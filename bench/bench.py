#!/usr/bin/env python3
"""Benchmark the running engine (./llm up) through its OpenAI API, thinking off.
Stdlib only. Speed: prefill 1K-250K, decode (prose/code, at 16K and 245K context), 2/4 parallel streams, image latency.
Accuracy: GSM8K, MATH-500, MMLU-Pro, HumanEval+, MBPP+, MMStar, ChartQA, stories (temp 0.7), needle at 32K/128K/250K.
Every question's answer is kept in the JSON.

  bench.py [--mode full|fast|quick] [--only speed,gsm8k,...] [--out FILE] [--resume] [--workers N] [--thinking]
Results: results/strata-<time>-<mode>.json (rewritten after every suite; --resume --out FILE continues a stopped run,
the per-item log is FILE.ckpt).
"""
import argparse, json, os, re, statistics, subprocess, sys, threading, time
from pathlib import Path

import client, suites
from client import RESULTS, Sizer, chat, decode_rate, with_image

HERE = Path(__file__).resolve().parent
ORDER = ["gsm8k", "mmlu_pro", "humaneval_plus", "mbpp_plus", "math500", "mmstar", "chartqa", "stories", "needle"]


# ---------------------------------------------------------------------------------------------- resource sampler
class Sampler(threading.Thread):
    """Peak host RAM in use, peak VRAM and peak container memory while the benchmark runs."""

    def __init__(self, container):
        super().__init__(daemon=True)
        self.container, self.stop_, self.peak = container, threading.Event(), {"host_used_gib": 0.0, "vram_gib": 0.0, "container": None}

    def run(self):
        n = 0
        while not self.stop_.wait(5):
            try:
                self.peak["host_used_gib"] = max(self.peak["host_used_gib"], client.meminfo_gib()["used_gib"])
                v = client.gpu_read("mem_info_vram_used")
                self.peak["vram_gib"] = max(self.peak["vram_gib"], (v or 0) / 2**30)
                n += 1
                if n % 12 == 1:
                    st = client.system_info(self.container)["container_mem"]
                    if st:
                        self.peak["container"] = st
            except Exception:                                # noqa: BLE001 - sampling must never kill the run
                pass


# ---------------------------------------------------------------------------------------------- speed
def image_latency(reps=3):
    """Time to first token for a question about a picture (6 distinct images: nothing is served from an image cache)."""
    ms, cq = suites.load("mmstar.jsonl"), suites.load("chartqa.jsonl")
    imgs = [("mmstar", x["image"]) for x in ms[:reps]] + [("chartqa", x["image"]) for x in cq[:reps]]
    chat([{"role": "user", "content": with_image("What is this?", ms[-1]["image"])}], max_tokens=4)     # warm-up
    out = []
    for kind, path in imgs:
        r = chat([{"role": "user", "content": with_image("Describe this image in one sentence.", path)}], max_tokens=8)
        out.append({"kind": kind, "image": path, "ttft_s": r["ttft"], "prompt_tokens": r["prompt_tokens"]})
        print(f"  image ttft {kind}: {r['ttft']:.2f}s ({r['prompt_tokens']} prompt tokens)", flush=True)
    t = [chat([{"role": "user", "content": f"Variant {i}: describe a harbor at dawn in one sentence."}], max_tokens=8)["ttft"]
         for i in range(3)]
    return {"runs": out, "median_ttft_s": statistics.median(x["ttft_s"] for x in out),
            "text_ttft_s": statistics.median(t)}


def speed(mode):
    quick, fast = mode == "quick", mode == "fast"
    out = {"prefill": [], "decode": [], "concurrency": []}
    sizer, reps = Sizer(), (1 if quick or fast else 3)
    r = chat([{"role": "user", "content": sizer.text(2000, 1) + "\nSay OK."}], max_tokens=4)         # warm-up + calibration
    sizer.learn(2000, r["prompt_tokens"])
    sizes = [1024, 4096, 16384, 65536] if quick else [4096, 16384, 65536, 131072, 250000] if fast else \
        [1024, 4096, 16384, 65536, 131072, 250000]
    for n in sizes:
        runs = []
        for _ in range(reps):
            msg = sizer.text(n, int.from_bytes(os.urandom(6), "big")) + "\n\nReply with the single word: done."
            r = chat([{"role": "user", "content": msg}], max_tokens=8)
            sizer.learn(n, r["prompt_tokens"])
            runs.append({"prompt_tokens": r["prompt_tokens"], "ttft_s": r["ttft"],
                         "prefill_tok_s": r["prompt_tokens"] / r["ttft"] if r["ttft"] else None})
            print(f"  prefill {n:>7}: {r['prompt_tokens']} tok, ttft {r['ttft']:.2f}s, {runs[-1]['prefill_tok_s']:.0f} tok/s", flush=True)
        out["prefill"].append({"target": n, "runs": runs, "median_tok_s": statistics.median(x["prefill_tok_s"] for x in runs),
                               "median_ttft_s": statistics.median(x["ttft_s"] for x in runs)})
    tasks = {"prose": "Write a long, detailed essay (at least 900 words) about the history of lighthouses.",
             "code": "Write a complete, well-commented Python module implementing a red-black tree with insert, delete, "
                     "search and in-order traversal, plus unit tests."}
    for ctx in ([0] if quick else [0, 245760] if fast else [0, 16384, 245760]):
        for name, task in (tasks.items() if ctx <= 16384 else [("prose", tasks["prose"])]):
            runs = []
            for _ in range(reps):
                pre = (sizer.text(ctx, int.from_bytes(os.urandom(6), "big")) + "\n\nIgnore the document above.\n") if ctx else ""
                r = chat([{"role": "user", "content": pre + task}], max_tokens=512)
                runs.append({"completion_tokens": r["completion_tokens"], "decode_tok_s": decode_rate(r), "ttft_s": r["ttft"]})
                print(f"  decode {name} @ctx {ctx}: {r['completion_tokens']} tok, {decode_rate(r) or 0:.1f} tok/s", flush=True)
            out["decode"].append({"task": name, "context": ctx, "runs": runs,
                                  "median_tok_s": statistics.median(x["decode_tok_s"] for x in runs if x["decode_tok_s"])})
    import concurrent.futures as cf
    for c in ([4] if quick or fast else [2, 4]):
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(c) as ex:
            rs = list(ex.map(lambda i: chat([{"role": "user", "content": tasks["prose"] + f" (variant {i})"}], max_tokens=256), range(c)))
        wall = time.perf_counter() - t0
        tot = sum(r["completion_tokens"] or 0 for r in rs)
        out["concurrency"].append({"streams": c, "total_tokens": tot, "wall_s": wall, "aggregate_tok_s": tot / wall,
                                   "per_stream_tok_s": statistics.median(decode_rate(r) or 0 for r in rs)})
        print(f"  concurrency {c}: {tot / wall:.1f} tok/s aggregate", flush=True)
    out["image"] = image_latency()
    return out


# ---------------------------------------------------------------------------------------------- main
def save(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["full", "fast", "quick"], default="full")
    ap.add_argument("--quick", action="store_true", help="same as --mode quick")
    ap.add_argument("--fast", action="store_true", help="same as --mode fast")
    ap.add_argument("--only", default="speed," + ",".join(ORDER))
    ap.add_argument("--out", help="result file (default results/strata-<time>-<mode>.json)")
    ap.add_argument("--resume", action="store_true", help="continue the run in --out: finished parts and items are kept")
    ap.add_argument("--workers", type=int, default=1, help="parallel requests for the accuracy suites")
    ap.add_argument("--thinking", action="store_true")
    a = ap.parse_args()
    import signal

    def on_term(*_):                                                  # `finally` below then saves peaks and the partial result;
        threading.Timer(90, os._exit, [143]).start()                  # a request that hangs must not keep the process alive
        sys.exit(143)
    signal.signal(signal.SIGTERM, on_term)
    mode = "quick" if a.quick else "fast" if a.fast else a.mode
    client.THINKING = a.thinking
    engine, workers = "strata", a.workers
    parts = a.only.split(",")
    RESULTS.mkdir(exist_ok=True)
    path = Path(a.out) if a.out else RESULTS / f"{engine}-{time.strftime('%Y%m%d-%H%M%S')}-{mode}.json"
    out = json.loads(path.read_text()) if a.resume and path.exists() else {}
    if not out:
        out = {"engine": engine, "model": os.environ.get("STRATA_MODEL"), "mode": mode,
               "thinking": client.THINKING, "workers": workers,
               "started": time.strftime("%Y-%m-%d %H:%M:%S"), "suites": {}}
    container = f"local-llm-{engine}-1"
    out["served"] = client.served_model()
    out.setdefault("system_loaded", client.system_info(container))
    ck = suites.Ckpt(str(path) + ".ckpt")
    sam = Sampler(container)
    sam.start()
    sz = lambda: suites.SIZES[out["mode"]]                    # noqa: E731
    plan = {
        "gsm8k": lambda: suites.gsm8k(sz()["gsm8k"], workers, ck),
        "mmlu_pro": lambda: suites.mmlu_pro(sz()["mmlu_pro"], workers, ck),
        "humaneval_plus": lambda: suites.humaneval_plus(sz()["humaneval"], workers, ck),
        "mbpp_plus": lambda: suites.mbpp_plus(sz()["mbpp"], workers, ck),
        "math500": lambda: suites.math500(sz()["math500"], workers, ck),
        "mmstar": lambda: suites.mmstar(sz()["mmstar"], workers, ck),
        "chartqa": lambda: suites.chartqa(sz()["chartqa"], workers, ck),
        "stories": lambda: suites.stories(sz()["stories"], workers, ck),
        "needle": lambda: suites.needle(out["mode"], ck),
    }
    try:
        if "speed" in parts and "speed" not in out:
            print(f"== speed  [{time.strftime('%H:%M:%S')}]", flush=True)
            out["speed"] = speed(mode)
            save(path, out)
        for name in ORDER:
            if name in parts and name not in out["suites"]:
                print(f"== {name}  [{time.strftime('%H:%M:%S')}]", flush=True)
                t0 = time.time()
                out["suites"][name] = plan[name]()
                out["suites"][name]["seconds"] = time.time() - t0
                save(path, out)
    finally:
        sam.stop_.set()
        prev = out.get("peak") or {}                       # a resumed run keeps the peaks of the part before the restart
        out["peak"] = {"host_used_gib": max(sam.peak["host_used_gib"], prev.get("host_used_gib") or 0),
                       "vram_gib": max(sam.peak["vram_gib"], prev.get("vram_gib") or 0),
                       "container": sam.peak["container"] or prev.get("container")}
        out["system_after"] = client.system_info(container)
        save(path, out)
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["complete"] = all(n in out["suites"] for n in ORDER if n in parts) and ("speed" in out or "speed" not in parts)
    save(path, out)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
