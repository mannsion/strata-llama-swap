"""Download the benchmark datasets (pinned revisions) and write what bench.py reads: bench/data/*.jsonl + images.
Runs in a throwaway container (./llm bench-data); the downloads are only ever parsed as data.
  python prep.py OUT_DIR"""
import base64, io, json, os, random, sys, urllib.request

import pyarrow.parquet as pq

OUT = sys.argv[1]
FILES = {   # repo, revision, file
    "gsm8k": ("openai/gsm8k", "740312add88f", "main/test-00000-of-00001.parquet"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "b189ec765aa7", "data/test-00000-of-00001.parquet"),
    "humanevalplus": ("evalplus/humanevalplus", "d32357cf319e", "test.jsonl"),
    "mbppplus": ("evalplus/mbppplus", "b2d74c91837c", "data/test-00000-of-00001-d5781c9c51e02795.parquet"),
    "math500": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79a", "test.jsonl"),
    "mmstar": ("Lin-Chen/MMStar", "bc98d668301d", "mmstar.parquet"),
    "chartqa": ("HuggingFaceM4/ChartQA", "b605b6e08b57", "data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet"),
}


def fetch(name):
    repo, rev, f = FILES[name]
    with urllib.request.urlopen(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{f}", timeout=600) as r:
        raw = r.read()
    print(f"{name}: {len(raw) / 1e6:.1f} MB", flush=True)
    if f.endswith(".jsonl"):
        return [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    return pq.read_table(io.BytesIO(raw)).to_pylist()


def dump(name, rows):
    with open(os.path.join(OUT, name), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def save_image(sub, key, img):
    raw = img["bytes"] if isinstance(img, dict) else base64.b64decode(img) if isinstance(img, str) else img
    ext = "png" if raw[:4] == b"\x89PNG" else "webp" if raw[8:12] == b"WEBP" else "jpg"
    os.makedirs(os.path.join(OUT, "images", sub), exist_ok=True)
    rel = f"images/{sub}/{key}.{ext}"
    with open(os.path.join(OUT, rel), "wb") as f:
        f.write(raw)
    return rel


dump("openai__gsm8k-main-test.jsonl", [{"question": r["question"], "answer": r["answer"]} for r in fetch("gsm8k")])
dump("TIGER-Lab__MMLU-Pro-default-test.jsonl", fetch("mmlu_pro"))
dump("humanevalplus.jsonl", [{k: r[k] for k in ("task_id", "prompt", "entry_point", "test")} for r in fetch("humanevalplus")])
dump("mbppplus.jsonl", [{k: r[k] for k in ("task_id", "prompt", "test_imports", "test_list", "test")} for r in fetch("mbppplus")])
dump("math500.jsonl", [{k: r[k] for k in ("unique_id", "problem", "answer", "subject", "level")} for r in fetch("math500")])
dump("mmstar.jsonl", [{"index": int(r["index"]), "question": r["question"], "answer": r["answer"], "category": r["category"],
                       "l2_category": r["l2_category"], "image": save_image("mmstar", int(r["index"]), r["image"])}
                      for r in fetch("mmstar")])
cq = fetch("chartqa")
dump("chartqa.jsonl", [{"id": i, "query": cq[i]["query"], "label": [str(x) for x in cq[i]["label"]],
                        "human_or_machine": cq[i]["human_or_machine"], "image": save_image("chartqa", i, cq[i]["image"])}
                       for i in sorted(random.Random(1234).sample(range(len(cq)), 200))])
print("done")
