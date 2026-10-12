"""OpenAI-API client shared by the bench scripts: one streamed chat request with timing, the filler/Sizer used for long
prompts, image parts, and the GPU/RAM probes. Stdlib only."""
import base64, concurrent.futures as cf, http.client, json, os, random, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE.parent / "results"
HOST, PORT = os.environ.get("API_HOST", "127.0.0.1"), int(os.environ.get("API_PORT", "8000"))
MODEL = os.environ.get("MODEL", "Qwen3.8-Flash-Next-Uncensored-i1-IQ4_XS")
THINKING = False


class ServerDown(RuntimeError):
    """The engine stopped answering (several requests in a row failed)."""


# ---------------------------------------------------------------------------------------------- requests
def image_part(path):
    """An OpenAI image_url part with the file inlined as a base64 data URL."""
    p = Path(path)
    if not p.is_absolute():
        p = DATA / p
    raw = p.read_bytes()
    mime = "image/png" if raw[:4] == b"\x89PNG" else "image/webp" if raw[8:12] == b"WEBP" else "image/jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(raw).decode()}"}}


def with_image(text, path):
    return [image_part(path), {"type": "text", "text": text}]



def chat(messages, max_tokens=256, temperature=0.0, timeout=1500, retries=1, **extra):
    """One streamed chat request. Returns text, usage and timing (ttft, decode span). Transient failures
    (connection reset, HTTP 5xx) are retried; a request that keeps failing raises."""
    for attempt in range(retries + 1):
        try:
            r = _chat(messages, max_tokens, temperature, timeout, **extra)
            return r
        except (ConnectionError, http.client.HTTPException, TimeoutError, OSError, RuntimeError) as e:
            if attempt == retries or "HTTP 4" in str(e):
                raise
            time.sleep(5 * (attempt + 1))


def _chat(messages, max_tokens, temperature, timeout, **extra):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": THINKING},
            "reasoning_effort": "medium" if THINKING else "none", **extra}
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    try:
        t0 = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:500]!r}")
        text, reasoning, usage, t_first, t_last, chunks, finish = [], [], None, None, None, 0, None
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                finish = ch.get("finish_reason") or finish
                piece = d.get("content") or ""
                rpiece = d.get("reasoning_content") or d.get("reasoning") or ""
                if piece or rpiece:
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
                    chunks += 1
                    text.append(piece)
                    reasoning.append(rpiece)
        t_end = time.perf_counter()
    finally:
        conn.close()
    usage = usage or {}
    return {"text": "".join(text), "reasoning": "".join(reasoning), "finish": finish,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "ttft": (t_first - t0) if t_first else None, "decode_s": (t_last - t_first) if t_first else None,
            "total_s": t_end - t0, "chunks": chunks}


def decode_rate(r):
    n = r.get("completion_tokens") or 0
    return (n - 1) / r["decode_s"] if r.get("decode_s") and n > 1 else None


def parallel(fn, items, workers=1):
    """Ordered map; a request that fails becomes {"error": ...} (via fn's own handling) and, when many fail in a row,
    ServerDown ends the run instead of burning hours on a dead server."""
    bad = [0]

    def guarded(x):
        try:
            r = fn(x)
            bad[0] = 0
            return r
        except Exception as e:                                  # noqa: BLE001 - recorded per item
            bad[0] += 1
            if bad[0] >= 6:
                raise ServerDown(f"{bad[0]} requests in a row failed, last: {e}") from e
            return {"error": f"{type(e).__name__}: {e}"[:300], "ok": False}
    if workers <= 1:
        return [guarded(x) for x in items]
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(guarded, items))


# ---------------------------------------------------------------------------------------------- long prompts
WORDS = ("alpha river copper lantern meadow quartz signal harbor velvet orbit canyon ember falcon glacier island "
         "jasmine kernel lattice marble nectar oasis prism quiver raven saffron tundra umbra vertex willow xenon "
         "yonder zephyr anchor beacon cobalt delta engine fabric granite helix ivory jungle kayak lumen magnet").split()


def filler(n_tokens, seed):
    """Unique text of roughly n_tokens (numbered lines of random words; a fresh seed defeats prefix caches)."""
    rnd = random.Random(seed)
    lines, est = [f"Document {seed:x}."], 0
    while est < n_tokens:
        lines.append(f"{len(lines)}. " + " ".join(rnd.choice(WORDS) for _ in range(12)) + ".")
        est += 17                                     # ~17 tokens per line, corrected by calibration below
    return "\n".join(lines)


class Sizer:
    """Maps a target token count to a filler size using the server's own prompt_tokens."""
    ratio = 1.0

    def text(self, n, seed):
        return filler(int(n * self.ratio * (0.98 if n > 131072 else 1.0)), seed)

    def learn(self, target, got):
        if got:
            self.ratio *= target / got


# ---------------------------------------------------------------------------------------------- machine probes
def gpu_dir():
    """sysfs dir of the Radeon AI PRO R9700 (the card with the most VRAM; card0/card1 swap between boots)."""
    best, size = None, 0
    for c in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        try:
            n = int((c / "device/mem_info_vram_total").read_text())
        except (OSError, ValueError):
            continue
        if n > size:
            best, size = c / "device", n
    return best


def gpu_read(name):
    d = gpu_dir()
    try:
        return int((d / name).read_text()) if d else None
    except (OSError, ValueError):
        return None


def meminfo_gib():
    m = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        m[k] = int(v.split()[0]) / 2**20
    return {"used_gib": m["MemTotal"] - m["MemAvailable"], "page_cache_gib": m["Cached"], "total_gib": m["MemTotal"]}


def system_info(container=None):
    vram = gpu_read("mem_info_vram_used")
    st = None
    if container:
        try:
            st = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
                                capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception:                                # noqa: BLE001
            st = None
    return {"vram_used_gib": vram / 2**30 if vram else None, "container_mem": st, "host": meminfo_gib()}


def served_model():
    import urllib.request
    with urllib.request.urlopen(f"http://{HOST}:{PORT}/v1/models", timeout=30) as f:
        return json.load(f)
