"""Accuracy suites for bench.py (thinking off, greedy unless noted). Each returns
   {"accuracy", "n", "ci95", "errors", "items": [...]}  with one item per question: id, ok, pred, gold, tokens, ...
so runs can be compared question by question.
Datasets live in bench/data (./llm bench-data) and are only ever parsed as data.
Model-written code runs only in the sandbox container (see sandbox/Dockerfile): no network, read-only, 1 GB, 256 pids."""
import json, math, os, random, re, subprocess, tempfile, threading, time, urllib.parse, urllib.request
from fractions import Fraction
from pathlib import Path

import client
from client import DATA, Sizer, chat, with_image

SANDBOX_IMAGE = "local-llm/sandbox:py312"

# suite sizes per mode: full is the real benchmark, fast a good-enough pass, quick a smoke test
SIZES = {
    "full": dict(gsm8k=300, math500=500, mmlu_pro=500, humaneval=164, mbpp=378, mmstar=400, chartqa=200, stories=30),
    "fast": dict(gsm8k=100, math500=150, mmlu_pro=150, humaneval=164, mbpp=100, mmstar=120, chartqa=60, stories=10),
    "quick": dict(gsm8k=8, math500=8, mmlu_pro=14, humaneval=6, mbpp=6, mmstar=12, chartqa=6, stories=3),
}
NEEDLE = {"full": ([32768, 131072, 250000], (0.1, 0.5, 0.9)), "fast": ([32768, 131072, 250000], (0.5,)),
          "quick": ([16384, 65536], (0.5,))}


# ---------------------------------------------------------------------------------------------- plumbing
class Ckpt:
    """Append-only log of finished items, so a restarted run continues where it stopped (same file = same run)."""

    def __init__(self, path):
        self.path, self.done, self.lock = Path(path) if path else None, {}, threading.Lock()
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    self.done.setdefault(r["suite"], {})[str(r["id"])] = r["item"]
                except ValueError:
                    pass                                         # a half-written last line

    def get(self, suite):
        return self.done.get(suite, {})

    def add(self, suite, id_, item):
        if self.path:
            with self.lock, self.path.open("a") as f:
                f.write(json.dumps({"suite": suite, "id": id_, "item": item}) + "\n")


def run(name, tasks, fn, workers=1, ck=None):
    """tasks: [(id, payload)]; fn(payload) -> item dict. Resumes finished ids from ck; a request that fails is
    recorded with an error (and retried on the next resume); six failures in a row mean the server is gone."""
    done = ck.get(name) if ck else {}
    todo = [t for t in tasks if str(t[0]) not in done]
    if done:
        print(f"  {name}: resuming, {len(done)} of {len(tasks)} items already done", flush=True)
    fails, lock, got, cnt, t0 = [0], threading.Lock(), {}, [0], time.time()

    def work(t):
        i, payload = t
        try:
            r = fn(payload)
            with lock:
                fails[0] = 0
        except Exception as e:                                   # noqa: BLE001 - recorded per item
            with lock:
                fails[0] += 1
                if fails[0] >= 6:
                    raise client.ServerDown(f"{name}: 6 requests in a row failed, last: {e}") from e
            return {"id": i, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
        r["id"] = i
        if ck:
            ck.add(name, i, r)
        with lock:
            cnt[0] += 1
            if cnt[0] % 25 == 0:
                print(f"  {name}: {cnt[0] + len(done)}/{len(tasks)} done, {time.time() - t0:.0f}s", flush=True)
        return r
    if workers <= 1:
        res = [work(t) for t in todo]
    else:
        import concurrent.futures as cf
        ex = cf.ThreadPoolExecutor(workers)
        try:
            res = list(ex.map(work, todo))
        finally:
            ex.shutdown(wait=True, cancel_futures=True)      # a stop or a dead server must not drain the whole queue
    got = {str(r["id"]): r for r in res}
    return [done.get(str(i)) or got[str(i)] for i, _ in tasks]


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def summarize(name, items, **extra):
    ok = sum(1 for r in items if r.get("ok"))
    n = len(items)
    res = {"accuracy": ok / n if n else None, "n": n, "ci95": wilson(ok, n),
           "errors": sum(1 for r in items if "error" in r), "truncated": sum(1 for r in items if r.get("finish") == "length"),
           **extra, "items": items}
    print(f"  {name}: {res['accuracy']:.1%} ({ok}/{n}), {res['errors']} errors, {res['truncated']} truncated", flush=True)
    return res


def load(name):
    return [json.loads(l) for l in (DATA / name).read_text().splitlines() if l.strip()]


def hf_rows(dataset, config, split, n):
    """First n rows of a public Hugging Face dataset via the datasets-server API, cached in data/ (JSONL)."""
    DATA.mkdir(exist_ok=True)
    cache = DATA / f"{dataset.replace('/', '__')}-{config}-{split}.jsonl"
    have = [json.loads(l) for l in cache.read_text().splitlines()] if cache.exists() else []
    off = len(have)
    while off < n:
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": off,
                                    "length": min(100, n - off)})
        with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{q}", timeout=60) as f:
            got = [r["row"] for r in json.load(f)["rows"]]
        if not got:
            break
        have += got
        off += len(got)
        cache.write_text("\n".join(json.dumps(r) for r in have))
    return have[:n]


def pick(keys, n, seed=1234, equal=False):
    """Indices of n items spread over the strata `keys` (one key per item), nested: pick(k, 100) is a subset of
    pick(k, 500), so a fast run is comparable item by item with a full one. Shares are proportional to the stratum
    size, or equal with equal=True. A fixed seed shuffles inside each stratum."""
    rnd, groups = random.Random(seed), {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    rank = []
    for k in sorted(groups):
        g = groups[k][:]
        rnd.shuffle(g)
        for pos, i in enumerate(g):
            rank.append(((pos + rnd.random()) / (1 if equal else len(g)), i))
    return sorted(i for _, i in sorted(rank)[:n])


def last_number(s):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return nums[-1].replace(",", "").rstrip(".") if nums else None


# ---------------------------------------------------------------------------------------------- GSM8K
def gsm8k(n, workers, ck):
    data = hf_rows("openai/gsm8k", "main", "test", 1319)
    idx = pick(["gsm8k"] * len(data), n, 1234)

    def one(x):
        r = chat([{"role": "user", "content": x["question"] + "\n\nSolve step by step. End with a line "
                   "'#### <number>' giving only the final numeric answer."}], max_tokens=1024)
        gold = x["answer"].split("####")[-1].strip().replace(",", "")
        m = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", r["text"])
        pred = m.group(1).replace(",", "") if m else last_number(r["text"])
        try:
            ok = pred is not None and abs(float(pred) - float(gold)) < 1e-6
        except ValueError:
            ok = False
        return {"ok": ok, "gold": gold, "pred": pred, "tokens": r["completion_tokens"], "finish": r["finish"]}
    return summarize("gsm8k", run("gsm8k", [(i, data[i]) for i in idx], one, workers, ck))


# ---------------------------------------------------------------------------------------------- MATH-500
def last_boxed(s):
    i = max(s.rfind("\\boxed"), s.rfind("\\fbox"))
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        depth += (s[k] == "{") - (s[k] == "}")
        if depth == 0:
            return s[j + 1:k]
    return None


UNITS = {"inches", "inch", "cm", "centimeters", "meters", "m", "feet", "ft", "units", "squareunits", "degrees", "degree",
         "cents", "dollars", "hours", "minutes", "seconds", "days", "miles", "mph", "sq", "cubic", "square", "years"}


def norm_math(s):
    s = s.strip().replace("\n", " ")
    s = re.sub(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|big|Big)(?![a-zA-Z])", "", s)
    for a in ("\\!", "\\,", "\\;", "\\:", "\\ ", "\\$", "$", "\\%", "%", "°", "\\degree", "\\displaystyle"):
        s = s.replace(a, " " if a == "\\ " else "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "", s)
    s = re.sub(r"\\(?:text|textbf|mbox|mathrm)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt\s*(\d)", r"\\sqrt{\1}", s)
    s = re.sub(r"\s+", "", s).rstrip(".")
    return s


def strip_assign(s):
    m = re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*=([^=]+)", s)
    return m.group(1) if m else s


class _Calc:
    """Evaluates a small LaTeX arithmetic subset to a float (numbers, + - * / ^, \\frac, \\sqrt, \\pi, \\cdot, \\times,
    brackets) by recursive descent. Anything else raises ValueError. No eval of any kind."""
    TOK = re.compile(r"\\[a-zA-Z]+|\d+\.?\d*|\.\d+|[-+*/^(){}\[\]]")

    def __init__(self, s):
        if re.sub(self.TOK, "", s):
            raise ValueError("unsupported")
        self.t, self.i = self.TOK.findall(s), 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, x=None):
        v = self.peek()
        if v is None or (x is not None and v != x):
            raise ValueError("syntax")
        self.i += 1
        return v

    def parse(self):
        v = self.expr()
        if self.peek() is not None:
            raise ValueError("trailing")
        return v

    def expr(self):
        v = self.term()
        while self.peek() in ("+", "-"):
            v = v + self.term() if self.eat() == "+" else v - self.term()
        return v

    def starts_atom(self):
        p = self.peek()
        return p is not None and (p[0].isdigit() or p[0] == "." or p in ("(", "{", "[", "\\frac", "\\sqrt", "\\pi"))

    def term(self):
        v = self.unary()
        while True:
            p = self.peek()
            if p in ("*", "\\cdot", "\\times"):
                self.eat()
                v *= self.unary()
            elif p in ("/", "\\div"):
                self.eat()
                d = self.unary()
                if d == 0:
                    raise ValueError("div0")
                v /= d
            elif self.starts_atom():
                v *= self.power()
            else:
                return v

    def unary(self):
        if self.peek() in ("-", "+"):
            return -self.unary() if self.eat() == "-" else self.unary()
        return self.power()

    def power(self):
        b = self.atom()
        if self.peek() == "^":
            self.eat()
            e = self.unary() if self.peek() != "{" else self.group("{", "}")
            if abs(e) > 400 or (abs(b) > 1e6 and abs(e) > 20):
                raise ValueError("huge")
            b = b ** e
            if isinstance(b, complex):
                raise ValueError("complex")
        return b

    def group(self, o, c):
        self.eat(o)
        v = self.expr()
        self.eat(c)
        return v

    def atom(self):
        p = self.peek()
        if p is None:
            raise ValueError("eof")
        if p[0].isdigit() or p[0] == ".":
            return float(self.eat())
        if p == "\\pi":
            self.eat()
            return math.pi
        if p == "\\frac":
            self.eat()
            a = self.group("{", "}")
            b = self.group("{", "}")
            if b == 0:
                raise ValueError("div0")
            return a / b
        if p == "\\sqrt":
            self.eat()
            k = 2.0
            if self.peek() == "[":
                k = self.group("[", "]")
            x = self.group("{", "}")
            if x < 0 or k == 0:
                raise ValueError("neg root")
            return x ** (1 / k)
        if p == "(":
            return self.group("(", ")")
        if p == "{":
            return self.group("{", "}")
        if p == "[":
            return self.group("[", "]")
        raise ValueError("atom")


def num_value(s):
    s = strip_assign(s)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    m = re.fullmatch(r"(.*?)([A-Za-z]+)", s)
    if m and m.group(2).lower() in UNITS and m.group(1):
        s = m.group(1)
    try:
        return _Calc(s).parse()
    except (ValueError, OverflowError, ZeroDivisionError, RecursionError):
        return None


def split_top(s):
    """'(1,2]' -> ('(', ']', ['1', '2']);  '1,2,3' -> ('', '', [...]);  None when there is no list."""
    open_, close = "", ""
    if len(s) >= 2 and s[0] in "([" and s[-1] in ")]":
        open_, close, s = s[0], s[-1], s[1:-1]
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return (open_, close, parts) if len(parts) > 1 else None


def math_equal(pred, gold, depth=0):
    if pred is None:
        return False
    a, b = norm_math(pred), norm_math(gold)
    if a == b or strip_assign(a) == b or a == strip_assign(b) or strip_assign(a) == strip_assign(b):
        return True
    na, nb = num_value(a), num_value(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= 1e-6 * max(1.0, abs(na), abs(nb))
    la, lb = split_top(a), split_top(b)
    if la and lb and depth < 3 and la[:2] == lb[:2] and len(la[2]) == len(lb[2]):
        return all(math_equal(x, y, depth + 1) for x, y in zip(la[2], lb[2]))
    return a.lower() == b.lower()


def math500(n, workers, ck):
    data = load("math500.jsonl")
    idx = pick([x["level"] for x in data], n, 1234)

    def one(x):
        r = chat([{"role": "user", "content": x["problem"] + "\n\nSolve the problem step by step, then put the final "
                   "answer in \\boxed{}."}], max_tokens=1536)
        pred = last_boxed(r["text"])
        if pred is None:
            m = re.findall(r"answer is[:\s]*\$?([^\n$]+?)\$?\.?(?:\n|$)", r["text"])
            pred = m[-1].strip() if m else None
        return {"ok": math_equal(pred, x["answer"]), "gold": x["answer"], "pred": pred, "level": x["level"],
                "subject": x["subject"], "tokens": r["completion_tokens"], "finish": r["finish"]}
    return summarize("math500", run("math500", [(data[i]["unique_id"], data[i]) for i in idx], one, workers, ck))


# ---------------------------------------------------------------------------------------------- MMLU-Pro
def mmlu_pro(n, workers, ck):
    data = hf_rows("TIGER-Lab/MMLU-Pro", "default", "test", 12032)
    idx = pick([x["category"] for x in data], n, 1234)
    letters = "ABCDEFGHIJ"

    def one(x):
        opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(x["options"]))
        r = chat([{"role": "user", "content": f"{x['question']}\n\n{opts}\n\nAnswer with only the letter of the correct "
                   "option. Do not explain."}], max_tokens=512)
        pred = mc_letter(r["text"], letters)
        return {"ok": pred == x["answer"], "gold": x["answer"], "pred": pred, "category": x["category"],
                "tokens": r["completion_tokens"], "finish": r["finish"]}
    return summarize("mmlu_pro", run("mmlu_pro", [(data[i]["question_id"], data[i]) for i in idx], one, workers, ck))


# ---------------------------------------------------------------------------------------------- code (sandboxed)
SANDBOX = r'''
import json, subprocess, sys, pathlib, concurrent.futures as cf
def go(p):
    try:
        r = subprocess.run([sys.executable, str(p)], capture_output=True, timeout=%d)
        return p.stem, r.returncode == 0, "" if r.returncode == 0 else r.stderr.decode("utf-8", "replace")[-300:]
    except subprocess.TimeoutExpired:
        return p.stem, False, "timeout"
with cf.ThreadPoolExecutor(4) as ex:
    print(json.dumps({s: [ok, err] for s, ok, err in ex.map(go, sorted(pathlib.Path("/w").glob("*.py")))}))
'''


def run_sandboxed(programs, timeout=30):
    """programs {name: source} -> {name: (passed, error tail)}. Model code runs ONLY here: a throwaway container with no
    network, a read-only root, no host files except the programs (read-only), 1 GB of memory and 256 processes."""
    with tempfile.TemporaryDirectory() as d:
        for name, src in programs.items():
            Path(d, re.sub(r"[^A-Za-z0-9_]", "_", str(name)) + ".py").write_text(src)
        os.chmod(d, 0o755)
        for f in Path(d).iterdir():
            os.chmod(f, 0o644)
        r = subprocess.run(["docker", "run", "--rm", "--network", "none", "--memory", "1g", "--pids-limit", "256",
                            "--cpus", "4", "--read-only", "--tmpfs", "/tmp", "--user", "65534:65534",
                            "-v", f"{d}:/w:ro", SANDBOX_IMAGE, "python", "-c", SANDBOX % timeout],
                           capture_output=True, text=True, timeout=7200)
        if r.returncode:
            raise RuntimeError(f"sandbox failed: {r.stderr[-400:]}")
        out = json.loads(r.stdout.strip().splitlines()[-1])
    return {name: tuple(out.get(re.sub(r"[^A-Za-z0-9_]", "_", str(name)), [False, "missing"])) for name in programs}


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def humaneval_plus(n, workers, ck):
    data = load("humanevalplus.jsonl")
    data = data if n >= len(data) else data[::max(1, len(data) // n)][:n]

    def gen(x):
        r = chat([{"role": "user", "content": "Complete this Python function. Reply with the full function "
                   "(including the signature and any imports) in one ```python block.\n\n```python\n"
                   + x["prompt"] + "```"}], max_tokens=1024)
        return {"code": extract_code(r["text"]), "tokens": r["completion_tokens"], "finish": r["finish"]}
    gens = run("humaneval_plus_gen", [(x["task_id"], x) for x in data], gen, workers, ck)
    progs = {x["task_id"]: x["prompt"] + "\n" + g.get("code", "") + "\n\n" + x["test"] + f"\n\ncheck({x['entry_point']})\n"
             for x, g in zip(data, gens) if "error" not in g}
    res = run_sandboxed(progs)
    items = [{"id": x["task_id"], "ok": bool(res[x["task_id"]][0]) if x["task_id"] in res else False,
              "pred": "pass" if x["task_id"] in res and res[x["task_id"]][0] else "fail",
              "err": (res[x["task_id"]][1] if x["task_id"] in res else g.get("error", ""))[-200:],
              "tokens": g.get("tokens"), "finish": g.get("finish"), "code": g.get("code", "")}
             for x, g in zip(data, gens)]
    return summarize("humaneval_plus", items)


def mbpp_plus(n, workers, ck):
    data = load("mbppplus.jsonl")
    data = data if n >= len(data) else data[::max(1, len(data) // n)][:n]

    def gen(x):
        r = chat([{"role": "user", "content": f"{x['prompt']}\n\nYour code should pass this test:\n{x['test_list'][0]}\n\n"
                   "Reply with the complete Python code (imports, helper functions and the required function) in one "
                   "```python block."}], max_tokens=1024)
        return {"code": extract_code(r["text"]), "tokens": r["completion_tokens"], "finish": r["finish"]}
    gens = run("mbpp_plus_gen", [(x["task_id"], x) for x in data], gen, workers, ck)
    progs = {x["task_id"]: "\n".join(x["test_imports"]) + "\n" + g.get("code", "") + "\n\n" + x["test"] + "\n"
             for x, g in zip(data, gens) if "error" not in g}
    res = run_sandboxed(progs)
    items = [{"id": x["task_id"], "ok": bool(res[x["task_id"]][0]) if x["task_id"] in res else False,
              "pred": "pass" if x["task_id"] in res and res[x["task_id"]][0] else "fail",
              "err": (res[x["task_id"]][1] if x["task_id"] in res else g.get("error", ""))[-200:],
              "tokens": g.get("tokens"), "finish": g.get("finish"), "code": g.get("code", "")}
             for x, g in zip(data, gens)]
    return summarize("mbpp_plus", items)


# ---------------------------------------------------------------------------------------------- images
def mc_letter(text, letters="ABCD"):
    """The chosen option: a leading letter ('B', 'B: Glass', '(B)'), else the last 'answer is B' / '**B**' style mention,
    else the last stand-alone capital letter near the end (the model started to explain)."""
    t = text.strip()
    m = re.match(rf"^\W*([{letters}])(?:\s*$|[.):,*\]]|\s*:)", t)      # "B", "B.", "(B)", "B: Glass" - but not "I think ..."
    if m:
        return m.group(1)
    for pat in (rf"(?i:answer|option|choice)(?: is|:)?\s*[:\-]?\s*\**\(?([{letters}])\)?(?![A-Za-z])", rf"\*\*\(?([{letters}])\)?[.):]?\*\*"):
        ms = re.findall(pat, t)
        if ms:
            return ms[-1]
    ms = re.findall(rf"(?<![A-Za-z])([{letters}])(?![A-Za-z])", t[-60:])
    return ms[-1] if ms else None


def mmstar(n, workers, ck):
    data = load("mmstar.jsonl")
    idx = pick([x["category"] for x in data], n, 1234, equal=True)

    def one(x):
        r = chat([{"role": "user", "content": with_image(x["question"] + "\n\nRespond with only the letter of the correct "
                   "option (A, B, C or D). Do not explain.", x["image"])}], max_tokens=512)
        pred = mc_letter(r["text"])
        return {"ok": pred == x["answer"], "gold": x["answer"], "pred": pred, "category": x["category"],
                "l2": x["l2_category"], "tokens": r["completion_tokens"], "prompt_tokens": r["prompt_tokens"],
                "finish": r["finish"]}
    return summarize("mmstar", run("mmstar", [(data[i]["index"], data[i]) for i in idx], one, workers, ck))


def to_float(s):
    t = str(s).strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    try:
        return float(t)
    except ValueError:
        return None


def relaxed_ok(pred, golds):
    """ChartQA relaxed accuracy: numbers within 5% of the gold value, everything else exact (case-insensitive)."""
    p = pred.strip().rstrip(".")
    for g in golds:
        pf, gf = to_float(p), to_float(g)
        if pf is not None and gf is not None:
            if (gf == 0 and pf == 0) or (gf != 0 and abs(pf - gf) / abs(gf) <= 0.05):
                return True
        elif p.lower() == g.strip().lower():
            return True
    return False


def chartqa(n, workers, ck):
    data = load("chartqa.jsonl")
    idx = pick([x["human_or_machine"] for x in data], n, 1234, equal=True)

    def one(x):
        r = chat([{"role": "user", "content": with_image(x["query"] + "\nAnswer the question using a single word or "
                   "phrase.", x["image"])}], max_tokens=32)
        pred = (r["text"].strip().splitlines() or [""])[0]
        return {"ok": relaxed_ok(pred, x["label"]), "gold": x["label"][0], "pred": pred,
                "kind": "machine" if x["human_or_machine"] else "human", "tokens": r["completion_tokens"],
                "prompt_tokens": r["prompt_tokens"], "finish": r["finish"]}
    return summarize("chartqa", run("chartqa", [(data[i]["id"], data[i]) for i in idx], one, workers, ck))


# ---------------------------------------------------------------------------------------------- stories
WORD = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
FIRST = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
SECOND = {"you", "your", "yours", "yourself"}
THIRD = {"he", "she", "him", "her", "his", "hers", "they", "them", "their", "theirs"}


def strip_title(text):
    lines = text.strip().split("\n")
    if lines and len(lines[0].split()) <= 14 and (lines[0].startswith("#") or (lines[0].startswith("**") and lines[0].rstrip().endswith("**"))):
        lines = lines[1:]
    return "\n".join(lines).strip()


def story_metrics(text, spec):
    body = strip_title(text)
    ws = [w.lower() for w in WORD.findall(body)]
    n = len(ws)
    low = body.lower()
    hits = [m for m in spec["must"] if m.lower() in low]
    narr = [w.lower() for w in WORD.findall(re.sub(r'"[^"]*"|“[^”]*”', " ", body))] or ws
    f, s, t = (sum(w in S for w in narr) for S in (FIRST, SECOND, THIRD))
    nn = max(1, len(narr))
    pov = spec.get("pov")
    pov_ok = True if not pov else (f / nn >= 0.015 and f > t) if pov == "first" else \
        (s / nn >= 0.015 and s > f) if pov == "second" else (f / nn < 0.005 and t / nn >= 0.01)
    tail = re.sub(r"[^a-z ]", "", low[-250:])
    ends_ok = True if not spec.get("ends_with") else re.sub(r"[^a-z ]", "", spec["ends_with"].lower()) in tail
    grams = list(zip(ws, ws[1:], ws[2:]))
    return {"words": n, "in_range": spec["min_words"] <= n <= spec["max_words"],
            "must_hit": len(hits), "must_total": len(spec["must"]), "all_must": len(hits) == len(spec["must"]),
            "pov_ok": pov_ok, "ends_ok": ends_ok, "distinct3": len(set(grams)) / len(grams) if grams else 0.0}


def stories(n, workers, ck):
    specs = json.loads((DATA / "stories.json").read_text())
    specs = specs if n >= len(specs) else specs[::max(1, len(specs) // n)][:n]

    def one(spec):
        r = chat([{"role": "user", "content": spec["prompt"]}], max_tokens=min(2400, int(spec["max_words"] * 1.8) + 100),
                 temperature=0.7, seed=1)
        m = story_metrics(r["text"], spec)
        m.update({"text": r["text"], "genre": spec["genre"], "tokens": r["completion_tokens"], "finish": r["finish"],
                  "prompt": spec["prompt"]})
        m["ok"] = m["in_range"] and m["all_must"] and m["pov_ok"] and m["ends_ok"]
        return m
    items = run("stories", [(s["id"], s) for s in specs], one, workers, ck)
    ok = [r for r in items if "error" not in r]
    avg = lambda k: sum(float(r[k]) for r in ok) / len(ok) if ok else None      # noqa: E731
    return summarize("stories", items, range_rate=avg("in_range"), must_rate=avg("all_must"),
                     must_fraction=sum(r["must_hit"] for r in ok) / max(1, sum(r["must_total"] for r in ok)),
                     pov_rate=avg("pov_ok"), distinct3=avg("distinct3"), words_mean=avg("words"))


# ---------------------------------------------------------------------------------------------- long context
def needle(mode, ck):
    sizes, depths = NEEDLE[mode]
    sizer, tasks = Sizer(), []
    r = chat([{"role": "user", "content": sizer.text(4000, 7) + "\nSay OK."}], max_tokens=4)
    sizer.learn(4000, r["prompt_tokens"])

    def one(t):
        n, dp = t
        seed = random.getrandbits(48)
        code = f"{random.Random(seed).randrange(10**7, 10**8)}"
        lines = sizer.text(n, seed).split("\n")
        lines.insert(max(1, int(len(lines) * dp)), f"IMPORTANT: the secret access code is {code}.")
        r = chat([{"role": "user", "content": "\n".join(lines) + "\n\nWhat is the secret access code in the document? "
                   "Reply with only the code."}], max_tokens=32)
        ok = code in r["text"]
        print(f"  needle {r['prompt_tokens']} tok @ {dp:.0%}: {'PASS' if ok else 'FAIL'} ({r['ttft']:.0f}s)", flush=True)
        return {"ok": ok, "context": r["prompt_tokens"], "target": n, "depth": dp, "pred": "hit" if ok else "miss",
                "ttft_s": r["ttft"]}
    return summarize("needle", run("needle", [(f"{n}@{dp}", (n, dp)) for n in sizes for dp in depths], one, 1, ck))
