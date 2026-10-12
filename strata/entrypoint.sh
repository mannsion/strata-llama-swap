#!/bin/sh
# Prepares the pack and MTP draft layer once (under /data), writes the engine config from env, starts the server.
# Env (set by compose.yaml): MODEL_GGUF, MMPROJ, VISION=gpu|cpu|none, CONTEXT, KV, PREFILL, SPEC, EXTRA_ARGS.
set -eu
cd /opt/strata
PY=/opt/strata/.venv/bin/python
DATA=${STRATA_DATA:-/data}
GGUF=${MODEL_GGUF:?set MODEL_GGUF}
NAME=${MODEL_NAME:-strata}
PACK=$DATA/packs/$(basename "$GGUF" .gguf)
export STRATA_GGUF_PY=/opt/strata/third_party/llama.cpp/gguf-py

if [ ! -f "$PACK/native_experts.txt" ] || [ ! -f "$PACK/tokenizer/vocab.json" ]; then
  echo "Packing $GGUF -> $PACK (one time)"
  $PY tools/iq_pack.py --gguf "$GGUF" --out "$PACK" --compat-bf16
fi
MTP=${MTP_RT:-$DATA/mtp/rt}
if [ ! -f "$MTP/experts.bin" ]; then
  MTP=$DATA/mtp
  if [ ! -f "$MTP/rt/experts.bin" ]; then
    echo "Building the MTP draft layer (one time, ~5 GB download)"
    $PY tools/mtp_fetch.py fetch --out "$MTP"
    $PY tools/mtp_pack.py --src "$MTP" --experts q2_0 --out "$MTP/mtp-q2_0.gguf"
    $PY tools/mtp_rt.py --gguf "$MTP/mtp-q2_0.gguf" --out "$MTP/rt"
  fi
  MTP=$MTP/rt
fi

CFG=$DATA/strata.json
$PY - "$CFG" "$GGUF" "$PACK" "$MTP" "$NAME" <<'PYEOF'
import json, os, sys
from pathlib import Path
import setup
cfg_path, gguf, pack, rt, name = sys.argv[1:]
setup.refresh_draft_vocab(Path(rt), os.environ.get("DRAFT_VOCAB", "cjk"))
e = os.environ.get
ctx = int(e("CONTEXT", "262144"))
ple = e("PLE_GGUF") or gguf
args = ["--pack", pack, "--native", gguf] + ([] if ple == "none" else ["--ple-gguf", ple]) + [
        "--expert-profile", "data/expert-profile.bin", "--expert-cache", e("EXPERT_CACHE", "auto"),
        "--prefill", e("PREFILL", "auto"), "--spec", e("SPEC", "4"), "--spec-min-p", e("SPEC_MIN_P", "0.5"),
        "--mtp", rt, "--max-context", str(ctx), "--kv", e("KV", "int8")]
if ctx >= 65536:
    args += ["--kv-resident", e("KV_RESIDENT", "32768")]
if e("RESIDENT_BUDGET_GIB"):
    args += ["--resident-budget-gib", e("RESIDENT_BUDGET_GIB")]
vision = e("VISION", "gpu")
reserve = e("VRAM_RESERVE_MIB") or ("700" if vision != "none" else "")
if vision != "none":
    args += ["--vision"]
if reserve:
    args += ["--vram-reserve-mib", reserve]
args += e("EXTRA_ARGS", "").split()
env = {}
table = setup.hipblaslt_table(e("GPU_ARCH", "gfx1201"), ["/opt/rocm/lib"])
if table:
    env["STRATA_HIPBLASLT_TUNING"] = str(table)
for k in ("STRATA_HIP_WMMA", "STRATA_PF_FUSED", "STRATA_SELECT_WMMA", "STRATA_PREFILL_RING", "STRATA_RESIDENT_PIN"):
    if e(k):
        env[k] = e(k)
cfg = {"exe": "/opt/strata/engine/strata", "args": args, "cwd": "/opt/strata",
       "tokenizer": f"{pack}/tokenizer", "model_name": name, "log": f"{os.path.dirname(cfg_path)}/strata.log",
       "lib_dirs": ["/opt/rocm/lib"], "backend": "hip", "env": env}
if vision != "none":
    # strata-vision is the stock encoder (llama.cpp mtmd); "gpu" makes serve.server start it with --gpu, and this
    # image builds it with ggml's HIP backend, so it runs on the same Radeon as the engine
    cfg["vision"] = {"exe": "/opt/strata/engine/strata-vision", "mmproj": e("MMPROJ", ""), "model": gguf,
                     "gpu": vision == "gpu",
                     "max_tokens": int(e("VISION_TOKENS") or (1024 if vision == "gpu" else 300))}
    if vision == "cpu":
        cfg["vision"]["threads"] = max(1, (os.cpu_count() or 8) // 2)
    if not Path(cfg["vision"]["mmproj"]).is_file():
        sys.exit(f"MMPROJ not found: {cfg['vision']['mmproj']!r}")
Path(cfg_path).write_text(json.dumps(cfg, indent=1))
print(json.dumps(cfg, indent=1))
PYEOF

exec $PY -m serve.server --engine strata --config "$CFG" --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
