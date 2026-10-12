# Strata for the Radeon AI PRO R9700 (gfx1201), built from vendor/strata.
# ROCm 7.2.4: its hipBLASLt (1.2.2) matches the shipped gfx1201 tuning table.
ARG ROCM_VERSION=7.2.4
FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8 \
    HIP_PLATFORM=amd ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl git libatomic1 libgomp1 ninja-build \
        python3 python3-pip python3-venv hipblas-dev hipblaslt-dev rocblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/strata
COPY . .

RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt

# The engine as setup.py builds it for HIP, plus the K-quant prompt kernels and the extra expert kernels (so other GGUF
# quants of the model work too). Native CPU code: build on the machine that runs it.
ARG GPU_ARCH=gfx1201
RUN .venv/bin/python - <<'PYEOF'
import json, os, pathlib, shutil
import setup
arch = os.environ.get("GPU_ARCH", "gfx1201")
root = pathlib.Path("/opt/rocm")
bitcode = next(p for p in (root / "lib/llvm/amdgcn/bitcode", root / "amdgcn/bitcode") if p.is_dir())
llama = setup.get_llama_cpp()
setup.cmake_build(setup.ROOT, setup.ROOT / "build-hip", "strata",
    ["-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF", "-DSTRATA_BUILD_TESTS=OFF",
     "-DSTRATA_PREFILL_MMQ=ON", "-DSTRATA_MMQ_KQUANTS=ON", "-DSTRATA_Q6K_EXPERTS=ON", "-DSTRATA_ORCA_Q4KS_MMQ=ON",
     f"-DCMAKE_HIP_ARCHITECTURES={arch}",
     f"-DCMAKE_HIP_COMPILER={root / 'llvm/bin/clang++'}", f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root}",
     f"-DCMAKE_PREFIX_PATH={root}", f"-DCMAKE_HIP_FLAGS=--rocm-path={root} --rocm-device-lib-path={bitcode}",
     f"-DSTRATA_GGML_DIR={llama}"], None, "")
eng = setup.ROOT / "engine"
eng.mkdir(exist_ok=True)
shutil.copy2(setup.ROOT / "build-hip" / setup.EXE, eng / setup.EXE)
(eng / "BUILD.json").write_text(json.dumps({"source": "local-hip", "backend": "hip", "archs": [arch],
    "version": setup.source_version(), "lib_dirs": [str(root / "lib")], "vision": "none",
    "src": setup.source_hash(setup.ENGINE_SOURCES)}, indent=1))
PYEOF
RUN rm -rf build-hip

# The image encoder on the GPU.  tools/vision only knows ggml's CUDA backend (STRATA_VISION_CUDA); setup.py has no AMD
# GPU encoder yet.  Same llama.cpp checkout, but with ggml's HIP backend switched on from outside (-DGGML_HIP=ON), the
# ROCm clang as HIP compiler and gfx1201 as the only code target.  The binary is the stock strata-vision: serve.server
# starts it with --gpu from the config's "vision" section (written by entrypoint.sh).
RUN .venv/bin/python - <<'PYEOF'
import json, os, pathlib, shutil
import setup
arch = os.environ.get("GPU_ARCH", "gfx1201")
root = pathlib.Path("/opt/rocm")
bitcode = next(p for p in (root / "lib/llvm/amdgcn/bitcode", root / "amdgcn/bitcode") if p.is_dir())
llama = setup.get_llama_cpp()
setup.cmake_build(setup.ROOT / "tools" / "vision", setup.ROOT / "build-vision", "strata-vision",
    [f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_CUDA=OFF", "-DSTRATA_PORTABLE=OFF", "-DGGML_HIP=ON",
     "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",       # static ggml-hip objects must be PIE-compatible (Ubuntu links PIE)
     f"-DGPU_TARGETS={arch}", f"-DAMDGPU_TARGETS={arch}", f"-DCMAKE_HIP_ARCHITECTURES={arch}",
     f"-DCMAKE_HIP_COMPILER={root / 'llvm/bin/clang++'}", f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root}",
     f"-DCMAKE_PREFIX_PATH={root}", f"-DCMAKE_HIP_FLAGS=--rocm-path={root} --rocm-device-lib-path={bitcode}"],
    None, "")
eng = setup.ROOT / "engine"
shutil.copy2(setup.ROOT / "build-vision" / "bin" / setup.VEXE, eng / setup.VEXE)
meta = json.loads((eng / "BUILD.json").read_text())
meta.update({"vision": "gpu", "vision_src": setup.source_hash(setup.VISION_SOURCES)})
(eng / "BUILD.json").write_text(json.dumps(meta, indent=1))
PYEOF
RUN rm -rf build-vision && ldd engine/strata-vision | grep -E "hip|hsa|rocblas|hipblas" || true

COPY --from=files --chmod=755 entrypoint.sh /usr/local/bin/strata-entrypoint
ENTRYPOINT ["/usr/local/bin/strata-entrypoint"]
