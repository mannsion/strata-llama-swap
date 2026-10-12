# Runtime for model-written code: python 3.12 + numpy (the EvalPlus tests compare with np.allclose).
# bench.py runs it with --network none --read-only --user 65534 --memory 1g --pids-limit 256.
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.2.6
