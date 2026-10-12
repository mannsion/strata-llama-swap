# strata-llama-swap

Run **Qwen3.8-Flash-Next-Uncensored** (a large mixture-of-experts model) on a single **AMD Radeon AI PRO R9700** at 70–90 tokens/s,
with 256K context and image input, behind one OpenAI- and Anthropic-compatible API.

- **Engine:** [Strata](https://github.com/Niko1221/Strata), built for ROCm / gfx1201. Experts that don't fit in the 32 GB
  of VRAM are served from system RAM.
- **Front door:** [llama-swap](https://github.com/mostlygeek/llama-swap). It lists the models and starts whichever one a
  request names.
- **Vision on the GPU:** Strata's image encoder is built with ggml's HIP backend. Upstream only ships CUDA or CPU.
- **Everything runs in Docker**, with a single script, `./llm`.

## Models

| Model id | Build | Size |
|---|---|---|
| `Qwen3.8-Flash-Next-Uncensored-i1-IQ4_XS` **(default)** | [mradermacher i1-IQ4_XS](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-Uncensored-i1-GGUF), imatrix-calibrated | 97 GB |
| `Qwen3.8-Flash-Next-Uncensored-IQ4_XS` | [mradermacher IQ4_XS](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-Uncensored-GGUF), static | 98 GB |

One model is loaded at a time. A request for the other one swaps it in, which takes about 30 s.

## Performance

R9700 + Ryzen 9 7950X + 128 GB DDR5 at 3600 MT/s, thinking off, greedy decoding:

| | |
|---|---|
| Output, prose | ~72–77 tok/s |
| Output, code | ~88–91 tok/s |
| Output at 240K context | ~60 tok/s |
| Prompt reading | 1,500 (4K) – 2,400 (64K) tok/s |
| First token | 0.4 s for short prompts · ~1.3 s with an image · ~2 min for a 250K prompt |
| Memory | 31 GiB VRAM · ~68 GiB RAM |
| Accuracy (fast run) | GSM8K 93% · HumanEval 100% · MMLU-Pro 52% · needle in a haystack to 260K: 3/3 |

Strata was also compared head to head with [Radiance](https://codeberg.org/StillDeadcode/radiance) on the same model.
Strata was 40–77% faster at output and used 30 GiB less RAM, with the same accuracy. Full table:
**[results/RESULTS.md](results/RESULTS.md)**.

Output speed is limited mostly by **system RAM bandwidth**, because the experts that don't fit in VRAM are read from RAM
for every token. Two DIMMs running at their rated EXPO speed beat four running at 3600.

## Requirements

- An AMD Radeon AI PRO R9700 (gfx1201) on Linux, with the `amdgpu` kernel driver
- Docker with the compose plugin
- **≥ 96 GB of system RAM** (about 68 GiB is used), and about 200 GB of disk
- The [`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) for the downloads

## Quick start

```sh
git clone --recursive https://github.com/mannsion/strata-llama-swap.git && cd strata-llama-swap
./llm setup       # writes .env: finds the R9700 (not the iGPU), your groups and paths
./llm download    # ~195 GB from Hugging Face
./llm build       # Strata for gfx1201 (~20 min), llama-swap, the bench sandbox
./llm up          # http://127.0.0.1:8000/v1
```

The first start packs the model once and fetches Strata's MTP draft layer (~5 GB), which takes a few minutes. Later
starts take about 30 s.

To serve other machines, set `BIND_ADDR` in `.env` to the machine's LAN address and list the host names clients will use
in `ALLOWED_HOSTS`. Then open the port in your firewall. **There is no API key**, so only do this on a network you trust.

## API

| | |
|---|---|
| `GET /v1/models` | Both models, with `architecture.input_modalities`, `context_length` and tool support |
| `POST /v1/chat/completions` | OpenAI Chat Completions, streaming |
| `POST /v1/responses` | OpenAI Responses |
| `POST /v1/messages` | Anthropic Messages (+ `/v1/messages/count_tokens`) |
| `GET /running` | Which model is loaded |

Images go in as base64 `data:` URLs (`image_url` / `input_image` / Anthropic `image` blocks).

Thinking is on by default. To turn it off:

| API | Field |
|---|---|
| Chat Completions | `"chat_template_kwargs": {"enable_thinking": false}` |
| Responses | `"reasoning": {"effort": "none"}` |
| Messages | `"thinking": {"type": "disabled"}` |

```sh
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next-Uncensored-i1-IQ4_XS",
  "messages": [{"role": "user", "content": "Hello"}],
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Send one request at a time. Parallel requests don't add throughput on this setup.

## How it works

```
client ──► llama-swap  (BIND_ADDR:8000)
              │  reads "model", stops the running model, docker compose up's the requested one, waits for /health
              ▼
           Strata container  (127.0.0.1 only) ── R9700 (by PCI path) + experts in system RAM
```

- `compose.yaml`: llama-swap plus one service per model. The model services are only ever started by llama-swap.
- `swap/config.yaml`: the model list, its capabilities, and the default model that's loaded at startup.
- `strata/`: the Strata image (ROCm 7.2.4, gfx1201, HIP vision encoder) and its entrypoint.
- `bench/`: the benchmark.
- `results/`: published results.

## Benchmark

```sh
./llm bench-data              # once: GSM8K, MATH-500, MMLU-Pro, HumanEval+, MBPP+, MMStar, ChartQA (pinned revisions)
./llm bench --mode fast       # speed + all accuracy suites; --mode full for bigger samples, --quick to smoke-test
MODEL=Qwen3.8-Flash-Next-Uncensored-IQ4_XS ./llm bench --only speed,mmstar
```

It covers speed (prefill 1K–250K, decode at 0/16K/245K context, parallel streams, time to first token with an image),
math, knowledge, code, vision, stories (with checked rules) and long-context retrieval. Model-written code runs in a
network-less, read-only sandbox container. Results are written to `results/`.

## Commands

| | |
|---|---|
| `./llm up` / `stop` | Start / stop the API and the loaded model |
| `./llm status` | Containers and the loaded model |
| `./llm logs [service]` | llama-swap (default), `strata-i1-iq4_xs` or `strata-iq4_xs` |
| `./llm build` | Rebuild the images |

## Troubleshooting

- **`403 Host '…' is not allowed`**: add the host name clients use to `ALLOWED_HOSTS` in `.env`. Requests by IP address
  always work.
- **Unreachable from other machines**: check `BIND_ADDR`, then your firewall. Containers use host networking, so the
  port is the host's own port.
- **Slow output, or swapping**: check free RAM (`free -g`). The model needs about 68 GiB besides the OS.
- **Docker builds can't reach the internet**: the builds already use `--network host`, for hosts whose firewall drops
  Docker's bridge traffic.
