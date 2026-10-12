# Strata vs Radiance

Same source model (OrcaRouter Qwen3.8-Flash-Next-Uncensored), two different 4-bit builds; one Radeon AI PRO R9700 32 GB, Ryzen 9 7950X, 128 GB DDR5 @3600. Greedy, thinking off. Prefill = prompt tokens / time to first token (fresh prompts, no cache). Decode = output tokens/s after the first token, MTP speculation on.

| | Strata | Radiance |
|---|---|---|
| Model repo | mradermacher/Qwen3.8-Flash-Next-Uncensored-GGUF | StillDeadcode/qwen3.8-next-flash-uncensored-fp8-iq4r-moe |
| Format | IQ4_XS GGUF (98 GB) | .rad IQ4R (122 GB) |
| Quant recipe | IQ4_XS gate/up, IQ4_NL/Q5_1 down experts; Q8_0 dense (BF16 compat projections); MTP draft from base model | 4-bit codebook experts (GPTQ, WHT), int8 W8A8 trunk, FP8 KV; model's own MTP head |
| Load time | 20 s | 65 s |
| VRAM used (loaded) | 31.4 GiB | 31.7 GiB |
| Container RAM | 67.7GiB / 124.7GiB | 98.19GiB / 124.7GiB |
| Prefill 4K | 1,487 tok/s (TTFT 2.8 s) | 1,594 tok/s (TTFT 2.6 s) |
| Prefill 16K | 2,245 tok/s (TTFT 7.4 s) | 1,746 tok/s (TTFT 9.5 s) |
| Prefill 64K | 2,425 tok/s (TTFT 27.9 s) | 1,779 tok/s (TTFT 38.1 s) |
| Prefill 128K | 2,244 tok/s (TTFT 58.8 s) | 1,747 tok/s (TTFT 75.4 s) |
| Prefill 244K | 1,961 tok/s (TTFT 125.9 s) | 1,711 tok/s (TTFT 144.1 s) |
| Decode code | 91.3 tok/s | 65.4 tok/s |
| Decode prose | 72.1 tok/s | 40.7 tok/s |
| Decode prose @240K ctx | 60.4 tok/s | 41.7 tok/s |
| 4 parallel streams | 72.2 tok/s total | 109.5 tok/s total |
| GSM8K | 93.3% (56/60) | 95.0% (57/60) |
| MMLU-Pro | 51.7% (31/60) | 55.0% (33/60) |
| HumanEval pass@1 | 100.0% (50/50) | 98.0% (49/50) |
| Long-context retrieval | 100.0% (3/3) | 100.0% (3/3) |

strata: 2026-10-12 04:00:44 → 2026-10-12 04:24:36
radiance: 2026-10-12 04:25:43 → 2026-10-12 04:48:41
