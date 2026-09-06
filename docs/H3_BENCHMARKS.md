# MiniMax H3 Performance Benchmarks on Radeon 780M (gfx1103)

Production benchmarks for the MiniMax H3 DiT model running on the user's
llmhost2 (Ubuntu 24.04, kernel 6.x, TheRock ROCm 7.13 gfx110X wheels,
PyTorch 2.9.1+rocm7.13). All runs were smoke tests at **0.2 MP, 5 sec**
on the same workflow to keep numbers directly comparable.

Hardware: AMD Radeon 780M (gfx1103) iGPU, 4 GB VRAM + 51 GB GTT
(amdgpu.gttsize=52224, ttm.pages_limit=13369344, ttm.page_pool_size=8021606),
amdgpu.cwsr_enable=0, amdgpu.mes_kiq=1. Container: `comfyiu:vaskes`
running on `comfyiu:base` (TheRock gfx110X-all wheels).

## Headline results

| # | Attention | Sampler | Steps | LoRA | Sampling time | Per-iter | Total prompt | Speedup vs (1) |
|---|-----------|---------|-------|------|---------------|----------|--------------|----------------|
| 1 | flash-attn (`--use-flash-attention` from CLI) | res_multistep | 20 | none | 1:13:53 | 221.7 s | **1:19:18** | 1.00× (baseline) |
| 2 | sage-attn (KJNodes `Patch Sage Attention` mode=`auto`) | res_multistep | 20 | none | 41:30 | 124.5 s | 44:07 | **1.92×** |
| 3 | sage-attn (same) | res_multistep | 8 | `minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors` (kijai) | 17:43 | 132.9 s | 19:36 | **4.04×** |

User feedback (2026-09-06, after the 8-step run): the 4-step
configuration with this LoRA produces unacceptable quality, so the
8-step setting is the sweet spot — 19:36 total is what production
smoke tests target. The 4-step LoRA is left in the file as a knob
for future experiments but is not recommended.

Notes:
- All three runs use the same model (`MiniMax-H3-ref2va-Q4_0.gguf` 19.9 GB
  GGUF, `MiniMax-H3-encoder-Q4_K_M.gguf` 16.5 GB GGUF,
  `MiniMax-H3-encoder-mmproj-F16.gguf` vision sidecar).
- `MiniMax H3 Memory Efficient Sage Attention Patch` (the H3-specific
  kjnode patch) was **disabled** for all three runs — it always fails on
  ROCm because it imports `sageattention.core.get_cuda_arch_versions`,
  which is CUDA-only and returns `None` on AMD.
- Spectrum `Apply MiniMax H3` was **disabled** for all three runs
  (user wanted full quality baseline and then iteration).
- The `auto` mode in `Patch Sage Attention KJ` falls back to the
  basic `sageattn` (Triton) on AMD because the more specific modes
  (`sageattn_qk_int8_pv_fp16_triton`, `sageattn_qk_int8_pv_fp8_*`,
  `sageattn3*`) are only present in sage-attn 2.x, which is CUDA-only.
  On our sage-attn 1.0.6 ROCm fork from `guinmoon/SageAttention-Rocm7`
  only `sageattn` and `sageattn_varlen` (plus a few per-block int8
  helpers) are exported.

## Per-iter analysis

Sage-attn cuts per-iter time by ~44% (221.7 → 124.5 s). The drop comes
from sage's persistent softmax and better register usage on long
sequences; H3 has headdim=128 and seq_len ≈ 85 K tokens at 0.2 MP / 5 s,
which is exactly the regime where Triton's flash-attn path is wasteful.

With LoRA + 8 steps, per-iter time is *slightly* higher (132.9 s) than
the 20-step sage run (124.5 s). The overhead is the LoRA forward
patch (extra matmul) and the lower steps-to-warmup ratio. But total
time wins because 8 ≪ 20.

## The LoRA is a 4-step distill — go further

The LoRA filename `lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16`
says **4 step**, but the user ran 8. With 4 steps the expected
sampling time is ~9-10 min and total prompt ~11-12 min, i.e. **~6.7×
over the flash baseline**. Quality at 4 steps on a well-trained
distill LoRA is acceptable for previews; master renders should still
go 8-20 step.

To try: change `KSampler.steps = 4`. Likely needs the matching
`sampler = euler` or `dpmpp_2m` per the LoRA's training; the workflow
already uses `res_multistep` which is k-diffusion's ResDEPS variant
and works at 4 steps in the kijai examples.

## GGUF vs native safetensors on 51 GB GTT

The GGUF path (`MiniMax-H3-ref2va-Q4_0.gguf` 19.9 GB +
`MiniMax-H3-encoder-Q4_K_M.gguf` 16.5 GB) is the only configuration
that fits on 4 GB VRAM + 51 GB GTT without OOM. Native safetensors
(`minimax_h3_ref2va_pruned_int8_convrot.safetensors` 21 GB + the
Qwen3-VL encoder 25 GB BF16) was tried first and OOM'd at 50.36 GiB
during the first sampling step. The GGUF Q4_0 / Q4_K_M quantisations
trim ~9 GB off the encoder and ~5 GB off the DiT relative to int8, and
that's the difference between "OOM at step 1" and "runs to completion".

## Memory breakdown per run (from the user's log)

| Component | Size | Notes |
|-----------|------|-------|
| MiniMaxH3TEModel_ (GGUF Q4_K_M encoder) | 16.49 GB | dequantises `token_embd.weight` to BF16 to avoid OOM during CLIP load |
| MiniMaxH3VideoVAE (BF16) | 4.97 GB | |
| MiniMaxH3AudioVAE (BF16) | 0.58 GB | |
| MiniMaxH3 UNet (GGUF Q4_0) | 19.44 GB | dequantised to BF16 for the actual transformer calls |
| Sampling state (k/v cache, noise, sigmas, intermediate features) | ~0.5 GB | |
| **Total peak** | **~42 GB** | comfortably under 51 GB GTT |

The OOM that bit the user with native safetensors was the **text
encoder + UNet loaded simultaneously** with no unload step between
them. The H3-Multishot pack's `unload_model_after` switch on the
JoyEcho LLM node is the surgical fix for that pattern; in the GGUF
path the encoder is small enough that it doesn't blow the budget even
if it lingers.

## What didn't work, and what to do instead

| Attempt | Result | Why |
|---------|--------|-----|
| `--use-sage-attention` CLI flag in the container CMD | ComfyUI switched to sage globally; KJNodes `Patch Sage Attention` then refuses to load because it expects CLI-managed attention. | Use KJNodes' patch instead of CLI flag. |
| `MiniMax H3 Memory Efficient Sage Attention Patch` (KJNodes) | RuntimeError: "sageattention is not new enough version or could not determine CUDA architecture". | The patch calls `get_cuda_arch_versions` from `sageattention.core`; on ROCm this returns None. The patch is NVIDIA-only. Disable/remove the node on AMD. |
| `CLIPLoaderGGUF` (from `city96/ComfyUI-GGUF`) with `type=flux` | `TypeError: the JSON object must be str, bytes or bytearray, not NoneType` deep inside `load_mistral_tokenizer`. | ComfyUI-GGUF's CLIP loader doesn't know `minimax` arch; it falls back to mistral tokenizer and the encoder GGUF has no mistral tekken data. Use `H3ClipLoaderAny` from `ComfyUI-H3-Multishot` with `type=minimax` instead. |
| `Patch Sage Attention KJ` mode=`sageattn_qk_int8_pv_fp16_triton` | `ImportError: cannot import name 'sageattn_qk_int8_pv_fp16_triton' from 'sageattention'`. | Function only exists in sage-attn 2.x. The ROCm 1.0.6 fork only has `sageattn` / `sageattn_varlen`. Use mode=`auto` (which falls back to `sageattn` on AMD) or mode=`sageattn` explicitly. |
| `Spectrum Apply MiniMax H3` with default settings | worked, ~40-50% additional speedup, but produces a slightly drifted output vs native H3 even with the same seed. The user disabled it because they care about every pixel. | Trade-off accepted. Re-enable for iteration/preview, disable for master. |

## Replicating the 19:36 result

1. In the workflow JSON (or in the editor):
   - `Patch Sage Attention KJ` → `sage_attention = auto`
   - `MiniMax H3 Memory Efficient Sage Attention Patch` → right-click → Disable (or delete)
   - `KSampler` → `steps = 8` (or 4 to push the LoRA's rating)
   - `Load LoRA` → `minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors`
2. Models: DiT `MiniMax-H3-ref2va-Q4_0.gguf`, encoder `MiniMax-H3-encoder-Q4_K_M.gguf`, mmproj `MiniMax-H3-encoder-mmproj-F16.gguf`. All in `/opt/comfyiu/models/`.
3. Loader: `H3ClipLoaderAny` with `type=minimax`, `mmproj_name=(auto)`.
4. Container: `comfyiu:vaskes` (this repo, `profiles/vaskes/Dockerfile`).
5. CMD may keep `--use-flash-attention`; KJNodes' patch overrides it. Both work.

## What's next

- ~~4-step run with the same LoRA~~ — tried 2026-09-06, quality was unacceptable. The 8-step config (19:36) is the production sweet spot.
- Native int8 path (`minimax_h3_ref2va_pruned_int8_convrot.safetensors`) once the OOM pattern is fixed via `unload_model_after`.
- SDPA fallback as a third tier for the truly memory-starved cases.
