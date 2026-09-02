# MiniMax H3 t2va Benchmark Trace

This is a live trace of the first successful H3 t2va inference on 780M. In progress as of 2026-09-02.

## Setup

- **Model**: `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (20 GB)
- **Text encoder**: `minimaxH3INT4Convrot_qwen3vl32bInt4.safetensors` (15 GB, int4 quantized)
- **Video VAE**: `minimax_h3_video_vae_int8_convrot.safetensors` (3 GB)
- **Audio VAE**: `minimax_h3_audio_vae_fp32.safetensors` (0.6 GB)
- **Workflow**: `workspace/h3_t2v_workflow.json` (832×480, 124 frames = 5 sec, 20 steps, `res_multistep` sampler, `beta` scheduler, seed=42, prompt "A cat sitting on a windowsill, soft morning light, slow pan, calm atmosphere")
- **Flags**: `--use-flash-attention --enable-triton-backend --supports-fp8-compute --disable-pinned-memory --gpu-only`

## Memory budget

| Component | Size |
|---|---|
| Text encoder int4 | 15 GB |
| H3 diffusion int8 | 20 GB |
| Video VAE int8 | 3 GB |
| Audio VAE fp32 | 0.6 GB |
| Intermediates | ~1-2 GB |
| **Total** | **~40 GB** (just fits in our 40 GB GTT budget) |

## Why int4 text encoder (not int8)?

The int8 version is 26 GB. With diffusion (20 GB) + VAE (4 GB) + intermediates, we'd need ~52 GB. We have 40 GB. **OOM guaranteed.**

Int4 saves 11 GB at small quality cost. For text encoding, int4 is fine — the encoder outputs conditioning vectors, not pixels.

## Timeline

| Time | Event | GPU% | GTT | Notes |
|---|---|---|---|---|
| t=0 | Submit | 0 | 200 MB | Prompt accepted, prompt_id assigned |
| t=0-2s | Text encoder loading | 0-5% | 200 MB → 15 GB | qwen3vl int4 reading from disk |
| t=2-10s | H3 diffusion loading | 0-15% | 15 GB → 20 GB | Paged load from disk |
| t=10-30s | Warmup, scheduler setup | 15-40% | 25-30 GB | MIOpen search, kernel compile |
| t=30-50s | Denoising starts | 99% | 28-35 GB | First step, GTT grows as activations cache |
| t=50-2000s | Denoising 20 steps | 95-99% | 35-40 GB | GPU saturated, ~100s per step |
| t=2000+s | VAE decode (video+audio) | TBD | TBD | Decodes 124 frames |
| TBD | Save mp4 | TBD | TBD | ffmpeg encode to h264 |

**ETA for 20 steps on 780M**: ~1.5-3 hours (vs ~5-10 min on RTX 4090).

## Notes

- The 780M has only 6 CUs (RDNA3, RDNA2 had 8-12 in similar APUs)
- H3 is a packed DiT (joint video+audio), much bigger than SD 1.5 UNet
- The bottleneck is **compute**, not memory or I/O
- 99% GPU utilization confirms we're compute-bound (not waiting on disk or offload)

## Memory pressure recipe (the bits that work)

```bash
# MUST: disable pinned memory to free 42 GB
--disable-pinned-memory

# MUST: keep everything on GPU (we have enough GTT)
--gpu-only

# MUST: pick attention backend (pick ONE)
--use-flash-attention       # works for H3 (headdim 64-128)
# OR --use-ck-attention      # also works, slightly slower on H3
# OR --use-sage-attention    # headdim must match — H3 might be OK

# OPTIONAL: fp8 compute (works for H3, not faster than fp16 for our small batch)
--supports-fp8-compute
--enable-triton-backend
```

## What doesn't work

- **int8 text encoder** (26 GB) — OOM
- **`--pinned-memory`** enabled — adds 42 GB pinned buffer, OOM
- **`--lowvram`** — tries to offload to CPU, we have 16 GB CPU RAM but ComfyUI's offload pattern doesn't free GTT fast enough
- **All custom nodes from the ref2va workflow** (PathchSageAttentionKJ, MiniMaxH3MemoryEfficientSageAttentionPatch, SpectrumApplyMiniMaxH3) — these are dreamline-specific custom nodes that aren't in the public ComfyUI or the official H3 release. Standard ComfyUI nodes work fine for t2va.
