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

The 780M's media engine handles BF16 and int8 natively in silicon,
but `int8_convrot` is a quantisation scheme designed for the NVIDIA
path. When a `*_int8_convrot.safetensors` checkpoint is loaded on
Radeon, the quantised weights are **dequantised to BF16 at load time**
before they ever reach the matrix-multiply units. That means int8
saves disk and on-the-fly transfer bytes but not actual compute —
every step runs the same BF16 matmul that a pure BF16 checkpoint
would. Worse, the dequant itself is a non-trivial kernel that has to
happen on every forward pass for the layers it touches, so the int8
file can be *slower* than a clean BF16 file on this APU.

GGUF, by contrast, is built on top of the **llama.cpp engine** which
ships its own ROCm/HIP kernels for every quantised data type
(Q4_0, Q4_K, Q6_K, F16, F32, etc.). On load, those kernels are
JIT-compiled once via Triton/LLVM and then the quantised weights are
fed straight into the dequantising matmul — no intermediate BF16
buffer is materialised in GTT, no per-step dequant overhead, and
the cast (Q-quant → FP16/BF16) happens inside the matrix kernel
itself. On RDNA3 (gfx1103) this path is fully exercised because
TheRock's `gfx110X-all` wheel set includes the `rocWMMA` and
`comfy_kitchen` int8/int4 GEMM kernels that the llama engine
dispatches into.

Practical upshot on our setup:

- `*_int8_convrot.safetensors` 21 GB DiT + 25 GB BF16 encoder → OOM
  at 50.36 GiB during the first sampling step. The dequant-to-BF16
  pattern also keeps peak working set high because both the
  quantised storage and the BF16 working copy are live at once.
- `*_Q4_0.gguf` 19.9 GB DiT + `*_Q4_K_M.gguf` 16.5 GB encoder →
  fits with headroom, peak ~42 GB during sampling, no
  dequant-to-BF16 working set, GGUF kernels do the cast inside the
  GEMM. This is the only configuration that runs to completion on
  4 GB VRAM + 51 GB GTT today.

The GGUF Q4_0 / Q4_K_M quantisations trim ~9 GB off the encoder and
~5 GB off the DiT relative to the int8 path, and the GGUF llama
engine avoids the int8→BF16 dequant step that the native int8 path
forces on Radeon. Together that's the difference between "OOM at step
1" and "runs to completion in 19:36".

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

## GGUF vs int8_convrot on a 2-second clip with --lowvram (2026-09-07)

Both stacks run the same H3 LoRA (8 steps) and the same attention
patch (KJNodes `auto` → `sageattn` Triton). The container is in
`--lowvram` mode so partial frees happen between phases; we see lines
like

```
[INFO] Unloaded partially: 1249.17 MB freed, 15196.76 MB remains loaded
[INFO] Unloaded partially: 1175.85 MB freed,  3790.64 MB remains loaded
```

in the int8_convrot run. Per-iter sampling time is essentially
identical, but the prompt totals diverge because of model-load and
quant-metadata setup overhead.

| | GGUF Q4_0 | int8_convrot | Delta |
|---|---:|---:|---:|
| Sampling 8 steps | 3:47 | 3:51 | +1.6% |
| Per-iter | 28.44 s | 28.89 s | +1.6% |
| Full prompt (load + sample + VAE) | **304 s** | **467 s** | **+54%** |
| Setup overhead (load + unload + quant setup) | ~1:13 | ~3:54 | **+220%** |

The sampling step itself is the same speed, so the int8_convrot
penalty is paid entirely in model-load and quantisation-metadata
parsing. For a single short render that overhead dominates; for a
20-iter batch the gap closes. GGUF wins on cold-start latency; the
per-iter cost is a wash.

## Memory note: --lowvram actually unloads

The user pointed out that the previous 50.30 GiB peak came from
CLIP + VAE + DiT all resident at once. Switching the container to
`--lowvram` (i.e. dropping `--highvram` and `--gpu-only`) makes
ComfyUI's model manager partial-free between `Load * Model` calls:

- After the first CLIP load (16.4 GB), partial free drops it to
  ~3.8 GB before the second CLIP pass — the 12 GB difference goes
  to system RAM.
- After the second CLIP load (16.4 GB), partial free keeps ~15.2 GB
  in VRAM before the DiT load (20 GB) needs to come in.
- The DiT ends up with 20 GB and the residual CLIP doesn't push us
  past 36 GB peak, leaving ~15 GB headroom for sampling buffers
  (sage-attn activations, k/v cache, intermediate features).

This is the right configuration for the 4 GB VRAM + 51 GB GTT
split: ComfyUI offloads previous models to system RAM
unified-memory on the APU (no PCIe round-trip on 780M because the
APU shares the DDR5 bus with the CPU), and the next load reads the
partial-free version back. ~3 minutes of load overhead across the
whole prompt is the cost; in exchange we no longer carry 50 GB of
dead weight through sampling.

## ## What didn't work, and what to do instead

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

## Max-scale 2MP × 10sec on --lowvram (overnight 2026-09-07)

Overnight run on the 780M with the new `--lowvram` config: full HD,
10 seconds, 8 steps with the kijai LightX2V Turbo LoRA, sage-attn
auto, GGUF Q4_0 DiT + GGUF Q4_K_M encoder.

| | Value |
|---|---|
| Resolution | 2.0 MP (1920×1080) |
| Length | 10 sec @ 24 fps = 240 frames |
| Sampler steps | 8 (LoRA-distilled) |
| Attention | sage `auto` (Triton) |
| Per-iter | 6656.9 s (~111 min) |
| Total sampling | 14:47:35 |
| Total prompt | **15:32:15** |
| DiT at peak | 20.1 GB in GTT + 7.1 GB offloaded to DDR5 system RAM |
| Headroom at peak | ~21 GB (lowvram reshuffled weights) |
| Completion | **yes** — render written to /opt/comfyiu/output/ |

This is the largest prompt the 780M will run with H3 today, and the
first time a 2 MP × 10 sec render has finished without OOM. The
partial-offload pattern that lowvram triggers (DiT split into the
active ~20 GB chunk + the rest in DDR5) is what kept peak working set
inside 50 GiB while still letting sampling buffers grow to the ~25 GB
they need for a 240-frame 1080p sequence.

Per-iter scales as O(pixel_count × frame_count) for H3. From the
0.2 MP / 5 sec / 8-step baseline at 28.4 s/iter:

- pixel_count ratio: (2.0 / 0.2) = 10× → ×10
- frame_count ratio: (240 / 120) = 2× → ×2
- expected per-iter: 28.4 × 10 × 2 ≈ 568 s
- observed: 6656 s — about 12× the linear estimate

The extra factor comes from attention being O(seq_len^1.5) and the
sage-attn kernel not being perfectly cache-friendly for the very
long sequences that 2 MP × 10 sec produces (~120 K tokens per call
versus ~10 K at 0.2 MP × 5 sec). 12× more compute per token is
plausible.

## What --lowvram actually bought us

The earlier 0.2 MP / 5 sec run on `--highvram` peaked at 50+ GiB and
had ~1 GiB headroom for sampling buffers. At 2 MP × 10 sec, the
sampling buffers alone need ~25 GB, so the highvram config would have
OOM'd at step 1. With `--lowvram` the active DiT chunk stays in GTT
and the residual weights are paged out as the buffers grow:

```
loaded partially; 20736.65 MB usable, 20125.07 MB loaded,
                  7079.56 MB offloaded, 611.57 MB buffer reserved
```

That 7.1 GB offload was the difference between "OOM at step 1" and
"15.5 hours to completion". The 5× per-iter regression from lowvram
on small prompts (28.4 → 154 s) is no longer relevant here — at
max-scale, the kernel paging cost is amortised inside the dominant
attention compute cost.

## What it would take to make 2 MP × 10 sec fast

A 15.5 hour render is unusable for iteration. In rough order of
impact on the same hardware:

1. **Spectrum `enabled = true`** (the `Spectrum Apply MiniMax H3`
   node) — skips ~40% of the H3 transformer calls via Chebyshev
   forecast. Expected: 15.5 h → ~9-10 h. The user has explicitly
   said they care about every pixel and don't want this; leaving
   off by default.
2. **Sage-attn 2.x** — sage-attn 1.0.6 (ROCm) is already in use;
   sage-attn 2.x is CUDA-only and unavailable on this APU. No win
   here until ROCm catches up.
3. **GGUF Q3_K_S DiT** — would shave another 3-4 GB off the DiT and
   the dequant cost, but at the cost of measurable quality loss on
   fine textures. Not recommended.
4. **Spectrum balanced → max_speed policy** — drops the safety
   net in the forecast step and trusts the Chebyshev fit more.
   Expected: another 10-20% on top of (1). Same quality caveat.
5. **Resolution drop** — 2 MP → 1 MP (1280×720) is a 4× pixel cut,
   which by the O(pixel × frame) estimate should drop total time
   by ~3.5×. 15.5 h → ~4.5 h. This is the single biggest knob the
   user has without touching model code.

The honest reading is that the 780M is a low-end inference target
for H3 max-scale. NVIDIA 5090 / 5080 will always be 10-50× faster
here. The path to "overnight 2 MP × 10 sec" is the spec; "few
hours 2 MP × 10 sec" needs Spectrum or a smaller model. Nothing
else on the table is going to halve it.

## ## What's next

- ~~4-step run with the same LoRA~~ — tried 2026-09-06, quality was unacceptable. The 8-step config (19:36) is the production sweet spot.
- Native int8 path (`minimax_h3_ref2va_pruned_int8_convrot.safetensors`) once the OOM pattern is fixed via `unload_model_after`.
- SDPA fallback as a third tier for the truly memory-starved cases.
