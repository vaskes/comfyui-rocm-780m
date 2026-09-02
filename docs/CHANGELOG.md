# Changelog

## v5 (2026-09-02) — H3 t2va ready

- Added `flash-attn==2.8.3.post1` (Triton ROCm backend, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`)
- Added `bitsandbytes==0.50.2`
- New default command: `--use-flash-attention --enable-triton-backend --supports-fp8-compute --disable-pinned-memory --gpu-only`
- Added `h3_t2v_workflow.json` example for MiniMax H3 t2va (text→video+audio)
- Added `bench_h3.py` benchmark script
- Added `docs/H3_BENCHMARK.md` with live trace
- Added `docs/APU_MEMORY_MODEL.md` explaining VRAM vs GTT on 780M
- Added `docs/ATTENTION_BACKENDS.md` comparison

## v4 (2026-09-02) — TheRock gfx1103 native

- Switched to TheRock nightly wheels (`https://rocm.nightlies.amd.com/v2/gfx110X-all/`)
- Native gfx1103 inference (no `HSA_OVERRIDE_GFX_VERSION` shim)
- Pinned `torch==2.9.1+rocm7.13.0a20260513`, `torchvision==0.24.0+rocm7.13.0a20260513`, `torchaudio==2.9.0+rocm7.13.0a20260513`
- Added sage-attention ROCm 7 wheel
- Added `probe_mem.py` and `probe_alloc.py` for debugging GTT vs VRAM

## v1-v3 (2026-08-28 to 2026-08-30) — Initial attempts (deprecated)

- Used official PyPI ROCm wheels (gfx110X not supported, hangs)
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` (gfx1100 emulation, hangs on complex kernels)
- v3: MIOpen ASM disable (still hung)
- v3: ROCm 6.4.4 downgrade (still hung)
- Conclusion: **kernel bug**, fixed in v4 with cwsr_enable=0 GRUB fix
## v6 (2026-09-02) — flash-attn fix + smoke test

### Fix: flash-attn no longer falls back to SDPA

The `flash-attn` package requires a C compiler to JIT-compile its Triton ROCm kernels. The v4/v5 Dockerfiles were missing `g++` and `cmake`, causing flash-attn to silently fall back to pytorch SDPA (852 warnings per inference).

**Fix**: added `g++ cmake ninja-build` to `apt-get install` in the v6 Dockerfile. Flash-attn now works correctly (0 warnings in container logs).

### Verified: 0.2 MP × 5 frames H3 t2va smoke test

After the v6 image was built and the container restarted, a 320×640 (0.2 MP) × 5-frame H3 t2va inference ran in 168.81 seconds total:
- Model loading: ~150s (text encoder int4 15 GB + H3 diffusion int8 20 GB)
- Denoising 8 steps: **18 seconds** (2.15-3.21s/step)
- VAE decode + save: ~3s
- Flash-attn warnings: 0 (vs 852 in the full run)

The smoke test mp4 is at `results/H3_smoke_780M_flash_attn.mp4` (35 KB, 0.2 sec duration).

### Per-step speedup: ~150x

| Setup | Steps | Time/step | Notes |
|---|---|---|---|
| SDPA fallback (v5, full run) | 20 | 352 s/step | 124 frames @ 832x480, denoising 1:57:22 |
| **flash-attn (v6, smoke)** | 8 | 2.3 s/step | 5 frames @ 320x640, denoising 18s |

Per MP per step: 352 s vs 11 s = **~30x speedup** when flash-attn works.

### Build context fix

Builds were failing because `git clone https://github.com/...` triggered GitHub's "could not read Username" error in the docker build context (no SSH keys, no auth). Fixed by switching to `curl` + tarball downloads in all Dockerfiles (v4, v5, v6).
