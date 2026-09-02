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
