# APU Memory Model: Why GTT, not VRAM, is the right thing to watch on 780M

## TL;DR

On a Radeon 780M APU, `rocm-smi` and `mem_info_vram_used` will **always show ~107 MiB** even with a 20 GB model in GPU memory. The actual model is in `mem_info_gtt_used`. This is **correct behavior for an APU**, not a bug.

## What's VRAM and what's GTT on an APU?

The 780M has no physical HBM. Both "VRAM" and "GTT" are carved out of system RAM at boot time by BIOS:

| Region | Size (typical) | Source | What lives there |
|---|---|---|---|
| VRAM | 16 GB (BIOS) | system RAM, dedicated to GPU | display buffers, kernel-level GPU state, mode switching |
| GTT | 40 GB (rest of system RAM) | system RAM, mapped to GPU | model tensors, intermediate activations, scratch space |

Both are **physically the same DDR5**. The difference is just allocation policy and access path.

## What we observed

Probing with a synthetic test (`workspace/probe_mem.py`):
- `torch.cuda.get_device_properties(0).total_memory` = 40 GiB (16 GB VRAM + 24 GB visible GTT)
- Allocate 12 GB of fp16 tensors on `cuda:0`:
  - `torch.cuda.memory_allocated()` = 12 GB (looks right)
  - `mem_info_vram_used` (sysfs) = 107 MiB (no change!)
  - `mem_info_gtt_used` (sysfs) = 12 GB+ increase

Conclusion: **TheRock gfx1103 wheels put all torch tensors in GTT, not VRAM.**

## Why does it work?

GPU code paths (conv2d, matmul, attention) only need the data to be in GPU-accessible memory. GTT is GPU-accessible — just accessed through a translation table (GART = Graphics Address Remapping Table) rather than being a fixed GPU-side allocation.

The performance penalty is real (GTT is slightly slower than VRAM for repeated access) but small for compute-bound workloads like diffusion.

## The bug we **didn't** have

Several prior reports of "VRAM=0" on APUs were misdiagnosed as bugs in the kernel driver. They're not. The right metric to watch is `mem_info_gtt_used` (in MiB), not `mem_info_vram_used`.

```bash
# Watch memory pressure
watch -n 1 'echo "vram=$(($(cat /sys/class/drm/card0/device/mem_info_vram_used)/1024/1024)) MiB  gtt=$(($(cat /sys/class/drm/card0/device/mem_info_gtt_used)/1024/1024)) MiB"'

# Or use our gpu_monitor.py
python3 scripts/gpu_monitor.py 60 0.5
```

## Where this matters for MiniMax H3

For H3 t2va, the model is 20 GB. With int4 text encoder (15 GB), the total is 40 GB. We have exactly 40 GB GTT. **It's tight.** Add `--disable-pinned-memory` to save 42 GB of pinned-memory headroom (which we don't need for the inference, only for async data transfers that we're not doing).
