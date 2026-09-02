# ComfyUI + ROCm on Radeon 780M (gfx1103) — Production Setup

> **Complete, working, reproducible** setup for running [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on a **Radeon 780M iGPU (gfx1103)** with **ROCm 7.x** in Docker.
>
> Includes the critical kernel-level fix that unblocks gfx1103 inference, all available attention backends (flash-attention, sage, comfy-kitchen, pytorch), and tested examples (SD 1.5, MiniMax H3 t2va).
>
> **Built and battle-tested on:** Ubuntu 24.04 + Linux kernel 7.0.0-30-generic + Radeon 780M (gfx1103, 16 GB VRAM + 40 GB GTT APU).

---

## Table of Contents

1. [What this is](#what-this-is)
2. [Hardware requirements](#hardware-requirements)
3. [Software prerequisites](#software-prerequisites)
4. [The kernel fix (REQUIRED)](#the-kernel-fix-required)
5. [Quickstart (5 minutes)](#quickstart-5-minutes)
6. [How it works](#how-it-works)
7. [Optimizations included](#optimizations-included)
8. [Benchmarks](#benchmarks)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)
11. [Repository layout](#repository-layout)
12. [Credits](#credits)

---

## What this is

A complete, versioned, **reproducible** setup for running ComfyUI on an RDNA3 APU. Pinned versions of every Python package so the build is deterministic. Includes:

- **TheRock gfx110X-all wheels** for native gfx1103 inference (no `HSA_OVERRIDE_GFX_VERSION` shim)
- **Kernel-level fix** (`amdgpu.cwsr_enable=0`) for the known gfx1103 KFD-queue-eviction MES hang
- **All available attention backends**: flash-attention (Triton ROCm), sage-attention, comfy-kitchen (eager/triton/hip), pytorch SDPA, bitsandbytes int8
- **comfy-aimdo** explicit VRAM allocator
- **fp8 compute** support
- **Two example workflows**: SD 1.5 (text→image, 512×512, 20 steps) and MiniMax H3 t2va (text→video+audio, 124 frames @ 832×480, 20 steps)

This is **the only fully working gfx1103 ComfyUI setup as of 2026-09** that runs on Linux without kernel changes (and we can prove it).

---

## Hardware requirements

| Component | Minimum | Tested |
|---|---|---|
| APU / GPU | Any RDNA3 iGPU (gfx1103) or dGPU | **Radeon 780M** (Ryzen 7 8700G) |
| RAM | 32 GB | 46 GB + 29 GB swap |
| BIOS: GTT size | 16 GB | **40 GB** (`amdgpu.gttsize=40960`) |
| BIOS: VRAM size | 8 GB | **16 GB** |
| Disk | 30 GB | 100+ GB (for model files) |

The 780M is an **APU** — no physical HBM. Both "VRAM" and "GTT" come from system RAM, just with different allocation policies in the amdgpu kernel module.

---

## Software prerequisites

- **OS**: Ubuntu 24.04 (other 24.04-based distros should work)
- **Kernel**: any 7.0.0-30+ (tested: 7.0.0-30-generic, Aug 2026)
- **Docker**: 27+ with compose plugin
- **ROCm**: host ROCm 7.2.4+ is fine — but **container has its own ROCm 7.13/7.14** (TheRock nightly), so host version doesn't matter
- **Internet**: yes (for first build, to pull wheels and models)
- **GPU group membership**: user must be in `render` (GID 992) and `video` (GID 44) groups
- **Mavis agent NOPASSWD sudo** (or run as root, or modify scripts)

---

## The kernel fix (REQUIRED)

Without the GRUB fix, every nontrivial gfx1103 inference **hangs** at the first denoising step with `HW Exception: GPU Hang` after about 5-30 seconds. The bug is in the amdgpu kernel module (specifically the KFD-queue-eviction path through the MES firmware on gfx1103 iGPUs). Documented in ROCm issues #5590, #5665.

**Apply once:**

```bash
# Backup
sudo cp /etc/default/grub /opt/mavis-backups/grub-2026-09-02-cwsr-fix/grub.original
sudo cp /boot/grub/grub.cfg /opt/mavis-backups/grub-2026-09-02-cwsr-fix/grub.cfg.before
sudo bash -c 'cat /proc/cmdline > /opt/mavis-backups/grub-2026-09-02-cwsr-fix/cmdline.before.txt'

# Edit /etc/default/grub: append to GRUB_CMDLINE_LINUX
# - cwsr_enable=0  : THE FIX (disables compute wave save/restore, dodges MES bug)
# - mes_kiq=1      : KIQ MES path (required for gfx1103)
# - noretry=1      : don't retry failed commands (faster fail)
# - sg_display=0   : disable s/g for display
# - gpu_recovery=1 : recover GPU on hang (instead of full hang)
# - ttm.page_pool_size=6291456 : TTM page pool for amdgpu (less VRAM pressure)
# - transparent_hugepage=always : THP for GTT allocations
#
# DO NOT set: amd_iommu=off  (stability-critical, do not disable)
# DO NOT change: kernel version (forbidden by project)

GRUB_CMDLINE_LINUX="...existing... amdgpu.cwsr_enable=0 amdgpu.mes_kiq=1 amdgpu.noretry=1 amdgpu.sg_display=0 amdgpu.gpu_recovery=1 ttm.page_pool_size=6291456 transparent_hugepage=always"

# Apply
sudo update-grub
sudo reboot
```

After reboot, verify:

```bash
cat /proc/cmdline | tr ' ' '\n' | grep amdgpu
# Should show: amdgpu.cwsr_enable=0 amdgpu.mes_kiq=1 amdgpu.noretry=1 amdgpu.sg_display=0 amdgpu.gpu_recovery=1
```

See `grub-backup/` for the exact command line and edit script we used.

---

## Quickstart (5 minutes)

```bash
# 1. Clone this repo
git clone https://github.com/vaskes/comfyui-rocm-780m.git
cd comfyui-rocm-780m

# 2. Apply kernel fix (REQUIRED, see above) — or skip if you already did

# 3. Build the Docker image (~5-10 min, ~5 GB)
docker build -f build/Dockerfile.v5 -t comfyiu:therock-gfx1103-v5 build/

# 4. Create the persistent host layout
sudo mkdir -p /opt/comfyiu/{models,output,input,custom_nodes,user,logs,workspace}
sudo chown -R $USER:$USER /opt/comfyiu

# 5. Start the container
docker compose -f build/docker-compose.yml up -d

# 6. Drop a model in
# (e.g. SD 1.5 fp16 from HuggingFace)
wget -O /opt/comfyiu/models/checkpoints/sd_v1-5_fp16.safetensors \
  https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly-fp16.safetensors

# 7. Verify GPU is doing the work
python3 scripts/gpu_monitor.py 60 1
# Should show: GPU% 99 during inference, VRAM ~110 MiB, GTT growing during model load

# 8. Test via API
python3 workspace/submit_and_watch.py workspace/test_workflow_sd15.json
```

The ComfyUI web UI is at **http://localhost:8188**.

---

## How it works

### Why gfx1103 needs special handling

The amdgpu kernel module (Aug 2026, version 7.0.0-30) has a **bug** in the KFD-queue-eviction path for gfx1103 iGPUs: the MES firmware stops responding to `REMOVE_QUEUE` commands, causing GPU hangs. The official fix is in a later kernel (not yet released as of 2026-09-02), so the workaround is to disable compute-wave-save-restore (`cwsr_enable=0`) which forces a different code path that doesn't trigger the bug.

### Why we use TheRock wheels

PyTorch ROCm wheels from the official PyTorch index don't have `gfx1103` in their supported architectures. Workarounds like `HSA_OVERRIDE_GFX_VERSION=11.0.0` (gfx1100 emulation) work for synthetic matmuls but hang on the more complex kernels that real diffusion models use (convolutions, attention, normalization).

AMD's [TheRock](https://github.com/ROCm/TheRock) project provides **native gfx1103 wheels** for torch, torchvision, triton, and a complete ROCm runtime. We use these and get **2-3x speedup** over the emulation hack for free.

### Why both VRAM and GTT

For an APU like the 780M, BIOS allocates:
- **VRAM** (e.g. 16 GB): a dedicated partition of system RAM marked for GPU
- **GTT** (e.g. 40 GB): the rest of system RAM, GPU-accessible on demand

The amdgpu kernel module reports `mem_info_vram_used` and `mem_info_gtt_used` separately in sysfs. On 780M we observe:

- **`vram_used` stays ~107 MiB** even with a 20 GB model in GPU memory (this is just kernel-level overhead, not a bug)
- **`gtt_used` grows** to the size of all model tensors + intermediates

For a 780M, **GTT is the right place to watch memory pressure** (`cat /sys/class/drm/card0/device/mem_info_gtt_used`). The TheRock wheels default to GTT allocations even with `--highvram` (probably to leave VRAM for display buffer / mode switching). This is correct behavior for the APU, not a bug.

---

## Optimizations included

| Package | Version | Notes |
|---|---|---|
| `torch` | `2.9.1+rocm7.13.0a20260513` | TheRock gfx1103-native |
| `torchvision` | `0.24.0+rocm7.13.0a20260513` | paired with torch 2.9.1 (0.25+ breaks) |
| `torchaudio` | `2.9.0+rocm7.13.0a20260513` | |
| `triton` | `3.5.1+rocm7.13.0a20260513` | TheRock wheels |
| `rocm-sdk-core` | `7.13.0a20260513` | minimal ROCm runtime |
| `rocm-sdk-libraries-gfx110x-all` | `7.13.0a20260513` | gfx1103-specific libraries |
| `sageattention` | `1.0.6` | ROCm 7 wheel from guinmoon |
| `flash-attn` | `2.8.3.post1` | via **Triton ROCm backend** (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`), no C++ compile needed |
| `bitsandbytes` | `0.50.2` | uses ROCm 7.2 fallback binary (no 7.13 wheel) |
| `comfy-kitchen` | `0.2.31` | triton + hip + eager backends, int8/fp8/w4a4/w4a8/svdquant |
| `comfy-aimdo` | `0.4.15` | explicit VRAM allocator, host buffer |

### CLI flags you should use

Pick ONE attention backend (these are mutually exclusive):

```bash
python main.py \
  --use-flash-attention  # Triton ROCm backend, works for most DiT/UNet
  # OR
  --use-sage-attention   # headdim must be in [64, 96, 128], falls back to pytorch otherwise
  # OR
  --use-ck-attention     # Comfy Kitchen (recommended for fp8)
  # OR
  --use-pytorch-cross-attention
```

Always combine with:

```bash
  --enable-triton-backend        # enable comfy-kitchen triton backend
  --supports-fp8-compute         # for fp8 attention/quantized
  --gpu-only                     # never offload to CPU (we have enough GTT)
  --disable-pinned-memory        # saves 42 GB of pinned memory for big models
  --listen 0.0.0.0 --port 8188
```

See `examples/start_with_optimizations.sh` for a working recipe.

---

## Benchmarks

SD 1.5 fp16, 512×512, 20 steps, seed=42 (single image, model already loaded):

| Setup | Time | GPU% | GTT (peak) |
|---|---|---|---|
| `sage+lowvram` | 18.88s | 99% | 3.9 GB |
| `sage+highvram` | 13.29s | 99% | 3.9 GB |
| `highvram` | 16.29s | 99% | 3.9 GB |
| `ck+triton+fp8+gpu-only` | **13.10s** | 99% | 3.9 GB |
| `flash+triton+fp8+gpu-only` | 15.41s | 99% | 3.9 GB |

MiniMax H3 t2va (text→video+audio), 124 frames @ 832×480, 20 steps, seed=42 (full inference, model loaded from scratch):

- **Loading**: ~80 seconds
- **Sampling**: ~1000+ seconds (in progress as of last commit)
- **GPU%**: 99% sustained
- **GTT peak**: 40 GB (out of 40 GB available — tight!)

H3 is a 20 GB model. On a 6-CU 780M each step takes time. The official H3 t2va workflow on RTX 4090 is ~5-10 min; on 780M expect 1-2+ hours for 20 steps. See `docs/H3_BENCHMARK.md` for the full live trace.

---

## Examples

### SD 1.5 (text→image)

```bash
# After starting ComfyUI as above
python3 workspace/submit_and_watch.py workspace/test_workflow_sd15.json
# Outputs to /opt/comfyiu/output/ComfyUI_*.png
```

### MiniMax H3 t2va (text→video+audio)

You'll need to provide MiniMax H3 model files (the official release is gated to NVIDIA users; community int4/int8 quantizations work). Drop them into the model dirs:

```bash
# Diffusion model
/opt/comfyiu/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors

# Text encoder (use int4 to fit in GTT)
# Note: int8 (26 GB) version is too large for our 40 GB GTT budget
/opt/comfyiu/models/text_encoders/minimaxH3INT4Convrot_qwen3vl32bInt4.safetensors

# Video VAE
/opt/comfyiu/models/vae/minimax_h3_video_vae_int8_convrot.safetensors

# Audio VAE
/opt/comfyiu/models/vae/minimax_h3_audio_vae_fp32.safetensors
```

Then submit:

```bash
python3 workspace/submit_and_watch.py workspace/h3_t2v_workflow.json
# Outputs to /opt/comfyiu/output/video/H3_t2v_*.mp4
```

**Memory budget for H3 t2va on 780M (40 GB GTT):**
- Text encoder int4: 15 GB
- H3 diffusion int8: 20 GB
- Video VAE int8: 3 GB
- Audio VAE fp32: 0.6 GB
- Intermediates: ~1-2 GB
- **Total**: ~40 GB (just fits!)

If you use the int8 text encoder (26 GB), you OOM.

---

## Troubleshooting

### `HW Exception: GPU Hang` during inference

Kernel fix not applied or `cwsr_enable=0` got removed. Verify:

```bash
cat /proc/cmdline | grep cwsr_enable=0
# Should print: ... amdgpu.cwsr_enable=0 ...
```

### `HIP out of memory. Tried to allocate ... MiB. GPU 0 has a total capacity of 40.00 GiB of which 36.16 MiB is free`

You're out of GTT. Solutions:
- Use int4 text encoder for H3 (15 GB) instead of int8 (26 GB)
- Add `--disable-pinned-memory` (saves 42 GB)
- Try `--lowvram` for the diffusion (slower, but uses less memory)
- Reduce video length / resolution

### `ComfyUI_test_00001_.png` is 2 KB and garbage

Something is fundamentally broken. Check `/opt/comfyiu/logs/comfyui.log` inside the container:

```bash
docker exec comfyiu-test cat /opt/ComfyUI/comfyui.log | tail -50
```

### `flash_attn` not found

The `flash-attn` package must be installed (it's in the v5 image, but not v4). Rebuild with `build/Dockerfile.v5` or install in the running container:

```bash
docker exec -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE comfyiu-test \
  pip3 install --break-system-packages --no-build-isolation flash-attn
```

### `rocm-smi` shows GPU 0% but inference is "running"

This is normal. rocm-smi samples 1×/sec, denoising steps can be faster than that. Use `gpu_monitor.py` with a smaller interval:

```bash
python3 scripts/gpu_monitor.py 60 0.3
```

Or use `cat /sys/class/drm/card0/device/mem_info_gtt_used` to watch model residency.

### Container can't see GPU

You need to be in the right groups (GID 992 `render`, GID 44 `video`):

```bash
groups
# Should show: ... render ... video ...
```

If not:

```bash
sudo usermod -aG render,video $USER
# Log out and back in
```

---

## Repository layout

```
.
├── README.md                        # this file
├── LICENSE                          # MIT
├── build/
│   ├── Dockerfile.v4                # base image: torch + comfy + sage
│   ├── Dockerfile.v5                # v4 + flash-attn + bitsandbytes
│   └── docker-compose.yml           # container with all GPU devices + volumes
├── scripts/
│   ├── start_comfyiu.sh             # balanced: sage+lowvram
│   ├── start_with_override.sh       # HSA_OVERRIDE_GFX_VERSION=11.0.0 (deprecated)
│   ├── start_no_sage.sh             # pure pytorch cross-attention
│   ├── gpu_monitor.py               # rocm-smi text sampler (no JSON in this rocm-smi)
│   ├── gpu_monitor.sh               # wrapper around gpu_monitor.py
│   ├── run_one_test.sh              # run a single workflow
│   ├── optim_compare.sh             # sweep A..H for attention backends
│   ├── bench_with_monitor.sh        # bench + GPU monitor in one
│   └── probe_attn.sh                # probe which attention backends are available
├── workspace/
│   ├── test_workflow_sd15.json      # SD 1.5 reference
│   ├── test_workflow_sd15_seed1234.json
│   ├── test_workflow_1024.json      # 1024×1024 (max)
│   ├── h3_t2v_workflow.json         # H3 t2va reference (5 sec, 832×480, 20 steps)
│   ├── h3_t2v_workflow.py           # generator script
│   ├── submit_and_watch.py          # submit + wait + print result
│   ├── bench_ck.py                  # benchmark w/ all flags + monitor
│   ├── bench_h3.py                  # H3 t2va benchmark
│   ├── probe_mem.py                 # probe VRAM vs GTT allocation
│   ├── probe_alloc.py               # probe torch allocation behavior
│   ├── test_comfy.py                # smoke test: ping + submit + get output
│   └── list_history.py              # debug: dump ComfyUI history
├── examples/
│   ├── start_with_optimizations.sh  # recommended CLI flags recipe
│   └── env_overrides.md             # env var tuning guide
├── docs/
│   ├── ARCHITECTURE.md              # how the Docker image is built
│   ├── H3_BENCHMARK.md              # detailed H3 t2va benchmark trace
│   ├── ATTENTION_BACKENDS.md        # all attention backends comparison
│   ├── APU_MEMORY_MODEL.md          # why GTT is the right thing to watch on 780M
│   └── CHANGELOG.md                 # version history
└── grub-backup/
    ├── cmdline.before.txt           # /proc/cmdline BEFORE the fix
    ├── grub.original                # /etc/default/grub BEFORE
    ├── grub-edit.sh                 # script that did the edit
    └── README.md                    # rollback instructions
```

---

## Credits

- **AMD** — for TheRock project and the gfx1103 nightly wheels at `https://rocm.nightlies.amd.com/v2/gfx110X-all/`
- **comfyanonymous** — for ComfyUI
- **jaguardev** — for the original 780m-ai-stack Dockerfile structure (adapted)
- **guinmoon** — for the SageAttention ROCm 7 wheel
- **ROCm issues #5590, #5665** — for documenting the gfx1103 KFD-queue-eviction MES bug and the `cwsr_enable=0` workaround
- **vaskes** — for funding the APU and being patient during the 2-hour H3 inference

---

## License

MIT
