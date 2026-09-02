# Operations Cheatsheet

Everything you need to manage this ComfyUI setup on llmhost2 (Radeon 780M, gfx1103).

## TL;DR

```bash
# Is it running?
sg docker -c "docker ps" | grep comfyiu-test
# Or
curl -s http://127.0.0.1:8188/system_stats | head -1

# Not running? Start it.
sudo systemctl start comfyiu.service
# Or directly:
sg docker -c "docker start comfyiu-test"

# Logs (follow)
sg docker -c "docker logs -f comfyiu-test"

# Submit a workflow
python3 /opt/comfyiu/workspace/submit_and_watch.py /opt/comfyiu/workspace/<workflow>.json
```

## What's where

| Path | What's in it |
|---|---|
| `/opt/comfyiu/` | ComfyUI host layout (models, output, input, custom_nodes, user, logs, workspace) |
| `/opt/comfyiu/models/diffusion_models/` | H3 diffusion weights (20 GB) |
| `/opt/comfyiu/models/text_encoders/` | qwen3vl int4 (15 GB) and int8 (26 GB) |
| `/opt/comfyiu/models/vae/` | H3 video VAE (4 variants) + audio VAE |
| `/opt/comfyiu/models/loras/` | H3 turbo 8-step LoRA (1.9 GB) |
| `/opt/comfyiu/output/video/` | Generated mp4 files |
| `/opt/comfyiu/output/` | Generated images, subdirs by save-prefix |
| `/opt/comfyiu/workspace/` | Python test scripts, JSON workflows |
| `/opt/comfyiu/scripts/` | Shell helpers (start_comfyiu.sh, gpu_monitor.py, ...) |
| `/opt/comfyiu/logs/` | Container logs, bench traces, smoke runs |
| `/opt/comfyiu-rocm-780m/` | Git repo, Dockerfile, docs, scripts (this repo) |
| `/etc/systemd/system/comfyiu.service` | Auto-start service |
| `/opt/mavis-backups/grub-2026-09-02-cwsr-fix/` | GRUB fix backup (cwsr_enable=0 etc.) |
| `/opt/H3_*.mp4` | Reference outputs (H3_t2v_780M_first_success.mp4, H3_smoke_*) |

## Container

| | |
|---|---|
| Name | `comfyiu-test` |
| Image | `comfyiu:therock-gfx1103-v6` (5 GB, with gcc for flash-attn) |
| Command | `python main.py --use-flash-attention --enable-triton-backend --supports-fp8-compute --disable-pinned-memory --gpu-only --listen 0.0.0.0 --port 8188` |
| Env | `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` |
| GPU | `/dev/kfd /dev/dri` + GID 992 (render), 44 (video) |
| Network | host |
| Restart policy | `unless-stopped` |
| Port | 8188 (host) |

## Quick commands

### Service management
```bash
sudo systemctl status comfyiu.service
sudo systemctl start comfyiu.service
sudo systemctl stop comfyiu.service
sudo systemctl restart comfyiu.service
```

### Container
```bash
sg docker -c "docker ps"                          # status
sg docker -c "docker logs -f comfyiu-test"        # tail logs
sg docker -c "docker exec -it comfyiu-test bash"  # shell
sg docker -c "docker restart comfyiu-test"        # restart
sg docker -c "docker rm -f comfyiu-test"          # delete (data persists in /opt/comfyiu)
```

### Submit workflows
```bash
# SD 1.5 reference
python3 /opt/comfyiu/workspace/submit_and_watch.py /opt/comfyiu/workspace/test_workflow_sd15.json

# H3 t2v full (5 sec, 832x480, 20 steps, ~2h on v5 SDPA, ~5-7 min on v6 flash-attn)
python3 /opt/comfyiu/workspace/submit_and_watch.py /opt/comfyiu/workspace/h3_t2v_workflow.json

# H3 t2v smoke (0.2 sec, 320x640, 8 steps, ~3 min on v6 flash-attn)
python3 /opt/comfyiu/workspace/submit_and_watch.py /opt/comfyiu/workspace/h3_smoke_workflow.json
```

### Inspect history
```bash
# List all jobs
sg docker -c "docker exec comfyiu-test ls -la /opt/ComfyUI/output/video/"

# Check progress of running job
sg docker -c "docker exec comfyiu-test bash -c 'cat /opt/ComfyUI/logs/comfyui.log 2>/dev/null | tail -20'"

# GPU/GTT usage
watch -n 1 'cat /sys/class/drm/card0/device/mem_info_{vram,gtt}_used'
```

### Resource monitoring
```bash
# One-shot GPU status
sudo /opt/rocm/bin/rocm-smi

# Continuous monitor
python3 /opt/comfyiu/scripts/gpu_monitor.py 60 0.5
```

## Image management

```bash
# List all versions
sg docker -c "docker images" | grep therock

# Roll back to v5 (no gcc, flash-attn broken)
sg docker -c "docker stop comfyiu-test && docker rm comfyiu-test"
sg docker -c "docker run -d --name comfyiu-test --network host --device /dev/kfd --device /dev/dri --group-add 992 --group-add 44 --security-opt seccomp=unconfined --cap-add SYS_PTRACE -v /opt/comfyiu/models:/opt/ComfyUI/models -v /opt/comfyiu/output:/opt/ComfyUI/output -v /opt/comfyiu/input:/opt/ComfyUI/input -v /opt/comfyiu/custom_nodes:/opt/ComfyUI/custom_nodes -v /opt/comfyiu/user:/opt/ComfyUI/user -v /opt/comfyiu/logs:/opt/ComfyUI/logs -v /opt/comfyiu/workspace:/workspace comfyiu:therock-gfx1103-v5 python main.py --use-flash-attention --enable-triton-backend --supports-fp8-compute --disable-pinned-memory --gpu-only --listen 0.0.0.0 --port 8188"

# Rebuild v6 from source
cd /opt/comfyui-rocm-780m
sg docker -c "docker build -f build/Dockerfile.v6 -t comfyiu:therock-gfx1103-v6 build/"
```

## Recipes for different attention backends

```bash
# Currently running: flash-attn + comfy-kitchen triton + fp8
python main.py --use-flash-attention --enable-triton-backend --supports-fp8-compute --disable-pinned-memory --gpu-only

# Alternative: comfy-kitchen only (works for quantized/fp8)
python main.py --use-ck-attention --enable-triton-backend --supports-fp8-compute --gpu-only

# Alternative: sage (headdim must be in [64, 96, 128])
python main.py --use-sage-attention --highvram

# Fallback: pytorch (always works, slowest)
python main.py --use-pytorch-cross-attention --highvram
```

## GRUB fix (REQUIRED for 780M to work)

The amdgpu kernel module has a KFD-queue-eviction MES bug for gfx1103. Without the fix, every inference hangs after 5-30s.

**Verify**:
```bash
cat /proc/cmdline | grep cwsr_enable=0
# Should print: ... amdgpu.cwsr_enable=0 ...
```

**Backup of original GRUB config** is in `/opt/mavis-backups/grub-2026-09-02-cwsr-fix/`. See `grub-backup/README.md` in the git repo for full details and rollback procedure.

## Reboot procedure

After reboot:
1. Container should auto-start via `comfyiu.service`
2. If not: `sudo systemctl start comfyiu.service`
3. Verify: `curl -s http://127.0.0.1:8188/system_stats | head -1` should return JSON
4. Check flash-attn works: `sg docker -c "docker logs comfyiu-test 2>&1 | grep -c 'Flash Attention failed'"` should be 0

## Known issues

1. **`amd_iommu=off` is FORBIDDEN** — stability-critical
2. **Kernel change is FORBIDDEN** — fixes must work with kernel 7.0.0-30
3. **GitHub may rate-limit `git clone` in docker build** — use curl + tarball instead (Dockerfiles already fixed)
4. **MIOpen fdb.txt warning** — `File is unreadable: .../gfx1103_6.HIP.fdb.txt` — benign, just a MIOpen cache file

## Updating the setup

After making changes to Dockerfiles, scripts, or docs:
```bash
cd /opt/comfyui-rocm-780m
# Edit files
sudo -n git add -A
sudo -n git commit -m "describe change"
sudo -n git push origin main
```

After changing models in /opt/comfyiu/models/, no git action needed — models are gitignored. Just verify the container sees them:
```bash
sg docker -c "docker exec comfyiu-test ls /opt/ComfyUI/models/<type>/"
```

## Architecture

```
Host (Ubuntu 24.04, kernel 7.0.0-30)
├── amdgpu (with cwsr_enable=0)  ← 780M iGPU
│   ├── /dev/kfd, /dev/dri        ← exposed to container
│   ├── VRAM: 16 GB (mostly empty)
│   └── GTT: 40 GB (where models live)
│
├── Docker
│   ├── comfyiu:therock-gfx1103-v6 (5 GB)
│   │   └── ROCm 7.13 TheRock + gcc + flash-attn + sage + comfy-kitchen + comfy-aimdo
│   │
│   └── container: comfyiu-test
│       └── ComfyUI 0.34.0 on Python 3.12, port 8188
│
├── systemd
│   └── comfyiu.service    ← auto-starts container at boot
│
└── Files (persistent)
    ├── /opt/comfyiu/                ← container data
    ├── /opt/comfyui-rocm-780m/      ← git repo (source of truth)
    ├── /opt/mavis-backups/grub-2026-09-02-cwsr-fix/  ← GRUB fix backup
    └── /opt/H3_*.mp4                ← reference outputs
```
