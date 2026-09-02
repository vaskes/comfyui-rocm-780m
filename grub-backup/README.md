# GRUB Backup — gfx1103 kernel fix

This is the exact set of changes applied to `/etc/default/grub` on the test host to fix the gfx1103 KFD-queue-eviction MES hang.

## What was changed

Added these kernel parameters to `GRUB_CMDLINE_LINUX`:

```
amdgpu.cwsr_enable=0
amdgpu.mes_kiq=1
amdgpu.noretry=1
amdgpu.sg_display=0
amdgpu.gpu_recovery=1
ttm.page_pool_size=6291456
transparent_hugepage=always
```

## What each one does

| Parameter | Purpose | Source |
|---|---|---|
| `amdgpu.cwsr_enable=0` | **THE FIX.** Disables compute-wave-save-restore, which on gfx1103 triggers the MES firmware bug. Without this, GPU hangs after 5-30 seconds of any nontrivial inference. | ROCm issues #5590, #5665 |
| `amdgpu.mes_kiq=1` | Use KIQ (Kernel Interface Queue) for MES, required for gfx1103. | ROCm docs |
| `amdgpu.noretry=1` | Don't retry failed commands, fail fast (debugging) | jaguar 780m-ai-stack |
| `amdgpu.sg_display=0` | Don't use s/g for display, more VRAM for compute | jaguardev |
| `amdgpu.gpu_recovery=1` | Try to recover GPU on hang instead of leaving it dead | jaguardev |
| `ttm.page_pool_size=6291456` | TTM (Translation Table Maps) page pool = 6 GB, reduces GTT pressure | jaguardev |
| `transparent_hugepage=always` | THP for GTT allocations, less fragmentation | jaguardev |

## What is INTENTIONALLY NOT changed

- **`amd_iommu=off`** — stability-critical, do NOT disable IOMMU
- **kernel version** — project policy: no kernel upgrades or downgrades
- **anything in `/etc/modprobe.d/`** — could conflict with the GRUB cmdline
- **BIOS settings** — GTT 40 GB and VRAM 16 GB are correct for our use case

## How to apply

```bash
# Backup current state
sudo cp /etc/default/grub /opt/comfyui-rocm-780m/grub-backup/grub.original
sudo cp /boot/grub/grub.cfg /opt/comfyui-rocm-780m/grub-backup/grub.cfg.before
sudo bash -c 'cat /proc/cmdline > /opt/comfyui-rocm-780m/grub-backup/cmdline.before.txt'

# Edit /etc/default/grub, append to GRUB_CMDLINE_LINUX
sudo nano /etc/default/grub
# Append: amdgpu.cwsr_enable=0 amdgpu.mes_kiq=1 amdgpu.noretry=1 amdgpu.sg_display=0 amdgpu.gpu_recovery=1 ttm.page_pool_size=6291456 transparent_hugepage=always

# Apply
sudo update-grub
sudo reboot
```

## How to verify

After reboot:

```bash
cat /proc/cmdline | tr ' ' '\n' | grep amdgpu
# Should show all the new parameters
```

## How to roll back

```bash
sudo cp /opt/comfyui-rocm-780m/grub-backup/grub.original /etc/default/grub
sudo update-grub
sudo reboot
```

## Files in this directory

- `grub.original` — `/etc/default/grub` BEFORE the edit
- `cmdline.before.txt` — `/proc/cmdline` BEFORE the edit
- `grub-edit.sh` — script that did the edit (for reproducibility)
