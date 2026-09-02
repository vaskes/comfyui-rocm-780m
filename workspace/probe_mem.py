#!/usr/bin/env python3
"""Probe where torch actually puts tensors — VRAM or GTT (system RAM)."""
import torch
import subprocess
import re

def rocm_vram_used_mib():
    try:
        out = subprocess.run(["/opt/rocm/bin/rocm-smi", "--showmeminfo=VRAM"],
                             capture_output=True, text=True, timeout=5)
        for line in out.stdout.splitlines():
            m = re.search(r"VRAM Total Used Memory \(B\):\s*(\d+)", line)
            if m:
                return int(m.group(1)) // (1024 * 1024)
    except Exception:
        pass
    return -1

def torch_vram_allocated_mib():
    return torch.cuda.memory_allocated() // (1024 * 1024)

def torch_vram_reserved_mib():
    return torch.cuda.memory_reserved() // (1024 * 1024)

def torch_total_mib():
    return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)

print("=" * 60)
print("Before allocation")
print("=" * 60)
print(f"  torch total memory:    {torch_total_mib()} MiB  (16 GB VRAM + GTT reported as one)")
print(f"  torch allocated:        {torch_vram_allocated_mib()} MiB")
print(f"  torch reserved:         {torch_vram_reserved_mib()} MiB")
print(f"  rocm-smi VRAM used:     {rocm_vram_used_mib()} MiB")
print()

# Allocate a 1.6 GB tensor (SD UNet size) on cuda:0
print("=" * 60)
print("Allocating 1.6 GB tensor on cuda:0 ...")
print("=" * 60)
x = torch.randn(1024, 1024, 1024, dtype=torch.float16, device="cuda")  # 2 GB
y = torch.randn(1024, 1024, 1024, dtype=torch.float16, device="cuda")  # 2 GB more
print(f"  torch allocated:        {torch_vram_allocated_mib()} MiB")
print(f"  torch reserved:         {torch_vram_reserved_mib()} MiB")
print(f"  rocm-smi VRAM used:     {rocm_vram_used_mib()} MiB")
print()

# Now do a matmul
print("=" * 60)
print("Doing matmul ...")
print("=" * 60)
import time
torch.cuda.synchronize()
t0 = time.time()
z = torch.matmul(x, y.T)
torch.cuda.synchronize()
print(f"  matmul done in {time.time()-t0:.3f}s")
print(f"  result mean: {z.float().mean().item():.4f}")
print()

# Try with even larger allocation
print("=" * 60)
print("Allocating 12 GB tensor (would fit in 16 GB VRAM, NOT in 8 GB)...")
print("=" * 60)
try:
    big = torch.randn(3, 1024, 1024, 1024, dtype=torch.float16, device="cuda")  # 6 GB
    bigger = torch.randn(3, 1024, 1024, 1024, dtype=torch.float16, device="cuda")  # 6 GB more = 12 GB
    print(f"  allocated 12 GB on cuda:0")
    print(f"  torch allocated:        {torch_vram_allocated_mib()} MiB")
    print(f"  torch reserved:         {torch_vram_reserved_mib()} MiB")
    print(f"  rocm-smi VRAM used:     {rocm_vram_used_mib()} MiB")
except torch.cuda.OutOfMemoryError as e:
    print(f"  OOM: {e}")
print()

# Final test
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"  If rocm-smi shows ~12 GB used for 12 GB tensor: model is in VRAM")
print(f"  If rocm-smi shows ~0-100 MB used: model is in GTT (system RAM)")
