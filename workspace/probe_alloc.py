#!/usr/bin/env python3
"""Probe where torch places allocations on gfx1103 — VRAM or GTT."""
import os
import sys
import torch

def read_sysfs(field):
    try:
        with open(f"/sys/class/drm/card0/device/mem_info_{field}") as f:
            return int(f.read().strip())
    except Exception as e:
        return -1

def report(label):
    v = read_sysfs("vram_used") / (1024*1024)
    g = read_sysfs("gtt_used") / (1024*1024)
    ta = torch.cuda.memory_allocated() / (1024*1024)
    tr = torch.cuda.memory_reserved() / (1024*1024)
    print(f"  [{label}]  sysfs VRAM={v:.0f} MiB  GTT={g:.0f} MiB  |  torch alloc={ta:.0f} MiB  reserved={tr:.0f} MiB")

# Test 1: small 100 MB allocation
print("test 1: 100 MB tensor on cuda:0")
report("before")
x = torch.randn(1024, 1024, 64, dtype=torch.float16, device="cuda")  # 128 MB fp16
torch.cuda.synchronize()
report("after 128 MB alloc")

# Test 2: bigger 1.6 GB
print("\ntest 2: 1.6 GB tensor on cuda:0")
x2 = torch.randn(1024, 1024, 1024, dtype=torch.float16, device="cuda")  # 2 GB
torch.cuda.synchronize()
report("after 2 GB alloc")

# Test 3: 6 GB (more than VRAM limit if VRAM is exclusive)
print("\ntest 3: try 6 GB (more than usual VRAM)")
try:
    x3 = torch.randn(2, 1024, 1024, 1024, dtype=torch.float16, device="cuda")  # 4 GB
    x4 = torch.randn(2, 1024, 1024, 1024, dtype=torch.float16, device="cuda")  # 4 GB
    torch.cuda.synchronize()
    report("after 8 GB alloc")
except Exception as e:
    print(f"  OOM: {e}")
    report("after OOM")

# Test 4: try with PYTORCH_NO_CUDA_MEMORY_CACHING
print("\ntest 4: try with PYTORCH_NO_CUDA_MEMORY_CACHING=1")
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
try:
    torch.cuda.empty_cache()
    x5 = torch.randn(2, 1024, 1024, 1024, dtype=torch.float16, device="cuda")  # 4 GB
    torch.cuda.synchronize()
    report("after 4 GB alloc")
except Exception as e:
    print(f"  ERR: {e}")
    report("after err")

print("\nKEY QUESTION: where does the 4 GB model go? VRAM or GTT?")
print("If sysfs GTT grows by ~4 GB: model is in GTT (slow, but works)")
print("If sysfs VRAM grows by ~4 GB: model is in VRAM (fast)")
