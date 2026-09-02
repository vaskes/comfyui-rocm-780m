#!/usr/bin/env python3
"""Test: can we force torch to allocate to VRAM instead of GTT on gfx1103?"""
import os, sys
import torch

# Strategy 1: GPU_MAX_HEAP_SIZE — limits how much GTT torch can use
# Strategy 2: PYTORCH_CUDA_ALLOC_CONF — allocator config
# Strategy 3: HSA override flags

def report(label):
    v = int(open("/sys/class/drm/card0/device/mem_info_vram_used").read()) / (1024*1024)
    g = int(open("/sys/class/drm/card0/device/mem_info_gtt_used").read()) / (1024*1024)
    ta = torch.cuda.memory_allocated() / (1024*1024)
    print(f"  [{label}]  VRAM={v:.0f} MiB  GTT={g:.0f} MiB  | torch alloc={ta:.0f} MiB")

# Try different envs at module load
print(f"arg: {sys.argv[1] if len(sys.argv)>1 else 'none'}")

strategy = sys.argv[1] if len(sys.argv) > 1 else "default"

if strategy == "max_heap_4gb":
    os.environ["GPU_MAX_HEAP_SIZE"] = str(4 * 1024**3)  # 4 GB max
    print("set GPU_MAX_HEAP_SIZE=4G")
elif strategy == "max_heap_8gb":
    os.environ["GPU_MAX_HEAP_SIZE"] = str(8 * 1024**3)
    print("set GPU_MAX_HEAP_SIZE=8G")
elif strategy == "max_heap_12gb":
    os.environ["GPU_MAX_HEAP_SIZE"] = str(12 * 1024**3)
    print("set GPU_MAX_HEAP_SIZE=12G")
elif strategy == "max_heap_14gb":
    os.environ["GPU_MAX_HEAP_SIZE"] = str(14 * 1024**3)
    print("set GPU_MAX_HEAP_SIZE=14G")
elif strategy == "garbage_collect":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:512"
    print("set PYTORCH_CUDA_ALLOC_CONF")
elif strategy == "hsa_override":
    # Try HSA flag to force VRAM
    pass

print(f"\nGPU device count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
print()

report("init")

print("\nallocating 2 GB...")
x = torch.randn(1024, 1024, 1024, dtype=torch.float16, device="cuda")
report("after 2 GB")

print("\nallocating another 2 GB...")
y = torch.randn(1024, 1024, 1024, dtype=torch.float16, device="cuda")
report("after 4 GB total")

print("\nallocating another 4 GB...")
try:
    z = torch.randn(2, 1024, 1024, 1024, dtype=torch.float16, device="cuda")
    report("after 8 GB total")
except Exception as e:
    print(f"OOM: {e}")
    report("after OOM")
