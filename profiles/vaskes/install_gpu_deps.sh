#!/bin/bash
# Idempotent install of GPU-only packages (flash-attn).
# These cannot be installed during docker build because their setup.py
# imports torch to detect HIP/CUDA arch — but docker build has no GPU.
# Run once at container first start (or manually via docker exec).

set -e

# Marker file: if /opt/ComfyUI/.gpu_deps_installed exists, skip
MARKER=/opt/ComfyUI/.gpu_deps_installed
if [ -f "$MARKER" ]; then
    echo "[gpu-deps] Already installed, marker $MARKER exists. Skipping."
    exit 0
fi

echo "[gpu-deps] First run — installing flash-attn + bitsandbytes"
echo "[gpu-deps] This takes ~3-5 minutes (Triton JIT compiles flash-attn kernels)"
echo

# Check flash-attn
if python3 -c "import flash_attn" 2>/dev/null; then
    echo "[gpu-deps] flash-attn already importable, skipping"
else
    echo "[gpu-deps] Installing flash-attn==2.8.3.post1"
    pip3 install --break-system-packages --no-build-isolation \
        "flash-attn==2.8.3.post1"
fi

# Check bitsandbytes
if python3 -c "import bitsandbytes" 2>/dev/null; then
    echo "[gpu-deps] bitsandbytes already importable, skipping"
else
    echo "[gpu-deps] Installing bitsandbytes"
    pip3 install --break-system-packages "bitsandbytes"
fi

# Create marker
touch "$MARKER"
echo "[gpu-deps] Done. Marker $MARKER created."
