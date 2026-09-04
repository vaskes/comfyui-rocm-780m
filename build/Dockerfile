# ComfyUI + ROCm TheRock (gfx1103 native) for Radeon 780M
# PLANETARY BASE: builds comfyiu:base — the minimum needed to run ComfyUI
# on an AMD APU (gfx1103) with ROCm 7.13 / TheRock wheels. No custom nodes
# are installed here. For a specific workflow setup, see ../profiles/.
#
# Built per official ComfyUI README: rocm.nightlies.amd.com/v2/gfx110X-all/
#
# MUTUALLY EXCLUSIVE NOTES (ROCm vs nvidia):
# - This is the ROCm/gfx1103 build. For nvidia, use a separate image with
#   FROM nvidia/cuda and PyTorch.org wheels (cu130/cu132).
# - SageAttention: ROCm uses guinmoon wheel, NOT thu-ml (CUDA-only).
# - flash-attn: installed at first run via entrypoint (needs GPU; build has
#   no GPU access for the wheel build).
# - Skip nvidia-only packages: nvidia-vfx, onnxruntime-gpu, deepspeed.
#
# Constraints: no kernel change, no amd_iommu=off, no host ROCm change.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# System packages
# - python3.12 (default in 24.04, matches wheel cp312)
# - ROCm deps (libhsa-runtime64, libamd-comgr, libdrm-amdgpu) — needed even for TheRock wheels
# - ComfyUI runtime deps (ffmpeg, libGL, libxcb, etc.)
# - C/C++ toolchain (g++, cmake, ninja-build) — required for Triton JIT to compile
#   flash-attn kernels on first use. Without this, flash-attn silently falls back to SDPA.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    ca-certificates curl wget git aria2 \
    libgfortran5 libatomic1 libquadmath0 \
    libhsa-runtime64-1 libamd-comgr2 libdrm-amdgpu1 \
    libelf1 libnuma1 libssl3 \
    libxcb1 libxext6 libxrender1 libsm6 libxi6 \
    libgl1 libglib2.0-0 libgomp1 \
    ffmpeg sox libportaudio2 libportaudiocpp0 \
    libudev1 pkg-config \
    g++ gcc cmake ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure python3.12 is the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /opt

# --- TheRock wheels for gfx110X-all (official ComfyUI RDNA3 path) ---
# MUTUALLY EXCLUSIVE: this is the ROCm TheRock install. Do NOT mix with
# PyTorch.org cu130/cu132 wheels.
ARG TORCH_IDX=https://rocm.nightlies.amd.com/v2/gfx110X-all/
RUN pip3 install --break-system-packages --pre \
    rocm-sdk-core \
    rocm-sdk-libraries-gfx110x-all \
    --index-url ${TORCH_IDX} \
    --no-deps

# PyTorch stack from same gfx110X-all index, pinned
RUN pip3 install --break-system-packages --pre \
    "torch==2.9.1+rocm7.13.0a20260513" \
    "torchvision==0.24.0+rocm7.13.0a20260513" \
    "torchaudio==2.9.0+rocm7.13.0a20260513" \
    "triton" \
    --index-url ${TORCH_IDX}

# --- ComfyUI core ---
ARG COMFYUI_REF=master
RUN curl -L -s -o /tmp/comfyui.tar.gz https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/${COMFYUI_REF}.tar.gz && \
    tar xzf /tmp/comfyui.tar.gz && \
    mv ComfyUI-${COMFYUI_REF} /opt/ComfyUI && \
    rm /tmp/comfyui.tar.gz
WORKDIR /opt/ComfyUI

# Install ComfyUI requirements (filter out torch/torchvision/triton which we have from TheRock)
RUN grep -vE '^(torch|torchvision|triton|torchaudio)\b' requirements.txt > /tmp/req_filtered.txt && \
    pip3 install --break-system-packages -r /tmp/req_filtered.txt

# === CC/CXX for Triton JIT (flash-attn first-run compile) ===
ENV CC=gcc
ENV CXX=g++

# --- bitsandbytes (works on ROCm with TheRock wheels) ---
# flash-attn installed at first run via entrypoint (needs GPU; build has none)
RUN pip3 install --break-system-packages --no-build-isolation \
    "bitsandbytes" \
    || echo "WARNING: bitsandbytes install failed (will retry at runtime)"

# --- ROCm flash-attn CUDA-stub for SeedVR2 compatibility ---
# SeedVR2 does: import flash_attn_2_cuda  # noqa: F401
# The ROCm flash-attn wheel uses the Triton backend, not the CUDA C++ ext.
# We create an empty stub so SeedVR2's detection check passes.
# This is harmless: SeedVR2 never calls functions from this module.
RUN cat > /usr/local/lib/python3.12/dist-packages/flash_attn_2_cuda.py << "STUB_EOF"
# ROCm compatibility stub. The ROCm flash-attn wheel uses the Triton
# backend instead of the CUDA C++ extension. SeedVR2 only needs this
# module to be importable; actual attention uses flash_attn_varlen_func.
__all__ = []
STUB_EOF

# --- Runtime ROCm environment ---
ENV HSA_ENABLE_SDMA=0
ENV HSA_USE_SVM=0
ENV TORCH_CUDNN_ENABLED=0
ENV TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
ENV COMFYUI_ENABLE_MIOPEN=1
ENV MIOPEN_FIND_MODE=2
ENV MIOPEN_FIND_ENFORCE=1
ENV MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
ENV MIOPEN_DEBUG_CONV_FFT=0
ENV MIOPEN_DEBUG_CONV_DIRECT=0
ENV MIOPEN_DEBUG_CONV_WINOGRAD=0
ENV PYTORCH_TUNABLEOP_ENABLED=1
ENV PYTORCH_TUNABLEOP_TUNING=0
ENV PYTORCH_HIP_ALLOC_CONF=backend:native,garbage_collection_threshold:0.7,max_split_size_mb:256,expandable_segments:True
ENV AMD_LOG_LEVEL=0
ENV AMD_SERIALIZE_KERNEL=0

WORKDIR /opt/ComfyUI

# Install GPU-only deps at first run (flash-attn).
# These need GPU access to compile, which docker build does not provide.
COPY install_gpu_deps.sh /usr/local/bin/install_gpu_deps.sh
RUN chmod 755 /usr/local/bin/install_gpu_deps.sh
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 755 /usr/local/bin/entrypoint.sh

# Persistent host volumes (for custom installs + models)
# NOTE: custom_nodes/ NOT in volume list — nodes are baked into the image.
# To override from host: comment out the VOLUME block or rsync.
VOLUME ["/opt/ComfyUI/models", "/opt/ComfyUI/output", "/opt/ComfyUI/input", "/opt/ComfyUI/user"]

EXPOSE 8188

# Entrypoint: installs GPU deps on first run, then execs CMD
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command with standard optimizations.
# MUTUALLY EXCLUSIVE attention backends (pick ONE — see docs/ATTENTION_BACKENDS.md):
#   --use-flash-attention (Triton ROCm; works for most DiT/UNet) ← DEFAULT
#   --use-ck-attention   (comfy-kitchen; best with --supports-fp8-compute)
#   --use-sage-attention (headdim must be in [64, 96, 128])
#   (no flag)            pytorch SDPA fallback (slowest but always works)
CMD ["python", "main.py", \
     "--use-flash-attention", \
     "--enable-triton-backend", \
     "--supports-fp8-compute", \
     "--disable-pinned-memory", \
     "--gpu-only", \
     "--listen", "0.0.0.0", \
     "--port", "8188"]
