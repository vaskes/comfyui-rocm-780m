# ComfyUI + ROCm TheRock (gfx1103 native) for Radeon 780M
# Built per official ComfyUI README: rocm.nightlies.amd.com/v2/gfx110X-all/
# Inspired by jaguardev/780m-ai-stack (TheRock multi-stage), but on ubuntu:24.04 base
# Constraints: no kernel change, no amd_iommu=off, no host ROCm change

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# System packages
# - python3.12 (default in 24.04, matches wheel cp312)
# - ROCm deps (libhsa-runtime64, libamd-comgr, libdrm-amdgpu) — needed even for TheRock wheels
# - ComfyUI runtime deps (ffmpeg, libGL, libxcb, etc.)
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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure python3.12 is the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /opt

# --- TheRock wheels for gfx110X-all (official ComfyUI RDNA3 path) ---
# - torch/torchvision/triton compiled natively for gfx1103 (no HSA_OVERRIDE_GFX_VERSION needed)
# - rocm-sdk-core + rocm-sdk-libraries-gfx110x-all provide the gfx1103-specific runtime
# - uses pre-release (--pre) because these are alpha/beta nightly builds
ARG TORCH_IDX=https://rocm.nightlies.amd.com/v2/gfx110X-all/
# Install ONLY from gfx110X-all (no --extra-index-url PyPI — would let pip grab CUDA torch)
RUN pip3 install --break-system-packages --pre \
    rocm-sdk-core \
    rocm-sdk-libraries-gfx110x-all \
    --index-url ${TORCH_IDX} \
    --no-deps

# PyTorch stack (torch + torchvision + triton) from same gfx110X-all index, pinned
# - Pinned because pip with --extra-index-url picks highest version from any index,
#   and PyPI has newer stable torch that doesn't know about gfx1103
# - torchvision on gfx110X-all: 0.27.0a0+rocm7.13.0a (paired with torch 2.9.1)
RUN pip3 install --break-system-packages --pre \
    "torch==2.9.1+rocm7.13.0a20260513" \
    "torchvision==0.24.0+rocm7.13.0a20260513" \
    "torchaudio==2.9.0+rocm7.13.0a20260513" \
    "triton" \
    --index-url ${TORCH_IDX}

# --- ComfyUI ---
ARG COMFYUI_REF=master
RUN git clone --depth 1 -b ${COMFYUI_REF} https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI
WORKDIR /opt/ComfyUI

# Install ComfyUI requirements (filter out torch/torchvision/triton which we have from TheRock)
RUN grep -vE '^(torch|torchvision|triton|torchaudio)\b' requirements.txt > /tmp/req_filtered.txt && \
    pip3 install --break-system-packages -r /tmp/req_filtered.txt

# --- ComfyUI-Manager (handy for custom nodes later) ---
RUN git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git /opt/ComfyUI/custom_nodes/ComfyUI-Manager && \
    pip3 install --break-system-packages -r /opt/ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt || true

# --- SageAttention for ROCm 7 (jaguardev's binary wheel) ---
RUN pip3 install --break-system-packages --no-cache-dir \
    https://github.com/guinmoon/SageAttention-Rocm7/releases/download/v1.0.6_rocm7/sageattention-1.0.6-py3-none-any.whl \
    || echo "WARNING: sageattention-1.0.6 install failed, will try without"

# --- Runtime environment ---
# jaguardev's env, with notes from past attempts on this host
# Key changes vs default:
#   HSA_ENABLE_SDMA=0  (jaguar; previous attempt had 1 which hung)
#   HSA_USE_SVM=0      (jaguar; previous attempt had default)
#   TORCH_CUDNN_ENABLED=0  (avoid cudnn path entirely)
#   TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1  (try AOTriton flash/efficient)
#   FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE  (Triton flash-attn path)
#   MIOPEN_FIND_MODE=2  (cache then tune)
#   MIOPEN_FIND_ENFORCE=1  (re-search DB)
#   PYTORCH_TUNABLEOP_TUNING=0  (don't autotune at runtime — slow + risky)
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
# gfx1103-native: do NOT set HSA_OVERRIDE_GFX_VERSION (gfx110X-all wheels know about gfx1103)

# Make /opt/ComfyUI a workspace; mount points the user expects
WORKDIR /opt/ComfyUI
VOLUME ["/opt/ComfyUI/models", "/opt/ComfyUI/output", "/opt/ComfyUI/input", "/opt/ComfyUI/custom_nodes", "/opt/ComfyUI/user"]

EXPOSE 8188

# Default command: ComfyUI with sage attention (jaguar's "balanced" variant)
# Override in docker-compose for aggressive variants
CMD ["python", "main.py", \
     "--listen", "0.0.0.0", \
     "--port", "8188", \
     "--use-sage-attention", \
     "--enable-manager"]
