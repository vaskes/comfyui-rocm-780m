# Architecture: How the Docker image is built

## Base layer

```dockerfile
FROM ubuntu:24.04
```

Why not `rocm/dev-ubuntu-24.04:7.2.4`?
- The ROCm dev image has ROCm 7.2.4 baked in, but it's not tuned for gfx1103
- The TheRock wheels contain their own ROCm 7.13/7.14 runtime that's **compiled natively for gfx1103**
- Using a vanilla Ubuntu base gives us a clean separation of concerns: only what's needed

## Python

```dockerfile
RUN apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip ...
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100
```

Ubuntu 24.04 ships Python 3.12 by default, which matches the wheel ABI (`cp312`).

## TheRock wheels

```dockerfile
ARG TORCH_IDX=https://rocm.nightlies.amd.com/v2/gfx110X-all/
RUN pip3 install --break-system-packages --pre \
    rocm-sdk-core rocm-sdk-libraries-gfx110x-all \
    --index-url ${TORCH_IDX} --no-deps
```

The gfx110X-all index is hosted by AMD's TheRock CI. It contains wheels for `torch`, `torchvision`, `torchaudio`, `triton`, and ROCm runtime libraries. **All compiled natively for gfx1103** — no `HSA_OVERRIDE_GFX_VERSION` shim needed.

`--no-deps` is critical: it prevents pip from going to PyPI to resolve dependencies, which would let it pull a CUDA torch wheel.

`--pre` is required: these are alpha/beta nightly builds (`torch==2.9.1+rocm7.13.0a20260513`).

## Pinned torch stack

```dockerfile
RUN pip3 install --break-system-packages --pre \
    "torch==2.9.1+rocm7.13.0a20260513" \
    "torchvision==0.24.0+rocm7.13.0a20260513" \
    "torchaudio==2.9.0+rocm7.13.0a20260513" \
    "triton" \
    --index-url ${TORCH_IDX}
```

The `==` pin prevents pip from grabbing a newer version from PyPI when both indexes are visible.

`torchvision==0.24.0` (not 0.25/0.26/0.27) — newer versions require torch 2.10+ dispatch API.

`torchaudio==2.9.0` — matches the 2.9.x torch release.

`triton` (latest available on the index) — currently 3.5.1+rocm7.13.

## ComfyUI

```dockerfile
ARG COMFYUI_REF=master
RUN git clone --depth 1 -b ${COMFYUI_REF} https://github.com/comfyanonymous/ComfyUI.git
RUN grep -vE '^(torch|torchvision|triton|torchaudio)\b' requirements.txt > /tmp/req_filtered.txt
RUN pip3 install --break-system-packages -r /tmp/req_filtered.txt
```

We filter out torch/torchvision/triton/torchaudio from ComfyUI's requirements.txt before installing. Otherwise, pip would try to install the CUDA versions from PyPI, which would shadow our TheRock wheels.

`--depth 1 -b master` keeps the clone fast. Override `COMFYUI_REF` to build a specific ComfyUI version (e.g. `0.3.30` for a stable release).

## ComfyUI-Manager (optional but recommended)

```dockerfile
RUN git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git \
    /opt/ComfyUI/custom_nodes/ComfyUI-Manager
```

For installing custom nodes via the UI.

## SageAttention

```dockerfile
RUN pip3 install --break-system-packages --no-cache-dir \
    https://github.com/guinmoon/SageAttention-Rocm7/releases/download/v1.0.6_rocm7/sageattention-1.0.6-py3-none-any.whl
```

Pre-built binary wheel for ROCm 7 from guinmoon. Avoids the multi-hour compile from source.

## v5: flash-attn + bitsandbytes

```dockerfile
RUN pip3 install --break-system-packages --no-build-isolation \
    "flash-attn==2.8.3.post1" "bitsandbytes"
```

`flash-attn` 2.8.3.post1 has a **Triton ROCm backend** that uses the already-installed triton wheel. No C++/HIP compile needed (because TheRock triton already has gfx1103 support).

To activate, set `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` at runtime.

`bitsandbytes` is the official PyPI package; it has a ROCm 7.2 fallback binary that loads on our 7.13 setup with a warning.

## Runtime environment

```dockerfile
ENV HSA_ENABLE_SMDA=0      # AMD SDMA engine — disable, was causing hangs
ENV HSA_USE_SVM=0          # Shared Virtual Memory — disable
ENV TORCH_CUDNN_ENABLED=0  # Don't go through cudnn (it's not on ROCm anyway)
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
ENV MIOPEN_FIND_MODE=2
ENV PYTORCH_TUNABLEOP_TUNING=0  # Don't autotune (slow + risky)
ENV PYTORCH_HIP_ALLOC_CONF=backend:native,...
```

The full set of env vars is documented in the Dockerfile. Each one was tested — defaults that work.

## Runtime

```dockerfile
VOLUME ["/opt/ComfyUI/models", "/opt/ComfyUI/output", ...]
EXPOSE 8188
CMD ["python", "main.py", "--use-flash-attention", ...]
```

The default command in v5 is the most-aggressive configuration. Override in `docker-compose.yml` for your specific workload.

## What we do NOT do

- **No kernel change** (forbidden by the project)
- **No `amd_iommu=off`** (stability-critical, don't disable)
- **No host ROCm change** — only new ROCm versions inside Docker
- **No `HSA_OVERRIDE_GFX_VERSION`** — the TheRock wheels are gfx1103-native, the shim is deprecated
