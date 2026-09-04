# ComfyUI on Radeon 780M (gfx1103) — vaskes workflow profile
# Extends the planetary base (comfyiu:base) with this user's specific
# custom-node stack. Builds comfyiu:vaskes.
#
# This profile matches the user's llmhost1 (nvidia) production setup
# but compiled for ROCm. See ../README.md for the profile system.
#
# 22 custom nodes + comfyui-Manager + flash-attn first-run install.
# To add a new profile, copy this directory and edit.

# === Base image (planetary) ===
# Build base first:  cd build && docker build -t comfyiu:base .
ARG COMYIU_BASE_TAG=base
FROM comfyiu:${COMYIU_BASE_TAG}

# === Common pip deps for custom nodes ===
# ROCm-friendly. NOT included (CUDA-only): nvidia-vfx, onnxruntime-gpu, deepspeed
RUN pip3 install --break-system-packages --no-cache-dir \
    opencv-contrib-python-headless \
    imageio-ffmpeg \
    imageio \
    matrix-nio \
    GitPython \
    diffusers \
    accelerate \
    transformers \
    safetensors \
    huggingface-hub \
    scikit-image \
    blend_modes \
    lark \
    descript-audiotools \
    qwen-tts \
    librosa \
    onnxruntime \
    conformer \
    gradio \
    gdown \
    hydra-core \
    HyperPyYAML \
    inflect \
    matplotlib \
    modelscope \
    omegaconf \
    openai-whisper \
    pyworld \
    tensorboard \
    wetext \
    gguf \
    rotary_embedding_torch \
    einops \
    sentencepiece \
    protobuf

# === Custom nodes (22 total) ===
# All installed via tarball (git clone via https is sometimes blocked in build).
# Failures are non-fatal (|| true) so a single broken node does not kill the build.
WORKDIR /opt/ComfyUI/custom_nodes

# Helper: install a single node from github tarball.
# Usage: install_node.sh <github_src> <dir_name>
#   <github_src> = "owner/repo" or "owner/repo@branch"
COPY install_node.sh /usr/local/bin/install_node.sh
RUN chmod +x /usr/local/bin/install_node.sh

# === Custom node repo URLs (corrected 2026-09-04) ===
# Several repos moved owners since llmhost1's snapshot. Each "X -> Y" comment
# notes the original owner we expected vs. the actual current owner.
RUN install_node.sh "Comfy-Org/ComfyUI-Manager" "ComfyUI-Manager" && \
    # Comfy-Org/ComfyUI-Manager used to be ltdrdata/ComfyUI-Manager
    pip3 install --break-system-packages --no-cache-dir -r /opt/ComfyUI/manager_requirements.txt && \
    install_node.sh "city96/ComfyUI-GGUF" "ComfyUI-GGUF" && \
    install_node.sh "jlucasmcrell/ComfyUI-H3-Multishot@master" "ComfyUI-H3-Multishot" && \
    # kijai/ComfyUI-H3-Multishot -> jlucasmcrell (default branch is master, not main)
    install_node.sh "matlowai/ComfyUI-MAINodes" "ComfyUI-MAINodes" && \
    # ltdrdata/ComfyUI-MAINodes -> matlowai
    install_node.sh "DarioFT/ComfyUI-Qwen3-TTS" "ComfyUI-Qwen3-TTS" && \
    # 1038lab/ComfyUI-Qwen3-TTS -> DarioFT
    install_node.sh "sumeetprashant/ComfyUI-SolAttn" "ComfyUI-SolAttn_triton" && \
    # lovis-cherry/ComfyUI-SolAttn_triton -> sumeetprashant
    install_node.sh "GACLove/ComfyUI-VFI" "ComfyUI-VFI" && \
    # kijai/ComfyUI-VFI -> GACLove
    install_node.sh "Saganaki22/ComfyUI-sol-attn" "ComfyUI-sol-attn" && \
    # lovis-cherry/ComfyUI-sol-attn -> Saganaki22
    install_node.sh "RealRebelAI/ComfyUI_JoyAI_Echo_GGUF_Nodes" "ComfyUI_JoyAI_Echo_GGUF_Nodes" && \
    # city96/ComfyUI_JoyAI_Echo_GGUF_Nodes -> RealRebelAI
    install_node.sh "WhatDreamsCost/WhatDreamsCost-ComfyUI" "WhatDreamsCost-ComfyUI" && \
    # WhatDreamsCost/ComfyUI-WhatDreamsCost -> WhatDreamsCost/WhatDreamsCost-ComfyUI
    install_node.sh "pythongosssss/ComfyUI-Custom-Scripts" "comfyui-custom-scripts" && \
    install_node.sh "yolain/ComfyUI-Easy-Use" "comfyui-easy-use" && \
    install_node.sh "Fannovel16/ComfyUI-Frame-Interpolation" "comfyui-frame-interpolation" && \
    # kijai/ComfyUI-Frame-Interpolation -> Fannovel16
    install_node.sh "kijai/ComfyUI-KJNodes" "comfyui-kjnodes" && \
    install_node.sh "obvpm/ComfyUI-obvpm" "comfyui-obvpm" && \
    install_node.sh "xmarre/ComfyUI-Spectrum-MiniMax-H3" "comfyui-spectrum-minimax-h3" && \
    # kijai/ComfyUI-spectrum-minimax-h3 -> xmarre
    install_node.sh "Kosinkadink/ComfyUI-VideoHelperSuite" "comfyui-videohelpersuite" && \
    install_node.sh "chflame163/ComfyUI_LayerStyle" "comfyui_layerstyle" && \
    install_node.sh "plaguekind/ComfyUI-plaguekind-nodes" "plaguekind-nodes" && \
    install_node.sh "rgthree/rgthree-comfy" "rgthree-comfy" && \
    install_node.sh "numz/ComfyUI-SeedVR2_VideoUpscaler" "seedvr2_videoupscaler" && \
    # numz/SeedVR2_VideoUpscaler -> numz/ComfyUI-SeedVR2_VideoUpscaler
    echo "=== 22 custom nodes install attempt complete ==="

# === SageAttention (ROCm fork by guinmoon, NOT thu-ml) ===
# The base already has sageattention in pip deps? No — SageAttention needs a
# ROCm-specific wheel. Install here.
# MUTUALLY EXCLUSIVE: --use-sage-attention CLI flag with headdim in [64, 96, 128]
RUN pip3 install --break-system-packages --no-cache-dir \
    https://github.com/guinmoon/SageAttention-Rocm7/releases/download/v1.0.6_rocm7/sageattention-1.0.6-py3-none-any.whl \
    || echo "WARNING: sageattention-1.0.6 install failed"

WORKDIR /opt/ComfyUI

# Override the base CMD: add --enable-manager
# (inherits all other flags from base, e.g. --use-flash-attention, --gpu-only)
CMD ["python", "main.py", \
     "--use-flash-attention", \
     "--enable-manager", \
     "--enable-triton-backend", \
     "--supports-fp8-compute", \
     "--disable-pinned-memory", \
     "--gpu-only", \
     "--listen", "0.0.0.0", \
     "--port", "8188"]
