#!/bin/bash
# Try with HSA_OVERRIDE_GFX_VERSION=11.0.0 + various workarounds per the search results
set -e
sg docker -c "docker rm -f comfyiu-test 2>/dev/null || true"
sg docker -c "docker run -d \
  --name comfyiu-test \
  --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add 992 --group-add 44 \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -v /opt/comfyiu/models:/opt/ComfyUI/models \
  -v /opt/comfyiu/output:/opt/ComfyUI/output \
  -v /opt/comfyiu/input:/opt/ComfyUI/input \
  -v /opt/comfyiu/custom_nodes:/opt/ComfyUI/custom_nodes \
  -v /opt/comfyiu/user:/opt/ComfyUI/user \
  -v /opt/comfyiu/logs:/opt/ComfyUI/logs \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -e MIOPEN_FIND_MODE=1 \
  -e HSA_ENABLE_SDMA=0 \
  -e PYTORCH_TUNABLEOP_ENABLED=0 \
  comfyiu:therock-gfx1103-v4 \
  python main.py --use-quad-cross-attention --disable-pinned-memory --disable-async-offload --reserve-vram 2 --lowvram --listen 0.0.0.0 --port 8188"
echo container_started
