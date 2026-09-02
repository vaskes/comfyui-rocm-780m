#!/bin/bash
# Start ComfyUI with all available optimizations for the 780M.
# Use this as your default starting point. Tune from here.
set -e

cd /opt/Comfyiu || { echo "ComfyUI not installed at /opt/Comfyiu"; exit 1; }

# Stop any running container
docker rm -f comfyiu-test 2>/dev/null || true

# Start with all optimizations
docker run -d \
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
  -v /opt/comfyiu/workspace:/workspace \
  -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  comfyiu:therock-gfx1103-v5 \
  python main.py \
    --use-flash-attention \
    --enable-triton-backend \
    --supports-fp8-compute \
    --disable-pinned-memory \
    --gpu-only \
    --listen 0.0.0.0 \
    --port 8188

# Wait for it to be ready
echo "Waiting for ComfyUI to start..."
for i in {1..30}; do
    if curl -sf -m 2 http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        echo "ComfyUI is up at http://127.0.0.1:8188"
        exit 0
    fi
    sleep 2
done

echo "ComfyUI did not start in 60s. Check: docker logs comfyiu-test"
exit 1
