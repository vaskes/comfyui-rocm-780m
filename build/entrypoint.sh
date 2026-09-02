#!/bin/bash
# Container entrypoint — runs install_gpu_deps.sh on first start, then execs CMD
set -e
SCRIPT_DIR=/opt/comfyui-rocm-780m/scripts
if [ -f "$SCRIPT_DIR/install_gpu_deps.sh" ]; then
    bash "$SCRIPT_DIR/install_gpu_deps.sh"
fi
exec "$@"
