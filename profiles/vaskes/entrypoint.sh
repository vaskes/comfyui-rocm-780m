#!/bin/bash
# Container entrypoint — runs install_gpu_deps.sh on first start, then execs CMD
set -e
SCRIPT_FILE=/usr/local/bin/install_gpu_deps.sh
if [ -f "$SCRIPT_FILE" ]; then
    bash "$SCRIPT_FILE"
fi
exec "$@"
