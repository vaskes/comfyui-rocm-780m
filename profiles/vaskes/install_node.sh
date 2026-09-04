#!/bin/bash
# install_node.sh — clone a GitHub repo as a ComfyUI custom node
# Usage: install_node.sh <github_src> <dir_name>
#   <github_src> = "owner/repo" or "owner/repo@branch"
#   <dir_name>   = directory name under custom_nodes/

set -u
src="$1"
dir="$2"
ref="main"
if [[ "$src" == *"@"* ]]; then
    ref="${src##*@}"
    src="${src%@*}"
fi
echo "=== Installing $dir from $src @ $ref ==="
rm -rf "$dir" 2>/dev/null
if curl -L -s -f -o /tmp/cnode.tar.gz "https://github.com/$src/archive/refs/heads/$ref.tar.gz"; then
    mkdir -p "$dir"
    if tar xzf /tmp/cnode.tar.gz -C "$dir" --strip-components=1 2>/dev/null; then
        rm -f /tmp/cnode.tar.gz
        if [ -f "$dir/requirements.txt" ]; then
            pip3 install --break-system-packages --no-cache-dir -r "$dir/requirements.txt" 2>&1 | tail -3 || true
        fi
        if [ -f "$dir/install.py" ]; then
            (cd "$dir" && python3 install.py 2>&1 | tail -3) || true
        fi
        echo "    [OK] $dir"
    else
        echo "    [FAIL: tar extract] $dir"
    fi
else
    echo "    [FAIL: download] $dir"
fi
