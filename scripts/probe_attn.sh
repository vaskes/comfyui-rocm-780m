#!/bin/bash
# Probe attention backends available in the comfyiu container
set +e
echo "=== torch version ==="
python3 -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())" 2>&1
echo
echo "=== pip list (attention-related) ==="
pip3 list --break-system-packages 2>/dev/null | grep -iE "sage|flash|triton|bitsandbytes|kitchen|attn|amd|aiter" | head -20
echo
echo "=== check imports ==="
for mod in sageattention flash_attn bitsandbytes comfy_kitchen comfy_aimdo amd_aiter; do
    python3 -c "import $mod; print('$mod:', getattr($mod, '__version__', 'OK'))" 2>&1 | head -1
done
echo
echo "=== triton ==="
python3 -c "import triton; print('triton:', triton.__version__)" 2>&1
echo
echo "=== comfyui attention backends ==="
python3 -c "
import sys
sys.path.insert(0, '/opt/ComfyUI')
from comfy.ldm.modules.attention import attention_backends
print('Available backends:', list(attention_backends.keys()) if isinstance(attention_backends, dict) else 'not a dict')
" 2>&1 | head -20
echo
echo "=== comfyui args ==="
grep -E "args.use_|args.enable_|args.cpu_" /opt/ComfyUI/comfy/cli_args.py 2>&1 | head -30
echo
echo "=== comfy-kitchen backends detected ==="
python3 -c "
import sys
sys.path.insert(0, '/opt/ComfyUI')
import comfy_kitchen
print('comfy_kitchen OK')
print(dir(comfy_kitchen))
" 2>&1 | head -20
