# Attention Backends Comparison

The ComfyUI CLI has 5 mutually exclusive attention backends. Pick the one that matches your model.

| Backend | Flag | Best for | Notes |
|---|---|---|---|
| **pytorch SDPA** | (default) | small models, fallback | Uses `torch.nn.functional.scaled_dot_product_attention`. Always works. Slower than specialized backends. |
| **pytorch cross-attention** | `--use-pytorch-cross-attention` | SD 1.5/2, simple UNets | Sub-quadratic cross-attention. Old default. |
| **sage-attention** | `--use-sage-attention` | most models, but **headdim must be in [64, 96, 128]** | Falls back silently to pytorch if headdim mismatch. SD 1.5 attention has headdim 40/80 → falls back. |
| **flash-attention** | `--use-flash-attention` | DiT, large models, anything with big attention matrices | Best choice for H3. On gfx1103 uses **Triton ROCm backend** (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`). No C++ compile needed. |
| **Comfy Kitchen** | `--use-ck-attention` | fp8, quantized models, w4a8/w4a4/svdquant | Custom kernels via Triton or HIP backends. Combine with `--supports-fp8-compute` and `--enable-triton-backend`. |

## Benchmarks on 780M (SD 1.5, 512×512, 20 steps, seed=42)

| Setup | Time | Note |
|---|---|---|
| `sage+lowvram` | 18.88s | headdim 40/80, sage falls back to pytorch |
| `sage+highvram` | 13.29s | same fall-back, just less offloading |
| `highvram` | 16.29s | default pytorch attention |
| `ck+triton+fp8+gpu-only` | **13.10s** | Comfy Kitchen triton backend, fp8 compute |
| `flash+triton+fp8+gpu-only` | 15.41s | flash-attn via Triton ROCm |

For SD 1.5, **comfy-kitchen wins** (small headdim, custom fp8/int8 matmul, no offload).

For DiT (like H3), **flash-attention wins** (headdim is usually 64-128, big attention matrices benefit most from flash).

## How to check what's available

Inside the container:

```python
# Use comfy-kitchen's introspection
from comfy_kitchen import list_backends
print(list_backends())
# ['eager', 'triton', 'hip', 'cuda']  # cuda is disabled on ROCm
```

Check that your chosen backend is enabled:

```bash
# Should print: "Found comfy_kitchen backend triton: {'available': True, ...}"
docker logs comfyiu-test 2>&1 | grep "Found comfy_kitchen backend"
```

For flash-attn on gfx1103:

```bash
docker exec comfyiu-test python3 -c "
import os
assert os.environ.get('FLASH_ATTENTION_TRITON_AMD_ENABLE') == 'TRUE', 'set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE'
import flash_attn
from flash_attn import flash_attn_func
q = torch.randn(1, 4, 512, 64, dtype=torch.float16, device='cuda')
k = torch.randn(1, 4, 512, 64, dtype=torch.float16, device='cuda')
v = torch.randn(1, 4, 512, 64, dtype=torch.float16, device='cuda')
out = flash_attn_func(q, k, v)
print('flash_attn OK, out shape:', out.shape)
"
```

## Recommended starting recipe

For any new model, try this order:

```bash
# 1. Comfy Kitchen (best for fp8/quantized)
python main.py --use-ck-attention --enable-triton-backend --supports-fp8-compute --gpu-only

# 2. Flash Attention (best for big attention matrices)
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python main.py --use-flash-attention --gpu-only

# 3. Sage (best if headdim is 64/96/128)
python main.py --use-sage-attention --highvram

# 4. PyTorch (always works, slower)
python main.py --use-pytorch-cross-attention --highvram
```

Watch `gtt_used` and `gpu%` with `python3 scripts/gpu_monitor.py 60 0.3`. If GPU% > 90% sustained, the backend is doing real work. If GPU% is low, the bottleneck is elsewhere (loading, offloading, VAE decode).
