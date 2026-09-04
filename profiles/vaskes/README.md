# vaskes profile

A full H3 t2v workflow stack for one user's llmhost1 (nvidia) production
setup, compiled for ROCm on Radeon 780M.

## What's in this profile

### Custom nodes (22)

1. **ComfyUI-Manager** (`Comfy-Org/ComfyUI-Manager`) — UI for installing more nodes
2. **ComfyUI-GGUF** (`city96/ComfyUI-GGUF`) — GGUF model loader
3. **ComfyUI-H3-Multishot** (`jlucasmcrell`) — MiniMax-H3 long-form chaining
4. **ComfyUI-MAINodes** (`matlowai`) — MiniMax-H3 motion lab / contact-sheet
5. **ComfyUI-Qwen3-TTS** (`DarioFT`) — Qwen3-TTS custom voice / cloning
6. **ComfyUI-SolAttn_triton** (`sumeetprashant`) — NVIDIA Sol-Attn (sparse attention)
7. **ComfyUI-VFI** (`GACLove`) — video frame interpolation (RIFE)
8. **ComfyUI-sol-attn** (`Saganaki22`) — Sol-Attn MiniMax-H3 nodes
9. **ComfyUI_JoyAI_Echo_GGUF_Nodes** (`RealRebelAI`) — JoyAI-Echo GGUF
10. **WhatDreamsCost-ComfyUI** — LTX Director + various nodes
11. **comfyui-custom-scripts** (`pythongosssss`) — UI scripts
12. **comfyui-easy-use** (`yolain`) — convenience nodes
13. **comfyui-frame-interpolation** (`Fannovel16`) — VFI (multi-algo)
14. **comfyui-kjnodes** (`kijai`) — misc nodes
15. **comfyui-obvpm** (`obvpm`) — misc nodes
16. **comfyui-spectrum-minimax-h3** (`xmarre`) — Spectrum speedup for H3
17. **comfyui-videohelpersuite** (`Kosinkadink`) — video load/save
18. **comfyui_layerstyle** (`chflame163`) — layer effects
19. **plaguekind-nodes** (`plaguekind`) — misc nodes
20. **rgthree-comfy** (`rgthree`) — UI + node improvements
21. **seedvr2_videoupscaler** (`numz/ComfyUI-SeedVR2_VideoUpscaler`) — video upscaling
22. **ComfyUI-VideoHelperSuite** (above 17) — see Kosinkadink

### Pip deps (added on top of base)

```
opencv-contrib-python-headless  imageio-ffmpeg  imageio  matrix-nio
GitPython  diffusers  accelerate  transformers  safetensors
huggingface-hub  scikit-image  blend_modes  lark  descript-audiotools
qwen-tts  librosa  onnxruntime  conformer  gradio  gdown
hydra-core  HyperPyYAML  inflect  matplotlib  modelscope
omegaconf  openai-whisper  pyworld  tensorboard  wetext
gguf  rotary_embedding_torch  einops  sentencepiece  protobuf
```

ROCm-friendly. NOT included (CUDA-only): nvidia-vfx, onnxruntime-gpu, deepspeed.

### CLI flags (CMD in Dockerfile)

```
--use-flash-attention     # Triton ROCm backend (gfx1103 native)
--enable-manager          # UI for installing more nodes
--enable-triton-backend   # Triton kernels for some ops
--supports-fp8-compute    # comfy-kitchen fp8 ops
--disable-pinned-memory   # saves 42 GB pinned memory buffer
--gpu-only                # no CPU offload
--listen 0.0.0.0           # bind all interfaces
--port 8188               # ComfyUI default port
```

### Env vars (docker-compose)

| Var | Value | Why |
|---|---|---|
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `TRUE` | Triton ROCm backend for flash-attn |
| `BNB_ROCM_VERSION` | `72` | bnb has no ROCm 7.13 prebuilt; fall back to 7.2 binaries (format: digits only, no dot — 72 not 7.2) |
| `MIOPEN_LOG_LEVEL` | `3` | suppress gfx1103 kernel-db warning (MIOpen JIT-falls-back, harmless) |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | APU benefits |

## Build & run

```bash
# 1. Build planetary base (one-time)
cd ../../build
docker build -t comfyiu:base .

# 2. Build this profile
cd ../..
docker build -f profiles/vaskes/Dockerfile -t comfyiu:vaskes profiles/vaskes

# 3. Run
docker compose -f profiles/vaskes/docker-compose.yml up -d

# 4. Check
docker logs -f comfyui
```

## Custom-node volume mount

This profile mounts `/opt/comfyiu/custom_nodes:/opt/ComfyUI/custom_nodes`
as a host volume. This means:
- Nodes added via ComfyUI-Manager persist across container restarts.
- The image's 22 nodes are NOT auto-synced to the host (mount hides them).
- To re-sync image's 22 nodes to host (e.g. after a profile rebuild):
  ```bash
  docker compose -f profiles/vaskes/docker-compose.yml down
  docker run -d --name comfyiu-tmp comfyiu:vaskes sleep infinity
  sudo docker cp comfyiu-tmp:/opt/ComfyUI/custom_nodes/. /opt/comfyiu/custom_nodes/
  sudo chown -R <user>:<group> /opt/comfyiu/custom_nodes/
  docker stop comfyiu-tmp && docker rm comfyiu-tmp
  docker compose -f profiles/vaskes/docker-compose.yml up -d
  ```

## Known issues / fixes baked in

| Issue | Fix |
|---|---|
| flash-attn pip install needs GPU access (fails in docker build) | installed via entrypoint on first run (3-5 min) |
| SeedVR2 detection looks for `flash_attn_2_cuda` (CUDA-only) | stub created in base image (`flash_attn_2_cuda.py` is a no-op) |
| qwen-tts pins `accelerate==1.12.0` | pre-installed in base before qwen-tts |
| bnb no ROCm 7.13 prebuilt | `BNB_ROCM_VERSION=72` env var falls back to ROCm 7.2 binaries |
| MIOpen no gfx1103 conv kernel db | `MIOPEN_LOG_LEVEL=3` hides the JIT-fallback warning |
| 11 custom-node repos moved owners | corrected in `Dockerfile` with `X -> Y` comments |
| `jlucasmcrell/ComfyUI-H3-Multishot` default branch is `master`, not `main` | `install_node.sh` accepts `@branch` syntax |

## MUTUALLY EXCLUSIVE things (read before changing)

- **ROCm vs nvidia**: this profile is ROCm. Don't mix TheRock wheels with
  PyTorch.org cu130/cu132 wheels. For nvidia, use a separate image.
- **SageAttention**: ROCm uses `guinmoon/SageAttention-Rocm7` wheel, NOT
  thu-ml (which is CUDA-only). `--use-sage-attention` CLI flag requires
  headdim in [64, 96, 128] for the model, otherwise falls back to pytorch.
- **Attention backends** (pick ONE — see docs/ATTENTION_BACKENDS.md):
  - `--use-flash-attention` (default; Triton ROCm; works for most DiT/UNet)
  - `--use-ck-attention` (comfy-kitchen; best with `--supports-fp8-compute`)
  - `--use-sage-attention` (headdim in [64, 96, 128] only)
  - (no flag) pytorch SDPA fallback

## Original Dockerfile provenance

This profile was originally a single Dockerfile (v6) in `build/`. It was
split into "planetary base" + "private profile" on 2026-09-04 to keep the
main repo focused on the ROCm-780M bring-up, not on this user's specific
custom-node stack.
