#!/usr/bin/env python3
"""Build H3 t2v smoke-test workflow JSON: 0.2 MP, 5 frames.

H3 frame grid: 17k+5. 5 frames is k=0 (minimum).
0.2 MP = 200,000 pixels, with both width and height multiples of 32.

Try 320x640 = 204,800 px (≈ 0.2 MP) — fits H3 max-pixel constraint
(BASE_SHORT_EDGE=768, MAX_PIXELS=768*1344=1,032,192). 320x640 is way under.
"""
import json
from pathlib import Path

# Models (int4 text encoder to fit in 40 GB GTT, just like the full run)
DIFFUSION_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "minimaxH3INT4Convrot_qwen3vl32bInt4.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_int8_convrot.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"

# 0.2 MP smoke test config
PROMPT = "A cat sitting on a windowsill, soft morning light, slow pan, calm atmosphere"
WIDTH = 320        # 320 × 640 = 204,800 px ≈ 0.2 MP
HEIGHT = 640
LENGTH = 5         # 5 frames (H3 minimum, k=0 in 17k+5 grid)
SEED = 42
STEPS = 8          # fewer steps for smoke test (vs 20 in full run)


def make_node(id, class_type, inputs=None, title=None):
    n = {"class_type": class_type, "inputs": inputs or {}}
    if title:
        n["_meta"] = {"title": title}
    return (id, n)


nodes = {
    **dict([make_node("1", "UNETLoader",
                      {"unet_name": DIFFUSION_MODEL, "weight_dtype": "default"},
                      "Load H3 Diffusion")]),
    **dict([make_node("2", "CLIPLoader",
                      {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"},
                      "Load H3 Text Encoder (qwen3vl int4)")]),
    **dict([make_node("3", "VAELoader",
                      {"vae_name": VAE_VIDEO},
                      "Load H3 Video VAE")]),
    **dict([make_node("4", "VAELoader",
                      {"vae_name": VAE_AUDIO},
                      "Load H3 Audio VAE")]),

    **dict([make_node("5", "EmptyMiniMaxH3LatentAV",
                      {"width": WIDTH, "height": HEIGHT, "length": LENGTH},
                      "Empty H3 AV Latent (5 frames)")]),

    **dict([make_node("6", "MiniMaxH3ImageToVideo",
                      {
                          "prompt": PROMPT,
                          "width": WIDTH,
                          "height": HEIGHT,
                          "length": LENGTH,
                          "clip": ["2", 0],
                          "vae": ["3", 0],
                      },
                      "H3 t2v Conditioning")]),

    **dict([make_node("7", "MiniMaxH3SigmaShift",
                      {"model": ["1", 0],
                       "shift_video": 12.0,
                       "shift_audio": 3.0},
                      "ModelSampling H3")]),

    **dict([make_node("8", "RandomNoise",
                      {"noise_seed": SEED},
                      "Random Noise")]),

    **dict([make_node("9", "KSamplerSelect",
                      {"sampler_name": "res_multistep"},
                      "KSampler Select")]),
    **dict([make_node("10", "BasicScheduler",
                      {"scheduler": "beta", "steps": STEPS, "denoise": 1.0,
                       "model": ["7", 0]},
                      "Basic Scheduler")]),
    **dict([make_node("11", "BasicGuider",
                      {"model": ["7", 0],
                       "conditioning": ["6", 0]},
                      "Basic Guider")]),
    **dict([make_node("12", "SamplerCustomAdvanced",
                      {"noise": ["8", 0],
                       "guider": ["11", 0],
                       "sampler": ["9", 0],
                       "sigmas": ["10", 0],
                       "latent_image": ["6", 1]},
                      "Sampler Custom Advanced")]),

    **dict([make_node("13", "VAEDecode",
                      {"samples": ["12", 0], "vae": ["3", 0]},
                      "VAE Decode Video")]),
    **dict([make_node("14", "VAEDecodeAudio",
                      {"samples": ["12", 0], "vae": ["4", 0]},
                      "VAE Decode Audio")]),

    **dict([make_node("15", "CreateVideo",
                      {"fps": 24, "bit_depth": 8, "color_space": "sRGB",
                       "images": ["13", 0], "audio": ["14", 0]},
                      "Create Video")]),

    **dict([make_node("16", "SaveVideo",
                      {"filename_prefix": "video/H3_smoke",
                       "format": "auto", "format.codec": "auto", "codec": "auto",
                       "video-preview": "",
                       "video": ["15", 0]},
                      "Save Video")]),
}

# ComfyUI uses node IDs as keys in a flat dict
workflow = {str(k): v for k, v in nodes.items()}

out_path = "/opt/comfyiu/workspace/h3_smoke_workflow.json"
with open(out_path, "w") as f:
    json.dump(workflow, f, indent=2)
print(f"wrote {out_path}")
print(f"config: {WIDTH}x{HEIGHT} = {WIDTH*HEIGHT} pixels ({WIDTH*HEIGHT/1e6:.2f} MP), {LENGTH} frames, {STEPS} steps")
print(f"nodes: {len(nodes)}")
