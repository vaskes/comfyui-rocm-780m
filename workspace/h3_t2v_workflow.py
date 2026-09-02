#!/usr/bin/env python3
"""Build official H3 t2v workflow JSON using only standard ComfyUI nodes."""
import json
from pathlib import Path


def make_node(id, class_type, inputs=None, title=None):
    n = {"class_type": class_type, "inputs": inputs or {}}
    if title:
        n["_meta"] = {"title": title}
    return (id, n)


# Models the user placed
DIFFUSION_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
# Use the int4 text encoder (15 GB) instead of int8 (26 GB) — saves 11 GB
TEXT_ENCODER = "minimaxH3INT4Convrot_qwen3vl32bInt4.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_int8_convrot.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"

PROMPT = "A cat sitting on a windowsill, soft morning light, slow pan, calm atmosphere"

WIDTH = 832
HEIGHT = 480
LENGTH = 124  # 5 seconds @ 24fps (trained range starts here)
SEED = 42
STEPS = 20

nodes = {
    # 1. Load models
    **dict([make_node("1", "UNETLoader",
                      {"unet_name": DIFFUSION_MODEL, "weight_dtype": "default"},
                      "Load H3 Diffusion")]),
    **dict([make_node("2", "CLIPLoader",
                      {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"},
                      "Load H3 Text Encoder (qwen3vl)")]),
    **dict([make_node("3", "VAELoader",
                      {"vae_name": VAE_VIDEO},
                      "Load H3 Video VAE")]),
    **dict([make_node("4", "VAELoader",
                      {"vae_name": VAE_AUDIO},
                      "Load H3 Audio VAE")]),

    # 2. Empty AV latent (will be replaced by MiniMaxH3ImageToVideo output)
    **dict([make_node("5", "EmptyMiniMaxH3LatentAV",
                      {"width": WIDTH, "height": HEIGHT, "length": LENGTH},
                      "Empty H3 AV Latent")]),

    # 3. Image-to-Video conditioning (this is t2v: no first/last frames, just text)
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

    # 4. Model sampling config (sigma shift for H3)
    **dict([make_node("7", "MiniMaxH3SigmaShift",
                      {"model": ["1", 0],
                       "shift_video": 12.0,
                       "shift_audio": 3.0},
                      "ModelSampling H3")]),

    # 5. Random noise
    **dict([make_node("8", "RandomNoise",
                      {"noise_seed": SEED},
                      "Random Noise")]),

    # 6. Sampler (use the ref2va pattern: BasicScheduler + SamplerCustomAdvanced + BasicGuider)
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

    # 7. VAE Decode (video + audio)
    **dict([make_node("13", "VAEDecode",
                      {"samples": ["12", 0], "vae": ["3", 0]},
                      "VAE Decode Video")]),
    **dict([make_node("14", "VAEDecodeAudio",
                      {"samples": ["12", 0], "vae": ["4", 0]},
                      "VAE Decode Audio")]),

    # 8. Create video
    **dict([make_node("15", "CreateVideo",
                      {"fps": 24, "bit_depth": 8, "color_space": "sRGB",
                       "images": ["13", 0], "audio": ["14", 0]},
                      "Create Video")]),

    # 9. Save video
    **dict([make_node("16", "SaveVideo",
                      {"filename_prefix": "video/H3_t2v",
                       "format": "auto", "format.codec": "auto", "codec": "auto",
                       "video-preview": "",
                       "video": ["15", 0]},
                      "Save Video")]),
}

# ComfyUI uses node IDs as keys in a flat dict
workflow = {str(k): v for k, v in nodes.items()}

out_path = "/workspace/h3_t2v_workflow.json"
with open(out_path, "w") as f:
    json.dump(workflow, f, indent=2)
print(f"wrote {out_path}")
print(f"nodes: {list(nodes.keys())}")
