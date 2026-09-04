# Profiles

This directory contains **workflow-specific profiles** for ComfyUI on Radeon 780M.
The planetary base image (`comfyiu:base`, built from `../build/Dockerfile`)
contains only the minimum to run ComfyUI on gfx1103 — no custom nodes.

Each profile is a thin layer on top of the base, adding a specific
set of custom nodes + pip deps + entrypoint flags for a user's workflow.

## Why profiles?

The planetary problem ("make ComfyUI run on a Radeon 780M APU with ROCm 7.13")
is solved by `comfyiu:base`. But every user has different custom nodes and
pip deps. Keeping those in the main repo would bloat the base image and
make it hard to share between users.

A profile is a self-contained, opinionated extension:
- inherits base ROCm stack
- adds the user's specific custom nodes
- adds the user's specific pip deps
- sets the user's specific CLI flags (e.g. `--enable-manager`)

## Available profiles

| Profile | Use case | Custom nodes |
|---|---|---|
| [`vaskes/`](vaskes/) | One user's full H3 t2v + GGUF + VFI + Qwen3-TTS + SeedVR2 stack | 22 |

## Adding a new profile

1. Copy the closest existing profile:
   ```bash
   cp -R profiles/vaskes profiles/yourname
   ```
2. Edit `Dockerfile`: add/remove pip deps and `install_node.sh` calls.
3. Edit `docker-compose.yml`: tweak env vars, container_name, command.
4. Build and run:
   ```bash
   # First build base (if you haven't yet)
   cd build && docker build -t comfyiu:base .
   # Then build your profile
   cd ..
   docker build -f profiles/yourname/Dockerfile -t comfyiu:yourname profiles/yourname
   docker compose -f profiles/yourname/docker-compose.yml up -d
   ```

## Profile layout

```
profiles/<name>/
├── Dockerfile         # FROM comfyiu:base + this profile's deps + custom nodes
├── docker-compose.yml # env vars + volume mounts + container_name for this profile
├── install_node.sh    # (optional) custom-node install helper
├── install_gpu_deps.sh # (optional) flash-attn / bitsandbytes first-run install
├── entrypoint.sh      # (optional) container entrypoint
└── README.md          # what's in this profile
```

## Sharing

Profiles are intentionally simple (just a Dockerfile + compose + helper scripts).
You can fork a profile, tweak it, and share it back via PR. Profiles are
inherently personal (everyone's workflow is different), so the expectation
is that the maintainer of each profile owns and updates it.

## The planetary base

If you don't need a custom node stack, just use the base:

```bash
cd build
docker build -t comfyiu:base .
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined --group-add 992 --group-add 44 \
    -p 8188:8188 -v /opt/comfyiu/models:/opt/ComfyUI/models \
    comfyiu:base
```

See `../README.md` for the full planetary documentation.
