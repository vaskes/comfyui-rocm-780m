## Known issue: flash-attn Triton ROCm backend requires C compiler

If you see this in container logs:

```
Flash Attention failed, using default SDPA:
Failed to find C compiler. Please specify via CC environment variable
or set triton.knobs.build.impl.
```

It means Triton can't JIT-compile its kernels. The fix:

1. **Make sure your Dockerfile has `g++`, `cmake`, `ninja-build` in `apt-get install`**:
   ```dockerfile
   RUN apt-get install -y --no-install-recommends \
       python3.12 python3.12-venv python3.12-dev python3-pip \
       ...
       g++ cmake ninja-build
   ```
2. Or in a running container:
   ```bash
   docker exec -u root comfyiu-test apt-get install -y g++ cmake ninja-build
   docker restart comfyiu-test
   ```
3. Or pass env vars (doesn't always work):
   ```bash
   -e CC=gcc -e CXX=g++ -e triton.knobs.build.impl=python
   ```

When this happens, **flash-attn silently falls back to pytorch SDPA** — inference still works, but is ~1.3x slower for H3-style workloads (5.7 min/step vs ~4.5 min/step on the same 780M).

`comfy-kitchen` (`--use-ck-attention`) does NOT have this problem because its kernels are pre-compiled .so/.cubin files shipped in the wheel. Use it as a safe alternative:

```bash
python main.py --use-ck-attention --enable-triton-backend --supports-fp8-compute --gpu-only
```

This is documented in our live benchmark trace (`docs/H3_BENCHMARK.md`) — our first H3 t2va run hit exactly this issue, fell back to SDPA, and still produced correct output (just slower than it could have been with proper flash-attn).
