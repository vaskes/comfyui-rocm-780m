#!/usr/bin/env python3
"""Submit workflow and poll for completion, with monitor in same process."""
import argparse
import json
import sys
import time
import urllib.request
import urllib.error
import subprocess
from pathlib import Path


def submit_prompt(host, port, workflow):
    url = f"http://{host}:{port}/prompt"
    body = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["prompt_id"]


def wait_completion(host, port, prompt_id, timeout=600):
    url = f"http://{host}:{port}/history/{prompt_id}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                history = json.loads(resp.read())
            if prompt_id in history:
                return history[prompt_id]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                pass  # not started yet
            else:
                raise
        except Exception as e:
            print(f"  poll err: {e}", file=sys.stderr)
        time.sleep(0.5)
    raise TimeoutError(f"Prompt {prompt_id} not done in {timeout}s")


def get_gpu_stats():
    try:
        out = subprocess.run(
            ["/opt/rocm/bin/rocm-smi", "--showuse", "--showmeminfo=VRAM"],
            capture_output=True, text=True, timeout=5,
        )
        gpu = "?"
        vram_used = "?"
        vram_total = "?"
        for line in out.stdout.splitlines():
            if "GPU use" in line:
                gpu = line.split(":")[-1].strip()
            if "VRAM Total Used" in line:
                vram_used = int(line.split(":")[-1].strip()) // (1024*1024)
            if "VRAM Total Memory" in line:
                vram_total = int(line.split(":")[-1].strip()) // (1024*1024)
        return gpu, vram_used, vram_total
    except Exception as e:
        return "?", "?", "?"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8188)
    p.add_argument("--workflow", required=True)
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    wf = json.loads(Path(args.workflow).read_text())
    print(f"Submitting workflow: {args.workflow}")

    # Pre-warm: load model
    print("Submitting prompt...")
    t0 = time.time()
    pid = submit_prompt(args.host, args.port, wf)
    print(f"  prompt_id: {pid}")

    print("Polling for completion (with GPU monitor)...")
    last_t = 0
    peak_gpu = 0
    peak_vram = 0
    while True:
        elapsed = time.time() - t0
        gpu, vram_used, vram_total = get_gpu_stats()
        if gpu != "?":
            try:
                if int(gpu) > peak_gpu: peak_gpu = int(gpu)
            except: pass
            try:
                if int(vram_used) > peak_vram: peak_vram = int(vram_used)
            except: pass
        # Print every 2 sec
        if int(elapsed) // 2 != last_t // 2:
            print(f"  t={elapsed:5.1f}s  GPU={gpu}%  VRAM={vram_used}/{vram_total} MiB")
            last_t = int(elapsed)
        try:
            url = f"http://{args.host}:{args.port}/history/{pid}"
            with urllib.request.urlopen(url, timeout=2) as resp:
                history = json.loads(resp.read())
            if pid in history:
                elapsed = time.time() - t0
                print(f"\n✓ COMPLETED in {elapsed:.2f}s")
                print(f"  peak GPU%:  {peak_gpu}")
                print(f"  peak VRAM:  {peak_vram} MiB")
                # Output filenames
                for nid, out in history[pid].get("outputs", {}).items():
                    for img in out.get("images", []):
                        print(f"  image[{nid}]: {img.get('filename')}")
                return 0
        except urllib.error.HTTPError:
            pass
        except Exception as e:
            pass
        if elapsed > args.timeout:
            print(f"\n✗ TIMEOUT after {args.timeout}s")
            print(f"  peak GPU%:  {peak_gpu}")
            print(f"  peak VRAM:  {peak_vram} MiB")
            return 1
        time.sleep(0.5)


if __name__ == "__main__":
    sys.exit(main())
