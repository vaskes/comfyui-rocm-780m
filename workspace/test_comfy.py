#!/usr/bin/env python3
"""
Test script: submit a workflow to ComfyUI API, time the generation.
Usage: python3 test_comfy.py --host 127.0.0.1 --port 8188
"""
import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def submit_prompt(host: str, port: int, workflow: dict) -> str:
    """POST /prompt, return prompt_id."""
    url = f"http://{host}:{port}/prompt"
    body = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["prompt_id"]


def wait_completion(host: str, port: int, prompt_id: str, timeout: float = 600) -> dict:
    """Poll /history/{prompt_id} until status indicates done."""
    url = f"http://{host}:{port}/history/{prompt_id}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                history = json.loads(resp.read())
            if prompt_id in history:
                return history[prompt_id]
        except urllib.error.HTTPError:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Prompt {prompt_id} did not complete in {timeout}s")


def wait_server(host: str, port: int, timeout: float = 120) -> None:
    """Wait for /system_stats_info endpoint to respond."""
    url = f"http://{host}:{port}/system_stats"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Server {host}:{port} not ready in {timeout}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8188)
    p.add_argument("--workflow", default="test_workflow_sd15.json")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--wait-server", type=int, default=120)
    args = p.parse_args()

    print(f"Waiting for ComfyUI at {args.host}:{args.port} ...")
    wait_server(args.host, args.port, args.wait_server)
    print("  server ready")

    wf = json.loads(Path(args.workflow).read_text())
    print(f"Submitting workflow: {args.workflow}")
    t0 = time.time()
    pid = submit_prompt(args.host, args.port, wf)
    print(f"  prompt_id: {pid}")

    print("Waiting for completion ...")
    result = wait_completion(args.host, args.port, pid, args.timeout)
    elapsed = time.time() - t0
    print(f"  completed in {elapsed:.1f}s")

    # Extract output image info
    if "outputs" in result:
        for node_id, out in result["outputs"].items():
            if "images" in out:
                for img in out["images"]:
                    print(f"  output[{node_id}]: {img.get('filename')}  "
                          f"({img.get('subfolder', '')}{'/' + img['subfolder'] if img.get('subfolder') else ''})")

    print(f"TOTAL ELAPSED: {elapsed:.2f} seconds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
