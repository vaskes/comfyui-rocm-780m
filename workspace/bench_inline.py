#!/usr/bin/env python3
"""Single Python process: monitor GPU + submit workflow + print results."""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

LOG = "/opt/comfyiu/logs/vt_inline.log"
WORKFLOW = "/opt/comfyiu/workspace/test_workflow_sd15.json"
HOST = "127.0.0.1"
PORT = 8188

# Clear log
open(LOG, "w").close()


def monitor(stop_event, log):
    with open(log, "a") as f:
        f.write("# t(s) gpu% vram_used_mib vram_total_mib power_w temp_c\n")
        f.flush()
    t0 = time.time()
    while not stop_event.is_set():
        t = time.time() - t0
        try:
            out = subprocess.run(
                ["/opt/rocm/bin/rocm-smi", "--showuse", "--showmeminfo=VRAM", "--showtemp", "--showpower"],
                capture_output=True, text=True, timeout=5,
            )
            gpu = "0"
            vu = vt = pw = tc = "?"
            for line in out.stdout.splitlines():
                m = re.search(r"GPU use \(%\):\s*(\d+)", line)
                if m: gpu = m.group(1)
                m = re.search(r"VRAM Total Used Memory \(B\):\s*(\d+)", line)
                if m: vu = int(m.group(1)) // (1024 * 1024)
                m = re.search(r"VRAM Total Memory \(B\):\s*(\d+)", line)
                if m: vt = int(m.group(1)) // (1024 * 1024)
                m = re.search(r"Current Socket Graphics Package Power \(W\):\s*([\d.]+)", line)
                if m: pw = m.group(1)
                m = re.search(r"Temperature \(Sensor edge\) \(C\):\s*([\d.]+)", line)
                if m: tc = m.group(1)
            with open(log, "a") as f:
                f.write(f"{t:.2f} {gpu} {vu} {vt} {pw} {tc}\n")
        except Exception as e:
            with open(log, "a") as f:
                f.write(f"{t:.2f} ERR {e}\n")
        time.sleep(0.3)
    print("monitor done")


def submit(workflow, log):
    with open(workflow) as f:
        wf = json.load(f)
    body = json.dumps({"prompt": wf}).encode()
    t0 = time.time()
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/prompt",
        data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    pid = data["prompt_id"]
    print(f"submitted prompt_id={pid} at t={time.time()-t0:.2f}s")
    # poll for completion
    while True:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/history/{pid}", timeout=5) as r:
                hist = json.loads(r.read())
            if pid in hist:
                elapsed = time.time() - t0
                print(f"completed in {elapsed:.2f}s")
                with open(log, "a") as f:
                    f.write(f"# completed in {elapsed:.2f}s\n")
                return elapsed
        except Exception:
            pass
        if time.time() - t0 > 120:
            print("timeout!")
            return -1
        time.sleep(0.3)


stop = threading.Event()
mon_thread = threading.Thread(target=monitor, args=(stop, LOG), daemon=True)
mon_thread.start()
time.sleep(2)  # let monitor get baseline

# Submit and wait
elapsed = submit(WORKFLOW, LOG)

# Let monitor run a bit more to see post-inference
time.sleep(3)
stop.set()
mon_thread.join(timeout=3)

# Stats
peak_gpu = 0
peak_vram = 0
with open(LOG) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3 or parts[1] == "ERR":
            continue
        try:
            g = int(parts[1])
            v = int(parts[2])
            if g > peak_gpu: peak_gpu = g
            if v > peak_vram: peak_vram = v
        except:
            pass

print(f"\npeak GPU% = {peak_gpu}")
print(f"peak VRAM  = {peak_vram} MiB")
print(f"inference  = {elapsed:.2f}s")
