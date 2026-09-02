#!/usr/bin/env python3
"""Submit SD 1.5 workflow + monitor GPU/VRAM/GTT during inference."""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

LOG = "/opt/comfyiu/logs/bench_ck.log"
WORKFLOW = "/opt/comfyiu/workspace/test_workflow_sd15.json"
HOST = "127.0.0.1"
PORT = 8188


def read_sysfs(field):
    try:
        with open(f"/sys/class/drm/card0/device/mem_info_{field}") as f:
            return int(f.read().strip())
    except Exception:
        return -1


def report_state(stop_event, log):
    with open(log, "a") as f:
        f.write("# t(s) gpu% vram_mib gtt_mib vram_total_mib gtt_total_mib power_w temp_c\n")
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
            pw = tc = "?"
            for line in out.stdout.splitlines():
                m = re.search(r"GPU use \(%\):\s*(\d+)", line)
                if m: gpu = m.group(1)
                m = re.search(r"Current Socket Graphics Package Power \(W\):\s*([\d.]+)", line)
                if m: pw = m.group(1)
                m = re.search(r"Temperature \(Sensor edge\) \(C\):\s*([\d.]+)", line)
                if m: tc = m.group(1)
            v = read_sysfs("vram_used") / (1024*1024)
            g = read_sysfs("gtt_used") / (1024*1024)
            vt = read_sysfs("vram_total") / (1024*1024)
            gt = read_sysfs("gtt_total") / (1024*1024)
            with open(log, "a") as f:
                f.write(f"{t:.2f} {gpu} {v:.0f} {g:.0f} {vt:.0f} {gt:.0f} {pw} {tc}\n")
        except Exception as e:
            with open(log, "a") as f:
                f.write(f"{t:.2f} ERR {e}\n")
        time.sleep(0.3)


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
    print(f"submitted prompt_id={pid} at t={time.time()-t0:.2f}s", flush=True)
    while True:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{PORT}/history/{pid}", timeout=5) as r:
                hist = json.loads(r.read())
            if pid in hist:
                elapsed = time.time() - t0
                print(f"completed in {elapsed:.2f}s", flush=True)
                with open(log, "a") as f:
                    f.write(f"# completed in {elapsed:.2f}s\n")
                return elapsed
        except Exception:
            pass
        if time.time() - t0 > 180:
            print("timeout!", flush=True)
            return -1
        time.sleep(0.3)


stop = threading.Event()
mon_thread = threading.Thread(target=report_state, args=(stop, LOG), daemon=True)
mon_thread.start()
time.sleep(2)
elapsed = submit(WORKFLOW, LOG)
time.sleep(2)
stop.set()
mon_thread.join(timeout=3)

# Stats
peak_gpu = 0
peak_vram = 0
peak_gtt = 0
with open(LOG) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4 or parts[1] == "ERR":
            continue
        try:
            g = int(parts[1])
            v = float(parts[2])
            gt = float(parts[3])
            if g > peak_gpu: peak_gpu = g
            if v > peak_vram: peak_vram = v
            if gt > peak_gtt: peak_gtt = gt
        except:
            pass

print(f"\npeak GPU%   = {peak_gpu}")
print(f"peak VRAM    = {peak_vram:.0f} MiB")
print(f"peak GTT     = {peak_gtt:.0f} MiB")
print(f"inference    = {elapsed:.2f}s")
