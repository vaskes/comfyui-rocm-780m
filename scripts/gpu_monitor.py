#!/usr/bin/env python3
"""Sample rocm-smi every N seconds, log GPU%, VRAM used/total, power, temp."""
import subprocess
import time
import sys
import os
import re

DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 60
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
LOG = sys.argv[3] if len(sys.argv) > 3 else "/opt/comfyiu/logs/gpu_monitor.log"

ROCM_SMI = "/opt/rocm/bin/rocm-smi"

os.makedirs(os.path.dirname(LOG), exist_ok=True)
with open(LOG, "w") as f:
    f.write(f"# GPU monitor: duration={DURATION}s interval={INTERVAL}s log={LOG}\n")
    f.write("# t(s) gpu% vram_used_mib vram_total_mib power_w temp_c\n")
    f.flush()


def parse(text):
    gpu_use = "N/A"
    vram_used_b = "N/A"
    vram_total_b = "N/A"
    power_w = "N/A"
    temp_c = "N/A"
    m = re.search(r"GPU use \(%\):\s*(\d+)", text)
    if m: gpu_use = int(m.group(1))
    m = re.search(r"VRAM Total Used Memory \(B\):\s*(\d+)", text)
    if m: vram_used_b = int(m.group(1))
    m = re.search(r"VRAM Total Memory \(B\):\s*(\d+)", text)
    if m: vram_total_b = int(m.group(1))
    m = re.search(r"Current Socket Graphics Package Power \(W\):\s*([\d.]+)", text)
    if m: power_w = float(m.group(1))
    m = re.search(r"Temperature \(Sensor edge\) \(C\):\s*([\d.]+)", text)
    if m: temp_c = float(m.group(1))
    # Convert VRAM bytes → MiB
    vram_used_mib = vram_used_b / (1024 * 1024) if isinstance(vram_used_b, int) else "N/A"
    vram_total_mib = vram_total_b / (1024 * 1024) if isinstance(vram_total_b, int) else "N/A"
    return gpu_use, vram_used_mib, vram_total_mib, power_w, temp_c


t0 = time.time()
end = t0 + DURATION
while time.time() < end:
    t = time.time() - t0
    try:
        out = subprocess.run(
            [ROCM_SMI, "--showuse", "--showmeminfo=VRAM", "--showtemp", "--showpower"],
            capture_output=True, text=True, timeout=5,
        )
        gpu_use, vu, vt, pw, tc = parse(out.stdout)
        with open(LOG, "a") as f:
            f.write(f"{t:.1f} {gpu_use} {vu:.0f} {vt:.0f} {pw} {tc}\n")
    except Exception as e:
        with open(LOG, "a") as f:
            f.write(f"{t:.1f} ERR {e}\n")
    time.sleep(INTERVAL)
print(f"Done. Wrote {LOG}")

