#!/bin/bash
# /opt/comfyiu/scripts/gpu_monitor.sh
# Sample rocm-smi every N seconds, write timestamped log
DURATION=${1:-60}
INTERVAL=${2:-1}
LOG=${3:-/opt/comfyiu/logs/gpu_monitor.log}
echo "GPU monitor: duration=${DURATION}s, interval=${INTERVAL}s, log=${LOG}"
mkdir -p "$(dirname "$LOG")"
> "$LOG"
echo "# Started $(date)" >> "$LOG"
echo "# t(s) gpu% vram% vram_used_gb vram_total_gb power temp" >> "$LOG"
END=$(($(date +%s) + DURATION))
T0=$(date +%s)
while [ $(date +%s) -lt $END ]; do
    T=$(($(date +%s) - T0))
    # rocm-smi: parse VRAM and GPU%
    LINE=$(/opt/rocm/bin/rocm-smi --showuse --showmeminfo=used --showtemp --showpower --csv 2>/dev/null | tail -1)
    echo "$T $LINE" >> "$LOG"
    sleep "$INTERVAL"
done
echo "# Done $(date)" >> "$LOG"
