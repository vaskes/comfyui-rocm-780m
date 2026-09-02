#!/bin/bash
# Benchmark with high-frequency GPU monitoring
set -e
LOG=/opt/comfyiu/logs/bench_full_$(date +%Y%m%d_%H%M%S).log
WORKFLOW=${1:-/opt/comfyiu/workspace/test_workflow_sd15.json}
DURATION=${2:-120}
TEST_TIMEOUT=${3:-180}
INTERVAL=${4:-0.2}

echo "=== benchmark started $(date) ===" | tee -a $LOG
echo "workflow=$WORKFLOW duration=$DURATION test_timeout=$TEST_TIMEOUT interval=$INTERVAL" | tee -a $LOG

# Start monitor at high frequency
python3 /opt/comfyiu/scripts/gpu_monitor.py $DURATION $INTERVAL /opt/comfyiu/logs/_monitor_full.log >> $LOG 2>&1 &
MON=$!
echo "monitor pid=$MON" | tee -a $LOG
sleep 1

# Run test
echo "=== test start $(date) ===" | tee -a $LOG
timeout $TEST_TIMEOUT python3 /opt/comfyiu/workspace/submit_and_watch.py \
    --host 127.0.0.1 --port 8188 --workflow "$WORKFLOW" --timeout $((TEST_TIMEOUT - 10)) 2>&1 | tee -a $LOG
TEST_RC=${PIPESTATUS[0]}
echo "test_rc=$TEST_RC" | tee -a $LOG

# Stop monitor
kill $MON 2>/dev/null
wait $MON 2>/dev/null

# Stats
echo
echo "=== monitor stats (rows with GPU>0) ===" | tee -a $LOG
awk 'NR>2 && $2 != "ERR" && $2+0 > 0' /opt/comfyiu/logs/_monitor_full.log >> $LOG 2>&1
echo "=== peak stats ===" | tee -a $LOG
awk 'NR>2 && $2 != "ERR" {if ($2+0 > maxgpu) maxgpu=$2+0; if ($3+0 > maxvram) maxvram=$3+0} END {print "max GPU%=", maxgpu, " max VRAM MiB=", maxvram, " (total VRAM =", $4, "MiB)"}' /opt/comfyiu/logs/_monitor_full.log >> $LOG 2>&1
echo "=== count of samples with GPU>0 ===" | tee -a $LOG
awk 'NR>2 && $2 != "ERR" && $2+0 > 0' /opt/comfyiu/logs/_monitor_full.log | wc -l | tee -a $LOG
echo "=== total monitor samples ===" | tee -a $LOG
awk 'NR>2 && $2 != "ERR"' /opt/comfyiu/logs/_monitor_full.log | wc -l | tee -a $LOG

echo "=== output files ===" | tee -a $LOG
ls -la /opt/comfyiu/output/ >> $LOG 2>&1

echo "=== done $(date) ===" | tee -a $LOG
echo "FULL LOG: $LOG"
