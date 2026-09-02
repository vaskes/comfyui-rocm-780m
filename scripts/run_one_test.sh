#!/bin/bash
# Single test runner: starts container, runs test, stops container, appends to summary
# Usage: bash run_one_test.sh <name> <comfy_args...>
set -e
NAME=$1
shift
COMFY_ARGS="$@"
OUT_DIR=/opt/comfyiu/logs/optim_compare
WORKFLOW=/opt/comfyiu/workspace/test_workflow_sd15.json
SUMMARY="$OUT_DIR/_summary.txt"

mkdir -p "$OUT_DIR"

echo "====================================================" >> "$SUMMARY"
echo "TEST: $NAME  ($COMFY_ARGS)" >> "$SUMMARY"
echo "  start $(date +%H:%M:%S)" >> "$SUMMARY"

# Stop any running
sg docker -c "docker rm -f comfyiu-test 2>/dev/null" >/dev/null 2>&1
sleep 2

# Start
sg docker -c "docker run -d \
  --name comfyiu-test \
  --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add 992 --group-add 44 \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -v /opt/comfyiu/models:/opt/ComfyUI/models \
  -v /opt/comfyiu/output:/opt/ComfyUI/output \
  -v /opt/comfyiu/input:/opt/ComfyUI/input \
  -v /opt/comfyiu/custom_nodes:/opt/ComfyUI/custom_nodes \
  -v /opt/comfyiu/user:/opt/ComfyUI/user \
  -v /opt/comfyiu/logs:/opt/ComfyUI/logs \
  comfyiu:therock-gfx1103-v4 \
  python main.py $COMFY_ARGS --listen 0.0.0.0 --port 8188" >/dev/null 2>&1

# Wait for server
SERVER_READY=0
for i in $(seq 1 30); do
    if curl -m 2 -s http://127.0.0.1:8188/system_stats -o /dev/null 2>/dev/null; then
        SERVER_READY=1
        break
    fi
    sleep 2
done
if [ "$SERVER_READY" = "0" ]; then
    echo "  SERVER NOT READY" >> "$SUMMARY"
    sg docker -c "docker rm -f comfyiu-test" >/dev/null 2>&1
    exit 1
fi
sleep 5  # comfy init time

# Monitor
MON_LOG="$OUT_DIR/${NAME}_monitor.log"
python3 /opt/comfyiu/scripts/gpu_monitor.py 120 0.3 "$MON_LOG" >/dev/null 2>&1 &
MON=$!
sleep 1

# Test
TEST_LOG="$OUT_DIR/${NAME}_test.log"
timeout 90 python3 /opt/comfyiu/workspace/submit_and_watch.py \
    --host 127.0.0.1 --port 8188 --workflow "$WORKFLOW" --timeout 60 \
    > "$TEST_LOG" 2>&1
TEST_RC=$?

sleep 1
kill $MON 2>/dev/null
wait $MON 2>/dev/null

# Collect
COMPLETED=$(grep "COMPLETED" "$TEST_LOG" | head -1)
PEAK_GPU=$(awk 'NR>2 && $2 != "ERR" {if ($2+0 > g) g=$2+0} END {print g+0}' "$MON_LOG")
PEAK_VRAM=$(awk 'NR>2 && $2 != "ERR" {if ($3+0 > v) v=$3+0} END {print v+0}' "$MON_LOG")
SAMPLES_GT0=$(awk 'NR>2 && $2+0 > 0' "$MON_LOG" | wc -l)
TOTAL_SAMPLES=$(awk 'NR>2 && $2 != "ERR"' "$MON_LOG" | wc -l)
IMG=$(grep "image\[" "$TEST_LOG" | head -1)

echo "  $COMPLETED" >> "$SUMMARY"
echo "  peak GPU%=$PEAK_GPU  peak VRAM=${PEAK_VRAM} MiB  samples_with_GPU>0=$SAMPLES_GT0/$TOTAL_SAMPLES  $IMG" >> "$SUMMARY"
echo "  test_rc=$TEST_RC  end $(date +%H:%M:%S)" >> "$SUMMARY"
echo

# Stop container
sg docker -c "docker rm -f comfyiu-test" >/dev/null 2>&1
sleep 3
