#!/bin/bash
# Optimization comparison: benchmark each ComfyUI config against the same workflow
# Usage: bash optim_compare.sh <workflow.json> [output_dir]
# Compares: time per step, peak GPU%, peak VRAM, output file
set -e

WORKFLOW=${1:-/opt/comfyiu/workspace/test_workflow_sd15.json}
OUT_DIR=${2:-/opt/comfyiu/logs/optim_compare_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT_DIR"
echo "Output dir: $OUT_DIR" | tee -a "$OUT_DIR/_summary.txt"

# Stop any running comfyiu
sg docker -c "docker rm -f comfyiu-test 2>/dev/null" >/dev/null || true
sleep 2

# Each config: name|docker_args(comfy_args)
# Note: sage falls back to pytorch for SD 1.5 (headdim 40/80 not in [64,96,128])
CONFIGS=(
    "A_baseline_lowvram_sage|--use-sage-attention --lowvram"
    "B_highvram_sage|--use-sage-attention --highvram"
    "C_highvram_pytorch|--highvram"
    "D_lowvram_pytorch|--lowvram"
    "E_flash_highvram|--use-flash-attention --highvram"
    "F_flash_lowvram|--use-flash-attention --lowvram"
    "G_normalvram|--normalvram"
    "H_pytorch_cross_highvram|--use-pytorch-cross-attention --highvram"
)

for cfg in "${CONFIGS[@]}"; do
    NAME=${cfg%%|*}
    ARGS=${cfg#*|}
    echo
    echo "===================================================="
    echo "TEST: $NAME  ($ARGS)"
    echo "====================================================" | tee -a "$OUT_DIR/_summary.txt"
    echo "TEST: $NAME  ($ARGS)" | tee -a "$OUT_DIR/_summary.txt"

    # Start container with these args
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
      python main.py $ARGS --listen 0.0.0.0 --port 8188" >/dev/null 2>&1

    # Wait for server
    for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        if curl -m 2 -sw "%{http_code}" http://127.0.0.1:8188/system_stats -o /dev/null 2>/dev/null | grep -q 200; then
            break
        fi
        sleep 2
    done
    sleep 3

    # Wait a bit more for comfy to fully init
    sleep 3

    # Start monitor
    MON_LOG="$OUT_DIR/${NAME}_monitor.log"
    python3 /opt/comfyiu/scripts/gpu_monitor.py 180 0.3 "$MON_LOG" >/dev/null 2>&1 &
    MON=$!
    sleep 1

    # Run test
    TEST_LOG="$OUT_DIR/${NAME}_test.log"
    echo "  test start $(date +%H:%M:%S)" | tee -a "$OUT_DIR/_summary.txt"
    timeout 150 python3 /opt/comfyiu/workspace/submit_and_watch.py \
        --host 127.0.0.1 --port 8188 --workflow "$WORKFLOW" --timeout 120 \
        > "$TEST_LOG" 2>&1
    TEST_RC=$?

    # Stop monitor
    sleep 1
    kill $MON 2>/dev/null
    wait $MON 2>/dev/null

    # Collect stats
    COMPLETED=$(grep "COMPLETED" "$TEST_LOG" | head -1)
    PEAK_GPU=$(awk 'NR>2 && $2 != "ERR" {if ($2+0 > g) g=$2+0} END {print g+0}' "$MON_LOG")
    PEAK_VRAM=$(awk 'NR>2 && $2 != "ERR" {if ($3+0 > v) v=$3+0} END {print v+0}' "$MON_LOG")
    IMG=$(grep "image\[" "$TEST_LOG" | head -1)

    echo "  $COMPLETED" | tee -a "$OUT_DIR/_summary.txt"
    echo "  peak GPU%=$PEAK_GPU  peak VRAM=${PEAK_VRAM} MiB  $IMG" | tee -a "$OUT_DIR/_summary.txt"
    echo "  test_rc=$TEST_RC" | tee -a "$OUT_DIR/_summary.txt"

    # Stop container
    sg docker -c "docker rm -f comfyiu-test" >/dev/null 2>&1
    sleep 3
done

echo
echo "===================================================="
echo "ALL TESTS COMPLETE"
echo "===================================================="
echo "Summary saved to: $OUT_DIR/_summary.txt"
