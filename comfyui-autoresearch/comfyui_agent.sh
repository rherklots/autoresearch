#!/bin/bash
# comfyui_agent.sh — Autonomous ComfyUI LPIPS optimization agent
# Polls for trigger files, runs LPIPS on Mac, writes results
# Designed to run 24/7 as a background job on Mac (via launchd or manual)

TRIGGER_DIR="/Volumes/scripts/comfyui/comfyui-autoresearch"
RESULT_FILE="$TRIGGER_DIR/.last_result.json"
LOG="$TRIGGER_DIR/agent.log"

echo "[$(date)] comfyui_agent.sh gestart — polling $TRIGGER_DIR" | tee -a "$LOG"

while true; do
  # Check for trigger file
  for trigger in "$TRIGGER_DIR"/.trigger_*.json; do
    [[ -e "$trigger" ]] || continue

    params=$(cat "$trigger")
    seed=$(echo "$params" | python3 -c "import sys,json; print(json.load(sys.stdin).get('seed',999))")
    steps=$(echo "$params" | python3 -c "import sys,json; print(json.load(sys.stdin).get('steps',30))")
    guidance=$(echo "$params" | python3 -c "import sys,json; print(json.load(sys.stdin).get('guidance',2.5))")
    max_shift=$(echo "$params" | python3 -c "import sys,json; print(json.load(sys.stdin).get('max_shift',1.15))")
    base_shift=$(echo "$params" | python3 -c "import sys,json; print(json.load(sys.stdin).get('base_shift',0.5))")
    run_id=$(echo "$params" | python3 -c "import sys,json; print(json.load(sys.stdin).get('run_id','unknown'))")

    echo "[$(date)] Trigger ontvangen: run_id=$run_id seed=$seed" | tee -a "$LOG"

    # Run score script
    cd "$TRIGGER_DIR"
    python3 score_script.py \
      --seed "$seed" \
      --steps "$steps" \
      --guidance "$guidance" \
      --max_shift "$max_shift" \
      --base_shift "$base_shift" \
      --ref_dir "./refs" \
      --output_dir "./outputs" \
      > "$TRIGGER_DIR/run_${run_id}.log" 2>&1

    # Extract results
    if grep -q "^val_score:" "$TRIGGER_DIR/run_${run_id}.log"; then
      val_score=$(grep "^val_score:" "$TRIGGER_DIR/run_${run_id}.log" | awk '{print $2}')
      gen_time=$(grep "^generation_time:" "$TRIGGER_DIR/run_${run_id}.log" | awk '{print $2}')
      file_size=$(grep "^file_size_kb:" "$TRIGGER_DIR/run_${run_id}.log" | awk '{print $2}')

      # Write result JSON
      cat > "$RESULT_FILE" <<EOF
{
  "run_id": "$run_id",
  "val_score": $val_score,
  "generation_time": $gen_time,
  "file_size_kb": $file_size,
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
      echo "[$(date)] Resultaat: val_score=$val_score ($gen_time s, ${file_size}KB)" | tee -a "$LOG"
    else
      echo "[$(date)] FOUT: score_script.py produceerde geen val_score" | tee -a "$LOG"
    fi

    rm -f "$trigger"
  done

  sleep 3
done
