#!/bin/bash
# Waits until fewer than 3 run-experiment processes are alive, then launches
# the pima_diabetes experiment detached, with a job descriptor so the
# dashboard Training view tracks it (mirrors src/lib/qednet.ts behavior).
set -u

ROOT=/home/z/my-project
JOBS_DIR=$ROOT/experiments/jobs
LOG_DIR=$ROOT/experiments/logs
mkdir -p "$JOBS_DIR" "$LOG_DIR"

LAUNCH_LOG=$LOG_DIR/pima_launcher.log
echo "$(date '+%F %T') launcher started" >> "$LAUNCH_LOG"

while true; do
  running=$(pgrep -f "qednet.cli run-experiment" 2>/dev/null | wc -l)
  if [ "$running" -lt 3 ]; then
    break
  fi
  sleep 30
done

JOB_ID="job_$(date +%s | base36 2>/dev/null || python3 -c 'print(format(int(__import__("time").time()), "x"))')"
# simpler: build id the same way qednet.ts does (ts36 + random)
JOB_ID=$(python3 - <<'PY'
import time, random, string
ts = format(int(time.time() * 1000), "x")
rnd = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
print(f"job_{ts}_{rnd}")
PY
)
LOG_FILE=$JOBS_DIR/${JOB_ID}.log
STARTED_MS=$(python3 -c 'import time; print(int(time.time()*1000))')

cd "$ROOT/backend"
setsid nohup python3 -m qednet.cli run-experiment \
  --config "$ROOT/configs/experiments/pima_diabetes.yaml" \
  --exp-id exp_pima_diabetes \
  >> "$LOG_FILE" 2>&1 < /dev/null &
PID=$!

cat > "$JOBS_DIR/${JOB_ID}.json" <<EOF
{
  "jobId": "$JOB_ID",
  "experimentId": "exp_pima_diabetes",
  "configName": "pima_diabetes",
  "status": "running",
  "pid": $PID,
  "startedAt": $STARTED_MS,
  "endedAt": null,
  "logFile": "$LOG_FILE"
}
EOF

echo "$(date '+%F %T') launched pima_diabetes: jobId=$JOB_ID pid=$PID" >> "$LAUNCH_LOG"
