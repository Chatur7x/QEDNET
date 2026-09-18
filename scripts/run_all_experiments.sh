#!/bin/bash
# Run the four real experiments sequentially (background launcher).
cd /home/z/my-project
export PYTHONPATH=/home/z/my-project/backend
for cfg in breast_cancer heart_disease parkinsons pima_diabetes; do
  echo "[$(date '+%H:%M:%S')] starting $cfg" >> experiments/logs/run.log
  python3 -m qednet.cli run-experiment --config configs/experiments/$cfg.yaml \
    --verbose >> experiments/logs/$cfg.log 2>&1
  echo "[$(date '+%H:%M:%S')] finished $cfg (exit $?)" >> experiments/logs/run.log
done
echo "[$(date '+%H:%M:%S')] ALL EXPERIMENTS DONE" >> experiments/logs/run.log
