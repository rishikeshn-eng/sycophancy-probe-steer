#!/bin/bash
# usage: run_job.sh <logname> <python args...>   (detached, holds the shared GPU lock)
cd "$(dirname "$0")/scripts"
LOG=../logs/$1.log; shift
nohup ~/ai-safety-portfolio/_shared/gpulock.sh ../.venv/bin/python -u "$@" > "$LOG" 2>&1 < /dev/null &
echo "started pid $! -> $LOG"
