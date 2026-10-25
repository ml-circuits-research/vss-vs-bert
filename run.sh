#!/usr/bin/env bash
# Runs a python script and saves the log with the date and the current commit hash in its name (copied from the main repository's
# experiment/run.sh; the process pattern of the secondary guard is adapted to this repository's layout; nothing else changed).
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: ./run.sh script.py [arguments]"; exit 1; }
# logs/ is ignored by the check: new logs are always uncommitted at run time. Documents (*.md) are ignored too (2026-10-16): the hash in the
# log name must identify the CODE that ran; documents are edited while runs are queued.
if [ -n "$(git status --porcelain 2>/dev/null | grep -v "logs/" | grep -v "\.md$")" ]; then
  echo "There are uncommitted code changes. Commit first (git add -A && git commit), so that the log corresponds to a commit."; exit 1
fi
# protection (2026-10-16, after the machine hung on 2026-10-15): one run at a time, enough free memory, a wall-clock cap
LOCK=logs/.run.lock
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "Another run.sh is active (pid $(cat "$LOCK"), lock $LOCK). Runs are sequential; wait for it or stop it first."; exit 1
fi
# secondary guard (the lock is the primary one): other experiment processes, EXCLUDING this script's own process tree (a parent queue.sh or
# shell whose command line mentions a script name must not make run.sh refuse its own launch — judgment 2026-10-16 E6)
anc=""; p=$$; while [ "$p" -gt 1 ]; do anc="$anc $p"; p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); [ -n "$p" ] || break; done
others=$(pgrep -f "python3? (models|vss|data|eval)/[a-z_0-9]+\.py|python3? v[0-9]+_" | grep -vxF -f <(printf '%s\n' $anc) || true)
if [ -n "$others" ]; then
  echo "Another experiment process is running (pids: $others). Runs are sequential on this machine; wait for it or stop it first."; ps -o pid,args -p $(echo $others | tr ' ' ',') | head -4; exit 1
fi
avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
if [ "$avail_gb" -lt 40 ]; then echo "Only ${avail_gb} GB available; refusing to start (need 40)."; exit 1; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0   # (2026-10-25, responses R3) set and dict-from-set iteration order must not depend on the per-process string hash
export VSA_MEM_FRACTION=${VSA_MEM_FRACTION:-0.5} VSA_MIN_AVAILABLE_GB=${VSA_MIN_AVAILABLE_GB:-16} VSA_MAX_HOURS=${VSA_MAX_HOURS:-6}
mkdir -p logs
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
hash=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
name=$(basename "$1" .py)
out="logs/$(date +%Y-%m-%d_%H%M)_${name}_${hash}.txt"
{ echo "# $(date -Iseconds)  commit=$hash  cmd: python3 $*  (guard: mem fraction $VSA_MEM_FRACTION, min available $VSA_MIN_AVAILABLE_GB GB, max $VSA_MAX_HOURS h; MemAvailable at start ${avail_gb} GB)"; echo "# $(python3 --version 2>&1)  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"; } > "$out"
( timeout --signal=TERM $((${VSA_MAX_HOURS%.*}*3600+600)) python3 -u "$@" 2>&1 | tee -a "$out" ) &
pid=$!
# log-advance watchdog: if the log has not grown for STALL_MIN minutes, kill the run (default 30)
STALL_MIN=${STALL_MIN:-30}; last=$(stat -c %s "$out"); stalled=0
while kill -0 $pid 2>/dev/null; do
  sleep 60; now=$(stat -c %s "$out")
  if [ "$now" -eq "$last" ]; then stalled=$((stalled+1)); else stalled=0; last=$now; fi
  if [ $stalled -ge $STALL_MIN ]; then echo "# watchdog: log did not advance for $STALL_MIN min; killing the run" | tee -a "$out"; pkill -TERM -P $pid; kill -TERM $pid; break; fi
done
wait $pid
echo "log saved: $out"
