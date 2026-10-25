#!/usr/bin/env bash
# power_sampler.sh (responses R12, Q8 d): a sidecar that samples the GPU power draw once per second while a run.sh run holds the lock, and
# writes logs/power_<label>.csv (timestamp, watts). It touches no model code. Integrate afterwards with eval/power_energy.py.
# Usage (from the repository root, right after starting ./run.sh): eval/power_sampler.sh <label> [lock file, default logs/.run.lock]
# With concurrent runs (run.sh slots, R22) the GPU power is that of all of them: label the energy "concurrent (N runs)".
set -euo pipefail
label=${1:?label}; lock=${2:-logs/.run.lock}; out=logs/power_${label}.csv; mkdir -p logs
for _ in $(seq 120); do [ -f "$lock" ] && break; sleep 1; done
pid=$(cat "$lock"); echo "timestamp,watts" > "$out"
nvidia-smi --query-gpu=timestamp,power.draw --format=csv,noheader,nounits -l 1 >> "$out" &
smi=$!; while kill -0 "$pid" 2>/dev/null; do sleep 5; done; kill "$smi" 2>/dev/null || true
echo "power samples written: $out"
