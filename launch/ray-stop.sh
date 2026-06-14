#!/usr/bin/env bash
# launch/ray-stop.sh — tear down Ray + vLLM containers between model runs
#
# Run on HEAD node; also run the worker equivalent on the worker node.
# Pattern from project log: "clean ray restart" between each model.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "=== Stopping vLLM serve + Ray on HEAD ==="
docker stop vllm-serve 2>/dev/null || true
docker rm   vllm-serve 2>/dev/null || true
docker stop ray-head   2>/dev/null || true
docker rm   ray-head   2>/dev/null || true

echo ""
echo "=== Also stop on WORKER (run on ${WORKER_IP}) ==="
echo "    docker stop vllm-worker ray-worker 2>/dev/null; docker rm vllm-worker ray-worker 2>/dev/null"
echo ""
echo "Wait ~10 s for ports to clear before starting the next model."
