#!/usr/bin/env bash
# launch/bench.sh — run the benchmark against the live vLLM server
#
# Workload is identical across all 8 runs (verbatim from Notion methodology):
#   vllm bench serve, random dataset, ISL=OSL=1024, 1000 prompts, seed 42
#
# The bench client runs inside the same vLLM container image so no extra
# install is needed.
#
# Run this FROM A CLUSTER NODE (not from a Mac/laptop) so the client
# traffic does not traverse the upstream link and pollute the switch-traffic
# measurement.  For AWQ-TP and gpt-oss-TP specifically, run from the WORKER
# node (off-head) — see the respective serve scripts for the reason.
#
# Usage:
#   bash launch/bench.sh <model-name> [max-concurrency] [extra flags...]
#
# Examples:
#   bash launch/bench.sh "nvidia/Llama-3.3-70B-Instruct-FP4"
#   bash launch/bench.sh "Qwen/Qwen2.5-72B-Instruct-AWQ" 48
#   bash launch/bench.sh "openai/gpt-oss-120b" 48 --ignore-eos

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

MODEL="${1:?Usage: bench.sh <model-name> [max-concurrency] [extra-flags...]}"
MAX_CONCURRENCY="${2:-128}"
shift 2 2>/dev/null || shift 1 2>/dev/null || true
EXTRA_FLAGS=("$@")

RESULTS_DIR="${RESULTS_DIR:-./bench-results}"
mkdir -p "${RESULTS_DIR}"

echo "=== Benchmarking ${MODEL} (conc=${MAX_CONCURRENCY}) ==="
echo "    Server:   http://localhost:${SERVE_PORT}"
echo "    Results:  ${RESULTS_DIR}"
echo "    Started:  $(date)"

# vllm bench serve is bundled in the same container image.
docker run --rm \
    --network host \
    "${VLLM_IMAGE}" \
    python -m vllm.entrypoints.benchmark_serving \
        --backend openai-chat \
        --model "${MODEL}" \
        --host localhost \
        --port "${SERVE_PORT}" \
        --endpoint /v1/chat/completions \
        --dataset-name random \
        --random-input-len 1024 \
        --random-output-len 1024 \
        --num-prompts 1000 \
        --max-concurrency "${MAX_CONCURRENCY}" \
        --seed 42 \
        "${EXTRA_FLAGS[@]}" \
        --save-result \
        --result-dir "${RESULTS_DIR}"

echo ""
echo "Benchmark complete: $(date)"
echo "Results saved to ${RESULTS_DIR}"
