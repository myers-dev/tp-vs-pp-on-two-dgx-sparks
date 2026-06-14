#!/usr/bin/env bash
# launch/serve-qwen25-72b-tp.sh
# Serve Qwen/Qwen2.5-72B-Instruct-AWQ with Tensor Parallelism (TP=2)
#
# Engine: vLLM 0.19.0 (nvcr.io/nvidia/vllm:26.04-py3)
# Parallelism: --tensor-parallel-size 2
# Quant: AWQ-Int4  (--quantization awq_marlin)
#
# Result from the post: 364 tok/s output, 17.1 Gb/s inter-node, conc 48
#
# IMPORTANT — concurrency caveat (verbatim from project log):
#   AWQ-dequant + enforce-eager + cross-node all-reduce at concurrency 128
#   saturates both GB10 nodes (sshd dies, bench hangs 0/1000).  This was
#   reproduced twice.  The successful run used --max-concurrency 48 with
#   the bench client on the WORKER node (off-head).
#   See bench.sh: use CONCURRENCY=48 for this model's TP run.
#
# Prerequisites:
#   1. source launch/env.sh
#   2. Ray cluster up (launch/ray-start.sh)
#   3. Run bench.sh from the WORKER node (10.88.1.119), not from HEAD.
#
# Run on HEAD node.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

MODEL="Qwen/Qwen2.5-72B-Instruct-AWQ"
MODEL_PATH="${MODELS_DIR}/Qwen2.5-72B-Instruct-AWQ"

echo "=== Serving ${MODEL} TP=2 on port ${SERVE_PORT} ==="
echo "    WARNING: run bench.sh from WORKER (${WORKER_IP}), not HEAD."
echo "    WARNING: use --max-concurrency 48 (not 128) — see header comment."

docker run -d --name vllm-serve \
    "${DOCKER_FLAGS[@]}" \
    -e VLLM_HOST_IP="${HEAD_IP}" \
    "${VLLM_IMAGE}" \
    python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${MODEL}" \
        --tensor-parallel-size 2 \
        --quantization awq_marlin \
        --enforce-eager \
        --port "${SERVE_PORT}" \
        --host 0.0.0.0

echo ""
echo "Waiting for server to be ready (~120 s)..."
until curl -sf "http://localhost:${SERVE_PORT}/health" > /dev/null 2>&1; do
    sleep 5
done
echo "Server ready."
echo "Run from WORKER: bash launch/bench.sh ${MODEL} 48"
