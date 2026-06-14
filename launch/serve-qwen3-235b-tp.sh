#!/usr/bin/env bash
# launch/serve-qwen3-235b-tp.sh
# Serve Qwen/Qwen3-235B-A22B with Tensor Parallelism (TP=2)
#
# Engine: vLLM 0.19.0 (nvcr.io/nvidia/vllm:26.04-py3)
# Parallelism: --tensor-parallel-size 2
# Arch: MoE (235B total / ~22B active per token)
# Quant: NVFP4
#
# Result from the post: 307 tok/s output, 17.4 Gb/s inter-node, conc 128
#
# MoE TP finding: vLLM shards expert weights with TP (not EP/all-to-all),
# so MoE TP traffic follows the same 2L×h×tok/s law as dense.
# 235B lower absolute traffic than 70B because h=4096 (vs 8192) despite
# more layers (94 vs 80).
#
# --enforce-eager is REQUIRED (multi-node TP hangs during CUDA-graph
# capture on GB10 regardless of model).
#
# Prerequisites:
#   1. source launch/env.sh
#   2. Ray cluster up (launch/ray-start.sh)
#
# Run on HEAD node.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# TEMPLATED — model ID and local path
MODEL="Qwen/Qwen3-235B-A22B"
MODEL_PATH="${MODELS_DIR}/Qwen3-235B-A22B-FP4"

echo "=== Serving ${MODEL} TP=2 on port ${SERVE_PORT} ==="

docker run -d --name vllm-serve \
    "${DOCKER_FLAGS[@]}" \
    -e VLLM_HOST_IP="${HEAD_IP}" \
    "${VLLM_IMAGE}" \
    python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${MODEL}" \
        --tensor-parallel-size 2 \
        --enforce-eager \
        --port "${SERVE_PORT}" \
        --host 0.0.0.0

echo ""
echo "Waiting for server to be ready (~120-240 s for 235B)..."
until curl -sf "http://localhost:${SERVE_PORT}/health" > /dev/null 2>&1; do
    sleep 10
done
echo "Server ready.  Run: bash launch/bench.sh ${MODEL}"
