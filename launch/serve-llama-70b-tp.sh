#!/usr/bin/env bash
# launch/serve-llama-70b-tp.sh
# Serve nvidia/Llama-3.3-70B-Instruct-FP4 with Tensor Parallelism (TP=2)
#
# Engine: vLLM 0.19.0 (nvcr.io/nvidia/vllm:26.04-py3)
# Parallelism: --tensor-parallel-size 2 (each layer sharded across both nodes)
# Quant: NVFP4 (pre-quantized; no --quantization flag)
#
# Result from the post: 404 tok/s output, 37.6 Gb/s inter-node, conc 128
#
# NOTE: multi-node TP deadlocks during CUDA-graph capture on GB10.
# --enforce-eager is REQUIRED (verbatim from project log — skips capture).
# TRT-LLM was dropped for this reason; vLLM + Ray + enforce-eager works.
#
# Prerequisites:
#   1. source launch/env.sh
#   2. Ray cluster up (launch/ray-start.sh)
#
# Run on HEAD node.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

MODEL="nvidia/Llama-3.3-70B-Instruct-FP4"
MODEL_PATH="${MODELS_DIR}/Llama-3.3-70B-Instruct-FP4"

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
echo "Waiting for server to be ready..."
until curl -sf "http://localhost:${SERVE_PORT}/health" > /dev/null 2>&1; do
    sleep 5
done
echo "Server ready.  Run: bash launch/bench.sh ${MODEL}"
