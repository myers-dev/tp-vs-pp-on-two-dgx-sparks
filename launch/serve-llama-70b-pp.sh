#!/usr/bin/env bash
# launch/serve-llama-70b-pp.sh
# Serve nvidia/Llama-3.3-70B-Instruct-FP4 with Pipeline Parallelism (PP=2)
#
# Engine: vLLM 0.19.0 (nvcr.io/nvidia/vllm:26.04-py3)
# Parallelism: --pipeline-parallel-size 2 (one node per stage)
# Quant: NVFP4 (loaded as-is; no --quantization flag needed — model is
#         pre-quantized by NVIDIA)
#
# Result from the post: 355 tok/s output, 0.19 Gb/s inter-node, conc 128
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

echo "=== Serving ${MODEL} PP=2 on port ${SERVE_PORT} ==="

# VLLM_HOST_IP: verbatim from project log — prevents mis-resolution to
# docker0/loopback.  Set to HEAD IP for the head container.
docker run -d --name vllm-serve \
    "${DOCKER_FLAGS[@]}" \
    -e VLLM_HOST_IP="${HEAD_IP}" \
    "${VLLM_IMAGE}" \
    python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${MODEL}" \
        --pipeline-parallel-size 2 \
        --enforce-eager \
        --port "${SERVE_PORT}" \
        --host 0.0.0.0

echo ""
echo "Waiting for server to be ready..."
until curl -sf "http://localhost:${SERVE_PORT}/health" > /dev/null 2>&1; do
    sleep 5
done
echo "Server ready.  Run: bash launch/bench.sh ${MODEL}"
