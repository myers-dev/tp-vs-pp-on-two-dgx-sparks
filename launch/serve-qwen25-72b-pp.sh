#!/usr/bin/env bash
# launch/serve-qwen25-72b-pp.sh
# Serve Qwen/Qwen2.5-72B-Instruct-AWQ with Pipeline Parallelism (PP=2)
#
# Engine: vLLM 0.19.0 (nvcr.io/nvidia/vllm:26.04-py3)
# Parallelism: --pipeline-parallel-size 2
# Quant: AWQ-Int4
#   --quantization awq_marlin  (verbatim flag for vLLM AWQ serving)
#   TEMPLATED note: vLLM 0.19.0 uses "awq_marlin" for fused AWQ kernels.
#   If your vLLM version requires just "awq", adjust accordingly.
#
# Result from the post: 181 tok/s output, 0.098 Gb/s inter-node, conc 128
#
# Note: AWQ activations remain bf16 regardless of weight quantization
# (weight-only quant by design; Lin et al., 2023).  PP traffic follows
# the same h×throughput law as dense NVFP4 runs — confirmed by cross-check
# in the project log: 0.19×(181/355)=0.097 vs measured 0.098.
#
# Prerequisites:
#   1. source launch/env.sh
#   2. Ray cluster up (launch/ray-start.sh)
#
# Run on HEAD node.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

MODEL="Qwen/Qwen2.5-72B-Instruct-AWQ"
MODEL_PATH="${MODELS_DIR}/Qwen2.5-72B-Instruct-AWQ"

echo "=== Serving ${MODEL} PP=2 on port ${SERVE_PORT} ==="

docker run -d --name vllm-serve \
    "${DOCKER_FLAGS[@]}" \
    -e VLLM_HOST_IP="${HEAD_IP}" \
    "${VLLM_IMAGE}" \
    python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${MODEL}" \
        --pipeline-parallel-size 2 \
        --quantization awq_marlin \
        --enforce-eager \
        --port "${SERVE_PORT}" \
        --host 0.0.0.0

echo ""
echo "Waiting for server to be ready (~120-180 s)..."
until curl -sf "http://localhost:${SERVE_PORT}/health" > /dev/null 2>&1; do
    sleep 5
done
echo "Server ready.  Run: bash launch/bench.sh ${MODEL}"
