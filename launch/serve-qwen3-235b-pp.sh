#!/usr/bin/env bash
# launch/serve-qwen3-235b-pp.sh
# Serve Qwen/Qwen3-235B-A22B with Pipeline Parallelism (PP=2)
#
# Engine: vLLM 0.19.0 (nvcr.io/nvidia/vllm:26.04-py3)
# Parallelism: --pipeline-parallel-size 2
# Arch: MoE (235B total / ~22B active per token)
# Quant: NVFP4 — TEMPLATED note: the NFS path used in the experiment was
#   nvidia/Qwen3-235B-A22B-FP4 (the NVIDIA-quantized FP4 variant on NGC/HF).
#   Adjust MODEL_PATH to match your local copy.
#
# Result from the post: 224 tok/s output, 0.060 Gb/s inter-node, conc 128
#
# MoE PP finding: each PP stage holds entire layers (all experts) on one node;
# no cross-node expert routing.  PP traffic for MoE is the same h×throughput
# formula as dense — MoE adds nothing to PP network cost.
#
# Prerequisites:
#   1. source launch/env.sh
#   2. Ray cluster up (launch/ray-start.sh)
#
# Run on HEAD node.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

# TEMPLATED — model ID and local path; adjust quant suffix if using HF hub name
MODEL="Qwen/Qwen3-235B-A22B"
MODEL_PATH="${MODELS_DIR}/Qwen3-235B-A22B-FP4"

echo "=== Serving ${MODEL} PP=2 on port ${SERVE_PORT} ==="

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
echo "Waiting for server to be ready (~120-240 s for 235B)..."
until curl -sf "http://localhost:${SERVE_PORT}/health" > /dev/null 2>&1; do
    sleep 10
done
echo "Server ready.  Run: bash launch/bench.sh ${MODEL}"
