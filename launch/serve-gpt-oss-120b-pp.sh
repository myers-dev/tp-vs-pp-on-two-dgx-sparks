#!/usr/bin/env bash
# launch/serve-gpt-oss-120b-pp.sh
# Serve openai/gpt-oss-120b with Pipeline Parallelism (PP=2)
#
# Engine: vLLM 0.19.0 (nvcr.io/nvidia/vllm:26.04-py3)
# Parallelism: --pipeline-parallel-size 2
# Arch: MoE (~120B total / ~5.1B active)
# Quant: MXFP4
#   TEMPLATED note: vLLM 0.19.0 auto-detects MXFP4 from the model config.
#   No --quantization flag was needed in the experiment (vLLM reads
#   quantization_config from config.json).
#
# Result from the post: 222 tok/s total (short-output caveat below),
#   0.205 Gb/s inter-node, conc 128
#
# SHORT-OUTPUT CAVEAT (verbatim from project log):
#   gpt-oss emits early EOS on random prompts (~130 tok avg vs 1024
#   requested).  The bench finishes in ~10 min vs ~50-90 for other models.
#   Total tok/s is PREFILL-WEIGHTED and not directly comparable to the other
#   models.  The PP vs TP ratio within gpt-oss is still valid.
#
# Prerequisites:
#   1. source launch/env.sh
#   2. Ray cluster up (launch/ray-start.sh)
#
# Run on HEAD node.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

MODEL="openai/gpt-oss-120b"
MODEL_PATH="${MODELS_DIR}/gpt-oss-120b"

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
echo "Waiting for server to be ready (~120-240 s for 120B)..."
until curl -sf "http://localhost:${SERVE_PORT}/health" > /dev/null 2>&1; do
    sleep 10
done
echo "Server ready.  Run: bash launch/bench.sh ${MODEL}"
