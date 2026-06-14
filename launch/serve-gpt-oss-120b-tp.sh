#!/usr/bin/env bash
# launch/serve-gpt-oss-120b-tp.sh
# Serve openai/gpt-oss-120b with Tensor Parallelism (TP=2)
#
# Engine: vLLM 0.19.0 (nvcr.io/nvidia/vllm:26.04-py3)
# Parallelism: --tensor-parallel-size 2
# Arch: MoE (~120B total / ~5.1B active)
# Quant: MXFP4 (auto-detected from config.json)
#
# Result from the post: 2641 tok/s total, 22.6 Gb/s inter-node, conc 48
#   (also prefill-weighted — see caveat below)
#
# CONCURRENCY CAVEAT (verbatim from project log):
#   conc 128 caused head-node saturation (all-reduce churn killed sshd).
#   Successful run used --max-concurrency 48 with bench client on WORKER.
#   Run bench.sh from WORKER node (10.88.1.119).
#
# --ignore-eos NOTE (verbatim from project log):
#   --ignore-eos was passed to the bench but did NOT force full 1024-token
#   outputs; gpt-oss emits its own end-of-turn (~55 tok/req) that ignore_eos
#   does not suppress.  bench.sh includes --ignore-eos to match the
#   experiment; remove it if you want to compare without it.
#
# Prerequisites:
#   1. source launch/env.sh
#   2. Ray cluster up (launch/ray-start.sh)
#   3. Run bench.sh from WORKER (10.88.1.119), not from HEAD.
#
# Run on HEAD node.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

MODEL="openai/gpt-oss-120b"
MODEL_PATH="${MODELS_DIR}/gpt-oss-120b"

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
        --enforce-eager \
        --port "${SERVE_PORT}" \
        --host 0.0.0.0

echo ""
echo "Waiting for server to be ready (~120-240 s for 120B)..."
until curl -sf "http://localhost:${SERVE_PORT}/health" > /dev/null 2>&1; do
    sleep 10
done
echo "Server ready."
echo "Run from WORKER: bash launch/bench.sh ${MODEL} 48 --ignore-eos"
