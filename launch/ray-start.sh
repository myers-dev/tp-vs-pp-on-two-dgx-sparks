#!/usr/bin/env bash
# launch/ray-start.sh — start a two-node Ray cluster inside vLLM containers
#
# Run this before any per-model serve script.  Run ray-stop.sh between models.
#
# Pattern verbatim from the project log: Ray is started inside the same
# nvcr.io/nvidia/vllm:26.04-py3 container as vLLM; --enforce-eager is passed
# to the vllm serve command (not to Ray itself).  Node IPs are pinned via
# VLLM_HOST_IP to prevent mis-resolution to docker0/loopback.
#
# Usage (run on HEAD node):
#   source launch/env.sh
#   bash launch/ray-start.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "=== Starting Ray HEAD on ${HEAD_IP} ==="
# TEMPLATED — adjust --num-cpus / --num-gpus if your GB10 reports differently.
docker run -d --name ray-head \
    "${DOCKER_FLAGS[@]}" \
    -e VLLM_HOST_IP="${HEAD_IP}" \
    "${VLLM_IMAGE}" \
    ray start --head \
        --node-ip-address="${HEAD_IP}" \
        --port="${RAY_PORT}" \
        --num-cpus=128 \
        --num-gpus=1 \
        --block

echo ""
echo "=== Starting Ray WORKER on ${WORKER_IP} (run on worker node, or via ssh) ==="
echo "    ssh ${WORKER_IP} docker run --rm --network host --ipc host \\"
echo "        --shm-size=10g --device /dev/infiniband --ulimit memlock=-1 \\"
echo "        -v ${MODELS_DIR}:${MODELS_DIR} \\"
echo "        -e NCCL_IB_HCA=${NCCL_IB_HCA} \\"
echo "        -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE} \\"
echo "        -e NCCL_IGNORE_CPU_AFFINITY=${NCCL_IGNORE_CPU_AFFINITY} \\"
echo "        -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \\"
echo "        -e UCX_NET_DEVICES=${UCX_NET_DEVICES} \\"
echo "        -e GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} \\"
echo "        -e TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME} \\"
echo "        -e RAY_memory_monitor_refresh_ms=${RAY_memory_monitor_refresh_ms} \\"
echo "        -e VLLM_HOST_IP=${WORKER_IP} \\"
echo "        ${VLLM_IMAGE} \\"
echo "        ray start --address=${HEAD_IP}:${RAY_PORT} \\"
echo "            --node-ip-address=${WORKER_IP} \\"
echo "            --num-cpus=128 --num-gpus=1 --block &"
echo ""
echo "After Ray is up on both nodes, run a per-model serve script."
