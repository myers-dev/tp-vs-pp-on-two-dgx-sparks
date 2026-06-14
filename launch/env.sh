#!/usr/bin/env bash
# launch/env.sh — shared serving environment for all TP/PP runs
#
# Source this file before calling any per-model launch script, or let each
# launch script source it.  All values below are verbatim from the Notion
# methodology notes (NCCL/RoCE env, container image, port) unless marked
# TEMPLATED.
#
# Cluster nodes used in the post:
#   HEAD   gx10-9c8c   10.88.1.120   (rank 0 / Ray head)
#   WORKER gx10-f0ff   10.88.1.119   (rank 1 / Ray worker)
#                        ^-- .121 (gx10-f0ff) was also used mid-experiment
#
# Usage:
#   source launch/env.sh
#   # then run one of the per-model scripts

# ── container ──────────────────────────────────────────────────────────────
# Verbatim from the post and project log.
export VLLM_IMAGE="nvcr.io/nvidia/vllm:26.04-py3"   # vLLM 0.19.0

# ── model paths on NFS ─────────────────────────────────────────────────────
# TEMPLATED — adjust to match your NFS mount point.
export MODELS_DIR="/mnt/models"

# ── per-node IPs ───────────────────────────────────────────────────────────
# TEMPLATED — set to your actual node IPs.
export HEAD_IP="10.88.1.120"
export WORKER_IP="10.88.1.119"
export RAY_PORT="6379"
export SERVE_PORT="8000"

# ── NCCL / RoCE environment (verbatim from Notion methodology notes) ────────
# Applied to BOTH nodes' containers.
#
# NCCL_IB_HCA: names the exact RDMA (RoCE) NICs NCCL uses for cross-node GPU
#   communication.  Without it NCCL may not select the RDMA fabric, causing
#   slow or hung collectives.
export NCCL_IB_HCA="rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1"

# NCCL_IB_DISABLE=0: explicitly enables NCCL's RDMA transport (vs TCP
#   fallback); this is the 200 Gb/s path.
export NCCL_IB_DISABLE="0"

# NCCL_IGNORE_CPU_AFFINITY=1: stops NCCL from pinning threads to CPU cores
#   based on GB10's mis-reported topology, which can stall communication.
export NCCL_IGNORE_CPU_AFFINITY="1"

# NCCL_SOCKET_IFNAME: the interface NCCL uses for its bootstrap/control
#   handshake (not bulk data); must be routable between nodes.
export NCCL_SOCKET_IFNAME="enP2p1s0f1np1"

# UCX_NET_DEVICES / GLOO_SOCKET_IFNAME / TP_SOCKET_IFNAME: interface pinning
#   for UCX, PyTorch Gloo backend, and vLLM TP coordination sockets.
export UCX_NET_DEVICES="enP2p1s0f1np1"
export GLOO_SOCKET_IFNAME="enP2p1s0f1np1"
export TP_SOCKET_IFNAME="enP2p1s0f1np1"

# RAY_memory_monitor_refresh_ms=0: keeps Ray from OOM-killing workers under
#   memory pressure on GB10 unified memory.
export RAY_memory_monitor_refresh_ms="0"

# ── docker run flags (verbatim from project log; used on both nodes) ────────
# --device /dev/infiniband : expose the RoCE NIC to the container
# --ulimit memlock=-1      : let NCCL pin memory for RDMA
#                            (without this: ibv_reg_mr "Cannot allocate memory")
# --network host           : required for Ray multi-node discovery
# --ipc host               : shared memory for GPU peer access
# --shm-size=10g           : TEMPLATED — tune to your system
DOCKER_FLAGS=(
    --rm
    --network host
    --ipc host
    --shm-size=10g
    --device /dev/infiniband
    --ulimit memlock=-1
    -v "${MODELS_DIR}:${MODELS_DIR}"
    -e NCCL_IB_HCA="${NCCL_IB_HCA}"
    -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE}"
    -e NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY}"
    -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
    -e UCX_NET_DEVICES="${UCX_NET_DEVICES}"
    -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}"
    -e TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME}"
    -e RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms}"
)
export DOCKER_FLAGS
