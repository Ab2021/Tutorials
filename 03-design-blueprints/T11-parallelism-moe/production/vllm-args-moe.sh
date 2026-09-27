#!/usr/bin/env bash
# T11 — WideEP MoE replica launch (REFERENCE-GRADE, NOT EXECUTED HERE)
#
# This script was NOT run in the environment that produced this blueprint: there is no
# GPU, no ROCm/CUDA runtime and no model checkpoint present. It is written to be
# correct-on-inspection and to encode the topology the HLD argues for. Every flag below
# traces to a relationship stated in the corpus [T] or to the derivation in the LLD [D].
#
# Shape it launches:  attention replicated (DP attention), experts split (EP),
#                     TP confined inside the node, PP held at 2.
#
# NOTE: flag names on a real vLLM/llm-d build move between releases. Treat the VALUES as
# the design decision and the SPELLING as something to re-check against your build.

set -euo pipefail

MODEL="${MODEL:-/models/<your-moe-checkpoint>}"

# --- parallelism --------------------------------------------------------------
# TP must not exceed the GPUs in one node: a TP collective rides the intra-node fabric
# and a cross-node TP group puts a latency-critical collective on the slow link. [T]
TP_SIZE="${TP_SIZE:-8}"

# Pipeline depth. Held at 2: the bubble is (PP-1)/(M+PP-1), so PP=2 at M=8 costs ~11%,
# while PP=8 costs ~47%. Deeper needs many more microbatches to hide. [T]
PP_SIZE="${PP_SIZE:-2}"

# Expert parallelism. experts_per_gpu = total_experts / ep.  [T]
# The value here is what sim/planner.py ranked for a 16-GPU replica on 192 GB parts.
EP_SIZE="${EP_SIZE:-16}"

# Data parallelism carries the replicated attention blocks and the request stream.
DP_SIZE="${DP_SIZE:-16}"

# --- the MoE path -------------------------------------------------------------
# Fused: top-k permute -> grouped GEMMs -> unpermute, with reduction and scaling folded
# in. The naive path is 2 all-to-all + 6 kernels; the fused path is 3. The all-to-all
# count is unchanged -- the win is launch and intermediate-materialisation overhead. [T]
ENABLE_FUSED_MOE=1
MOE_PERMUTE_FUSED=1
MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-1}"

# --- KV budget ----------------------------------------------------------------
# The planner's binding constraint is usually weight memory. Requesting a large KV cache
# without checking that the weights fit first is the classic OOM-at-startup. [D]
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

# --- all-to-all transport -----------------------------------------------------
# The all-to-all rides the fabric the wide dimension spans, which is inter-node here.
# Its bandwidth, not the GPU, is what degrades first under load. [D]
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export VLLM_ALL2ALL_BACKEND="${VLLM_ALL2ALL_BACKEND:-deepep_low_latency}"

# --- expert placement ---------------------------------------------------------
# Redundant experts let a hot expert be served locally instead of crossing the fabric.
# This is a memory-for-bandwidth trade and must be budgeted against KV. [D]
export VLLM_MOE_REDUNDANT_EXPERTS="${VLLM_MOE_REDUNDANT_EXPERTS:-0}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --pipeline-parallel-size "${PP_SIZE}" \
  --enable-expert-parallel \
  --expert-parallel-size "${EP_SIZE}" \
  --data-parallel-size "${DP_SIZE}" \
  --enable-expert-parallel-attention \
  --enable-fused-moe \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --disable-log-requests

# WHY DP ATTENTION: with experts split across the wide dimension, replicating the
# attention block means no collective is needed to serve a token's attention -- only the
# MoE layer communicates. That is the whole point of the WideEP split. [T]
#
# WHERE THIS FAILS: if the fabric degrades, every token still crosses it once per MoE
# layer, and the replica's latency tracks the fabric rather than the GPU. The HLD's
# degradation ladder responds by narrowing EP and accepting a tighter KV budget.
