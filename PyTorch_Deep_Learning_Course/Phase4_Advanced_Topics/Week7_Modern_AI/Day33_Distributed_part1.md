# Day 33: Distributed Training - Deep Dive

> **Phase**: 4 - Advanced Topics
> **Week**: 7 - Modern AI
> **Topic**: NCCL, Ring All-Reduce, and ZeRO Stages

## 1. Communication Primitives (NCCL)

NVIDIA Collective Communications Library.
Optimized for NVLink/InfiniBand.
*   **Broadcast**: One sends to all.
*   **Scatter**: One sends chunks to all.
*   **Gather**: All send chunks to one.
*   **All-Gather**: All send chunks to all (Everyone gets everything).
*   **Reduce**: All send to one, aggregating (Sum).
*   **All-Reduce**: All send to all, aggregating (Sum). Crucial for DDP.

## 2. Ring All-Reduce

Naive All-Reduce is $O(N^2)$ bandwidth.
**Ring All-Reduce**:
*   Organize GPUs in a logical ring.
*   Pass chunks to neighbor.
*   Bandwidth independent of number of GPUs. $O(N)$.
*   Optimal for large clusters.

## 3. ZeRO (Zero Redundancy Optimizer)

DeepSpeed / FSDP concepts.
*   **Stage 1**: Shard Optimizer States (Adam moments). 4x memory saving.
*   **Stage 2**: Shard Gradients. 2x memory saving.
*   **Stage 3**: Shard Parameters. Linear memory saving ($1/N$).
*   **Offload**: Move states to CPU/NVMe.

## 4. Gradient Accumulation vs Distributed

*   **Grad Accumulation**: Saves memory on Single GPU. Increases Time.
*   **Distributed**: Saves Time (DDP) or Memory (FSDP) using Multiple GPUs.

## 5. Effective Batch Size

$$ \text{Effective BS} = \text{Per\_GPU\_BS} \times \text{Num\_GPUs} \times \text{Accum\_Steps} $$
*   Large batch size requires **Linear Scaling Rule** for Learning Rate.
*   $LR_{new} = LR_{base} \times k$.
*   Or LARS/LAMB optimizers for stability.
