"""T07 -- KV cache: allocation, sharing, prefix caching, tiering.

The KV cache is the resource that limits concurrency in every real deployment. This package
models the four mechanisms that decide how much of it you get to use:

  allocator.py     block allocation -- paged vs contiguous, waste arithmetic, refcounted pool
  prefix_cache.py  content-addressed reuse, radix-chained keys, sibling sharing, copy-on-write
  tiering.py       HBM -> DRAM -> SSD, the transfer-cost model, retention policy
  kvmath.py        bytes per token, the concurrency ceiling, quantization effects
  experiments.py   seven questions an operator actually asks

Provenance is inline in each module: [T] corpus, [R] supporting repo, [D] derived.
"""
from .allocator import (BLOCK_SIZE, BlockPool, BlockTable, contiguous_waste, paged_waste,
                        expected_waste, make_length_distribution, block_size_table)
from .prefix_cache import (PrefixCache, block_hash, fork, copy_on_write, sharing_saving)
from .tiering import (Medium, POOLED_MEMORY, RDMA, TCP, NVME_LOCAL, offload_cost,
                      recompute_cost, breakeven, preemption_choice, RetentionPolicy)
from .kvmath import (KvSpec, MODELS, kv_table, max_concurrency, kv_quant_effect,
                     offload_headroom, recompute_cliff)
from . import experiments

__all__ = [
    "BLOCK_SIZE", "BlockPool", "BlockTable", "contiguous_waste", "paged_waste",
    "expected_waste", "make_length_distribution", "block_size_table",
    "PrefixCache", "block_hash", "fork", "copy_on_write", "sharing_saving",
    "Medium", "POOLED_MEMORY", "RDMA", "TCP", "NVME_LOCAL", "offload_cost",
    "recompute_cost", "breakeven", "preemption_choice", "RetentionPolicy",
    "KvSpec", "MODELS", "kv_table", "max_concurrency", "kv_quant_effect",
    "offload_headroom", "recompute_cliff", "experiments",
]
