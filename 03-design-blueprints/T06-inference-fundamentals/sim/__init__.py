"""T06 capacity and latency model: closed-form, no simulation, no RNG."""
from .roofline import (
    ModelSpec, HardwareSpec, TrafficSpec, LLAMA_31, H100_CLASS,
    flops_per_token, bytes_moved, intensity, classify, prefill_crossover,
    attention_mlp_ratio,
)
from .kv import (
    kv_bytes_per_token, kv_total, weights_bytes, weight_replicas_needed,
    free_after_weights, max_concurrency, gqa_saving, kv_dtype_effect,
)
from .latency import (
    ttft, decode_throughput, itl, queue_wait, e2e, required_itl,
    required_output_len, goodput, throughput_proxy,
)

__all__ = [
    "ModelSpec", "HardwareSpec", "TrafficSpec", "LLAMA_31", "H100_CLASS",
    "flops_per_token", "bytes_moved", "intensity", "classify", "prefill_crossover",
    "attention_mlp_ratio",
    "kv_bytes_per_token", "kv_total", "weights_bytes", "weight_replicas_needed",
    "free_after_weights", "max_concurrency", "gqa_saving", "kv_dtype_effect",
    "ttft", "decode_throughput", "itl", "queue_wait", "e2e", "required_itl",
    "required_output_len", "goodput", "throughput_proxy",
]
