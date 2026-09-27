"""Topology planner for MoE serving — the mechanism this blueprint exists to prove.

This module contains NO hardware calls and NO measured numbers. It models the memory and
communication relationships described in the corpus and lets you enumerate the tradeoff. Every
quantity it prints is a model output, not a benchmark.

Provenance: the relationships (experts_per_gpu = experts / ep, the fused kernel path, TP confined
to a node, the pipeline bubble) come from
`refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
and are marked [T]. The specific coefficients are derived here and marked [D].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

# --------------------------------------------------------------------------- model


@dataclass(frozen=True)
class ModelSpec:
    """Architecture parameters. 61 layers / 7168 hidden / 256 experts / top-k 8 is a
    representative large-MoE shape in the class the corpus discusses [T] (256 experts);
    the exact figures here are an example, not a claim about a named model. [D]"""

    name: str = "example-moe-256"
    layers: int = 61
    hidden: int = 7168
    experts: int = 256          # [T] total expert count, class of DeepSeek-V3
    top_k: int = 8              # [T] experts activated per token
    kv_heads: int = 8
    head_dim: int = 128
    dtype_bytes: int = 2        # fp16


@dataclass(frozen=True)
class Accelerator:
    name: str = "mi300-class"
    memory_gb: float = 192.0    # [T] MI300 192 GB / MI355 288 GB named in the corpus
    usable_fraction: float = 0.85   # runtime + fragmentation headroom [D]


@dataclass(frozen=True)
class Fabric:
    gpus_per_node: int = 8
    intra_node_bw_gbps: float = 400.0   # illustrative; TP collectives ride here [D]
    inter_node_bw_gbps: float = 50.0    # illustrative; EP all-to-all rides here [D]


@dataclass(frozen=True)
class Sharding:
    tp: int = 1
    pp: int = 1
    dp: int = 1
    ep: int = 1

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.dp

    def label(self) -> str:
        return f"TP{self.tp} PP{self.pp} DP{self.dp} EP{self.ep}"


@dataclass
class MemoryPlan:
    experts_per_gpu: int
    expert_weight_gb: float
    attention_weight_gb: float
    weights_per_gpu_gb: float
    kv_bytes_per_token: float
    kv_per_gpu_gb: float
    max_tokens_per_gpu: float
    fits: bool
    binding_constraint: str


@dataclass
class TopologySpec:
    sharding: Sharding
    memory: MemoryPlan
    bubble_fraction: float
    a2a_payload_bytes: float
    a2a_peers: int
    a2a_penalty: float
    score: float
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- arithmetic


def attention_weight_gb(m: ModelSpec) -> float:
    """Attention block, replicated on every DP rank. 4 x H^2 assumes Q,K,V,O. [D]"""
    return m.layers * 4 * m.hidden * m.hidden * m.dtype_bytes / 1e9


def kv_bytes_per_token(m: ModelSpec) -> float:
    """2 (K and V) x layers x kv_heads x head_dim x dtype. [D]"""
    return 2 * m.layers * m.kv_heads * m.head_dim * m.dtype_bytes


def plan_memory(m: ModelSpec, a: Accelerator, s: Sharding) -> MemoryPlan:
    """Pure arithmetic. No I/O, no device query."""
    experts_per_gpu = m.experts // s.ep if s.ep else m.experts
    # gate + up + down projections per expert; 3 x H^2 is the conventional gated MLP [D]
    expert_gb = experts_per_gpu * 3 * m.hidden * m.hidden * m.dtype_bytes / 1e9
    # Pipeline stages split the layers, so a PP replica holds 1/pp of the expert weights.
    expert_gb /= s.pp
    attn_gb = attention_weight_gb(m) / s.tp
    weights_gb = expert_gb + attn_gb

    budget_gb = a.memory_gb * a.usable_fraction
    kv_gb = budget_gb - weights_gb
    kvt = kv_bytes_per_token(m)
    max_tokens = (kv_gb * 1e9 / kvt) if kv_gb > 0 else 0.0

    if kv_gb <= 0:
        binding = "weights"
    elif kv_gb < weights_gb:
        binding = "kv"
    else:
        binding = "weights"
    return MemoryPlan(
        experts_per_gpu=experts_per_gpu,
        expert_weight_gb=expert_gb,
        attention_weight_gb=attn_gb,
        weights_per_gpu_gb=weights_gb,
        kv_bytes_per_token=kvt,
        kv_per_gpu_gb=max(kv_gb, 0.0),
        max_tokens_per_gpu=max_tokens,
        fits=kv_gb > 0,
        binding_constraint=binding,
    )


def bubble_fraction(pp: int, microbatches: int) -> float:
    """(P-1)/(M+P-1). [D] At PP=2, M=8 -> 1/9 ~= 11%; at PP=8 -> 7/15 ~= 47%."""
    if pp <= 1:
        return 0.0
    return (pp - 1) / (microbatches + pp - 1)


def a2a_payload_bytes(m: ModelSpec) -> float:
    """Tokens dispatched to top_k experts and returned. Independent of EP degree --
    a token goes to top_k experts wherever they live. [D]"""
    return m.top_k * m.hidden * m.dtype_bytes * 2


def a2a_penalty(m: ModelSpec, f: Fabric, s: Sharding, tokens_in_flight: int) -> float:
    """What *does* grow with EP is the peer count and therefore the per-message overhead.

    payload is constant; overhead scales with (ep - 1) peers and the inter/intra bandwidth
    ratio. Modelled, not measured. [D]
    """
    peers = max(s.ep - 1, 0)
    if peers == 0:
        return 0.0
    payload = a2a_payload_bytes(m) * tokens_in_flight
    # a node-local peer exchanges over the intra-node fabric; every peer beyond that is
    # assumed to pay the inter-node rate. Rough, and stated as such.
    local = min(peers, f.gpus_per_node - 1)
    remote = max(peers - local, 0)
    seconds = (payload * local / (f.intra_node_bw_gbps * 1e9)
               + payload * remote / (f.inter_node_bw_gbps * 1e9))
    return seconds


# --------------------------------------------------------------------------- enumeration


def enumerate_shardings(m: ModelSpec, f: Fabric, world_size: int) -> Iterator[Sharding]:
    """Yield every structurally valid decomposition of `world_size`.

    Hard constraints enforced here, so a caller can never rank an illegal shape:
      * tp <= gpus_per_node          -- TP stays INSIDE the node [T]
      * tp divides world_size        -- and the node, so a TP group is node-local
      * ep divides experts exactly   -- otherwise experts are unowned
      * world_size == tp*pp*dp       -- a mismatch hangs a collective
    """
    for tp in range(1, min(f.gpus_per_node, world_size) + 1):
        if world_size % tp:
            continue
        rest = world_size // tp
        for pp in range(1, rest + 1):
            if rest % pp:
                continue
            dp = rest // pp
            for ep in range(1, m.experts + 1):
                if m.experts % ep:
                    continue
                # EP ranks are drawn from the DP x TP pool: an EP group must fit on the
                # devices that hold experts, which is dp*tp here.
                if ep > dp * tp:
                    continue
                yield Sharding(tp=tp, pp=pp, dp=dp, ep=ep)


def score(spec: TopologySpec, microbatches: int) -> float:
    """Rank candidates. A MODEL of the tradeoff, not a throughput measurement. [D]"""
    if not spec.memory.fits:
        raise ValueError("score() called on an infeasible candidate")
    concurrency_proxy = spec.memory.max_tokens_per_gpu
    efficiency = (1.0 - spec.bubble_fraction)
    comm = 1.0 / (1.0 + spec.a2a_penalty * 1000.0)   # scale-free damping of the seconds term
    return concurrency_proxy * efficiency * comm


def plan(m: ModelSpec, a: Accelerator, f: Fabric, world_size: int,
         microbatches: int = 8, tokens_in_flight: int = 4096) -> list[TopologySpec]:
    """Enumerate, filter to feasible, score, rank descending."""
    out: list[TopologySpec] = []
    for s in enumerate_shardings(m, f, world_size):
        mem = plan_memory(m, a, s)
        if not mem.fits:
            continue
        cand = TopologySpec(
            sharding=s,
            memory=mem,
            bubble_fraction=bubble_fraction(s.pp, microbatches),
            a2a_payload_bytes=a2a_payload_bytes(m),
            a2a_peers=max(s.ep - 1, 0),
            a2a_penalty=a2a_penalty(m, f, s, tokens_in_flight),
            score=0.0,
        )
        cand.score = score(cand, microbatches)
        cand.notes.append(f"experts/gpu={mem.experts_per_gpu}")
        cand.notes.append(f"binding={mem.binding_constraint}")
        out.append(cand)
    out.sort(key=lambda c: c.score, reverse=True)
    return out


# --------------------------------------------------------------------------- validation


def niah_recall(concurrency: int, ctx_tokens: int, ceiling_tokens: float,
                needles: int = 10) -> float:
    """A MODEL of long-context degradation as KV pressure rises -- not a measurement.

    Two effects, both described in the corpus [T] and encoded here [D]:
      1. recall falls gracefully as concurrency x context approaches the KV ceiling;
      2. beyond the ceiling, KV recomputation kicks in and recall collapses -- the
         corpus documents a cliff at 28k inputs / concurrency 256.

    Both context length and concurrency drive the pressure, which is why all three NIAH
    shapes must be swept rather than one.
    """
    if ceiling_tokens <= 0:
        return 0.0
    pressure = (concurrency * ctx_tokens) / ceiling_tokens
    if pressure <= 1.0:
        # slow graceful decay: 100% at zero pressure, ~85% at the ceiling
        return needles * (1.0 - 0.15 * pressure)
    # past the ceiling: recomputation, and recall falls off a shelf
    overshoot = pressure - 1.0
    return max(0.0, needles * (0.85 - 3.0 * overshoot))


def fits_without_ep(m: ModelSpec, a: Accelerator, f: Fabric, world_size: int,
                    microbatches: int = 8) -> bool:
    """Can this model be served on this accelerator with EP=1?

    This is the question that decides whether Expert Parallelism is a *choice* or a
    *necessity* -- and it is the reason WideEP exists at all. [D]
    """
    for s in enumerate_shardings(m, f, world_size):
        if s.ep == 1 and plan_memory(m, a, s).fits:
            return True
    return False


def passes(recall: float, threshold: int = 7) -> bool:
    """The corpus's harness threshold: >= 7 of 10 needles. [T]"""
    return recall >= threshold
