"""Latency decomposition -- every span must map to a term of the end-to-end equation.

The corpus's worked example [T]:

    "Here is one real request drawn as a stack of spans. The total was 1.2 seconds. Retrieval took
     90 milliseconds. Generation took over a second. The bottleneck is obvious the moment you can
     see it."                                          -- LLM Observability: Traces, Spans, OTel

A trace is only useful if it decomposes rather than just measures. This module builds a span tree,
computes the exact end-to-end total from its leaves, and then asks the two questions an operator
actually has:

    "where did the time go?"        -> by_name(): which span NAME owns the most total time
    "what should I optimize?"       -> attribution(): which span has the headroom

**The mistake this module exists to prevent** is ranking spans by COUNT instead of by TOTAL. In an
agentic workload there are many tool calls and few retrievals, so the tool span wins on count and
loses on time. Optimizing the wrong one is invisible in a metrics-only dashboard, because a metric
has no name attached -- which is the entire argument for traces.

Provenance: [T] transcript, [R] supporting repo, [D] derived.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Span:
    """One step of a request. Children are nested, and the parent's duration is the wall clock."""

    name: str
    ms: float
    kind: str = "internal"          # internal | retrieval | llm | tool | guardrail | queue
    children: list["Span"] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def add(self, *children: "Span") -> "Span":
        self.children.extend(children)
        return self

    @property
    def self_ms(self) -> float:
        """Time spent in this span itself, not in its children."""
        return max(0.0, self.ms - sum(c.ms for c in self.children))


def walk(span: Span, depth: int = 0, out: list | None = None) -> list[tuple[int, Span]]:
    """Depth-first flatten, parents before children."""
    out = [] if out is None else out
    out.append((depth, span))
    for c in span.children:
        walk(c, depth + 1, out)
    return out


def end_to_end_ms(root: Span) -> float:
    """The wall-clock total. Equal to the root's duration by construction -- asserted below."""
    return root.ms


def by_name(root: Span) -> dict[str, dict]:
    """Total milliseconds per span NAME, with count and share of the end-to-end time.

    Aggregating by name rather than by span id is what makes a trace summarisable: the same
    `retrieval` step appears once, but `tool_call` appears many times, and only the aggregate is
    comparable against the total.
    """
    total = end_to_end_ms(root)
    acc: dict[str, dict] = {}
    for _, s in walk(root):
        a = acc.setdefault(s.name, {"name": s.name, "kind": s.kind, "count": 0, "total_ms": 0.0,
                                    "self_ms": 0.0})
        a["count"] += 1
        a["total_ms"] += s.ms
        a["self_ms"] += s.self_ms
    for a in acc.values():
        a["pct_of_e2e"] = a["total_ms"] / total * 100.0 if total else 0.0
    return acc


def leaves(root: Span) -> list[Span]:
    """Leaf spans only -- the level at which the decomposition actually sums to the total."""
    return [s for _, s in walk(root) if not s.children]


def decomposition_check(root: Span) -> dict:
    """Does the tree add up? A trace whose spans do not sum to the total is a broken instrument.

    This is a real failure mode: spans timed independently across processes accumulate clock skew,
    and the leaf sum drifts from the root. Reporting the residual is how you notice.
    """
    leaf_sum = sum(s.ms for s in leaves(root))
    residual = leaf_sum - root.ms
    return {"leaf_sum_ms": leaf_sum, "root_ms": root.ms, "residual_ms": residual,
            "residual_pct": residual / root.ms * 100.0 if root.ms else 0.0,
            "consistent": abs(residual / root.ms) < 0.02 if root.ms else True}


def attribution(root: Span, target_ms: float = 300.0) -> dict:
    """Where a latency budget should go, ranked by SELF time rather than by span count.

    Two traps are avoided here, and both produce a confidently wrong answer:

      * **The root span is excluded.** It owns 100% of the wall clock by construction, so including
        it makes "the top span by total" trivially the request itself. An attribution table that
        says "your request is slow" is not an attribution table.
      * **Ranking is by `self_ms`, not `total_ms`.** A parent's total already contains its children's,
        so ranking by total double-counts every nested span: `llm_turn` and its own `decode` child
        both appear, and the parent looks like a second, separate cost. Self time is the time a span
        spends ITSELF, and self times across the tree sum to the end-to-end total exactly.

    Returns both rankings side by side, because the disagreement between them is the finding, and
    also `count_discriminates` -- in a chat turn every span occurs once, so a count ranking is
    arbitrary and should not be presented as if it meant something.
    """
    names = [a for k, a in by_name(root).items() if k != root.name]
    by_self = sorted(names, key=lambda a: a["self_ms"], reverse=True)
    by_count = sorted(names, key=lambda a: a["count"], reverse=True)
    top = by_self[0] if by_self else None
    count_top = by_count[0] if by_count else None
    discriminating = bool(count_top and count_top["count"] > 1)
    return {
        "by_self": by_self,
        "by_count": by_count,
        "rank_by_self": [a["name"] for a in by_self],
        "rank_by_count": [a["name"] for a in by_count],
        "count_discriminates": discriminating,
        "disagree": bool(discriminating and top and count_top and top["name"] != count_top["name"]),
        "top_by_self": top["name"] if top else None,
        "top_by_count": count_top["name"] if count_top else None,
        "top_self_ms": top["self_ms"] if top else 0.0,
        "top_share_pct": (top["self_ms"] / end_to_end_ms(root) * 100.0) if top else 0.0,
        # if you removed the top span entirely, would you hit the target?
        "max_possible_saving_ms": top["self_ms"] if top else 0.0,
        "target_reachable": bool(top and top["self_ms"] >= target_ms),
    }


# ----------------------------------------------------------------------------------------------
# Two request shapes. Both are modelled from the corpus's description, not measured.
# ----------------------------------------------------------------------------------------------

def chat_request(retrieval_ms: float = 90.0, tool_ms: float = 0.0, n_tools: int = 0,
                 queue_ms: float = 40.0, ttft_ms: float = 120.0, output_tokens: int = 180,
                 itl_ms: float = 4.80, retries: int = 0) -> Span:
    """A RAG chat turn: the corpus's 1.2 s example [T], with every term named.

    end_to_end = queue + retrieval + prompt_assembly + guardrail_in + (TTFT + N x ITL)
                 + guardrail_out + serialize + retries
    """
    gen_ms = ttft_ms + output_tokens * itl_ms
    root = Span("request", 0.0, kind="internal")
    root.add(
        Span("queue_wait", queue_ms, kind="queue"),
        Span("prompt_assembly", 15.0),
        Span("guardrail_in", 20.0, kind="guardrail", meta={"position": "input"}),
        Span("retrieval", retrieval_ms, kind="retrieval",
             meta={"doc_ids": ["policy-v2", "policy-v1"], "top_k": 8}),
    )
    gen = Span("generate", gen_ms, kind="llm",
               meta={"model": "8b-instruct", "input_tokens": 4200, "output_tokens": output_tokens})
    gen.add(Span("ttft", ttft_ms, kind="llm"), Span("decode", output_tokens * itl_ms, kind="llm"))
    root.add(gen)
    if n_tools:
        for i in range(n_tools):
            root.add(Span("tool_call", tool_ms, kind="tool", meta={"tool": f"tool_{i}"}))
    root.add(Span("guardrail_out", 30.0, kind="guardrail", meta={"position": "output"}))
    root.add(Span("serialize", 20.0))
    for i in range(retries):
        root.add(Span("retry", 200.0, kind="tool", meta={"attempt": i + 1}))
    root.ms = sum(c.ms for c in root.children)
    return root


def agentic_request(n_tools: int = 12, tool_ms: float = 8.0, llm_turns: int = 5,
                    llm_turn_ms: float = 380.0, retrieval_ms: float = 60.0) -> Span:
    """A long-horizon agent: MANY small spans and FEW large ones.

    This is the shape where ranking by count and ranking by self time disagree, which is the point of
    the experiment. The corpus notes agentic traffic has become the majority of inference load
    ("about 70% of the inference traffic these days is agentic" [T], llm-d), so this is the shape
    that matters in production, not the chat turn.

    `n_tools` is exact: the remainder is distributed over the first turns rather than truncated, so
    the caller can rely on the span count it asked for.
    """
    root = Span("request", 0.0, kind="internal")
    root.add(Span("queue_wait", 25.0, kind="queue"), Span("retrieval", retrieval_ms, kind="retrieval"))
    base, extra = divmod(n_tools, llm_turns)
    for i in range(llm_turns):
        t = Span("llm_turn", llm_turn_ms, kind="llm", meta={"turn": i + 1})
        t.add(Span("ttft", 90.0, kind="llm"), Span("decode", llm_turn_ms - 90.0, kind="llm"))
        root.add(t)
        for j in range(base + (1 if i < extra else 0)):
            root.add(Span("tool_call", tool_ms, kind="tool", meta={"tool": f"t{i}_{j}"}))
    root.add(Span("guardrail_out", 35.0, kind="guardrail"))
    root.ms = sum(c.ms for c in root.children)
    return root


def render_tree(root: Span, depth: int = 0) -> list[str]:
    """A text trace, the way it appears in a UI."""
    lines = []
    for d, s in walk(root):
        pad = "  " * d
        pct = s.ms / root.ms * 100.0 if root.ms else 0.0
        lines.append(f"{pad}{s.name:<18} {s.ms:8.1f} ms  {pct:5.1f}%")
    return lines
