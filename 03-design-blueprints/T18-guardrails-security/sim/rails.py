"""T18 -- rails: why four rails are not 1-(1-r)^4.

The corpus's claim [T] is *"No single rail is enough. It is the layering, the redundancy, that makes
the system genuinely hard to break."* That is true. The arithmetic people use to express it is not.

A rail's catch rate is not a scalar. It is conditional on two things the scalar model throws away:

  * **TECHNIQUE.** A pattern rail catches the patterns it was written for and essentially nothing
    else -- a regex cannot match a family it has no rule for. A learned classifier generalises a
    little past its training families. Neither catches a technique neither was built for, and the
    technique is the attacker's choice, not yours.
  * **PATH.** An input rail never sees an indirect injection, because the payload never travels
    through the user's message. Coverage is not competence, and the corpus's central observation --
    *"teams reliably add the obvious input and output rails, and reliably skip the two that matter
    most: filtering the retrieved context and gating the tool calls"* [T] -- is a COVERAGE statement
    wearing a competence sentence's clothes.

So a rail here is `(sees, covers, r, residual)` and compound catch is computed over an explicit joint
distribution of (technique, path). The scalar model is kept alongside it, because **the gap between
the two numbers is the headline finding of this topic**, and because the gap is largest exactly where
the corpus says teams are weakest.

The four rails modelled here are the four DETECTION positions (two at the input position, one at
retrieval, one at the output). The fifth control the corpus names -- the tool-call rail -- is
deliberately absent from this module: it does not detect anything, it removes a capability. That
makes it a different kind of object and it lives in `gating.py`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ------------------------------------------------------------------------------------------------
# The two axes the scalar model discards
# ------------------------------------------------------------------------------------------------

TECHNIQUES = ("known_pattern", "paraphrase", "structural", "novel")
PATHS = ("direct", "retrieved", "tool_output", "uploaded_doc")

TECHNIQUE_NOTE = {
    "known_pattern": "matches a rule somebody already wrote",
    "paraphrase": "the same attack in different words -- needs a model, not a regex",
    "structural": "a shape, not a phrase -- exfiltration markers, encoded payloads, delimiter breaks",
    "novel": "a technique nobody has a rule or a training example for yet",
}

PATH_NOTE = {
    "direct": "the user types it -- the only path an input rail can see",
    "retrieved": "inside a document the RAG system retrieved -- the user never typed it",
    "tool_output": "inside a tool's JSON response, after a legitimate call",
    "uploaded_doc": "inside a file the user attached without reading it",
}


@dataclass(frozen=True)
class Rail:
    """One detection rail. `r` is competence; `covers` is reach; `residual` is generalisation."""

    name: str
    kind: str                 # "pattern" | "classifier"
    r: float                  # catch rate WITHIN competence
    residual: float           # catch rate OUTSIDE competence
    sees: frozenset           # techniques this rail can see at all
    covers: frozenset         # paths that reach this rail
    note: str = ""

    def catch(self, technique: str, path: str) -> float:
        """Conditional catch rate. Zero when the payload never reaches this rail."""
        if path not in self.covers:
            return 0.0
        return self.r if technique in self.sees else self.residual


# ------------------------------------------------------------------------------------------------
# The attack mix
# ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AttackClass:
    name: str
    technique: str
    path: str
    weight: float             # share of observed attack attempts
    severity: int             # 1..5, consequence if it succeeds


# An explicit joint distribution rather than two independent marginals, because the real world is
# not independent: novel techniques arrive through external content far more often than through the
# user's own typing. The severity column is what makes the tail visible -- it is not in the corpus,
# it is the thing the corpus's "measure safety" instruction [T] requires you to attach.
ATTACK_MIX = (
    AttackClass("direct_jailbreak",      "paraphrase",    "direct",       0.10, 3),
    AttackClass("direct_pattern",        "known_pattern", "direct",       0.06, 2),
    AttackClass("rag_injection_known",   "known_pattern", "retrieved",    0.17, 3),
    AttackClass("rag_injection_para",    "paraphrase",    "retrieved",    0.15, 4),
    AttackClass("rag_injection_struct",  "structural",    "retrieved",    0.06, 4),
    AttackClass("rag_injection_novel",   "novel",         "retrieved",    0.08, 5),
    AttackClass("tool_output_injection", "structural",    "tool_output",  0.12, 4),
    AttackClass("uploaded_doc_known",    "known_pattern", "uploaded_doc", 0.09, 3),
    AttackClass("uploaded_doc_novel",    "novel",         "uploaded_doc", 0.05, 5),
    AttackClass("exfil_via_output",      "structural",    "retrieved",    0.07, 5),
    AttackClass("novel_exfil",           "novel",         "tool_output",  0.05, 5),
)

# ------------------------------------------------------------------------------------------------
# The four detection rails, at the four positions the corpus names
# ------------------------------------------------------------------------------------------------

ALL_PATHS = frozenset(PATHS)

INPUT_PATTERN = Rail(
    "input_pattern", "pattern", r=0.88, residual=0.00,
    sees=frozenset({"known_pattern"}), covers=frozenset({"direct"}),
    note="regex / blocklist for 'ignore previous instructions', DAN, [INST], <|system|>",
)
INPUT_CLASSIFIER = Rail(
    "input_classifier", "classifier", r=0.92, residual=0.06,
    sees=frozenset({"known_pattern", "paraphrase"}), covers=frozenset({"direct"}),
    note="learned injection detector on the user message",
)
RETRIEVAL_RAIL = Rail(
    "retrieval_rail", "classifier", r=0.90, residual=0.05,
    sees=frozenset({"known_pattern", "paraphrase", "structural"}),
    covers=frozenset({"retrieved", "tool_output", "uploaded_doc"}),
    note="the layer teams skip -- scans untrusted content before it enters the context",
)
OUTPUT_VALIDATOR = Rail(
    "output_validator", "classifier", r=0.85, residual=0.03,
    sees=frozenset({"known_pattern", "paraphrase", "structural"}), covers=ALL_PATHS,
    note="scans the reply for exfiltration markers, encoded payloads, instruction echoes",
)

FULL_STACK = (INPUT_PATTERN, INPUT_CLASSIFIER, RETRIEVAL_RAIL, OUTPUT_VALIDATOR)

# The two rails the corpus says teams reliably build, and the one it says they skip for detection.
OBVIOUS_TWO = (INPUT_PATTERN, INPUT_CLASSIFIER)
SKIPPED_ONE = (RETRIEVAL_RAIL,)

RAIL_BY_NAME = {r.name: r for r in FULL_STACK}


# ------------------------------------------------------------------------------------------------
# Compound catch, done two ways
# ------------------------------------------------------------------------------------------------

def scalar_catch(stack) -> float:
    """The arithmetic the corpus's sentence invites: 1 - prod(1 - r_i).

    This treats every rail as an unconditional filter with a fixed scalar competence. It is what a
    design review produces when the rails are described in prose instead of measured per technique.
    """
    p_fail = 1.0
    for rail in stack:
        p_fail *= (1.0 - rail.r)
    return 1.0 - p_fail


def stack_catch(stack, technique: str, path: str) -> float:
    """The conditional model: a rail only contributes on a path it covers and a technique it sees."""
    p_fail = 1.0
    for rail in stack:
        p_fail *= (1.0 - rail.catch(technique, path))
    return 1.0 - p_fail


def measured_catch(stack, mix=ATTACK_MIX) -> float:
    """Traffic-weighted mean catch over the attack mix."""
    total = sum(a.weight for a in mix)
    return sum(a.weight * stack_catch(stack, a.technique, a.path) for a in mix) / total


def severity_weighted_catch(stack, mix=ATTACK_MIX) -> float:
    """Catch weighted by consequence instead of by frequency.

    This is the statistic the corpus's *"run a red team suite and track the injection catch rate"* [T]
    does NOT ask for, and the experiments show why it should.
    """
    denom = sum(a.weight * a.severity for a in mix)
    return sum(a.weight * a.severity * stack_catch(stack, a.technique, a.path) for a in mix) / denom


def catch_by_class(stack, mix=ATTACK_MIX) -> list:
    out = []
    for a in mix:
        out.append({
            "name": a.name, "technique": a.technique, "path": a.path,
            "weight": a.weight, "severity": a.severity,
            "catch": stack_catch(stack, a.technique, a.path),
            "escaped_share": a.weight * (1.0 - stack_catch(stack, a.technique, a.path)),
        })
    return sorted(out, key=lambda r: r["catch"])


def catch_by_technique(stack, mix=ATTACK_MIX) -> list:
    out = []
    for t in TECHNIQUES:
        w = sum(a.weight for a in mix if a.technique == t)
        if w == 0.0:
            continue
        c = sum(a.weight * stack_catch(stack, a.technique, a.path) for a in mix
                if a.technique == t) / w
        out.append({"technique": t, "share": w, "catch": c})
    return out


def catch_by_path(stack, mix=ATTACK_MIX) -> list:
    out = []
    for p in PATHS:
        w = sum(a.weight for a in mix if a.path == p)
        if w == 0.0:
            continue
        c = sum(a.weight * stack_catch(stack, a.technique, a.path) for a in mix
                if a.path == p) / w
        out.append({"path": p, "share": w, "catch": c})
    return out


def leave_one_out(stack) -> list:
    """What each rail is actually worth, given the rest of the stack."""
    base = measured_catch(stack)
    out = []
    for rail in stack:
        reduced = tuple(r for r in stack if r.name != rail.name)
        out.append({
            "rail": rail.name,
            "without": measured_catch(reduced),
            "marginal": base - measured_catch(reduced),
        })
    return sorted(out, key=lambda r: -r["marginal"])


def build_order(pool=FULL_STACK, mix=ATTACK_MIX) -> list:
    """Greedy: repeatedly add the rail that buys the most catch. The corpus's build order [T] is
    'tool-call rail first'; this computes the order for the DETECTION rails only, which is a
    different question and the one teams answer badly."""
    chosen = []
    remaining = list(pool)
    steps = []
    current = 0.0
    while remaining:
        scored = []
        for rail in remaining:
            c = measured_catch(tuple(chosen) + (rail,), mix)
            scored.append((c - current, c, rail))
        gain, c, rail = max(scored, key=lambda s: (s[0], -len(s[2].covers)))
        chosen.append(rail)
        remaining = [r for r in remaining if r.name != rail.name]
        steps.append({"step": len(chosen), "rail": rail.name, "gain": gain, "catch": c})
        current = c
    return steps


def ceiling_catch(stack, mix=ATTACK_MIX) -> float:
    """The stack's own theoretical maximum: every rail at r=1.0 within its competence.

    The distance between `measured_catch` and this number is the entire remaining headroom of the
    detection approach. If it is small, then "improve the classifier" is not a lever -- it is a
    rounding error -- and the design argument has to move somewhere else.
    """
    perfect = tuple(
        Rail(r.name, r.kind, r=1.0, residual=r.residual, sees=r.sees, covers=r.covers, note=r.note)
        for r in stack
    )
    return measured_catch(perfect, mix)


def evasion_sensitivity(stack, mix=ATTACK_MIX) -> dict:
    """How much of the stack's protection survives a technique nobody has seen.

    Answers the design question a scalar catch rate cannot: if every attacker switched to a novel
    technique tomorrow, what is left? The answer is the residual terms and nothing else.
    """
    shifted = tuple(
        AttackClass(a.name.replace("_known", "_novel"), "novel", a.path, a.weight, a.severity)
        for a in mix
    )
    return {
        "current": measured_catch(stack, mix),
        "all_novel": measured_catch(stack, shifted),
        "floor": _floor_catch(stack, mix),
    }


def _floor_catch(stack, mix=ATTACK_MIX) -> float:
    """The best any technique shift can do, i.e. catch when every rail is on its residual."""
    return min(stack_catch(stack, "novel", p) for p in PATHS)


def severity_skew(stack, mix=ATTACK_MIX) -> dict:
    """The tail-over-mean check, stated for this topic: frequency-weighted vs consequence-weighted.

    If the consequence-weighted catch is LOWER than the frequency-weighted catch, the stack is
    tuned to what is common rather than to what is expensive -- and the corpus's headline metric
    ("track the injection catch rate") is a mean with no name attached.
    """
    m = measured_catch(stack, mix)
    s = severity_weighted_catch(stack, mix)
    return {"mean_catch": m, "severity_catch": s, "skew": s - m,
            "mean_catch_rate": 1.0 - m, "severity_escape_rate": 1.0 - s}


def mix_summary(mix=ATTACK_MIX) -> dict:
    by_path, by_tech, by_sev = {}, {}, {}
    for a in mix:
        by_path[a.path] = by_path.get(a.path, 0.0) + a.weight
        by_tech[a.technique] = by_tech.get(a.technique, 0.0) + a.weight
        by_sev[a.severity] = by_sev.get(a.severity, 0.0) + a.weight
    return {
        "total_weight": sum(a.weight for a in mix),
        "by_path": by_path, "by_technique": by_tech, "by_severity": by_sev,
        "mean_severity": sum(a.weight * a.severity for a in mix) / sum(a.weight for a in mix),
        "direct_share": sum(a.weight for a in mix if a.path == "direct"),
    }


# ------------------------------------------------------------------------------------------------
# Latency, because every rail is on the critical path and the cheap ones are not the fast ones
# ------------------------------------------------------------------------------------------------

RAIL_LATENCY_MS = {
    # per-request cost unless noted. Ordering is the claim (the cheapest rail carries the largest
    # blast radius); the microseconds are illustrative.
    "input_pattern": (0.4, 1.5),      # (p50, p99)
    "input_classifier": (4.0, 12.0),
    "retrieval_rail": (6.0, 22.0),    # PER RETRIEVED CHUNK at k=5 -- the dominant line item
    "output_validator": (5.0, 18.0),
    "tool_call_gate": (0.2, 0.9),     # cheap, structural, and the largest blast radius
}


def rail_latency(stack, k_chunks: int = 5, cached_frac: float = 0.0) -> dict:
    """Rail latency with an optional per-chunk verdict cache.

    The corpus does not discuss rail cost at all; this is the accounting its *"safety is only real
    if you are measuring it"* [T] instruction implies, and it is the reason the retrieval rail is
    the one teams skip for cost rather than for principle.
    """
    rows = []
    total_p50 = total_p99 = 0.0
    for rail in stack:
        p50, p99 = RAIL_LATENCY_MS[rail.name]
        mult = k_chunks if rail.name in ("retrieval_rail",) else 1
        if rail.name == "retrieval_rail":
            mult = k_chunks * (1.0 - cached_frac)
        rows.append({"rail": rail.name, "calls": mult, "ms_p50": p50 * mult, "ms_p99": p99 * mult})
        total_p50 += p50 * mult
        total_p99 += p99 * mult
    return {"rows": rows, "p50": total_p50, "p99": total_p99}


# ------------------------------------------------------------------------------------------------
# Trust tags: a rail's reach depends on a label that has to survive the pipeline
# ------------------------------------------------------------------------------------------------

TRUST_RANK = {
    "system": 5,
    "user": 4,
    "retrieved_trusted": 4,
    "tool_output": 3,
    "retrieved_untrusted": 2,
    "unknown": 0,          # the tag was lost
}


def tag_survival(hops: int, loss_per_hop: float) -> float:
    """P(the trust tag is still attached when the content reaches the model)."""
    return (1.0 - loss_per_hop) ** hops


def effective_privilege(hops: int, loss_per_hop: float, fail_open: bool) -> dict:
    """What fraction of untrusted content arrives with instruction-level privilege.

    The design decision this measures is a one-word one: when the tag is missing, does the pipeline
    FAIL OPEN (treat it as the user's own words -- privileged) or FAIL CLOSED (treat it as scraped
    content -- restricted)? Fail-open turns a plumbing bug into a security control bypass.
    """
    survival = tag_survival(hops, loss_per_hop)
    lost = 1.0 - survival
    privileged = lost * (1.0 if fail_open else 0.0)
    return {
        "survival": survival,
        "lost": lost,
        "privileged_share": privileged,
        "restricted_share": survival + (lost if not fail_open else 0.0),
    }


def tag_loss_false_refusal(hops: int, loss_per_hop: float, legitimate_share: float) -> float:
    """Fail-closed has a cost: legitimate first-party content that lost its tag gets restricted."""
    return (1.0 - tag_survival(hops, loss_per_hop)) * legitimate_share
