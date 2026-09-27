"""The candidate pool: construction, diversity measurement, and the gate.

Design point carried from HLD sec 5.5: `n` candidates are not `n` distinct candidates.
At temperature 0.2, 100 draws produce only ~20 unique outputs [T] CMU lecture 12. The
unique fraction is therefore measured, never assumed, and the gate sits BEFORE scoring so
that no verifier compute is spent on a pool that cannot support a selection.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    """One completion.

    `logprob` is the generator's own score and is an INPUT, never the selection signal: a
    model's log-probability is biased toward what it already produces, so selecting on it
    degenerates to greedy decoding with extra steps. It is carried for diagnostics only.
    """
    id: str
    text: str
    length: int
    logprob: float
    meta: dict = field(default_factory=dict)


def make_rng(seed: int) -> random.Random:
    """The stdlib generator, explicitly seeded, so two experiments cannot interfere."""
    return random.Random(seed)


def make_candidate(cid: str, text: str, length: int, logprob: float, meta: dict | None = None) -> Candidate:
    return Candidate(id=cid, text=text, length=length, logprob=logprob, meta=dict(meta or {}))


def unique_fraction(candidates: list[Candidate], dedup_key=None) -> float:
    """Distinct candidates / total candidates.

    Computed from the exact dedup key. Whether near-duplicates should count as distinct is a
    TASK decision, and this function declines to make it silently: a task that wants semantic
    deduplication passes its own `dedup_key` and the count follows it.
    """
    if not candidates:
        return 0.0
    if dedup_key is None:
        keys = {c.text for c in candidates}
    else:
        keys = {dedup_key(c) for c in candidates}
    return len(keys) / len(candidates)


def build_pool(prompt: str, candidates: list[Candidate], floor: float = 0.5,
               dedup_key=None) -> dict:
    """Build a Pool. Never raises on a degenerate pool -- it records the fact.

    The caller decides whether to abort, regenerate or proceed, because those are POLICY
    choices the HLD assigns to the pipeline rather than to the pool builder.
    """
    uf = unique_fraction(candidates, dedup_key)
    return {
        "prompt": prompt,
        "candidates": list(candidates),
        "requested_n": len(candidates),
        "unique_fraction": uf,
        "degenerate": uf < floor,
    }


def diversity_gate(pool: dict, floor: float) -> tuple[bool, str]:
    """(passes, reason). The value of this gate is not statistical -- it is that no verifier
    compute is spent on a pool that cannot support a selection."""
    uf = pool["unique_fraction"]
    if uf < floor:
        return False, f"unique_fraction {uf:.3f} < floor {floor:.3f}: pool is degenerate"
    return True, f"unique_fraction {uf:.3f} >= floor {floor:.3f}"


def expected_unique(n: int, m: int) -> float:
    """Expected distinct outcomes when n draws come from m equally likely phrasings.

    A model for the temperature/diversity relationship: at low temperature the model
    concentrates on few phrasings, so m is small and a large n buys little.
    """
    if m <= 1:
        return 1.0
    return m * (1.0 - (1.0 - 1.0 / m) ** n)


def phrasings_at_temperature(temperature: float, base_m: int = 20) -> int:
    """The corpus datum: at temperature 0.2, 100 draws give ~20 unique outputs [T].

    So m(0.2) = 20. We scale linearly in temperature and floor at 1.
    """
    if temperature <= 0.0:
        return 1
    return max(1, int(round(base_m * (temperature / 0.2))))


def draw_pool(n: int, temperature: float, rng: random.Random, base_m: int = 20,
              quality_of=None) -> list[Candidate]:
    """Draw a pool whose DISTINCTNESS follows the temperature model above.

    Candidates are labelled c0..c{m-1} by phrasing; duplicates get distinct ids so the pool
    is exactly n long while its unique fraction follows the model.
    """
    m = phrasings_at_temperature(temperature, base_m)
    out = []
    for i in range(n):
        phrasing = rng.randrange(m)
        quality = quality_of(phrasing) if quality_of else 0.5
        out.append(make_candidate(
            cid=f"c{i:03d}",
            text=f"phrasing-{phrasing:03d}",
            length=100 + 20 * phrasing,
            logprob=-0.1 * i,
            meta={"quality": quality, "family": "A"},
        ))
    return out


def cosine_decay(x: float) -> float:
    """A tiny helper used by the length experiments: 1 at x=0, 0 at x=1, smooth."""
    x = max(0.0, min(1.0, x))
    return 0.5 * (1.0 + math.cos(math.pi * x))
