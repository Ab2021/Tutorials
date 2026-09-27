"""T05 verifiers and best-of-N: a toy selection service.

Nothing here calls a model. Every scorer is an explicit, inspectable arithmetic function
whose biases are PARAMETERS, so the experiments can turn each bias on and off and observe
the difference. A simulated unbiased oracle would make the whole package vacuous.
"""
from .candidates import (
    Candidate, make_candidate, unique_fraction, build_pool, diversity_gate, make_rng,
)
from .verifiers import (
    programmatic_score, orm_score, prm_score, judge_score, tiered_score, measure_agreement,
)
from .selection import select, kl_bound, acceptance_probability, rejection_sample

__all__ = [
    "Candidate", "make_candidate", "unique_fraction", "build_pool", "diversity_gate",
    "make_rng", "programmatic_score", "orm_score", "prm_score", "judge_score",
    "tiered_score", "measure_agreement", "select", "kl_bound",
    "acceptance_probability", "rejection_sample",
]
