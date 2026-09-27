"""T09 -- speculative decoding: exactness, acceptance, draft cost, and the batch regime.

Speculative decoding is usually presented as a speedup. It is three separate things, and conflating
them is why deployments disappoint:

  * an EXACT sampler whose correctness rests on one construction (residual resampling) -- remove it
    and the output distribution drifts while the text stays fluent                 -> acceptance.py
  * a family of DRAFTERS whose cost `c` varies by more than an order of magnitude, and whose cost
    is a bigger lever than their quality at useful draft lengths                    -> drafters.py
  * a REGIME-DEPENDENT optimisation that converts spare tensor cores into tokens, and therefore
    stops working exactly when the batch grows enough to consume them               -> regimes.py

The three are separable and must be reasoned about separately: a perfect drafter loses money in the
compute-bound regime, and a free drafter (n-gram) cannot drift. Provenance is inline: [T] corpus,
[R] supporting repo, [D] derived.
"""
from .acceptance import (normalise, residual, rejection_mass, acceptance_rate, total_variation,
                         speculative_step, naive_step, empirical_distribution, exactness_report,
                         decay_profile, tokens_per_step)
from .drafters import (draft_model_cost, mtp_cost, ngram_cost, medusa_tree_cost,
                       ngram_acceptance, drift_penalty, tree_expected_accepts,
                       linear_expected_accepts)
from .regimes import (speedup, gamma_sweep, optimal_gamma, marginal_gamma_gain,
                      arithmetic_intensity, memory_bound_batch_limit, step_cost,
                      speedup_vs_batch, deployment_verdict)
from . import experiments

__all__ = [
    "normalise", "residual", "rejection_mass", "acceptance_rate", "total_variation",
    "speculative_step", "naive_step", "empirical_distribution", "exactness_report",
    "decay_profile", "tokens_per_step",
    "draft_model_cost", "mtp_cost", "ngram_cost", "medusa_tree_cost",
    "ngram_acceptance", "drift_penalty", "tree_expected_accepts", "linear_expected_accepts",
    "speedup", "gamma_sweep", "optimal_gamma", "marginal_gamma_gain",
    "arithmetic_intensity", "memory_bound_batch_limit", "step_cost",
    "speedup_vs_batch", "deployment_verdict", "experiments",
]
