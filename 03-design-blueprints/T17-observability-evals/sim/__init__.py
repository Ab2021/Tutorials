"""T17 -- observability, tracing and evaluation: the three planes and the loop that joins them.

The organising claim [D]: **observability is three planes, not one.** A team with only the first can
tell you the GPU is busy and nothing else. The corpus's own two sentences about the dashboard
"nobody owns" [T] are this claim stated as a failure mode.

  * **system observability** -- metrics: TTFT, ITL, goodput, queue depth, KV occupancy. "Is it
    healthy, fast and affordable?"                                                        -> sampling.py
  * **trace observability** -- nested spans per request, so a slow or wrong answer can be
    DECOMPOSED rather than merely measured. "Where did the time go?"                       -> latency.py
  * **quality evaluation** -- offline and online scoring of outputs and trajectories. "Was it
    any good?"                                                                              -> judge.py
                                                                                             -> stats.py
                                                                                             -> gate.py
  * and the edge that joins them: traces -> judge -> dataset -> gate. Without it the first two are
    a storage bill                                                                          -> loop.py

The through-line, and the reason the modules are separable: **each plane answers a question the
others cannot.** Metrics cannot attribute; traces cannot judge; a judge cannot tell you what
happened. The corpus gives the same list twice -- "trace every request as a tree of nested spans",
"build on the OpenTelemetry standard", "put P95 latency, cost and error rate on a dashboard", and
"close the loop by feeding traces into your evaluations and alerts" [T] -- and the fourth item is
the one that turns the other three into a system.

Provenance inline: [T] corpus, [R] supporting repo, [D] derived. Stdlib only, no GPU, no network.
"""
from .latency import (Span, walk, end_to_end_ms, by_name, leaves, decomposition_check, attribution,
                      chat_request, agentic_request, render_tree)
from .sampling import (Policy, DEFAULT_POLICIES, expected_kept, storage_cost_gb, p_zero_captured,
                       requests_until_first_trace, frontier, redaction_is_orthogonal)
from .judge import (Answer, Pair, JudgeParams, Correction, GoldSet, synthetic_gold_set,
                    judge_score, judge_pair, judge_pointwise, both_orders_verdict, agreement,
                    cohen_kappa, evaluate, DEFAULT_JUDGE, correction_ladder, position_bias_sweep,
                    verbosity_effect)
from .stats import (normal_cdf, standard_error, win_rate_sigma, minimum_detectable_effect,
                    observed_gain_distribution, false_improvement_rate, detection_power,
                    tail_representation, eval_set_sizing, required_n)
from .gate import (ScoreSet, Gate, make_release, gate_matrix, disagreement,
                   regression_hidden_by_mean)
from .loop import (FailureClass, failure_taxonomy, LoopRun, run_loop, compare, open_loop_cost)
from . import experiments

__all__ = [
    "Span", "walk", "end_to_end_ms", "by_name", "leaves", "decomposition_check", "attribution",
    "chat_request", "agentic_request", "render_tree",
    "Policy", "DEFAULT_POLICIES", "expected_kept", "storage_cost_gb", "p_zero_captured",
    "requests_until_first_trace", "frontier", "redaction_is_orthogonal",
    "Answer", "Pair", "JudgeParams", "Correction", "GoldSet", "synthetic_gold_set", "judge_score",
    "judge_pair", "judge_pointwise", "both_orders_verdict", "agreement", "cohen_kappa", "evaluate",
    "DEFAULT_JUDGE", "correction_ladder", "position_bias_sweep", "verbosity_effect",
    "normal_cdf", "standard_error", "win_rate_sigma", "minimum_detectable_effect",
    "observed_gain_distribution", "false_improvement_rate", "detection_power",
    "tail_representation", "eval_set_sizing", "required_n",
    "ScoreSet", "Gate", "make_release", "gate_matrix", "disagreement", "regression_hidden_by_mean",
    "FailureClass", "failure_taxonomy", "LoopRun", "run_loop", "compare", "open_loop_cost",
    "experiments",
]
