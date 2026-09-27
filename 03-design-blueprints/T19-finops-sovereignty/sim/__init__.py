"""T19 -- finops and sovereignty: what a discount is worth, what a unit decides, and what a posture
can actually evidence.

The organising claim [D]: **cost and sovereignty are the same question asked twice -- "who controls
the layer that turns my data into value" -- and both are settled by measurement and attestation, not
by intent.** The tokenomics talk states the cost half ("optimization without measurement is just
guessing" [T]); the sovereign-AI talk states the governance half ("you can't legislate a memory
dump" [T]). Neither half is a number until you have chosen a unit and an evidence type, and both of
those choices are made before any measurement happens.

Five modules, and the reason they are separable is that each one is a DIFFERENT KIND of mistake:

  * **The stack.** Eight lines on the bill with wildly different sizes, and a discount that applies
    to one line read as if it applied to all of them. A cache read "roughly 10x cheaper" [T] is a
    statement about a PRICE, and a programme banks `line_reduction x share_of_bill`. -> stack.py
  * **The levers.** The corpus gives a ladder and a headline ("roughly a tenth of the cost" [T]).
    The headline is testable. Ranking by claimed discount and ranking by achievable saving give
    different orders, two kinds of lever compose differently, and the ladder has a ceiling that is
    not where the headline is.                                                        -> levers.py
  * **The unit.** The Feb->Apr figures -- 30c to 16c, 630 lines to 91, 8.2 files to 3.6 -- are three
    signs from two rows, and which one you quote is a decision made before the arithmetic. The
    workload mix then puts a time axis on the same shape.                                -> units.py
  * **The posture.** Sovereignty is a four-dimensional vector with prerequisites, and continuity is
    a minimum rather than a mean. Residency is a filter and never a score, because a legal property
    has no error tolerance.                                                        -> sovereignty.py
  * **Attribution.** The precondition for all of it, and the one metric whose coverage falls as the
    thing it measures grows, because the untagged spend is where the growth is.   -> attribution.py

Four claims the corpus makes are quantified here rather than repeated:

  * a **20% throughput penalty** for confidential computing is a **25% cost uplift**, not 20%;
  * the levers' **ceiling is 60.6%**, against a headline of 90%;
  * **30c to 16c** is a 47% saving per artifact, a 22% increase per file and a 270% increase per
    line of delivered work;
  * a **score-based residency router leaks**, and a filter leaks zero, at any noise level.

Provenance inline: [T] corpus transcript, [R] supporting repo, [D] derived. Stdlib only -- no numpy,
no GPU, no network. Every number reproducible from `python run.py`.
"""
from .stack import (
    PREFIX_SHARE_OF_INPUT, INPUT_SHARE_OF_BILL, Layer, BILL, BILL_TOTAL, LAYER_BY_NAME,
    CACHED_TODAY, CACHED_TARGET,
    percent_to_multiple, cache_read_price, prefix_line, caching_saving, discount_sweep,
    asymptotic_reduction, write_premium_break_even, caching_npv,
    reasoning_amplification, reasoning_gating_saving, layer_table,
)
from .levers import (
    ALL, Lever, LEVERS, LEVER_BY_NAME,
    achievable, bill_share, rank_by_achievable, rank_by_claim, ordering_disagreement,
    apply_order, conflict_pairs, cheapest_first, free_wins, reduction_levers, pricing_levers,
    value_per_ease, order_saving_first, order_free_first, order_ease_first, ordering_comparison,
    ceiling, max_single_discount, headline_check,
)
from .units import (
    Artifact, FEB, APR, WORKLOADS, UNIT_NOTE, HEADLINE_UNIT,
    normalised, unit_ranking, deflation_check, metric_families,
    Workload, MIX, GROWTH_FACTOR,
    mix_summary, tail_ratio, growth_scenario, cost_per_task_vs_call, where_the_money_is,
    reasoning_invisibility,
)
from .sovereignty import (
    DIMENSIONS, DIMENSION_QUESTION, PREREQ, Posture, POSTURES, POSTURE_BY_NAME,
    effective, claimed_vs_effective, posture_table, mean_vs_min,
    tee_uplift, attested_blended_uplift, tee_budget_table, attestation_breakeven_cost,
    normal_cdf, residency_score_leak, residency_leaks, residency_compliance,
    CONTINUITY_LAYERS, CONTINUITY_NOTE, continuity, continuity_examples,
    THREE_WAY_TRUST, trust_gap_closers, open_weights_spectrum, spectrum_position,
)
from .attribution import (
    Team, TEAMS, GROWTH_FACTOR as ATTR_GROWTH_FACTOR,
    coverage, untagged_share_of_growth, governance_ladder, shadow_gap, cap_from_mean_vs_tail,
)
from . import experiments

__all__ = [
    "PREFIX_SHARE_OF_INPUT", "INPUT_SHARE_OF_BILL", "Layer", "BILL", "BILL_TOTAL",
    "LAYER_BY_NAME", "CACHED_TODAY", "CACHED_TARGET",
    "percent_to_multiple", "cache_read_price", "prefix_line", "caching_saving", "discount_sweep",
    "asymptotic_reduction", "write_premium_break_even", "caching_npv",
    "reasoning_amplification", "reasoning_gating_saving", "layer_table",
    "ALL", "Lever", "LEVERS", "LEVER_BY_NAME",
    "achievable", "bill_share", "rank_by_achievable", "rank_by_claim", "ordering_disagreement",
    "apply_order", "conflict_pairs", "cheapest_first", "free_wins", "reduction_levers", "pricing_levers",
    "value_per_ease", "order_saving_first", "order_free_first", "order_ease_first",
    "ordering_comparison", "ceiling", "max_single_discount", "headline_check",
    "Artifact", "FEB", "APR", "WORKLOADS", "UNIT_NOTE", "HEADLINE_UNIT",
    "normalised", "unit_ranking", "deflation_check", "metric_families",
    "Workload", "MIX", "GROWTH_FACTOR",
    "mix_summary", "tail_ratio", "growth_scenario", "cost_per_task_vs_call", "where_the_money_is",
    "reasoning_invisibility",
    "DIMENSIONS", "DIMENSION_QUESTION", "PREREQ", "Posture", "POSTURES", "POSTURE_BY_NAME",
    "effective", "claimed_vs_effective", "posture_table", "mean_vs_min",
    "tee_uplift", "attested_blended_uplift", "tee_budget_table", "attestation_breakeven_cost",
    "normal_cdf", "residency_score_leak", "residency_leaks", "residency_compliance",
    "CONTINUITY_LAYERS", "CONTINUITY_NOTE", "continuity", "continuity_examples",
    "THREE_WAY_TRUST", "trust_gap_closers", "open_weights_spectrum", "spectrum_position",
    "Team", "TEAMS", "ATTR_GROWTH_FACTOR",
    "coverage", "untagged_share_of_growth", "governance_ladder", "shadow_gap",
    "cap_from_mean_vs_tail",
    "experiments",
]
