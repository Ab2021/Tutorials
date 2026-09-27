"""T18 -- guardrails and security: the competence matrix, the gate, and the decay.

The organising claim [D]: **a guardrail is not a filter, it is a competence matrix plus a capability
decision -- and only one of the two still has room.**

  * **Detect.** A rail's catch rate is conditional on the attack TECHNIQUE and on the PATH the
    payload travelled. Described as a scalar it produces the arithmetic everyone quotes,
    `1 - prod(1 - r_i)`, which overstates protection by tens of points precisely where the corpus
    says teams are weakest.                                                             -> rails.py
  * **Gate.** The layer with the largest blast radius detects nothing. It removes a capability, and
    that is why it is the only control whose strength does not depend on recognising the attack --
    and why it is the one with an order of magnitude still available.      -> gating.py
  * **Balance.** Both of the above cost something: false refusals, latency, human attention. The
    corpus gives the trade as a warning ("a system that blocks everything is safe and useless");
    this is the same trade as arithmetic, with an optimum that moves by orders of magnitude between
    two businesses.                                                                     -> policy.py
  * **Decay.** And none of it holds still. "A rail you tested in March may be bypassed by June" is
    a schedule, and a schedule is a number.                                             -> decay.py

The through-line, and the reason the four modules are separable: **the three failure modes of a
guardrail stack are a metric that averages away the class that matters, a control whose strength is
somebody's attention, and a schedule.** Each module measures one of them.

Provenance inline: [T] corpus, [R] supporting repo, [D] derived. Stdlib only, no GPU, no network.
"""
from .rails import (
    TECHNIQUES, PATHS, TECHNIQUE_NOTE, PATH_NOTE, TRUST_RANK,
    Rail, AttackClass, ATTACK_MIX, FULL_STACK, OBVIOUS_TWO, SKIPPED_ONE, RAIL_BY_NAME,
    INPUT_PATTERN, INPUT_CLASSIFIER, RETRIEVAL_RAIL, OUTPUT_VALIDATOR, ALL_PATHS,
    scalar_catch, stack_catch, measured_catch, severity_weighted_catch, ceiling_catch,
    catch_by_class, catch_by_technique, catch_by_path, leave_one_out, build_order,
    evasion_sensitivity, severity_skew, mix_summary,
    RAIL_LATENCY_MS, rail_latency, tag_survival, effective_privilege, tag_loss_false_refusal,
)
from .gating import (
    READ, WRITE_LOW, EXTERNAL, IRREVERSIBLE, RISK_ORDER, RISK_NOTE, MIN_TRUST,
    ALLOW, ALLOW_LOGGED, DRY_RUN, APPROVAL, DENY, OUTCOME_NOTE,
    Tool, TOOLS, TOOL_BY_NAME, HARM_CAPABLE, BLAST_RADIUS, Decision,
    session_trust, decide, gate_matrix,
    P_TARGET_IRREVERSIBLE, P_TARGET_EXTERNAL, SCHEMA_CATCH,
    BASE_APPROVER_ACCURACY, APPROVER_ACCURACY_DECAY_DEPTH,
    gate_failure, harm_rate, lever_table,
    ApprovalQueue, approver_accuracy_at_depth, queue_profile, sweep_queue,
    approval_surface, undo_vs_approve, gate_under_tag_loss,
)
from .policy import (
    normal_cdf, normal_pdf, Detector, CHEAP_FILTER, PRECISE_CHECK,
    Scenario, SCENARIOS, Segment, SEGMENTS,
    expected_cost, cost_curve, optimal_threshold, closed_form_theta,
    cascade, find_operating_point, cascade_vs_single, segment_fpr,
)
from .decay import (
    DecayParams, Cadence, CADENCES, run_decay, cadence_table, cadence_for_trough,
    metric_blindness, R_LIBRARY, R_RESIDUAL, SEV_LIBRARY, SEV_NOVEL,
)
from . import experiments

__all__ = [
    "TECHNIQUES", "PATHS", "TECHNIQUE_NOTE", "PATH_NOTE", "TRUST_RANK",
    "Rail", "AttackClass", "ATTACK_MIX", "FULL_STACK", "OBVIOUS_TWO", "SKIPPED_ONE",
    "RAIL_BY_NAME", "INPUT_PATTERN", "INPUT_CLASSIFIER", "RETRIEVAL_RAIL", "OUTPUT_VALIDATOR",
    "ALL_PATHS",
    "scalar_catch", "stack_catch", "measured_catch", "severity_weighted_catch", "ceiling_catch",
    "catch_by_class", "catch_by_technique", "catch_by_path", "leave_one_out", "build_order",
    "evasion_sensitivity", "severity_skew", "mix_summary",
    "RAIL_LATENCY_MS", "rail_latency", "tag_survival", "effective_privilege",
    "tag_loss_false_refusal",
    "READ", "WRITE_LOW", "EXTERNAL", "IRREVERSIBLE", "RISK_ORDER", "RISK_NOTE", "MIN_TRUST",
    "ALLOW", "ALLOW_LOGGED", "DRY_RUN", "APPROVAL", "DENY", "OUTCOME_NOTE",
    "Tool", "TOOLS", "TOOL_BY_NAME", "HARM_CAPABLE", "BLAST_RADIUS", "Decision",
    "session_trust", "decide", "gate_matrix",
    "P_TARGET_IRREVERSIBLE", "P_TARGET_EXTERNAL", "SCHEMA_CATCH",
    "BASE_APPROVER_ACCURACY", "APPROVER_ACCURACY_DECAY_DEPTH",
    "gate_failure", "harm_rate", "lever_table",
    "ApprovalQueue", "approver_accuracy_at_depth", "queue_profile", "sweep_queue",
    "approval_surface", "undo_vs_approve", "gate_under_tag_loss",
    "normal_cdf", "normal_pdf", "Detector", "CHEAP_FILTER", "PRECISE_CHECK",
    "Scenario", "SCENARIOS", "Segment", "SEGMENTS",
    "expected_cost", "cost_curve", "optimal_threshold", "closed_form_theta",
    "cascade", "find_operating_point", "cascade_vs_single", "segment_fpr",
    "DecayParams", "Cadence", "CADENCES", "run_decay", "cadence_table", "cadence_for_trough",
    "metric_blindness", "R_LIBRARY", "R_RESIDUAL", "SEV_LIBRARY", "SEV_NOVEL",
    "experiments",
]
