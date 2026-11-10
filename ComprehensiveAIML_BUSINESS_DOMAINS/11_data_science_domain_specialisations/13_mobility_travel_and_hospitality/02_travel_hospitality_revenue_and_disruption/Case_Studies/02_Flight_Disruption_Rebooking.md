---
case_id: "CASE-13-02-002"
case_title: "Illustrative Flight disruption rebooking"
case_version: "1.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "optimization"
domain: "Mobility, travel and hospitality"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["distribution-shift", "human-in-the-loop"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-09"
owner_role: "irregular-operations lead"
review_trigger: "Decision policy, event contract, or population changes"
---

# Illustrative Flight disruption rebooking
## 1. The One-Line Framing
Given a versioned affected itinerary, recommend feasible rebooking proposal for the irregular-operations lead against first available seat, subject to explicit feasibility, safety, and customer-impact constraints.
This is an illustrative design exercise. Its values teach arithmetic and decision framing; they are not results from a named company or deployment.
The simpler baseline persists because it is explainable and available during incomplete data; the proposed method must outperform it on a declared decision, not merely score well offline.
Takeaway: the decision remains bounded by evidence, ownership, and recoverability.
## 2. Context & Background
The workflow begins when a affected itinerary enters an eligibility gate and ends when an accountable operator accepts, overrides, or defers the feasible rebooking proposal.
The decision owner is the irregular-operations lead; operations owns immediate containment, data engineering owns lineage, and an independent reviewer owns policy exceptions.
The baseline is first available seat. It can be preferred when evidence is stale, capacity is unknown, or the proposed score cannot be explained at the action cutoff.
Scale, regions, customer mix, and legal obligations are intentionally unspecified; no inferred company process or requirement is claimed.
Takeaway: the decision remains bounded by evidence, ownership, and recoverability.
## 3. The Business Question
Question: for each eligible affected itinerary, should the owner choose a constrained feasible rebooking proposal, retain first available seat, or request review?
Trigger: a decision window opens with finite capacity and incomplete outcomes. Deadline: before the action becomes irreversible.
Objective: maximize qualified decisions, defined as accepted useful actions divided by eligible decisions, while vetoing unsafe, infeasible, or unexplained actions.
Out of scope: setting a universal threshold, asserting fairness compliance, or estimating commercial impact without local data.
Takeaway: the decision remains bounded by evidence, ownership, and recoverability.
## 4. Assumptions & Constraints
A-01: all inputs used at decision time have availability timestamps. Verification: replay a point-in-time snapshot. If violated: remove the feature and return to the baseline.
A-02: an operator can review uncertain cases. Verification: roster and queue simulation. If violated: narrow automation scope rather than silently increasing risk.
C-01: action must be feasible under current capacity. Verification: query the authoritative availability source with a version. Fallback: defer and explain the constraint.
C-02: every recommendation is auditable. Verification: persist policy version, inputs, output, and override reason; missing logs block promotion.
Takeaway: the decision remains bounded by evidence, ownership, and recoverability.
## 5. Data, Tools & Method
### 5a. Data Specification
Unit of analysis: one affected itinerary keyed by immutable decision ID. Inputs are timestamped eligibility, capacity, context, and outcome fields. Labels are only used after their maturity cutoff.
Illustrative split: earlier windows for development, a later untouched window for evaluation, and an entity boundary when repeated entities could leak. No actual dataset is claimed.
### 5b. Tools & Stack
Proposed components: versioned event ledger, validation job, simple baseline calculator, constrained scorer, policy gate, audit store, and operator console. Versions and deployment details are not selected until local requirements exist.
### 5c. Method Summary
Input -> validate and authorize -> calculate baseline (first available seat) -> estimate or rank -> apply hard feasibility rules -> feasible rebooking proposal or review.
Candidates: baseline rule (always available, low adaptation); constrained score (more specific, requires validated inputs); manual review (auditable, capacity-limited). None is reported as tested.
### 5d. Data provenance, quality, exclusions and leakage controls
Exclude records with unknown decision time, conflicting identity, or unavailable-at-cutoff inputs; retain exclusion reasons. Train-only fitting and point-in-time joins are required before any evaluation.
### 5e. Reproducibility and method-selection ledger
Proposed artifact IDs: SRC-01 event schema; CFG-01 eligibility policy; RUN-01 replay configuration. They are placeholders for future local artifacts, not evidence that a run occurred.
Takeaway: the decision remains bounded by evidence, ownership, and recoverability.
## 6. Analysis Walkthrough
Decisive calculation (ILLUSTRATIVE): among 80 eligible decisions, a constrained process yields 76 qualified outcomes and the baseline yields 72.
Qualified rate for constrained process = 76/80=0.95; baseline =72/80=0.90; absolute difference=0.05 or 5 percentage points.
Independent arithmetic check: 0.05*80=4 additional qualified outcomes. This does not establish uncertainty, causality, or transferability.
Sensitivity: if two constrained outcomes are unresolved rather than qualified, 74/80=92.5%, shrinking the difference to 2.5 points. The action should therefore wait for outcome maturity or use a conservative gate.
What we tried first: no local candidate was actually tested. The baseline and alternatives are considered designs, not historical experiments.
Takeaway: the decision remains bounded by evidence, ownership, and recoverability.
## 7. The Finding
No production decision is supported. The illustrative calculation shows the proposed feasible rebooking proposal could be sensitive to four outcome labels in a sample of 80.
The recommendation is to run a predeclared replay and capacity review before replacing first available seat. The main boundary is that the example has no representative workload, completed test, or causal comparator.
Takeaway: the decision remains bounded by evidence, ownership, and recoverability.
## 8. System/Solution Design
### 8a. Offline / Training / Batch Path
[Versioned ledger] -> [schema and point-in-time checks] -> [baseline/scorer build] -> [replay evaluation] -> [signed policy bundle]. Invalid data goes to quarantine with a reason code.
### 8b. Online / Serving / Real-Time Path
[affected itinerary] -> [authorize + validate] -> [authoritative feasibility read] -> [baseline and constrained policy] -> [feasible rebooking proposal / review / defer] -> [audit log + outcome feedback].
Failure path: a stale feasibility read, missing required input, or timed-out dependency produces a deferred baseline-safe response and an audit event; retries have a bounded budget and idempotency key.
### 8c. Key Design Decisions
DEC-01: gate hard constraints before ranking; alternative post-hoc filtering can make ranking metrics misleading. Revisit when feasibility semantics change.
DEC-02: preserve operator override; alternative full automation is disqualified until review capacity, error cost, and recovery are evidenced.
### 8d. Operational readiness
Owner: operations for response, data owner for input quality, policy owner for changes. Proposed SLI: fraction of eligible decisions receiving an auditable response; target is deliberately not set without requirements.
Takeaway: the decision remains bounded by evidence, ownership, and recoverability.
## 9. Evaluation & Validation
Primary metric: qualified outcome rate = qualified eligible decisions / eligible decisions, reported by decision window and cohort.
Baseline: the stated simple policy applied to the same frozen replay population. Guardrails: infeasible action count, unresolved outcome rate, override rate, latency, and cohort disparity.
Protocol: temporal replay, independent sample audit, capacity stress, and failure injection. Status: not executed. Pass/fail thresholds, sample size, and uncertainty method require a local owner before testing.
Validity threat: policy changes eligibility and exposure, so a post-policy observed rate may not be comparable to baseline without a designed experiment.
Takeaway: the decision remains bounded by evidence, ownership, and recoverability.
## 10. Action Taken
No action taken as of 2026-11-09: this is an illustrative curriculum exercise.
Proposed action: approve a reversible shadow replay only after schema, access, operator, and recovery owners accept the protocol.
Stop authority: the operations lead may disable the constrained path if feasibility, audit, or guardrail checks fail.
Takeaway: the decision remains bounded by evidence, ownership, and recoverability.
## 11. Outcome & Results
Outcome not yet measured. The only values in this case are constructed teaching arithmetic, not deployment, pilot, or observed results.
Measurement plan: record the frozen population, baseline output, constrained output, action acceptance, and guardrails for a predeclared window; report missingness and uncertainty before deciding.
Takeaway: the decision remains bounded by evidence, ownership, and recoverability.
## 12. Failure Modes & What Went Wrong
F-01: apparent gain from future information. Diagnostic: availability-time audit. Containment: remove field. Prevention: point-in-time replay. Provenance: hypothetical.
F-02: infeasible recommendation from stale capacity. Diagnostic: authoritative version mismatch. Containment: defer. Prevention: freshness gate and recovery test. Provenance: hypothetical.
F-03: queue overload from review surge. Diagnostic: queue age and abandonment. Containment: narrow eligibility. Prevention: capacity simulation. Provenance: hypothetical.
F-04: aggregate improvement hides a harmed cohort. Diagnostic: cohort numerator/denominator ledger. Containment: pause policy. Prevention: declared guardrail and independent review. Provenance: hypothetical.
Takeaway: the decision remains bounded by evidence, ownership, and recoverability.
## 13. Transferable Lesson
Canonical pattern: evaluation-mismatch. General principle: a score or optimizer is useful only when its unit, action, feasibility, and outcome match the owner’s decision.
Preconditions: trustworthy event time, versioned inputs, a safe baseline, and authority to defer. Avoid when no action can change or no outcome can be observed.
Portability test: replay the exact decision contract on a contrasting population; if eligibility or action semantics change, rebuild rather than reuse the metric.
Takeaway: the decision remains bounded by evidence, ownership, and recoverability.
## 14. Connections & Cross-References
Related method: the local cheat sheets define denominators, audit signals, and capacity-aware selection. No external case comparison is asserted.
Open question: what population, review capacity, and outcome maturity would make a local evaluation decision-ready? Owner: domain analytics lead.
### Interview Pack
Takeaway: the decision remains bounded by evidence, ownership, and recoverability.
**Q01 / judgment. Prompt:** Frame the decision grain and baseline.
**Strong answer:** define the action contract, identify what is known at the cutoff, calculate only from the frozen eligible population, and state the safe fallback.
**Follow-up:** halve capacity or delay labels; a strong answer narrows automation or defers rather than inventing confidence.
**Alternatives/red flags/scoring:** a simpler baseline is acceptable when it meets constraints; claiming an illustrative rate is realized impact is a red flag. Score 0–3 for unsupported to defended reasoning.
**Answer support:** Sections 4–12 and the illustrative arithmetic in Section 6.
**Q02 / judgment. Prompt:** Identify the most dangerous leakage path.
**Strong answer:** define the action contract, identify what is known at the cutoff, calculate only from the frozen eligible population, and state the safe fallback.
**Follow-up:** halve capacity or delay labels; a strong answer narrows automation or defers rather than inventing confidence.
**Alternatives/red flags/scoring:** a simpler baseline is acceptable when it meets constraints; claiming an illustrative rate is realized impact is a red flag. Score 0–3 for unsupported to defended reasoning.
**Answer support:** Sections 4–12 and the illustrative arithmetic in Section 6.
**Q03 / judgment. Prompt:** Compute 76/80 and compare it with 72/80.
**Strong answer:** define the action contract, identify what is known at the cutoff, calculate only from the frozen eligible population, and state the safe fallback.
**Follow-up:** halve capacity or delay labels; a strong answer narrows automation or defers rather than inventing confidence.
**Alternatives/red flags/scoring:** a simpler baseline is acceptable when it meets constraints; claiming an illustrative rate is realized impact is a red flag. Score 0–3 for unsupported to defended reasoning.
**Answer support:** Sections 4–12 and the illustrative arithmetic in Section 6.
**Q04 / judgment. Prompt:** Explain why the result is not causal.
**Strong answer:** define the action contract, identify what is known at the cutoff, calculate only from the frozen eligible population, and state the safe fallback.
**Follow-up:** halve capacity or delay labels; a strong answer narrows automation or defers rather than inventing confidence.
**Alternatives/red flags/scoring:** a simpler baseline is acceptable when it meets constraints; claiming an illustrative rate is realized impact is a red flag. Score 0–3 for unsupported to defended reasoning.
**Answer support:** Sections 4–12 and the illustrative arithmetic in Section 6.
**Q05 / judgment. Prompt:** Choose an action when capacity is stale.
**Strong answer:** define the action contract, identify what is known at the cutoff, calculate only from the frozen eligible population, and state the safe fallback.
**Follow-up:** halve capacity or delay labels; a strong answer narrows automation or defers rather than inventing confidence.
**Alternatives/red flags/scoring:** a simpler baseline is acceptable when it meets constraints; claiming an illustrative rate is realized impact is a red flag. Score 0–3 for unsupported to defended reasoning.
**Answer support:** Sections 4–12 and the illustrative arithmetic in Section 6.
**Q06 / judgment. Prompt:** Design an idempotent failure response.
**Strong answer:** define the action contract, identify what is known at the cutoff, calculate only from the frozen eligible population, and state the safe fallback.
**Follow-up:** halve capacity or delay labels; a strong answer narrows automation or defers rather than inventing confidence.
**Alternatives/red flags/scoring:** a simpler baseline is acceptable when it meets constraints; claiming an illustrative rate is realized impact is a red flag. Score 0–3 for unsupported to defended reasoning.
**Answer support:** Sections 4–12 and the illustrative arithmetic in Section 6.
**Q07 / judgment. Prompt:** Select a guardrail that can veto the primary metric.
**Strong answer:** define the action contract, identify what is known at the cutoff, calculate only from the frozen eligible population, and state the safe fallback.
**Follow-up:** halve capacity or delay labels; a strong answer narrows automation or defers rather than inventing confidence.
**Alternatives/red flags/scoring:** a simpler baseline is acceptable when it meets constraints; claiming an illustrative rate is realized impact is a red flag. Score 0–3 for unsupported to defended reasoning.
**Answer support:** Sections 4–12 and the illustrative arithmetic in Section 6.
**Q08 / judgment. Prompt:** Diagnose a rate increase with rising exclusions.
**Strong answer:** define the action contract, identify what is known at the cutoff, calculate only from the frozen eligible population, and state the safe fallback.
**Follow-up:** halve capacity or delay labels; a strong answer narrows automation or defers rather than inventing confidence.
**Alternatives/red flags/scoring:** a simpler baseline is acceptable when it meets constraints; claiming an illustrative rate is realized impact is a red flag. Score 0–3 for unsupported to defended reasoning.
**Answer support:** Sections 4–12 and the illustrative arithmetic in Section 6.
**Q09 / judgment. Prompt:** Change the review-capacity constraint.
**Strong answer:** define the action contract, identify what is known at the cutoff, calculate only from the frozen eligible population, and state the safe fallback.
**Follow-up:** halve capacity or delay labels; a strong answer narrows automation or defers rather than inventing confidence.
**Alternatives/red flags/scoring:** a simpler baseline is acceptable when it meets constraints; claiming an illustrative rate is realized impact is a red flag. Score 0–3 for unsupported to defended reasoning.
**Answer support:** Sections 4–12 and the illustrative arithmetic in Section 6.
**Q10 / judgment. Prompt:** Explain the rollback boundary.
**Strong answer:** define the action contract, identify what is known at the cutoff, calculate only from the frozen eligible population, and state the safe fallback.
**Follow-up:** halve capacity or delay labels; a strong answer narrows automation or defers rather than inventing confidence.
**Alternatives/red flags/scoring:** a simpler baseline is acceptable when it meets constraints; claiming an illustrative rate is realized impact is a red flag. Score 0–3 for unsupported to defended reasoning.
**Answer support:** Sections 4–12 and the illustrative arithmetic in Section 6.
**Q11 / judgment. Prompt:** Specify a point-in-time replay.
**Strong answer:** define the action contract, identify what is known at the cutoff, calculate only from the frozen eligible population, and state the safe fallback.
**Follow-up:** halve capacity or delay labels; a strong answer narrows automation or defers rather than inventing confidence.
**Alternatives/red flags/scoring:** a simpler baseline is acceptable when it meets constraints; claiming an illustrative rate is realized impact is a red flag. Score 0–3 for unsupported to defended reasoning.
**Answer support:** Sections 4–12 and the illustrative arithmetic in Section 6.
**Q12 / judgment. Prompt:** Transfer the design to a contrasting domain.
**Strong answer:** define the action contract, identify what is known at the cutoff, calculate only from the frozen eligible population, and state the safe fallback.
**Follow-up:** halve capacity or delay labels; a strong answer narrows automation or defers rather than inventing confidence.
**Alternatives/red flags/scoring:** a simpler baseline is acceptable when it meets constraints; claiming an illustrative rate is realized impact is a red flag. Score 0–3 for unsupported to defended reasoning.
**Answer support:** Sections 4–12 and the illustrative arithmetic in Section 6.
**Replay control 001: identity reconciliation contract.** Perturbation: missing identifier. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 001: identity reconciliation oracle.** compare immutable-key cardinality; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 001: identity reconciliation failure branch.** If missing identifier makes the action ambiguous, quarantine the record; record the rejected alternative and user-facing reason.
**Replay control 001: identity reconciliation recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 001: identity reconciliation interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 002: timezone normalization contract.** Perturbation: conflicting timestamps. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 002: timezone normalization oracle.** convert all source times to UTC; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 002: timezone normalization failure branch.** If conflicting timestamps makes the action ambiguous, defer to baseline; record the rejected alternative and user-facing reason.
**Replay control 002: timezone normalization recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 002: timezone normalization interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 003: late-event watermark contract.** Perturbation: delayed source feed. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 003: late-event watermark oracle.** replay with a closed watermark; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 003: late-event watermark failure branch.** If delayed source feed makes the action ambiguous, freeze the affected window; record the rejected alternative and user-facing reason.
**Replay control 003: late-event watermark recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 003: late-event watermark interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 004: eligibility version contract.** Perturbation: policy migration. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 004: eligibility version oracle.** diff rule versions; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 004: eligibility version failure branch.** If policy migration makes the action ambiguous, require owner approval; record the rejected alternative and user-facing reason.
**Replay control 004: eligibility version recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 004: eligibility version interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 005: capacity snapshot contract.** Perturbation: stale availability. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 005: capacity snapshot oracle.** read the authoritative version; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 005: capacity snapshot failure branch.** If stale availability makes the action ambiguous, reject the constrained output; record the rejected alternative and user-facing reason.
**Replay control 005: capacity snapshot recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 005: capacity snapshot interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 006: input availability contract.** Perturbation: post-cutoff field. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 006: input availability oracle.** remove unavailable fields; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 006: input availability failure branch.** If post-cutoff field makes the action ambiguous, remove the feature; record the rejected alternative and user-facing reason.
**Replay control 006: input availability recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 006: input availability interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 007: policy signature contract.** Perturbation: unsigned rule bundle. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 007: policy signature oracle.** verify signature and hash; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 007: policy signature failure branch.** If unsigned rule bundle makes the action ambiguous, roll back the bundle; record the rejected alternative and user-facing reason.
**Replay control 007: policy signature recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 007: policy signature interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 008: outcome maturity contract.** Perturbation: unresolved outcome. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 008: outcome maturity oracle.** stratify by label age; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 008: outcome maturity failure branch.** If unresolved outcome makes the action ambiguous, mark result provisional; record the rejected alternative and user-facing reason.
**Replay control 008: outcome maturity recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 008: outcome maturity interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 009: override ownership contract.** Perturbation: unassigned override. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 009: override ownership oracle.** require named approver; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 009: override ownership failure branch.** If unassigned override makes the action ambiguous, route to review; record the rejected alternative and user-facing reason.
**Replay control 009: override ownership recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 009: override ownership interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 010: duplicate suppression contract.** Perturbation: duplicate delivery. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 010: duplicate suppression oracle.** replay duplicate events; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 010: duplicate suppression failure branch.** If duplicate delivery makes the action ambiguous, deduplicate before action; record the rejected alternative and user-facing reason.
**Replay control 010: duplicate suppression recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 010: duplicate suppression interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 011: cohort assignment contract.** Perturbation: missing identifier. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 011: cohort assignment oracle.** compare immutable-key cardinality; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 011: cohort assignment failure branch.** If missing identifier makes the action ambiguous, quarantine the record; record the rejected alternative and user-facing reason.
**Replay control 011: cohort assignment recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 011: cohort assignment interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 012: fallback selection contract.** Perturbation: conflicting timestamps. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 012: fallback selection oracle.** convert all source times to UTC; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 012: fallback selection failure branch.** If conflicting timestamps makes the action ambiguous, defer to baseline; record the rejected alternative and user-facing reason.
**Replay control 012: fallback selection recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 012: fallback selection interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 013: retry idempotency contract.** Perturbation: delayed source feed. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 013: retry idempotency oracle.** replay with a closed watermark; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 013: retry idempotency failure branch.** If delayed source feed makes the action ambiguous, freeze the affected window; record the rejected alternative and user-facing reason.
**Replay control 013: retry idempotency recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 013: retry idempotency interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 014: quarantine routing contract.** Perturbation: policy migration. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 014: quarantine routing oracle.** diff rule versions; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 014: quarantine routing failure branch.** If policy migration makes the action ambiguous, require owner approval; record the rejected alternative and user-facing reason.
**Replay control 014: quarantine routing recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 014: quarantine routing interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 015: audit retention contract.** Perturbation: stale availability. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 015: audit retention oracle.** read the authoritative version; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 015: audit retention failure branch.** If stale availability makes the action ambiguous, reject the constrained output; record the rejected alternative and user-facing reason.
**Replay control 015: audit retention recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 015: audit retention interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 016: identity reconciliation contract.** Perturbation: post-cutoff field. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 016: identity reconciliation oracle.** remove unavailable fields; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 016: identity reconciliation failure branch.** If post-cutoff field makes the action ambiguous, remove the feature; record the rejected alternative and user-facing reason.
**Replay control 016: identity reconciliation recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 016: identity reconciliation interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 017: timezone normalization contract.** Perturbation: unsigned rule bundle. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 017: timezone normalization oracle.** verify signature and hash; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 017: timezone normalization failure branch.** If unsigned rule bundle makes the action ambiguous, roll back the bundle; record the rejected alternative and user-facing reason.
**Replay control 017: timezone normalization recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 017: timezone normalization interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 018: late-event watermark contract.** Perturbation: unresolved outcome. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 018: late-event watermark oracle.** stratify by label age; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 018: late-event watermark failure branch.** If unresolved outcome makes the action ambiguous, mark result provisional; record the rejected alternative and user-facing reason.
**Replay control 018: late-event watermark recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 018: late-event watermark interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 019: eligibility version contract.** Perturbation: unassigned override. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 019: eligibility version oracle.** require named approver; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 019: eligibility version failure branch.** If unassigned override makes the action ambiguous, route to review; record the rejected alternative and user-facing reason.
**Replay control 019: eligibility version recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 019: eligibility version interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 020: capacity snapshot contract.** Perturbation: duplicate delivery. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 020: capacity snapshot oracle.** replay duplicate events; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 020: capacity snapshot failure branch.** If duplicate delivery makes the action ambiguous, deduplicate before action; record the rejected alternative and user-facing reason.
**Replay control 020: capacity snapshot recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 020: capacity snapshot interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 021: input availability contract.** Perturbation: missing identifier. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 021: input availability oracle.** compare immutable-key cardinality; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 021: input availability failure branch.** If missing identifier makes the action ambiguous, quarantine the record; record the rejected alternative and user-facing reason.
**Replay control 021: input availability recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 021: input availability interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 022: policy signature contract.** Perturbation: conflicting timestamps. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 022: policy signature oracle.** convert all source times to UTC; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 022: policy signature failure branch.** If conflicting timestamps makes the action ambiguous, defer to baseline; record the rejected alternative and user-facing reason.
**Replay control 022: policy signature recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 022: policy signature interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 023: outcome maturity contract.** Perturbation: delayed source feed. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 023: outcome maturity oracle.** replay with a closed watermark; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 023: outcome maturity failure branch.** If delayed source feed makes the action ambiguous, freeze the affected window; record the rejected alternative and user-facing reason.
**Replay control 023: outcome maturity recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 023: outcome maturity interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 024: override ownership contract.** Perturbation: policy migration. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 024: override ownership oracle.** diff rule versions; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 024: override ownership failure branch.** If policy migration makes the action ambiguous, require owner approval; record the rejected alternative and user-facing reason.
**Replay control 024: override ownership recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 024: override ownership interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 025: duplicate suppression contract.** Perturbation: stale availability. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 025: duplicate suppression oracle.** read the authoritative version; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 025: duplicate suppression failure branch.** If stale availability makes the action ambiguous, reject the constrained output; record the rejected alternative and user-facing reason.
**Replay control 025: duplicate suppression recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 025: duplicate suppression interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 026: cohort assignment contract.** Perturbation: post-cutoff field. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 026: cohort assignment oracle.** remove unavailable fields; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 026: cohort assignment failure branch.** If post-cutoff field makes the action ambiguous, remove the feature; record the rejected alternative and user-facing reason.
**Replay control 026: cohort assignment recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 026: cohort assignment interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 027: fallback selection contract.** Perturbation: unsigned rule bundle. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 027: fallback selection oracle.** verify signature and hash; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 027: fallback selection failure branch.** If unsigned rule bundle makes the action ambiguous, roll back the bundle; record the rejected alternative and user-facing reason.
**Replay control 027: fallback selection recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 027: fallback selection interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 028: retry idempotency contract.** Perturbation: unresolved outcome. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 028: retry idempotency oracle.** stratify by label age; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 028: retry idempotency failure branch.** If unresolved outcome makes the action ambiguous, mark result provisional; record the rejected alternative and user-facing reason.
**Replay control 028: retry idempotency recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 028: retry idempotency interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 029: quarantine routing contract.** Perturbation: unassigned override. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 029: quarantine routing oracle.** require named approver; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 029: quarantine routing failure branch.** If unassigned override makes the action ambiguous, route to review; record the rejected alternative and user-facing reason.
**Replay control 029: quarantine routing recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 029: quarantine routing interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 030: audit retention contract.** Perturbation: duplicate delivery. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 030: audit retention oracle.** replay duplicate events; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 030: audit retention failure branch.** If duplicate delivery makes the action ambiguous, deduplicate before action; record the rejected alternative and user-facing reason.
**Replay control 030: audit retention recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 030: audit retention interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 031: identity reconciliation contract.** Perturbation: missing identifier. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 031: identity reconciliation oracle.** compare immutable-key cardinality; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 031: identity reconciliation failure branch.** If missing identifier makes the action ambiguous, quarantine the record; record the rejected alternative and user-facing reason.
**Replay control 031: identity reconciliation recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 031: identity reconciliation interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 032: timezone normalization contract.** Perturbation: conflicting timestamps. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 032: timezone normalization oracle.** convert all source times to UTC; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 032: timezone normalization failure branch.** If conflicting timestamps makes the action ambiguous, defer to baseline; record the rejected alternative and user-facing reason.
**Replay control 032: timezone normalization recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 032: timezone normalization interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 033: late-event watermark contract.** Perturbation: delayed source feed. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 033: late-event watermark oracle.** replay with a closed watermark; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 033: late-event watermark failure branch.** If delayed source feed makes the action ambiguous, freeze the affected window; record the rejected alternative and user-facing reason.
**Replay control 033: late-event watermark recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 033: late-event watermark interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 034: eligibility version contract.** Perturbation: policy migration. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 034: eligibility version oracle.** diff rule versions; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 034: eligibility version failure branch.** If policy migration makes the action ambiguous, require owner approval; record the rejected alternative and user-facing reason.
**Replay control 034: eligibility version recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 034: eligibility version interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 035: capacity snapshot contract.** Perturbation: stale availability. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 035: capacity snapshot oracle.** read the authoritative version; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 035: capacity snapshot failure branch.** If stale availability makes the action ambiguous, reject the constrained output; record the rejected alternative and user-facing reason.
**Replay control 035: capacity snapshot recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 035: capacity snapshot interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 036: input availability contract.** Perturbation: post-cutoff field. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 036: input availability oracle.** remove unavailable fields; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 036: input availability failure branch.** If post-cutoff field makes the action ambiguous, remove the feature; record the rejected alternative and user-facing reason.
**Replay control 036: input availability recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 036: input availability interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 037: policy signature contract.** Perturbation: unsigned rule bundle. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 037: policy signature oracle.** verify signature and hash; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 037: policy signature failure branch.** If unsigned rule bundle makes the action ambiguous, roll back the bundle; record the rejected alternative and user-facing reason.
**Replay control 037: policy signature recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 037: policy signature interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 038: outcome maturity contract.** Perturbation: unresolved outcome. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 038: outcome maturity oracle.** stratify by label age; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 038: outcome maturity failure branch.** If unresolved outcome makes the action ambiguous, mark result provisional; record the rejected alternative and user-facing reason.
**Replay control 038: outcome maturity recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 038: outcome maturity interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 039: override ownership contract.** Perturbation: unassigned override. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 039: override ownership oracle.** require named approver; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 039: override ownership failure branch.** If unassigned override makes the action ambiguous, route to review; record the rejected alternative and user-facing reason.
**Replay control 039: override ownership recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 039: override ownership interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 040: duplicate suppression contract.** Perturbation: duplicate delivery. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 040: duplicate suppression oracle.** replay duplicate events; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 040: duplicate suppression failure branch.** If duplicate delivery makes the action ambiguous, deduplicate before action; record the rejected alternative and user-facing reason.
**Replay control 040: duplicate suppression recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 040: duplicate suppression interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 041: cohort assignment contract.** Perturbation: missing identifier. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 041: cohort assignment oracle.** compare immutable-key cardinality; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 041: cohort assignment failure branch.** If missing identifier makes the action ambiguous, quarantine the record; record the rejected alternative and user-facing reason.
**Replay control 041: cohort assignment recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 041: cohort assignment interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 042: fallback selection contract.** Perturbation: conflicting timestamps. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 042: fallback selection oracle.** convert all source times to UTC; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 042: fallback selection failure branch.** If conflicting timestamps makes the action ambiguous, defer to baseline; record the rejected alternative and user-facing reason.
**Replay control 042: fallback selection recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 042: fallback selection interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 043: retry idempotency contract.** Perturbation: delayed source feed. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 043: retry idempotency oracle.** replay with a closed watermark; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 043: retry idempotency failure branch.** If delayed source feed makes the action ambiguous, freeze the affected window; record the rejected alternative and user-facing reason.
**Replay control 043: retry idempotency recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 043: retry idempotency interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 044: quarantine routing contract.** Perturbation: policy migration. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 044: quarantine routing oracle.** diff rule versions; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 044: quarantine routing failure branch.** If policy migration makes the action ambiguous, require owner approval; record the rejected alternative and user-facing reason.
**Replay control 044: quarantine routing recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 044: quarantine routing interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 045: audit retention contract.** Perturbation: stale availability. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 045: audit retention oracle.** read the authoritative version; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 045: audit retention failure branch.** If stale availability makes the action ambiguous, reject the constrained output; record the rejected alternative and user-facing reason.
**Replay control 045: audit retention recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 045: audit retention interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 046: identity reconciliation contract.** Perturbation: post-cutoff field. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 046: identity reconciliation oracle.** remove unavailable fields; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 046: identity reconciliation failure branch.** If post-cutoff field makes the action ambiguous, remove the feature; record the rejected alternative and user-facing reason.
**Replay control 046: identity reconciliation recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 046: identity reconciliation interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 047: timezone normalization contract.** Perturbation: unsigned rule bundle. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 047: timezone normalization oracle.** verify signature and hash; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 047: timezone normalization failure branch.** If unsigned rule bundle makes the action ambiguous, roll back the bundle; record the rejected alternative and user-facing reason.
**Replay control 047: timezone normalization recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 047: timezone normalization interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 048: late-event watermark contract.** Perturbation: unresolved outcome. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 048: late-event watermark oracle.** stratify by label age; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 048: late-event watermark failure branch.** If unresolved outcome makes the action ambiguous, mark result provisional; record the rejected alternative and user-facing reason.
**Replay control 048: late-event watermark recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 048: late-event watermark interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 049: eligibility version contract.** Perturbation: unassigned override. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 049: eligibility version oracle.** require named approver; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 049: eligibility version failure branch.** If unassigned override makes the action ambiguous, route to review; record the rejected alternative and user-facing reason.
**Replay control 049: eligibility version recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 049: eligibility version interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 050: capacity snapshot contract.** Perturbation: duplicate delivery. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 050: capacity snapshot oracle.** replay duplicate events; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 050: capacity snapshot failure branch.** If duplicate delivery makes the action ambiguous, deduplicate before action; record the rejected alternative and user-facing reason.
**Replay control 050: capacity snapshot recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 050: capacity snapshot interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 051: input availability contract.** Perturbation: missing identifier. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 051: input availability oracle.** compare immutable-key cardinality; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 051: input availability failure branch.** If missing identifier makes the action ambiguous, quarantine the record; record the rejected alternative and user-facing reason.
**Replay control 051: input availability recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 051: input availability interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 052: policy signature contract.** Perturbation: conflicting timestamps. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 052: policy signature oracle.** convert all source times to UTC; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 052: policy signature failure branch.** If conflicting timestamps makes the action ambiguous, defer to baseline; record the rejected alternative and user-facing reason.
**Replay control 052: policy signature recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 052: policy signature interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 053: outcome maturity contract.** Perturbation: delayed source feed. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 053: outcome maturity oracle.** replay with a closed watermark; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 053: outcome maturity failure branch.** If delayed source feed makes the action ambiguous, freeze the affected window; record the rejected alternative and user-facing reason.
**Replay control 053: outcome maturity recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 053: outcome maturity interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 054: override ownership contract.** Perturbation: policy migration. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 054: override ownership oracle.** diff rule versions; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 054: override ownership failure branch.** If policy migration makes the action ambiguous, require owner approval; record the rejected alternative and user-facing reason.
**Replay control 054: override ownership recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 054: override ownership interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 055: duplicate suppression contract.** Perturbation: stale availability. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 055: duplicate suppression oracle.** read the authoritative version; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 055: duplicate suppression failure branch.** If stale availability makes the action ambiguous, reject the constrained output; record the rejected alternative and user-facing reason.
**Replay control 055: duplicate suppression recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 055: duplicate suppression interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 056: cohort assignment contract.** Perturbation: post-cutoff field. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 056: cohort assignment oracle.** remove unavailable fields; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 056: cohort assignment failure branch.** If post-cutoff field makes the action ambiguous, remove the feature; record the rejected alternative and user-facing reason.
**Replay control 056: cohort assignment recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 056: cohort assignment interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 057: fallback selection contract.** Perturbation: unsigned rule bundle. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 057: fallback selection oracle.** verify signature and hash; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 057: fallback selection failure branch.** If unsigned rule bundle makes the action ambiguous, roll back the bundle; record the rejected alternative and user-facing reason.
**Replay control 057: fallback selection recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 057: fallback selection interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 058: retry idempotency contract.** Perturbation: unresolved outcome. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 058: retry idempotency oracle.** stratify by label age; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 058: retry idempotency failure branch.** If unresolved outcome makes the action ambiguous, mark result provisional; record the rejected alternative and user-facing reason.
**Replay control 058: retry idempotency recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 058: retry idempotency interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 059: quarantine routing contract.** Perturbation: unassigned override. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 059: quarantine routing oracle.** require named approver; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 059: quarantine routing failure branch.** If unassigned override makes the action ambiguous, route to review; record the rejected alternative and user-facing reason.
**Replay control 059: quarantine routing recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 059: quarantine routing interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 060: audit retention contract.** Perturbation: duplicate delivery. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 060: audit retention oracle.** replay duplicate events; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 060: audit retention failure branch.** If duplicate delivery makes the action ambiguous, deduplicate before action; record the rejected alternative and user-facing reason.
**Replay control 060: audit retention recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 060: audit retention interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 061: identity reconciliation contract.** Perturbation: missing identifier. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 061: identity reconciliation oracle.** compare immutable-key cardinality; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 061: identity reconciliation failure branch.** If missing identifier makes the action ambiguous, quarantine the record; record the rejected alternative and user-facing reason.
**Replay control 061: identity reconciliation recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 061: identity reconciliation interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 062: timezone normalization contract.** Perturbation: conflicting timestamps. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 062: timezone normalization oracle.** convert all source times to UTC; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 062: timezone normalization failure branch.** If conflicting timestamps makes the action ambiguous, defer to baseline; record the rejected alternative and user-facing reason.
**Replay control 062: timezone normalization recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 062: timezone normalization interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 063: late-event watermark contract.** Perturbation: delayed source feed. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 063: late-event watermark oracle.** replay with a closed watermark; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 063: late-event watermark failure branch.** If delayed source feed makes the action ambiguous, freeze the affected window; record the rejected alternative and user-facing reason.
**Replay control 063: late-event watermark recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 063: late-event watermark interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 064: eligibility version contract.** Perturbation: policy migration. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 064: eligibility version oracle.** diff rule versions; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 064: eligibility version failure branch.** If policy migration makes the action ambiguous, require owner approval; record the rejected alternative and user-facing reason.
**Replay control 064: eligibility version recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 064: eligibility version interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 065: capacity snapshot contract.** Perturbation: stale availability. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 065: capacity snapshot oracle.** read the authoritative version; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 065: capacity snapshot failure branch.** If stale availability makes the action ambiguous, reject the constrained output; record the rejected alternative and user-facing reason.
**Replay control 065: capacity snapshot recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 065: capacity snapshot interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 066: input availability contract.** Perturbation: post-cutoff field. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 066: input availability oracle.** remove unavailable fields; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 066: input availability failure branch.** If post-cutoff field makes the action ambiguous, remove the feature; record the rejected alternative and user-facing reason.
**Replay control 066: input availability recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 066: input availability interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 067: policy signature contract.** Perturbation: unsigned rule bundle. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 067: policy signature oracle.** verify signature and hash; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 067: policy signature failure branch.** If unsigned rule bundle makes the action ambiguous, roll back the bundle; record the rejected alternative and user-facing reason.
**Replay control 067: policy signature recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 067: policy signature interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 068: outcome maturity contract.** Perturbation: unresolved outcome. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 068: outcome maturity oracle.** stratify by label age; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 068: outcome maturity failure branch.** If unresolved outcome makes the action ambiguous, mark result provisional; record the rejected alternative and user-facing reason.
**Replay control 068: outcome maturity recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 068: outcome maturity interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 069: override ownership contract.** Perturbation: unassigned override. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 069: override ownership oracle.** require named approver; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 069: override ownership failure branch.** If unassigned override makes the action ambiguous, route to review; record the rejected alternative and user-facing reason.
**Replay control 069: override ownership recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 069: override ownership interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 070: duplicate suppression contract.** Perturbation: duplicate delivery. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 070: duplicate suppression oracle.** replay duplicate events; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 070: duplicate suppression failure branch.** If duplicate delivery makes the action ambiguous, deduplicate before action; record the rejected alternative and user-facing reason.
**Replay control 070: duplicate suppression recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 070: duplicate suppression interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 071: cohort assignment contract.** Perturbation: missing identifier. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 071: cohort assignment oracle.** compare immutable-key cardinality; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 071: cohort assignment failure branch.** If missing identifier makes the action ambiguous, quarantine the record; record the rejected alternative and user-facing reason.
**Replay control 071: cohort assignment recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 071: cohort assignment interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 072: fallback selection contract.** Perturbation: conflicting timestamps. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 072: fallback selection oracle.** convert all source times to UTC; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 072: fallback selection failure branch.** If conflicting timestamps makes the action ambiguous, defer to baseline; record the rejected alternative and user-facing reason.
**Replay control 072: fallback selection recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 072: fallback selection interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 073: retry idempotency contract.** Perturbation: delayed source feed. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 073: retry idempotency oracle.** replay with a closed watermark; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 073: retry idempotency failure branch.** If delayed source feed makes the action ambiguous, freeze the affected window; record the rejected alternative and user-facing reason.
**Replay control 073: retry idempotency recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 073: retry idempotency interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 074: quarantine routing contract.** Perturbation: policy migration. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 074: quarantine routing oracle.** diff rule versions; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 074: quarantine routing failure branch.** If policy migration makes the action ambiguous, require owner approval; record the rejected alternative and user-facing reason.
**Replay control 074: quarantine routing recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 074: quarantine routing interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 075: audit retention contract.** Perturbation: stale availability. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 075: audit retention oracle.** read the authoritative version; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 075: audit retention failure branch.** If stale availability makes the action ambiguous, reject the constrained output; record the rejected alternative and user-facing reason.
**Replay control 075: audit retention recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 075: audit retention interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 076: identity reconciliation contract.** Perturbation: post-cutoff field. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 076: identity reconciliation oracle.** remove unavailable fields; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 076: identity reconciliation failure branch.** If post-cutoff field makes the action ambiguous, remove the feature; record the rejected alternative and user-facing reason.
**Replay control 076: identity reconciliation recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 076: identity reconciliation interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 077: timezone normalization contract.** Perturbation: unsigned rule bundle. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 077: timezone normalization oracle.** verify signature and hash; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 077: timezone normalization failure branch.** If unsigned rule bundle makes the action ambiguous, roll back the bundle; record the rejected alternative and user-facing reason.
**Replay control 077: timezone normalization recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 077: timezone normalization interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 078: late-event watermark contract.** Perturbation: unresolved outcome. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 078: late-event watermark oracle.** stratify by label age; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 078: late-event watermark failure branch.** If unresolved outcome makes the action ambiguous, mark result provisional; record the rejected alternative and user-facing reason.
**Replay control 078: late-event watermark recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 078: late-event watermark interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 079: eligibility version contract.** Perturbation: unassigned override. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 079: eligibility version oracle.** require named approver; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 079: eligibility version failure branch.** If unassigned override makes the action ambiguous, route to review; record the rejected alternative and user-facing reason.
**Replay control 079: eligibility version recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 079: eligibility version interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 080: capacity snapshot contract.** Perturbation: duplicate delivery. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 080: capacity snapshot oracle.** replay duplicate events; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 080: capacity snapshot failure branch.** If duplicate delivery makes the action ambiguous, deduplicate before action; record the rejected alternative and user-facing reason.
**Replay control 080: capacity snapshot recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 080: capacity snapshot interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 081: input availability contract.** Perturbation: missing identifier. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 081: input availability oracle.** compare immutable-key cardinality; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 081: input availability failure branch.** If missing identifier makes the action ambiguous, quarantine the record; record the rejected alternative and user-facing reason.
**Replay control 081: input availability recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 081: input availability interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 082: policy signature contract.** Perturbation: conflicting timestamps. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 082: policy signature oracle.** convert all source times to UTC; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 082: policy signature failure branch.** If conflicting timestamps makes the action ambiguous, defer to baseline; record the rejected alternative and user-facing reason.
**Replay control 082: policy signature recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 082: policy signature interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 083: outcome maturity contract.** Perturbation: delayed source feed. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 083: outcome maturity oracle.** replay with a closed watermark; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 083: outcome maturity failure branch.** If delayed source feed makes the action ambiguous, freeze the affected window; record the rejected alternative and user-facing reason.
**Replay control 083: outcome maturity recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 083: outcome maturity interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 084: override ownership contract.** Perturbation: policy migration. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 084: override ownership oracle.** diff rule versions; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 084: override ownership failure branch.** If policy migration makes the action ambiguous, require owner approval; record the rejected alternative and user-facing reason.
**Replay control 084: override ownership recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 084: override ownership interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 085: duplicate suppression contract.** Perturbation: stale availability. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 085: duplicate suppression oracle.** read the authoritative version; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 085: duplicate suppression failure branch.** If stale availability makes the action ambiguous, reject the constrained output; record the rejected alternative and user-facing reason.
**Replay control 085: duplicate suppression recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 085: duplicate suppression interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 086: cohort assignment contract.** Perturbation: post-cutoff field. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 086: cohort assignment oracle.** remove unavailable fields; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 086: cohort assignment failure branch.** If post-cutoff field makes the action ambiguous, remove the feature; record the rejected alternative and user-facing reason.
**Replay control 086: cohort assignment recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 086: cohort assignment interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 087: fallback selection contract.** Perturbation: unsigned rule bundle. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 087: fallback selection oracle.** verify signature and hash; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 087: fallback selection failure branch.** If unsigned rule bundle makes the action ambiguous, roll back the bundle; record the rejected alternative and user-facing reason.
**Replay control 087: fallback selection recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 087: fallback selection interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 088: retry idempotency contract.** Perturbation: unresolved outcome. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 088: retry idempotency oracle.** stratify by label age; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 088: retry idempotency failure branch.** If unresolved outcome makes the action ambiguous, mark result provisional; record the rejected alternative and user-facing reason.
**Replay control 088: retry idempotency recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 088: retry idempotency interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 089: quarantine routing contract.** Perturbation: unassigned override. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 089: quarantine routing oracle.** require named approver; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 089: quarantine routing failure branch.** If unassigned override makes the action ambiguous, route to review; record the rejected alternative and user-facing reason.
**Replay control 089: quarantine routing recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 089: quarantine routing interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 090: audit retention contract.** Perturbation: duplicate delivery. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 090: audit retention oracle.** replay duplicate events; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 090: audit retention failure branch.** If duplicate delivery makes the action ambiguous, deduplicate before action; record the rejected alternative and user-facing reason.
**Replay control 090: audit retention recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 090: audit retention interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 091: identity reconciliation contract.** Perturbation: missing identifier. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 091: identity reconciliation oracle.** compare immutable-key cardinality; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 091: identity reconciliation failure branch.** If missing identifier makes the action ambiguous, quarantine the record; record the rejected alternative and user-facing reason.
**Replay control 091: identity reconciliation recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 091: identity reconciliation interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 092: timezone normalization contract.** Perturbation: conflicting timestamps. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 092: timezone normalization oracle.** convert all source times to UTC; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 092: timezone normalization failure branch.** If conflicting timestamps makes the action ambiguous, defer to baseline; record the rejected alternative and user-facing reason.
**Replay control 092: timezone normalization recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 092: timezone normalization interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 093: late-event watermark contract.** Perturbation: delayed source feed. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 093: late-event watermark oracle.** replay with a closed watermark; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 093: late-event watermark failure branch.** If delayed source feed makes the action ambiguous, freeze the affected window; record the rejected alternative and user-facing reason.
**Replay control 093: late-event watermark recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 093: late-event watermark interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 094: eligibility version contract.** Perturbation: policy migration. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 094: eligibility version oracle.** diff rule versions; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 094: eligibility version failure branch.** If policy migration makes the action ambiguous, require owner approval; record the rejected alternative and user-facing reason.
**Replay control 094: eligibility version recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 094: eligibility version interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 095: capacity snapshot contract.** Perturbation: stale availability. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 095: capacity snapshot oracle.** read the authoritative version; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 095: capacity snapshot failure branch.** If stale availability makes the action ambiguous, reject the constrained output; record the rejected alternative and user-facing reason.
**Replay control 095: capacity snapshot recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 095: capacity snapshot interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 096: input availability contract.** Perturbation: post-cutoff field. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 096: input availability oracle.** remove unavailable fields; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 096: input availability failure branch.** If post-cutoff field makes the action ambiguous, remove the feature; record the rejected alternative and user-facing reason.
**Replay control 096: input availability recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 096: input availability interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 097: policy signature contract.** Perturbation: unsigned rule bundle. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 097: policy signature oracle.** verify signature and hash; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 097: policy signature failure branch.** If unsigned rule bundle makes the action ambiguous, roll back the bundle; record the rejected alternative and user-facing reason.
**Replay control 097: policy signature recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 097: policy signature interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 098: outcome maturity contract.** Perturbation: unresolved outcome. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 098: outcome maturity oracle.** stratify by label age; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 098: outcome maturity failure branch.** If unresolved outcome makes the action ambiguous, mark result provisional; record the rejected alternative and user-facing reason.
**Replay control 098: outcome maturity recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 098: outcome maturity interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 099: override ownership contract.** Perturbation: unassigned override. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 099: override ownership oracle.** require named approver; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 099: override ownership failure branch.** If unassigned override makes the action ambiguous, route to review; record the rejected alternative and user-facing reason.
**Replay control 099: override ownership recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 099: override ownership interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 100: duplicate suppression contract.** Perturbation: duplicate delivery. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 100: duplicate suppression oracle.** replay duplicate events; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 100: duplicate suppression failure branch.** If duplicate delivery makes the action ambiguous, deduplicate before action; record the rejected alternative and user-facing reason.
**Replay control 100: duplicate suppression recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 100: duplicate suppression interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 101: cohort assignment contract.** Perturbation: missing identifier. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 101: cohort assignment oracle.** compare immutable-key cardinality; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 101: cohort assignment failure branch.** If missing identifier makes the action ambiguous, quarantine the record; record the rejected alternative and user-facing reason.
**Replay control 101: cohort assignment recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 101: cohort assignment interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 102: fallback selection contract.** Perturbation: conflicting timestamps. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 102: fallback selection oracle.** convert all source times to UTC; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 102: fallback selection failure branch.** If conflicting timestamps makes the action ambiguous, defer to baseline; record the rejected alternative and user-facing reason.
**Replay control 102: fallback selection recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 102: fallback selection interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 103: retry idempotency contract.** Perturbation: delayed source feed. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 103: retry idempotency oracle.** replay with a closed watermark; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 103: retry idempotency failure branch.** If delayed source feed makes the action ambiguous, freeze the affected window; record the rejected alternative and user-facing reason.
**Replay control 103: retry idempotency recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 103: retry idempotency interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 104: quarantine routing contract.** Perturbation: policy migration. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 104: quarantine routing oracle.** diff rule versions; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 104: quarantine routing failure branch.** If policy migration makes the action ambiguous, require owner approval; record the rejected alternative and user-facing reason.
**Replay control 104: quarantine routing recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 104: quarantine routing interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 105: audit retention contract.** Perturbation: stale availability. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 105: audit retention oracle.** read the authoritative version; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 105: audit retention failure branch.** If stale availability makes the action ambiguous, reject the constrained output; record the rejected alternative and user-facing reason.
**Replay control 105: audit retention recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 105: audit retention interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 106: identity reconciliation contract.** Perturbation: post-cutoff field. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 106: identity reconciliation oracle.** remove unavailable fields; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 106: identity reconciliation failure branch.** If post-cutoff field makes the action ambiguous, remove the feature; record the rejected alternative and user-facing reason.
**Replay control 106: identity reconciliation recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 106: identity reconciliation interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 107: timezone normalization contract.** Perturbation: unsigned rule bundle. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 107: timezone normalization oracle.** verify signature and hash; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 107: timezone normalization failure branch.** If unsigned rule bundle makes the action ambiguous, roll back the bundle; record the rejected alternative and user-facing reason.
**Replay control 107: timezone normalization recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 107: timezone normalization interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 108: late-event watermark contract.** Perturbation: unresolved outcome. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 108: late-event watermark oracle.** stratify by label age; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 108: late-event watermark failure branch.** If unresolved outcome makes the action ambiguous, mark result provisional; record the rejected alternative and user-facing reason.
**Replay control 108: late-event watermark recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 108: late-event watermark interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 109: eligibility version contract.** Perturbation: unassigned override. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 109: eligibility version oracle.** require named approver; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 109: eligibility version failure branch.** If unassigned override makes the action ambiguous, route to review; record the rejected alternative and user-facing reason.
**Replay control 109: eligibility version recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 109: eligibility version interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 110: capacity snapshot contract.** Perturbation: duplicate delivery. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 110: capacity snapshot oracle.** replay duplicate events; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 110: capacity snapshot failure branch.** If duplicate delivery makes the action ambiguous, deduplicate before action; record the rejected alternative and user-facing reason.
**Replay control 110: capacity snapshot recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 110: capacity snapshot interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 111: input availability contract.** Perturbation: missing identifier. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 111: input availability oracle.** compare immutable-key cardinality; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 111: input availability failure branch.** If missing identifier makes the action ambiguous, quarantine the record; record the rejected alternative and user-facing reason.
**Replay control 111: input availability recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 111: input availability interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 112: policy signature contract.** Perturbation: conflicting timestamps. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 112: policy signature oracle.** convert all source times to UTC; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 112: policy signature failure branch.** If conflicting timestamps makes the action ambiguous, defer to baseline; record the rejected alternative and user-facing reason.
**Replay control 112: policy signature recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 112: policy signature interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 113: outcome maturity contract.** Perturbation: delayed source feed. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 113: outcome maturity oracle.** replay with a closed watermark; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 113: outcome maturity failure branch.** If delayed source feed makes the action ambiguous, freeze the affected window; record the rejected alternative and user-facing reason.
**Replay control 113: outcome maturity recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 113: outcome maturity interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 114: override ownership contract.** Perturbation: policy migration. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 114: override ownership oracle.** diff rule versions; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 114: override ownership failure branch.** If policy migration makes the action ambiguous, require owner approval; record the rejected alternative and user-facing reason.
**Replay control 114: override ownership recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 114: override ownership interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 115: duplicate suppression contract.** Perturbation: stale availability. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 115: duplicate suppression oracle.** read the authoritative version; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 115: duplicate suppression failure branch.** If stale availability makes the action ambiguous, reject the constrained output; record the rejected alternative and user-facing reason.
**Replay control 115: duplicate suppression recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 115: duplicate suppression interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 116: cohort assignment contract.** Perturbation: post-cutoff field. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 116: cohort assignment oracle.** remove unavailable fields; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 116: cohort assignment failure branch.** If post-cutoff field makes the action ambiguous, remove the feature; record the rejected alternative and user-facing reason.
**Replay control 116: cohort assignment recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 116: cohort assignment interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 117: fallback selection contract.** Perturbation: unsigned rule bundle. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 117: fallback selection oracle.** verify signature and hash; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 117: fallback selection failure branch.** If unsigned rule bundle makes the action ambiguous, roll back the bundle; record the rejected alternative and user-facing reason.
**Replay control 117: fallback selection recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 117: fallback selection interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 118: retry idempotency contract.** Perturbation: unresolved outcome. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 118: retry idempotency oracle.** stratify by label age; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 118: retry idempotency failure branch.** If unresolved outcome makes the action ambiguous, mark result provisional; record the rejected alternative and user-facing reason.
**Replay control 118: retry idempotency recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 118: retry idempotency interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 119: quarantine routing contract.** Perturbation: unassigned override. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 119: quarantine routing oracle.** require named approver; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 119: quarantine routing failure branch.** If unassigned override makes the action ambiguous, route to review; record the rejected alternative and user-facing reason.
**Replay control 119: quarantine routing recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 119: quarantine routing interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 120: audit retention contract.** Perturbation: duplicate delivery. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 120: audit retention oracle.** replay duplicate events; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 120: audit retention failure branch.** If duplicate delivery makes the action ambiguous, deduplicate before action; record the rejected alternative and user-facing reason.
**Replay control 120: audit retention recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 120: audit retention interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 121: identity reconciliation contract.** Perturbation: missing identifier. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 121: identity reconciliation oracle.** compare immutable-key cardinality; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 121: identity reconciliation failure branch.** If missing identifier makes the action ambiguous, quarantine the record; record the rejected alternative and user-facing reason.
**Replay control 121: identity reconciliation recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 121: identity reconciliation interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 122: timezone normalization contract.** Perturbation: conflicting timestamps. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 122: timezone normalization oracle.** convert all source times to UTC; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 122: timezone normalization failure branch.** If conflicting timestamps makes the action ambiguous, defer to baseline; record the rejected alternative and user-facing reason.
**Replay control 122: timezone normalization recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 122: timezone normalization interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 123: late-event watermark contract.** Perturbation: delayed source feed. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 123: late-event watermark oracle.** replay with a closed watermark; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 123: late-event watermark failure branch.** If delayed source feed makes the action ambiguous, freeze the affected window; record the rejected alternative and user-facing reason.
**Replay control 123: late-event watermark recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 123: late-event watermark interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 124: eligibility version contract.** Perturbation: policy migration. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 124: eligibility version oracle.** diff rule versions; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 124: eligibility version failure branch.** If policy migration makes the action ambiguous, require owner approval; record the rejected alternative and user-facing reason.
**Replay control 124: eligibility version recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 124: eligibility version interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 125: capacity snapshot contract.** Perturbation: stale availability. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 125: capacity snapshot oracle.** read the authoritative version; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 125: capacity snapshot failure branch.** If stale availability makes the action ambiguous, reject the constrained output; record the rejected alternative and user-facing reason.
**Replay control 125: capacity snapshot recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 125: capacity snapshot interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 126: input availability contract.** Perturbation: post-cutoff field. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 126: input availability oracle.** remove unavailable fields; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 126: input availability failure branch.** If post-cutoff field makes the action ambiguous, remove the feature; record the rejected alternative and user-facing reason.
**Replay control 126: input availability recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 126: input availability interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 127: policy signature contract.** Perturbation: unsigned rule bundle. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 127: policy signature oracle.** verify signature and hash; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 127: policy signature failure branch.** If unsigned rule bundle makes the action ambiguous, roll back the bundle; record the rejected alternative and user-facing reason.
**Replay control 127: policy signature recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 127: policy signature interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 128: outcome maturity contract.** Perturbation: unresolved outcome. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 128: outcome maturity oracle.** stratify by label age; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 128: outcome maturity failure branch.** If unresolved outcome makes the action ambiguous, mark result provisional; record the rejected alternative and user-facing reason.
**Replay control 128: outcome maturity recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 128: outcome maturity interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 129: override ownership contract.** Perturbation: unassigned override. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 129: override ownership oracle.** require named approver; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 129: override ownership failure branch.** If unassigned override makes the action ambiguous, route to review; record the rejected alternative and user-facing reason.
**Replay control 129: override ownership recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 129: override ownership interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 130: duplicate suppression contract.** Perturbation: duplicate delivery. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 130: duplicate suppression oracle.** replay duplicate events; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 130: duplicate suppression failure branch.** If duplicate delivery makes the action ambiguous, deduplicate before action; record the rejected alternative and user-facing reason.
**Replay control 130: duplicate suppression recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 130: duplicate suppression interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 131: cohort assignment contract.** Perturbation: missing identifier. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 131: cohort assignment oracle.** compare immutable-key cardinality; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 131: cohort assignment failure branch.** If missing identifier makes the action ambiguous, quarantine the record; record the rejected alternative and user-facing reason.
**Replay control 131: cohort assignment recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 131: cohort assignment interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 132: fallback selection contract.** Perturbation: conflicting timestamps. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 132: fallback selection oracle.** convert all source times to UTC; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 132: fallback selection failure branch.** If conflicting timestamps makes the action ambiguous, defer to baseline; record the rejected alternative and user-facing reason.
**Replay control 132: fallback selection recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 132: fallback selection interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 133: retry idempotency contract.** Perturbation: delayed source feed. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 133: retry idempotency oracle.** replay with a closed watermark; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 133: retry idempotency failure branch.** If delayed source feed makes the action ambiguous, freeze the affected window; record the rejected alternative and user-facing reason.
**Replay control 133: retry idempotency recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 133: retry idempotency interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 134: quarantine routing contract.** Perturbation: policy migration. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 134: quarantine routing oracle.** diff rule versions; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 134: quarantine routing failure branch.** If policy migration makes the action ambiguous, require owner approval; record the rejected alternative and user-facing reason.
**Replay control 134: quarantine routing recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 134: quarantine routing interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 135: audit retention contract.** Perturbation: stale availability. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 135: audit retention oracle.** read the authoritative version; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 135: audit retention failure branch.** If stale availability makes the action ambiguous, reject the constrained output; record the rejected alternative and user-facing reason.
**Replay control 135: audit retention recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 135: audit retention interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 136: identity reconciliation contract.** Perturbation: post-cutoff field. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 136: identity reconciliation oracle.** remove unavailable fields; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 136: identity reconciliation failure branch.** If post-cutoff field makes the action ambiguous, remove the feature; record the rejected alternative and user-facing reason.
**Replay control 136: identity reconciliation recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 136: identity reconciliation interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 137: timezone normalization contract.** Perturbation: unsigned rule bundle. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 137: timezone normalization oracle.** verify signature and hash; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 137: timezone normalization failure branch.** If unsigned rule bundle makes the action ambiguous, roll back the bundle; record the rejected alternative and user-facing reason.
**Replay control 137: timezone normalization recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 137: timezone normalization interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 138: late-event watermark contract.** Perturbation: unresolved outcome. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 138: late-event watermark oracle.** stratify by label age; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 138: late-event watermark failure branch.** If unresolved outcome makes the action ambiguous, mark result provisional; record the rejected alternative and user-facing reason.
**Replay control 138: late-event watermark recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 138: late-event watermark interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 139: eligibility version contract.** Perturbation: unassigned override. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 139: eligibility version oracle.** require named approver; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 139: eligibility version failure branch.** If unassigned override makes the action ambiguous, route to review; record the rejected alternative and user-facing reason.
**Replay control 139: eligibility version recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 139: eligibility version interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 140: capacity snapshot contract.** Perturbation: duplicate delivery. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 140: capacity snapshot oracle.** replay duplicate events; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 140: capacity snapshot failure branch.** If duplicate delivery makes the action ambiguous, deduplicate before action; record the rejected alternative and user-facing reason.
**Replay control 140: capacity snapshot recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 140: capacity snapshot interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 141: input availability contract.** Perturbation: missing identifier. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 141: input availability oracle.** compare immutable-key cardinality; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 141: input availability failure branch.** If missing identifier makes the action ambiguous, quarantine the record; record the rejected alternative and user-facing reason.
**Replay control 141: input availability recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 141: input availability interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 142: policy signature contract.** Perturbation: conflicting timestamps. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 142: policy signature oracle.** convert all source times to UTC; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 142: policy signature failure branch.** If conflicting timestamps makes the action ambiguous, defer to baseline; record the rejected alternative and user-facing reason.
**Replay control 142: policy signature recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 142: policy signature interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 143: outcome maturity contract.** Perturbation: delayed source feed. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 143: outcome maturity oracle.** replay with a closed watermark; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 143: outcome maturity failure branch.** If delayed source feed makes the action ambiguous, freeze the affected window; record the rejected alternative and user-facing reason.
**Replay control 143: outcome maturity recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 143: outcome maturity interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 144: override ownership contract.** Perturbation: policy migration. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 144: override ownership oracle.** diff rule versions; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 144: override ownership failure branch.** If policy migration makes the action ambiguous, require owner approval; record the rejected alternative and user-facing reason.
**Replay control 144: override ownership recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 144: override ownership interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 145: duplicate suppression contract.** Perturbation: stale availability. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 145: duplicate suppression oracle.** read the authoritative version; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 145: duplicate suppression failure branch.** If stale availability makes the action ambiguous, reject the constrained output; record the rejected alternative and user-facing reason.
**Replay control 145: duplicate suppression recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 145: duplicate suppression interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 146: cohort assignment contract.** Perturbation: post-cutoff field. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 146: cohort assignment oracle.** remove unavailable fields; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 146: cohort assignment failure branch.** If post-cutoff field makes the action ambiguous, remove the feature; record the rejected alternative and user-facing reason.
**Replay control 146: cohort assignment recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 146: cohort assignment interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 147: fallback selection contract.** Perturbation: unsigned rule bundle. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 147: fallback selection oracle.** verify signature and hash; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 147: fallback selection failure branch.** If unsigned rule bundle makes the action ambiguous, roll back the bundle; record the rejected alternative and user-facing reason.
**Replay control 147: fallback selection recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 147: fallback selection interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 148: retry idempotency contract.** Perturbation: unresolved outcome. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 148: retry idempotency oracle.** stratify by label age; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 148: retry idempotency failure branch.** If unresolved outcome makes the action ambiguous, mark result provisional; record the rejected alternative and user-facing reason.
**Replay control 148: retry idempotency recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 148: retry idempotency interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 149: quarantine routing contract.** Perturbation: unassigned override. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 149: quarantine routing oracle.** require named approver; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 149: quarantine routing failure branch.** If unassigned override makes the action ambiguous, route to review; record the rejected alternative and user-facing reason.
**Replay control 149: quarantine routing recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 149: quarantine routing interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
**Replay control 150: audit retention contract.** Perturbation: duplicate delivery. Preserve the raw field, provenance, decision cutoff, and version so a reviewer can distinguish source correction from policy change.
**Replay control 150: audit retention oracle.** replay duplicate events; the expected signal is an explainable match or a quarantined mismatch, never silent coercion.
**Replay control 150: audit retention failure branch.** If duplicate delivery makes the action ambiguous, deduplicate before action; record the rejected alternative and user-facing reason.
**Replay control 150: audit retention recovery.** Restore from the last compatible policy and ledger snapshot, reconcile affected decisions, and test that replay cannot create a second effect.
**Replay control 150: audit retention interpretation.** This drill tests a stated contract rather than production performance; it identifies the measurement or approval evidence required before promotion.
## 15. Quick Reference Card
| Domain | Mobility, travel and hospitality illustrative decision exercise |
| Problem type | Flight disruption rebooking: bounded feasible rebooking proposal decision |
| Method | Versioned baseline plus feasibility gate, constrained alternative, and review fallback |
| Primary result | Not measured; 76/80 versus 72/80 is illustrative arithmetic only |
| Key lesson | Align event grain, action, capacity, and outcome before optimization |
| Tags | evaluation-mismatch; distribution-shift; human-in-the-loop |
| Case ID / version / as-of | CASE-13-02-002 / 1.0 / 2026-11-09 |
| Evidence level | illustrative; no named organization or deployment |
| Implementation stage | proposed |
| Baseline / primary metric | first available seat; qualified eligible outcome rate |
| Key limitation | No local data, source artifact, or executed validation |
| Review trigger / owner | Policy or event contract change / domain analytics owner |
