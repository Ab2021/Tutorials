---
case_id: "CASE-01-001"
case_title: "Select an AI use-case portfolio under review-capacity constraints"
case_version: "1.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "AI product portfolio planning"
primary_pattern: "evidence-to-operation decision"
secondary_patterns: ["human-in-the-loop"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-09"
owner_role: "portfolio steering owner"
source_index: "INDEX.md"
review_trigger: "Change in workload, harm estimate, review capacity, or proposed data boundary"
---

# Select an AI Use-Case Portfolio Under Review-Capacity Constraints

## 1. The One-Line Framing

Given three proposed workflow interventions scored at the request level, decide which can enter a limited pilot for a portfolio owner, subject to a 240-review/day capacity and a no-unapproved-side-effect constraint, compared with funding by enthusiasm. **Illustrative design exercise:** no organization, deployment, or measured outcome is asserted. The reusable pattern is evidence-to-operation decision; the surprising tradeoff is that a higher estimated gross benefit can be infeasible because review demand exceeds capacity.

## 2. Context & Background

A fictional operations group receives 2,000 eligible requests/day. It currently uses human handling, with no validated automation baseline. The portfolio owner must choose among: knowledge-answer drafting (A), document triage (B), and account-change execution (C). Users are operations staff; reviewers own release and escalation. The scope is one 30-day pilot decision, not vendor selection or autonomous deployment. Values below are constructed teaching inputs. The prior process persists because it provides authorization and correction, even if it is slow.

## 3. The Business Question

| Contract field | Decision |
|---|---|
| Question | Which proposals merit a reversible pilot without exceeding review capacity or allowing unapproved state changes? |
| Owner | portfolio steering owner |
| Trigger | budget cycle and backlog concern |
| Actions | pilot A, pilot B, pilot C, fund none, or collect evidence |
| Objective | maximize illustrative net value while satisfying hard gates |
| Hard constraints | <=240 reviews/day; no irreversible autonomous effect; authorized data only |
| Scope | 2,000 eligible requests/day for 30 illustrative days |
| Out of scope | production ROI, legal determination, vendor procurement |

A higher quality proxy does not equal business value if reviewers cannot process routed work or the action is not reversible.

## 4. Assumptions & Constraints

| ID | Assumption/constraint | Type | Status | Impact if wrong | Fallback |
|---|---|---|---|---|---|
| A-01 | 2,000 eligible requests/day | workload | illustrative | capacity calculation changes | re-size from observed arrivals |
| A-02 | review policy routes stated fraction | operational | unverified | queue grows or harms escape | immutable routing log and manual queue |
| A-03 | per-unit benefit/harm estimates are comparable | economics | illustrative | ranking reverses | measure baseline and pilot effects |
| C-01 | review capacity is 240/day | capacity | illustrative | pilot infeasible | lower intake or staff reviewers |
| C-02 | state-changing actions require approval | safety | required design constraint | unacceptable effect | propose-only mode |

The most decision-reversing assumptions are review capacity, routing fraction, and the validity of benefit estimates.

## 5. Data, Tools & Method

### 5a. Data Specification

The unit is one eligible request/day, keyed by an illustrative request ID. Inputs are proposal estimates, routed fraction, and action reversibility. There is no training data or organization data. A future pilot must record arrival time, decision, reviewer completion, correction, source authorization, and side-effect ID; exclude ineligible and duplicate requests with a declared rule.

### 5b. Tools & Stack

A spreadsheet or audited calculation is sufficient for this decision. Proposed production needs a versioned policy service, queue, reviewer console, immutable decision log, and approval endpoint. No tool is claimed as implemented.

### 5c. Method Summary

Compute required reviews `q*N`; reject any proposal over capacity or with an unapproved state change; then compare illustrative net value. The baseline is “fund none / human workflow.” A weighted score was considered but rejected because it can hide a hard safety violation.

### 5d. Provenance, Quality, Exclusions and Leakage

All values are ILLUSTRATIVE. A real evaluation must freeze eligibility before looking at outcomes, distinguish event time from review-completion time, and prevent later resolution fields entering prediction-time inputs. No source body in the local inventory supports results here.

### 5e. Reproducibility and Method-Selection Ledger

| ID | Location | Purpose | Accessibility | Limitation |
|---|---|---|---|---|
| SRC-01 | `02_AI_Market_Map.md` | workload/value formulas | local | teaching values only |
| RUN-01 | hand calculation below | capacity/ranking trace | reproducible by arithmetic | not executed system test |

## 6. Analysis Walkthrough

| Step | Input | Operation | Output | Interpretation |
|---|---|---|---|---|
| 1 | N=2,000/day | calculate `q*N` | review demand | capacity gate |
| 2 | reversibility field | apply zero-unapproved-effect gate | eligibility | safety gate |
| 3 | benefits/costs | calculate estimated daily value | ranking | only among eligible options |

| Proposal | Routed fraction q | Reviews/day | Estimated value/day | Action boundary |
|---|---:|---:|---:|---|
| A drafting | 0.08 | 160 | $600 | reviewer sends output |
| B triage | 0.15 | 300 | $750 | reviewer confirms route |
| C account action | 0.02 | 40 | $1,100 | changes account without approval |

**Checked walkthrough.** A requires `0.08*2,000=160` reviews/day, leaving 80/day headroom. B requires `0.15*2,000=300`, exceeding the 240/day limit by 60. C satisfies capacity but fails the side-effect gate. Independent arithmetic: 8% of 2,000 is 160; 15% is 300; 2% is 40.

Sensitivity: at N=3,000/day, A requires 240 reviews/day and has zero headroom; it should not scale under this assumption. If A’s routed fraction rises to 13%, it needs 260 reviews/day and fails. The decision changes before considering its illustrative value.

What we tried first: no actual portfolio was tried. The rejected **method**, not a historical run, is selecting by gross value alone; it would select C despite a disallowed autonomous effect.

## 7. The Finding

Pilot A only, with a 160-review/day expected load and a proposed hard cap below 240/day. Do not pilot B at stated routing and volume; redesign its policy or add staffed capacity. Do not pilot C as autonomous action; convert it to propose-only with explicit approval. This is an illustrative recommendation based on constructed inputs, not a measured portfolio result.

## 8. System/Solution Design

```text
request -> eligibility policy -> versioned proposal policy -> review queue
        -> reviewer approve/reject/escalate -> audited outcome -> monitoring
failure: queue cap -> hold new intake -> baseline/manual route -> owner alert
```

The policy service owns routing; the reviewer owns final external communication. Every request has a stable ID; retries must use it to avoid duplicate queue entries. The initial rollout has a daily intake cap of 160, a kill switch to baseline, and no direct state-changing tool. A later action path requires identity, authorization, idempotency, audit, and tested rollback; it is not enabled by this design.

## 9. Evaluation & Validation

| Metric | Definition | Comparator | Status |
|---|---|---|---|
| review arrivals | routed eligible requests/day | capacity 240/day | proposed measurement |
| completion latency | review completion minus arrival | manual baseline | not measured |
| correction/harm | reviewer rejection/correction by reason | baseline and slice | not measured |
| cost | all system plus review cost/request | current workflow | not measured |

Run a limited, time-bounded pilot using a frozen eligibility rule. Monitor arrival/completion backlog, quality rubric, source authorization, correction reasons, cost, and disparate workflow slices. Stop on prohibited data/action, persistent queue growth, or declared quality/harm gate failure. The evaluation has no executed results.

## 10. Action Taken

In this illustrative case, the recommended action is a **proposed** pilot of A with a 160/day intake cap and reviewer approval. B is held pending a routing redesign. C is changed to a suggestion-only workflow. No action was actually taken.

## 11. Outcome & Results

**Not measured.** The only result is a hand-derived capacity conclusion: A fits the stated simplified constraint; B and C each fail a distinct hard gate. A real outcome record must report numerator, denominator, window, comparator, uncertainty, and evidence label.

## 12. Failure Modes & What Went Wrong

| Failure | Competing causes | Diagnostic | Bounded response |
|---|---|---|---|
| Queue grows | volume, routing threshold, slow reviews | arrival vs completion time series | cap intake; investigate policy/service-time tail |
| Reviewer approves unsafe output | rubric gap, fatigue, missing evidence | sample corrections/escalations | revise rubric and restrict task |
| Duplicate work | retry, unstable request IDs | same ID appears twice | idempotency key and dedupe |
| Value estimate fails | baseline mismeasured, benefit double-counted | reconcile observed workflow ledger | remove claim; re-rank |
| Action occurs without authority | integration bypass, permission error | audit event has no approval ID | disable action path; investigate |

## 13. Transferable Lesson

Portfolio selection is a feasibility problem before it is a ranking problem. Use hard gates for authorization, reversibility, and operational capacity; compare expected value only among options that remain feasible. A human review label is inadequate until its arrival rate, service capacity, ownership, and failure path are explicit.

## 14. Connections & Cross-References

See [AI Use-Case Charter](../Cheat_Sheets/01_AI_Use_Case_Charter.md), [AI Market Map](../Cheat_Sheets/02_AI_Market_Map.md), and [Technology Due Diligence](../Cheat_Sheets/03_Technology_Due_Diligence.md).

1. **Why does B fail?** `300>240` reviews/day; it exceeds the capacity constraint.
2. **What is A’s headroom?** `240-160=80` reviews/day under the stated model.
3. **Why not choose C for highest value?** It has an unapproved state-changing action, a non-compensable gate failure.
4. **What changes at 3,000 requests/day for A?** A needs 240 reviews/day, leaving no headroom.
5. **Why is a weighted score wrong here?** It can conceal a hard safety/authorization failure.
6. **What is the baseline?** Funding none and retaining the current human workflow.
7. **How is duplicate work detected?** A stable request ID and queue/audit replay.
8. **What observation could reverse A’s recommendation?** A routing fraction above 12% at N=2,000, capacity decrease, or unacceptable correction/harm rate.
9. **What is the plausible wrong method?** Select the largest estimated benefit before applying gates.
10. **What is measured here?** Nothing operational; arithmetic only.
11. **Who owns a final action?** The designated reviewer, not the generation component.
12. **What must a real pilot add?** Frozen eligibility, versioned policy, audit, quality/harm rubric, workload traces, and rollback.

## 15. Quick Reference Card

| Field | Case rule |
|---|---|
| Decision | fund only feasible pilot candidates |
| Gate | `q*N <= review capacity`; no unapproved effect |
| Worked result | A: 160/day fits; B: 300/day fails; C: action gate fails |
| Evidence | illustrative arithmetic; not measured |
| Action | pilot A with cap and approval; redesign/hold B and C |
| Review trigger | workload, routing, capacity, or authorization changes |
