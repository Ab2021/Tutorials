---
collection: "healthcare-life-sciences-biomedical-ai"
file_id: "03"
topic: "Human factors, alert design, and recovery"
level: 2
status: "draft — substantive expansion required before publication"
version: "0.1"
created: "2026-11-09"
reviewed: null
review_trigger: "Change in clinical workflow, alert policy, staffing, user interface, or escalation ownership"
notation_style: "plain-text"
prerequisites: ["01_Clinical_Decision_Support_Evidence_and_Handoff.md"]
pattern_tags: ["human-factors", "alerts", "recovery"]
source_status: "OPEN — no original source body or local workflow study available"
owner_role: "Clinical operations and safety lead"
source_index: "INDEX.md"
---

# Human Factors: Alert Design, Review, and Recovery

> **Purpose:** Design an assistive alert so a person can identify its reason, choose or reject an action, and recover when the alert or its delivery fails.
>
> **Prerequisites:** [clinical decision-support evidence](01_Clinical_Decision_Support_Evidence_and_Handoff.md) and a mapped local workflow.
>
> **Key outcome:** Given an alert rate, review time, and shift capacity, you can calculate burden, define an acknowledgment/recovery invariant, and distinguish alert fatigue from transport failure.
>
> **This file does NOT cover:** clinical policy, autonomous action, or measured effectiveness of a particular interface.
>
> **Level:** 2 — Core Methods.
>
> **Evidence:** DERIVED workload relation; VERIFIED illustrative arithmetic; OPEN workflow evidence.

## 1. File Header

An alert is an interruption plus a decision request, not merely a model output. Its design must identify the recipient, urgency basis, evidence, available actions, ownership transfer, acknowledgment semantics, timeout, escalation, and audit record. An “acknowledged” alert is not proof that the proposed action was appropriate.

**Takeaway:** An alert is safe only when its workflow and recovery behavior are as explicit as its prediction rule.

## 2. Concept Map / Mental Model

```text
[eligible event] -> [policy/rule] -> [alert with reason/version]
                                      | unavailable recipient
                                      v
                               [named fallback/escalation]
                                      |
                                      v
[user sees] -> [acknowledge / defer / reject / act] -> [audit outcome]
                                      |
                                      v
                         [timeout, duplicate, delivery failure recovery]
```

The invariant is: every eligible alert reaches either a responsible recipient with a recorded disposition or a monitored exception queue. “Sent” is not delivery, “delivered” is not seen, and “seen” is not a clinical decision.

**Takeaway:** Separate creation, transport, presentation, acknowledgment, and action in both metrics and failure handling.

## 3. Foundational Definitions

| Term | Meaning | Boundary |
|---|---|---|
| alert burden | alerts requiring attention in a window | not the same as model volume if alerts are deduplicated |
| actionable alert | alert with an authorized recipient, evidence, and feasible action | not merely a high score |
| acknowledgment | recorded receipt/disposition signal | not agreement or action completion |
| escalation | time-bounded transfer when owner/path fails | not repeated notification forever |
| override/rejection | documented disagreement with system suggestion | not automatically an error |
| fatigue | degraded attention or response under interruption load | requires measurement, not assumed from count alone |

**Takeaway:** Terms that collapse delivery, attention, and action hide the very failures the interface must detect.

## 4. Core Formulas with Full Derivation

**Validity before use:** Workload arithmetic assumes a defined alert class, fixed average review time, and no overlap; it is a planning estimate, not a clinical staffing recommendation.

```text
FORMULA [HF-01]: H = A * m / 60
WHERE: H = review hours/shift; A = actionable alerts/shift [count];
       m = review minutes/alert [min/alert].
CONSTRAINT: average review time is representative; interruptions, escalation, and documentation excluded unless measured.
NUMERICAL EXAMPLE: A=180 alerts; m=3 min/alert; H=180*3/60=9 hours.
INTERPRETATION: one 8-hour reviewer cannot absorb this illustrative alert stream without backlog, sharing, or policy change.
```

If delivery requires acknowledgment by deadline `d`, completion rate is `acknowledged by d / eligible alerts`; report exception reasons separately. A high rate can hide inappropriate acknowledgment if recipients use a bulk-dismiss path.

**Boundary:** Reducing `A` by raising a threshold may lower burden while increasing missed opportunities. Workload cannot choose the threshold alone.

**Takeaway:** Quantify interruption demand and pair it with missed-alert and disposition evidence.

## 5. Worked Numerical Example

**ILLUSTRATIVE; arithmetic VERIFIED.** A worklist produces 180 actionable alerts per shift. At three minutes of initial review each, burden is nine hours. If an eight-hour capacity is binding, excess is one reviewer-hour. If 20 alerts are duplicated and deduplication safely collapses them, `A=160` and estimated initial burden becomes eight hours. That change must be tested against the risk that “duplicates” are distinct events needing separate action.

A five-minute acknowledgment target with 150 acknowledgments among 180 eligible alerts gives `150/180=83.3%`. The 30 exceptions must be separated into no delivery, no presentation, no recipient, delayed acknowledgment, and deliberately deferred review before treating the result as a user-performance metric.

**Takeaway:** A denominator and exception taxonomy prevent a superficially good acknowledgment number from concealing transport or staffing failure.

## 6. Visual / Geometric Intuition

Imagine alert load as arrivals to a bounded attention queue. When arrivals exceed safe review capacity, either waiting time grows or alerts are ignored, escalated, grouped, or displaced by other work. The UI controls which evidence is visible at the decision point; it cannot compensate for absent ownership or invalid inputs.

**Takeaway:** Alert fatigue is a queue-and-workflow problem as well as an interface problem.

## 7. Comparison / Selection Table

| Pattern | Use when | Advantage | Disqualifier | Recovery requirement |
|---|---|---|---|---|
| interruptive alert | urgent, actionable event | rapid attention | no timely action exists | acknowledged recipient and timeout escalation |
| passive worklist | prioritization is sufficient | fewer interruptions | time-critical event | stale-item and ownership monitoring |
| grouped/deduplicated alert | repeated evidence is genuinely same action | lowers burden | events have independent consequences | reversible grouping and audit members |
| abstain/fallback | input or authority is inadequate | prevents unsupported guidance | fallback unavailable | exception owner and baseline process |

**Takeaway:** Choose the least interruptive mechanism that still meets the defined response requirement.

## 8. Failure Modes & Edge Cases

| ID | Competing causes | Discriminating signal | Bounded correction / verification |
|---|---|---|---|
| HF-F01 alert fatigue | excessive volume; poor precision; staffing loss | high dismiss/defer rate with delivery intact, stratified by shift | change policy only after reviewing missed-event trade-off; observe disposition distribution |
| HF-F02 silent delivery failure | queue outage; address mapping error; client failure | created count exceeds delivered/presented count | route exception queue; test failed transport and recovery trace |
| HF-F03 duplicate suppression harms | repeated message; independent recurring event | inspect grouped event IDs and actions | make grouping reversible; sample audit before rollout |
| HF-F04 acknowledgment without review | bulk action; incentive; unclear UI | implausibly short latencies or repeated patterns | distinguish seen/acknowledged/action; usability review |
| HF-F05 escalation loop | absent owner; bad timeout; retry bug | repeated alerts without terminal disposition | cap retries; assign named escalation and verify closure |

**Takeaway:** Every alert failure needs a signal that distinguishes a human-workload problem from a technical-delivery problem.

## 9. Complexity & Cost Analysis

For `A` alerts, immutable audit writes and disposition reconciliation are `O(A)` records under ordinary append semantics. The material cost is attention: `A*m` minutes plus escalation and documentation. Storage, paging, and staffing prices are not measured here. Monitor backlog age and unassigned count, not only model score distribution.

**Takeaway:** The dominant cost of an alert system can be scarce reviewer attention, which must be explicitly budgeted.

## 10. Key Theorems / Results Reference

- **Conservation of work (DERIVED):** under the simple model, alert review minutes equal count times per-alert minutes.
- **End-to-end delivery principle:** a producer success response does not establish recipient presentation or acknowledgment.
- **Authority boundary:** an interface affordance must not create an action beyond the user’s assigned authority.

**Takeaway:** Instrument the whole path from eligibility through terminal disposition, not just the alert generator.

## 11. Interview / Exam Questions

1. **Compute review hours for 180 alerts at three minutes each.** Nine hours; state the fixed-average assumption.
2. **Why is “alert sent” inadequate?** Transport, presentation, ownership, and acknowledgment can still fail.
3. **What invariant should a critical alert have?** Terminal disposition by a responsible owner or a monitored exception path.
4. **How can deduplication be dangerous?** Similar-looking events may require independent actions; preserve members and reversibility.
5. **Does acknowledgment mean the recommendation was correct?** No; it records a workflow event, not validity or outcome.
6. **How do you distinguish fatigue from transport failure?** Compare created, delivered, presented, and disposition traces before interpreting response behavior.
7. **What changed constraint reverses a passive-worklist choice?** A verified time-critical response requirement may require interruptive escalation.
8. **What should happen on missing recipient identity?** Do not silently drop; route to an exception owner/fallback.
9. **What is OPEN?** Actual user study, local capacity, policy, and source-body evidence.
10. **Why track rejection?** It can expose interface, data, or policy mismatch and supports accountable review.

**Takeaway:** Interview answers should distinguish technical state transitions from human decisions.

## 12. Quick Reference Card

| Field | Rule |
|---|---|
| Alert contract | recipient, urgency, evidence, action choices, owner, timeout |
| Core sizing | `H=A*m/60`; disclose excluded work |
| Delivery states | created → delivered → presented → acknowledged → terminal disposition |
| Safety invariant | no eligible alert silently loses ownership |
| Duplicate rule | group only if action is truly shared and membership is auditable |
| Escalation | bounded retry, named fallback, monitored exception queue |
| Audit | policy/version, evidence, timestamps, dispositions, recovery events |
| Avoid | interpreting acknowledgment or lower volume as clinical benefit |
