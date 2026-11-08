---
collection: "program-orientation-ai-landscape"
file_id: "01"
topic: "AI use-case charter"
level: 1
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-07"
reviewed: null
source_status: "course-topic draft; visible directory consulted, original bodies unavailable"
---

# AI Use-Case Charter

> **Reference boundary:** [`Train/Templates/directory.txt`](../../../../Templates/directory.txt) is the source inventory. This course-topic draft is not a reconstruction of unavailable source materials.

## 1. Purpose

A charter prevents solution-first AI work. It makes the decision, user, baseline, evidence, risk, economics and owner explicit before selecting an LLM, predictive model, retrieval stack, automation or human workflow.

## 2. Required fields

| Field | Required statement |
|---|---|
| Decision | Who decides what, when, and with what consequence? |
| User/workflow | Eligible user, trigger, current path and fallback |
| Outcome | Measurable value and harm boundaries |
| Baseline | Current process metric/cost/error/latency |
| Intervention | Proposed model/system and what remains human-owned |
| Evidence | Data provenance, labels/ground truth, evaluation split and uncertainty |
| Risks | Privacy, security, safety, fairness, compliance and operational failure |
| Economics | Unit volume, marginal cost, expected benefit and capacity constraint |
| Gate | Promotion, rollback, owner and review date |

## 3. Decision rule

Expected net value is a hypothesis: `V = benefit - build_cost - run_cost - expected_harm_cost`. For 10,000 monthly decisions, $3 benefit/valid decision, 70% valid coverage, $8,000 build amortization and $5,000 run cost, illustrative gross benefit is `10000*.7*3=$21,000`; illustrative net before harm/risk is `$8,000`. This is not a business case until baseline, counterfactual, costs and harms are evidenced.

## 4. Select the simplest credible intervention

| Need | First candidate | Escalate only if |
|---|---|---|
| Current facts | Search/retrieval plus citations | Retrieval recall/assembly fails measured task |
| Deterministic rule | Rule/validator | Exceptions materially exceed manageable scope |
| Ranking/prediction | Statistical/ML model | Baseline cannot meet measured target |
| Open-ended assisted work | Human-in-loop LLM | Quality/safety/ownership gates are met |
| High-consequence action | Decision support/approval | Control, audit and rollback evidence exists |

## 5. Failure modes

| Failure | Check | Response |
|---|---|---|
| “AI” is solution without decision | Charter has no action owner/baseline | Stop and frame decision |
| Accuracy hides costly errors | Cost/harm slice and capacity analysis | Add hard gates/abstention/human review |
| Evaluation leaks workflow data | Dependency/time/provenance audit | Rebuild split and invalidate result |
| Pilot succeeds but cannot operate | Load, owner, incident and cost plan absent | Do not scale; define operating contract |

## 6. Validation

Compare intervention with current baseline on a frozen eligible population; report coverage, quality, harms, latency, cost and human workload by material slice. Run a limited rollout with immutable version, monitoring and tested rollback. A demo is not a promotion result.

## 7. Interview checks

1. **What comes before model choice?** The decision and baseline.
2. **Why state fallback?** Failure handling is part of product behavior.
3. **Why does gross benefit not prove value?** Counterfactual, costs and harms remain uncertain.
4. **When is human review required?** When consequence/uncertainty exceeds validated autonomy boundary.
5. **What promotes?** Predeclared quality, risk, economics and operating gates.

## 8. Quick reference

| Before building | Before promoting |
|---|---|
| Decision, user, baseline, data, risk and owner | Frozen evaluation, economics, operations, monitoring and rollback |

**Limitations:** No organization, workload or source lesson body was available. Numbers are illustrative.
