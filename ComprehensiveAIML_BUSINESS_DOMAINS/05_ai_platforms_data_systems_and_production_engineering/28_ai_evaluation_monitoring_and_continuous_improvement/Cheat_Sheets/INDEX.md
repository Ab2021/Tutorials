# AI Evaluation, Monitoring and Continuous Improvement — Cheat Sheets

**Status:** One unreviewed draft; all other collection topics remain planned. Follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master. The inventoried original source files were unavailable and their contents were not inspected.

## 1. Overview

This collection supports engineers evaluating deployed predictive and agent systems. The current draft distinguishes observable operational behavior, mature outcome quality, population shift and response. Readiness means defining the eligible population and labels, selecting a meaningful baseline and responding without treating a proxy as proof of impact.

## 2. Topic ladder

| Level | Reference | Outcome and prerequisite | Status |
|---|---|---|---|
| Practitioner | [Model evaluation and production monitoring](04_Model_Evaluation_and_Production_Monitoring.md) | Diagnose outcomes and data shift under delayed labels; requires confusion matrices and deployed-system basics | Draft; unreviewed |
| Planned | Evaluation matrix | Define workload, baselines, metrics, slices and release gates | Not written |
| Planned | Human-rater guide | Specify reviewer selection, calibration and disagreement handling | Not written |
| Planned | Production metric tree | Connect outcomes, proxies, operational signals and guardrails | Not written |
| Planned | Common Issues | Collection-wide distinct failure diagnoses | Not written |

## 3. Quick topic lookup

| Question | Reference | Condition and warning |
|---|---|---|
| Is model quality really declining? | [Worked example](04_Model_Evaluation_and_Production_Monitoring.md#5-worked-numerical-example) | Mature and representative labels are necessary for population-quality claims. |
| Does population stability index prove harm? | [Failure modes](04_Model_Evaluation_and_Production_Monitoring.md#8-failure-modes--edge-cases) | Feature shift is a diagnostic signal, not proof of degradation or causation. |
| When should the team respond? | [Comparison table](04_Model_Evaluation_and_Production_Monitoring.md#7-comparison--selection-table) | Consider impact, coverage, uncertainty, capacity and the available fallback. |

## 4. Reading paths

- **Beginner:** Review classification, cohorts, proportions and event logging first; then read the draft's mental model and definitions. Foundational evaluation-matrix material remains planned.
- **Practitioner:** Read the complete monitoring draft; reproduce its illustrative confusion-matrix and shift calculations, then define a real label-maturity and response contract.
- **Advanced:** Examine selection, segment weighting, delayed feedback, agent outcome definitions and intervention effects; design paired or randomized evaluations where feasible. The human-review and full metric-tree references remain planned.

## 5. Prerequisite skills

Understand confusion-matrix counts and denominators, sampling and selection bias, confidence-interval assumptions, event timestamps, features and model versions, and operational escalation ownership. For agent evaluation distinguish task completion from task success.

## 6. What this collection does not cover

The existing draft does not establish causal impact, guarantee safety, implement a complete evaluation platform, or supply version-specific monitoring integrations. Human-rater methodology, a comprehensive metric tree and a standalone issue catalog remain planned.

## 7. Master symbol glossary

| Symbol | Meaning | Scope / warning | Reference |
|---|---|---|---|
| TP, FP, FN, TN | Confusion-matrix counts for a defined cohort and outcome | Unlabeled cases must not silently enter the denominator | [Worked example](04_Model_Evaluation_and_Production_Monitoring.md#5-worked-numerical-example) |
| Precision | TP / (TP + FP) | Defined only for positive denominator; depends on prevalence and labels | [Formula section](04_Model_Evaluation_and_Production_Monitoring.md#4-core-formulas-with-full-derivation) |
| Recall | TP / (TP + FN) | Requires outcome ascertainment for the target population | [Formula section](04_Model_Evaluation_and_Production_Monitoring.md#4-core-formulas-with-full-derivation) |
| PSI | Population stability index comparing distributions | Bin choices and reference cohort matter; it does not prove outcome impact | [Worked example](04_Model_Evaluation_and_Production_Monitoring.md#5-worked-numerical-example) |

## 8. Quick Reference Card directory

| Card | Core rule | Use and critical condition |
|---|---|---|
| [Evaluation and monitoring](04_Model_Evaluation_and_Production_Monitoring.md#12-quick-reference-card) | Precision and recall use different denominators. | Labels, eligibility, maturity and selection must support the intended population inference. |

## 9. Common Issues locator

The collection-wide Common Issues artifact is **planned, not written**; no total is claimed. Use the draft's [failure modes and edge cases](04_Model_Evaluation_and_Production_Monitoring.md#8-failure-modes--edge-cases) for symptoms, competing explanations, mitigations and verification.

## 10. Sources, versions and maintenance

The draft uses illustrative and derived examples; no deployment outcome is measured. Revisit its metric, label, source and response contracts when the model, cohort, operating policy, deployment version or label-maturity process changes. The [source coverage brief](../../../00_SOURCE_COVERAGE_AGENTS.md#agent-observability) is a topic inventory, not source evidence.

**Additional planned briefs:** [PLAN-2026-IR001 and PLAN-2026-IR008](../../../00_REFERENCE_INFORMED_CASE_PLAN.md) cover dataset lineage and contamination, versioned evaluation manifests, calibrated judges, over-refusal, sandboxed scoring, paired comparisons and missing-suite failure tests. Their [corpus-contamination](../../../../INTERVIEW_CASES/Agentic_AI_System/AI_engg_challenges/synthetic_corpus_contamination.md) and [release-evaluation](../../../../INTERVIEW_CASES/Agentic_AI_System/LLM_Safety_Evaluation/release_evaluation_platform.md) examples are references only, not verified incidents or completed sheets. A synthetic-text classifier does not establish authorship; a judge score is not a deployment guarantee.

## 11. Collection acceptance checklist

- [ ] All planned topics and a distinct Common Issues artifact are authored and reviewed.
- [ ] Each file preserves all 12 sections, checked calculations or traces, substantive diagnostics, answered interviews and a final card.
- [ ] Metric population, labels, uncertainty, evidence status and review triggers are explicit.
- [ ] Local links, glossary, reading paths, issue locations and editorial statuses match completed files.
