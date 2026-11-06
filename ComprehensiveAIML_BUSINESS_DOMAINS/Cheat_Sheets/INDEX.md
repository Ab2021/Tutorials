# Cross-Domain Cheat Sheets

## 1. Overview

This planned collection supports architecture reviews across the course. Its sheets connect methods without replacing focused module references. No individual sheet or completed troubleshooting entry exists yet. [Course authoring contract](../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [master template](../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md) govern every future file. Owner: curriculum maintainer; collection version: 2.0-planned; substantive content review: not performed.

## 2. Topic ladder

| Level | ID | Planned sheet | Testable outcome and numerical/mechanism task | Prerequisites | Status |
|---|---|---|---|---|---|
| 1 | 01 | AI system design review | Turn a workload into explicit constraints; calculate throughput, concurrency and storage with units and limits | External: rates, units, service boundaries | planned |
| 2 | 02 | Model-to-production launch checklist | Define quality/reliability gates and a rollback-compatible release; calculate an error budget for a stated window | 01; module-specific evaluation prerequisites | planned |
| 2 | 03 | Agent risk and control matrix | Trace identity, tool authority, exact-action approval and retry state; calculate a bounded action/token budget | 01; agent and tool-contract modules | planned |
| 2 | 04 | Edge and robotics deployment gate | Reject a design violating timing, memory, power or safe-state constraints; calculate a complete cycle budget | 01; sensor/control prerequisites | planned |
| 3 | 05 | AI evaluation scorecard | Select denominators, uncertainty and guardrails across predictive, generative, agentic and multimodal tasks | 01–04; statistics and evaluation modules | planned |
| 3 | 06 | AI investment and vendor assessment | Compare cost per useful outcome including failed work, review labor and sensitivity, without unsupported current prices | 01–05 | planned |
| Final | 07 | 25 Common Issues | Distinguish 25 cross-domain failure mechanisms through diagnostic evidence, bounded fixes and verification | 01–06 | planned; 0 completed issues |

Each row is an authoring brief, not a link to unwritten content. Split independent calculations into focused sheets when needed; preserve the sequence and update IDs before publication.

## 3. Quick topic lookup

| Question | Planned destination | Use condition / warning |
|---|---|---|
| Can the system handle the workload? | 01, Sections 4–5 and 9 | Rates must include peaks, retries and state growth |
| Can this release be recovered? | 02, Sections 7–8 | Model rollback alone may leave incompatible state |
| May an agent execute this action? | 03, Sections 4 and 8 | Capability exposure is not permission |
| Is an on-device cycle feasible? | 04, Sections 5 and 9 | Mean latency is not a worst-case guarantee |
| Does the metric justify the decision? | 05, Sections 4–5 | Offline quality does not establish business impact |
| Is the vendor cost comparison valid? | 06, Sections 7 and 9 | Require dated currency/prices and comparable work |

## 4. Reading paths

- **Beginner:** 01 → 02 → 03; assumes unit arithmetic and service basics. Optional: 04 for physical systems.
- **Practitioner:** 01 → 02 → 03 → 04 → 05 → 06; complete each topic's external module prerequisites before its exercises.
- **Advanced:** the practitioner path → 07, plus relevant source-specific sheets in the [source crosswalk](../00_SOURCE_TOPIC_CROSSWALK.md).

## 5. Prerequisite skills

Use [Part 01 foundations](../01_foundations_computing_and_problem_solving/INDEX.md) for programming, systems and probability; [evaluation/platform modules](../05_ai_platforms_data_systems_and_production_engineering/INDEX.md) for model lifecycle and reliability; [edge modules](../07_edge_ai_robotics_and_cyber_physical_systems/INDEX.md) before device/control calculations. Specific earlier-sheet dependencies must be linked when those sheets exist.

## 6. Exclusions and handoffs

Detailed GPU kernels, modality objectives, actuarial models and sector workflows belong to the focused [source-coverage plans](../00_AUDIT_COVERAGE_MATRIX.md). This collection covers cross-domain selection and operating gates, not every method's derivation. No vendor architecture, regulatory applicability or production benchmark is established by a planning row.

## 7. Master symbol glossary

Notation standard: fenced plain-text equations. Define symbols locally even when registered here.

| Symbol | Meaning | Units / type | Domain | First planned use / overload warning |
|---|---|---|---|---|
| lambda | Arrival rate | events/s; nonnegative scalar | Stated workload/window | 01; not a model regularization parameter here |
| W | Mean time in a defined system | s; scalar | Stable accounting boundary | 01; not a weight matrix here |
| L | Mean items in that system | items; scalar | Same population/window as lambda and W | 01; not loss here |
| B | Backlog | events; nonnegative count | Defined queue/time | 01; distinguish bytes and batch size |
| C_useful | Total included cost / successful useful outcomes | currency/outcome | Positive outcome denominator, dated prices | 06; not all requests are useful outcomes |

Derivations and numerical examples remain to be authored; this glossary does not assert empirical queue or cost behavior.

## 8. Quick Reference Card directory

| Planned card | Core rule to establish | Critical assumption / avoid condition |
|---|---|---|
| 01, Section 12 | Dimensionally consistent workload sizing | Same accounting boundary; no tail-latency claim from a mean |
| 02, Section 12 | Promote and recover a compatible release unit | State/schema rollback must be viable |
| 03, Section 12 | Authorize exact actions, not just tools | Reject stale or mismatched approvals |
| 04, Section 12 | Enforce device and safety budgets before optimization | Do not infer safe operation from offline accuracy |
| 05, Section 12 | Validate decision metrics against comparators | Do not infer causality from uncontrolled changes |
| 06, Section 12 | Compare cost per useful outcome | Avoid incompatible workload/price bases |
| 07, Section 12 | Diagnose before mitigation | A symptom is not a confirmed cause |

Cards are planned; no nonexistent section anchors are presented as working links.

## 9. Common Issues locator

Final artifact: planned `07_25_Common_Issues.md`; **target 25, completed 0**. Allocate distinct failures across sizing/overload, evaluation validity, authorization/retry state, rollout compatibility, device deadlines and economics. Populate real ISSUE IDs, categories, symptoms, discriminating diagnostics, provenance and anchors only when entries are authored. The template's four categories and separate provenance labels are mandatory.

## 10. Sources, versions and maintenance

| Record | Scope | State / review trigger |
|---|---|---|
| [Master template](../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md) | Structure, equations, questions, troubleshooting | Applicable; recheck on template revision |
| [Course numerical standard](../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) | Canonical notation, evidence and checked-example requirements | Applicable; each sheet still needs its own verification |
| Sheet-specific primary sources | API/hardware/method/price claims | Not collected; verify when authoring and on dependency/price changes |

All seven planned artifacts have technical checks, numerical checks and interview-answer checks pending. Authors/reviewers are not assigned. Quarterly catalog review and source/dependency changes trigger maintenance; no substantive sheet-review date is claimed.

## 11. Acceptance checklist

- [x] Planned ladder, scope, three paths and source handoffs are explicit.
- [x] Unwritten sheets, cards and issues are not counted as completed.
- [ ] Author focused sheets with all 12 sections, derivations/invariants, checked numerical examples and at least four diagnostic failures each.
- [ ] Complete 10–15 answered interview questions per sheet and keep each card last.
- [ ] Complete 25 distinct actionable issues with honest provenance.
- [ ] Verify sources, prerequisite graph, calculations, paths/anchors, glossary, review records and publication gates.
