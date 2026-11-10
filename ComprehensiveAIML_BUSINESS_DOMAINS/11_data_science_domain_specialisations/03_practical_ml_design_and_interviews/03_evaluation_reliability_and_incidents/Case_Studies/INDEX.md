# Evaluation, Reliability and Incidents — Case Studies

## 1. Collection metadata
| Field | Value |
|---|---|
| Owner | Curriculum maintainer |
| Canonical ID format | `CASE-03-03-NNN` |
| Index state | planned catalog; no local case authored |
| Evidence policy | source access and evaluation execution are recorded independently |
| Review trigger | evaluand, version manifest, incident evidence, recovery contract, or source access changes |

## 2. Case catalog
| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-03-03-01 | RAG regression | evaluation/reliability | llm-agentic | Block, promote, or investigate a candidate release? | evaluation-mismatch | rollout-regression | not-measured | proposed | not reviewed | quality/safety suites unspecified |
| PLAN-03-03-02 | Forecast drift | evaluation/reliability | predictive-ml | Contain or continue after population change? | model-monitoring | label-delay | not-measured | proposed | not reviewed | outcome maturity absent |
| PLAN-03-03-03 | Streaming lag | evaluation/reliability | streaming-systems | Recover correctness under lag and replay? | incident-postmortem | cost-latency-tradeoff | not-measured | proposed | not reviewed | workload and oracle absent |
| PLAN-03-03-04 | Edge-model rollback | evaluation/reliability | hybrid | Revert safely under device/version constraints? | rollout-regression | distribution-shift | not-measured | proposed | not reviewed | device evidence absent |
| PLAN-03-03-05 | Agent action incident | evaluation/reliability | llm-agentic | Contain an unsafe action path and recover state? | incident-postmortem | human-in-the-loop | not-measured | proposed | not reviewed | no incident evidence |
| PLAN-2026-IR008 | Release-evaluation gate | evaluation/reliability | llm-agentic | Is evidence sufficient to promote a candidate? | evaluation-mismatch | data-leakage, rollout-regression | not-measured | proposed | not reviewed | reference-only brief |
| PLAN-2026-IR030 | Algorithm-invariant incident replay | evaluation/reliability | incident | Which invariant failed and what replay prevents recurrence? | incident-postmortem | correctness-contract, bounded-work | not-measured | proposed | not reviewed | reference-only brief |

## 3. Canonical reusable patterns
| Pattern | Signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | offline evidence omits a decision-critical failure | release selection | evaluand fully matches live decision |
| model-monitoring | operational signals shift before outcomes mature | deployed predictive system | no meaningful feedback signal |
| rollout-regression | guardrail worsens under staged exposure | reversible deployment | no comparable control or baseline |
| incident-postmortem | impact requires causal reconstruction | evidence/replay exists | narrative-only event |
| data-leakage | unavailable information inflates test result | suspiciously strong evaluation | temporal availability is verified |

## 4. Similarity and contrast map
| Cluster | Shared need | Critical distinction | Supporting method |
|---|---|---|---|
| RAG release / agent action | evidence before automation | answer quality differs from action authorization | manifests, sandboxing, approval |
| forecast drift / edge rollback | safe degradation | label delay differs from device compatibility | cohort maturity, version inventory |
| streaming lag / invariant replay | recovery correctness | queue progress differs from algorithm termination | replay oracle and state contract |

## 5. Questions and future-case backlog
Each case must become a bounded, source-honest decision exercise with 15 ordered sections, 5a–5e, input-to-output architecture, success and failed recovery paths, checked arithmetic or trace, sensitivity, claim ledger, and 12–18 answered interview prompts. Reference briefs IR008 and IR030 cannot donate unverified figures or observed-outcome claims.

## 6. Review and retirement log
| Scope | Editorial state | Last substantive review | Trigger |
|---|---|---|---|
| All planned cases | planned | none | primary evidence, bounded decision, and evaluation/replay artifact available |
| Index navigation | drafted | 2026-11-09 | planned catalog or source brief changes |

## 7. Pre-publish checks
- [x] Seven required collection-index sections and catalog fields are present.
- [ ] Each authored case meets the 15-section >=1000-line contract.
- [ ] Evidence, stage, editorial status, and result limitation are distinct.
- [ ] Cases contain actual calculation/trace checks and failure/recovery analysis.
- [ ] Independent evidence and source review are complete.
