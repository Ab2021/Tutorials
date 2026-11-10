# ML Product Sense — Case Studies

## 1. Collection metadata
| Field | Value |
|---|---|
| Owner | Curriculum maintainer |
| Canonical ID format | `CASE-03-02-NNN` |
| Index state | planned catalog; no local case authored |
| Evidence policy | distinguish illustrative, estimated, observed, published, simulated, and not-measured |
| Review trigger | decision contract, metric, population, policy, or source-access change |

## 2. Case catalog
| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-03-02-01 | Design an AI feature | ML product sense | hybrid | Is the proposed user decision valuable and feasible? | evaluation-mismatch | human-in-the-loop | not-measured | proposed | not reviewed | no bounded case or result yet |
| PLAN-03-02-02 | Reject an unjustified ML proposal | ML product sense | analytical | Should a simpler baseline be retained? | cost-latency-tradeoff | evaluation-mismatch | not-measured | proposed | not reviewed | no evidence package yet |
| PLAN-03-02-03 | Human review workflow | ML product sense | predictive-ml | How should scarce review be allocated? | human-in-the-loop | label-delay | not-measured | proposed | not reviewed | capacity and quality unknown |
| PLAN-03-02-04 | Metric conflict | ML product sense | analytical | Which metric trade-off permits action? | evaluation-mismatch | causal-confounding | not-measured | proposed | not reviewed | outcome definition absent |
| PLAN-03-02-05 | Rollout decision | ML product sense | hybrid | Should the feature expand, pause, or roll back? | rollout-regression | evaluation-mismatch | not-measured | proposed | not reviewed | no evaluation window yet |

## 3. Canonical reusable patterns
| Pattern | Signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | proxy improves while decision quality is unknown | metrics differ from value | valid action-level outcome exists |
| human-in-the-loop | uncertainty or consequence needs qualified review | review can alter outcome | review has no capacity/quality evidence |
| causal-confounding | observed association is mistaken for intervention effect | nonrandomized comparison | defensible randomization or assumptions exist |
| rollout-regression | a live guardrail worsens after release | staged rollout exists | comparison population changed |
| cost-latency-tradeoff | quality competes with cost, delay, or staff | constraint is binding | cost/delay is immaterial |

## 4. Similarity and contrast map
| Case family | Shared decision | Contrast to preserve | Method handoff |
|---|---|---|---|
| AI feature / unjustified proposal | whether to use ML | adoption decision versus rejection evidence | product rubric and baseline analysis |
| human review / metric conflict | safe value measurement | capacity allocation versus estimand choice | operations and experimentation |
| rollout | reversible exposure | live policy versus offline selection | reliability/canary collection |

## 5. Questions and future-case backlog
Every planned row requires a canonical CASE ID, one owner, input/output unit, baseline, constraint, evidence basis, numerical walkthrough, sensitivity, architecture/recovery path where applicable, and 12–18 answered interview prompts before it can be called drafted. The source-plan additions for recommendation, search, fraud, NLP, forecasting, agents, CV, and causal experiments remain distinct topic commitments.

## 6. Review and retirement log
| Scope | Editorial state | Last substantive review | Trigger |
|---|---|---|---|
| All catalog rows | planned | none | bounded evidence and a decision contract become available |
| Index navigation | drafted | 2026-11-09 | catalog or source plan changes |

## 7. Pre-publish checks
- [x] Seven required index sections and original catalog fields are present.
- [ ] Each linked title is a 15-section >=1000-line case with Section 15 last.
- [ ] Evidence, implementation stage, and editorial state are independently recorded.
- [ ] Claim ledger, failure/recovery paths, calculations, sensitivity, and interview packs are complete.
- [ ] Source evidence and independent review are complete.
