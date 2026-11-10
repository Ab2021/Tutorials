# Practical ML Design and Interviews — Cross-Domain Case Studies

## 1. Collection metadata
| Field | Value |
|---|---|
| Owner | Curriculum maintainer |
| Canonical ID format | `CASE-03-X-NNN` |
| Index state | reserved, no local case authored |
| Scope | decisions that require two or more child domains |
| Evidence rule | evidence level, implementation stage, and editorial state are separate |

## 2. Case catalog
| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| None | No cross-domain case registered | — | — | — | — | — | — | — | — | No artifact exists |

## 3. Canonical reusable patterns
| Pattern | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| decision-contract-mismatch | product action, ML output, and evaluation population disagree | an interface crosses child domains | same contract is verified end-to-end |
| evidence-to-recovery-gap | a release decision lacks a safe containment path | operational rollout | independent fallback is not viable |
| proxy-outcome-confusion | a local metric is mistaken for business or safety outcome | metrics cross workflows | outcome and action are aligned |

## 4. Similarity and contrast map
| Child artifact | Shared connection | Contrast preserved |
|---|---|---|
| [Feature-store sheet](../01_ml_system_design_production/Cheat_Sheets/03_Feature_Store_Design.md) | decision-time input validity | feature parity is not product value |
| [Drift case](../01_ml_system_design_production/Case_Studies/01_model_monitoring_drift_incident.md) | containment under uncertainty | feature shift is not causal harm |
| product/evaluation indexes | planned decision and evidence methods | no authored cases in those directories |

## 5. Questions and future-case backlog
A future cross-domain case needs a single consequential decision that cannot be taught adequately in one child collection; it must name the owner, input/output grain, baseline, constraint, evidence basis, and each domain handoff. It must meet the full 15-section case contract, including successful and failed recovery paths and a 12–18-question interview pack.

## 6. Review and retirement log
| Scope | Editorial state | Last substantive review | Trigger |
|---|---|---|---|
| Reserved collection | planned | none | bounded cross-domain decision becomes available |
| Index navigation | drafted | 2026-11-09 | child ownership or source plan changes |

## 7. Pre-publish checks
- [x] Seven-section index and original catalog fields are present.
- [ ] A bounded 15-section cross-domain case exists.
- [ ] Evidence/stage/editorial records and claim ledger exist.
- [ ] Sources and independent review are complete.
