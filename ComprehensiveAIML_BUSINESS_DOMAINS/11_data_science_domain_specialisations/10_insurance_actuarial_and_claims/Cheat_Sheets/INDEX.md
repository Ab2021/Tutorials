# Insurance, Actuarial and Claims — Cross-Domain Cheat Sheets

## 1. Overview
This collection is for reusable controls spanning pricing, claims, and model operations. It teaches an analyst to specify a model's intended use, preserve a release decision, and distinguish a monitored signal from proof of correctness. It assumes the domain sheets' insurance vocabulary and does not provide actuarial, legal, or regulatory advice. Readiness means a reader can reconstruct an approval, change, monitoring, and recovery decision from retained evidence.

## 2. Topic ladder
| Level | File | Testable outcome | Prerequisites | Status |
|---|---|---|---|---|
| Core | [01 Insurance Model Risk and Change Control](01_Insurance_Model_Risk_and_Change_Control.md) | Classify a proposed change and specify evidence, owner, monitor, and rollback boundary. | Basic model lifecycle and decision logging. | Draft; below 1,000 lines, with independent substantive review open. |
| Advanced | Model-risk governance and explainability | Reconcile intended use, explanation audience, validation, and approval scope. | 01 and domain method sheet. | Planned; source topic `11_Model_Risk_Governance_Explainability`. |
| Advanced | Compatible MLOps release and recovery | Verify compatible model, schema, policy, and routing versions. | 01; versioning and incident basics. | Planned; source topic `12_MLOps_Insurance_Production`. |
| Final | Common Insurance Analytics Issues | Diagnose 50 distinct actuarial, claims, and model-operation failures. | All completed sheets. | Planned; source title requires 50 complete issues. |

## 3. Quick topic lookup
| Question | File / section | Use condition | Failure warning |
|---|---|---|---|
| What must a model inventory record? | [01 §3](01_Insurance_Model_Risk_and_Change_Control.md#3-foundational-definitions) | Establishing intended use. | Inventory presence does not validate a model. |
| Is a change material? | [01 §3](01_Insurance_Model_Risk_and_Change_Control.md#3-foundational-definitions) | Data, feature, threshold, policy, vendor, or deployment change. | Do not classify by code diff alone. |
| What should be monitored with delayed labels? | [01 §4.2](01_Insurance_Model_Risk_and_Change_Control.md#42-general-delayed-outcome-performance) | Outcome quality is not immediately known. | Stable aggregate metrics can conceal slice harm. |

## 4. Reading paths
- **Beginner:** 01, then the pricing and reserving sheets in `../01_pricing_underwriting_and_reserving/Cheat_Sheets/`.
- **Practitioner:** 01; pricing; reserving; apply the change record to one proposed claims workflow.
- **Advanced:** complete the practitioner path, then the planned governance, release/recovery, and 50-issue artifacts when authored.

## 5. Prerequisite skills
Readers need versioned-artifact concepts, elementary probability and ratios, a distinction between decision-time and later data, and the ability to name a business owner. The child pricing and reserving collection provides concrete exposure and development examples.

## 6. Exclusions and handoffs
This collection does not determine a jurisdiction's requirements, approve a rate, select a reserve, or certify fairness. Pricing belongs in `../01_pricing_underwriting_and_reserving/`; claims workflow belongs in `../02_claims_fraud_customer_and_operations/`; climate scenario methods belong in `../03_insurance_mlops_model_risk_and_climate/`. Applicable legal, actuarial, privacy, and model-risk owners remain required handoffs.

## 7. Master symbol glossary
| Symbol / term | Meaning | Units / type | Domain / warning | First use |
|---|---|---|---|---|
| Intended use | Decision, population, owner, and permitted action | Structured record | Must be narrower than a model name | [01 §3](01_Insurance_Model_Risk_and_Change_Control.md#3-foundational-definitions) |
| Material change | Change capable of altering decision risk or control evidence | Classification | Not synonymous with a large code diff | [01 §3](01_Insurance_Model_Risk_and_Change_Control.md#3-foundational-definitions) |
| Mature outcome | Outcome whose definition and observation window are complete | Event/status | Not an immediate proxy label | [01 §3](01_Insurance_Model_Risk_and_Change_Control.md#3-foundational-definitions) |
| Override | Authorized departure from a model or policy output | Decision event | Requires reason and authority | [01 §3](01_Insurance_Model_Risk_and_Change_Control.md#3-foundational-definitions) |

## 8. Quick Reference Card directory
| Card | Rule | Use condition | Critical assumption | Avoid when |
|---|---|---|---|---|
| [Model risk and change control card](01_Insurance_Model_Risk_and_Change_Control.md#12-quick-reference-card) | Retain use, owner, data, version, evidence, approval, and recovery record together. | Any decision-support model change. | The record corresponds to the actual deployed bundle. | Treating a checklist as independent validation. |

## 9. Common Issues locator
No Common Issues artifact exists yet. The source inventory names `13_50_Common_Insurance_Analytics_Issues.md`; its contents were unavailable, but its visible 50-issue scope remains a required planned target. Local failure rows in 01 do not meet the 50-issue, 12-section artifact requirement.

## 10. Sources, versions and maintenance
| Artifact / source ID | Evidence and limitation | Editorial state | Review trigger |
|---|---|---|---|
| 01 | Visible source inventory plus generic derived control reasoning; no insurer implementation evaluated. | Draft; 12 sections exist, while the 1,000-line and substantive-review gates remain open. | Model-risk policy, workflow, source, or evidence change. |
| `Insurance_Cheat_Sheets/11_Model_Risk_Governance_Explainability.md` | Visible filename only; body unavailable. | Planned. | Source body becomes available. |
| `Insurance_Cheat_Sheets/12_MLOps_Insurance_Production.md` | Visible filename only; body unavailable. | Planned. | Source body becomes available. |
| `Insurance_Cheat_Sheets/13_50_Common_Insurance_Analytics_Issues.md` | Visible filename only; title commits 50 issues. | Planned. | Source body or collection scope changes. |

## 11. Collection acceptance checklist
- [x] One canonical index with prerequisite paths and source limits.
- [x] Visible source-topic commitments retained as planned work.
- [x] Card directory and issue locator distinguish actual from planned artifacts.
- [ ] Every indexed topic is an authored 12-section, 1,000-line sheet.
- [ ] A 50-issue final troubleshooting artifact exists.
- [ ] Independent numerical, source, navigation, and editorial review completed.
