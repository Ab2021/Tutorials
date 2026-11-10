# Insurance MLOps, Model Risk and Climate — Cheat Sheets

## 1. Overview
This collection will teach operational controls for model inventory, validation, monitoring, change control, explanation, fairness investigation, lineage, climate scenarios, and incident recovery. It is for readers who must defend a model-supported insurance workflow without inferring hidden source-course material or jurisdiction-specific obligations. A completed sheet must make an owner, evidence boundary, decision-time data, and recovery path reconstructible.

## 2. Topic ladder
| Level | Planned topic | Testable outcome | Prerequisites | Status |
|---|---|---|---|---|
| Foundation | Model inventory and validation | Specify intended use, evidence, validation boundary, and expiry. | Basic statistics and decision logging. | Planned. |
| Core | Drift, champion/challenger, explainability, fairness, audit, and lineage | Separate data/action/outcome change and select a bounded investigation. | Foundation; delayed-label awareness. | Planned. |
| Core | Climate risk and catastrophe scenarios | State scenario inputs, exposure basis, aggregation, and non-claim boundary. | Exposure and loss terminology. | Planned. |
| Advanced | MLOps production and incident response | Verify compatible artifacts, staged promotion, rollback, and reconciliation. | Versioned state and control ownership. | Planned; covers source `12_MLOps_Insurance_Production`. |
| Advanced | Customer analytics, CLV, and ensembles | Derive a stated-horizon customer value and compare a baseline with ensemble assumptions. | Conditional expectation and calibration. | Planned; covers sources 07/08. |
| Advanced | Regulatory eligibility retrieval | Filter by effective interval and jurisdiction before ranking; abstain to legal owner. | Versioned sources and retrieval evaluation. | Planned; PLAN-2026-IR013. |
| Final | 50 Common Insurance Analytics Issues | Diagnose 50 distinct complete issues. | All completed sheets. | Planned; source title specifies 50. |

## 3. Quick topic lookup
| Question | Planned home | Use condition | Failure warning |
|---|---|---|---|
| What belongs in a promotion bundle? | MLOps production §4–5 | Model, policy, schema, features, and routing can change. | Prompt-only rollback leaves incompatible dependencies. |
| How should a one-period CLV be calculated? | Customer analytics §4–5 | Margin, retention event, and discount timing are explicit. | Do not silently substitute an infinite-horizon formula. |
| Is a source eligible for a regulatory answer? | Regulatory eligibility retrieval §4, §8 | Effective time and jurisdiction metadata exist. | Retrieval ranking cannot repair ineligible material. |

## 4. Reading paths
- **Beginner:** model inventory/validation → drift/lineage → MLOps production.
- **Practitioner:** beginner path → explainability/fairness/audit → climate scenarios → incident response.
- **Advanced:** practitioner path → customer analytics/ensembles → eligibility retrieval → 50 Common Issues.

## 5. Prerequisite skills
Readers need probability, population-versus-slice reasoning, version-control concepts, event-time discipline, and ability to distinguish an observed outcome from an illustrative calculation. The cross-domain [model-risk draft](../../Cheat_Sheets/01_Insurance_Model_Risk_and_Change_Control.md) is a conceptual predecessor but is not yet master-template complete.

## 6. Exclusions and handoffs
This collection does not establish legal eligibility, actuarial approval, climate projections, fairness compliance, or a particular vendor's behavior. Pricing and reserves hand off to `../../01_pricing_underwriting_and_reserving/`; claims process controls hand off to `../../02_claims_fraud_customer_and_operations/`. Escalate actual regulatory, privacy, actuarial, and security decisions to accountable owners.

## 7. Master symbol glossary
| Symbol / term | Meaning | Units/type | Domain / warning | Planned first use |
|---|---|---|---|---|
| Effective interval | Time range in which a source or policy version applies | timestamp interval | Recorded time is not effective time | Eligibility retrieval |
| Mature outcome | Outcome available under a declared maturation rule | event/status | Not a proxy score | Drift/validation |
| Release manifest | Versioned set of compatible artifacts | structured record | A model hash alone is insufficient | MLOps production |
| One-period expected discounted margin | Margin earned if retained for a stated next period, discounted to present | currency | Not lifetime CLV | Customer analytics |
| Scenario | Explicit conditional input set, not a forecast | structured input | A scenario result is conditional | Climate risk |

## 8. Quick Reference Card directory
No completed cards exist. Planned cards will cover model inventory, tiering, climate-scenario documentation, compatible release/rollback, customer-value conditioning, and eligibility-before-ranking. A card link will be added only with the corresponding authored sheet.

## 9. Common Issues locator
No issue artifact exists. The visible source filename `13_50_Common_Insurance_Analytics_Issues.md` commits this collection to 50, not 25, complete distinct issue entries. Its original body remains unavailable; no content is attributed to it.

## 10. Sources, versions and maintenance
| Commitment | Basis / limitation | State | Recheck trigger |
|---|---|---|---|
| Source files 07, 08, 11, 12, 13 | Visible filenames from `Insurance_Analytical_Modelling`; bodies unavailable. | Planned. | Source access or curriculum revision. |
| Course outline and Course_Content directories | Visible but contents unavailable. | OPEN; not treated as lesson evidence. | Directory contents become available. |
| PLAN-2026-IR013 | Reference-informed design brief; no legal guidance or tested system claim. | Planned. | Authority metadata or reference correction. |

## 11. Collection acceptance checklist
- [x] Visible source commitments and unresolved sources remain explicit.
- [x] 50-issue target is preserved.
- [x] Prerequisite paths and handoffs are available.
- [ ] Every indexed topic has a 12-section, 1,000-line authored sheet.
- [ ] The 50-issue final artifact is authored and checked.
- [ ] Independent source, numerical, navigation, and editorial review is complete.
