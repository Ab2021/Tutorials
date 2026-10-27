# Module 26 — Quality, Evaluation, and Model Risk

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (14 lecture, 8 lab, 4 case clinic)
> **Track fit:** Common core; required for all tracks
> **Prerequisites:** Modules 06, 07, 10, 25
> **Tools:** Model inventory template, model card, backtesting harness, issue log
> **Industry question:** "How do we know this model is still fit for the decision, and who is accountable when it is not?"

---

## 1. Why industry cares

A demand model that is 8% better on average can still be wrong on the 200 items that drive half the margin. An inventory policy tuned on two years of stable lead times can fail in the first disruption. A vendor's black-box forecast can be accurate and unusable because no one can explain an override.

Model risk is the **risk of loss from a model being wrong, misused, or misunderstood**. In planning, the loss appears as stockouts, excess inventory, missed service, bad capital allocation, and lost credibility. It is not only a banking concept; regulators, auditors, and boards now ask planning and AI teams the same questions banks have asked for decades.

Three forces make this urgent in 2026:

1. **More models, more often.** Planning teams run hundreds of series-level models, not one.
2. **Vendor and foundation models.** The organization did not build the model and cannot fully inspect it.
3. **Regulation and audit.** The EU AI Act, ISO/IEC 42001, and internal audit functions expect documented intended use, testing, monitoring, and accountability.

## 2. Learning outcomes

By the end of this module, you can:

- Define a model and its scope for inventory and risk-tiering purposes.
- Build and maintain a model inventory with owners and versions.
- Tier models by materiality and complexity.
- Assess conceptual soundness against the decision.
- Design data-quality and representativeness tests.
- Run outcomes analysis, backtesting, benchmarking, and challenger comparisons.
- Write a model card and a validation dossier.
- Audit reproducibility and lineage.
- Define monitoring thresholds and revalidation triggers.
- Evaluate a vendor model and its documentation.
- Manage issues, severity, and remediation.

## 3. Mental model

```text
  INVENTORY ──> TIER ──> VALIDATE ──> APPROVE ──> MONITOR ──> REVALIDATE
      │           │         │            │           │            │
   owner,      materiality  conceptual   conditions  drift,       trigger,
   version,    × complexity soundness,   and limits  incidents,   change,
   use         × autonomy   data,        recorded    outcomes     retirement
                            outcomes
```

Model risk management is a **lifecycle**, not a one-time approval. The questions repeat at every stage:

- What decision does it support?
- What would make it wrong?
- How would we know?
- Who is accountable?
- What do we do when it fails?

### 3.1 What counts as a model

For inventory purposes, a model is a **quantitative method, system, or approach that processes inputs into outputs used in a decision**, including:

- Statistical and ML forecasts.
- Optimization and simulation models.
- Rules and heuristics embedded in planning systems.
- Vendor models and configured packages.
- Spreadsheets with material logic.
- LLM-based assistants that affect a decision.

Exclude purely descriptive reports with no decision output, but document the boundary.

## 4. Core concepts

### 4.1 Model inventory

Minimum fields:

| Field | Why |
|---|---|
| Model ID and name | Unique reference |
| Owner and validator | Accountability |
| Business decision | Purpose and scope |
| Outputs and users | Where it is used |
| Inputs and data sources | Lineage |
| Version and environment | Reproducibility |
| Tier | Control intensity |
| Last validation and next review | Currency |
| Known limitations | Honest use |
| Status | Active, retired, restricted |

An inventory that is not maintained is worse than none: it creates false assurance.

### 4.2 Tiering

Tier by **materiality** (financial, service, safety, regulatory impact) and **complexity/autonomy** (simple rule to opaque foundation model).

| Tier | Materiality | Complexity | Controls |
|---|---|---|---|
| 1 | High | High | Full independent validation, ongoing monitoring, board or committee approval |
| 2 | Medium | Medium | Validation, annual review, monitoring |
| 3 | Low | Low | Owner review, spot checks, documented assumptions |

Tiering is a judgment. Record the rationale, not only the tier.

### 4.3 Conceptual soundness

Ask whether the model's design is **appropriate for the decision**, not whether it is sophisticated.

| Question | Evidence |
|---|---|
| Is the target the decision variable? | Forecast of demand vs shipments vs orders |
| Is the horizon right? | Lead-time horizon, not one-step |
| Are the assumptions stated? | Stationarity, independence, capacity, lead time |
| Is the method appropriate? | Intermittent demand, hierarchy, censoring |
| Are the inputs valid? | Grain, coverage, latency, leakage |
| Is the output interpretable enough? | Override and explanation |
| Are the limitations documented? | Known failure regimes |

A model can be conceptually unsound even with a low error: it may be right for the wrong reason and fail when conditions change.

### 4.4 Data quality and representativeness

| Dimension | Test | Failure sign |
|---|---|---|
| Completeness | Null and row counts by key | Missing stores or weeks |
| Accuracy | Reconcile to source | Shipments ≠ orders |
| Timeliness | Event to availability lag | Forecast uses late data |
| Consistency | Duplicate and conflict checks | Two prices for one SKU |
| Validity | Range and domain rules | Negative demand |
| Representativeness | Coverage by segment | New stores absent |
| Stability | Schema and distribution drift | Renamed fields |

Representativeness is the most under-tested. A model trained on large stores can fail on small ones even with strong aggregate accuracy. Test performance by **slice**: store size, region, product class, volume band, promotion status, and age of item.

### 4.5 Outcomes analysis and backtesting

| Method | Question | Limitation |
|---|---|---|
| Backtest | Would it have worked historically? | Regime change |
| Rolling origin | Does it hold across time? | Costly |
| Benchmark | Does it beat naive and current process? | Benchmark quality |
| Challenger | Does an alternative do better? | Multiple testing |
| Sensitivity | What inputs matter? | Correlation, not causation |
| Stress test | What if lead time doubles? | Scenario credibility |
| Impact analysis | What decision would change? | Requires policy simulation |

Report **error and decision impact together**. A 5% WAPE improvement that changes no order is worth little; a 1% improvement on critical items may be worth a lot.

### 4.6 Metrics that matter for planning models

| Metric | Use | Caution |
|---|---|---|
| WAPE | Volume-weighted accuracy | Hides item-level tails |
| Bias | Systematic over/under | Averages mask offsetting errors |
| MASE | Scale-free comparison | Naive denominator can be tiny |
| Pinball loss | Quantile forecasts | Less intuitive to business |
| CRPS | Full distribution | Harder to explain |
| Fill rate impact | Decision outcome | Requires policy simulation |
| Stability | Forecast churn | High churn erodes trust |
| Slice performance | Fairness and robustness | Multiple comparisons |

### 4.7 Model documentation and the model card

A model card should state:

1. **Intended use** and out-of-scope uses.
2. **Decision and owner.**
3. **Inputs, grain, and lineage.**
4. **Method and assumptions.**
5. **Training period and validation design.**
6. **Performance overall and by slice.**
7. **Known limitations and failure regimes.**
8. **Monitoring metrics and thresholds.**
9. **Version, date, and validator.**
10. **Change log.**

A model card is not marketing. It is the document a new analyst reads before changing the model and an auditor reads after an incident.

### 4.8 Reproducibility and audit trail

Reproducibility requires:

- Code under version control with a commit hash.
- Data snapshot or hash.
- Pinned environment and dependencies.
- Recorded parameters and random seeds.
- Saved artifacts: model, forecasts, metrics.
- A run manifest that links all of the above.

If you cannot rerun last quarter's forecast, you cannot investigate why it was wrong.

### 4.9 Independence and validation

Validation is effective only when the validator can say no.

| Model | Builder | Validator |
|---|---|---|
| Demand forecast | Data science | Separate DS or external |
| Inventory policy | Operations research | Planning + finance |
| Vendor model | Vendor | Internal model risk |
| LLM assistant | Product team | Security + model risk |

Independence is a spectrum. At minimum, the validator should not be measured on the model's launch.

### 4.10 Ongoing monitoring

Monitor four layers:

| Layer | Metrics | Alert |
|---|---|---|
| Data | Nulls, row counts, schema, freshness | Missing feed |
| Model | Bias, WAPE, pinball, stability | Error above threshold |
| Decision | Override rate, order changes, service | Unexplained shift |
| Outcome | Fill rate, inventory, cost | Benefit not realized |

Define thresholds **before** launch. A threshold set after seeing the data is a rationalization.

### 4.11 Revalidation triggers

- Material change in data source or grain.
- Model or vendor version change.
- New product, market, or channel.
- A disruption that changes the demand regime.
- Performance breach for two consecutive cycles.
- A material incident or audit finding.
- Scheduled review (annual or per tier).

### 4.12 Vendor model evaluation

Ask for:

1. Documentation of method and assumptions.
2. Training data provenance and coverage.
3. Performance on your data, not a demo.
4. Explainability and override.
5. Monitoring and alerting.
6. Right to audit and access to logs.
7. Data ownership, residency, and egress.
8. Versioning and change notification.
9. Exit and transition support.
10. Indemnity and liability.

A vendor that will not provide these is a model risk you cannot manage.

## 5. Data, standards, and workflow

### 5.1 Standards and guidance

| Source | Scope | Status |
|---|---|---|
| Federal Reserve SR 11-7 / OCC 2011-12 | Model risk management in banking | Supervisory guidance, widely adopted |
| PRA SS1/23 | Model risk management principles for U.K. banks | In force |
| EU AI Act | AI systems, risk classes, documentation | Phased application 2025–2027 |
| ISO/IEC 42001 | AI management system | Published |
| NIST AI RMF | AI risk management framework | Voluntary |
| ISO 31000 | Risk management | Published |
| Internal model policy | Firm-specific | Binding |

Planning models are not bank models, but the SR 11-7 structure — inventory, tiering, validation, monitoring, governance — is the most usable template.

### 5.2 Validation workflow

1. Confirm scope and tier.
2. Review conceptual soundness.
3. Audit data quality and lineage.
4. Reproduce results.
5. Run outcomes analysis and challenger comparison.
6. Test sensitivity and stress.
7. Review documentation and controls.
8. Issue findings with severity.
9. Approve with conditions or reject.
10. Schedule monitoring and revalidation.

### 5.3 Issue severity

| Severity | Definition | Response |
|---|---|---|
| Critical | Wrong decision, regulatory breach, safety | Stop use, escalate |
| High | Material bias or control gap | Remediate before next cycle |
| Medium | Documentation or monitoring gap | Remediate within quarter |
| Low | Cosmetic or process | Track and fix |

## 6. Industry case studies

### 6.1 SR 11-7 and bank model risk

**Context:** After the 2008 crisis, U.S. supervisors formalized model risk management.
**Intervention:** SR 11-7 set expectations for inventory, validation, governance, and ongoing monitoring.
**Evidence:** The guidance is public and has been widely adopted beyond banking.
**Outcome:** Model risk became a named control function with independent validation.
**What transferred:** lifecycle structure, tiering, and independence.
**What did not:** the full banking governance apparatus is disproportionate for a small planning team; scale controls to tier.

### 6.2 Zillow Offers — algorithmic pricing failure

**Context:** Zillow's iBuying business used automated home-price models to make purchase offers.
**Intervention:** The model scaled purchasing decisions.
**Evidence:** In 2021 Zillow announced a write-down of inventory and wound down Zillow Offers, citing inability to forecast home prices accurately at scale. Public reporting described a roughly $300M+ write-down and substantial job losses.
**Outcome:** A well-documented case of model risk meeting capital risk.
**What transferred:** the link between model error and balance-sheet exposure; the need for stress testing and human review of high-variance decisions.
**What did not:** the conclusion that pricing models cannot work; the failure was in scaling and risk limits as much as in the model.

### 6.3 Knight Capital — change control failure

**Context:** A 2012 deployment introduced a code defect that executed unintended orders.
**Intervention:** Automated trading logic.
**Evidence:** The firm lost hundreds of millions of dollars in under an hour and was acquired soon after.
**Outcome:** A canonical case for deployment controls and rollback.
**What transferred:** model change is a risk event; deployment needs gates, canaries, and rollback.
**What did not:** the specifics of trading systems; the principle applies to any automated decision.

### 6.4 Demand model drift after a shock

**Context:** The 2020 pandemic broke historical demand patterns for many categories.
**Intervention:** Models trained on pre-2020 data continued to produce forecasts.
**Evidence:** Many firms reported large bias and manual overrides; some paused models.
**Outcome:** Drift monitoring and regime detection became mainstream.
**What transferred:** a model's validity is conditional on the data regime.
**What did not:** the idea that all history is useless; some stable categories retained signal.

## 7. Labs

### Lab 26.1 — Model inventory and tiering

Inventory ten models in a realistic planning stack. Assign tier, owner, and next review. Justify each tier.

**Deliverable:** Inventory table and a one-page tiering rationale.

**Acceptance criteria:** Decisions are named; materiality and complexity are scored; the riskiest model is identified.

### Lab 26.2 — Conceptual soundness review

Take a demand model and write a soundness review: target, horizon, assumptions, method, inputs, and failure regimes.

**Deliverable:** Two-page review with findings.

### Lab 26.3 — Challenger validation

Compare a production model with a challenger on rolling-origin folds. Report WAPE, bias, pinball, and slice performance. Decide whether the challenger is better.

**Deliverable:** Validation report and recommendation.

**Acceptance criteria:** Same folds and data; uncertainty reported; decision rule stated before results.

### Lab 26.4 — Data-quality audit

Run completeness, accuracy, timeliness, consistency, and representativeness tests on a demand dataset. Quantify each gap.

**Deliverable:** Data-quality scorecard and remediation list.

### Lab 26.5 — Model card and reproducibility audit

Write a model card and attempt to reproduce the model from its artifacts. Record every gap.

**Deliverable:** Model card and reproducibility memo.

### Lab 26.6 — Vendor model evaluation

Score a vendor model against the ten vendor questions. Identify the top three risks and the contractual protections required.

**Deliverable:** Vendor risk memo.

## 8. Assignment / project

Produce a validation dossier for one planning model.

**Required components:**

1. Scope, decision, owner, and tier.
2. Conceptual soundness assessment.
3. Data-quality and lineage review.
4. Reproduction evidence.
5. Outcomes analysis with rolling-origin backtest.
6. Challenger comparison.
7. Sensitivity and stress tests.
8. Performance by slice.
9. Model card.
10. Findings, severity, and remediation plan.
11. Monitoring plan and revalidation triggers.
12. Approval recommendation with conditions.

**Length:** 10–15 pages plus appendix.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Scope and tiering | 10% | Lists models | Correct tier and owner | Tier rationale tied to materiality and autonomy |
| Conceptual soundness | 20% | Describes method | Identifies assumptions and failure regimes | Challenges the target and decision fit |
| Data and lineage | 15% | Describes sources | Tests quality and representativeness | Quantifies gaps and impact |
| Outcomes analysis | 20% | Reports one metric | Rolling-origin and challenger | Decision impact and uncertainty |
| Documentation | 15% | Basic description | Complete model card | Model card usable by a new analyst |
| Monitoring and governance | 10% | Generic plan | Thresholds and triggers | Tied to decision and audit |
| Independence and judgment | 10% | Accepts builder's claims | Independent check | Willing to reject |

## 10. Regulatory, ethical, and commercial lens

- **EU AI Act:** high-risk obligations include risk management, data governance, documentation, logging, human oversight, and accuracy. Determine whether a planning model falls in scope with counsel.
- **Financial reporting:** inventory and revenue models may touch audited figures; validation must satisfy auditors.
- **Consumer protection:** pricing and credit models face fairness and explainability expectations.
- **Labor:** workforce planning models can affect people; document intended use and appeal paths.
- **Vendor contracts:** right to audit, change notification, data ownership, and exit are risk controls.
- **Human oversight:** an override path is not a rubber stamp; it needs authority, information, and accountability.
- **Responsible AI:** fairness across suppliers, stores, and customers; transparency about intended use and limits; accountability for automated decisions; privacy of customer and worker data; and a documented redress path when a model affects a person. These are controls, not a statement of values.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Inventory theater | List not updated | No owner | False assurance | Assign owner, review quarterly |
| Validation capture | Validator reports to builder | No independence | Rubber stamp | Separate reporting line |
| Metric monoculture | Only WAPE reported | Tails ignored | Service failure | Slice and decision metrics |
| Benchmark contamination | Foundation model trained on test | Leakage | Inflated score | Local holdout |
| Silent drift | No monitoring | Regime change | Bad orders | Thresholds and alerts |
| Documentation debt | "We know how it works" | Staff turnover | Unmaintainable | Model card gate |
| Vendor opacity | No method documentation | Black box | Unmanageable risk | Contract terms |
| Change without gate | Direct production edits | No rollback | Incident | Change control |
| Override blindness | Overrides not logged | Unknown behavior | Hidden failure | Log and analyze overrides |
| Revalidation never | Annual review skipped | Stale model | Gradual failure | Calendar and trigger |

## 12. Interview and discussion questions

1. What is model risk in a supply chain context? *(Hint: loss from wrong, misused, or misunderstood models.)*
2. How would you tier a demand model versus a warehouse simulation? *(Hint: materiality and autonomy.)*
3. What is conceptual soundness? *(Hint: fit between design and decision, not accuracy alone.)*
4. How do you test representativeness? *(Hint: slice performance and coverage.)*
5. Why is a random split invalid for time series? *(Hint: leakage.)*
6. When is a challenger model better? *(Hint: pre-registered decision rule, same folds, uncertainty.)*
7. What belongs in a model card? *(Hint: intended use, data, method, performance, limits, monitoring.)*
8. How do you validate a vendor model? *(Hint: documentation, your data, audit rights, exit.)*
9. What triggers revalidation? *(Hint: data, version, regime, breach, incident.)*
10. How do you make validation independent? *(Hint: reporting line and incentives.)*
11. What would you do if a model fails a stress test? *(Hint: restrict use, add buffer, remediate.)*
12. How do you monitor a model whose actuals arrive late? *(Hint: proxy metrics and backfill.)*

## 13. Key terms

| Term | Definition |
|---|---|
| Model risk | Loss from a model being wrong, misused, or misunderstood |
| Model inventory | Register of models, owners, uses, and status |
| Tiering | Assigning control intensity by materiality and complexity |
| Conceptual soundness | Fit between model design and decision |
| Outcomes analysis | Testing model output against realized outcomes |
| Challenger model | An alternative used to test the incumbent |
| Backtesting | Replaying history with the model |
| Stress test | Performance under extreme but plausible conditions |
| Model card | Structured documentation of purpose, data, performance, and limits |
| Reproducibility | Ability to recreate results from artifacts |
| Lineage | Traceable path from data source to output |
| Independence | Validator's freedom from builder pressure |
| Drift | Change in data or relationship that degrades performance |
| Revalidation | Formal re-review after a trigger |
| Issue severity | Ranking of findings by impact |
| Right to audit | Contractual access to vendor evidence |
| Human oversight | Meaningful review and override authority |
| Model change control | Gated process for changing a live model |
| Benchmark contamination | Test data seen during training |
| Slice analysis | Performance by subgroup |

## 14. Sources and further reading

### Standards and regulation

- Federal Reserve, SR 11-7, Guidance on Model Risk Management. https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- OCC, Bulletin 2011-12. https://www.occ.treas.gov/
- PRA, SS1/23, Model risk management principles for banks. https://www.bankofengland.co.uk/prudential-regulation/
- EU AI Act, Regulation (EU) 2024/1689. https://eur-lex.europa.eu/
- ISO/IEC 42001, AI management systems. https://www.iso.org/standard/81230.html
- NIST, AI Risk Management Framework. https://www.nist.gov/itl/ai-risk-management-framework

### Peer-reviewed and professional

- Sculley, D. et al., "Hidden Technical Debt in Machine Learning Systems," NeurIPS, 2015.
- Shmueli, G., "To Explain or to Predict?" *Statistical Science*, 2010.
- Breiman, L., "Statistical Modeling: The Two Cultures," *Statistical Science*, 2001.
- Hyndman, R. J., Athanasopoulos, G., *Forecasting: Principles and Practice*.
- Institute of Internal Auditors, model risk guidance.

### Industry

- Zillow Group, Q3 2021 shareholder letter and Zillow Offers wind-down reporting.
- Knight Capital, SEC order and public reporting on the 2012 incident.
- Bank model risk management practice notes from Deloitte, EY, and KPMG.

### Tools

- Model inventory and model card templates.
- MLflow or equivalent for experiment tracking.
- Great Expectations or dbt tests for data quality.
- Git and container images for reproducibility.

> **Next:** [Module 27](27_datasets_tools_benchmarks.md) grounds the model in datasets, tools, and benchmark evidence.
