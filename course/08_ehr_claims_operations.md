# Module 08 — EHR, Claims, and Operations Analytics

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 28 hours (10 lecture, 4 case clinic, 14 lab)
> **Track fit:** Health Systems core; useful for all tracks
> **Prerequisites:** Modules 02–06
> **Tools:** SQL, Python, pandas, scikit-learn, survival analysis, a BI tool
> **Industry question:** "Which patient, bed, clinic slot, or dollar should we act on next — and what happens if we are wrong?"

---

## 1. Why industry cares

Health systems run on operational decisions made thousands of times per day:

- Which patient needs a call today?
- Which bed will be free at 3 p.m.?
- Which clinic will run late?
- Which claim is likely to be denied?
- Which patient will not show up?
- Which admission can be avoided?
- Which case needs a social worker?

These decisions have direct financial and clinical consequences. They are also amenable to data science because they are frequent, measurable, and often supported by existing data.

This module focuses on the **unglamorous, high-value work** of health-system analytics: cohorting, risk stratification, utilization, capacity, and operations.

## 2. Learning outcomes

By the end of this module, you can:

- Map an operational decision to a data product and owner.
- Build EHR-based cohorts and features with correct temporal windows.
- Analyze claims data for utilization, cost, and care gaps.
- Develop and validate readmission, no-show, length-of-stay, and capacity models.
- Evaluate alert burden and workflow impact.
- Distinguish prediction from intervention effect.
- Use risk adjustment and comorbidity measures appropriately.
- Design operational dashboards that do not mislead.
- Communicate uncertainty to operational leaders.
- Avoid common health-system analytics traps.

## 3. Mental model: the operating system of a health system

```
Patients → Access → Encounters → Diagnostics → Treatment → Discharge → Follow-up
              │          │             │             │           │
           Scheduling   EHR          Labs/Imaging   Orders     Claims
              │          │             │             │           │
           No-show     Documentation  Turnaround     Capacity   Denials
              │          │             │             │           │
              └──────────┴─────────────┴─────────────┴───────────┘
                                   Data
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
               Prediction       Optimization      Monitoring
```

Operations analytics is a loop: measure, predict, act, measure again.

## 4. Decision taxonomy

| Decision | Time horizon | Typical method | Primary metric | Harm of error |
|---|---|---|---|---|
| No-show risk | Days | Classification | Show rate, staff utilization | Empty slot or overbooking |
| Readmission risk | 30 days | Classification/survival | Readmission rate | Wasted outreach or missed care |
| Deterioration | Hours | Time-series classification | Time to response | Delayed care |
| Length of stay | Days | Regression/survival | Bed-days | Poor capacity planning |
| ED demand | Hours–days | Forecasting | Wait time, boarding | Staffing mismatch |
| OR utilization | Days–weeks | Optimization/simulation | Utilization, delays | Idle OR or overtime |
| Claim denial | Weeks | Classification | Denial rate, revenue | Write-off or rework |
| Care gap | Months | Rule-based + prediction | Gap closure | Missed quality bonus |
| Sepsis | Hours | Classification | Time to antibiotics | Mortality |
| High-cost member | Year | Regression/classification | Cost, avoidable use | Misallocated care management |

## 5. EHR cohorts and features

### 5.1 Cohort construction

Define:

- **Index event:** admission, discharge, diagnosis, visit.
- **Eligibility:** age, condition, payer, setting.
- **Exclusion:** hospice, transfers, planned admissions, pregnancy, etc.
- **Observation window:** lookback for comorbidities and prior utilization.
- **Follow-up window:** outcome ascertainment.

### 5.2 Feature families

| Family | Examples | Temporal rule |
|---|---|---|
| Demographics | Age, sex, race/ethnicity, language | At index |
| Utilization | Prior admissions, ED visits, length of stay | Lookback only |
| Comorbidity | Elixhauser, Charlson, HCC | Lookback |
| Vitals | Last, mean, min, max, trend | Before time zero |
| Labs | Last, delta, abnormal flags | Before time zero |
| Medications | Count, class, changes | Before time zero |
| Procedures | Recent procedures | Before time zero |
| Notes | NLP-derived concepts | Before time zero |
| Social | Insurance, area deprivation, living situation | At index |
| Access | Distance, appointment wait time | At index |

### 5.3 Temporal windows

- Last value before time zero.
- Mean/min/max over 24h, 7d, 30d, 1y.
- Delta between first and last.
- Count of abnormal values.
- Time since last event.

**Rule:** every feature must have a timestamp and a rule for inclusion at decision time.

### 5.4 Comorbidity measures

- **Charlson Comorbidity Index:** weighted conditions, predicts mortality.
- **Elixhauser:** 30 comorbidity groups, predicts outcomes and cost.
- **HCC:** CMS hierarchical condition categories for risk adjustment.
- **LACE:** length of stay, acuity, comorbidity, ED visits for readmission.

Use the version appropriate to the population and payment model. Do not mix versions.

## 6. Claims analytics

### 6.1 Claims structure

- **Member:** eligibility, plan, enrollment periods.
- **Medical claim:** provider, place of service, diagnosis, procedure, date, allowed/paid amount.
- **Pharmacy claim:** drug, days supply, quantity, cost.
- **Institutional vs professional claims:** facility vs clinician.
- **Claim lines:** multiple services per claim.

### 6.2 Episodes of care

Group claims into episodes:

- Admission to discharge.
- Procedure plus 90-day follow-up.
- Chronic condition management over a year.

Episode logic must handle transfers, multiple claims, and outliers.

### 6.3 Cost

- **Allowed amount:** negotiated rate.
- **Paid amount:** what the payer paid.
- **Member cost-sharing:** deductible, copay.
- **Total cost of care:** medical plus pharmacy.
- **Risk adjustment:** compare like populations.

**Trap:** cost is a proxy for utilization and price, not health need. High cost may reflect good access or severe illness.

### 6.4 Care gaps

Examples:

- Missing HbA1c in diabetes.
- Missing mammogram.
- Missing statin in cardiovascular disease.
- Missed follow-up after discharge.
- Medication non-adherence.

Care-gap analytics combines rules, claims, and EHR data. It must account for care received outside the network.

## 7. Key models

### 7.1 Readmission

- Define planned vs unplanned.
- Exclude transfers and deaths.
- Use 30-day horizon.
- Include prior utilization and social factors.
- Calibrate; choose threshold based on intervention capacity.
- Evaluate the intervention, not just the model.

### 7.2 No-show

- Label: missed appointment without cancellation.
- Features: prior no-shows, lead time, appointment type, weather, transport, insurance.
- Action: reminders, overbooking, transportation support.
- Metric: show rate, staff idle time, patient access.
- Equity: do not penalize patients with access barriers.

### 7.3 Length of stay and capacity

- Predict expected discharge date.
- Forecast census and admissions.
- Optimize bed assignment.
- Model discharge barriers (social, therapy, placement).
- Use survival or quantile regression, not just mean LOS.

### 7.4 Deterioration

- Use time-series features.
- Define time zero and prediction horizon.
- Evaluate at alert threshold.
- Measure time to response and clinical outcomes.
- Monitor alert fatigue.

### 7.5 ED demand

- Forecast hourly/daily arrivals.
- Include seasonality, day of week, weather, local events.
- Evaluate with MAE/MAPE and operational impact.
- Integrate with staffing and bed management.

## 8. Operational evaluation

### 8.1 Metrics

| Domain | Metrics |
|---|---|
| Clinical | Mortality, readmission, complications, time to treatment |
| Operational | Length of stay, throughput, wait time, utilization |
| Financial | Cost per case, denial rate, revenue, margin |
| Access | Third-next-available, no-show rate, new patient wait |
| Experience | Patient satisfaction, clinician burden |
| Equity | Access and outcomes by subgroup |

### 8.2 Evaluation designs

- Pre/post with comparison group.
- Interrupted time series.
- Stepped-wedge.
- Randomized quality-improvement trial.
- Difference-in-differences.

**Rule:** a dashboard is not an intervention. Measure the action, not the view.

## 9. Industry case studies

### 9.1 Readmission prediction and transitional care

Readmission models are common but often modest. Their value depends on the intervention: medication reconciliation, early follow-up, home visits, or remote monitoring. Studies of transitional care show that the intervention design matters more than the model.

**Lesson:** pair prediction with a proven intervention and evaluate the combination.

Source: Kansagara, D. et al., *JAMA*, 2011. https://doi.org/10.1001/jama.2011.1515

### 9.2 Sepsis alerts in operation

Sepsis alerts can reduce time to treatment when paired with a response team, but they can also create alert fatigue and false positives.

**Lesson:** measure alert burden, clinician response, and outcomes, not only model AUC.

Source: Wong, A. et al., *JAMA Internal Medicine*, 2021. https://doi.org/10.1001/jamainternmed.2021.2626

### 9.3 No-show prediction

No-show models are used to target reminders and transportation support. The ethical risk is penalizing patients with structural barriers.

**Lesson:** the action should reduce barriers, not deny access.

Source: Dantas, L. F. et al., "No-shows in appointment scheduling — a systematic literature review," *Health Policy*, 2018. https://doi.org/10.1016/j.healthpol.2018.02.002

### 9.4 OR scheduling and capacity

Hospitals use predictive and optimization models to improve OR utilization, reduce cancellations, and manage post-anesthesia care unit capacity.

**Lesson:** optimization must include human constraints, turnover, and urgency.

Source: Guerriero, F., Guido, R., "Operational research in the management of the operating theatre: a survey," *Health Care Management Science*, 2011. https://doi.org/10.1007/s10729-010-9143-6

### 9.5 Risk adjustment and value-based care

Value-based contracts compare providers on cost and quality. Risk adjustment models (e.g., HCC) are used to account for patient complexity.

**Lesson:** risk adjustment is a payment policy, not a clinical truth. It can be gamed and can obscure inequity.

Source: CMS, *Risk Adjustment*. https://www.cms.gov/medicare/payment/medicare-advantage-rates-statistics/risk-adjustment

### 9.6 Revenue-cycle leakage and denial analytics

US provider margins remain thin, and denial management has become a hard-dollar AI use case. Industry estimates reported median final denial rates rising from 2.5% to 2.7% between 2024 and 2025, net revenue leakage growing from $38.6B to $48.4B, and hospitals spending about $18B overturning denials in 2025. These are industry estimates, not audited national statistics; verify the underlying reports before using them in a business case.

**Industry lesson:** denial prediction is only useful if paired with a workflow that prevents or appeals the denial before the filing deadline. The model must use only information available at the decision point, and the outcome metric is recovered revenue and clean-claim rate, not AUC.

Sources: Kaufman Hall, *National Hospital Flash Report*, February 2026. https://www.kaufmanhall.com/sites/default/files/2026-04/KH-NHFR_Report-February-2026-Metrics.pdf; HFMA, *Hospital Operating Margin Trends*. https://www.hfma.org/; AHA, *Costs of Caring 2026*. https://www.aha.org/system/files/media/file/2026/03/Costs-of-Caring-2026.pdf

### 9.7 Population health and public health analytics

Provider operations extend beyond individual encounters:

- **Screening and care gaps:** who is overdue for colorectal cancer screening, HbA1c, or hypertension follow-up?
- **Chronic disease registries:** diabetes, heart failure, asthma, CKD.
- **Social risk:** housing, food insecurity, transportation, language.
- **Immunization and outbreak response:** coverage, hesitancy, allocation.
- **Public health surveillance:** syndromic ED data, wastewater, notifiable diseases.
- **Program evaluation:** did the outreach program improve control or equity?

| Decision | Data | Metric | Design caution |
|---|---|---|---|
| Who gets outreach | EHR, claims, registry, SDOH | Gap closure, control rate | Selection and contactability |
| Where to allocate a program | Geography, utilization, deprivation | Reach, equity, cost per case | Ecological fallacy |
| Did the program work? | Pre/post, comparison region | Difference-in-differences | Secular trends |
| Is surveillance signal real? | ED, lab, wastewater | Timeliness, sensitivity | Reporting lag, denominators |

**Rule:** population health metrics require a denominator, a time window, and an equity lens. An aggregate improvement can hide a widening gap.

## 10. Labs

### Lab 8.1 — Build an EHR cohort

**Objective:** Define and extract a readmission cohort.

**Steps:**

1. Define index admission and exclusions.
2. Build a 1-year lookback.
3. Extract demographics, utilization, comorbidity, vitals, and labs.
4. Create a feature dictionary.
5. Check cohort counts and missingness.

**Deliverable:** cohort and data dictionary.

**Acceptance criteria:** temporal rules explicit; exclusions justified; counts reproducible.

### Lab 8.2 — Readmission model with calibration

**Objective:** Train and validate a 30-day readmission model.

**Steps:**

1. Split temporally.
2. Fit a logistic regression and gradient boosting.
3. Evaluate AUROC, AUPRC, Brier, and calibration.
4. Plot decision curves.
5. Choose a threshold based on program capacity.
6. Report subgroup performance.

**Deliverable:** model report.

**Acceptance criteria:** temporal split; calibration; threshold justified.

### Lab 8.3 — Claims cost analysis

**Objective:** Analyze cost and utilization for a chronic condition.

**Steps:**

1. Build episodes of care.
2. Compute total cost of care.
3. Compare by risk group.
4. Identify drivers of cost.
5. Discuss what claims cannot explain.

**Deliverable:** cost analysis.

**Acceptance criteria:** episode logic documented; outliers handled; limitations stated.

### Lab 8.4 — Capacity forecast

**Objective:** Forecast daily ED arrivals or inpatient census.

**Steps:**

1. Explore seasonality and day-of-week effects.
2. Build a baseline and a time-series model.
3. Evaluate MAE/MAPE.
4. Simulate staffing impact.
5. Communicate uncertainty.

**Deliverable:** forecast and operational memo.

**Acceptance criteria:** backtest; uncertainty intervals; operational interpretation.

### Lab 8.5 — No-show intervention analysis

**Objective:** Evaluate a reminder program.

**Steps:**

1. Define treated and comparison groups.
2. Use difference-in-differences or a randomized pilot.
3. Measure show rate and equity.
4. Assess staff workload.
5. Recommend rollout.

**Deliverable:** evaluation report.

**Acceptance criteria:** comparison group justified; equity analyzed; operational cost included.

### Lab 8.6 — Alert burden simulation

**Objective:** Estimate alerts per day at different thresholds.

**Steps:**

1. Use a validation cohort.
2. Compute alert volume at multiple thresholds.
3. Estimate clinician review time.
4. Compare with capacity.
5. Recommend a threshold and escalation path.

**Deliverable:** simulation and recommendation.

**Acceptance criteria:** capacity-based; false-positive burden quantified.

## 11. Assignment: operational decision product

Choose an operational decision and produce:

1. Decision charter.
2. Cohort and label definition.
3. Feature dictionary.
4. Model with temporal validation.
5. Calibration and decision curve.
6. Alert burden analysis.
7. Workflow integration map.
8. Equity analysis.
9. Value case.
10. Monitoring plan.

## 12. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Decision framing | 20% | Model without owner | Clear decision and workflow | Capacity and incentive aligned |
| Cohort and label | 20% | Convenience sample | Explicit definitions | Validated against chart review |
| Model quality | 20% | AUROC only | Calibration and utility | Temporal and subgroup |
| Operational fit | 20% | Ignores workflow | Basic integration | Alert burden and staffing modeled |
| Value and equity | 10% | Generic | Basic analysis | Quantified and actionable |
| Communication | 10% | Technical | Clear | Executive-ready |

## 13. Regulatory, ethical, and commercial lens

- **CDS:** operational models may be CDS or device functions depending on use; triage early.
- **Transparency:** patients and clinicians should know when algorithms influence care.
- **Equity:** no-show and risk models can perpetuate disadvantage. Design actions that expand access.
- **Payment:** value-based contracts create incentives; ensure models do not merely shift costs.
- **Procurement:** hospitals require security, integration, liability, and references.

## 14. Failure modes and red flags

- Predicting readmission without an intervention.
- Using discharge disposition as a feature.
- Mixing index and future utilization.
- Ignoring planned readmissions.
- Optimizing average LOS without considering outliers.
- Overbooking clinics without safety limits.
- Using cost as a proxy for need without validation.
- Reporting alert volume without clinician capacity.
- Ignoring care received outside the network.
- Deploying a dashboard with no owner.

## 15. Interview and discussion questions

1. **How would you build a readmission model?**
   Hint: cohort, exclusions, temporal split, calibration, intervention, threshold.
2. **Why is readmission a problematic target?**
   Hint: planned admissions, competing risk of death, social determinants, gaming.
3. **What is the difference between prediction and intervention?**
   Hint: prediction ranks; intervention changes outcomes; evaluate both.
4. **How do you measure alert burden?**
   Hint: alerts per patient-day, clinician review time, positive predictive value.
5. **What claims data cannot tell you?**
   Hint: severity, labs, adherence, out-of-network care, social context.
6. **How do you risk-adjust cost comparisons?**
   Hint: HCC or comorbidity models; avoid comparing dissimilar populations.
7. **What is a care gap?**
   Hint: recommended care not received; rules plus data.
8. **How would you forecast ED demand?**
   Hint: seasonality, day-of-week, weather, local events, backtesting.
9. **Why can a no-show model be unethical?**
   Hint: penalizes access barriers; action should reduce barriers.
10. **How do you evaluate a capacity intervention?**
    Hint: interrupted time series, comparison unit, operational metrics.
11. **What is the total cost of an alert?**
    Hint: clinician time, false positives, workflow disruption, maintenance.
12. **How do you choose a threshold for a care-management program?**
    Hint: capacity, intervention effect, harm ratio, equity.

## 16. Key terms

| Term | Definition |
|---|---|
| Index admission | Anchor encounter for cohort definition |
| Lookback | Period before index for features |
| Readmission | Unplanned inpatient return within a window |
| LOS | Length of stay |
| No-show | Missed appointment without cancellation |
| Care gap | Recommended care not received |
| HCC | Hierarchical Condition Category risk adjustment |
| Charlson / Elixhauser | Comorbidity indices |
| Episode of care | Grouped services around a condition or procedure |
| Alert burden | Volume and workload created by alerts |
| Capacity | Available beds, staff, or appointment slots |
| Throughput | Flow of patients through a system |
| Risk adjustment | Statistical adjustment for patient complexity |
| Stepped-wedge | Sequential rollout evaluation design |

## 17. Sources and further reading

### Methods and evidence
- Kansagara, D. et al., "Risk Prediction Models for Hospital Readmission: A Systematic Review," *JAMA*, 2011. https://doi.org/10.1001/jama.2011.1515
- Wong, A. et al., "External Validation of a Widely Implemented Proprietary Sepsis Prediction Model," *JAMA Internal Medicine*, 2021. https://doi.org/10.1001/jamainternmed.2021.2626
- Dantas, L. F. et al., "No-shows in appointment scheduling — a systematic literature review," *Health Policy*, 2018. https://doi.org/10.1016/j.healthpol.2018.02.002
- Guerriero, F., Guido, R., "Operational research in the management of the operating theatre: a survey," *Health Care Management Science*, 2011. https://doi.org/10.1007/s10729-010-9143-6

### Policy and data
- HFMA, *Hospital Operating Margin Trends*. https://www.hfma.org/
- AHA, *Costs of Caring 2026*. https://www.aha.org/
- CMS Risk Adjustment: https://www.cms.gov/medicare/payment/medicare-advantage-rates-statistics/risk-adjustment
- CMS Quality Measures: https://www.cms.gov/medicare/quality/measures
- AHRQ HCUP: https://hcup-us.ahrq.gov/
- Elixhauser Comorbidity Software: https://hcup-us.ahrq.gov/toolssoftware/comorbidity/comorbidity.jsp

### Tools
- scikit-learn: https://scikit-learn.org/
- statsmodels: https://www.statsmodels.org/
- Prophet: https://facebook.github.io/prophet/
- OR-Tools: https://developers.google.com/optimization

> **Next:** [Module 09](09_clinical_nlp.md) addresses the unstructured majority of clinical data: text.
