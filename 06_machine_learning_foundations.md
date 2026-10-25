# Module 06 — Machine Learning Foundations

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 32 hours (12 lecture, 4 case clinic, 16 lab)
> **Track fit:** Common core
> **Prerequisites:** Module 04; Python; scikit-learn basics helpful
> **Tools:** scikit-learn, XGBoost or LightGBM, pandas, matplotlib, SHAP, Fairlearn
> **Industry question:** "Does this model improve a decision, and can we prove it without fooling ourselves?"

---

## 1. Why industry cares

Most biomedical machine-learning projects fail for mundane reasons:

- The label does not mean what the team thinks it means.
- The split leaks information.
- The metric does not match the decision.
- The model is uncalibrated.
- Performance is not validated outside the development site.
- The threshold is arbitrary.
- Clinicians cannot tell when to trust it.
- Nobody owns it after deployment.

This module builds the habits that prevent those failures. It is deliberately less about exotic algorithms and more about **problem framing, validation, calibration, decision utility, and honest reporting**.

## 2. Learning outcomes

By the end of this module, you can:

- Translate a clinical or operational decision into a prediction task with a defined time zero, target population, label, and action.
- Identify and prevent common forms of data leakage.
- Choose a validation design that matches the deployment claim.
- Build a reproducible preprocessing and modeling pipeline.
- Select models based on discrimination, calibration, clinical utility, robustness, and simplicity.
- Evaluate performance with confidence intervals and subgroup analyses.
- Calibrate probabilities and choose decision thresholds using harms and benefits.
- Explain model behavior without overclaiming causality.
- Quantify uncertainty and detect distribution shift.
- Write a model development report that another team can reproduce.

## 3. Mental model: prediction is a decision aid

```
Decision → Target → Label → Features available at time zero → Split → Model → Calibration → Threshold → Utility → Monitoring
```

Every arrow is a potential failure. The model is only one box. Record exchangeability, label validity, leakage, calibration, and stability assumptions in the Module 01A ledger.

### 3.1 Prediction task specification

| Element | Question | Example |
|---|---|---|
| Decision | What action follows? | Enroll in care management |
| User | Who acts? | Nurse navigator |
| Time zero | When is the prediction made? | Hospital discharge |
| Horizon | How far ahead? | 30 days |
| Population | Who is eligible? | Adults with heart failure |
| Label | What outcome? | Unplanned readmission |
| Features | What is known at time zero? | Demographics, prior utilization, labs |
| Action | What changes? | Transitional care program |
| Cost asymmetry | What does an error cost? | Missed high-risk patient vs wasted outreach |
| Threshold | What risk triggers action? | ≥ 20% |

**Rule:** if you cannot specify time zero, you cannot prevent leakage.

## 4. Labels

### 4.1 Label sources

- Clinical event (e.g., death, stroke).
- Diagnosis code.
- Treatment order.
- Lab threshold.
- Imaging read.
- Chart review.
- Patient-reported outcome.
- Device alarm.
- Claims event.

Each source has a different error structure.

### 4.2 Label quality checks

- Prevalence over time and across sites.
- Agreement with chart review in a sample.
- Sensitivity to code definitions.
- Whether the label includes history or rule-out.
- Whether the label is available at the same time as features.
- Whether the label is a proxy for the true outcome.

### 4.3 Label leakage

Label leakage occurs when the label or its cause is encoded in the features.

Examples:

- A note written after the outcome mentions the outcome.
- A medication order placed because of the outcome.
- A lab measured after time zero.
- A discharge disposition that encodes the outcome.
- A billing code assigned after the event.

**Detection:** inspect feature timing, run ablation with and without suspicious features, and ask a clinician to review the top predictors.

## 5. Data splitting and validation

### 5.1 Split strategies

| Strategy | Use when | Limitation |
|---|---|---|
| Random | IID data, large sample | Leakage with repeated measures |
| Grouped | Multiple records per patient | Requires group identifiers |
| Temporal | Deployment will be later in time | Less data, drift |
| Site-held-out | Multi-site deployment | Site heterogeneity |
| Nested CV | Hyperparameter tuning and performance estimation | Compute cost |
| Bootstrap | Internal validation | Not external validity |

### 5.2 Leakage taxonomy

| Type | Example | Fix |
|---|---|---|
| Temporal | Future lab used to predict past event | Feature cutoff at time zero |
| Patient | Same patient in train and test | Group split |
| Site | Same hospital in train and test | Site-held-out |
| Preprocessing | Imputation/scaling fit on all data | Fit inside CV |
| Label | Post-outcome note | Remove post-time-zero features |
| Duplicate | Same encounter copied | Deduplicate |
| Target encoding | Encoding uses target across folds | Fit within folds |

### 5.3 Nested cross-validation

```
Outer loop: estimate generalization performance
  Inner loop: tune hyperparameters
```

Report the outer-loop performance. Do not tune on the test set.

## 6. Preprocessing

### 6.1 Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
```

Fit the entire pipeline inside cross-validation. Never fit imputers or scalers on the full dataset before splitting.

### 6.2 Categorical variables

- One-hot encoding for low cardinality.
- Target encoding only inside folds, with smoothing.
- Hashing for high cardinality.
- Group rare categories.

### 6.3 Missingness

- Median/mode imputation for simple baselines.
- Multiple imputation for inference.
- Missing indicators only if missingness is informative and not a proxy for protected attributes.
- Tree models can handle missingness natively, but the mechanism still matters.

### 6.4 Temporal features

- Time since last event.
- Counts in windows.
- Trend and variability.
- Seasonality.
- Avoid features that use future data.

## 7. Models

| Model | Strengths | Weaknesses | Use when |
|---|---|---|---|
| Logistic regression | Interpretable, calibrated, stable | Limited nonlinearity | Baseline, small data, regulatory clarity |
| Decision tree | Transparent | Unstable, overfits | Rule generation |
| Random forest | Robust, nonlinear | Large, less interpretable | Tabular baseline |
| Gradient boosting | Strong tabular performance | Tuning, overfitting, opacity | Structured data competitions |
| SVM | Effective in high dimensions | Scaling, calibration | Small/medium data |
| k-NN | Simple | Curse of dimensionality | Low-dimensional similarity |
| Naive Bayes | Fast | Strong independence assumption | Text baseline |
| Neural network | Flexible, multimodal | Data hungry, tuning | Images, signals, text |

**Industry guidance:** start with logistic regression and a gradient-boosted model. The difference in performance is often smaller than the difference in validation quality.

## 8. Class imbalance

- **Do not** blindly oversample before splitting; it leaks.
- Use class weights or threshold tuning.
- Report precision-recall AUC for rare outcomes.
- Report calibration; resampling changes the base rate.
- Consider the cost of false positives and false negatives.
- Evaluate at the operating threshold, not just at 0.5.

## 9. Metrics

### 9.1 Discrimination

- **AUROC:** probability that a random positive is ranked above a random negative. Threshold-independent, prevalence-independent, but not clinically intuitive.
- **AUPRC:** area under precision-recall curve; more informative for rare outcomes.
- **Sensitivity/specificity:** at a chosen threshold.
- **PPV/NPV:** prevalence-dependent.

### 9.2 Calibration

- **Calibration plot:** predicted vs observed risk by decile.
- **Calibration slope and intercept.**
- **Brier score:** mean squared error of probabilities; lower is better.
- **Log loss:** penalizes confident wrong predictions.

### 9.3 Clinical utility

- **Decision curve analysis:** net benefit across threshold probabilities.
- **Net reclassification improvement:** useful but often poorly reported.
- **Number needed to screen/treat.**
- **Alert burden:** alerts per 1,000 patient-days.

### 9.4 Confidence intervals

Report bootstrap or analytic confidence intervals for every primary metric. A single AUC without an interval is not evidence.

## 10. Calibration

### 10.1 Why calibration matters

A model that ranks well but predicts 40% risk when the true risk is 10% will cause over-treatment. Calibration is essential when probabilities drive decisions.

### 10.2 Methods

- **Platt scaling:** fit a logistic regression to the scores.
- **Isotonic regression:** flexible monotonic mapping; needs enough data.
- **Beta calibration:** flexible parametric.
- **Temperature scaling:** for neural networks.

**Rule:** calibrate on data not used for model fitting, and re-check calibration after deployment.

## 11. Threshold selection

The optimal threshold depends on:

- Prevalence.
- Cost of false positive vs false negative.
- Capacity of the intervention.
- Clinician preferences.
- Equity considerations.

**Decision-theoretic threshold:** treat if

\[
p \cdot B > (1-p) \cdot C
\]

where \(p\) is risk, \(B\) is benefit of true positive, and \(C\) is cost of false positive. Rearranged:

\[
p > \frac{C}{B + C}
\]

If the intervention can only handle 50 patients per week, the threshold is also a capacity constraint.

## 12. Interpretability

| Method | What it shows | Limitation |
|---|---|---|
| Coefficients | Direction and magnitude (on model scale) | Assumes model form |
| Odds ratios | Relative odds | Not risk ratios; confounding |
| SHAP | Feature contribution for a prediction | Correlated features, instability |
| Partial dependence | Average effect of a feature | Can create unrealistic combinations |
| ICE | Individual conditional expectation | Many lines, hard to summarize |
| LIME | Local surrogate | Unstable, not causal |
| Attention | Model focus | Not explanation of reasoning |

**Industry guidance:** use interpretability to debug and communicate, not to claim causation. For regulated products, prefer inherently interpretable models when performance is adequate.

## 13. Uncertainty

- **Aleatoric:** irreducible randomness.
- **Epistemic:** uncertainty about the model or parameters.
- **Distributional:** uncertainty about the population.

Methods:

- Bootstrap ensembles.
- Bayesian models.
- Conformal prediction.
- Quantile regression.
- Monte Carlo dropout (with caution).

**Clinical use:** communicate uncertainty in a way that changes action, e.g., "risk 18–32%, recommend clinician review" rather than a false-precision "24.7%."

## 14. Fairness and subgroup performance

### 14.1 Definitions

- **Demographic parity:** equal positive prediction rates.
- **Equal opportunity:** equal true positive rates.
- **Equalized odds:** equal TPR and FPR.
- **Predictive parity:** equal PPV.
- **Calibration within groups:** predicted risk matches observed risk.

These definitions can conflict. Choose based on the decision and the harm.

### 14.2 Analysis

- Report sample sizes and performance by subgroup.
- Check label quality by subgroup.
- Check feature availability by subgroup.
- Examine intersectional groups where sample size permits.
- Involve community stakeholders.

## 15. Model selection

Choose the simplest model that meets the decision requirements:

1. Define the minimum acceptable performance and calibration.
2. Compare a transparent baseline.
3. Test a more flexible model.
4. Evaluate robustness and subgroup performance.
5. Consider maintenance and regulatory burden.
6. Choose based on the full evidence, not a single metric.

## 16. Industry case studies

### 16.1 Sepsis prediction

Sepsis models illustrate label ambiguity, alert burden, and external validation challenges. A model can have good AUROC and still generate too many false alerts for a busy ward.

**Lesson:** evaluate at the workflow threshold and measure alert burden.

Source: Wong, A. et al., *JAMA Internal Medicine*, 2021. https://doi.org/10.1001/jamainternmed.2021.2626

### 16.2 Readmission prediction

Readmission models often have modest discrimination (AUROC ~0.60–0.70) and limited actionability. Their value depends on the intervention and the capacity to deliver it.

**Lesson:** a modest model can be useful if paired with an effective intervention; a strong model is useless without one.

Source: Kansagara, D. et al., "Risk Prediction Models for Hospital Readmission: A Systematic Review," *JAMA*, 2011. https://doi.org/10.1001/jama.2011.1515

### 16.3 Diabetic retinopathy screening

IDx-DR demonstrated that a narrow, well-validated imaging model with a clear workflow can be authorized for autonomous use.

**Lesson:** narrow intended use and prospective evidence beat broad claims.

Source: FDA De Novo DEN180001. https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf

### 16.4 Algorithmic bias in population health

The Obermeyer study showed that a cost proxy produced racial bias in care-management referrals.

**Lesson:** label choice is an ethical decision. Validate the proxy against the true need.

Source: Obermeyer, Z. et al., *Science*, 2019. https://doi.org/10.1126/science.aax2342

### 16.5 COVID-19 deterioration models

Many COVID-19 prognostic models were published early in the pandemic; a systematic review found most had high risk of bias and few were ready for clinical use.

**Lesson:** rapid model proliferation is not evidence. External validation and reporting quality matter.

Source: Wynants, L. et al., "Prediction models for diagnosis and prognosis of covid-19," *BMJ*, 2020. https://doi.org/10.1136/bmj.m1328

## 17. Labs

### Lab 6.1 — Build a leakage-free pipeline

**Objective:** Build a clinical prediction pipeline with proper splits and preprocessing.

**Steps:**

1. Define time zero and features.
2. Create a patient-grouped or temporal split.
3. Build a scikit-learn pipeline.
4. Fit preprocessing inside cross-validation.
5. Report AUROC, AUPRC, Brier score, and calibration.
6. Compare with a baseline.

**Deliverable:** notebook and model card draft.

**Acceptance criteria:** no leakage; confidence intervals; calibration plot.

### Lab 6.2 — Detect leakage

**Objective:** Find planted leakage in a dataset.

**Data:** Instructor adds post-outcome features, duplicate patients, and target encoding leakage.

**Steps:**

1. Inspect feature timing.
2. Run ablation.
3. Check duplicate records.
4. Review top predictors with a clinician.
5. Fix the pipeline and re-evaluate.

**Deliverable:** leakage audit.

**Acceptance criteria:** all planted leaks found; impact quantified.

### Lab 6.3 — Calibration and decision curves

**Objective:** Compare two models on calibration and net benefit.

**Steps:**

1. Plot reliability diagrams.
2. Compute Brier score and calibration slope.
3. Plot decision curves.
4. Choose a threshold based on harm ratio.
5. Explain the choice to a clinician.

**Deliverable:** calibration and decision-curve report.

**Acceptance criteria:** calibration assessed; threshold justified; net benefit interpreted.

### Lab 6.4 — Subgroup fairness audit

**Objective:** Evaluate performance across subgroups.

**Steps:**

1. Define relevant subgroups.
2. Report AUROC, calibration, sensitivity, and PPV by group.
3. Check sample sizes and confidence intervals.
4. Identify mechanisms for disparities.
5. Propose mitigation and monitoring.

**Deliverable:** fairness audit.

**Acceptance criteria:** no claims from tiny subgroups; mechanisms discussed; mitigation feasible.

### Lab 6.5 — Nested cross-validation and model selection

**Objective:** Compare logistic regression, random forest, and gradient boosting with nested CV.

**Steps:**

1. Define an outer loop for performance.
2. Tune hyperparameters in an inner loop.
3. Compare metrics and calibration.
4. Assess stability across folds.
5. Recommend a model.

**Deliverable:** comparison report.

**Acceptance criteria:** no test-set tuning; uncertainty reported; recommendation justified.

### Lab 6.6 — Uncertainty quantification

**Objective:** Add conformal prediction intervals to a risk model.

**Steps:**

1. Split calibration and test data.
2. Compute nonconformity scores.
3. Generate prediction sets at 90% coverage.
4. Evaluate coverage and set size.
5. Discuss clinical use.

**Deliverable:** uncertainty analysis.

**Acceptance criteria:** coverage validated; set size interpreted; limitations stated.

## 18. Assignment: model development report

Write a 2,000-word report:

1. Decision and prediction task.
2. Cohort and label definition.
3. Feature availability and leakage controls.
4. Split strategy.
5. Preprocessing and model.
6. Performance with confidence intervals.
7. Calibration and decision curve.
8. Subgroup performance.
9. Interpretability and limitations.
10. Deployment and monitoring needs.

## 19. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Problem framing | 20% | Predicts a label | Defines decision and time zero | Aligns action, threshold, and capacity |
| Validation | 25% | Random split | Grouped/temporal split | External and prospective plan |
| Metrics | 20% | AUROC only | Discrimination, calibration, utility | Decision-theoretic and subgroup |
| Reproducibility | 15% | Notebook only | Pipeline and seeds | Containerized and versioned |
| Interpretation | 10% | Overclaims | Cautious | Translates to clinical action |
| Communication | 10% | Technical | Clear | Decision-ready |

## 20. Regulatory, ethical, and commercial lens

- **FDA:** a model that informs diagnosis or treatment may be a device. Intended use and claims matter.
- **GMLP:** good machine-learning practice emphasizes representative data, independent test sets, reference standards, and monitoring.
- **Transparency:** model cards and documentation are increasingly expected.
- **Equity:** report subgroup performance; do not hide disparities.
- **Commercial:** a simpler model may be cheaper to validate, explain, and maintain. Total cost of ownership matters.

## 21. Failure modes and red flags

- Label defined after seeing the data.
- Random split with repeated patients.
- Imputation before splitting.
- Using future labs or notes.
- Optimizing AUROC only.
- No calibration.
- Threshold at 0.5 by default.
- No confidence intervals.
- No subgroup analysis.
- Overinterpreting SHAP.
- Ignoring alert burden.
- No monitoring plan.

## 22. Interview and discussion questions

1. **What is data leakage? Give three examples.**
   Hint: temporal, patient, preprocessing, label.
2. **Why is AUROC not enough?**
   Hint: calibration, prevalence, decision threshold, clinical utility.
3. **How do you choose a threshold?**
   Hint: harm ratio, capacity, prevalence, equity.
4. **What is calibration, and why does it matter?**
   Hint: predicted probabilities must match observed risk.
5. **How would you validate a model for a new hospital?**
   Hint: external validation, calibration, subgroup, workflow, monitoring.
6. **What is nested cross-validation?**
   Hint: inner tuning, outer performance estimation.
7. **When would you prefer logistic regression to gradient boosting?**
   Hint: small data, interpretability, calibration, regulatory clarity.
8. **How do you handle class imbalance?**
   Hint: weights, threshold, PR AUC, calibration; avoid pre-split resampling.
9. **What is a decision curve?**
   Hint: net benefit across threshold probabilities.
10. **How do you test for bias?**
    Hint: subgroup performance, label quality, feature availability, intersectional analysis.
11. **What is conformal prediction?**
    Hint: distribution-free prediction sets with coverage guarantees.
12. **What would you monitor after deployment?**
    Hint: data drift, calibration, alert burden, outcomes, subgroup performance.

## 23. Key terms

| Term | Definition |
|---|---|
| Time zero | Moment when prediction is made and features are frozen |
| Label leakage | Target information present in features |
| Group split | Split by patient or cluster to prevent leakage |
| Temporal split | Train on earlier data, test on later data |
| Nested CV | Inner tuning and outer performance estimation |
| AUROC | Area under receiver operating characteristic curve |
| AUPRC | Area under precision-recall curve |
| Brier score | Mean squared error of predicted probabilities |
| Calibration | Agreement between predicted and observed risk |
| Decision curve | Net benefit across thresholds |
| Conformal prediction | Distribution-free uncertainty sets |
| SHAP | Shapley-value feature attribution |
| Subgroup fairness | Performance equity across groups |
| Drift | Change in data or relationships after deployment |

## 24. Sources and further reading

### Methods
- Steyerberg, E. W., *Clinical Prediction Models*.
- Harrell, F. E., *Regression Modeling Strategies*.
- Hastie, T., Tibshirani, R., Friedman, J., *The Elements of Statistical Learning*.
- James, G. et al., *An Introduction to Statistical Learning*.
- Vickers, A. J., Elkin, E. B., "Decision curve analysis: a novel method for evaluating prediction models," *Medical Decision Making*, 2006. https://doi.org/10.1177/0272989X06295361

### Clinical AI
- Wong, A. et al., "External Validation of a Widely Implemented Proprietary Sepsis Prediction Model," *JAMA Internal Medicine*, 2021. https://doi.org/10.1001/jamainternmed.2021.2626
- Kansagara, D. et al., "Risk Prediction Models for Hospital Readmission," *JAMA*, 2011. https://doi.org/10.1001/jama.2011.1515
- Wynants, L. et al., "Prediction models for diagnosis and prognosis of covid-19," *BMJ*, 2020. https://doi.org/10.1136/bmj.m1328
- Obermeyer, Z. et al., "Dissecting racial bias in an algorithm used to manage the health of populations," *Science*, 2019. https://doi.org/10.1126/science.aax2342

### Guidance and tools
- FDA, *Good Machine Learning Practice*. https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
- TRIPOD+AI: https://www.tripod-statement.org/
- scikit-learn: https://scikit-learn.org/
- SHAP: https://shap.readthedocs.io/
- Fairlearn: https://fairlearn.org/
- MAPIE (conformal prediction): https://mapie.readthedocs.io/

> **Next:** [Module 07](07_deep_learning_multimodal_llms.md) extends these foundations to deep learning, multimodal data, LLMs, and agents.
