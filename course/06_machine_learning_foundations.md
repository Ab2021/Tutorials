# Module 06 — Machine Learning Foundations

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 32 hours (12 lecture, 10 lab, 10 project)
> **Track fit:** Common core
> **Prerequisites:** Module 04 (statistics, causal framing); Module 03 (data engineering) recommended; Python, pandas, scikit-learn
> **Tools:** Python 3.12, scikit-learn, LightGBM, XGBoost, SHAP, MAPIE or a conformal implementation, Fairlearn, MLflow
> **Industry question:** "What decision does this prediction change, what does each kind of error cost, and how do we know the model is right for the population it will run on?"

---

## 1. Why industry cares

A prediction model is not a deliverable. It is a component inside a decision: which shipment to expedite, which store-SKU to expedite inventory to, which supplier order to split, which customer order to promise a later date. The model earns its place only if the decision improves and the improvement is measurable.

Three predictions dominate supply chain machine learning:

| Prediction | Decision it changes | Cost of a false positive | Cost of a false negative |
|---|---|---|---|
| Late delivery risk | Expedite, re-route, notify customer | Expedite spend, unnecessary intervention | Penalty, service failure, churn, expedite after the fact |
| Stockout risk | Allocate, transfer, substitute | Excess inventory, obsolescence | Lost margin, lost basket, customer defection |
| Damage risk | Re-pack, change handling, claim | Handling cost, delay | Claim loss, replacement, customer dissatisfaction |

The costs are asymmetric and rarely equal. A late-delivery model tuned at a 0.5 threshold optimises accuracy for the modeller, not cost for the business. Changing the threshold is usually worth more money than changing the algorithm, and it is almost always the last thing a project does.

Four failure patterns account for most unsuccessful deployments:

| Failure | What it looks like | Consequence |
|---|---|---|
| Leakage | Backtest AUC of 0.94, production AUC of 0.71 | The model is deployed against a distribution it never saw |
| Wrong time zero | Features include information unavailable at decision time | The model cannot be run in production, or runs with stale inputs |
| Uncalibrated scores | Probabilities are ranked correctly but numerically wrong | Thresholds and expected-cost calculations are meaningless |
| No decision integration | The score exists; nobody's workflow uses it | Zero value, full maintenance cost |

Regulation now raises the cost of the fourth and third. The EU AI Act (Regulation 2024/1689) requires risk management, data governance, technical documentation, logging, and human oversight for high-risk systems, and a model affecting credit, employment, or access to essential services is squarely in scope. Even outside the Act, a model that changes how a supplier is treated or how a customer's order is prioritised is a governance object, not a notebook.

## 2. Learning outcomes

By the end of this module, you can:

- Define time zero, label, features, action, and cost asymmetry for a supply chain prediction problem.
- Construct a label from operational data and state its acquisition lag, noise, and censoring.
- Detect and eliminate all six leakage classes, and prove the elimination with a leakage audit.
- Choose among random, grouped, temporal, rolling-origin, and nested cross-validation splits and justify the choice.
- Build a preprocessing pipeline whose fit scope excludes validation and test data.
- Select among linear models, trees, forests, gradient boosting, and support vector machines on evidence, not preference.
- Handle class imbalance with resampling, class weights, or cost-sensitive learning, and explain why resampling changes calibration.
- Evaluate with RMSE, MAE, WAPE, AUC, AUPRC, precision, recall, and calibration, and explain when each is misleading.
- Calibrate probabilities, compute the Brier score and its decomposition, and read a decision curve.
- Choose a decision threshold from costs rather than convention, and quantify the value of doing so.
- Interpret models with SHAP and partial dependence, and state the limits of both.
- Quantify prediction uncertainty with conformal intervals and communicate coverage honestly.
- Audit a model for fairness across stores, regions, suppliers, and customer groups.
- Produce a model card that a validator, an auditor, and a planner can all use.

## 3. Mental model

### 3.1 The prediction contract

```
Time zero (t0): the moment the decision is made
   │
   ├─ Features: only what is knowable at t0
   ├─ Label:     the outcome observed after t0, within a defined window
   ├─ Action:    what the decision-maker does with the score
   └─ Cost:      what each error costs, in currency, at the decision
```

Every leakage failure is a violation of the time-zero contract. Every threshold failure is a violation of the cost contract. Every calibration failure is a violation of the probability contract.

### 3.2 Where the money is

```
Model quality   ────────────────►  small effect on value
Threshold choice ───────────────►  large effect on value
Decision integration ───────────►  largest effect on value
```

A 2-point AUC improvement on a late-delivery model typically moves cost by a few percent. Moving the threshold from 0.5 to the cost-optimal point, or routing the score into the expedite workflow at all, moves it by tens of percent. Teams consistently over-invest in the first and under-invest in the second and third.

### 3.3 The modelling loop

```
Decision → t0 → label → features → split → pipeline → model → calibrate
    → threshold → evaluate by segment → document → monitor → retrain
```

The loop runs anticlockwise as often as clockwise: monitoring reveals a segment failure, which sends you back to the label definition, not to the hyperparameters.

## 4. Core concepts

### 4.1 Decision, time zero, label, features, action, cost

| Element | Question | Example: late delivery |
|---|---|---|
| Decision | What changes if the score is high? | Expedite, re-route, or notify |
| Time zero | When is the decision made? | At load departure from the DC |
| Label | What is observed, over what window? | Delivery more than 30 minutes after the window, observed 1–5 days later |
| Features | What is knowable at t0? | Lane, carrier, distance, weight, forecast weather, dock congestion, order priority |
| Action | Who acts, on what threshold? | Transport planner, queue sorted by expected cost |
| Cost | What does each error cost? | False positive $12; false negative $125 |

Writing these six lines before touching data prevents most of the failures in §11.

### 4.2 Labels

A label is a constructed variable, and its construction is a modelling decision.

| Label | Definition choice | Acquisition lag | Main defect |
|---|---|---|---|
| Late delivery | vs requested date, vs confirmed date, vs appointment window | 1–10 days | Definition drift; missing timestamps for late loads |
| Stockout | Zero on-hand at a snapshot, or lost sale inferred | 1 day to never | Censoring; substitution hides the sale |
| Fraud (claims, returns) | Confirmed fraud vs suspected | 30–180 days | Label noise; investigation bias |
| Churn | No order in N days | N days after the fact | Window choice changes the base rate |
| Damage | Claim filed vs claim paid | 30–90 days | Only detected damage is labelled |
| Supplier failure | Missed commit vs late vs short | 1–30 days | Attribution between supplier and carrier |

Three properties must be documented for every label: **base rate**, **acquisition lag** (how long after t0 the label is known, which determines retraining cadence), and **noise mechanism** (why some positives are unlabelled). A model trained on "claims paid" learns the claims process as much as the damage process.

### 4.3 Leakage taxonomy

| Class | Mechanism | Example | Detection |
|---|---|---|---|
| Temporal | A feature is recorded after t0 | Using actual ship date to predict late delivery | Compare each feature's timestamp to t0 |
| Entity | The same entity appears in train and test | Same shipment in both folds via a duplicate key | Group-aware split; check entity overlap |
| Split | Split ignores time or group structure | Random split of a panel with strong seasonality | Compare random-split and temporal-split scores |
| Preprocessing | Scaler or imputer fitted on all data | Target encoding computed before splitting | Pipeline audit; fit inside folds only |
| Target | A feature is a proxy for the label | "Delivery exception code" set when a delay is logged | Univariate AUC per feature; suspiciously high |
| Duplicate | Near-duplicate rows across folds | Repeated order lines from an EDI retry | Hash on business key; deduplicate before splitting |

The single most useful diagnostic is the **suspiciously good result**. If a first model reaches AUC 0.95 on a supply chain outcome, assume leakage until proven otherwise. A useful discipline is to build a deliberately leaky model first, record its score, then remove the leak and treat the gap as the leakage premium.

### 4.4 Splits

| Split | Use when | Failure it prevents | Failure it introduces |
|---|---|---|---|
| Random | i.i.d. rows, no time or entity structure | Nothing in particular | Temporal and entity leakage |
| Grouped (by SKU, store, supplier) | Entities recur across rows | Entity leakage | Fewer groups than desired |
| Temporal (single holdout) | Forward-looking deployment | Temporal leakage | One period may be unrepresentative |
| Rolling origin | Time series and panel forecasting | Single-period luck | More compute; correlated folds |
| Nested CV | Hyperparameter tuning plus performance estimate | Optimistic tuning bias | Cost |
| Stratified group k-fold | Rare events plus entity structure | Class imbalance in small folds | Complexity |

For supply chain deployment, the default should be **grouped and temporal**: hold out future periods and unseen entities. A random split answers "can the model fit rows like these", which is not the question anyone is asking.

### 4.5 Preprocessing pipelines and fit scope

A pipeline binds preprocessing to the estimator so that every fit inside cross-validation refits the preprocessing on that fold's training data only.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

numeric = ["distance_km", "weight_kg", "prior_late_rate_90d", "dock_wait_min"]
categorical = ["carrier", "lane_region", "service_level", "vehicle_type"]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), numeric),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])

model = Pipeline([("pre", pre),
                  ("clf", LGBMClassifier(n_estimators=600, learning_rate=0.03,
                                         num_leaves=31, subsample=0.8,
                                         colsample_bytree=0.8, random_state=42))])
```

Three rules that prevent silent contamination: fit the pipeline, never the transformer, on the training fold; compute target and frequency encodings inside the fold; and use a fixed feature list generated from a point-in-time join rather than a table that is refreshed nightly.

### 4.6 Model families

| Family | Strength | Weakness | Supply chain fit |
|---|---|---|---|
| Linear / logistic | Transparent, stable, fast, well-calibrated by default | Misses interactions and nonlinearity | Baselines, regulated settings, small data |
| Decision tree | Interpretable rules | Unstable, overfits | Rule extraction for policy |
| Random forest | Robust, low tuning | Large, slower, poorly calibrated at the tails | Tabular risk with mixed types |
| Gradient boosting | Best tabular accuracy, handles missing values and interactions | Overfits without care; needs calibration | Late delivery, stockout, damage, fraud |
| Support vector machine | Good in high dimensions, kernel flexibility | Scaling required, slow on large n, no native probabilities | Niche; small, dense problems |
| Neural network | Sequence and high-cardinality structure | Data hungry, tuning intensive | Module 18 territory |

In tabular supply chain problems with tens of thousands to millions of rows, gradient boosting is the default and should be the benchmark against which anything else must justify itself.

### 4.7 Class imbalance and cost-sensitive learning

| Approach | Mechanism | Effect on calibration | When to use |
|---|---|---|---|
| Random undersampling | Drop majority rows | Shifts the base rate; probabilities too high | Fast experiments only |
| Random oversampling | Duplicate minority rows | Overfits duplicates | Rarely |
| SMOTE and variants | Synthetic minority points | Distorts the joint distribution | Contested; avoid for tabular risk |
| Class weights | Weight the loss | Shifts the effective prior | Good default |
| Cost-sensitive threshold | Keep the model, move the decision point | None; model unchanged | Best default when costs are known |
| Focal loss | Down-weights easy examples | Changes the objective | Research settings |

**The key insight:** resampling changes the base rate the model learns, which changes the meaning of the predicted probability. If the deployment population has a 4% late rate and training used 50/50 resampling, a score of 0.5 does not mean a 50% chance in deployment. Calibrate on a representative sample after resampling, or avoid resampling and move the threshold instead.

### 4.8 Metrics

| Metric | Definition | Use when | Fails when |
|---|---|---|---|
| RMSE | √mean((y − ŷ)²) | Large errors are disproportionately costly | Outliers dominate; units unclear |
| MAE | mean(\|y − ŷ\|) | Linear cost of error | Tail cost matters more than average |
| WAPE | Σ\|y − ŷ\| / Σ\|y\| | Volume-weighted accuracy across heterogeneous series | Very low volume; zero totals |
| AUC | Probability a random positive scores above a random negative | Ranking quality, balanced classes | Severe imbalance; ignores calibration |
| AUPRC | Area under precision-recall curve | Rare events (late, fraud, damage) | Not comparable across base rates |
| Precision | TP / (TP + FP) | Intervention cost is high | Ignores missed cases |
| Recall | TP / (TP + FN) | Missed cases are costly | Ignores false alarms |
| Brier | mean((p − y)²) | Probability quality | Scale depends on base rate |
| Calibration error | Mean \|predicted − observed\| by bin | Threshold and expected-cost work | Bin choice affects the value |

**AUC versus AUPRC.** With a 15% base rate, a random classifier has AUC 0.5 but AUPRC ≈ 0.15. AUPRC is therefore the honest summary for rare events, and it is not comparable across populations with different base rates. Report both, plus the base rate.

### 4.9 Calibration: worked example

A late-delivery model is scored on 1,000 shipments held out temporally. Scores are binned by predicted probability.

| Bin | n | Mean predicted | Observed late rate | Gap (observed − predicted) |
|---|---:|---:|---:|---:|
| 1 | 400 | 0.02 | 0.015 | −0.005 |
| 2 | 250 | 0.08 | 0.09 | +0.010 |
| 3 | 180 | 0.18 | 0.22 | +0.040 |
| 4 | 110 | 0.35 | 0.38 | +0.030 |
| 5 | 60 | 0.62 | 0.70 | +0.080 |
| Total | 1,000 | 0.1361 | 0.1519 | +0.0158 |

**Expected calibration error (ECE)**, weighted by bin size:

```
ECE = Σ (n_b/N) × |observed_b − predicted_b|
    = (400×0.005 + 250×0.010 + 180×0.040 + 110×0.030 + 60×0.080) / 1,000
    = (2.0 + 2.5 + 7.2 + 3.3 + 4.8) / 1,000
    = 19.8 / 1,000 = 0.0198
```

The model under-predicts overall: predicted mean 0.1361 against an observed rate of 0.1519, a ratio of 1.116. The miss is concentrated in the top bin, which is exactly where the decisions are made.

**Brier score.** Treating each bin's predicted probability as constant within the bin, the expected squared error decomposes into a calibration term and an irreducible variance term:

```
Brier = Σ (n_b/N) × [ (p_b − o_b)² + o_b(1 − o_b) ]
      = [400×(0.000025+0.014775) + 250×(0.0001+0.0819)
       + 180×(0.0016+0.1716) + 110×(0.0009+0.2356)
       + 60×(0.0064+0.2100)] / 1,000
      = (5.92 + 20.50 + 31.18 + 26.02 + 12.98) / 1,000
      = 96.60 / 1,000 = 0.0966
```

Baseline Brier using the base rate 0.1519 for every shipment: 0.1519 × 0.8481 = 0.1288.

```
Brier skill score = 1 − 0.0966 / 0.1288 = 1 − 0.750 = 0.250
```

The model improves on a constant base-rate forecast by 25% in squared-error terms. That is a real but modest gain, and it is the number to put next to AUC.

**Why this matters for the threshold.** An uncalibrated model with good ranking can still sort the queue correctly, but it cannot support an expected-cost calculation. If the model says 0.62 and the truth in that bin is 0.70, a cost calculation using 0.62 understates the case for intervention by 11%.

### 4.10 Decision curves and Brier decomposition

A decision curve plots net benefit against threshold probability:

```
Net benefit(p_t) = TP/N − (FP/N) × p_t/(1 − p_t)
```

for each candidate threshold, alongside the "treat all" and "treat none" strategies. The model is useful in the region where its curve is above both. This is the cleanest way to show a clinical or operations audience that a model adds value without arguing about AUC.

The Murphy decomposition of the Brier score is `Brier = reliability − resolution + uncertainty`. Reliability (calibration) should approach zero. Resolution (how much the forecast separates outcomes) should be large. Uncertainty is `p̄(1 − p̄)` and is a property of the problem, not the model. Reporting all three explains why two models with equal Brier scores can behave differently.

### 4.11 Threshold selection under asymmetric costs

If a false positive costs `C_FP` and a false negative costs `C_FN`, the expected cost of predicting positive is `(1−p)·C_FP`, and of predicting negative is `p·C_FN`. Predict positive when:

```
p > C_FP / (C_FP + C_FN)
```

**Worked example.** Expediting a shipment unnecessarily costs $12 (rebooking fee and handling). A missed late delivery costs $125 (expedite after the fact plus customer penalty and goodwill).

```
p* = 12 / (12 + 125) = 12 / 137 = 0.0876
```

Flag any shipment above 8.8% predicted risk — not 50%. Evaluating the bin table at each candidate threshold:

| Threshold | Shipments flagged | Late caught | Late missed | Cost (USD per 1,000) |
|---|---:|---:|---:|---:|
| 0.00 (flag all) | 1,000 | 151.9 | 0 | 12,000.00 |
| 0.0876 (cost-optimal) | 350 | 123.4 | 28.5 | 7,762.50 |
| 0.20 | 170 | 83.8 | 68.1 | 10,552.50 |
| 0.50 (default) | 60 | 42.0 | 109.9 | 14,457.50 |

The cost-optimal threshold saves $6,695 per 1,000 shipments against the default 0.5 — $6.70 per shipment, or $670,000 per 100,000 shipments. No change to the model. This is the highest-return hour of work in most supply chain ML projects.

The perfect-prediction floor is 151.9 × $12 = $1,822.80 per 1,000, so there is more value available from better features and labels than from threshold tuning, but the threshold is free and immediate.

### 4.12 Interpretability: SHAP and partial dependence

| Method | What it shows | Limit |
|---|---|---|
| SHAP values | Contribution of each feature to each prediction, additive | Correlated features split credit arbitrarily; not causal |
| Global SHAP summary | Mean absolute contribution per feature | Dominated by high-variance features |
| Partial dependence plot (PDP) | Average prediction as one feature varies | Assumes feature independence; extrapolates off-manifold |
| Accumulated local effects (ALE) | Conditional effect accounting for correlation | Less familiar; still not causal |
| Permutation importance | Loss increase when a feature is shuffled | Correlated features share importance |

**The causal warning.** SHAP explains the model, not the world. If carrier is correlated with lane difficulty, SHAP will attribute late risk to the carrier, and a planner will read that as "this carrier is bad". That is a causal claim the model cannot support. Use SHAP for debugging and for building trust, and use Module 04's methods for causal claims.

### 4.13 Uncertainty and conformal prediction

A point probability is not an interval. Conformal prediction produces distribution-free intervals with a finite-sample coverage guarantee under exchangeability.

For a regression model, split conformal: fit on a training set, compute absolute residuals on a calibration set, take the ⌈(n+1)(1−α)⌉/n quantile of residuals as `q̂`, and report `ŷ ± q̂`. For classification, use a nonconformity score such as `1 − p̂(y|x)` and build prediction sets.

| Property | Consequence |
|---|---|
| Coverage is marginal, not conditional | A 90% interval can under-cover a specific store or lane |
| Exchangeability is required | Distribution shift breaks the guarantee; use weighted conformal under known shift |
| Intervals adapt to difficulty | Wide where the model is unsure — useful for triage |
| Cheap to compute | A calibration set and a quantile |

Report coverage on a held-out period. A nominal 90% interval that achieves 78% in production is a documented failure, not a rounding error.

### 4.14 Fairness

| Dimension | Definition | Supply chain example |
|---|---|---|
| Demographic parity | Equal positive rate across groups | Expedite rate equal across regions |
| Equal opportunity | Equal true positive rate across groups | Late shipments caught equally well in rural and urban stores |
| Predictive parity | Equal precision across groups | Flags equally reliable for small and large suppliers |
| Calibration by group | Predicted probability matches outcome within each group | Model over-predicts lateness for a new carrier |
| Allocation harm | Downstream resource distribution | Low-volume stores receive fewer interventions |

Fairness is not a single metric, and the definitions conflict: a model cannot generally satisfy calibration and equal opportunity simultaneously when base rates differ. The professional task is to name which definition the decision requires, measure it, and document the trade-off.

For supply chain, the highest-risk cases are **supplier scorecards** (a biased model triggers penalties), **delivery equity** (rural or low-income areas systematically deprioritised), and **labour scheduling** (models that predict productivity can encode shift and demographic patterns).

### 4.15 Model selection and documentation

Select on the metric the decision uses, evaluated on the split that matches deployment, with a preference for the simpler model when performance is within noise.

```
1. Establish a naive baseline (base rate, or current rule).
2. Establish a linear or logistic benchmark.
3. Fit gradient boosting with modest tuning.
4. Evaluate on grouped, temporal holdout.
5. Bootstrap the metric difference; keep the simpler model if the CI includes zero.
6. Calibrate; select threshold by cost.
7. Evaluate by segment (store size, region, carrier, supplier).
8. Write the model card.
```

A model card contains: intended use and out-of-scope use; decision and threshold; training population and period; label definition and base rate; features and their time-zero availability; split strategy; metrics with confidence intervals; calibration results; segment performance; fairness audit; known failure modes; monitoring plan and retraining cadence; owner and approver.

## 5. Data, standards, and workflow

| Data object | Grain | Source | ML-specific requirement |
|---|---|---|---|
| Order | Order line | OMS / ERP | Point-in-time snapshot at t0 |
| Shipment | Shipment | TMS | Event timestamps with ingestion time |
| Delivery event | Event | Carrier scan, EPCIS | Label source; check completeness by carrier |
| Inventory snapshot | SKU × location × day | WMS | Snapshot time consistent across sites |
| Master data | Item, location, supplier | PIM / ERP | Versioned; as-of join, not current value |
| Weather | Grid × hour | NOAA or commercial | Forecast vintage, not realised weather |
| Promotion | SKU × store × week | TPM | Known at t0 or excluded |

**Point-in-time correctness** is the central discipline. Every feature table must carry the timestamp at which it was true and the timestamp at which it was loaded, and feature generation must join on `loaded_at <= t0`. A feature store that serves current values to a training set built from history is the most common source of silent leakage at scale.

**Standards.** GS1 GTIN and GLN for entity identity; GS1 EPCIS 2.0 for delivery and custody events, which is also the natural label source where partners support it; ASCM SCOR DS for the metric definitions behind late delivery and perfect order; ISO/IEC 42001 for AI management systems where the model is part of a governed process; NIST AI Risk Management Framework for the risk register structure.

**Workflow.** (1) Write the prediction contract. (2) Build labels with documented lag and noise. (3) Build point-in-time features. (4) Split by group and time. (5) Fit inside a pipeline. (6) Benchmark simple against complex. (7) Calibrate. (8) Select a threshold by cost. (9) Audit segments and fairness. (10) Write the model card and the monitoring plan.

## 6. Industry case studies

### 6.1 The Netflix Prize (2006–2009)

**Context.** Netflix offered US$1 million for a 10% improvement in root mean squared error (RMSE) over its own recommendation baseline, using a large anonymous ratings dataset.

**Design.** A public leaderboard with a held-out quiz set, later supplemented by a final held-out set to limit overfitting to the leaderboard.

**Outcome.** The prize was awarded in 2009 to a large ensemble. The winning solution was a blend of many models that was expensive to run and difficult to integrate, and Netflix did not deploy it in production — a widely reported outcome that became the standard cautionary tale about leaderboard optimisation.

**What transferred.** Three lessons. First, a metric improvement on a static split is not a deployment improvement. Second, ensembles that win competitions can be operationally infeasible. Third, leaderboard overfitting is real: the final holdout mattered, and teams that tuned to the public score degraded on it.

**What did not transfer.** The problem shape. Recommendations are a static, dense, single-stage prediction; supply chain predictions are temporal, sparse, and embedded in a physical workflow with hard constraints.

### 6.2 Google Flu Trends (2008–2015)

**Context.** Google Flu Trends estimated influenza-like illness from search query volumes, published as a near-real-time indicator ahead of official surveillance.

**Design.** A model relating query frequencies to Centers for Disease Control and Prevention (CDC) surveillance data, fitted on historical seasons and updated over time.

**Outcome.** Early performance was strong, but the model substantially overestimated prevalence in later seasons — reportedly by more than 50% during the 2012–13 season — and public updates were discontinued in 2015. Lazer, Kennedy, King, and Vespignani analysed the failure in *Science* in 2014, attributing it to algorithm dynamics, changing search behaviour, and overfitting to a short history.

**What transferred.** The failure mode is exactly the supply chain risk: a model fitted to a stable relationship, deployed into a changing behavioural and operational regime, with no mechanism to detect drift. It is the canonical argument for monitoring, for challenger models, and for holding a simple baseline in production alongside the complex one.

**What did not transfer.** The optimism. A high correlation over a short window is not evidence of a stable relationship, which is why Module 04's out-of-sample discipline applies to ML as much as to causal work.

### 6.3 Amazon's experimental recruiting model (reported 2018)

**Context.** Amazon reportedly built a machine learning system to screen job applicants, trained on a decade of historical applications, most of which came from men.

**Design.** A supervised model predicting hiring success from résumé features, with the historical hiring pattern as the label.

**Outcome.** Reuters reported in October 2018 that the system penalised résumés containing the word "women's" and downgraded graduates of two women's colleges, and that Amazon abandoned the effort. The model had learned the historical pattern rather than the intended criterion.

**What transferred.** The mechanism is label choice: "was hired" encoded a biased process, and the model reproduced it faithfully. Supply chain analogues are direct — a model trained on "was expedited" learns the biases of past expediting decisions, and a supplier scorecard trained on "was penalised" learns who was penalised, not who performed.

**What did not transfer.** The remedy. Removing the offending features is insufficient, because proxies remain. The defensible approach is to define the target as the outcome you want (performance, not hiring), to audit segment performance, and to keep a human decision with appeal rights.

### 6.4 Zillow Offers (2021)

**Context.** Zillow's iBuying business made cash offers on homes using automated valuation models, then resold them.

**Design.** A pricing model estimated home values; the business bought at the model price and carried inventory risk.

**Outcome.** In November 2021 Zillow announced it would wind down Zillow Offers, after a Q3 2021 inventory write-down of approximately US$304 million. The failure combined model error with a market shift and with the fact that the model's errors were correlated across the portfolio.

**What transferred.** Two lessons. First, **cost asymmetry and correlated error**: a valuation model that is unbiased on average can still lose heavily when errors cluster in one direction across inventory. Second, **the model was the decision**, and there was no threshold or human gate between a prediction and a capital commitment.

**What did not transfer.** The domain. Home prices are a thin, slow, and highly correlated market; most supply chain predictions repeat weekly across many SKUs, which gives faster feedback and more opportunities to correct.

## 7. Labs

### Lab 6.1 — Late-delivery risk without leakage

**Objective.** Build a calibrated late-delivery classifier with a proven point-in-time feature set and a cost-based threshold.

**Data.** Course shipment panel: 18 months of shipments with order, departure, delivery, and scan events; carrier, lane, distance, weight, service level, dock wait, and weather forecast vintage. A known-leaky variant is supplied for comparison.

**Steps.** (1) Write the prediction contract for six fields. (2) Build the label and report base rate, lag, and missingness. (3) Build point-in-time features and verify with a timestamp audit. (4) Split temporally and by carrier group. (5) Fit logistic and LightGBM inside pipelines. (6) Compare on AUC, AUPRC, and Brier. (7) Calibrate with isotonic regression on a held-out slice. (8) Select a threshold from the cost ratio and report the cost table. (9) Evaluate by carrier and lane distance band.

**Deliverable.** Notebook, model card, and a one-page cost memo.

**Acceptance criteria.** Every feature has a verified timestamp at or before t0; grouped and temporal split; AUPRC and base rate reported together; calibration plot and ECE; threshold derived from costs with the arithmetic shown; segment table with at least three segments.

**Stretch.** Fit the leaky variant and quantify the leakage premium in AUC; then identify which features caused it.

### Lab 6.2 — Leakage audit

**Objective.** Detect leakage in a supplied model and pipeline.

**Data.** A deliberately flawed repository: random split on a panel, target encoding fitted before splitting, a post-event feature, duplicated rows, and a label proxy.

**Steps.** (1) Run a feature-timestamp audit. (2) Compute univariate AUC per feature and flag outliers. (3) Compare random and temporal split scores. (4) Check entity overlap between folds. (5) Inspect the pipeline for fit-scope violations. (6) Write a findings table with severity and fix.

**Deliverable.** A leakage audit report with a findings table (issue, class, evidence, impact on the metric, fix) and a corrected pipeline.

**Acceptance criteria.** All five planted defects found; each classified correctly; the impact of each quantified or bounded; the corrected pipeline reproduces a materially lower and more credible score.

**Stretch.** Estimate how much of the original score was leakage by removing defects one at a time in sequence.

### Lab 6.3 — Calibration and decision curves

**Objective.** Show that calibration and threshold choice change the value of a model more than a modest AUC improvement does.

**Data.** Scores from Lab 6.1 plus a second, higher-AUC but uncalibrated model.

**Steps.** (1) Compute ECE and Brier for both. (2) Plot reliability diagrams. (3) Compute Brier skill score against the base rate. (4) Plot decision curves for both models plus treat-all and treat-none. (5) Compute total cost at each model's optimal threshold and at 0.5. (6) Report which model wins on AUC and which wins on cost.

**Deliverable.** Notebook and a two-page memo with reliability diagrams, decision curves, and a cost comparison table.

**Acceptance criteria.** ECE and Brier computed on held-out data; Brier skill score correct; decision curve includes both reference strategies; the memo identifies a case where the higher-AUC model is worse on cost.

**Stretch.** Add a conformal prediction set and report empirical coverage against nominal on a later period.

### Lab 6.4 — Fairness audit across stores and suppliers

**Objective.** Measure segment performance and decide whether the model is fit to deploy.

**Data.** Lab 6.1 scores joined to store format, region, and supplier size band.

**Steps.** (1) Compute AUC, AUPRC, precision at the chosen threshold, and calibration by segment. (2) Compute equal-opportunity difference and calibration-by-group difference. (3) Check whether base rates differ by segment and explain the consequence. (4) Run a SHAP analysis and identify whether a segment flag drives predictions. (5) Write a deployment recommendation with conditions.

**Deliverable.** Fairness audit table plus a recommendation memo with named conditions and a monitoring trigger.

**Acceptance criteria.** At least three segment dimensions; both equal-opportunity and calibration-by-group reported; base-rate differences stated; recommendation names a metric threshold that would trigger review.

**Stretch.** Apply a group-specific threshold and show the cost and fairness trade-off explicitly.

## 8. Assignment — Calibrated prediction model and model card

Deliver a deployable late-delivery or stockout-risk model with:

| Artifact | Requirement |
|---|---|
| Prediction contract | Six fields, signed off by a named decision owner |
| Feature specification | Every feature with source, timestamp rule, and null rate |
| Label specification | Definition, base rate, lag, noise mechanism |
| Split design | Grouped and temporal, with justification |
| Model comparison | Naive baseline, logistic, gradient boosting, with bootstrapped differences |
| Calibration report | Reliability diagram, ECE, Brier, skill score |
| Threshold analysis | Cost table across at least four thresholds, with the optimum marked |
| Segment analysis | AUC, AUPRC, precision, calibration by three segments |
| Fairness audit | Two definitions, measured, with the trade-off stated |
| Model card | Intended use, out-of-scope use, limits, monitoring plan, owner |
| Reproduction | Lockfile, seed, data snapshot hash, one-command rerun |

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Problem framing | 20% | Model without a decision | Decision and t0 defined | Contract complete, cost asymmetry quantified, owner signed off |
| Leakage control | 25% | Random split, no timestamp audit | Grouped or temporal split; audit performed | Point-in-time join proven; leakage premium quantified; audit documented |
| Model and evaluation | 20% | One model, accuracy only | Baseline plus one model, several metrics | Bootstrapped comparison; AUPRC and base rate; segment analysis |
| Calibration and threshold | 25% | Probabilities unvalidated; 0.5 threshold | Reliability checked; threshold discussed | ECE, Brier, skill score, decision curve, cost-optimal threshold with arithmetic |
| Documentation and governance | 10% | Notebook only | Basic README | Complete model card, fairness audit, monitoring triggers, reproduction package |

## 10. Regulatory, ethical, and commercial lens

| Area | Requirement or risk | Practical control |
|---|---|---|
| EU AI Act (Regulation 2024/1689) | Risk management, data governance, documentation, logging, human oversight for high-risk systems; enforcement from 2 August 2026 on the current timetable | Model card, decision log, human approval step, retained inputs |
| GDPR / CCPA-CPRA | Minimisation, lawful basis, automated decision rights | Exclude personal identifiers where the decision does not need them; document any automated decision affecting a person |
| NIST AI Risk Management Framework | Govern, map, measure, manage | Use as the risk-register structure for the model |
| ISO/IEC 42001 | AI management system requirements | Align model inventory and change control |
| Supplier fairness | A biased supplier model can trigger penalties | Audit by supplier size and region; provide an appeal path |
| Delivery equity | Optimising cost can deprioritise rural or low-income areas | Report service by region and store format; set a floor |
| Commercial | Model accuracy is not value | Report cost avoided at the chosen threshold, not AUC |

The ethical core is that these models allocate attention and resources. A late-delivery model decides which customer gets an expedite; a stockout model decides which store gets the last case. Neutrality is not achieved by ignoring group membership; it is achieved by measuring outcomes by group and stating the trade-off.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Temporal leakage | Backtest far better than pilot | Post-event features | Production collapse | Timestamp audit; point-in-time joins |
| Random split on a panel | Score drops sharply under temporal split | Seasonality and entity overlap | Overstated performance | Grouped and temporal splits |
| Preprocessing leakage | Encoder fitted on all rows | Statistics include test data | Optimistic CV | Pipelines; fit inside folds |
| Duplicate rows | Near-identical rows across folds | EDI retries, re-sent files | Inflated score | Deduplicate on business key before splitting |
| Target proxy | One feature has AUC 0.9 alone | Exception code encodes the label | Model learns the recording process | Univariate audit; drop and retest |
| Uncalibrated scores | Predicted mean ≠ observed rate | Loss function or resampling changes the prior | Wrong expected cost | Reliability diagram; isotonic or Platt scaling |
| Default threshold | 0.5 with asymmetric costs | Convention, not economics | Large avoidable cost | Cost-based threshold; cost table |
| Imbalance mishandled | Model predicts all negative | Accuracy looks high at 96% | Useless model accepted | AUPRC, recall, cost-sensitive evaluation |
| Drift unmonitored | Input distributions move; no alert | Regime change | Silent decay | Population stability index, performance monitoring |
| Fairness unmeasured | No segment metrics | Historical bias in labels | Discriminatory allocation | Segment audit; named fairness definition |
| Overfitting to leaderboard | Score rises with each submission | Repeated test-set evaluation | Generalisation failure | Holdout untouched until the end |
| No decision integration | Score exists; workflow unchanged | Project ends at the notebook | Zero value | Decision owner, threshold, and workflow from day one |

## 12. Interview and discussion questions

| # | Question | Answer hints |
|---:|---|---|
| 1 | What is time zero, and why does it dominate model design? | The moment of decision; every feature must be knowable then; leakage is a t0 violation. |
| 2 | Your model has AUC 0.94 in backtest. What is your first suspicion? | Leakage. Ask for a feature-timestamp audit and a temporal split. |
| 3 | Why is accuracy a bad metric for late-delivery risk? | Base rate dominates; predicting all-negative scores high; use AUPRC, recall, and cost. |
| 4 | Explain calibration to a transport manager. | If the model says 20% for 100 shipments, about 20 should be late; show the reliability table. |
| 5 | How do you choose a threshold? | Cost ratio: p* = C_FP/(C_FP + C_FN); show the cost table and the saving. |
| 6 | What does SHAP tell you, and what does it not? | Contribution to the model's prediction; not causal effect; correlated features split credit. |
| 7 | How would you detect drift before performance degrades? | Input distribution monitoring, PSI, feature-level alerts, delayed-label monitoring, challenger models. |
| 8 | Why can resampling break a probability model? | It changes the base rate the model learns; probabilities no longer reflect deployment. |
| 9 | What fairness definition would you use for a supplier scorecard? | Equal opportunity and calibration by group, with an appeal path; state the trade-off. |
| 10 | When is a simpler model the right answer? | Within noise of the complex model, regulated context, small data, need for explanation. |
| 11 | How do you know a model is ready for production? | Temporal holdout plus pilot, calibration, segment audit, threshold economics, monitoring plan. |
| 12 | A vendor says its model has 95% accuracy on late delivery. What do you ask? | Base rate, label definition, split, threshold, population, and what 95% refers to. |

## 13. Key terms

| Term | Definition |
|---|---|
| Time zero (t0) | The moment a decision is made and features must be knowable |
| Label | The constructed outcome variable the model predicts |
| Base rate | Prevalence of the positive class in the population |
| Leakage | Information in features that would not be available at t0 |
| Point-in-time join | Joining features as they were known at t0, not as they are now |
| Grouped split | Split that keeps an entity entirely in one fold |
| Temporal split | Split by time, training on the past and testing on the future |
| Rolling origin | Repeated temporal splits advancing the cutoff |
| Nested CV | Inner loop for tuning, outer loop for performance estimation |
| Pipeline | Object binding preprocessing and estimator so fit scope is controlled |
| Class imbalance | Positive class much rarer than negative |
| Cost-sensitive learning | Training or deciding with explicit error costs |
| AUC | Probability a random positive outranks a random negative |
| AUPRC | Area under the precision-recall curve |
| Precision / recall | Share of flags that are correct / share of positives found |
| WAPE | Sum of absolute errors over sum of actuals |
| Calibration | Agreement between predicted probabilities and observed frequencies |
| ECE | Expected calibration error, bin-weighted mean absolute gap |
| Brier score | Mean squared error of probabilities |
| Brier skill score | Improvement over a base-rate forecast |
| Decision curve | Net benefit across thresholds versus treat-all and treat-none |
| SHAP | Shapley-value attribution of a prediction to features |
| Partial dependence | Average prediction as one feature varies |
| Conformal prediction | Distribution-free intervals with marginal coverage |
| Coverage | Share of intervals containing the true value |
| Model card | Structured documentation of use, limits, and monitoring |
| Drift | Change in input or outcome distribution after training |

## 14. Sources and further reading

**Standards and regulation**

- Regulation (EU) 2024/1689 (Artificial Intelligence Act), consolidated text, EUR-Lex. Accessed 27 October 2026.
- NIST, *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, NIST AI 100-1, January 2023.
- ISO/IEC 42001:2023, *Information technology — Artificial intelligence — Management system*.
- ASCM, Supply Chain Operations Reference Digital Standard (SCOR DS). Accessed 27 October 2026.
- GS1, EPCIS 2.0 and General Specifications. Accessed 27 October 2026.

**Peer-reviewed and methodological**

- Lazer, D., Kennedy, R., King, G., and Vespignani, A. "The Parable of Google Flu: Traps in Big Data Analysis." *Science*, 343(6176), 2014.
- Shalev-Shwartz, S., and Ben-David, S. *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press, 2014.
- Hastie, T., Tibshirani, R., and Friedman, J. *The Elements of Statistical Learning*, 2nd ed. Springer, 2009.
- Brier, G. W. "Verification of Forecasts Expressed in Terms of Probability." *Monthly Weather Review*, 1950.
- Murphy, A. H. "A New Vector Partition of the Probability Score." *Journal of Applied Meteorology*, 1973.
- Vickers, A. J., and Elkin, E. B. "Decision Curve Analysis: A Novel Method for Evaluating Prediction Models." *Medical Decision Making*, 2006.
- Niculescu-Mizil, A., and Caruana, R. "Predicting Good Probabilities with Supervised Learning." *ICML*, 2005.
- Lundberg, S. M., and Lee, S.-I. "A Unified Approach to Interpreting Model Predictions." *NeurIPS*, 2017.
- Apley, D. W., and Zhu, J. "Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models." *Journal of the Royal Statistical Society B*, 2020.
- Angelopoulos, A. N., and Bates, S. "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." arXiv:2107.07511, 2021.
- Barocas, S., Hardt, M., and Narayanan, A. *Fairness and Machine Learning*. MIT Press, 2023. Open access.
- Kaufman, S., Rosset, S., Perlich, C., and Stitelman, O. "Leakage in Data Mining: Formulation, Detection, and Avoidance." *ACM Transactions on Knowledge Discovery from Data*, 2012.
- Sculley, D., et al. "Hidden Technical Debt in Machine Learning Systems." *NeurIPS*, 2015.

**Industry and practice**

- Reuters, "Amazon scraps secret AI recruiting tool that showed bias against women," 10 October 2018.
- Zillow Group, Q3 2021 shareholder letter and the November 2021 announcement on winding down Zillow Offers.
- Bennett, J., and Lanning, S. "The Netflix Prize." *KDD Cup and Workshop*, 2007.
- Mitchell, M., et al. "Model Cards for Model Reporting." *FAT\**, 2019.

**Datasets and tools**

- `scikit-learn` user guide, including `Pipeline`, `ColumnTransformer`, and calibration. https://scikit-learn.org — accessed 27 October 2026.
- LightGBM and XGBoost documentation. Accessed 27 October 2026.
- `shap`, `MAPIE`, and `Fairlearn` documentation. Accessed 27 October 2026.
- Scania APS Failure (UCI) and Bosch Production Line Performance (Kaggle) for cost-sensitive, imbalanced practice. Accessed 27 October 2026.

> **Note:** the case outcomes above are reported from public sources. Where a company did not publish a technical evaluation, the mechanism described is inference from the reported facts and is labelled as such — do not cite these as controlled studies of model performance.

---

> **Next:** `07_time_series_forecasting.md` — from prediction on independent rows to prediction along a calendar, with benchmarks, backtests, and honest error metrics.
