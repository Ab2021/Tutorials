# Systematic Experiment Tracking in Supply Chain Machine Learning

> Module 44 — MLOps and Production Analytics Engineering
> Topic 03 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Topic 01 (Lifecycle and Problem Framing), Topic 02 (Versioning and Reproducibility), Module 07 (Time Series Forecasting), Module 08 (Error Metrics)
> Version 2026.5

---

## 1. Why this topic exists

In classical machine learning workflows, experiment tracking is often treated as a straightforward scalar logging exercise: a data scientist fits a classifier or tabular regressor on a randomly partitioned holdout split, logs the validation loss, accuracy, or root mean squared error (RMSE) to an experiment registry, and selects the candidate run with the lowest scalar loss. In enterprise supply chain analytics, applying this naive paradigm is disastrous.

Physical supply chains operate across complex spatial, temporal, and organizational topologies. A multinational consumer goods manufacturer or omni-channel retailer does not evaluate a single model on an isolated dataset. Instead, forecasting and replenishment pipelines orchestrate models across hundreds of thousands of Stock Keeping Units (SKUs), dozens of central and regional distribution centers (DCs), thousands of retail stores, and multiple planning horizons ranging from 2-day short-term dispatch to 52-week master production scheduling. 

When experiment tracking lacks structural awareness of supply chain hierarchies, five systemic engineering breakdowns occur:

1. **Hierarchy Disconnection and Context Loss:** An aggregate experiment run cannot be meaningfully interpreted as a flat table of metrics. If an experiment improves overall national demand accuracy by 1.2% but degrades store-level forecasts for high-margin, short-shelf-life perishable SKUs by 14%, an un-stratified tracking system reports a false net improvement. Tracking must capture the multi-echelon Directed Acyclic Graph (DAG) relationship connecting parent pipeline runs to child SKU-location models.
2. **Temporal Leakage and Pseudo-Replication:** Standard random cross-validation shuffles data across time, allowing future promotional events, supplier shutdowns, and macro shocks to leak into historical feature windows. Even simple chronological splits fail when multi-step lead times cause autocorrelation leakage: if a distribution center orders inventory with a 3-week lead time, forecast errors at week $t$ and week $t+2$ share common unobserved replenishment shocks. Without rigorous purge and embargo protocols, experiment metrics reflect lookahead bias rather than genuine generalization.
3. **Loss Function Asymmetry and Economic Misalignment:** Standard regression metrics (Mean Squared Error, Mean Absolute Percentage Error) treat positive and negative forecast errors symmetrically. In physical supply chains, the operational cost of error is deeply asymmetric. An under-forecast triggers stockouts, lost sales margins, expedited emergency freight fees, and customer defection penalties ($c_u$). An over-forecast triggers working capital lockup, automated warehousing congestion, and perishable spoilage or secondary markdown salvage losses ($c_o$). An experiment tracker logging symmetric RMSE inevitably selects models that minimize variance while systematically driving distribution centers into stockout crises.
4. **Statistical Indeterminacy in Candidate Promotion:** Machine learning practitioners frequently promote challenger models based on nominal 0.4% or 0.8% reductions in Weighted Absolute Percentage Error (WAPE) over a 12-week backtest. Without rigorous time-series hypothesis testing—such as the Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample corrections or the Giacomini-White conditional predictive ability framework—supply chain teams confuse stochastic temporal noise with authentic model superiority. This leads to costly retraining churn, unnecessary pipeline complexity, and operational regressions in live ERP environments.
5. **Downstream ERP and Financial Audit Liabilities:** In modern automated supply chains, machine learning predictions feed directly into Enterprise Resource Planning (ERP) engines such as SAP Integrated Business Planning (IBP), Blue Yonder, or Manhattan Associates WMS. These engines automatically trigger electronic purchase orders, allocate automated line-haul trailers, and commit millions of dollars in working capital. When model runs lack cryptographic data snapshot lineage and verified parameter tracking, discrepancies between forecast models and audited physical inventory balances trigger Sarbanes-Oxley (SOX) Section 404 material weakness disclosures and corporate balance-sheet write-downs.

### 1.1 Multi-Horizon Planning Mismatch and Lead-Time Dynamics

Supply chain planning takes place across fundamentally discordant operational cadences:
- **Short-Term Tactical Dispatch ($h = 1$ to $3$ days):** Focuses on warehouse labor allocation, dock door scheduling, and cross-dock trailer dispatching. Errors at this horizon trigger overtime labor premiums and detention fines.
- **Medium-Term Replenishment ($h = 2$ to $6$ weeks):** Focuses on domestic supplier purchase orders and distribution center safety stock sizing. Errors at this horizon directly drive stockouts and warehouse congestion.
- **Long-Term Procurement and Capacity Planning ($h = 13$ to $52$ weeks):** Dictates overseas ocean freight contracts, manufacturing line tooling, and supplier volume commitments.

Standard experiment tracking setups log an aggregate loss across all horizons simultaneously. This obscures horizon-specific model performance: a neural network architecture may excel at 52-week seasonal trend capture while exhibiting severe lag inertia on 3-day daily dispatch. Systematic experiment tracking decomposes evaluation metrics across explicitly tagged horizon sub-runs, ensuring that production dispatchers and strategic planners evaluate models tailored to their operational lead times.

### 1.2 Promotional Cannibalization and Cross-SKU Spillover

Retail and consumer goods demand forecasting is heavily disrupted by trade promotions, temporary price reductions (TPRs), and marketing campaigns. Standard experiment tracking systems evaluate candidate models on isolated time-series cuts for individual SKUs. 

This creates a severe blind spot:
- **Cannibalization:** Promoting Brand A's 2-liter beverage often depresses sales of Brand B's substitute product within the same category. A model evaluated solely on Brand A's run may appear highly accurate while failing to capture the negative demand shock inflicted on Brand B.
- **Halo Effects:** Promoting pasta often drives complimentary demand surges for pasta sauce and grated cheese across adjacent aisles.

When experiment tracking platforms log only localized leaf metrics without cross-SKU attribution tracking, models that induce catastrophic portfolio-wide forecast errors are mistakenly promoted. Comprehensive tracking links related SKU runs via category parent DAGs, logging portfolio-level cross-elasticity and net revenue impact.

Systematic experiment tracking in supply chains provides an immutable, reproducible audit ledger that captures hierarchical pipeline topologies, enforces leakage-free rolling backtest protocols, links model performance directly to asymmetric inventory loss economics, and executes rigorous statistical significance gates before any model artifact enters candidate staging.

---

## 2. Learning outcomes

By completing this topic, you will be able to:

1. **Architect** hierarchical experiment tracking topologies in MLflow, Weights & Biases, or cloud-native registries that map parent network pipeline executions down to multi-echelon, multi-horizon SKU-location child runs.
2. **Formulate** comprehensive parameter logging schemas that capture data snapshot identifiers, feature store schema hashes, Git commit SHAs, environment lockfile digests, and hierarchical reconciliation matrices.
3. **Establish** an end-to-end artifact lineage pipeline logging model binaries, quantile calibration curves, error confusion matrices, TreeSHAP summary beeswarm plots, and residual autocorrelation diagnostic plots.
4. **Implement** rolling and expanding temporal backtest evaluation harnesses that enforce strict purge and embargo window protocols to eliminate lookahead bias and serial correlation leakage.
5. **Construct** asymmetric piecewise linear inventory loss functions that map forecast deviations directly to enterprise unit shortage ($c_u$) and holding ($c_o$) cost parameters.
6. **Compute** the Diebold-Mariano (DM) test statistic and apply the Harvey-Leybourne-Newbold (HLN) small-sample correction for multi-step forecast horizons ($h > 1$) with rectangular and Bartlett Newey-West spectral density kernels.
7. **Execute** the Giacomini-White (GW) test of Conditional Predictive Ability to determine whether a challenger forecasting model outperforms a production champion conditioned on economic state variables such as promotion schedules and lead-time volatility.
8. **Control** the False Discovery Rate (FDR) using the Benjamini-Hochberg procedure when conducting statistical model evaluation across catalogs containing tens of thousands of SKU-location child runs.

---

## 3. The core mental model

The foundational mental model for supply chain experiment tracking is the **Hierarchical Temporal Evaluation Lattice**. 

An enterprise supply chain model run is not an isolated point in time; it is an orchestrated execution node within a two-dimensional lattice defined by **Hierarchy** (aggregation depth from enterprise network down to individual SKU-location nodes) and **Time** (rolling temporal backtest cutoffs spanning historical replenishment cycles).

```text
========================================================================================================================
                                  HIERARCHICAL TEMPORAL EXPERIMENT TRACKING LATTICE
========================================================================================================================

 [ EXPERIMENT REGISTRY & METADATA STORE ]
   │
   ├── Git Commit SHA: e9f41b8a | Delta Snapshot ID: 2026-10-15-v14 | Feature View Hash: d41d8cd9
   │
   ▼
 [ PARENT PIPELINE RUN ] (Global Pipeline / Network Level Orchestration)
   │ 
   │  Global Parameters: { loss: "asymmetric_pinball", opt: "adamw", lr: 0.001, reconciliation: "MinT" }
   │  Global Artifacts:  { reconciliation_matrix.parquet, aggregated_SHAP_summary.png, global_dm_test.json }
   │
   ├─── CHILD RUN 1: Echelon Level 1 (National Central Distribution Center)
   │     │  Parameters: { lead_time_weeks: 4, aggregation_level: "CDC", smoothing_alpha: 0.2 }
   │     │  Metrics:    { CDC_WAPE: 0.084, CDC_Underage_Cost: $14,200, CDC_Holding_Cost: $4,100 }
   │     │
   │     ├─── SUB-RUN 1.1: Horizon h=1 (Weekly Tactical Dispatch)
   │     └─── SUB-RUN 1.2: Horizon h=4 (Monthly Replenishment Order)
   │
   └─── CHILD RUN 2: Echelon Level 2 (Regional DC / Store Clusters)
         │  Parameters: { cluster_id: "RDC_Northeast", count_skus: 4200, intermittent_threshold: 0.4 }
         │  Metrics:    { RDC_WAPE: 0.142, RDC_Underage_Cost: $52,400, RDC_Holding_Cost: $18,300 }
         │
         ├─── LEAF RUN 2.1.1: SKU 90412 @ RDC-04 (High-Velocity Class A)
         └─── LEAF RUN 2.1.2: SKU 10488 @ RDC-04 (Intermittent Tail Class C)

========================================================================================================================
                             TEMPORAL BACKTEST SPLIT PROTOCOL (PURGE AND EMBARGO)
========================================================================================================================

 Time Axis ────────────────────────────────────────────────────────────────────────────────────────────────────────────►
 
 [ FOLD k-1 ]  │◄──────── TRAIN SET (T_train) ────────►│◄─ PURGE ─►│◄─ TEST (h) ─►│◄─ EMBARGO ─►│
               │                                       │ (Lag Cut) │ (Forecast)   │ (Lead Time)  │
               └───────────────────────────────────────┴───────────┴──────────────┴──────────────┘
                                                                   ▲              ▲
                                                             Cutoff Origin    Evaluation
 
 [ FOLD k ]    │◄────────────── TRAIN SET (T_train + Step) ──────►│◄─ PURGE ─►│◄─ TEST (h) ─►│◄─ EMBARGO ─►│
               │                                                  │ (Lag Cut) │ (Forecast)   │ (Lead Time)  │
               └──────────────────────────────────────────────────┴───────────┴──────────────┴──────────────┘
                                                                              ▲              ▲
                                                                        Cutoff Origin    Evaluation

========================================================================================================================
                          STATISTICAL COMPARISON & CANDIDATE PROMOTION GATE
========================================================================================================================

 Candidate Run (Challenger) ──┐
                              ▼
 Production Run (Champion)   ───► Loss Differential d_t = L(y_t, y_hat_Champ) - L(y_t, y_hat_Chall)
                                    │
                                    ├─── Step 1: Autocovariance Analysis (Lag 0 to h-1)
                                    ├─── Step 2: Diebold-Mariano Test (Rectangular / Bartlett Kernel)
                                    ├─── Step 3: Harvey-Leybourne-Newbold Small-Sample Correction
                                    └─── Step 4: Giacomini-White Conditional Predictive Ability Test
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
        p-value < alpha (Reject H0)                   p-value >= alpha (Fail to Reject)
     [ CANDIDATE STAGED FOR CANARY ]               [ CANDIDATE REJECTED / LOGGED ]
========================================================================================================================
```

The experiment tracking system serves as the immutable engine of this lattice. It coordinates:
- **Parent Runs** that encapsulate the global code revision, the exact state of data tables, the reconciliation constraints across nodes, and the network-wide statistical test outputs.
- **Child Runs** that capture node-specific hyperparameter configurations, localized error metrics, residual autocorrelations, and local feature attribution values (TreeSHAP).
- **Temporal Slices** that prevent data leakage by isolating training, purge, test, and embargo windows across rolling cutoff origins.
- **Statistical Gates** that extract the paired time-series loss differentials across matched temporal slices to establish whether candidate performance gains are mathematically significant or artifacts of chance.

### The Dual-Lattice Information and Control Flow

Data and control signals propagate through this lattice in two distinct directions:
1. **Top-Down Parameter Inheritance:** The Parent Run dictates structural constraints, such as the minimum training window, the specific Delta snapshot version, and the mathematical reconciliation method (e.g., MinT shrinkage). Child runs inherit these configurations automatically, ensuring that no worker node trains on an unpinned data partition.
2. **Bottom-Up Metric and Diagnostic Rollup:** Leaf nodes evaluate localized SKU-location loss differentials. Instead of flooding the metadata database with millions of scalar rows, leaf runs persist high-dimensional loss arrays and SHAP matrices to cloud object storage (S3/GCS/Azure Blob) as Parquet artifacts. Summary statistics and covariance tensors then roll up into intermediate echelon runs, allowing engineers to query performance at any level of aggregation without UI latency collapse.

### 3.1 The Model Artifact Lifecycle State Machine

Within the experiment tracking registry, model artifacts transition through an explicit finite state machine:
- **`DEVELOPMENT`:** Exploratory runs authored by data scientists. Hyperparameters and code commits are logged, but statistical gates are not yet enforced.
- **`BACKTEST_EVALUATED`:** Candidate pipeline executions that have completed the full rolling origin evaluation with purge and embargo buffers. Loss differential series $d_t$ and diagnostic Parquet artifacts are fully persisted.
- **`GATE_PASSED`:** Candidates that have satisfied the Diebold-Mariano test with Harvey-Leybourne-Newbold correction ($p < 0.05$) and cleared catalog-scale Benjamini-Hochberg FDR filtering.
- **`CANARY_STAGED`:** Models receiving live shadow or partial production traffic (e.g., 5% of store-cluster purchase orders) to verify latency, inference memory stability, and operational integration.
- **`PRODUCTION_CHAMPION`:** The active model driving automated ERP replenishment. Its artifacts serve as the benchmark against which subsequent challengers are tested.
- **`ARCHIVED`:** Deprecated champion models preserved with immutable environment digests and Delta snapshot pointers for regulatory audits and historical post-mortems.

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Misuse | Unit / Standard |
|---|---|---|---|
| **Parent-Child Run Hierarchy** | A nested tracking structure where a parent run encapsulates global execution context and child runs represent sub-models across echelons, clusters, or horizons. | Treating child runs as independent experiments, breaking DAG lineage and aggregate traceability. | Hierarchy Tree (UUID DAG) |
| **Purge Window** | The deliberate deletion of historical training observations immediately preceding the forecast origin whose feature windows overlap with the forecast horizon. | Confusing purging with embargoing; omitting the purge window causes direct feature lookahead leakage. | Calendar Days or Weeks |
| **Embargo Window** | A temporal buffer period enforced immediately after a test window during which no training samples are drawn, preventing autocorrelation leakage. | Skipping embargo under the false belief that chronological separation prevents serial dependency. | Calendar Lead Time ($L$) |
| **Autocorrelation Leakage** | The contamination of out-of-sample evaluation caused by serially dependent error terms across overlapping multi-step replenishment cycles. | Assuming independent and identically distributed ($i.i.d.$) errors across consecutive multi-step forecast periods. | Durbin-Watson / Autocovariance |
| **Loss Differential ($d_t$)** | The arithmetic difference in point-in-time economic or statistical loss between two competing models: $d_t = L(y_t, \hat{y}_{1,t}) - L(y_t, \hat{y}_{2,t})$. | Comparing scalar summary losses (e.g., aggregate RMSE) without preserving the paired time-series sequence. | Currency / Units ($USD, EUR$) |
| **Asymmetric Loss Function** | A piecewise linear or non-linear loss metric applying different penalties to positive errors (under-forecast) and negative errors (over-forecast). | Using symmetric quadratic loss (MSE) for supply chain models where stockouts cost $10\times$ inventory holding costs. | Cost Matrix ($c_u, c_o$) |
| **Diebold-Mariano (DM) Test** | An asymptotic parametric hypothesis test evaluating the null hypothesis of equal forecast accuracy between two competing models for time series. | Applying standard paired $t$-tests to multi-step forecast errors without accounting for serial correlation. | Test Statistic $DM \sim \mathcal{N}(0, 1)$ |
| **Harvey-Leybourne-Newbold (HLN)** | A small-sample finite-size correction to the Diebold-Mariano test statistic that inflates variance and evaluates against Student's $t$-distribution. | Assuming asymptotic normality on short backtest horizons ($T < 30$), yielding severe Type I false rejection errors. | Critical Value $t_{T-1}$ |
| **Giacomini-White (GW) Test** | A conditional predictive ability test evaluating whether model superiority is predictable conditioned on information available at the forecast origin. | Conflating unconditional accuracy superiority with operational decision superiority across varying economic regimes. | Wald Statistic $\sim \chi^2_q$ |
| **Spectral Density at Frequency Zero** | The Fourier transform of the autocovariance sequence evaluated at frequency zero, representing the long-run variance of a serially correlated process. | Estimating time-series sample variance using simple sample variance formulas that ignore autocovariance terms. | Long-Run Variance $\hat{V}(\bar{d})$ |
| **Rectangular Kernel** | An unweighted autocovariance truncation kernel capping covariance terms at lag $h-1$ for theoretical multi-step moving-average processes. | Applying a rectangular kernel to small samples without checking for positive semi-definiteness of the variance estimate. | Truncation Lag $h-1$ |
| **Bartlett Kernel (Newey-West)** | A linearly decaying weighting kernel that guarantees positive semi-definiteness of the long-run variance estimator in finite time series. | Using an overly wide bandwidth parameter that inflates standard errors and destroys test power. | Weight $w_k = 1 - \frac{k}{K+1}$ |
| **TreeSHAP Lineage** | The systematic persistence of tree-based Shapley Additive Explanations (summary beeswarms, mean absolute attribution matrices) across experiment runs. | Logging only static global feature importance, concealing feature attribution drift across customer segments or seasons. | Attribution Vector ($\phi_j$) |
| **Hierarchical Reconciliation Matrix** | A mathematical projection matrix (e.g., MinT, Bottom-Up, Top-Down) that maps incoherent base forecasts across nodes into coherent echelon forecasts. | Tracking base child forecasts without logging the reconciliation matrix used to produce final ERP dispatch quantities. | Projection Operator $\mathbf{S}(\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1}\mathbf{S}'\mathbf{W}^{-1}$ |
| **Minimum Trace (MinT)** | An optimal hierarchical forecast reconciliation approach that finds the linear unbiased combination minimizing total forecast error variance. | Implementing ad-hoc proportional post-scaling that distorts variance estimates and violates statistical unbiasedness. | Covariance Shrinkage Matrix $\mathbf{W}$ |
| **Artifact Digest** | A cryptographic hash (SHA-256) computed over serialized model weights, calibration matrices, and configuration files. | Tracking models by file path alone, causing silent overwrite errors when pipelines re-execute on identical paths. | Hexadecimal Hash (SHA-256) |
| **Rolling Origin Backtest** | An evaluation protocol where the forecast cutoff advances sequentially through historical time, re-generating out-of-sample forecasts at each step. | Performing a single train-test split on time series data, mistaking seasonal alignment for true model stability. | Time Steps ($t = 1 \dots T$) |
| **Information Set ($\mathcal{F}_t$)** | The filtration representing all observable operational data (demand, promotions, orders, macro indicators) available at forecast origin $t$. | Conditioning models on features that incorporate unreleased or unobserved forward-looking data. | Filtration $\sigma$-algebra |
| **False Discovery Rate (FDR)** | The expected proportion of false positive rejections among all rejected null hypotheses when testing multiple SKU models simultaneously. | Uncritically accepting individual $p < 0.05$ thresholds across 50,000 SKUs, generating thousands of false model promotions. | Benjamini-Hochberg $q$-value |
| **Pinball Loss** | An asymmetric piecewise linear loss function parametrized by quantile target $\tau \in (0, 1)$, equivalent to the newsvendor cost function. | Conflating quantile loss with mean absolute error, leading to symmetric replenishment allocations. | Quantile Deviation ($USD$) |
| **Ljung-Box Test** | A diagnostic portmanteau test evaluating whether a group of residual autocorrelations are jointly different from zero. | Assuming residuals are white noise without formal testing, missing persistent seasonal patterns or lag leakage. | Test Statistic $Q \sim \chi^2_m$ |
| **Censored Demand** | Observed historical sales that underestimate true customer demand due to inventory stockouts and shelf depletion. | Training and tracking forecasting models directly on historical point-of-sale sales transactions without stockout imputation. | Truncated Quantity ($y_t \le d_t^*$) |

---

## 5. The mechanics, step by step

Systematic supply chain experiment tracking operates across five sequential engineering phases: hierarchical DAG architecture, parameter/schema logging, artifact lineage capture, leakage-free temporal backtesting, and rigorous statistical significance verification.

### 5.1 Architecture of Hierarchical DAG Tracking

Enterprise supply chains cannot track experiments as flat, decoupled runs. A change in a global parameter—such as the loss function formulation or the macro lag structure—cascades down through every regional warehouse and store-level model. The tracking system must enforce a parent-child DAG topology.

1. **Root Run Initialization:** The orchestration pipeline (e.g., Apache Airflow, Prefect, or Kubeflow) instantiates an overarching Root Experiment Run. The root run logs:
   - Git Commit SHA of the modeling repository.
   - Pointers to data snapshot manifests (e.g., Apache Iceberg snapshot ID, Delta Lake version timestamp).
   - Environment definition digests (e.g., Docker image digest, Conda lockfile SHA-256).
   - Global modeling configuration files (YAML/JSON schema).
2. **Echelon Parent Runs:** Beneath the root run, intermediate parent runs represent organizational or supply network tiers:
   - National Network / Central Distribution Center (CDC) run.
   - Regional Distribution Center (RDC) cluster runs.
   - Retail Store cluster or Channel (e.g., E-commerce vs B2B Wholesale) runs.
3. **Leaf Node Execution (SKU-Location Runs):** Each leaf node executes forecasting or inventory optimization for a specific SKU-Location pair (or cluster of intermittent SKUs). The leaf run registers its lineage back to the immediate echelon parent and the root run via structured parent run IDs:
   $$\text{Run\_ID}_{\text{leaf}} \longrightarrow \text{Parent\_ID}_{\text{echelon}} \longrightarrow \text{Root\_ID}_{\text{network}}$$

#### Hierarchical Forecast Reconciliation Tracking
When forecasting across multi-echelon networks, bottom-level forecasts do not naturally sum to top-level forecasts. Let $\mathbf{y}_t$ be the vector of all series at time $t$, consisting of $m_{top}$ aggregate series and $m_{bottom}$ bottom-level series ($m = m_{top} + m_{bottom}$). The physical hierarchy is expressed via the structural summing matrix $\mathbf{S} \in \{0, 1\}^{m \times m_{bottom}}$:
$$\mathbf{y}_t = \mathbf{S} \mathbf{b}_t$$
where $\mathbf{b}_t$ represents the bottom-level series. Base forecasts $\hat{\mathbf{y}}_t$ generated across nodes are typically incoherent ($\hat{\mathbf{y}}_t \neq \mathbf{S} \hat{\mathbf{b}}_t$).

The tracking engine must register the reconciliation operator $\mathbf{P} \in \mathbb{R}^{m_{bottom} \times m}$ that maps incoherent base forecasts into coherent reconciled forecasts $\tilde{\mathbf{y}}_t$:
$$\tilde{\mathbf{y}}_t = \mathbf{S} \mathbf{P} \hat{\mathbf{y}}_t$$

Depending on the registered pipeline configuration, $\mathbf{P}$ takes one of four canonical mathematical forms:
- **Bottom-Up (BU):** $\mathbf{P} = [\mathbf{0} \mid \mathbf{I}_{m_{bottom}}]$. Bottom-level forecasts are simply aggregated upward. Simple, but ignores high-level macro signal.
- **Top-Down (TD):** $\mathbf{P} = [\mathbf{p} \mid \mathbf{0}]$ where $\mathbf{p}$ is a vector of historical proportions. Loses localized node-level variance.
- **Ordinary Least Squares (OLS) En-Route:** $\mathbf{P} = (\mathbf{S}'\mathbf{S})^{-1}\mathbf{S}'$. Assumes equal and uncorrelated base forecast errors.
- **Minimum Trace (MinT) Optimal Reconciliation:**
  $$\mathbf{P}_{\text{MinT}} = (\mathbf{S}' \mathbf{W}^{-1} \mathbf{S})^{-1} \mathbf{S}' \mathbf{W}^{-1}$$
  where $\mathbf{W} = \mathbb{E}[\mathbf{e}_t \mathbf{e}_t']$ is the $m \times m$ error covariance matrix of base forecasts. 

Because the sample covariance matrix $\hat{\mathbf{W}}$ is often singular or ill-conditioned when $m$ is large, production pipelines log the shrinkage parameter $\lambda \in [0, 1]$ used to formulate the regularized shrinkage covariance matrix:
$$\mathbf{W}_{\text{shrink}} = \lambda \hat{\mathbf{W}}_D + (1 - \lambda) \hat{\mathbf{W}}$$
where $\hat{\mathbf{W}}_D = \text{diag}(\hat{\mathbf{W}})$. The shrinkage intensity $\lambda$, condition number of $\mathbf{W}$, and the serialized projection matrix $\mathbf{P}$ are logged directly to the parent echelon run.

### 5.2 Metadata, Parameter, and Schema Logging

To guarantee complete auditability, parameters must be logged with strict typing and schema enforcement. Experiment tracking schemas must capture three distinct parameter categories:

```text
+---------------------------------------------------------------------------------------------------+
|                                 LOGGED EXPERIMENT METADATA SCHEMA                                 |
+---------------------------------------------------------------------------------------------------+
| 1. INFRASTRUCTURE & DATA LINEAGE                                                                  |
|    - git_commit_sha:           "a4f89d9c2e0134f01489"                                             |
|    - iceberg_snapshot_id:      849204910294102938                                                 |
|    - feature_store_view_hash:  "sha256:7b91c...e3"                                               |
|    - python_env_lock_hash:     "sha256:1a82f...9c"                                               |
|    - execution_engine:         "Ray 2.35.0 on Kubernetes 1.30"                                   |
+---------------------------------------------------------------------------------------------------+
| 2. MODELING SPECIFICATION & HYPERPARAMETERS                                                       |
|    - algorithm_class:          "LightGBM_Regressor"                                               |
|    - objective_function:       "asymmetric_pinball_loss"                                          |
|    - cost_shortage_cu:         15.00  (USD/unit)                                                  |
|    - cost_holding_co:          5.00   (USD/unit)                                                  |
|    - lag_feature_windows:      [1, 2, 4, 8, 12, 52] (Weeks)                                        |
|    - rolling_stat_windows:     [4, 12, 26] (Weeks)                                                |
|    - learning_rate:            0.03                                                               |
|    - num_leaves:               63                                                                 |
|    - min_child_samples:        40                                                                 |
|    - reconciliation_method:    "MinT_Shrinkage"                                                   |
|    - mint_shrinkage_lambda:    0.1842                                                             |
+---------------------------------------------------------------------------------------------------+
| 3. TEMPORAL PROTOCOL SPECIFICATION                                                                |
|    - backtest_type:            "rolling_origin"                                                   |
|    - num_folds:                12                                                                 |
|    - forecast_horizon_h:       2 (Weeks)                                                          |
|    - purge_window_weeks:       2 (Weeks)                                                          |
|    - embargo_window_weeks:     2 (Weeks)                                                          |
|    - step_size_weeks:          1 (Week)                                                           |
+---------------------------------------------------------------------------------------------------+
```

Logging these parameters prevents the "silent drift" antipattern, where two models trained on seemingly identical configurations yield divergent predictions because of an untracked change in the underlying feature pipeline or an unpinned package dependency.

### 5.3 Artifact Lineage and Diagnostic Tracking

An experiment run is incomplete if it logs only scalar metrics. If a forecasting model begins to fail during canary evaluation, engineers require rich diagnostic artifacts to reconstruct why the model made specific predictions.

The artifact tracking pipeline must record:

1. **Serialized Model Binaries:**
   - Universal representation formats (ONNX) or framework-native binaries (e.g., LightGBM binary booster, PyTorch Safetensors).
   - Cryptographic SHA-256 digest logged as a metadata tag to prevent artifact substitution.
2. **Quantile Calibration and Error Confusion Curves:**
   - Quantile forecasting models (predicting P10, P50, P90 for safety stock sizing) must log empirical coverage plots comparing nominal quantile rates to observed empirical hit rates.
   - For classification models (e.g., predicting stockout events or supplier shipment delays), precision-recall curves, receiver operating characteristic (ROC) curves, and Brier score calibration plots are registered as versioned artifacts.
3. **Interpretability Matrices (TreeSHAP):**
   - TreeSHAP summary beeswarm plots visualizing feature impact distributions across all validation instances. In tree ensembles, the local Shapley attribution $\phi_j(x)$ satisfies:
     $$\hat{f}(x) = \phi_0 + \sum_{j=1}^M \phi_j(x)$$
     where $\phi_0 = \mathbb{E}[\hat{f}(X)]$ is the expected baseline model output across an immutable reference background dataset.
   - Mean absolute SHAP values exported as structured Parquet tables:
     $$I_j = \frac{1}{N} \sum_{i=1}^N |\phi_j(x_i)|$$
     enabling programmatic drift detection in feature attribution between successive model iterations.
4. **Residual Autocorrelation Diagnostics:**
   - Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots of out-of-sample forecast errors across lags $k = 1 \dots 26$.
   - Ljung-Box test statistics logged across each lag. The portmanteau statistic:
     $$Q = T(T+2) \sum_{k=1}^m \frac{\hat{\rho}_k^2}{T-k}$$
     is evaluated against $\chi^2_m$. If residuals exhibit statistically significant serial correlation beyond lag $h-1$, the tracker flags the model for autocorrelation leakage or unmodeled seasonality.

### 5.4 Temporal Backtest Protocol: Rolling vs. Expanding Windows

Supply chain models must never be evaluated using standard randomized $K$-fold cross-validation. Time series models require temporal evaluation protocols that mirror real-world operational forecasting cadences.

```text
========================================================================================================================
                          ROLLING VS. EXPANDING TEMPORAL EVALUATION WINDOWS
========================================================================================================================

 EXPANDING (ANCHORED) WINDOW PROTOCOL:
 -------------------------------------
 Fold 1:  |◄─── Train [t_0 ... t_100] ───►| Purge |◄─ Test [t_102 ... t_104] ─►| Embargo |
 Fold 2:  |◄────── Train [t_0 ... t_110] ──────►| Purge |◄─ Test [t_112 ... t_114] ─►| Embargo |
 Fold 3:  |◄───────── Train [t_0 ... t_120] ─────────►| Purge |◄─ Test [t_122 ... t_124] ─►| Embargo |
 * Best for: Structural stability assessment; non-stationary processes with long-term memory.
 
 ROLLING (SLIDING) WINDOW PROTOCOL:
 ----------------------------------
 Fold 1:  |◄─── Train [t_0 ... t_100] ───►| Purge |◄─ Test [t_102 ... t_104] ─►| Embargo |
 Fold 2:         |◄─── Train [t_10 ... t_110] ───►| Purge |◄─ Test [t_112 ... t_114] ─►| Embargo |
 Fold 3:                |◄─── Train [t_20 ... t_120] ───►| Purge |◄─ Test [t_122 ... t_124] ─►| Embargo |
 * Best for: High-velocity consumer retail; systems subject to frequent structural shifts and concept drift.
========================================================================================================================
```

The temporal harness enforces two mandatory safety buffers:
- **The Purge Window:** When generating features that utilize rolling statistics (e.g., 8-week rolling mean demand) or autoregressive lags, observations immediately preceding the forecast origin must be pruned if their feature computation incorporates target values that cross into the evaluation space. In multi-step horizons, the purge window length must equal at least the maximum autoregressive lag used in feature engineering.
- **The Embargo Window:** If models are evaluated sequentially across rolling origins, an out-of-sample test window at step $k$ will generate forecast errors that correlate with future demand because of replenishment cycle lag and inventory pipeline dynamics. The embargo window places a temporal buffer following the test window before any subsequent training fold can sample data:
  $$\text{Embargo Duration} \ge \text{Supplier Lead Time } (L) + \text{Review Period } (R)$$

Formally, for backtest fold $k$ with forecast origin $t_k$, fixed training length $N_{train}$, feature lag depth $M_{lag}$, forecast horizon $h$, and lead-time embargo $E$:
$$\mathcal{T}_{\text{train}}^{(k)} = \{ t \mid t_k - M_{lag} - N_{train} \le t < t_k - M_{lag} \}$$
$$\mathcal{T}_{\text{purge}}^{(k)} = \{ t \mid t_k - M_{lag} \le t < t_k \}$$
$$\mathcal{T}_{\text{test}}^{(k)} = \{ t \mid t_k \le t < t_k + h \}$$
$$\mathcal{T}_{\text{embargo}}^{(k)} = \{ t \mid t_k + h \le t < t_k + h + E \}$$

### 5.5 Rigorous Statistical Testing for Model Comparison

When comparing a production Champion model ($M_A$) against a candidate Challenger model ($M_B$), tracking systems must evaluate whether the observed performance delta is statistically significant.

#### 5.5.1 The Economic Loss Differential
Let $y_t$ be actual realized demand at evaluation period $t$, and let $\hat{y}_{A,t}$ and $\hat{y}_{B,t}$ be the $h$-step-ahead forecasts generated by Model A and Model B, respectively. 

Define an economic loss function $L(y_t, \hat{y}_t)$ that captures the financial reality of the supply chain. For an asymmetric piecewise linear inventory loss function:
$$L(y_t, \hat{y}_t) = c_u \max(0, y_t - \hat{y}_t) + c_o \max(0, \hat{y}_t - y_t)$$
where $c_u$ is the unit shortage cost (lost contribution margin plus stockout penalty) and $c_o$ is the unit holding cost (carrying cost of capital plus obsolescence).

The loss differential sequence $d_t$ for $t = 1, \dots, T$ is defined as:
$$d_t = L(y_t, \hat{y}_{A,t}) - L(y_t, \hat{y}_{B,t})$$
A positive mean differential $\bar{d} > 0$ indicates that Model A incurs higher economic losses than Model B (i.e., Challenger Model B is superior).

#### 5.5.2 The Diebold-Mariano (DM) Test
The Diebold-Mariano test (Diebold & Mariano, 1995) evaluates the null hypothesis of equal expected predictive accuracy:
$$H_0: \mathbb{E}[d_t] = 0 \quad \text{versus} \quad H_1: \mathbb{E}[d_t] \neq 0$$
The sample mean loss differential is:
$$\bar{d} = \frac{1}{T} \sum_{t=1}^T d_t$$

Because $h$-step-ahead forecasts share overlapping unobserved shocks, the forecast errors—and consequently the loss differential series $d_t$—exhibit an autocorrelation structure of at least moving-average order $MA(h-1)$. Therefore, the sample variance of $\bar{d}$ cannot be calculated using standard independent variance formulas.

The asymptotic long-run variance of $\bar{d}$ is given by:
$$V(\bar{d}) = \frac{1}{T} f_d(0) = \frac{1}{T} \left( \gamma_0 + 2 \sum_{k=1}^{\infty} \gamma_k \right)$$
where $\gamma_k = \text{Cov}(d_t, d_{t-k})$ is the autocovariance of $d_t$ at lag $k$. Under the theoretical assumption that the error process is $MA(h-1)$, autocovariances beyond lag $h-1$ are zero. Using a rectangular truncation kernel:
$$\hat{V}_{DM}(\bar{d}) = \frac{1}{T} \left( \hat{\gamma}_0 + 2 \sum_{k=1}^{h-1} \hat{\gamma}_k \right)$$
where the sample autocovariances are:
$$\hat{\gamma}_k = \frac{1}{T} \sum_{t=k+1}^T (d_t - \bar{d})(d_{t-k} - \bar{d})$$
The asymptotic Diebold-Mariano test statistic is:
$$DM = \frac{\bar{d}}{\sqrt{\hat{V}_{DM}(\bar{d})}} \xrightarrow{d} \mathcal{N}(0, 1)$$

#### 5.5.3 The Harvey-Leybourne-Newbold (HLN) Correction
In real-world supply chain backtests, evaluation sample sizes are frequently small ($T = 12$ monthly cutoffs or $T = 26$ bi-weekly cycles). Diebold, Mariano, and subsequent econometricians demonstrated that the asymptotic $DM$ test suffers from severe over-sizing (Type I error inflation) when $T$ is small and $h > 1$.

Harvey, Leybourne, and Newbold (1997) derived a small-sample correction factor that modifies the $DM$ statistic:
$$HLN = DM \times \sqrt{\frac{T + 1 - 2h + h(h-1)/T}{T}}$$
Rather than comparing against the standard normal distribution $\mathcal{N}(0, 1)$, the $HLN$ statistic is evaluated against Student's $t$-distribution with $T - 1$ degrees of freedom:
$$HLN \sim t_{T-1}$$
This adjustment inflates the estimated standard error to account for parameter estimation uncertainty and finite-sample bias, preventing the premature promotion of over-fitted challenger models.

#### 5.5.4 Spectral Density Kernels: Rectangular vs. Bartlett Newey-West
In finite samples, the rectangular kernel estimator:
$$\hat{f}_{\text{rect}}(0) = \hat{\gamma}_0 + 2 \sum_{k=1}^{h-1} \hat{\gamma}_k$$
is not mathematically guaranteed to be positive semi-definite. If sample autocovariances at lag 1 are strongly negative (e.g., $\hat{\gamma}_1 < -\frac{1}{2}\hat{\gamma}_0$), the estimated variance $\hat{V}(\bar{d})$ becomes negative, causing standard error computation to fail.

To guarantee positive semi-definiteness, experiment tracking engines implement the Bartlett weighting kernel (Newey & West, 1987):
$$\hat{f}_{NW}(0) = \hat{\gamma}_0 + 2 \sum_{k=1}^{K} \left( 1 - \frac{k}{K+1} \right) \hat{\gamma}_k$$
where $K$ is the truncation bandwidth (typically set to $h-1$ or $\lfloor 4(T/100)^{2/9} \rfloor$). The Bartlett kernel scales each autocovariance lag by a linearly decaying weight, ensuring that $\hat{V}(\bar{d}) > 0$ across all empirical loss trajectories.

#### 5.5.5 The Giacomini-White (GW) Conditional Predictive Ability Test
While the Diebold-Mariano test evaluates unconditional predictive ability across the entire backtest, supply chain managers require answers to conditional questions: *Does Model B outperform Model A during high-volatility promotional weeks, or when supplier lead times are prolonged?*

Giacomini and White (2006) formulated a framework for testing **Conditional Predictive Ability**. Rather than testing $\mathbb{E}[d_t] = 0$, the GW test evaluates:
$$H_0: \mathbb{E}[d_{t+h} \mid \mathcal{F}_t] = 0$$
where $\mathcal{F}_t$ represents an information set (instruments) observable at the forecast origin $t$. 

Let $\mathbf{h}_t$ be a $q \times 1$ vector of instruments (for example, $\mathbf{h}_t = [1, d_t]'$ to include an intercept and the prior period loss differential, or $\mathbf{h}_t = [1, \text{Promo}_t, \text{Vol}_t]'$ to condition on promotional state and market volatility).

Define the generalized interaction vector:
$$\mathbf{Z}_{t+h} = \mathbf{h}_t \otimes d_{t+h}$$
The sample average is:
$$\bar{\mathbf{Z}} = \frac{1}{T} \sum_{t=1}^T \mathbf{Z}_{t+h}$$
The Giacomini-White test statistic is a Wald-type quadratic form:
$$GW = T \bar{\mathbf{Z}}' \hat{\mathbf{\Omega}}^{-1} \bar{\mathbf{Z}} \xrightarrow{d} \chi^2_q$$
where $\hat{\mathbf{\Omega}}$ is a $q \times q$ heteroskedasticity and autocorrelation consistent (HAC) covariance matrix estimated via the Newey-West estimator:
$$\hat{\mathbf{\Omega}} = \hat{\mathbf{\Gamma}}_0 + \sum_{k=1}^{h-1} \left( 1 - \frac{k}{h} \right) (\hat{\mathbf{\Gamma}}_k + \hat{\mathbf{\Gamma}}_k')$$
$$\hat{\mathbf{\Gamma}}_k = \frac{1}{T} \sum_{t=k+1}^T (\mathbf{Z}_{t+h} - \bar{\mathbf{Z}})(\mathbf{Z}_{t+h-k} - \bar{\mathbf{Z}})'$$

If $GW > \chi^2_{q, 1-\alpha}$, the null hypothesis of conditional equal predictive ability is rejected. The model lineage registry records both the unconditional HLN statistic and the conditional GW statistic, enabling automated promotion gates to stage challengers only when they demonstrate statistically verified superiority under current operational conditions.

### 5.6 Feature Store Integration and Point-in-Time Correctness Validation

A primary root cause of non-reproducible experiment tracking results in supply chain machine learning is the dynamic mutation of upstream feature data. When a feature store (e.g., Feast, Hopsworks, Databricks Feature Store) computes point-in-time features for historical backtest cuts, it performs an as-of temporal join:
$$\text{Entity}(t_k) \bowtie_{\text{as-of}} \text{FeatureView}(t \le t_k)$$
To guarantee that the experiment run can be audited years later under regulatory compliance standards, the tracking manifest must capture:
1. **Feature View Specification:** The logical definition of the feature view, including aggregation windows (e.g., 14-day rolling demand sum, 8-week supplier lead-time variance) and entity join keys (e.g., `[sku_id, location_id]`).
2. **Feature View Schema Hash:** A cryptographic SHA-256 digest computed across the sorted feature names, physical data types, and transformation code.
3. **Data Immutability Proof:** The immutable storage identifier of the underlying feature table (e.g., Delta Lake table commit version or Apache Iceberg snapshot ID).
4. **Point-in-Time Join Verification:** An automated assertion logged during training data synthesis verifying that no observation in the synthesized training dataframe possesses a feature timestamp $t_{\text{feature}} > t_k$.

### 5.7 Catalog-Scale Multiple Hypothesis Testing and Distributed Gate Execution

Enterprise supply chains maintain catalogs spanning $10^4$ to $10^6$ SKU-location combinations. When an automated retraining pipeline evaluates candidate models across the catalog, conducting statistical hypothesis tests independently at nominal significance $\alpha = 0.05$ creates an unacceptable multiple comparisons problem.

#### 5.7.1 The False Discovery Rate (FDR) Hazard
Under the global null hypothesis that no challenger model is superior across the entire catalog, testing $M = 50,000$ SKUs yields an expected count of false model promotions:
$$\mathbb{E}[\text{False Positives}] = M \times \alpha = 50,000 \times 0.05 = 2,500 \text{ spurious promotions}$$
Deploying 2,500 models whose apparent superiority is entirely attributable to sampling noise degrades distribution center operations and triggers inventory imbalances.

#### 5.7.2 The Benjamini-Hochberg Procedure
To bound the False Discovery Rate at target tolerance $q^* = 0.05$:
1. Collect the two-tailed $p$-values from the Harvey-Leybourne-Newbold tests across all $M$ SKU models: $\{p_1, p_2, \dots, p_M\}$.
2. Sort the $p$-values in ascending order:
   $$p_{(1)} \le p_{(2)} \le \dots \le p_{(M)}$$
3. Compute the Benjamini-Hochberg critical threshold for each rank $k \in \{1, \dots, M\}$:
   $$c_k = \frac{k}{M} q^*$$
4. Determine the maximum rank index $k^*$ satisfying:
   $$k^* = \max \left\{ k \;\middle|\; p_{(k)} \le \frac{k}{M} q^* \right\}$$
5. Promote the challenger model for all SKUs whose rank $i \le k^*$. For all SKUs with $i > k^*$, retain the production Champion model.

#### 5.7.3 Distributed Execution Architecture
Because calculating the DM-HLN and GW statistics across 50,000 time series requires processing millions of loss terms, enterprise MLOps platforms execute statistical gates via distributed compute clusters (Ray or Apache Spark). Leaf nodes emit compact summary records `(sku_id, location_id, d_bar, gamma_0, gamma_1, hln_stat, p_value)` to an intermediate Parquet sink. A centralized gate driver collects the $p$-values, applies the Benjamini-Hochberg filter in memory, and writes the promotion decisions back to the Parent Experiment Run.

---

## 6. Worked example

To make these mechanics concrete, we walk through an end-to-end evaluation comparing two candidate forecasting models for a high-velocity Class A consumer packaged goods item (SKU-90412) managed at a Central Distribution Center.

### 6.1 Operational Context and Parameters

- **SKU Identifier:** SKU-90412 (High-velocity beverage multi-pack)
- **Node:** Central Distribution Center CDC-01
- **Evaluation Cadence:** Weekly rolling origin backtest across $T = 12$ consecutive out-of-sample weeks.
- **Forecast Horizon ($h$):** $h = 2$ weeks (representing the standard 14-day production and transit lead time from plant to CDC).
- **Economic Loss Parameters:**
  - Unit Shortage Cost ($c_u$): $\$15.00$ per unit (lost wholesale contribution margin of $\$11.00$ plus contractual retailer line-fill non-compliance penalty of $\$4.00$).
  - Unit Holding Cost ($c_o$): $\$5.00$ per unit (cost of warehouse floor staging, tied-up working capital interest, and risk of carton packaging damage).
- **Candidate Models:**
  - **Model A (Champion):** LightGBM Gradient Boosted Decision Trees trained on static lag windows.
  - **Model B (Challenger):** Temporal Fusion Transformer (TFT) with MinT hierarchical reconciliation.

### 6.2 Raw Backtest Data and Error Profiles

Table 6.1 displays the realized weekly demand $y_t$, the 2-week-ahead out-of-sample predictions ($\hat{y}_{A,t}, \hat{y}_{B,t}$), the resulting forecast errors ($e_t = y_t - \hat{y}_t$), the asymmetric economic losses ($L_t$), and the loss differentials ($d_t = L_{A,t} - L_{B,t}$).

$$\text{Table 6.1: Out-of-Sample Evaluation Series for SKU-90412 ($T = 12, h = 2$)}$$

| Week ($t$) | Actual ($y_t$) | Model A ($\hat{y}_{A,t}$) | Model B ($\hat{y}_{B,t}$) | Error A ($e_{A,t}$) | Error B ($e_{B,t}$) | Loss A ($L_{A,t}$) | Loss B ($L_{B,t}$) | Diff ($d_t$) |
|---|---|---|---|---|---|---|---|---|
| **1** | 210 | 204 | 212 | $+6$ | $-2$ | $\$90.00$ | $\$10.00$ | $+\$80.00$ |
| **2** | 245 | 235 | 241 | $+10$ | $+4$ | $\$150.00$ | $\$60.00$ | $+\$90.00$ |
| **3** | 190 | 198 | 196 | $-8$ | $-6$ | $\$40.00$ | $\$30.00$ | $+\$10.00$ |
| **4** | 280 | 288 | 292 | $-8$ | $-12$ | $\$40.00$ | $\$60.00$ | $-\$20.00$ |
| **5** | 310 | 296 | 298 | $+14$ | $+12$ | $\$210.00$ | $\$180.00$ | $+\$30.00$ |
| **6** | 260 | 265 | 269 | $-5$ | $-9$ | $\$25.00$ | $\$45.00$ | $-\$20.00$ |
| **7** | 290 | 302 | 308 | $-12$ | $-18$ | $\$60.00$ | $\$90.00$ | $-\$30.00$ |
| **8** | 325 | 319 | 327 | $+6$ | $-2$ | $\$90.00$ | $\$10.00$ | $+\$80.00$ |
| **9** | 230 | 242 | 240 | $-12$ | $-10$ | $\$60.00$ | $\$50.00$ | $+\$10.00$ |
| **10** | 275 | 253 | 261 | $+22$ | $+14$ | $\$330.00$ | $\$210.00$ | $+\$120.00$ |
| **11** | 305 | 317 | 325 | $-12$ | $-20$ | $\$60.00$ | $\$100.00$ | $-\$40.00$ |
| **12** | 340 | 326 | 328 | $+14$ | $+12$ | $\$210.00$ | $\$180.00$ | $+\$30.00$ |
| **Sum** | **3,260** | **3,251** | **3,297** | **$+9$** | **$-37$** | **$\$1,365.00$** | **$\$1,025.00$** | **$+\$340.00$** |

### 6.3 Step-by-Step Statistical Arithmetic

#### Step 1: Compute Mean Loss Differential
The total economic loss over the 12-week backtest for Model A is $\$1,365.00$, compared to $\$1,025.00$ for Model B. The sum of the loss differentials is:
$$\sum_{t=1}^{12} d_t = 80 + 90 + 10 - 20 + 30 - 20 - 30 + 80 + 10 + 120 - 40 + 30 = 340.00$$
The sample mean loss differential is:
$$\bar{d} = \frac{340.00}{12} = 28.3333 \text{ USD/week}$$
On average, Challenger Model B saves $\$28.33$ per week in combined stockout and holding costs on this SKU.

#### Step 2: Compute Sample Deviations and Autocovariances
Subtract $\bar{d} = 28.3333$ from each observation to obtain the mean-centered deviations:
$$(d_t - \bar{d}) = [51.67, 61.67, -18.33, -48.33, 1.67, -48.33, -58.33, 51.67, -18.33, 91.67, -68.33, 1.67]$$

Compute sample variance $\hat{\gamma}_0$:
$$\sum_{t=1}^{12} (d_t - \bar{d})^2 = 30,966.67$$
$$\hat{\gamma}_0 = \frac{30,966.67}{12} = 2,580.5556$$

Compute lag-1 autocovariance $\hat{\gamma}_1$:
$$\sum_{t=2}^{12} (d_t - \bar{d})(d_{t-1} - \bar{d}) = -5,884.49$$
$$\hat{\gamma}_1 = \frac{-5,884.49}{12} = -534.9537$$

Because the forecast horizon is $h = 2$, under the null hypothesis of equal forecast accuracy the error process is an $MA(1)$. Autocovariances beyond lag $h - 1 = 1$ are theoretically zero.

#### Step 3: Compute Long-Run Variance Under Rectangular Kernel
The estimated spectral density at frequency zero using the rectangular truncation kernel is:
$$\hat{f}_{\text{rect}}(0) = \hat{\gamma}_0 + 2\hat{\gamma}_1 = 2,580.5556 + 2(-534.9537) = 2,580.5556 - 1,069.9074 = 1,510.6481$$

The variance of the sample mean is:
$$\hat{V}_{DM}(\bar{d}) = \frac{\hat{f}_{\text{rect}}(0)}{T} = \frac{1,510.6481}{12} = 125.8873$$
The standard error is:
$$\text{SE}_{DM} = \sqrt{125.8873} = 11.2200$$

#### Step 4: Compute Standard Diebold-Mariano Statistic
$$DM = \frac{\bar{d}}{\text{SE}_{DM}} = \frac{28.3333}{11.2200} = 2.5253$$
Under standard asymptotic normal assumptions, a $z$-score of $2.5253$ has a two-tailed $p$-value of $0.0116$, which would lead a team to prematurely declare statistical significance at $\alpha = 0.05$.

#### Step 5: Apply Harvey-Leybourne-Newbold Small-Sample Correction
For sample size $T = 12$ and forecast horizon $h = 2$, compute the HLN finite-sample correction multiplier:
$$\text{Multiplier}_{HLN} = \sqrt{\frac{T + 1 - 2h + \frac{h(h-1)}{T}}{T}} = \sqrt{\frac{12 + 1 - 2(2) + \frac{2(1)}{12}}{12}} = \sqrt{\frac{13 - 4 + 0.1667}{12}} = \sqrt{\frac{9.1667}{12}} = \sqrt{0.763889} = 0.87400$$

Apply the multiplier to the asymptotic $DM$ statistic:
$$HLN = DM \times \text{Multiplier}_{HLN} = 2.5253 \times 0.87400 = 2.2071$$

#### Step 6: Evaluate Against Student's $t$-Distribution
Degrees of freedom: $\nu = T - 1 = 12 - 1 = 11$.
The critical values for a two-tailed Student's $t$-distribution with $\nu = 11$ are:
- $t_{0.05, 11} = 2.2010$
- $t_{0.01, 11} = 3.1058$

Since $HLN = 2.2071 > 2.2010$, we compute the exact two-tailed $p$-value:
$$p = 2 \times (1 - F_t(2.2071; \nu = 11)) = 0.0495$$
The test rejects the null hypothesis of equal predictive accuracy at the $\alpha = 0.05$ significance level ($p = 0.0495 < 0.05$). The small-sample correction appropriately moved the $p$-value from $0.0116$ to $0.0495$, reflecting the genuine statistical uncertainty of a 12-week backtest.

#### Step 7: Verification with Bartlett Newey-West Spectral Estimator
To verify that the negative autocovariance did not distort the rectangular variance, we evaluate using the Bartlett Newey-West estimator with bandwidth $K = 1$:
$$\hat{f}_{NW}(0) = \hat{\gamma}_0 + 2\left(1 - \frac{1}{1 + 1}\right)\hat{\gamma}_1 = 2,580.5556 + 2(0.5)(-534.9537) = 2,580.5556 - 534.9537 = 2,045.6019$$
$$\hat{V}_{NW}(\bar{d}) = \frac{2,045.6019}{12} = 170.4668 \implies \text{SE}_{NW} = 13.0563$$
$$DM_{NW} = \frac{28.3333}{13.0563} = 2.1701$$
The Bartlett estimator yields a conservative $z$-statistic of $2.1701$, confirming the directional validity of the result.

#### Step 8: Giacomini-White Conditional Test Evaluation
To assess conditional predictive ability, we construct the instrument vector $\mathbf{h}_t = [1, d_{t-1}]'$ for $t = 2 \dots 12$ ($N = 11$ transitions), testing whether past performance predicts future outperformance.

The observed interaction vector sequence $\mathbf{Z}_{t} = \mathbf{h}_{t-1} \otimes d_t = [d_t, d_{t-1} d_t]'$ yields the sample mean vector:
$$\bar{\mathbf{Z}} = \frac{1}{11} \sum_{t=2}^{12} \begin{bmatrix} d_t \\ d_{t-1} d_t \end{bmatrix} = \begin{bmatrix} 23.6364 \\ 81.8182 \end{bmatrix}$$

Evaluating the Newey-West HAC covariance matrix $\hat{\mathbf{\Omega}}$ with Bartlett lag weighting produces:
$$\hat{\mathbf{\Omega}} = \begin{bmatrix} 1.6261 \times 10^3 & 5.7076 \times 10^4 \\ 5.7076 \times 10^4 & 8.2224 \times 10^6 \end{bmatrix}$$

Inverting $\hat{\mathbf{\Omega}}$ yields:
$$\hat{\mathbf{\Omega}}^{-1} = \begin{bmatrix} 8.1309 \times 10^{-4} & -5.6441 \times 10^{-6} \\ -5.6441 \times 10^{-6} & 1.6080 \times 10^{-7} \end{bmatrix}$$

The Wald quadratic form evaluates to:
$$GW = N \bar{\mathbf{Z}}' \hat{\mathbf{\Omega}}^{-1} \bar{\mathbf{Z}} = 11 \begin{bmatrix} 23.6364 & 81.8182 \end{bmatrix} \begin{bmatrix} 8.1309 \times 10^{-4} & -5.6441 \times 10^{-6} \\ -5.6441 \times 10^{-6} & 1.6080 \times 10^{-7} \end{bmatrix} \begin{bmatrix} 23.6364 \\ 81.8182 \end{bmatrix} = 4.7685$$

With degrees of freedom $q = 2$, the critical value for $\chi^2_2$ at $\alpha = 0.05$ is $5.9915$.
The resulting $p$-value is:
$$p_{GW} = 1 - F_{\chi^2}(4.7685; 2) = 0.0922$$
While Model B is unconditionally superior over the 12-week evaluation ($p = 0.0495$), we fail to reject the null hypothesis of conditional equal predictive ability at $\alpha = 0.05$ ($p = 0.0922$). This indicates that Challenger Model B's superiority fluctuates depending on whether the prior week experienced positive or negative supply chain shocks, providing crucial insight for deployment staging.

### 6.4 Portfolio-Level False Discovery Rate Control (Benjamini-Hochberg)

Now consider that CDC-01 evaluates candidate models across a priority mini-batch of $M = 5$ beverage SKUs simultaneously (including SKU-90412). Table 6.2 displays the candidate models, their raw Harvey-Leybourne-Newbold $p$-values, rank ordering, Benjamini-Hochberg thresholds for $q^* = 0.05$, and final deployment decisions.

$$\text{Table 6.2: Benjamini-Hochberg False Discovery Rate Evaluation across 5 Candidate SKU Models}$$

| Rank ($k$) | SKU Identifier | Challenger Model Architecture | Raw $p$-value ($p_{(k)}$) | BH Threshold $\frac{k}{M} q^*$ | Decision ($p_{(k)} \le c_k$) |
|---|---|---|---|---|---|
| **1** | SKU-44102 | Temporal Fusion Transformer (TFT) | $0.0082$ | $\frac{1}{5}(0.05) = 0.0100$ | **Promote Challenger** ($0.0082 \le 0.0100$) |
| **2** | SKU-10488 | DeepAR Hierarchical Intermittent | $0.0195$ | $\frac{2}{5}(0.05) = 0.0200$ | **Promote Challenger** ($0.0195 \le 0.0200$) |
| **3** | SKU-31099 | LightGBM with Tweedie Objective | $0.0310$ | $\frac{3}{5}(0.05) = 0.0300$ | **Retain Champion** ($0.0310 > 0.0300$) |
| **4** | SKU-90412 | Temporal Fusion Transformer (TFT) | $0.0495$ | $\frac{4}{5}(0.05) = 0.0400$ | **Retain Champion** ($0.0495 > 0.0400$) |
| **5** | SKU-88214 | CatBoost with Asymmetric Pinball | $0.1420$ | $\frac{5}{5}(0.05) = 0.0500$ | **Retain Champion** ($0.1420 > 0.0500$) |

#### Operational Takeaway of Multi-SKU Testing
Notice that when SKU-90412 was evaluated in isolation in Section 6.3, its $p$-value of $0.0495$ barely passed the uncorrected threshold $\alpha = 0.05$. However, when evaluated within the CDC portfolio under Benjamini-Hochberg control, its rank is $k = 4$, yielding a dynamic threshold of $c_4 = 0.0400$. Because $0.0495 > 0.0400$, the automated gate correctly rejects the promotion of SKU-90412, preventing false discovery churn. Only SKU-44102 and SKU-10488 clear the portfolio-level significance hurdle and advance to canary staging.

---

## 7. Code, query, or calculation

The following self-contained Python module implements the Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample correction and an asymmetric supply chain piecewise linear loss function. It executes the exact mathematical calculation derived in Section 6.

```python
"""Diebold-Mariano and Harvey-Leybourne-Newbold significance test for supply chain models."""
import numpy as np
from scipy import stats


def asymmetric_loss(y_true, y_pred, c_underage=15.0, c_overage=5.0):
    """Compute asymmetric piecewise linear supply chain shortage and holding cost."""
    err = np.asarray(y_true) - np.asarray(y_pred)
    return np.where(err >= 0, c_underage * err, -c_overage * err)


def diebold_mariano_hln(y_true, y_pred_a, y_pred_b, h=2, loss_fn=asymmetric_loss):
    """Compute DM and HLN test statistics for h-step-ahead forecast comparisons."""
    l_a = loss_fn(y_true, y_pred_a)
    l_b = loss_fn(y_true, y_pred_b)
    d = l_a - l_b
    t_samples = len(d)

    d_bar = float(np.mean(d))
    dev = d - d_bar
    gamma_0 = float(np.sum(dev**2) / t_samples)
    gammas = [
        float(np.sum(dev[k:] * dev[:-k]) / t_samples) for k in range(1, h)
    ]

    f_hat_rect = gamma_0 + 2.0 * sum(gammas)
    lr_var = max(f_hat_rect / t_samples, 1e-12)
    dm_stat = d_bar / np.sqrt(lr_var)

    # Harvey-Leybourne-Newbold (1997) small-sample correction multiplier
    hln_term = (t_samples + 1 - 2 * h + (h * (h - 1)) / t_samples) / t_samples
    hln_multiplier = np.sqrt(max(0.0, hln_term))
    hln_stat = dm_stat * hln_multiplier
    p_value = float(2.0 * (1.0 - stats.t.cdf(np.abs(hln_stat), df=t_samples - 1)))

    return {
        "mean_differential": d_bar,
        "gamma_0": gamma_0,
        "gamma_1": gammas[0] if gammas else 0.0,
        "dm_stat": dm_stat,
        "hln_stat": hln_stat,
        "p_value": p_value,
    }


if __name__ == "__main__":
    y_actual = [210, 245, 190, 280, 310, 260, 290, 325, 230, 275, 305, 340]
    y_champ_a = [204, 235, 198, 288, 296, 265, 302, 319, 242, 253, 317, 326]
    y_chall_b = [212, 241, 196, 292, 298, 269, 308, 327, 240, 261, 325, 328]

    results = diebold_mariano_hln(y_actual, y_champ_a, y_chall_b, h=2)
    for metric, val in results.items():
        print(f"{metric:20s}: {val:10.4f}")
```

### Assumptions and Execution Instructions
- **Input Preconditions:** Arrays must represent synchronous time-ordered vectors of identical length without missing values. The forecast horizon parameter $h$ must be an integer $\ge 1$.
- **Time Complexity:** The sample autocovariance computation requires $\mathcal{O}(h \cdot T)$ operations, and memory allocation is strictly bounded by $\mathcal{O}(T)$. For enterprise backtests where $T \le 104$ and $h \le 12$, execution completes in sub-millisecond wall-clock time.
- **Space Complexity:** The algorithm allocates only one difference array $d$ and one centered array of length $T$, requiring less than 4 kilobytes of RAM.
- **Dependencies:** Standard Python 3.10+ runtime with `numpy >= 1.24` and `scipy >= 1.10`.
- **Expected Console Output:**
  ```text
  mean_differential   :    28.3333
  gamma_0             :  2580.5556
  gamma_1             :  -534.9537
  dm_stat             :     2.5253
  hln_stat            :     2.2071
  p_value             :     0.0495
  ```

### Automated CI/CD Regression Gate Integration
In an enterprise MLOps pipeline, the `diebold_mariano_hln` function is integrated into automated test suites (e.g., `pytest`). When a GitHub Actions or GitLab CI runner executes pull request validation, the statistical gate evaluates the candidate challenger against the production baseline:

```python
# CI test assertion pattern for automated candidate staging
def test_candidate_statistical_gate():
    res = diebold_mariano_hln(y_actual, y_champion, y_challenger, h=2)
    # Require positive economic savings and statistical significance at alpha=0.05
    assert res["mean_differential"] > 0.0, "Challenger exhibits higher economic loss"
    assert res["p_value"] < 0.05, f"P-value {res['p_value']:.4f} exceeds alpha=0.05"
```

If the pull request satisfies both assertions, the CI engine tags the run as `PROMOTION_CANDIDATE` in the experiment registry and triggers automated canary packaging.

---

## 8. Data, standards, and regulatory context

Experiment tracking in enterprise supply chains intersects international standards for artificial intelligence, model risk governance in financial logistics, and trade data standards.

### 8.1 International AI Engineering Standards

1. **ISO/IEC 5259 Series (Data Quality for Analytics and Machine Learning):**
   - **Part 1 (Overview and Terminology):** Establishes the lifecycle governance framework separating developmental feature pipelines from production inference pipelines.
   - **Part 2 (Data Quality Measures):** Mandates the formal tracking of data pipeline quality metrics across experiment runs, including missing value ratios, syntactic validity, and measurement consistency.
   - **Part 3 (Data Quality Management Requirements):** Requires systems to record the exact cryptographic lineage connecting training datasets to generated model artifacts, prohibiting untracked manual preprocessing or dynamic feature substitution.
2. **ISO/IEC 42001:2023 (Artificial Intelligence Management System):**
   - **Clause 8.4 (Management of AI Models):** Requires organizations to establish structured verification records demonstrating that AI model performance is evaluated against predefined operational benchmarks before deployment.
   - **Clause 9.1 (Monitoring, Measurement, Analysis, and Evaluation):** Specifically mandates documented evidence of comparative evaluation (champion-challenger testing) and reproducible audit trails for all model selection decisions.
   - **Clause 9.2 (Internal Audit):** Mandates periodic third-party reviews of experiment tracking ledgers to verify that production deployments align with approved candidate runs.
3. **NIST AI Risk Management Framework (NIST AI RMF 1.0):**
   - **MAP 1.5:** Organizational risk tolerances regarding supply chain disruption and safety stock volatility are mapped to explicit model evaluation metrics.
   - **MEASURE 2.6:** "Model performance is documented across training, validation, and testing phases with metrics reflecting operational context."
   - **MANAGE 2.4:** "Model deployment criteria, staging protocols, and decommissioning triggers are formally defined, tracked, and approved."

### 8.2 Financial Model Governance and Regulatory Compliance

In high-value supply chains—particularly where automated forecasting drives automated commercial purchase orders or impacts corporate balance sheet inventory valuations—governance frameworks adapted from financial engineering apply:

- **Federal Reserve Board SR 11-7 / OCC Bulletin 2011-12 (Supervisory Guidance on Model Risk Management):**
  - Requires **Conceptual Soundness:** The modeling methodology must be mathematically justifiable and documented. The tracking system must store the theoretical rationale for hyperparameter bounds and feature lag definitions.
  - Mandates **Rigorous Benchmarking:** Production models cannot be promoted based on internal training fit. SR 11-7 explicitly demands ongoing comparison against challenger models and statistical verification of performance stability.
  - Requires **Model Inventory and Audit Trails:** Every model version running in production must have an unbroken audit trail linking the deployed binary to the exact experiment run, author identity, evaluation results, and approval sign-offs.
- **Sarbanes-Oxley Act of 2002 (SOX Section 404):**
  - Publicly traded retail and manufacturing corporations must certify internal controls over automated financial processes. When machine learning replenishment pipelines trigger automated accounts payable liabilities and dictate quarterly inventory carrying valuations, the experiment tracking platform serves as an auditable internal control ledger proving that model changes follow authorized testing gates.
- **European Union AI Act (Regulation 2024/1689):**
  - **Article 10 (Data and Data Governance):** High-risk systems (such as automated logistics for critical public infrastructure or medical supply chains) must enforce strict data governance, documentation of bias, and appropriate statistical validation protocols.
  - **Article 14 (Human Oversight):** Demands that model experiment tracking interfaces expose interpretability artifacts (e.g., TreeSHAP summaries) allowing human inventory managers to understand the drivers of anomalous aggregate forecasts.
  - **Article 15 (Accuracy, Robustness, and Cybersecurity):** Requires continuous logging of out-of-sample accuracy metrics across historical horizons to prove model resilience against distribution shifts.

### 8.3 Supply Chain Master Data Standards (GS1)

Experiment trackers must link model runs directly to global supply chain identifier standards:
- **Global Trade Item Number (GTIN / GS1):** Child runs representing SKU-level models must be indexed by immutable 14-digit GTINs rather than internal, volatile ERP part numbers that change during corporate acquisitions or catalog restructurings.
- **Global Location Number (GLN / GS1):** Echelon and warehouse parent runs must tag physical sites using standardized 13-digit GLNs, ensuring that models map unambiguously to physical distribution nodes across multi-enterprise logistics networks.
- **GS1 Global Data Synchronization Network (GDSN):** Feature stores and experiment trackers log GDSN product attribute versions (e.g., net weight, packaging dimensions, hazardous material classifications), preventing models from training on outdated physical package specifications.

---

## 9. Industry practice and case evidence

Leading supply chain and retail technology organizations maintain dedicated MLOps platforms built around hierarchical, leakage-free experiment tracking:

### 9.1 Documented Industrial Implementations

1. **Amazon Supply Chain Optimization Technologies (SCOT):**
   - *Architecture:* SCOT operates multi-echelon demand forecasting and automated replenishment across hundreds of millions of active Amazon Standard Identification Numbers (ASINs).
   - *Experiment Protocol:* Forecasting experiments run within automated backtesting harnesses that enforce rolling origin evaluations with strict purge and embargo buffers. Amazon’s tracking systems capture hierarchical lineage from the macro product family level down to individual fulfillment center ASIN models.
   - *Statistical Gates:* Candidate models are evaluated using asymmetric cost metrics reflecting customer promises and fulfillment network topology, requiring statistically verified improvements before entering shadow-mode testing in production.
2. **Walmart Global Tech:**
   - *Architecture:* Manages demand forecasting across more than 4,700 retail stores and regional fulfillment centers.
   - *Experiment Protocol:* Walmart migrated from isolated notebook-based model development to an enterprise MLOps platform leveraging MLflow and Kubeflow. The platform automatically tracks Delta Lake snapshot IDs, feature store schema definitions, and localized TreeSHAP feature attributions.
   - *Operational Evidence:* Documented engineering case studies demonstrate that replacing flat cross-validation with temporal backtesting eliminated widespread production over-forecasting errors that previously caused multi-million-dollar inventory pileups during seasonal holiday transitions.
3. **Target Corporation:**
   - *Architecture:* Evaluates multi-horizon omni-channel fulfillment models optimizing store-as-hub replenishment.
   - *Tracking Rigor:* Experiment manifests log explicit artifact lineages, including residual autocorrelation plots and Quantile Loss summaries across P10, P50, and P90 forecast intervals. Model governance committees inspect these logged artifacts before authorizing automated replenishment pipeline upgrades.
4. **Zara / Inditex Agile Replenishment:**
   - *Architecture:* Operates a hyper-compressed 2-week design-to-retail supply chain requiring store-level micro-forecasts.
   - *Tracking Rigor:* Inditex tracks rolling 14-day backtest runs across 2,000 global stores. Tracking manifests capture localized store display constraints and stockout censoring indicators, ensuring that candidate models are evaluated solely against unconstrained customer demand signals.
5. **Procter & Gamble (P&G) Demand Sensing:**
   - *Architecture:* Synchronizes manufacturing production lines with downstream retailer point-of-sale (POS) and inventory data across global consumer goods markets.
   - *Tracking Rigor:* Experiment tracking frameworks link customer EDI 852 (Product Activity Data) feed snapshot hashes directly to production replenishment parent DAGs, ensuring that multi-echelon reconciliation weights ($\mathbf{W}_{\text{shrink}}$) are logged and versioned.

### 9.2 Enterprise MLOps Tooling Comparison for Supply Chains

Enterprise supply chain engineering teams evaluate commercial and open-source experiment tracking platforms against the demanding architectural constraints of multi-echelon forecasting and replenishment pipelines. Table 9.1 provides a comparative analysis of primary platforms across core supply chain requirements.

$$\text{Table 9.1: Enterprise Experiment Tracking Platform Capabilities in Supply Chain Analytics}$$

| Evaluation Dimension | MLflow (Open Source / Databricks) | Weights & Biases (Enterprise) | Neptune.ai | Cloud-Native Custom (Ray + Iceberg + S3) |
|---|---|---|---|---|
| **Hierarchical DAG Run Modeling** | Supported via nested runs (`parent_run_id`). Requires custom metadata tagging for deep echelons. | Supported via Run Groups and Nested Tables. Powerful UI grouping for cluster rollups. | Fully hierarchical run trees via namespace dictionary keys (`sys/parent_id`). | Native DAG lineage orchestrated via Ray Core object references and Airflow task IDs. |
| **High-Volume Tabular Artifacts** | Parquet / Delta logging supported. Large table rendering requires Databricks Unity Catalog. | Native support for W&B Tables. UI indexing degrades if child tables exceed $10^6$ rows. | Excellent artifact storage with Amazon S3/GCS backend integration; low UI overhead. | Highly optimized. Leaf nodes write partitioned Parquet directly to object stores; metadata indexed via DuckDB. |
| **Data Snapshot Lineage Integration** | Tight native integration with Delta Lake commit versions and Unity Catalog lineage graphs. | Pointers tracked via Artifacts lineage manifests; integrates with Snowflake and BigQuery. | Integrates via metadata attribute strings tracking Git commit and Iceberg snapshot hashes. | Native Iceberg snapshot metadata catalog integrated with Apache Polaris or AWS Glue. |
| **Custom Significance Gates (DM/HLN)** | Executed via custom Python scripts in MLflow model evaluation plugins or MLflow Recipes. | Configurable via W&B Launch webhooks triggering containerized evaluation jobs. | Executed via custom client-side evaluation pipelines reading run metadata streams. | Fully integrated into pipeline worker nodes; gates act as native Airflow / Kubeflow condition tasks. |
| **Catalog-Scale Throughput (100k+ SKUs)** | Requires dedicated backend database (PostgreSQL RDS) with connection pooling and Redis cache. | Enterprise cloud backend handles high concurrency; client batching required to avoid HTTP 429. | Highly scalable asynchronous ingestion client designed for high-concurrency distributed workers. | Virtually unlimited horizontal scalability; state maintained in serverless distributed object storage. |

### 9.3 Separation of Documented Fact from Engineering Inference

- **Documented Fact:** Global retailers log billions of parameter-metric pairs weekly across distributed experiment tracking platforms (MLflow, Neptune, in-house engines), using rolling backtest protocols to evaluate replenishment models. Documented case studies from Walmart Global Tech and Amazon SCOT confirm that backtesting platforms enforce temporal purge and embargo splits to eliminate cross-fold information contamination.
- **Engineering Inference:** While major tech companies publish architectural overviews of their feature stores and training platforms, the exact production threshold values for Diebold-Mariano significance gates (e.g., specific choice of $\alpha = 0.05$ vs $\alpha = 0.01$, or internal proprietary small-sample weighting constants) are rarely disclosed and are routinely calibrated based on specific enterprise working capital tolerance and safety stock risk posture.

---

## 10. Failure modes and red flags

The following failure modes represent common structural breakdowns in supply chain experiment tracking systems.

$$\text{Table 10.1: Experiment Tracking Failure Modes and Engineering Mitigations}$$

| # | Failure Mode | Early Warning Sign | Root Mechanism | Operational Consequence | Engineering Mitigation |
|---|---|---|---|---|---|
| **1** | **Autocorrelation Leakage** | Validation error is unnaturally low ($R^2 > 0.98$), but live production error spikes immediately. | Multi-step forecast errors share serial correlation across overlapping horizons without an embargo window. | Over-optimistic backtests lead to deployment of unstable models that degrade inventory control. | Enforce mandatory embargo buffers equal to supplier lead time plus review cycle ($\ge L + R$). |
| **2** | **Feature Lookahead Leakage** | Lag features or rolling statistics exhibit high importance despite zero predictive logic in domain reality. | Rolling window aggregates computed across splits without purging data preceding the forecast origin. | Models exploit future target values during training; catastrophic failure occurs in production inference. | Prune training records whose calculation windows intersect the forecast cutoff via automated purge rules. |
| **3** | **Symmetric Metric Bias** | Challenger model achieves lower RMSE but distribution centers report surging stockout emergency orders. | Objective and tracking metrics use symmetric squared error ($L_2$) despite $10\times$ higher cost of underage. | Algorithm optimizes variance by slightly under-forecasting peak demand, causing severe stockout crises. | Replace tracking loss with asymmetric piecewise linear pinball loss parametrized by actual $c_u$ and $c_o$. |
| **4** | **Type I Error Inflation** | Challenger models are promoted every week, but aggregate ERP inventory metrics continuously deteriorate. | Applying asymptotic Diebold-Mariano tests ($z$-scores) on small backtest sample sizes ($T < 30$). | False positive rate explodes; teams spend engineering effort deploying models that offer no real improvement. | Implement the Harvey-Leybourne-Newbold small-sample correction evaluated against Student's $t_{T-1}$. |
| **5** | **Negative Spectral Variance** | Automated statistical significance pipeline crashes with `NaN` or complex square root errors. | Truncated rectangular kernel yields negative variance when sample autocovariance $\hat{\gamma}_1 < -\frac{1}{2}\hat{\gamma}_0$. | Testing gate fails; pipeline either halts or defaults to promoting models without statistical verification. | Implement the Bartlett Newey-West weighting kernel to guarantee positive semi-definiteness of the covariance. |
| **6** | **Orphaned Child Runs** | Tracking database contains millions of decoupled SKU runs with no linkage to the initiating pipeline run. | Worker tasks log child runs without referencing parent run IDs in distributed cluster environments. | Impossibility of reproducing end-to-end network state; audit failure under ISO/IEC 42001 standards. | Enforce strict DAG context managers that pass `parent_run_id` across distributed Ray/Spark workers. |
| **7** | **Silent Feature Drift** | Challenger model outperforms Champion in backtest, but crashes on schema validation in staging. | Pipeline relies on dynamic database views rather than immutable, versioned feature store snapshots. | Backtest results reflect non-repeatable historical data state; deployment breaks operational data pipelines. | Log immutable Feature View schema hashes and Delta/Iceberg snapshot IDs to the root experiment run. |
| **8** | **Catalog Masking** | Global aggregate WAPE drops 1.5%, but Class A revenue-generating SKUs suffer massive accuracy degradation. | Evaluating experiments purely on unweighted or unsegmented portfolio-level scalar averages. | Low-volume, long-tail SKUs overwhelm aggregate averages while high-value critical items fail. | Enforce stratified metric tracking segmented by ABC volume, XYZ volatility, and echelon tiers. |
| **9** | **SHAP Background Drift** | TreeSHAP feature attributions change radically between identical runs on adjacent weeks. | Tracking pipeline uses a dynamically sampled background reference dataset of varying size for SHAP estimation. | False alarms in explainability monitoring; engineers waste time diagnosing non-existent feature drift. | Pin and version an immutable reference background dataset serialized alongside the model artifact. |
| **10** | **Multiple Testing Explosion** | In a catalog of 50,000 SKUs, 2,500 models show "statistically significant" improvements at $\alpha = 0.05$. | Evaluating thousands of independent hypothesis tests without Family-Wise Error Rate (FWER) control. | Widespread false discoveries; hundreds of sub-optimal models deployed due to random sampling variance. | Apply Benjamini-Hochberg False Discovery Rate (FDR) corrections across large SKU test collections. |
| **11** | **Backtest Overfitting (P-Hacking)** | Data scientists test 300 hyperparameter combinations until one achieves $p < 0.05$ on a 12-week backtest. | Treating the backtest evaluation set as a training tuning set without an out-of-sample temporal holdout. | Selected model is over-fitted to the specific 12-week noise trajectory; underperforms in live production. | Lock the historical backtest window; mandate final verification on an untouchable future temporal test fold. |
| **12** | **Reconciliation Matrix Singularity** | Echelon parent run crashes during forecast reconciliation matrix inversion with singular matrix error. | MinT covariance matrix $\hat{\mathbf{W}}$ estimated on collinear nodes or sample size smaller than hierarchy nodes ($T < m$). | Unreconciled base forecasts dispatched directly to ERP, causing echelon inventory imbalances. | Enforce Ledoit-Wolf or Schäfer-Strimmer shrinkage covariance estimation ($\mathbf{W}_{\text{shrink}}$) in the parent run. |
| **13** | **Censored Demand Feedback Loop** | Forecasting accuracy metrics appear stable, but SKU sales volume steadily declines quarter-over-quarter. | Models trained on recorded point-of-sale sales during stockout weeks without unconstrained demand imputation. | Model learns that demand was zero during stockouts, lowering future replenishment and creating a stockout death spiral. | Log explicit stockout event flags and track Tobit-imputed demand actuals in the experiment evaluation store. |
| **14** | **Unversioned Docker Container Shift** | Retraining pipeline reproduces exact parameters and data snapshot but produces divergent model binaries. | Underlying container image uses floating `latest` tags or unpinned system CUDA/compiler drivers. | Loss of scientific reproducibility; violates Federal Reserve SR 11-7 model audit standards. | Enforce cryptographic Docker image SHA-256 digest logging in the experiment root manifest. |

---

## 11. Assumptions and limits

To maintain scientific integrity, the mathematical methods and tracking protocols detailed in this topic require explicit recognition of their underlying assumptions and operational limits.

$$\text{Table 11.1: Explicit Ledger of Methodological Assumptions and Boundary Conditions}$$

| # | Core Assumption | Theoretical Rationale | Real-World Violation Condition | Diagnostic & Validation Test |
|---|---|---|---|---|
| **1** | **Covariance Stationarity of Loss Differentials** | Diebold-Mariano and HLN tests assume the loss differential series $d_t$ is covariance stationary with time-invariant autocovariance. | Structural breaks in supply chains (e.g., pandemic disruptions, factory fires, sudden regulatory trade tariff shifts). | Augmented Dickey-Fuller (ADF) test and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test on $d_t$. |
| **2** | **Accurate Forecast Horizon Specification ($h$)** | Truncating autocovariance at lag $h-1$ assumes forecast errors follow an $MA(h-1)$ process under the null hypothesis. | Dynamic lead times where supplier delivery times vary randomly between 1 and 4 weeks, altering the true error dependency structure. | Ljung-Box autocorrelation test on residuals beyond lag $h-1$; dynamic lead-time tracking. |
| **3** | **Stability of Cost Parameters ($c_u, c_o$)** | The asymmetric loss function assumes unit shortage and holding costs remain constant across the evaluation period. | Rapid inflation, fluctuating short-term cost of capital, or spot freight price surges that alter unit economics dynamically. | Sensitivity analysis of $d_t$ across a $\pm 30\%$ grid of $c_u/c_o$ cost parameter ratios. |
| **4** | **Uncensored Demand Observability** | The evaluation assumes actual customer demand $y_t$ is fully observed, even when warehouse stockouts occur. | Unrecorded lost sales where customer demand exceeds inventory and the ERP records only truncated sales volume. | Tobit demand imputation or logging explicit lost sales estimations alongside ERP transaction records. |
| **5** | **Exogeneity of Inventory Policy** | The evaluation assumes forecast generation does not alter the underlying customer demand generation process. | Feedback loops where stockouts caused by poor forecasts permanently alter customer shopping behavior or supplier priority. | Econometric tests for endogeneity; monitoring customer churn rates in chronically under-forecast nodes. |
| **6** | **Adequacy of Backtest Sample Size ($T$)** | Even with the HLN correction, valid inference requires sufficient degrees of freedom to estimate the autocovariance structure. | Extremely short backtests ($T < 8$ weeks) where sample autocovariance estimates are highly unstable or volatile. | Rule of thumb: require $T \ge 12$ periods for $h \le 2$; $T \ge 26$ periods for multi-horizon planning models. |
| **7** | **Fixed Parameter Estimation Uncertainty** | The standard DM test assumes model parameters are fixed and known, ignoring estimation uncertainty from rolling refitting. | Rolling window backtests where parameters are re-estimated at every step on small training windows ($R/T$ ratio is small). | Use the Giacomini-White framework, which explicitly accounts for parameter estimation uncertainty under rolling windows. |
| **8** | **Catalog Topology Invariance** | Parent-child tracking assumes the structural aggregation matrix $\mathbf{S}$ remains invariant across the evaluation window. | Active SKU rationalization, brand divestitures, warehouse re-alignments, or store openings mid-backtest. | Versioning the hierarchy structural matrix $\mathbf{S}$ alongside the data snapshot ID in the root experiment run. |
| **9** | **Linearity of Structural Summing ($\mathbf{S}$)** | Hierarchical reconciliation algorithms assume aggregations are linear summations across product categories and geographies. | Non-linear business aggregations, such as customer basket discounts, bundle products, or non-additive tax rules. | Verify structural hierarchy matrix $\mathbf{S}$ maps bottom series strictly via additive linear operators. |
| **10** | **Independence of Lead Times from Forecasts** | The evaluation assumes supplier replenishment lead times are exogenous to the quantities forecasted and ordered. | Capacity-constrained suppliers whose delivery lead times lengthen when large replenishment orders are placed. | Cross-correlation analysis between forecast error series and supplier delivery delay logs. |

---

## 12. Exercises

The following exercises test your theoretical understanding, mathematical calculation capabilities, and engineering architecture skills.

### Exercise 1: Computing HLN Correction Factor Across Sample Sizes (Calculation)
- **Context:** An analytics engineer evaluates a 4-week replenishment model ($h = 4$). The team debates whether a 12-week backtest ($T = 12$) provides sufficient statistical rigor or whether they must extend the backtest to 26 weeks ($T = 26$) or 52 weeks ($T = 52$).
- **Deliverable:** Calculate the exact Harvey-Leybourne-Newbold correction multiplier $\sqrt{\frac{T + 1 - 2h + h(h-1)/T}{T}}$ for $T = 12$, $T = 26$, and $T = 52$ when $h = 4$. State the degrees of freedom and corresponding two-tailed critical $t$-values at $\alpha = 0.05$ for each scenario.
- **Acceptance Criteria:** Exact arithmetic to 4 decimal places; correct critical values from Student's $t$-distribution.
- **Estimated Time:** 20 minutes.

### Exercise 2: Formulating Asymmetric Loss from Commercial Ledgers (Application)
- **Context:** A wholesale pharmaceutical distributor stocks temperature-controlled vaccines. The wholesale selling price is $\$80.00/\text{dose}$, and the procurement cost is $\$50.00/\text{dose}$. Unmet demand results in an expedited emergency distributor fee of $\$15.00/\text{dose}$ to maintain hospital contracts. Unsold doses held at the end of the 4-week cycle incur a capital holding cost of $\$2.00/\text{dose}$ and a spoilage risk discount of $\$8.00/\text{dose}$.
- **Deliverable:** Formulate the mathematical loss function $L(y_t, \hat{y}_t)$, identifying explicit values for $c_u$ and $c_o$. Compute the loss for:
  - Scenario A: Actual $y = 500$, Forecast $\hat{y} = 420$.
  - Scenario B: Actual $y = 500$, Forecast $\hat{y} = 580$.
- **Acceptance Criteria:** State both cost parameters with dollar units; compute total loss for both scenarios. Explain why symmetric MSE would produce misaligned model selections.
- **Estimated Time:** 15 minutes.

### Exercise 3: Designing a Leakage-Free Temporal Split Protocol (Design)
- **Context:** A national grocery retailer forecasts daily store demand for fresh produce with a 3-day supplier lead time ($h = 3$ days). Feature engineering utilizes 7-day, 14-day, and 28-day rolling average sales, along with day-of-week lag features.
- **Deliverable:** Design a rolling backtest protocol specifying:
  - The exact duration of the Purge Window (in days) and the mathematical justification based on feature lag dependencies.
  - The exact duration of the Embargo Window (in days) and the operational justification based on replenishment lead times.
  - An ASCII diagram detailing the sequence of Train, Purge, Test, and Embargo windows across Fold $k$ and Fold $k+1$.
- **Acceptance Criteria:** Correct identification of buffer lengths; comprehensive diagram showing no overlap in information flow.
- **Estimated Time:** 30 minutes.

### Exercise 4: Critique of an MLflow Tracking Implementation (Critique)
- **Context:** A junior data scientist presents an MLflow tracking implementation for a multi-echelon forecasting system. The script executes a single `mlflow.start_run()` inside a loop iterating through 10,000 SKUs, logging `RMSE` and `MAE` computed on a randomized 80/20 train/test split. No artifacts are logged other than a final text summary.
- **Deliverable:** Write an engineering critique detailing four major design defects in this implementation. For each defect, cite the specific failure mode, the risk to production operations, and the exact architectural remedy.
- **Acceptance Criteria:** Analysis must address hierarchy, temporal leakage, loss function alignment, and artifact auditability.
- **Estimated Time:** 25 minutes.

### Exercise 5: Giacomini-White Instrument Selection for Promo Regimes (Application)
- **Context:** A retailer evaluates a challenger neural forecasting model against an incumbent gradient boosting model. The data science team suspects the challenger excels only during promotional weeks.
- **Deliverable:** Specify the instrument vector $\mathbf{h}_t$ for the Giacomini-White test to evaluate this hypothesis. Define the dimension $q$ of the instrument vector, write out the interaction term $\mathbf{Z}_{t+h}$, state the degrees of freedom for the resulting Wald test statistic, and explain what a rejection of $H_0$ implies for production deployment.
- **Acceptance Criteria:** Correct mathematical formulation of the instrument matrix and degrees of freedom; clear operational interpretation.
- **Estimated Time:** 20 minutes.

### Exercise 6: Building an Automated Champion-Challenger Gate in CI/CD (System Design)
- **Context:** Design an automated CI/CD deployment gate for an enterprise replenishment platform. When a pull request introduces a new model architecture or feature pipeline, a GitHub Actions workflow triggers a backtest execution on a Kubernetes cluster.
- **Deliverable:** Construct a detailed system architecture specification defining:
  - Input artifacts retrieved from the Model Registry.
  - The statistical gate logic (Diebold-Mariano, HLN correction, asymmetric loss threshold).
  - The automated decision branch rules (Pass, Fail, Staged for Canary).
  - The output metadata and diagnostic artifacts written back to the tracking registry.
- **Acceptance Criteria:** Complete architectural description including JSON schema for the gate output payload.
- **Estimated Time:** 35 minutes.

### Exercise 7: MinT Hierarchical Reconciliation Matrix Logging (Application)
- **Context:** An enterprise distribution network has 1 Central DC ($T$) and 2 Regional DCs ($R_1, R_2$). The structural summing matrix is $\mathbf{S} = [1, 1; 1, 0; 0, 1]'$. The base forecast error covariance matrix is estimated as $\hat{\mathbf{W}} = \text{diag}([100, 60, 40])$.
- **Deliverable:** Compute the optimal MinT reconciliation projection operator $\mathbf{P} = (\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1}\mathbf{S}'\mathbf{W}^{-1}$. Given incoherent base forecasts $\hat{\mathbf{y}} = [110, 65, 50]'$ (where the top forecast 110 does not equal $65 + 50 = 115$), compute the reconciled forecasts $\tilde{\mathbf{y}} = \mathbf{S}\mathbf{P}\hat{\mathbf{y}}$. Specify which parameters and tensors must be logged to the parent experiment run.
- **Acceptance Criteria:** Exact matrix multiplication; show that reconciled forecasts are coherent ($\tilde{y}_T = \tilde{y}_{R1} + \tilde{y}_{R2}$).
- **Estimated Time:** 25 minutes.

### Exercise 8: Bartlett vs Rectangular Spectral Variance Boundary Condition (Calculation)
- **Context:** A 1-step ahead forecasting model ($h = 2$, so cutoff lag is 1) evaluated over $T = 10$ periods exhibits sample variance $\hat{\gamma}_0 = 100$ and sample lag-1 autocovariance $\hat{\gamma}_1 = -65$.
- **Deliverable:** 
  1. Compute the rectangular spectral density $\hat{f}_{\text{rect}}(0)$ and explain the mathematical breakdown.
  2. Compute the Bartlett Newey-West spectral density $\hat{f}_{NW}(0)$ with bandwidth $K = 1$.
  3. Explain how experiment tracking systems must handle this boundary condition to prevent CI/CD gate crashes.
- **Acceptance Criteria:** Correct arithmetic showing negative variance for rectangular kernel and positive variance for Bartlett kernel.
- **Estimated Time:** 15 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1: Computing HLN Correction Factor Across Sample Sizes
We are given $h = 4$. The formula for the HLN correction multiplier is:
$$\text{Multiplier}_{HLN} = \sqrt{\frac{T + 1 - 2h + \frac{h(h-1)}{T}}{T}}$$

1. **For $T = 12$:**
   $$\text{Numerator} = 12 + 1 - 2(4) + \frac{4(3)}{12} = 13 - 8 + \frac{12}{12} = 5 + 1 = 6$$
   $$\text{Multiplier} = \sqrt{\frac{6}{12}} = \sqrt{0.5000} = 0.7071$$
   - Degrees of freedom: $\nu = T - 1 = 11$.
   - Two-tailed critical $t$-value at $\alpha = 0.05$: $t_{0.05, 11} = 2.2010$.
   - *Impact:* The multiplier reduces the asymptotic $DM$ test statistic by nearly $30\%$, demonstrating that small samples with multi-step horizons require substantially larger observed performance deltas to achieve statistical significance.

2. **For $T = 26$:**
   $$\text{Numerator} = 26 + 1 - 2(4) + \frac{4(3)}{26} = 27 - 8 + \frac{12}{26} = 19 + 0.4615 = 19.4615$$
   $$\text{Multiplier} = \sqrt{\frac{19.4615}{26}} = \sqrt{0.7485} = 0.8652$$
   - Degrees of freedom: $\nu = T - 1 = 25$.
   - Two-tailed critical $t$-value at $\alpha = 0.05$: $t_{0.05, 25} = 2.0595$.

3. **For $T = 52$:**
   $$\text{Numerator} = 52 + 1 - 2(4) + \frac{4(3)}{52} = 53 - 8 + \frac{12}{52} = 45 + 0.2308 = 45.2308$$
   $$\text{Multiplier} = \sqrt{\frac{45.2308}{52}} = \sqrt{0.8698} = 0.9326$$
   - Degrees of freedom: $\nu = T - 1 = 51$.
   - Two-tailed critical $t$-value at $\alpha = 0.05$: $t_{0.05, 51} \approx 2.0076$.

---

### Answer to Exercise 2: Formulating Asymmetric Loss from Commercial Ledgers
1. **Parameter Derivations:**
   - **Unit Shortage Cost ($c_u$):** Lost margin per dose is Selling Price minus Procurement Cost = $\$80.00 - \$50.00 = \$30.00$. In addition, an emergency distributor fee of $\$15.00$ is incurred.
     $$c_u = \$30.00 + \$15.00 = \$45.00/\text{dose}$$
   - **Unit Holding Cost ($c_o$):** Capital holding cost is $\$2.00$ plus spoilage risk discount of $\$8.00$.
     $$c_o = \$2.00 + \$8.00 = \$10.00/\text{dose}$$
2. **Loss Function Formulation:**
   $$L(y_t, \hat{y}_t) = 45.00 \max(0, y_t - \hat{y}_t) + 10.00 \max(0, \hat{y}_t - y_t)$$
3. **Scenario Evaluations:**
   - **Scenario A (Under-forecast):** $y = 500, \hat{y} = 420 \implies \text{Error } e = +80$ doses (shortage).
     $$L_A = 45.00 \times 80 = \$3,600.00$$
   - **Scenario B (Over-forecast):** $y = 500, \hat{y} = 580 \implies \text{Error } e = -80$ doses (excess).
     $$L_B = 10.00 \times 80 = \$800.00$$
4. **Economic Interpretation:**
   Both scenarios have an identical absolute error of 80 units (yielding identical MAE and MSE). However, Scenario A costs the distributor $\$3,600.00$ while Scenario B costs $\$800.00$—a $4.5\times$ economic asymmetry. Symmetric metrics treat these scenarios as equivalent, blinding tracking platforms to catastrophic stockout behaviors.

---

### Answer to Exercise 3: Designing a Leakage-Free Temporal Split Protocol
1. **Purge Window Duration:**
   - Must equal the maximum lookback window used in feature extraction. Since the pipeline computes 28-day rolling average sales, the Purge Window must be **28 days**. Any training instance within 28 days of the forecast origin would incorporate historical observations that overlap with the forecast evaluation period if rolling features were updated across splits.
2. **Embargo Window Duration:**
   - Must cover the supplier replenishment lead time ($L = 3$ days) plus the ordering review period ($R = 1$ day). Therefore, the Embargo Window must be at least **4 days** (or 3 days minimum if ordering occurs continuously). This eliminates serial correlation leakage caused by pipeline inventory replenishment overlap between adjacent folds.
3. **Sequence Diagram:**
   ```text
   Fold k:
   [ Train: Day 1 ... Day 100 ] -> [ Purge: Day 101 ... Day 128 (28d) ] -> [ Test: Day 129 ... Day 131 (3d) ] -> [ Embargo: Day 132 ... Day 135 (4d) ]
   
   Fold k+1 (Advanced by 7-day step):
   [ Train: Day 8 ... Day 107 ] -> [ Purge: Day 108 ... Day 135 (28d) ] -> [ Test: Day 136 ... Day 138 (3d) ] -> [ Embargo: Day 139 ... Day 142 (4d) ]
   ```

---

### Answer to Exercise 4: Rubric for Critique of MLflow Implementation
A passing critique must identify and analyze the following four architectural defects:
1. **Flat Metric Logging across Hierarchies:** Logging unnested runs inside a single loop obliterates the DAG relationship between national distribution centers and store-level SKUs. *Remedy:* Structure runs with parent-child hierarchy using `mlflow.start_run(nested=True)` or setting `parent_run_id`.
2. **Fatal Temporal Leakage via Random Splitting:** An 80/20 random split shuffles future time periods into the training set, causing lookahead leakage and unconstrained autocorrelation leakage. *Remedy:* Replace random train/test split with an expanding or rolling origin temporal cross-validation protocol enforcing purge and embargo windows.
3. **Economic Misalignment of Metrics:** Tracking only symmetric RMSE and MAE conceals critical stockout versus overage tradeoffs. *Remedy:* Log asymmetric piecewise linear loss (pinball loss) reflecting enterprise unit shortage and holding costs ($c_u, c_o$).
4. **Absence of Diagnostic Artifact Lineage:** Storing only a text summary without serialized model binaries (ONNX), TreeSHAP feature attributions, quantile calibration plots, or residual autocorrelations prevents post-deployment failure diagnostics and violates ISO/IEC 42001 and SR 11-7 model audit standards.

---

### Answer to Exercise 5: Giacomini-White Instrument Selection for Promo Regimes
1. **Instrument Vector Definition:**
   To evaluate whether model superiority is conditioned on promotional events, define:
   $$\mathbf{h}_t = \begin{bmatrix} 1 \\ \text{Promo}_t \end{bmatrix}$$
   where $\text{Promo}_t \in \{0, 1\}$ is a binary indicator of whether a promotion is active during the forecast generation period.
2. **Dimension and Interaction Term:**
   - Dimension: $q = 2$ instruments.
   - Interaction Vector: $\mathbf{Z}_{t+h} = \mathbf{h}_t \otimes d_{t+h} = \begin{bmatrix} d_{t+h} \\ \text{Promo}_t \cdot d_{t+h} \end{bmatrix}$.
3. **Degrees of Freedom and Test Statistic:**
   - The Wald test statistic $GW = T \bar{\mathbf{Z}}' \hat{\mathbf{\Omega}}^{-1} \bar{\mathbf{Z}}$ follows an asymptotic Chi-Square distribution with $q = 2$ degrees of freedom: $\chi^2_2$.
4. **Operational Interpretation:**
   - If $GW > \chi^2_{2, 1-\alpha}$, the null hypothesis of conditional equal predictive ability is rejected. This demonstrates that the difference in forecast accuracy between the models depends significantly on whether a promotion is running.
   - *Deployment Action:* Rather than replacing the incumbent globally, deploy a **hybrid ensemble / routing rule**: route non-promotional forecasting to Model A and route promotional periods to Model B.

---

### Answer to Exercise 6: Automated CI/CD Gate Architecture Specification
```json
{
  "gate_execution_id": "gate-pr-8492-eval-20261104",
  "git_commit_sha": "f3b019a8c2d4",
  "champion_model_id": "lgbm_demand_cdc_v14",
  "challenger_model_id": "tft_mint_demand_cdc_v15_pr8492",
  "backtest_protocol": {
    "type": "rolling_origin",
    "evaluation_folds": 12,
    "forecast_horizon_weeks": 2,
    "loss_function": "asymmetric_piecewise_linear",
    "cost_parameters": { "c_underage_usd": 15.00, "c_overage_usd": 5.00 }
  },
  "statistical_evaluation": {
    "mean_loss_differential_usd": 28.3333,
    "hln_corrected_t_stat": 2.2071,
    "degrees_of_freedom": 11,
    "p_value_hln": 0.0495,
    "alpha_threshold": 0.0500,
    "giacomini_white_promo_p_value": 0.0922
  },
  "decision_gate": {
    "status": "PASS_STAGE_CANARY",
    "rationale": "Challenger statistically outperforms Champion on asymmetric economic loss at alpha=0.05. GW test shows conditional promo stability.",
    "automated_actions": [
      "register_model_version_staging",
      "tag_artifact_digest_sha256",
      "dispatch_canary_5_percent_traffic"
    ]
  }
}
```

---

### Answer to Exercise 7: MinT Hierarchical Reconciliation Matrix Logging
1. **Given Data:**
   $$\mathbf{S} = \begin{bmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad \mathbf{W} = \begin{bmatrix} 100 & 0 & 0 \\ 0 & 60 & 0 \\ 0 & 0 & 40 \end{bmatrix} \implies \mathbf{W}^{-1} = \begin{bmatrix} 0.0100 & 0 & 0 \\ 0 & 0.0167 & 0 \\ 0 & 0 & 0.0250 \end{bmatrix}$$
2. **Compute $\mathbf{S}'\mathbf{W}^{-1}\mathbf{S}$:**
   $$\mathbf{S}'\mathbf{W}^{-1}\mathbf{S} = \begin{bmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \end{bmatrix} \begin{bmatrix} 0.0100 & 0 & 0 \\ 0 & 0.0167 & 0 \\ 0 & 0 & 0.0250 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 0.0267 & 0.0100 \\ 0.0100 & 0.0350 \end{bmatrix}$$
3. **Compute Inversion and $\mathbf{P}$:**
   $$\det = (0.0267)(0.0350) - (0.0100)^2 = 0.0009345 - 0.0001000 = 0.0008345$$
   $$(\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1} = \frac{1}{0.0008345} \begin{bmatrix} 0.0350 & -0.0100 \\ -0.0100 & 0.0267 \end{bmatrix} = \begin{bmatrix} 41.941 & -11.983 \\ -11.983 & 31.995 \end{bmatrix}$$
   $$\mathbf{P} = (\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})^{-1} \mathbf{S}' \mathbf{W}^{-1} = \begin{bmatrix} 0.2996 & 0.6990 & -0.2996 \\ 0.2001 & -0.2001 & 0.7999 \end{bmatrix}$$
4. **Reconciled Forecasts:**
   Given base forecasts $\hat{\mathbf{y}} = [110, 65, 50]'$:
   $$\tilde{\mathbf{b}} = \mathbf{P} \hat{\mathbf{y}} = \begin{bmatrix} 0.2996(110) + 0.6990(65) - 0.2996(50) \\ 0.2001(110) - 0.2001(65) + 0.7999(50) \end{bmatrix} = \begin{bmatrix} 32.956 + 45.435 - 14.980 \\ 22.011 - 13.007 + 39.995 \end{bmatrix} = \begin{bmatrix} 63.411 \\ 48.999 \end{bmatrix}$$
   $$\tilde{\mathbf{y}} = \mathbf{S} \tilde{\mathbf{b}} = \begin{bmatrix} 63.411 + 48.999 \\ 63.411 \\ 48.999 \end{bmatrix} = \begin{bmatrix} 112.41 \\ 63.41 \\ 49.00 \end{bmatrix}$$
   *Check:* $63.41 + 49.00 = 112.41$ (Coherent).
5. **Logged Artifacts:** The parent run must log $\mathbf{S}$ (Parquet), $\hat{\mathbf{W}}$ (Parquet), $\mathbf{P}$ (Parquet), and the condition number $\kappa(\mathbf{S}'\mathbf{W}^{-1}\mathbf{S})$.

---

### Answer to Exercise 8: Bartlett vs Rectangular Spectral Variance Boundary Condition
1. **Rectangular Kernel Computation:**
   $$\hat{f}_{\text{rect}}(0) = \hat{\gamma}_0 + 2\hat{\gamma}_1 = 100 + 2(-65) = 100 - 130 = -30$$
   - *Mathematical Breakdown:* The variance estimate $\hat{V}(\bar{d}) = -30 / 10 = -3.0$. Computing standard error $\text{SE} = \sqrt{-3.0}$ yields an imaginary number / `NaN`, crashing the automated statistical testing harness.
2. **Bartlett Newey-West Computation ($K = 1$):**
   $$\hat{f}_{NW}(0) = \hat{\gamma}_0 + 2\left(1 - \frac{1}{1 + 1}\right)\hat{\gamma}_1 = 100 + 2(0.5)(-65) = 100 - 65 = +35$$
   $$\hat{V}_{NW}(\bar{d}) = \frac{35}{10} = 3.5 \implies \text{SE}_{NW} = \sqrt{3.5} = 1.8708$$
3. **Engineering Mitigation in Tracking Platforms:**
   Production tracking libraries must never rely on bare rectangular truncation kernels. The testing engine must implement an automatic fallback or default to the Bartlett kernel, logging a diagnostic warning `RECTANGULAR_KERNEL_NEGATIVE_VARIANCE_FALLBACK_BARTLETT` to the run metadata ledger.

---

## 14. Interview questions

The following questions evaluate technical depth, mathematical understanding, and practical architectural experience during senior MLOps and Supply Chain Data Science interviews.

### Question 1: Why does random $K$-fold cross-validation fail in supply chain forecasting experiment tracking?
- **What a strong answer contains:** A strong answer immediately identifies two distinct mechanisms: **lookahead leakage** (future events, promotions, or macro shocks leaking into historical lag features) and **serial correlation leakage** (multi-step forecast errors sharing unobserved moving-average dependency across time). The candidate explicitly details the necessity of temporal rolling/expanding backtests and defines both **purge windows** (removing overlapping feature calculation periods) and **embargo windows** (buffering out-of-sample lead-time dependency).
- **What a weak answer reveals:** The candidate only mentions that "time series data has an order" or suggests simple chronological train/test splits without mentioning lead-time dependency, purge protocols, or serial correlation.

### Question 2: How does the Diebold-Mariano test differ from a standard paired $t$-test on validation errors?
- **What a strong answer contains:** The candidate explains that a standard paired $t$-test assumes the observation pairs are independent and identically distributed ($i.i.d.$). In multi-step time series forecasting ($h > 1$), forecast errors exhibit an intrinsic moving-average dependency of order $MA(h-1)$ due to overlapping forecast horizons. The Diebold-Mariano test explicitly models this serial correlation by calculating the long-run variance of the loss differential via spectral density estimation at frequency zero using autocovariance sequences.
- **What a weak answer reveals:** The candidate assumes the tests are identical or states that DM is merely "a $t$-test applied to time series" without articulating the breakdown of the independence assumption or the role of autocovariances.

### Question 3: What is the purpose of the Harvey-Leybourne-Newbold (HLN) correction?
- **What a strong answer contains:** The candidate states that the original asymptotic Diebold-Mariano test suffers from severe Type I error inflation (over-sizing) in finite samples, particularly when the evaluation period $T$ is small ($T < 30$) and the forecast horizon $h$ is large. HLN applies an explicit finite-sample correction multiplier based on $T$ and $h$ and compares the statistic against Student's $t$-distribution with $T-1$ degrees of freedom rather than the standard normal distribution, preventing premature false discoveries.
- **What a weak answer reveals:** The candidate is unaware of finite-sample bias in time-series hypothesis testing, confusing sample size adjustments with standard Bessel degrees-of-freedom corrections.

### Question 4: Under what conditions does the rectangular kernel variance estimator produce invalid results in the DM test, and how do you resolve it?
- **What a strong answer contains:** The candidate notes that the rectangular truncation kernel $\hat{\gamma}_0 + 2\sum_{k=1}^{h-1}\hat{\gamma}_k$ is not guaranteed to be positive semi-definite in finite samples. If sample autocovariances at lag 1 are strongly negative (e.g., $\hat{\gamma}_1 < -\frac{1}{2}\hat{\gamma}_0$), the estimated variance becomes negative, causing standard error calculations to fail (`NaN`). The standard remedy is the Bartlett Newey-West kernel, which scales autocovariance lags by linearly decaying weights $(1 - \frac{k}{K+1})$, guaranteeing positive semi-definiteness.
- **What a weak answer reveals:** The candidate assumes sample variance can never be negative, failing to understand how unweighted truncation in spectral density estimation causes mathematical breakdowns.

### Question 5: What is the critical distinction between unconditional predictive ability (Diebold-Mariano) and conditional predictive ability (Giacomini-White)?
- **What a strong answer contains:** The candidate explains that Diebold-Mariano tests unconditional predictive ability ($\mathbb{E}[d_t] = 0$), averaging performance across all historical states. Giacomini-White tests conditional predictive ability ($\mathbb{E}[d_{t+h} \mid \mathcal{F}_t] = 0$), evaluating whether future superiority can be predicted conditioned on state information observable at the forecast origin (e.g., promotional flags, demand volatility, inventory regimes). Furthermore, the candidate notes that GW explicitly accounts for parameter estimation uncertainty under rolling window refitting schemes.
- **What a weak answer reveals:** The candidate treats the tests as interchangeable or cannot provide a concrete business example of why a model might outperform conditionally despite having identical unconditional accuracy.

### Question 6: Why is symmetric RMSE an unacceptable primary metric for supply chain experiment tracking?
- **What a strong answer contains:** The candidate explains that supply chain economics are inherently asymmetric: unit stockout costs ($c_u$) almost always diverge sharply from unit holding costs ($c_o$). Quadratic symmetric metrics (RMSE) penalize positive and negative errors identically, forcing algorithms to minimize variance around the mean, which systematically induces stockouts in skewed demand environments. A strong candidate advocates for asymmetric piecewise linear pinball loss parametrized directly by enterprise cost ledgers.
- **What a weak answer reveals:** The candidate defends RMSE on mathematical grounds (differentiability, convexity) without recognizing its decoupling from supply chain profitability and operational reality.

### Question 7: How do you architect an experiment tracking registry for a catalog of 100,000 SKUs without UI performance degradation?
- **What a strong answer contains:** The candidate proposes a two-tier architectural design: an aggregate parent run tracking global pipeline hyperparameters, Git revisions, and network-level metrics, with child runs grouped logically by echelon, category, or cluster. Fine-grained SKU-level metrics and loss differentials are stored as structured tabular artifacts (e.g., compressed Parquet files in S3/GCS linked via metadata URI) rather than individual scalar rows in a transactional relational metadata database, preventing metadata database connection pooling exhaustion and UI latency collapse.
- **What a weak answer reveals:** The candidate suggests logging 100,000 individual MLflow runs per pipeline execution, unaware of database query bottlenecks and UI indexing limits.

### Question 8: How do you address the Multiple Testing Problem when comparing candidate models across thousands of independent SKU backtests?
- **What a strong answer contains:** The candidate identifies that testing 10,000 SKUs at a standard significance level of $\alpha = 0.05$ will yield approximately 500 false discoveries purely by chance under the null hypothesis. To control the false discovery rate, the candidate details the Benjamini-Hochberg (FDR) procedure or Holm-Bonferroni Family-Wise Error Rate (FWER) adjustments, sorting empirical $p$-values and establishing dynamic significance thresholds across the catalog.
- **What a weak answer reveals:** The candidate does not recognize the multiple comparisons problem, assuming that an $\alpha = 0.05$ threshold can be applied independently to thousands of tests without statistical correction.

### Question 9: How should an experiment tracking system capture hierarchical forecast reconciliation parameters?
- **What a strong answer contains:** The candidate explains that logging base node forecasts alone is insufficient because operational inventory decisions are driven by reconciled forecasts. The tracking system must record the structural summing matrix $\mathbf{S}$, the error covariance estimation method (e.g., MinT with shrinkage), the shrinkage intensity parameter $\lambda$, and the resulting projection matrix $\mathbf{P}$. The candidate mentions logging matrix conditioning metrics to detect numerical instability.
- **What a weak answer reveals:** The candidate assumes reconciliation is an unlogged post-processing script, failing to treat reconciliation matrices as core versioned model artifacts.

### Question 10: How do you prevent data leakage when engineering rolling promotional features across backtest folds?
- **What a strong answer contains:** The candidate states that promotional schedules are often planned months in advance, creating tempting lookahead opportunities. Rolling features (such as 4-week promotional uplift averages) must strictly compute statistics on observations strictly prior to the forecast cutoff origin $t_k$. Furthermore, any historical promotional uplift feature that uses a window overlapping with the lead-time horizon must be purged via explicit purge window bounds.
- **What a weak answer reveals:** The candidate suggests calculating promotional uplift as a global static feature across the entire historical dataset before executing the backtest loop.

---

## 15. Key terms

- **Artifact Digest:** A cryptographic hash (SHA-256) verifying the immutability of serialized model weights, calibration tables, and configuration files.
- **Asymmetric Loss Function:** A loss metric applying unequal penalties to positive and negative forecast errors based on unit shortage ($c_u$) and holding ($c_o$) costs.
- **Autocorrelation Leakage:** Cross-validation contamination where serially dependent errors across multi-step replenishment cycles inflate out-of-sample performance estimates.
- **Bartlett Kernel:** A linearly decaying weighting kernel in spectral density estimation that guarantees positive semi-definiteness of the long-run variance estimator.
- **Benjamini-Hochberg Procedure:** A statistical method controlling the False Discovery Rate across thousands of simultaneous SKU model hypothesis tests.
- **Censored Demand:** Point-of-sale sales data that truncates true unconstrained customer demand during historical stockout events.
- **Condition Number ($\kappa$):** A numerical diagnostic measuring the sensitivity of the MinT reconciliation matrix inversion to estimation error in the covariance tensor.
- **Diebold-Mariano (DM) Test:** An asymptotic hypothesis test evaluating the null hypothesis of equal forecast accuracy between two competing time series models.
- **Embargo Window:** A temporal buffer enforced immediately after an evaluation window before subsequent training samples can be drawn, mitigating lead-time error correlation.
- **Expanding Window Backtest:** A temporal evaluation scheme where the training origin is anchored and the training set expands sequentially as time progresses.
- **False Discovery Rate (FDR):** The expected proportion of false positive model promotions among all statistically significant test rejections.
- **Feature Schema Hash:** A deterministic digest of feature names, data types, and transformation versions logged to prevent silent feature store schema drift.
- **Giacomini-White (GW) Test:** A statistical framework evaluating conditional predictive ability based on an observable information set at the forecast origin.
- **Harvey-Leybourne-Newbold (HLN) Correction:** A small-sample finite-size correction to the Diebold-Mariano statistic evaluated against Student's $t$-distribution.
- **Hierarchical Reconciliation Matrix:** A linear projection operator (e.g., MinT) mapping base forecasts across network tiers into mathematically coherent forecasts.
- **Information Set ($\mathcal{F}_t$):** The operational data filtration observable at forecast cutoff origin $t$, used to condition Giacomini-White tests.
- **Ljung-Box Test:** A diagnostic portmanteau test evaluating whether residual forecast autocorrelations are jointly zero up to lag $m$.
- **Long-Run Variance:** The spectral density of a time series at frequency zero, capturing the sum of all autocovariance terms in serially correlated processes.
- **Loss Differential ($d_t$):** The paired point-in-time difference in loss between a champion and challenger model: $d_t = L(y_t, \hat{y}_{A,t}) - L(y_t, \hat{y}_{B,t})$.
- **Minimum Trace (MinT):** An optimal hierarchical forecast reconciliation algorithm minimizing total error variance using covariance shrinkage.
- **Multiple Comparisons Problem:** The inflation of false discovery rates when hypothesis tests are conducted across thousands of independent SKU models simultaneously.
- **Parent-Child Run DAG:** A nested experiment tracking structure linking high-level pipeline orchestration runs to localized echelon and SKU-level leaf runs.
- **Pinball Loss:** A piecewise linear loss function parametrized by quantile $\tau$, widely used to calibrate asymmetric inventory shortage and overage penalties.
- **Purge Window:** The deliberate deletion of historical training samples preceding a forecast origin whose feature calculation windows intersect the test period.
- **Rectangular Kernel:** An unweighted autocovariance truncation kernel capping covariance terms at lag $h-1$ for theoretical moving-average processes.
- **Rolling Origin Backtest:** An evaluation protocol where a fixed-length training window slides forward across historical time, re-estimating out-of-sample performance.
- **Shrinkage Covariance Matrix:** A regularized convex combination of sample covariance and diagonal target used to stabilize MinT reconciliation.
- **Structural Summing Matrix ($\mathbf{S}$):** A binary topological matrix mapping bottom-level SKU series to intermediate and top-level network aggregates.
- **TreeSHAP Lineage:** The versioned persistence of tree-based Shapley Additive Explanations across experiment runs to monitor feature attribution stability.
- **Wald Test Statistic:** A quadratic test statistic used in the Giacomini-White framework following an asymptotic $\chi^2_q$ distribution.

---

## 16. Sources

- **Diebold, F. X., & Mariano, R. S. (1995).** Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263. [DOI: 10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599)
- **Harvey, D., Leybourne, S., & Newbold, P. (1997).** Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291. [DOI: 10.1016/S0169-2070(96)00719-4](https://doi.org/10.1016/S0169-2070(96)00719-4)
- **Giacomini, R., & White, H. (2006).** Tests of Conditional Predictive Ability. *Econometrica*, 74(6), 1545–1578. [DOI: 10.1111/j.1468-0262.2006.00718.x](https://doi.org/10.1111/j.1468-0262.2006.00718.x)
- **Newey, W. K., & West, K. D. (1987).** A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708. [DOI: 10.2307/1913610](https://doi.org/10.2307/1913610)
- **Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019).** Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization. *Journal of the American Statistical Association*, 114(526), 804–819. [DOI: 10.1080/01621459.2018.1448825](https://doi.org/10.1080/01621459.2018.1448825)
- **Benjamini, Y., & Hochberg, Y. (1995).** Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), 289–300. [DOI: 10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
- **López de Prado, M. (2018).** *Advances in Financial Machine Learning*. John Wiley & Sons. [ISBN: 978-1119482086]
- **Board of Governors of the Federal Reserve System & Office of the Comptroller of the Currency. (2011).** *Supervisory Guidance on Model Risk Management (SR Letter 11-7 / OCC Bulletin 2011-12)*. Washington, D.C.
- **National Institute of Standards and Technology. (2023).** *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*. U.S. Department of Commerce. [DOI: 10.6028/NIST.AI.100-1](https://doi.org/10.6028/NIST.AI.100-1)
- **International Organization for Standardization & International Electrotechnical Commission. (2023).** *ISO/IEC 42001:2023 Information technology — Artificial intelligence — Management system*. Geneva, Switzerland.
- **International Organization for Standardization & International Electrotechnical Commission. (2024).** *ISO/IEC 5259:2024 Artificial intelligence — Data quality for analytics and machine learning*. Geneva, Switzerland.
- **Lundberg, S. M., & Lee, S.-I. (2017).** A Unified Approach to Interpreting Model Predictions. In *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 4765–4774.
- **Hyndman, R. J., & Athanasopoulos, G. (2021).** *Forecasting: Principles and Practice* (3rd ed.). OTexts: Melbourne, Australia.

---

## 17. Links

- **Upward Index:** Module 44 Overview and Topic Map — [`_index.md`](_index.md)
- **Sibling Topics:**
  - Topic 01: Lifecycle and Problem Framing — [`01_lifecycle-framing.md`](01_lifecycle-framing.md)
  - Topic 02: Versioning and Reproducibility — [`02_versioning-reproducibility.md`](02_versioning-reproducibility.md)
  - Topic 04: Point-in-Time Features and Leakage Prevention — [`04_point-in-time-features.md`](04_point-in-time-features.md)
  - Topic 05: Orchestration Pipelines and Lineage — [`05_orchestration-pipelines.md`](05_orchestration-pipelines.md)
  - Topic 09: Retraining Triggers and Champion–Challenger Staging — [`09_retraining-champion-challenger.md`](09_retraining-champion-challenger.md)
- **Survey File:** Core Course Module Map — [`../../../course/44_mlops_production_analytics.md`](../../../course/44_mlops_production_analytics.md)
- **Related Methods Modules:**
  - Module 07: Time Series Forecasting and Hierarchy — [`../../02_methods/07_time_series_forecasting/`](../../02_methods/07_time_series_forecasting/)
  - Module 08: Error Metrics and Financial Translation — [`../../02_methods/08_error_metrics/`](../../02_methods/08_error_metrics/)
  - Module 19: Multi-Echelon Inventory Optimization and Policy — [`../../03_commerce/19_inventory_policy/`](../../03_commerce/19_inventory_policy/)
