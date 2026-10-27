# Module 27 — Datasets, Tools, and Benchmarks

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 18 hours (6 lecture, 8 lab, 4 project)
> **Track fit:** Common core
> **Prerequisites:** Modules 01A, 02, 03, 07
> **Tools:** Python, DuckDB, Kaggle CLI, Git, a reference manager, a benchmark notebook
> **Industry question:** "Which dataset and tool should I trust for this decision — and what will the benchmark hide?"

---

## 1. Why this module exists

Most supply chain analytics courses use one clean dataset. Industry uses many messy ones. The choice of dataset determines what you can learn, what you can claim, and what will fail in production.

There are three common traps:

1. **Benchmark transfer.** A method that wins M5 does not automatically win in a pharmaceutical cold chain or a semiconductor allocation problem. The data-generating process is different, and the cost of error is different.
2. **License blindness.** A dataset that is free for a competition may not be usable commercially. A partner dataset may forbid redistribution. A synthetic dataset may encode unrealistic behavior.
3. **Metric gaming.** A leaderboard rewards a metric, not a decision. A model can improve WRMSSE while worsening the tail that drives stockouts.

This module builds a **dataset and tool selection discipline**. It is a catalogue, but it is also a warning label.

## 2. Learning outcomes

By the end of this module, you can:

- Select a dataset based on decision fit, grain, history, label quality, and license.
- Read a dataset card and identify its known defects and censoring.
- Distinguish public, competition, commercial, partner, and synthetic data.
- Reproduce a benchmark and explain why its result may not transfer.
- Choose a forecasting, optimization, simulation, and MLOps tool stack for a given constraint.
- Audit a dataset for leakage, contamination, and representativeness.
- Build a personal resource library with licenses and access dates.
- Write a benchmark critique that a technical reviewer will accept.

## 3. Mental model: dataset fit is decision fit

```
Decision → Required grain → Required history → Required labels → Required exogeneity
        → License and access → Quality and freshness → Benchmark comparability
```

A dataset is not "good" in the abstract. It is fit for a decision or it is not.

### 3.1 The seven fit tests

| Test | Question | Failure symptom |
|---|---|---|
| Decision fit | Does the data support the action? | Model predicts something nobody can act on |
| Grain fit | Is it at the decision grain? | Forecasting SKU-week, planning SKU-store-week |
| History fit | Enough cycles for seasonality and events? | One year of data, two Christmases missing |
| Label fit | Is the outcome observable and timely? | Late delivery label arrives after the planning cutoff |
| Exogeneity | Are drivers available at decision time? | Using a post-decision price or promotion |
| License fit | Can the data be used for this purpose? | Competition data used commercially |
| Quality fit | Is it complete, accurate, and stable? | Master-data drift, missing stores, censored sales |

### 3.2 Data provenance classes

| Class | Example | Can use commercially? | Can redistribute? | Typical risk |
|---|---|---|---|---|
| Public open | UCI, World Bank | Usually yes, check license | Usually yes with attribution | Quality, age |
| Competition | Kaggle M5 | Check competition rules and license | Usually no raw redistribution | Metric-specific, censored |
| Commercial | Nielsen, Circana, GfK | Under contract | No | Cost, coverage |
| Partner | Retailer POS, carrier EDI | Under data-sharing agreement | No | Access, latency, completeness |
| Internal | ERP, WMS, TMS | Yes, within policy | No | Master data, access controls |
| Synthetic | Synthea-like, generated | Usually yes | Yes | Unrealistic behavior |
| Scraped | Web prices, reviews | Legally uncertain | No | Terms of service, reliability |

> **Rule:** "publicly downloadable" is not the same as "free to use." Read the license.

## 4. Dataset catalogue

### 4.1 Retail and demand

| Dataset | Grain and scale | Period | Access | Known issues | Best use |
|---|---|---|---|---|---|
| M5 (Walmart) | Daily SKU-store, 42,840 series, 28-day horizon | 2011–2016 | Kaggle | Censored by stockouts; US retail only | Hierarchical forecasting, promotions |
| M4 | 100,000 series, six frequencies | Varies | Public | Mixed domains; not supply chain specific | Method benchmarking |
| Corporación Favorita | Daily store-item sales and promotions | 2013–2017 | Kaggle | Grocery; missing data; Ecuador | Promotions, perishables |
| Rossmann | Daily store sales, 1,115 stores | 2013–2015 | Kaggle | Store-level, not SKU; Germany | Store forecasting, holidays |
| UCI Online Retail II | Transaction lines | 2009–2011 | UCI | Returns, cancellations, one UK retailer | Basket, returns, cleaning |
| DataCo Smart Supply Chain | Orders and shipping | 2015–2018 | Mendeley | Synthetic-like; limited realism | Teaching late delivery |
| Instacart Market Basket | Anonymized orders | 2017 | Public | No prices or inventory | Basket affinity |
| Dunnhumby Complete Journey | Household transactions | 2 years | Registration | Panel, not full demand | Customer analytics |

### 4.2 Manufacturing, quality, and maintenance

| Dataset | Grain and scale | Access | Known issues | Best use |
|---|---|---|---|---|
| NASA C-MAPSS | Turbofan sensor trajectories, four subsets | Public | Simulated degradation | Remaining useful life |
| Bosch Production Line | Production line measurements | Kaggle | Extreme imbalance; anonymized | Quality classification |
| SECOM | 1,567 rows, 590 sensors | UCI | High-dimensional, missing | Fault detection |
| Scania APS Failure | Truck component failures | UCI | Cost-sensitive classes | Failure prediction |
| MetroPT | Air compressor telemetry | UCI | Imperfect anomaly labels | Anomaly detection |
| SKAB | Water pump anomalies | GitHub | Small | Streaming anomaly detection |

### 4.3 Logistics, trade, and external signals

| Dataset | Grain | Access | Known issues | Best use |
|---|---|---|---|---|
| UN Comtrade | Country-product-year trade flows | Public API | Reporting lags, re-exports | Trade and sourcing |
| World Bank LPI | Country logistics index | Public | Survey-based, periodic | Network screening |
| GDELT | Event mentions | Public API | Noise, duplication | Disruption signals |
| ACLED | Conflict events | Registration | Coverage bias | Supplier risk |
| NOAA | Weather and climate | Public | Station coverage | Demand and disruption |
| Open Supply Hub | Facility locations | Public | Self-reported | Supplier mapping |
| Freightos FBX / Drewry WCI | Freight rate indices | Commercial/public summaries | Lane-specific | Transport scenarios |

### 4.4 Benchmark suites

| Benchmark | What it measures | Why it matters | Caveat |
|---|---|---|---|
| M5 | Hierarchical retail forecasting with promotions | Realistic retail structure and WRMSSE | US Walmart, 28-day horizon |
| M4 | General forecasting across frequencies | Broad method comparison | Mixed domains |
| Favorita | Grocery demand with promotions | Perishables and promotion effects | Ecuador, missing data |
| Rossmann | Store sales | Holiday and store effects | Store-level only |
| GEFCom | Energy load forecasting | Probabilistic and hierarchical | Energy-specific |
| TSFM-Bench | Time-series foundation models | Zero/few/full-shot comparison | Contamination risk |
| GIFT-Eval | Held-out foundation-model evaluation | Reduces leakage | Newer, evolving |

## 5. Benchmark pitfalls

### 5.1 The six ways a benchmark misleads

1. **Metric mismatch.** WRMSSE rewards hierarchy-weighted accuracy; your decision may depend on the 95th percentile.
2. **Censoring.** Stockouts remove demand from the data. A model learns "demand is low" when the shelf was empty.
3. **Contamination.** Foundation models may have seen the test period. One audit found only about 7% of datasets were never used in pretraining or fine-tuning.
4. **Temporal leakage.** Random splits on time-series data inflate performance.
5. **Domain mismatch.** A model trained on US grocery may fail on pharmaceutical cold chain.
6. **Leaderboard overfitting.** Repeated public-leaderboard submission turns the test set into a validation set.

### 5.2 The M5 lesson, stated carefully

M5 had 42,840 daily series, 1,913 training days, and a 28-day horizon. The winner combined 220 LightGBM models with a Tweedie objective and achieved WRMSSE 0.520, 22.4% better than the ES bu benchmark. All top-50 methods were ML and beat the statistical benchmark by more than 14%.

What transfers:

- Cross-learning across related series.
- Exogenous variables such as price, promotions, and holidays.
- Forecast combination.
- Time-based validation.

What does not transfer automatically:

- The exact hierarchy weights.
- The promotion structure of another retailer.
- The assumption that ML always beats a well-tuned statistical model.
- The 28-day horizon.

### 5.3 Foundation-model caution

TSFM-Bench (KDD 2025) found no single dominant model across 21 datasets; zero-shot models beat task-specific models on 10 of 21; scaling laws did not hold strictly; LLM-based models generally underperformed. A 2025 audit found serious contamination risk. A separate electricity-price study found a classical bi-seasonal MSTL model was never statistically beaten.

The correct conclusion is not "foundation models are bad." It is: **evaluate them against strong local baselines, on your data, with a contamination audit.**

## 6. Tool stack

### 6.1 Selection criteria

| Criterion | Question |
|---|---|
| Fit | Does it support the method and constraints? |
| Maturity | Is it maintained and documented? |
| Performance | Does it meet the latency and scale requirement? |
| License | Is commercial use permitted? |
| Operability | Can the team run, monitor, and upgrade it? |
| Cost | License, cloud, and people cost |
| Ecosystem | Integration with the warehouse and planning tools |
| Lock-in | How hard is it to replace? |

### 6.2 Forecasting

| Tool | Strength | Limitation |
|---|---|---|
| statsforecast | Fast classical models, scalable | Limited deep learning |
| sktime | Unified time-series API | Learning curve |
| darts | Broad model set, easy API | Memory at scale |
| GluonTS | Probabilistic deep learning | Framework-specific |
| LightGBM | Strong tabular and cross-learning baseline | Needs feature engineering |
| Prophet | Interpretable trend/seasonality | Weak on complex promotions |
| PyTorch Forecasting | Deep models and covariates | Engineering overhead |

### 6.3 Optimization

| Tool | Strength | Limitation |
|---|---|---|
| OR-Tools | Free, strong routing and CP-SAT | Less familiar to OR purists |
| Pyomo | Modeling language, many solvers | Solver licensing for large MIP |
| HiGHS | Open-source LP/MIP | Fewer features than commercial |
| Gurobi / CPLEX | Fast, robust, support | Cost |
| PuLP | Simple LP modeling | Limited advanced features |
| cvxpy | Convex optimization | Not for general MIP |

### 6.4 Simulation and data

| Tool | Use |
|---|---|
| SimPy | Discrete-event simulation in Python |
| AnyLogic / Arena | Commercial simulation with visualization |
| Salabim | Python simulation with animation |
| DuckDB | Local analytical SQL |
| Polars | Fast dataframes |
| dbt | SQL transformation and tests |
| Airflow / Dagster / Prefect | Orchestration |
| Kafka | Event streaming |
| Delta Lake / Iceberg | Table formats and lineage |

### 6.5 MLOps and evaluation

| Tool | Use |
|---|---|
| MLflow | Tracking, registry, packaging |
| DVC | Data and model versioning |
| Evidently | Drift and data-quality reports |
| Great Expectations | Data-quality tests |
| Optuna | Hyperparameter search |
| SHAP | Feature attribution |
| Fairlearn | Fairness metrics |
| Docker / Kubernetes | Packaging and deployment |
| Terraform | Infrastructure as code |

## 7. Dataset selection framework

### 7.1 Scorecard

| Dimension | Weight | 1 (poor) | 3 (adequate) | 5 (strong) |
|---|---:|---|---|---|
| Decision fit | 25% | No action link | Partial link | Direct link to decision |
| Grain | 15% | Wrong grain | Aggregatable | Exact grain |
| History | 15% | <1 cycle | 1–2 cycles | 3+ cycles with events |
| Label quality | 15% | Proxy, late | Reasonable | Observed, timely |
| License | 10% | Unclear | Restricted | Clear commercial use |
| Quality | 10% | Major gaps | Known issues | Audited |
| Freshness | 10% | Stale | Periodic | Timely |

A score below 3 on license or decision fit is a stop sign, regardless of total.

### 7.2 Data card template

```markdown
# Dataset card

## Purpose
## Source and owner
## Collection method and period
## Grain and size
## License and permitted use
## Known defects
- Missingness:
- Censoring:
- Duplicates:
- Master-data issues:
- Coverage bias:
## Label definition and timing
## Recommended use
## Not recommended for
## Access date
```

## 8. Labs

### Lab 27.1 — Dataset audit

**Objective:** Audit M5 or Favorita against the seven fit tests.

**Steps:** Download the data; read the license; profile grain, missingness, and zero sales; check stockout censoring; identify promotion fields; write the data card.

**Deliverable:** Data card and a one-page fit memo.

**Acceptance criteria:** License stated; grain and period correct; at least three defects documented; a clear "fit for / not fit for" verdict.

### Lab 27.2 — Benchmark reproduction

**Objective:** Reproduce a seasonal-naive and one ML baseline on a public dataset.

**Steps:** Build a rolling-origin backtest; compute WAPE, MASE, and bias; compare with the published benchmark; explain any gap.

**Deliverable:** Notebook and a 500-word comparison.

**Acceptance criteria:** Same split and metric; results reproducible; gap explained by data or method differences.

### Lab 27.3 — License and provenance audit

**Objective:** Determine whether a dataset can be used in a commercial project.

**Steps:** Read the license and terms; identify restrictions on redistribution, commercial use, and derivative models; write a memo for legal review.

**Deliverable:** License memo with a recommendation.

**Acceptance criteria:** Quotes the relevant clauses; distinguishes public download from commercial permission.

### Lab 27.4 — Tool selection

**Objective:** Choose a stack for a given constraint.

**Steps:** Define the decision, scale, latency, budget, and team skills; score three options; run a small proof of concept; document the trade-off.

**Deliverable:** Tool decision record.

**Acceptance criteria:** Criteria are explicit; proof of concept runs; lock-in and cost are addressed.

### Lab 27.5 — Benchmark critique

**Objective:** Critique a leaderboard claim.

**Steps:** Read the competition report and the winning solution; identify the metric, split, and any leakage; compare with a production decision; write five risks.

**Deliverable:** Two-page critique.

**Acceptance criteria:** Metric and split correct; at least three transfer risks; a recommendation.

## 9. Assignment

**Personal resource library and dataset audit (individual, 10 hours).**

Build a library of at least 15 datasets and 10 tools. For each dataset, record:

- Link, owner, license, access date.
- Grain, period, size.
- Known defects.
- One decision it could support.
- One decision it could not support.

For each tool, record:

- Category, license, maturity.
- Strength and limitation.
- One lab or project where you used it.

Then write a 700-word memo recommending a dataset and stack for one capstone archetype, with a risk section.

## 10. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Dataset fit | 25% | Chooses by popularity | Applies fit tests | Explains trade-offs and censoring |
| License and provenance | 20% | Assumes public means free | Reads license | Distinguishes use, redistribution, and model rights |
| Benchmark literacy | 20% | Repeats leaderboard | Explains metric and split | Identifies contamination and transfer limits |
| Tool judgment | 15% | Lists tools | Chooses by fit | Documents operability, cost, and lock-in |
| Resource library | 10% | Links only | Complete cards | Maintained with review dates |
| Communication | 10% | Dense list | Clear memo | Decision-ready with risks |

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Benchmark worship | "It won M5" | Metric and domain mismatch | Production failure | Local baseline and backtest |
| License violation | "It was on Kaggle" | Competition rules ignored | Legal risk | License review |
| Censored demand | Zero sales treated as zero demand | Stockouts | Under-forecast | Censoring model or stockout flag |
| Synthetic realism | Perfectly behaved data | Generator assumptions | False confidence | Validate against real data |
| Stale data | Last update years ago | No maintenance | Wrong decisions | Freshness check |
| Contamination | Foundation model scores too well | Pretraining overlap | Inflated evaluation | Held-out or post-cutoff test |
| Tool sprawl | Five overlapping libraries | No standard | Maintenance burden | Standard stack |
| Vendor lock-in | Proprietary format | Migration cost | Negotiating weakness | Exit plan |
| Small sample | 20 series | Overfitting | Poor generalization | Pooling or simpler model |
| No data card | Undocumented joins | Hidden defects | Reproducibility failure | Card and dictionary |

## 12. Interview and discussion questions

1. **How do you choose a dataset for a forecasting project?** Hint: decision, grain, history, labels, license, quality, freshness.
2. **What is censored demand, and how do you handle it?** Hint: stockouts hide demand; use stockout flags, substitution models, or inventory-aware likelihoods.
3. **Why might a model that wins M5 fail in your company?** Hint: different hierarchy, promotions, horizon, cost, and data quality.
4. **How do you audit a foundation model for contamination?** Hint: compare training cutoff, use post-cutoff or held-out data, check benchmark lineage.
5. **What does "public" mean for data licensing?** Hint: downloadable is not commercial; check terms.
6. **How do you compare two forecasting libraries?** Hint: accuracy, speed, probabilistic support, covariates, maintenance, license.
7. **When would you use a commercial solver?** Hint: large MIP, solve time, support, reliability; compare with open-source HiGHS.
8. **How do you evaluate a simulation tool?** Hint: input modeling, validation, experimentation, cost, integration.
9. **What data would you refuse to use?** Hint: unlicensed, coerced, personal without basis, or competitor non-public.
10. **How do you keep a resource library current?** Hint: review dates, owners, change triggers.

## 13. Key terms

| Term | Definition |
|---|---|
| Dataset card | A structured summary of a dataset's provenance, grain, license, and defects. |
| Censoring | Demand that is unobserved because stock ran out. |
| Contamination | Test data that appeared in model pretraining or tuning. |
| Benchmark | A standard dataset and metric used for comparison. |
| Leaderboard overfitting | Repeated tuning against a public test set. |
| Grain | The level of detail of a record. |
| Provenance | The origin and history of data. |
| License | Legal terms governing use and redistribution. |
| Synthetic data | Data generated to mimic real data. |
| External validity | Whether a result transfers to another setting. |
| Temporal validation | Evaluation on a later period. |
| Held-out set | Data not used for training or tuning. |
| Tool lock-in | Cost of migrating away from a tool. |
| Data contract | An agreement on schema, semantics, and quality. |
| Operational data store | A system holding current operational data. |

## 14. Sources and further reading

### Datasets
- M5 Forecasting — Accuracy. https://www.kaggle.com/competitions/m5-forecasting-accuracy
- M5 methods and results. https://github.com/Mcompetitions/M5-methods
- Corporación Favorita. https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting
- Rossmann Store Sales. https://www.kaggle.com/competitions/rossmann-store-sales
- UCI Online Retail II. https://archive.ics.uci.edu/dataset/502/online+retail+ii
- NASA C-MAPSS. https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- UN Comtrade. https://comtradeplus.un.org/
- World Bank LPI. https://lpi.worldbank.org/
- GDELT. https://www.gdeltproject.org/
- Open Supply Hub. https://opensupplyhub.org/

### Benchmarks and methods
- Makridakis, S., Spiliotis, E., Assimakopoulos, V., "The M5 Accuracy competition," 2021. https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf
- TSFM-Bench, KDD 2025. https://arxiv.org/html/2410.11802v5
- "Time Series Foundation Models: Benchmarking Challenges and Requirements," 2025. https://arxiv.org/html/2510.13654v1
- Hyndman, R. J., Athanasopoulos, G., *Forecasting: Principles and Practice*. https://otexts.com/fpp3/

### Tools
- statsforecast. https://github.com/Nixtla/statsforecast
- sktime. https://www.sktime.net/
- darts. https://unit8co.github.io/darts/
- OR-Tools. https://developers.google.com/optimization
- Pyomo. https://www.pyomo.org/
- HiGHS. https://highs.dev/
- SimPy. https://simpy.readthedocs.io/
- DuckDB. https://duckdb.org/
- dbt. https://www.getdbt.com/
- MLflow. https://mlflow.org/
- Evidently. https://www.evidentlyai.com/

> **Next:** [Module 28](28_sources_and_further_reading.md) teaches source evaluation and keeps the library honest.
