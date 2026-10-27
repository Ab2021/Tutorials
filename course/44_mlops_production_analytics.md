# Module 44 — MLOps and Production Analytics Engineering

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (10 lecture, 12 lab, 4 project)
> **Track fit:** Advanced elective; Data Science and Platform tracks
> **Prerequisites:** Modules 06, 07, 08, 19
> **Tools:** Python, Git, MLflow, DVC, Docker, Airflow or Prefect, Kubernetes, Prometheus
> **Industry question:** "The notebook has an accuracy of 0.91 — why is the planner still using a spreadsheet?"

---

## 1. Why industry cares

Most supply chain models do not fail because the algorithm is wrong. They fail because:

- The features used in production differ from the features used in training.
- The model silently degrades when demand patterns shift.
- Nobody knows which model version produced last week's forecast.
- A retrain breaks a downstream planning system and nobody notices for a month.
- The model cannot be reproduced, audited, or explained to a regulator.

**MLOps** is the discipline that keeps analytics running: versioned data, tracked experiments, automated pipelines, governed deployment, and continuous monitoring. It is the difference between a demonstration and a decision system.

This module is deliberately standalone. Other modules teach models; this one teaches the engineering and governance that make a model survive contact with operations.

## 2. Learning outcomes

By the end of this module, you can:

- Describe the ML lifecycle from problem framing to retirement.
- Version data, code, configuration, and models reproducibly.
- Track experiments and compare runs on a common basis.
- Build and maintain a feature pipeline with point-in-time correctness.
- Orchestrate batch and streaming pipelines with dependencies and retries.
- Package, serve, and roll out models: batch, real-time, and shadow.
- Design CI/CD/CT pipelines for analytics.
- Monitor data drift, concept drift, and performance with delayed labels.
- Choose retraining policies and champion–challenger designs.
- Govern models under frameworks such as SR 11-7 and internal model risk policy.
- Decide build versus buy for the analytics platform.
- Estimate the cost and reliability of a production analytics system.

## 3. Mental model

```text
   DATA ──> FEATURES ──> TRAIN ──> VALIDATE ──> REGISTER ──> DEPLOY ──> MONITOR
     │          │           │          │            │           │          │
   version   point-in-   experiment  backtest    versioned   canary    drift and
   and       time join   tracking    and gates   artefact    shadow    accuracy
   lineage                                                                 │
     └──────────────────── feedback and retraining ─────────────────────────┘

   Every arrow is automated. Every artefact is versioned. Every change is reversible.
```

The lifecycle is a **loop**, not a project. The loop's speed and safety determine how much value the analytics actually delivers.

### 3.1 The four planes

| Plane | Purpose | Typical tooling |
|---|---|---|
| Data | Versioned, quality-checked inputs | Lakehouse, DVC, dbt |
| Feature | Reusable, point-in-time features | Feature store |
| Model | Train, evaluate, register | MLflow, model registry |
| Operations | Serve, monitor, retrain | Airflow, Kubernetes, Prometheus |

## 4. Core concepts

### 4.1 The ML lifecycle

| Stage | Question | Artefact |
|---|---|---|
| Framing | What decision does this change? | Problem statement, success metric |
| Data | Is the data fit for purpose? | Dataset and data card |
| Features | What signals are stable and available? | Feature definitions |
| Training | What model, with what hyperparameters? | Trained model, metrics |
| Validation | Does it beat the baseline out of sample? | Backtest report |
| Deployment | How does it reach the decision? | Service or batch job |
| Monitoring | Is it still good? | Dashboards, alerts |
| Retirement | When do we stop using it? | Decommission record |

A model without a decision is a hobby. The first question is always which decision changes and who acts on it.

### 4.2 Versioning everything

A model is the product of four inputs:

\[
\text{model} = f(\text{code}, \text{data}, \text{config}, \text{environment})
\]

If any of the four is unversioned, the result is not reproducible. Minimum practice:

- **Code:** Git, with the commit hash recorded in the model artefact.
- **Data:** immutable snapshots or versioned tables (DVC, Delta, Iceberg).
- **Config:** hyperparameters and feature lists in version control, not in notebooks.
- **Environment:** pinned dependencies and a container image digest.

Reproducibility is not a research nicety. It is what lets you explain, to an auditor or a customer, why a forecast was what it was.

### 4.3 Experiment tracking

Each run should record:

- Dataset version and row count.
- Feature set version.
- Hyperparameters.
- Metrics, by segment and not only aggregate.
- Artefacts: model file, plots, confusion data.
- Environment and code commit.

Comparing runs requires a **common evaluation protocol** — same time split, same horizon, same metric definition. Most "model improvement" claims are the result of a changed evaluation, not a better model.

### 4.4 Point-in-time correctness and leakage

Feature leakage is the most common cause of a model that looks excellent and fails in production.

Examples in supply chain:

- Using the final inventory position to predict a shortage that was already known.
- Using a corrected demand figure that was not available at forecast time.
- Joining a supplier's current credit rating to a historical order.
- Aggregating over a period that includes the outcome.

The fix is a **point-in-time join**: for each prediction timestamp, only data with an effective timestamp at or before it may be used. This must be enforced in the pipeline, not by convention.

### 4.5 Orchestration

A pipeline is a directed acyclic graph of tasks:

```text
   extract ──> validate ──> build features ──> train ──> evaluate ──> register
                   │                              │
                   └──> quality gate (fail)       └──> backtest gate (fail)
```

Requirements:

- **Idempotency:** re-running produces the same result.
- **Retries and alerting:** failures are visible, not silent.
- **Dependencies:** tasks run in order, with explicit data contracts.
- **Backfills:** history can be rebuilt.
- **Lineage:** every output traces to its inputs.

A cron job with no monitoring is not orchestration.

### 4.6 Serving patterns

| Pattern | Latency | Use | Trade-off |
|---|---|---|---|
| Batch scoring | Hours | Weekly forecast, replenishment | Simple, stale between runs |
| Micro-batch | Minutes | Near-real-time reorder | Moderate complexity |
| Real-time API | Milliseconds | Pricing, routing, fraud | Operational burden |
| Streaming | Continuous | Telematics, anomaly detection | Highest complexity |
| Shadow | None | Safe evaluation | Extra compute |
| Canary | Partial traffic | Gradual rollout | Needs traffic split |

Most supply chain decisions are **batch or micro-batch**. Real-time serving is justified only when the decision is real-time and the value of latency exceeds the operational cost.

### 4.7 CI/CD/CT

- **CI (continuous integration):** code tests, data contract tests, model unit tests.
- **CD (continuous delivery):** automated packaging and deployment, gated by approval.
- **CT (continuous training):** automated retraining triggered by schedule, drift, or performance.

A deployment gate should check:

- Backtest improvement over champion, by segment.
- Data quality and completeness.
- Bias and stability checks.
- Latency and cost within budget.
- Rollback path exists and is tested.

### 4.8 Monitoring

Three layers:

1. **Operational:** pipeline success, latency, error rate, cost.
2. **Data:** schema, nulls, ranges, distribution drift.
3. **Model:** accuracy where labels exist, proxy metrics where they do not.

Population Stability Index for a feature:

\[
\text{PSI} = \sum_i (a_i - e_i)\ln(a_i/e_i)
\]

where a is actual share and e is expected share in bucket i.

Expected shares of 40/40/20 and actual shares of 25/50/25:

\[
\text{PSI} = (0.25-0.40)\ln\frac{0.25}{0.40} + (0.50-0.40)\ln\frac{0.50}{0.40} + (0.25-0.20)\ln\frac{0.25}{0.20}
\]

\[
= 0.0705 + 0.0223 + 0.0112 = 0.104
\]

Common thresholds: below 0.1 stable, 0.1–0.25 moderate shift, above 0.25 significant. A shift is a prompt to investigate, not proof of degradation.

### 4.9 Delayed labels

Forecast accuracy is only known after the period ends. A weekly forecast evaluated with a four-week lag means a problem can run for a month before it is visible.

Mitigations:

- Track **proxy metrics** available immediately: bias, forecast dispersion, input completeness.
- Monitor **leading indicators**: order pattern changes, promotion calendars, supplier lead-time shifts.
- Use **short-horizon** accuracy as an early signal.
- Set alert thresholds on proxies calibrated against historical failures.

### 4.10 Retraining policy

Options:

| Policy | Trigger | Advantage | Risk |
|---|---|---|---|
| Fixed schedule | Weekly or monthly | Predictable | Wastes effort when stable |
| Drift-triggered | PSI or error threshold | Responsive | Noisy triggers |
| Performance-triggered | Backtest degradation | Directly relevant | Delayed labels |
| Continuous | Every new data point | Current | Cost and instability |
| Manual | Analyst request | Controlled | Slow and political |

Most teams should use a **scheduled baseline plus a drift or performance trigger**, with a documented rollback.

### 4.11 Champion–challenger and shadow deployment

- **Champion:** the model in use.
- **Challenger:** a candidate run in parallel.
- **Shadow:** challenger scores without affecting decisions, so predictions can be compared against outcomes.

This is the safest way to test a new model on live data. It also produces the evidence needed for governance approval.

### 4.12 Model risk governance

Regulated sectors expect model risk management:

- **Inventory:** every model is registered with an owner and purpose.
- **Documentation:** data, assumptions, methodology, limitations.
- **Validation:** independent review and ongoing monitoring.
- **Change control:** material changes are approved and recorded.
- **Use test:** evidence the model is used and understood.

SR 11-7 (US Federal Reserve) is the canonical example; equivalent expectations exist in banking, insurance, and increasingly in any regulated supply chain. Even outside regulation, the documentation is what makes a model defensible.

### 4.13 Build versus buy

| Factor | Build | Buy |
|---|---|---|
| Differentiation | High | Low |
| Data sensitivity | Internal | Vendor acceptable |
| Time to value | Slow | Fast |
| Total cost | Higher staff, lower licence | Licence and lock-in |
| Control | Full | Limited |
| Talent | Required | Vendor-supplied |

The common failure is building a platform for a commodity need, or buying a platform for a differentiating one.

### 4.14 Worked example — retraining economics

A demand model retrains monthly. Each retrain costs €600 in compute and 6 engineer-hours at €90, so €1,140 per month, or €13,680 per year.

The model's error rises by about 0.4 percentage points of MAPE per month without retraining. On €50 million of revenue with a 25% margin, a 1-point MAPE improvement is worth roughly €125,000 in reduced mismatch cost (a deliberately simple assumption). A 0.4-point monthly degradation is worth about €50,000 per month, or €600,000 per year.

The retraining cost is trivial relative to the degradation. But the calculation also shows the opposite: if the model is stable, monthly retraining may be unnecessary. The decision is empirical, not habitual.

### 4.15 Worked example — monitoring coverage

A team runs 14 models. Monitoring is defined as: pipeline health, data quality, drift, and performance.

| Layer | Models covered | Coverage |
|---|---:|---:|
| Pipeline health | 14 | 100% |
| Data quality | 11 | 79% |
| Drift | 8 | 57% |
| Performance | 5 | 36% |

The aggregate "we monitor our models" claim is false for two-thirds of the portfolio at the performance layer. Reporting coverage by layer is more honest than reporting a single number, and it points directly to the next investment.

## 5. Data, standards, and workflow

### 5.1 Data and system sources

| Data | Source | Use |
|---|---|---|
| Raw events | ERP, WMS, TMS, POS | Training data |
| Master data | MDM | Entity consistency |
| Feature tables | Warehouse / feature store | Model inputs |
| Experiment runs | Tracking server | Comparison and audit |
| Model registry | Registry | Deployment and lineage |
| Production logs | Service | Monitoring |
| Ground truth | Actuals, reconciled | Performance |
| Incidents | Case management | Root cause |

### 5.2 Standards and frameworks

- **CRISP-DM** for process structure (still useful as a shared vocabulary).
- **SR 11-7** for model risk management.
- **ISO/IEC 42001** for AI management systems.
- **NIST AI Risk Management Framework.**
- **EU AI Act** where applicable, including documentation and logging duties.
- **Data contracts and schema registries** for interface stability.
- **OpenLineage / OpenMetadata** for lineage.
- **SLSA** and signed artefacts for build integrity.

### 5.3 Workflow

1. Frame the decision and success metric.
2. Version and validate the data.
3. Build point-in-time features.
4. Train and track experiments.
5. Validate out of sample and by segment.
6. Register and approve the model.
7. Deploy with a rollback path.
8. Monitor operations, data, and performance.
9. Retrain or retire on evidence.

## 6. Industry case studies

### 6.1 Forecast platform consolidation

**Context:** A retailer ran dozens of spreadsheet and script-based forecasts.
**Intervention:** A central forecasting service with versioned data and a model registry.
**Evidence:** Industry and vendor case reporting (grade C/D).
**Outcome:** Forecast accuracy improved mainly through data consistency and monitoring, not model sophistication.
**What transferred:** most of the gain comes from discipline, not algorithms.
**What did not:** specific accuracy gains.

### 6.2 Model degradation after a demand shock

**Context:** A demand model trained on pre-shock data degraded sharply.
**Intervention:** Drift monitoring and rapid retraining.
**Evidence:** Industry reporting and academic work on concept drift (grade B/C).
**Outcome:** Teams with monitoring detected the shift within weeks; teams without it detected it through stockouts.
**What transferred:** monitoring is the difference between a controlled degradation and an operational surprise.
**What did not:** specific error magnitudes.

### 6.3 Regulatory model risk review

**Context:** A financial institution's supply chain finance model was reviewed under model risk policy.
**Intervention:** Independent validation and documentation.
**Evidence:** Regulatory guidance and supervisory findings (grade B).
**Outcome:** The model was approved with conditions on monitoring and documentation.
**What transferred:** governance is a design requirement, not paperwork added at the end.
**What did not:** the specific institution.

### 6.4 A composite failure — the feature that leaked

**Context:** A churn model for a distributor used a field updated after cancellation.
**Intervention:** No point-in-time join.
**Evidence:** Composite teaching case.
**Outcome:** Offline accuracy was excellent; production performance collapsed.
**What transferred:** leakage is a pipeline bug, not a modelling choice.
**What did not:** specific metrics.

## 7. Labs

### 7.1 Lab 44.1 — Reproducible pipeline

Build a training pipeline with versioned data and code. Re-run it from a clean environment and reproduce the same metric.

**Deliverable:** Pipeline and reproduction log.

### 7.2 Lab 44.2 — Experiment tracking

Run five model variants with tracked parameters and metrics. Compare by segment and produce a recommendation.

**Deliverable:** Tracking report and comparison.

### 7.3 Lab 44.3 — Point-in-time features

Build a feature table for a demand model with effective dating. Demonstrate and fix a leakage bug.

**Deliverable:** Feature pipeline and leakage test.

### 7.4 Lab 44.4 — Serving and rollout

Package a model as a batch job and an API. Deploy as shadow, then canary, with a rollback.

**Deliverable:** Deployment and rollout log.

### 7.5 Lab 44.5 — Monitoring and drift

Implement PSI monitoring on three features and a delayed-label accuracy monitor. Define alert thresholds and a response playbook.

**Deliverable:** Monitoring dashboard and playbook.

### 7.6 Lab 44.6 — Model governance pack

Write a model risk document for one of your models: purpose, data, methodology, validation, limitations, monitoring, and change control.

**Deliverable:** Governance pack.

## 8. Assignment / project

Take an analytics model from notebook to production.

**Required components:**

1. Problem framing and decision link.
2. Data versioning and quality checks.
3. Feature pipeline with point-in-time correctness.
4. Experiment tracking and comparison.
5. Validation protocol and backtest.
6. Registry entry and deployment plan.
7. Monitoring and alerting.
8. Retraining and rollback policy.
9. Cost and reliability estimate.
10. Governance documentation.
11. Risks, assumptions, and limits.

**Length:** 12–16 pages plus repository.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Reproducibility | 20% | Notebook reruns | Versioned code and data | Full lineage and rebuild |
| Features | 20% | Ad hoc joins | Point-in-time table | Leakage tests and contracts |
| Deployment | 15% | Manual copy | Automated pipeline | Canary with rollback |
| Monitoring | 20% | Logs only | Drift and accuracy | Proxies and calibrated alerts |
| Governance | 15% | README | Model card | Audit-ready pack |
| Economics | 10% | No cost view | Cost estimate | Value and reliability trade-off |

## 10. Regulatory, ethical, and commercial lens

- **Model risk:** regulated firms must validate and monitor models.
- **AI regulation:** the EU AI Act and similar regimes impose documentation, logging, and human oversight.
- **Privacy:** training data may contain personal data; retention and minimisation apply.
- **Fairness:** allocation models can discriminate; test by protected group where lawful to do so.
- **Transparency:** planners need to understand and challenge model output.
- **Security:** the model pipeline is an attack surface; see Module 43.
- **Vendor lock-in:** platform choices have long commercial consequences.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Notebook in production | Manual runs | No pipeline | Silent failure | Orchestrated pipeline |
| Leakage | Offline ≫ online | Future data used | Wrong decisions | Point-in-time joins |
| No versioning | "Which model?" | Untracked artefacts | Irreproducible | Registry and hashes |
| Silent pipeline failure | Stale output | No alerting | Stale decisions | Health monitoring |
| Drift ignored | Accuracy falls | Distribution shift | Cost and stockouts | Drift monitoring |
| Retrain breaks downstream | Schema change | Contract breach | Outage | Data contracts and canary |
| Metric gaming | Aggregate only | Segments ignored | Hidden harm | Segment metrics |
| Governance afterthought | No documentation | Retrofit | Audit failure | Documentation as code |

## 12. Interview and discussion questions

1. What is MLOps, and why is it separate from data science? *(Hint: running and governing models.)*
2. What four things must be versioned for reproducibility? *(Hint: code, data, config, environment.)*
3. What is feature leakage, and how do you prevent it? *(Hint: point-in-time joins.)*
4. How do you compare two model runs fairly? *(Hint: same split, horizon, metric.)*
5. When is real-time serving justified? *(Hint: only when the decision is real-time.)*
6. What is a shadow deployment? *(Hint: score without acting.)*
7. How do you monitor a model with delayed labels? *(Hint: proxies and leading indicators.)*
8. What is PSI? *(Hint: distribution shift measure.)*
9. What triggers a retrain in your design? *(Hint: schedule, drift, performance.)*
10. What belongs in a model risk document? *(Hint: purpose, data, method, validation, limits, monitoring.)*
11. Build or buy the feature store? *(Hint: differentiation and sensitivity.)*
12. How do you roll back a bad model? *(Hint: versioned artefacts and tested path.)*

## 13. Key terms

| Term | Definition |
|---|---|
| MLOps | Practices for deploying and operating ML |
| Model registry | Catalogue of model versions and metadata |
| Feature store | Centralised, reusable feature definitions |
| Point-in-time join | Using only data available at prediction time |
| Leakage | Using future or outcome information |
| Experiment tracking | Recording runs, parameters, and metrics |
| CI/CD/CT | Integration, delivery, and continuous training |
| Shadow deployment | Scoring without affecting decisions |
| Canary release | Gradual traffic shift to a new version |
| Drift | Change in data or relationships over time |
| PSI | Population stability index |
| Delayed label | Outcome known only after a lag |
| Model card | Structured model documentation |
| SR 11-7 | US supervisory guidance on model risk |
| Data contract | Enforced schema and quality agreement |
| Lineage | Traceability of data and artefacts |

## 14. Sources and further reading

### Standards and regulation

- Federal Reserve, SR 11-7 model risk management guidance. https://www.federalreserve.gov/
- NIST AI Risk Management Framework. https://www.nist.gov/itl/ai-risk-management-framework
- ISO/IEC 42001 AI management systems. https://www.iso.org/
- EU AI Act. https://eur-lex.europa.eu/
- OpenLineage. https://openlineage.io/
- SLSA. https://slsa.dev/

### Reports and official sources

- Google, *Rules of Machine Learning* and MLOps guidance. https://developers.google.com/machine-learning/guides/rules-of-ml
- Microsoft, MLOps maturity model documentation.
- MLops.community and industry surveys.

### Peer-reviewed and professional

- Sculley, D., et al., "Hidden Technical Debt in Machine Learning Systems", NeurIPS 2015.
- Breck, E., et al., "The ML Test Score", 2017.
- Gama, J., et al., concept drift adaptation survey.
- CRISP-DM process documentation.

### Industry

- MLflow documentation. https://mlflow.org/
- Apache Airflow and Prefect documentation. https://airflow.apache.org/ https://www.prefect.io/
- DVC documentation. https://dvc.org/
- Evidently AI, drift monitoring guidance. https://www.evidentlyai.com/

> **Next:** [Module 45](45_plant_layout_detailed_engineering.md) closes the elective set with the physical design of facilities and the engineering interfaces that constrain them.
