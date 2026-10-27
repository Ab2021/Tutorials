# Module 21 — Deployment and MLOps for Planning Systems

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 30 hours (14 lecture, 12 lab, 4 project clinic)
> **Track fit:** Common core; required for all tracks
> **Prerequisites:** Modules 03, 06, 07, 20
> **Tools:** Python, Git, Docker, Airflow or Dagster, dbt, DuckDB or a warehouse, MLflow, Prometheus
> **Industry question:** "Can this forecast or plan run every cycle, be trusted by planners, and be fixed when it breaks?"

---

## 1. Why industry cares

A planning model has a cadence. Demand is forecast weekly, inventory is reviewed daily, routes are planned overnight, and S&OP runs monthly. If the model does not arrive on time, with the right grain, in the system the planner uses, it does not exist.

Deployment is where most analytics value is lost:

- The notebook works on the analyst's laptop but not on the warehouse's data.
- The forecast runs, but nobody knows it is stale.
- The model is accurate, but the planner cannot override it.
- A data feed changes schema and the pipeline fails silently.
- A vendor upgrades the planning suite and the integration breaks.
- An incident happens and no one can reproduce the forecast that caused it.

MLOps for supply chain is not only model serving. It is **the industrial discipline of producing a decision-grade artifact on a schedule, with evidence, controls, and a recovery path**.

## 2. Learning outcomes

By the end of this module, you can:

- Choose a deployment pattern for batch, API, embedded, and streaming planning use cases.
- Design a layered data and model architecture with clear contracts.
- Version code, data, configuration, and model artifacts together.
- Build a CI/CD pipeline with data-quality and backtest gates.
- Deploy a forecast or policy service with shadow and canary stages.
- Instrument freshness, drift, performance, and override metrics.
- Define retraining triggers and a champion–challenger process.
- Write a rollback and incident runbook.
- Estimate run cost and capacity.
- Explain the audit trail required for a planning decision.

## 3. Mental model

```text
  SOURCE ──> INGEST ──> TRANSFORM ──> FEATURES ──> TRAIN ──> REGISTRY
                                                              │
                                                              v
  PLAN ──> DECIDE <── SERVE <── VALIDATE <── PACKAGE <── APPROVE
    │                     │
    v                     v
  MONITOR ──> DETECT ──> RETRAIN / ROLLBACK ──> REVIEW
```

Two loops matter:

- **The build loop:** code change → test → package → validate → approve.
- **The run loop:** schedule → data → score → publish → monitor → respond.

The build loop is fast and infrequent. The run loop is slow, repetitive, and where trust is won or lost.

### 3.1 Batch versus real time

| Pattern | Use | Latency | Complexity |
|---|---|---|---|
| Batch scoring | Weekly demand, daily inventory | Hours | Low |
| Micro-batch | Intraday replenishment | Minutes | Medium |
| API | Planner query, what-if | Milliseconds | Medium |
| Streaming | Event-driven ETA, exceptions | Seconds | High |
| Embedded | Inside ERP/planning suite | Native | Vendor-dependent |

Most supply chain decisions are batch or micro-batch. Do not build streaming infrastructure for a weekly forecast.

## 4. Core concepts

### 4.1 Architecture layers

| Layer | Responsibility | Typical tools |
|---|---|---|
| Source | ERP, WMS, TMS, POS, EDI, telemetry | Connectors, CDC |
| Bronze | Raw, immutable landing | Object storage, warehouse |
| Silver | Conformed, deduplicated, typed | dbt, SQL |
| Gold | Decision-ready marts | dbt, warehouse |
| Feature | Reusable model inputs | Feature store or gold tables |
| Model | Training, evaluation, registry | Python, MLflow |
| Serve | Batch job, API, embedded | Airflow, FastAPI, vendor |
| Observe | Metrics, logs, alerts | Prometheus, Grafana, SIEM |

### 4.2 Data contracts

A data contract states:

- Schema and types.
- Grain and primary key.
- Freshness SLA.
- Null and range rules.
- Ownership and escalation.
- Change process.

Contracts prevent the classic failure: an upstream team renames a column and the forecast silently uses nulls.

### 4.3 Versioning and reproducibility

A forecast artifact should be reproducible from:

| Component | Identifier |
|---|---|
| Code | Git commit SHA |
| Data | Snapshot date or content hash |
| Environment | Container image digest |
| Parameters | Config file or run record |
| Model | Registry version |
| Randomness | Seed |

Store the **run manifest** with the output. Without it, an incident review becomes archaeology.

### 4.4 Testing strategy

| Test | What it catches | Gate |
|---|---|---|
| Unit | Logic errors | Every commit |
| Data quality | Schema, nulls, ranges, duplicates | Every run |
| Backtest | Accuracy regression | Every model change |
| Reconciliation | Hierarchy consistency | Every run |
| Integration | Contract mismatch | Every deploy |
| Shadow | Real-world disagreement | Before promotion |
| Canary | Production impact | Before full rollout |

A backtest gate needs a **pre-registered threshold**. "Do not deploy if WAPE is worse than the incumbent by more than 2%" is a gate; "review the numbers" is not.

### 4.5 CI/CD/CT

- **CI:** lint, unit tests, data tests, build image.
- **CD:** deploy to staging, run integration, promote.
- **CT (continuous training):** retrain on a schedule or trigger, validate, register, and promote through the same gates.

The model is an artifact like software. It should not be copied into production by hand.

### 4.6 Monitoring

| Layer | Metric | Alert condition |
|---|---|---|
| Freshness | Last successful run, max event lag | Older than SLA |
| Volume | Row count vs expected | ±20% |
| Schema | Column set and types | Any change |
| Distribution | PSI, KS, mean shift | Above threshold |
| Accuracy | WAPE, bias, pinball | Above threshold for 2 cycles |
| Stability | Forecast churn | Unexplained jump |
| Decision | Override rate, order change | Shift beyond band |
| Business | Fill rate, inventory, cost | Benefit not realized |
| System | Latency, errors, saturation | SLO breach |

**Delayed labels:** demand actuals may arrive days or weeks late. Use proxy metrics (order intake, shipments, sensor counts) and backfill when actuals land.

### 4.7 Drift

| Drift type | Example | Response |
|---|---|---|
| Data drift | Promotion mix changes | Investigate, add feature |
| Concept drift | Customer behavior changes | Retrain, reweight |
| Label drift | Returns policy changes | Redefine target |
| Upstream drift | New store codes | Fix mapping |
| Schema drift | ERP upgrade | Contract update |

Drift detection is not a retraining trigger by itself. Investigate before automating a response.

### 4.8 Retraining policy

| Policy | When | Risk |
|---|---|---|
| Fixed schedule | Weekly/monthly | Retrains on bad data |
| Triggered by drift | Metric breach | Chasing noise |
| Triggered by performance | Backtest gap | Late |
| Continuous | Every run | Cost and instability |
| Manual | Analyst judgment | Staleness |

Most planning teams should use **scheduled retraining with a performance gate**, plus an incident trigger.

### 4.9 Champion–challenger

Run the incumbent (champion) and a challenger in parallel:

1. Score both on the same data.
2. Compare on pre-registered metrics and slices.
3. Shadow the challenger without affecting decisions.
4. Promote only after a defined period and a human review.
5. Keep the champion artifact for rollback.

### 4.10 Rollback and incident response

Rollback requires:

- The previous artifact version.
- A tested restore command.
- Compatible schemas.
- A decision on what to do with forecasts already published.

An incident runbook names the incident commander, the communication channel, the customer impact, and the rollback authority. Test it before you need it.

### 4.11 Security and secrets

- No credentials in code or notebooks.
- Secrets in a managed vault.
- Least-privilege service accounts.
- Signed images and dependency scanning.
- Audit logs for model and data access.
- Network egress restricted to approved endpoints.

### 4.12 Cost and capacity

| Cost driver | Lever |
|---|---|
| Warehouse compute | Partitioning, incremental models |
| Training compute | Spot instances, smaller models |
| Serving | Batch over API where possible |
| Storage | Retention and compression |
| Observability | Sampling and retention |
| Vendor seats | Usage-based negotiation |

A daily full retrain of thousands of series can cost more than the inventory it saves. Model the run cost as part of the value case (Module 23).

## 5. Data, standards, and workflow

### 5.1 Reference pipeline

```text
06:00  ingest POS, orders, inventory snapshots
06:20  data-quality gate (schema, nulls, volume)
06:40  build features and hierarchy
07:00  score forecast (champion)
07:10  reconcile and publish to planning system
07:20  compute metrics and drift
07:30  publish dashboard and alerts
08:00  planner review and override window
12:00  freeze plan for the cycle
```

### 5.2 Operational SLOs

| SLO | Target | Measurement |
|---|---|---|
| Freshness | Forecast by 07:00 local | Run completion |
| Availability | 99.5% of cycles | Successful runs |
| Accuracy | WAPE within agreed band | Backtest and live |
| Completeness | 100% of active SKU-locations | Row count |
| Latency | API p95 < 500 ms | Histogram |
| Recovery | RTO 4 h, RPO 1 cycle | Drill |

### 5.3 Change control

Every production change records:

- What changed and why.
- Who approved.
- Test evidence.
- Expected impact.
- Rollback plan.
- Post-deployment review date.

## 6. Industry case studies

### 6.1 UPS ORION — deploying optimization at scale

**Context:** UPS needed drivers and dispatchers to adopt optimized routes across a large network.
**Intervention:** ORION recommended routes, with operational rules and driver discretion.
**Evidence:** UPS publicly described multi-year rollout, mileage savings, and explicit change-management work. Treat the savings as company claims.
**Outcome:** A reference for deploying optimization into a human workflow.
**What transferred:** the deployment was as much about dispatcher trust and exception handling as about the algorithm.
**What did not:** the exact savings and timeline depend on network density.

### 6.2 Google's ML Test Score — operational readiness

**Context:** Google researchers proposed a rubric for production ML readiness.
**Intervention:** Tests across data, model, infrastructure, and monitoring.
**Evidence:** Breck et al., "The ML Test Score," *IEEE Big Data*, 2017.
**Outcome:** A widely used checklist for production readiness.
**What transferred:** readiness is measurable before launch, not discovered in production.
**What did not:** the full rubric is heavy for a small planning team; select the tests that match the tier.

### 6.3 Analytics engineering with dbt — contracts and tests

**Context:** Analytics teams needed version-controlled, tested transformations.
**Intervention:** SQL models with tests, documentation, and lineage.
**Evidence:** dbt's public documentation and widespread adoption; vendor claims about productivity should be treated as vendor claims.
**Outcome:** Data transformation became software engineering.
**What transferred:** tests and lineage are prerequisites for trustworthy planning data.
**What did not:** dbt alone does not solve orchestration, model serving, or organizational ownership.

### 6.4 A silent pipeline failure — composite

**Context:** A retailer's daily replenishment forecast ran successfully but used a stale inventory snapshot.
**Intervention:** The pipeline had no freshness check on the snapshot table.
**Evidence:** Planners noticed unusual orders after three days; the root cause was an upstream job that had stopped updating.
**Outcome:** Excess inventory in some stores and stockouts in others.
**What transferred:** a successful job is not a successful outcome; freshness must be tested.
**What did not:** the claim that orchestration tools prevent this; the control was missing, not the tool.

Use this composite as a teaching case, not a citation.

## 7. Labs

### Lab 21.1 — Batch pipeline with orchestration

Build a DAG that ingests, validates, transforms, scores, reconciles, and publishes a forecast. Add a failure branch and an alert.

**Deliverable:** DAG code and a run log.

**Acceptance criteria:** Idempotent; retries safe; data-quality gate blocks bad input; run manifest stored.

### Lab 21.2 — Model registry and promotion

Register two model versions with metrics and artifacts. Promote one through staging to production with a gate.

**Deliverable:** Registry entries and promotion evidence.

**Acceptance criteria:** Versions immutable; metrics recorded; rollback to the prior version demonstrated.

### Lab 21.3 — Shadow and canary

Run a challenger in shadow mode for one cycle. Compare forecasts, overrides, and service impact. Then canary 10% of items.

**Deliverable:** Comparison report and promotion decision.

**Acceptance criteria:** Same data and period; pre-registered metrics; decision rule stated before results.

### Lab 21.4 — Monitoring and incident

Instrument freshness, drift, and accuracy. Inject a stale feed and a schema change. Confirm alerts and write the incident timeline.

**Deliverable:** Dashboard, alert rules, and post-incident review.

### Lab 21.5 — Cost and capacity

Estimate the monthly cost of the pipeline at 10× volume. Identify the top three cost drivers and two optimizations.

**Deliverable:** Cost model and optimization memo.

## 8. Assignment / project

Produce a deployment and monitoring plan for a forecasting or inventory system.

**Required components:**

1. Decision, cadence, and consumers.
2. Architecture and data contracts.
3. Deployment pattern and schedule.
4. Versioning and run manifest.
5. Test and gate strategy.
6. Shadow and canary plan.
7. Monitoring metrics and thresholds.
8. Retraining policy and champion–challenger.
9. Rollback and incident runbook.
10. SLOs, cost estimate, and owner.

**Length:** 8–12 pages plus diagrams.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Architecture | 15% | Notebook to production | Layered design and contracts | Failure isolation and cost awareness |
| Reproducibility | 15% | Manual steps | Versioned artifacts | One-command rebuild and manifest |
| Testing and gates | 20% | Unit tests only | Data and backtest gates | Pre-registered thresholds and shadow |
| Monitoring | 20% | Logs | Freshness, drift, accuracy | Decision and business metrics |
| Recovery | 15% | No plan | Rollback steps | Tested drill and RTO/RPO |
| Operations | 15% | No owner | SLOs and runbook | Cost, capacity, and change control |

## 10. Regulatory, ethical, and commercial lens

- **Audit trail:** regulated decisions need logs of inputs, model version, and output.
- **EU AI Act:** logging, human oversight, and accuracy obligations may apply to high-risk systems.
- **Change control:** financial and safety-relevant models need approval before promotion.
- **Vendor dependence:** managed services can simplify operations but obscure the model and raise exit costs.
- **Labor:** automation changes planner roles; plan the transition.
- **Environmental cost:** compute has a carbon footprint; right-size training and serving.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Notebook handoff | "It works on my machine" | No packaging | Cannot run | Container and CI |
| Silent staleness | Job succeeds, data old | No freshness test | Bad orders | Freshness gate |
| Schema break | Upstream rename | No contract | Nulls or crash | Contracts and tests |
| Manual promotion | Copy to production | No registry | Untraceable version | Registry and gates |
| Metric drift | Accuracy falls slowly | Regime change | Service loss | Monitoring and retrain |
| Override blindness | No override logging | Unknown behavior | Hidden failure | Log and analyze |
| Retrain churn | Daily model changes | No stability gate | Planner distrust | Champion–challenger |
| Cost blowout | Full refresh daily | No incremental design | Budget overrun | Partition and incremental |
| Secret leak | Key in notebook | Bad practice | Breach | Vault and scanning |
| No rollback | "We'll fix forward" | Untested restore | Extended outage | Rollback drill |

## 12. Interview and discussion questions

1. Batch or real time for weekly demand? *(Hint: batch; latency is not the constraint.)*
2. What belongs in a run manifest? *(Hint: code, data, environment, params, seed, model.)*
3. How do you test a forecast pipeline? *(Hint: unit, data, backtest, integration, shadow.)*
4. What is a data contract? *(Hint: schema, grain, freshness, ownership, change.)*
5. How do you detect a stale forecast? *(Hint: freshness metric and event lag.)*
6. When do you retrain? *(Hint: schedule plus performance gate plus incident.)*
7. What is a champion–challenger? *(Hint: parallel scoring and gated promotion.)*
8. How do you roll back a published forecast? *(Hint: prior artifact, republish, communicate.)*
9. How do you monitor with delayed actuals? *(Hint: proxies and backfill.)*
10. What SLO would you set for a planning service? *(Hint: freshness, availability, accuracy.)*
11. How do you control run cost? *(Hint: incremental models, batch, right-sizing.)*
12. What would you log for an audit? *(Hint: inputs, version, output, override, approver.)*

## 13. Key terms

| Term | Definition |
|---|---|
| MLOps | Practices for deploying and operating ML systems |
| Data contract | Agreement on schema, grain, freshness, and ownership |
| Run manifest | Record linking code, data, environment, and output |
| Model registry | Versioned store of models and metadata |
| Feature store | Reusable, consistent model inputs |
| CI/CD | Continuous integration and delivery |
| CT | Continuous training |
| Shadow deployment | Running a model without affecting decisions |
| Canary | Limited production rollout |
| Drift | Change in data or relationships |
| Delayed label | Outcome observed after a lag |
| Champion–challenger | Incumbent versus candidate comparison |
| Rollback | Restoring a prior version |
| SLO | Service-level objective |
| RTO | Recovery time objective |
| RPO | Recovery point objective |
| Idempotent | Re-running produces the same result |
| Lineage | Traceable path from source to output |
| Backtest gate | Deployment block based on accuracy regression |
| Audit trail | Evidence of inputs, decisions, and approvals |

## 14. Sources and further reading

### Standards and regulation

- EU AI Act, Regulation (EU) 2024/1689 (logging and human oversight). https://eur-lex.europa.eu/
- ISO/IEC 42001, AI management systems. https://www.iso.org/standard/81230.html
- NIST AI RMF. https://www.nist.gov/itl/ai-risk-management-framework

### Peer-reviewed and professional

- Sculley, D. et al., "Hidden Technical Debt in Machine Learning Systems," NeurIPS, 2015.
- Breck, E. et al., "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction," IEEE Big Data, 2017.
- Google SRE, *Site Reliability Engineering*. https://sre.google/sre-book/table-of-contents/
- Hulten, G. et al., "Building Intelligent Systems" (operational perspective).
- Kleppmann, M., *Designing Data-Intensive Applications*.

### Industry

- MLflow documentation. https://mlflow.org/docs/
- Apache Airflow documentation. https://airflow.apache.org/docs/
- Dagster documentation. https://docs.dagster.io/
- dbt documentation. https://docs.getdbt.com/
- Feast documentation. https://docs.feast.dev/
- Kubernetes documentation. https://kubernetes.io/docs/
- Prometheus documentation. https://prometheus.io/docs/

> **Next:** [Module 22](22_privacy_security_ethics_responsible_ai.md) adds the security, privacy, and responsible-AI controls that production planning systems require.
