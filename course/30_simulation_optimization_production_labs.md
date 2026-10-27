# Module 30 — Simulation, Optimization, and Real-Time Production Labs

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 28 hours (2 lecture, 24 lab, 2 review)
> **Track fit:** Advanced elective; recommended for Manufacturing and Logistics tracks
> **Prerequisites:** Modules 10–13, 20, 21; Python, Docker, basic Linux
> **Tools:** Python, SimPy, OR-Tools, DuckDB, Kafka, Docker, Kubernetes, Terraform, Prometheus, Grafana, a cloud account or local equivalents
> **Industry question:** "Can you build, test, deploy, and recover a planning system — not only a model?"

---

## 1. Why this module exists

A model in a notebook is a hypothesis. A planning system is a set of engineering and operational choices: how data arrives, how the model is scheduled, how a planner overrides it, how a failure is detected, and how the service is recovered.

This module is a lab manual. It contains ten labs that move from a simulation of a warehouse to a containerized planning service with observability and disaster recovery. Each lab is deliberately small enough to complete in 2–4 hours and realistic enough to expose the failure modes that matter.

The goal is not to make you a platform engineer. The goal is to make you **dangerous enough to design a system that operations can trust** and to have an informed conversation with the engineers who will run it.

## 2. Learning outcomes

By the end of this module, you can:

- Build and validate a discrete-event simulation of a fulfillment or production process.
- Formulate and solve a network design, routing, or inventory problem.
- Build an end-to-end forecasting pipeline with backtesting and artifact retention.
- Ingest and process streaming events with event-time semantics.
- Package a planning service in a container and deploy it to Kubernetes.
- Provision infrastructure with Terraform and explain the state and security model.
- Instrument a service with metrics, logs, and alerts, and define SLOs.
- Simulate and recover from a failure.
- Write a runbook that another engineer can follow at 02:00.

## 3. Lab environment

### 3.1 Baseline stack

| Component | Local option | Cloud option |
|---|---|---|
| Python | 3.11+ virtual environment | Same |
| Data | DuckDB + Parquet | Warehouse |
| Stream | Redpanda or Kafka in Docker | Managed Kafka |
| Container | Docker Desktop | Container registry |
| Orchestration | k3d or kind | Managed Kubernetes |
| Infrastructure | Terraform with local provider | Terraform with cloud provider |
| Metrics | Prometheus + Grafana in Docker | Managed monitoring |
| Object storage | Local filesystem or MinIO | Object storage |

### 3.2 Repository layout

```
labs/
  30_simulation_optimization/
    01_des_warehouse/
      README.md
      model.py
      requirements.txt
    02_network_milp/
    03_vrp/
    04_inventory_sim/
    05_forecast_pipeline/
    06_streaming/
    07_k8s_service/
    08_terraform/
    09_observability/
    10_disaster_recovery/
    docker-compose.yml
```

### 3.3 Rules

- Every lab has a README with objective, inputs, steps, expected output, and acceptance criteria.
- Pin dependencies. Record the Python and tool versions.
- Do not commit secrets. Use environment variables and a local `.env` that is git-ignored.
- Every lab must run from a clean checkout with one command.
- Every lab must state what it does not model.

## 4. Lab 1 — Discrete-event simulation of a fulfillment center

**Time:** 3 hours.
**Objective:** Model receiving, putaway, picking, packing, and shipping, and test a peak scenario.

**Data:** Synthetic order arrivals, SKU mix, process times, staffing, cutoff times.

**Steps:**

1. Define entities: order, tote, picker, packer, dock.
2. Fit arrival and service distributions from sample data. Do not use a normal distribution by default.
3. Build the SimPy process for order release, picking, packing, and dispatch.
4. Add resources: pickers, packers, docks, carts.
5. Add a cutoff constraint: orders released after 14:00 ship next day.
6. Run 30 replications with different seeds.
7. Report throughput, cycle time, utilization, and cutoff misses with confidence intervals.
8. Validate against a simple Little's law calculation.
9. Test a 30% volume increase and a 20% labor reduction.
10. Document what the model omits: travel paths, congestion, breakdowns, returns.

**Deliverable:** `model.py`, a results table, and a one-page memo.

**Acceptance criteria:** Model runs; replications and seeds recorded; confidence intervals reported; validation explained; at least two scenarios compared.

**Stretch:** Add a breakdown process and measure the effect on cutoff misses.

## 5. Lab 2 — Capacitated network design with MILP

**Time:** 3 hours.
**Objective:** Choose which DCs to open and how to flow product to customers under capacity and service constraints.

**Data:** Candidate DCs, fixed and variable costs, customer demand, transport costs, capacity, service distance.

**Steps:**

1. Define sets: plants, candidate DCs, customers, products.
2. Define parameters: demand, capacity, fixed cost, variable cost, distance.
3. Define variables: open DC (binary), flow (continuous).
4. Objective: minimize fixed plus transport cost.
5. Constraints: demand satisfaction, capacity, single-source or split rules, service radius.
6. Solve with OR-Tools or Pyomo + HiGHS.
7. Check solver status and infeasibility.
8. Run a base case and three scenarios: demand +20%, transport +15%, one DC unavailable.
9. Report cost, open DCs, utilization, and service.
10. Compare with the current network.

**Deliverable:** Model, solver log, scenario table, and recommendation memo.

**Acceptance criteria:** Formulation is correct; solver status is optimal or feasible; scenario results are reproducible; recommendation states the trade-off.

**Stretch:** Add carbon cost per tonne-km and plot a cost-carbon frontier.

## 6. Lab 3 — Vehicle routing with time windows

**Time:** 3 hours.
**Objective:** Build a capacitated VRP with time windows and compare it with a greedy route.

**Data:** Depot, 100–300 stops, demand, service time, time windows, vehicle capacity, travel matrix.

**Steps:**

1. Build a travel matrix (Euclidean or OSRM).
2. Implement a greedy nearest-neighbor baseline.
3. Build the OR-Tools routing model with capacity, time windows, and vehicle count.
4. Add a penalty for dropped stops.
5. Solve with a time limit.
6. Report cost, vehicles, distance, on-time rate, and solve time.
7. Test a tighter time window and a 20% demand increase.
8. Inspect the routes for plausibility.
9. Document assumptions: static traffic, no breaks, no driver hours.

**Deliverable:** `route.py`, route map or table, baseline comparison, and memo.

**Acceptance criteria:** Feasible routes; all constraints respected; baseline compared; solve time reported; assumptions listed.

**Stretch:** Add a heterogeneous fleet and a carbon objective.

## 7. Lab 4 — Inventory policy simulation

**Time:** 3 hours.
**Objective:** Compare (s,S), periodic review, and base-stock policies under stochastic demand and lead time.

**Data:** Synthetic or empirical demand and lead-time distributions; holding, ordering, and shortage costs.

**Steps:**

1. Fit demand and lead-time distributions. Check overdispersion.
2. Implement a discrete-event or daily simulation for each policy.
3. Calibrate the policy parameters to a target fill rate.
4. Run 100 replications.
5. Report fill rate, cycle service level, average inventory, ordering frequency, and cost.
6. Compare with the newsvendor or safety-stock formula.
7. Test a lead-time increase of 50%.
8. Test a demand variance increase of 50%.
9. Plot the service-inventory frontier.
10. State which policy is robust and why.

**Deliverable:** Simulation code, frontier plot, and policy recommendation.

**Acceptance criteria:** Distributions documented; replications and seeds; service and inventory reported with uncertainty; formula and simulation compared.

**Stretch:** Add a two-echelon system and test MEIO.

## 8. Lab 5 — End-to-end forecasting pipeline

**Time:** 4 hours.
**Objective:** Build a reproducible pipeline from raw demand data to a backtested forecast and a stored artifact.

**Data:** M5, Favorita, or a company extract.

**Steps:**

1. Ingest raw data into a bronze layer.
2. Clean and conform to a silver layer: grain, calendar, hierarchy, stockout flags.
3. Build features: lags, rolling statistics, promotions, holidays.
4. Implement seasonal naive and LightGBM baselines.
5. Backtest with a rolling origin over at least four folds.
6. Compute WAPE, MASE, bias, and pinball loss.
7. Store forecasts, metrics, and the model artifact with a version.
8. Write a data-quality test suite.
9. Create a run manifest: data hash, code commit, parameters, seed.
10. Rerun from scratch and confirm identical metrics.

**Deliverable:** Pipeline, metrics table, manifest, and reproducibility note.

**Acceptance criteria:** One command runs the pipeline; metrics reproducible; leakage checks documented; artifacts versioned.

**Stretch:** Add hierarchical reconciliation and compare WRMSSE.

## 9. Lab 6 — Streaming shipment events

**Time:** 3 hours.
**Objective:** Ingest shipment events, compute event-time ETA, and emit an exception.

**Data:** Synthetic shipment events: pickup, in-transit, delay, delivery, with event timestamps and ingestion timestamps.

**Steps:**

1. Start Kafka or Redpanda in Docker.
2. Produce events with a Python producer, including late and out-of-order events.
3. Consume with a consumer that maintains event-time state.
4. Compute a simple ETA from planned transit and elapsed time.
5. Detect a delay beyond a threshold.
6. Emit an exception with a reason and priority.
7. Measure latency and late-event handling.
8. Test a burst of 10,000 events.
9. Write a consumer lag alert.
10. Document exactly-once or at-least-once semantics.

**Deliverable:** Producer, consumer, exception output, and latency report.

**Acceptance criteria:** Late events handled; order independent of arrival; lag measured; semantics stated.

**Stretch:** Add a watermark and a windowed on-time rate.

## 10. Lab 7 — Containerized planning service on Kubernetes

**Time:** 4 hours.
**Objective:** Serve a forecast or policy recommendation as an API and deploy it locally.

**Steps:**

1. Wrap the model in a FastAPI service with `/health`, `/forecast`, and `/metrics`.
2. Add input validation and a model version header.
3. Build a multi-stage Docker image.
4. Write a Kubernetes Deployment, Service, ConfigMap, and Secret reference.
5. Set resource requests and limits.
6. Add readiness and liveness probes.
7. Deploy to k3d or kind.
8. Send 1,000 requests and measure p50, p95, and p99 latency.
9. Roll out a new version and roll back.
10. Document the rollback command.

**Deliverable:** Service code, manifests, load-test results, and runbook.

**Acceptance criteria:** Service starts from a clean cluster; probes work; rollback demonstrated; limits set; secrets not baked into the image.

**Stretch:** Add a horizontal pod autoscaler and a canary.

## 11. Lab 8 — Infrastructure as code with Terraform

**Time:** 3 hours.
**Objective:** Provision storage, a database, and a container registry reproducibly.

**Steps:**

1. Define variables for region, environment, and project.
2. Provision an object store bucket and a warehouse or database.
3. Provision a container registry.
4. Add lifecycle rules and encryption.
5. Use a remote state backend or a local state with a lock.
6. Run `terraform plan`, inspect the diff, then `apply`.
7. Change one variable and inspect the plan.
8. Destroy the environment and confirm no orphan resources.
9. Add a policy check for public buckets.
10. Document cost and teardown.

**Deliverable:** Terraform modules, plan output, and cost note.

**Acceptance criteria:** Plan is reviewed before apply; state is managed; teardown is clean; no public data exposure.

**Stretch:** Add a second environment with a workspace.

## 12. Lab 9 — Observability and cost

**Time:** 3 hours.
**Objective:** Instrument the service and define an SLO.

**Steps:**

1. Expose request count, error rate, latency histogram, and forecast freshness.
2. Add structured logs with a request ID.
3. Add a data-quality metric: null rate, row count, last update.
4. Start Prometheus and Grafana.
5. Build a dashboard with four panels: traffic, errors, latency, freshness.
6. Define an SLO: 99% of requests under 500 ms, 99.5% availability.
7. Create alerts for error budget burn and stale data.
8. Simulate a stale-data incident and confirm the alert.
9. Estimate monthly cost from requests, storage, and compute.
10. Write an on-call runbook.

**Deliverable:** Dashboard, alert rules, SLO definition, and cost estimate.

**Acceptance criteria:** Metrics are labeled; alert fires in a simulated incident; SLO is measurable; runbook is actionable.

**Stretch:** Add distributed tracing.

## 13. Lab 10 — Disaster recovery

**Time:** 3 hours.
**Objective:** Recover a planning service and its data after a simulated failure.

**Steps:**

1. Define RTO and RPO for the service and the forecast artifacts.
2. Back up the database and object store.
3. Simulate loss of the service pod, the database, and the object store, one at a time.
4. Restore from backup and measure recovery time.
5. Verify forecast reproducibility after restore.
6. Test a regional failover or a rebuild from code and data.
7. Document the gap between actual and target RTO/RPO.
8. Write a post-incident review.
9. Update the runbook.
10. Schedule a restore drill.

**Deliverable:** DR plan, restore evidence, and post-incident review.

**Acceptance criteria:** RTO/RPO stated; restore demonstrated; data integrity checked; gaps documented.

**Stretch:** Automate the restore and measure it.

## 14. Assessment and sources

### 14.1 Assessment

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Simulation validity | 15% | Model runs | Inputs fitted and validated | Sensitivity and limitations documented |
| Optimization correctness | 15% | Infeasible or wrong objective | Correct formulation and solve | Sensitivity, scenarios, and business interpretation |
| Pipeline reproducibility | 15% | Notebook only | One-command pipeline | Manifest, tests, and identical rerun |
| Streaming semantics | 10% | Ignores event time | Handles late events | Watermarks, lag, and failure semantics |
| Deployment | 15% | Runs locally | Container and probes | Rollback, autoscaling, and resource limits |
| Observability | 10% | Logs only | Metrics and dashboard | SLO, alerts, and cost |
| Disaster recovery | 10% | No plan | Backup and restore | Measured RTO/RPO and drill |
| Runbook quality | 10% | Vague | Step-by-step | Usable at 02:00 by another engineer |

### 14.2 Sources

- SimPy. https://simpy.readthedocs.io/
- OR-Tools. https://developers.google.com/optimization
- Pyomo. https://www.pyomo.org/
- HiGHS. https://highs.dev/
- Kafka documentation. https://kafka.apache.org/documentation/
- Kubernetes documentation. https://kubernetes.io/docs/
- Terraform documentation. https://developer.hashicorp.com/terraform/docs
- Prometheus. https://prometheus.io/docs/
- Grafana. https://grafana.com/docs/
- Google SRE Book. https://sre.google/sre-book/table-of-contents/
- Law, A. M., *Simulation Modeling and Analysis*.
- Bertsimas, D., Sim, M., "The Price of Robustness," *Operations Research*, 2004.

> **Next:** Return to [Module 25](25_capstone_hackathons.md) to apply these labs in the capstone, or [Module 21](21_deployment_mlops_planning_systems.md) for the monitoring and change-control framework.
