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

## 3A. Lab 0 — Environment bootstrap and reproducibility contract

**Time:** 2 hours.
**Objective:** Prove that the lab environment can be rebuilt from a clean checkout before any model is written.

**Steps:**

1. Create the repository from the provided layout. Do not copy a working environment by hand.
2. Pin Python and every dependency in `requirements.txt` or `pyproject.toml`.
3. Add a `Makefile` or task runner with `make setup`, `make data`, `make test`, and `make lab N`.
4. Create a synthetic data generator with a fixed seed for every lab that needs data.
5. Add a `doctor` command that checks Python version, Docker, available memory, and required ports.
6. Run `make setup` in a clean container and record the wall-clock time.
7. Add a smoke test that imports every lab module and fails if a dependency is missing.
8. Write the reproducibility contract: data version, code commit, seed, and expected artifact hash.
9. Document the teardown command and the disk space each lab needs.
10. Commit the lock file and the environment report.

**Deliverable:** Repository, lock file, `doctor` output, and environment report.

**Acceptance criteria:** Clean checkout runs `make setup` without manual steps; smoke test passes; every lab has a pinned environment; teardown is documented.

**Stretch:** Build the environment in CI and publish the image.

## 3B. Data contracts for the labs

Every lab consumes data. A lab that starts from an undocumented CSV teaches the wrong habit. Define a contract before writing model code.

| Lab | Dataset | Grain | Required columns | Contract checks |
|---|---|---|---|---|
| 1 | Synthetic FC events | event | timestamp, order_id, process, resource, duration | No negative durations; arrivals monotone |
| 2 | Network scenario | lane | origin, destination, product, demand, cost, capacity, distance | Demand sums to total; capacities positive |
| 3 | Routing instance | stop | id, x, y, demand, window_start, window_end, service_time | Windows feasible; depot present |
| 4 | Demand and lead time | sku × day | date, sku, demand, lead_time, stockout_flag | No missing dates; flags consistent |
| 5 | Public or company sales | store × sku × day | date, store, sku, sales, price, promotion | Hierarchy complete; calendar aligned |
| 6 | Shipment events | event | shipment_id, event_type, event_time, ingest_time | Event time may precede ingest time |
| 7 | Model inputs | request | features, model_version, request_id | Schema version matches service |
| 8 | Infrastructure config | resource | name, region, size, retention | No public access; tags present |
| 9 | Telemetry | metric sample | timestamp, metric, labels, value | Cardinality bounded; clock synced |
| 10 | Backups | snapshot | dataset, version, checksum, taken_at | Checksum verifies after restore |

For each contract, write three things:

1. **A schema file** (JSON Schema, Pydantic model, or a dbt contract) that the pipeline enforces.
2. **A test suite** that fails loudly on a violation, not silently coerces it.
3. **A version** that changes whenever the meaning of a column changes, even if the name does not.

The synthetic generator is part of the lab, not a convenience. It should have a fixed seed, a documented distribution, and at least one deliberately awkward property — late events, a stockout stretch, or a missing window — so that the student meets the failure before production does.

## 3C. Cost and safety guardrails

Labs run on real accounts, so they need limits.

- Set a budget alert before provisioning anything.
- Use the smallest instance class that completes the lab.
- Destroy infrastructure at the end of every session; do not leave clusters running overnight.
- Tag every resource with owner, lab, and expiry.
- Never use production data; use synthetic or public data.
- Never put credentials in code, notebooks, images, or chat.
- Treat a public bucket as an incident, not a warning.
- Record the cost of each lab in the environment report.

A student who can build a pipeline but cannot estimate its monthly cost has not finished the lab.

## 3D. Local-only path when Docker is unavailable

Some learners cannot run containers. The labs still work, with reduced fidelity, if the substitutions are stated.

| Lab | Full path | Local-only substitute | What is lost |
|---|---|---|---|
| 0 | Docker build | Python virtual environment plus lock file | Image reproducibility |
| 1–4 | Local Python | Same | Nothing material |
| 5 | Warehouse and object store | DuckDB and local Parquet | Warehouse-specific SQL |
| 6 | Kafka | Python queue or a file tailer | Broker semantics and lag |
| 7 | Kubernetes | FastAPI on localhost | Probes, rollout, limits |
| 8 | Terraform cloud | Terraform with a local provider | Real cloud policy |
| 9 | Prometheus and Grafana | Structured logs plus a simple metrics endpoint | Dashboards and alert routing |
| 10 | Backup and restore | File copy and checksum verification | Managed-service recovery |

The local-only path must still produce a manifest, a one-command run, and a teardown note. The point of the lab is the discipline, not the vendor.

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

## 13A. Integration lab — one decision, end to end

**Time:** 6 hours.
**Objective:** Connect at least three labs into one decision-grade workflow and defend the result.

Choose one decision: a peak fulfillment plan, a network redesign, a routing policy, or an inventory policy. Then:

1. State the decision, owner, cadence, and cost of error.
2. Produce the forecast or demand distribution that the decision consumes.
3. Run the optimization or simulation that chooses the action.
4. Test the action under at least two stressed inputs.
5. Instrument the workflow with a freshness check and a failure alert.
6. Package the workflow so another person can rerun it from a clean checkout.
7. Write a two-page decision memo with the baseline, the recommended action, the uncertainty, and the limits.
8. Present a five-minute defense and answer one challenge about validity and one about operations.

**Deliverable:** Integrated repository, decision memo, and defense.

**Acceptance criteria:** The workflow runs end to end; the baseline is explicit; the recommendation follows from the evidence; the limits are stated; a second person can rerun it.

**Stretch:** Add a human-in-the-loop override and measure its effect on the decision.

### 13A.1 Common failure modes across the labs

| Failure | Early warning | Fix |
|---|---|---|
| Simulation without validation | Results look plausible, no comparison | Validate against a known period and Little's law |
| Optimization with the wrong objective | Solver succeeds, business rejects the answer | Rewrite the objective with the decision owner |
| Pipeline that only runs once | Notebook state, manual steps | One-command run and a manifest |
| Streaming that ignores event time | Metrics change with arrival order | Event-time processing and watermarks |
| Deployment without rollback | "It works on my machine" | Versioned image and a tested rollback |
| Observability without an SLO | Many dashboards, no decision | Define the SLO and the error budget |
| Backup without a restore drill | Backup jobs are green | Restore and measure RTO/RPO |

## 13B. The twenty-minute architecture review

Every lab portfolio ends with a review. The reviewer plays the operations lead, not the data scientist. Twenty minutes, six questions, no slides.

1. **What decision does this system change, and who owns it?** If the answer is a dashboard, the system is not a decision system.
2. **What happens when the data is late, missing, or wrong?** The student should describe the failure mode and the visible signal, not only the happy path.
3. **What does a planner see, and what can they override?** An override path that is undocumented is a shadow process.
4. **How do you know the output is still valid?** Look for a freshness metric, a drift check, and an owner.
5. **What does it cost per month, and what would you turn off first?** Cost awareness separates an engineer from a notebook author.
6. **How would you rebuild this from nothing in one day?** The answer is the repository, the manifest, and the runbook.

Score each answer as absent, partial, or complete. Three or more absent answers means the system is a demo. The review is deliberately about operability, because the modeling has already been assessed in the labs.

## 13C. The ten-minute demo

The demo is part of the assessment and follows a fixed script.

1. **Minute 0–2: the decision.** State the decision, the owner, and the cost of being wrong. No architecture yet.
2. **Minute 2–4: the baseline.** Show what happens today and the number the project must beat.
3. **Minute 4–6: the run.** Execute the pipeline from a clean state and show the artifact and metrics.
4. **Minute 6–8: the failure.** Break an input or stop a service and show the alert, the fallback, and the recovery.
5. **Minute 8–10: the limits.** State what the system does not model, what would change the recommendation, and what you would do next.

A demo that shows only the happy path has not shown the system. The failure segment is the one that distinguishes a production lab from a notebook exercise, and it is the segment reviewers remember.

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

### 14.2 Lab-to-module map

| Lab | Reinforces | Skill assessed |
|---|---|---|
| 0 | 03, 30 | Reproducibility and environment discipline |
| 1 | 13, 20 | Discrete-event simulation and validation |
| 2 | 11 | Mixed-integer formulation and scenario analysis |
| 3 | 12 | Routing with real constraints |
| 4 | 10 | Inventory policy under uncertainty |
| 5 | 03, 07, 08 | Pipeline engineering and backtesting |
| 6 | 19 | Event-time processing and exception logic |
| 7 | 18, 26 | Service packaging and model versioning |
| 8 | — | Infrastructure as code and cost control |
| 9 | 26 | SLOs, monitoring, and data freshness |
| 10 | 26 | Recovery, RTO/RPO, and post-incident review |
| Integration | 21–24, 28 | End-to-end decision ownership |

### 14.3 Portfolio submission checklist

Submit one repository, not ten notebooks. The repository passes when:

- [ ] `make setup` works from a clean clone without manual steps.
- [ ] Every lab has a README with objective, steps, and acceptance criteria.
- [ ] Every lab runs with one command and a fixed seed.
- [ ] A manifest records data hash, code commit, parameters, and seed.
- [ ] Test suites cover schema, data quality, and at least one behavior.
- [ ] No secrets, credentials, or production data are committed.
- [ ] Each lab states what it does not model.
- [ ] The integration lab has a two-page decision memo.
- [ ] Cost estimates and teardown commands are present.
- [ ] A second person has rerun at least two labs successfully.

### 14.4 What distinguishes an expert portfolio

The difference between competent and expert is not model sophistication. It is whether the system can be trusted and operated:

- The baseline is explicit, and the improvement over it is measured.
- Failure is designed for: stale data, late events, missing inputs, and a rollback path.
- The decision owner is named, and the override is documented.
- The cost is known, and the cheapest way to run it is known.
- The limits are stated before the reviewer asks.
- Another engineer can rebuild it from the repository in a day.

A portfolio that demonstrates these six things is worth more than one with a larger model and no runbook.

### 14.5 Sources

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
- Humble, J., Farley, D., *Continuous Delivery*.
- Beyer, B. et al., *Site Reliability Engineering*.
- Kleppmann, M., *Designing Data-Intensive Applications*.
- Nygard, M., *Release It!*

> **Portfolio note:** A complete portfolio is a small production system, not a collection of exercises. Keep the repository, the manifest, and the runbook together; they are the evidence that the work can survive contact with operations.

> **Next:** Return to [Module 28](28_domain_case_library.md) to apply these labs to a domain project, or [Module 26](26_quality_evaluation_model_risk.md) for the monitoring and change-control framework.
