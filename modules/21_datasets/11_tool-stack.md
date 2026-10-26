# Tool Stack

> Module 21 — Datasets, Tools, and Benchmarks
> Topic 11 of 12
> Estimated study time: 3 hours
> Prerequisites: Topic 01, Topic 02
> Version 2026.3

## 1. Why this topic exists

A tool stack is an argument about how work will be done, reproduced, and maintained. Most stack choices are made by familiarity, fashion, or the last conference talk. The expensive failures are not "the wrong library"; they are stacks that cannot be reproduced, cannot be operated by the team that owns them, or cannot be licensed for the product.

The failures are specific:

- A pipeline works on a laptop and fails in production because the environment was never pinned.
- A team adopts a distributed engine for a dataset that fits in memory, and spends more time on operations than on analysis.
- A copyleft dependency is embedded in a distributed product, and legal review stops the release.
- An orchestration tool is chosen for its UI, and the team cannot debug a failed run at 2 a.m.
- A model is trained in one framework and cannot be served without a rewrite.
- A data quality tool reports thousands of warnings that nobody triages.
- A governance tool is deployed, and the audit trail does not cover the actual processing.

This topic is a judgement guide, not a catalogue. For each function it states what a tool is good for, its licence class, and its characteristic failure. It also states when a simpler tool wins, because the most common stack error is over-engineering a problem that a single-node tool solves.

The stack is also a governance artefact. The licence, the supply chain, the version pin, and the data flow determine whether the work can be audited and deployed. A stack review is therefore a risk review.

> **Note:** Licences and project governance change. The licence class in this topic is a starting point for legal review, not a legal opinion. Verify the current licence and the current maintainer before adopting a tool in a regulated product.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** a function to at least two candidate tools and state the trade-off. Assessed by Exercise 1 and self-check item 1.
2. **Score** a stack on fit, reproducibility, licence, team skill, cost, and operations. Assessed by Exercise 2.
3. **Distinguish** permissive, copyleft, and proprietary licences and their distribution consequences. Assessed by Exercise 3.
4. **Measure** setup time, runtime, memory, and lines of code in a bake-off. Assessed by Exercise 4.
5. **Identify** the characteristic failure mode of each tool in a proposed stack. Assessed by Exercise 5.
6. **Decide** when a simpler tool wins over a distributed or managed alternative. Assessed by Exercise 6.
7. **Design** a reproducible environment with pinned versions, containers, and data versioning. Assessed by Exercise 7.
8. **Write** a stack decision record with the rejected options and the switching cost. Assessed by Exercise 8.

## 3. The core mental model

A stack is a chain from data to decision. Each layer has a job, a failure mode, and an operating cost.

```text
   SOURCE
   files | database | API | stream
        |
        v
   STORAGE
   object store | warehouse | lakehouse | local
        |
        v
   TRANSFORMATION
   SQL | dataframes | dbt | notebooks
        |
        v
   ORCHESTRATION
   scheduler | workflow | manual
        |
        v
   MODELLING
   classical | deep | foundation
        |
        v
   EVALUATION AND GOVERNANCE
   metrics | fairness | privacy | lineage
        |
        v
   SERVING
   batch | API | edge | embedded
```

**The fit rule.** The tool must fit the data size, the latency, the team, and the governance. A tool that fits the data and not the team is a maintenance liability.

**The reproducibility rule.** A result is reproducible if another person can rerun it with the same code, environment, data version, and parameters. The stack must make that possible, not merely possible in principle.

**The licence rule.** A permissive licence permits commercial use and distribution with attribution. A copyleft licence requires derivative distributions to use the same licence. A proprietary licence is governed by a contract. The distribution model determines which matters.

**The operations rule.** Every tool has an operating cost: upgrades, monitoring, on-call, and debugging. A tool that is powerful and unoperable by the team is a net loss.

**The exit rule.** A stack choice is reversible at a cost. The decision record should state the switching cost and the trigger for switching.

| Layer | Job | Common over-engineering | Simpler alternative |
|---|---|---|---|
| Storage | Hold data durably | Distributed lakehouse for 50 GB | Parquet on object storage |
| Transformation | Clean and join | Spark for a single machine | DuckDB or Polars |
| Orchestration | Schedule and retry | A cluster scheduler for three jobs | Cron or a workflow file |
| Modelling | Fit and evaluate | A custom distributed trainer | scikit-learn or XGBoost |
| Serving | Answer requests | Kubernetes for one model | A container and a queue |
| Governance | Track and audit | A platform for one team | Git, hashes, and a data card |

## 4. Definitions and vocabulary

### 4.1 Stack and operations terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Stack | The set of tools used end to end. | Treated as one product | Set |
| Layer | A function in the stack. | Confused with a vendor | Function |
| Bake-off | A controlled comparison of two stacks. | A demo | Experiment |
| Switching cost | The cost of replacing a tool. | Ignored | Person-weeks |
| Lock-in | Dependence on a vendor or format. | Treated as unavoidable | Risk |
| Supply chain | The dependencies and their sources. | Ignored until an incident | Risk |
| SBOM | Software bill of materials. | Treated as a security guarantee | Document |
| Pinning | Fixing a version or digest. | Ranges in production | Version |
| Reproducible environment | An environment that can be recreated. | A requirements file with ranges | Container |
| Operational burden | The work to run and maintain. | Underestimated | Hours per month |

### 4.2 Data and transformation terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Parquet | Columnar file format. | Treated as a database | Format |
| DuckDB | Embedded analytical database. | Treated as a server | Engine |
| PostgreSQL | Relational database server. | Used for analytics at scale | Engine |
| dbt | SQL transformation framework. | Treated as an orchestrator | Framework |
| Airflow | Workflow orchestrator. | Used for streaming | Orchestrator |
| Dagster | Data orchestrator with assets. | Treated as a scheduler only | Orchestrator |
| Prefect | Workflow orchestration. | Treated as a data catalog | Orchestrator |
| Spark | Distributed data processing. | Used for small data | Engine |
| Polars | Single-node dataframe library. | Treated as distributed | Library |
| pandas | Single-node dataframe library. | Used for out-of-memory data | Library |

### 4.3 Modelling and serving terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| scikit-learn | Classical machine learning library. | Used for deep learning | Library |
| XGBoost | Gradient boosting library. | Treated as a complete pipeline | Library |
| PyTorch | Deep learning framework. | Treated as a serving system | Framework |
| MONAI | Medical imaging deep learning toolkit. | Treated as a clinical validator | Toolkit |
| ONNX | Model interchange format. | Assumed to preserve performance | Format |
| Triton | Inference server. | Used for batch-only jobs | Server |
| BentoML | Model serving framework. | Treated as an orchestrator | Framework |
| vLLM | LLM inference engine. | Treated as a model | Engine |
| FastAPI | Python API framework. | Used for long batch jobs | Framework |

### 4.4 Failure and reliability terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Failure mode | The way a tool fails. | Discovered in production | Category |
| Blast radius | The scope of a failure. | Ignored | Scope |
| Retry | Re-running a failed job. | Without idempotency | Policy |
| Idempotency | A rerun produces the same result. | Assumed | Property |
| Backpressure | Slowing input when downstream is saturated. | Ignored | Control |
| Circuit breaker | Stopping calls to a failing dependency. | Absent | Pattern |
| Dead letter queue | Storage for failed messages. | Messages lost | Queue |
| Time to recovery | Time to restore service. | Not measured | Duration |
| Runbook | The procedure for a failure. | Missing | Document |
| On-call | The person responsible for incidents. | Undefined | Rota |

### 4.5 Governance terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| MLflow | Experiment tracking and model registry. | Treated as a data catalog | Platform |
| DVC | Data version control. | Treated as a backup | Tool |
| Great Expectations | Data quality validation. | Treated as a monitoring system | Library |
| Evidently | Model and data drift monitoring. | Treated as harm detection | Library |
| Fairlearn | Fairness assessment. | Treated as a fairness guarantee | Library |
| OpenDP | Differential privacy library. | Treated as anonymisation | Library |
| Presidio | PII detection and redaction. | Treated as clinical de-identification | Library |
| OpenLineage | Lineage metadata standard. | Treated as a catalog | Standard |
| SemVer | Semantic versioning. | Ignored in pinning | Convention |
| SPDX | Licence and SBOM standard. | Treated as legal review | Standard |

## 5. The mechanics, step by step

The procedure has 18 steps in four phases: define the job and constraints, compare candidates, prototype and measure, and decide and document.

### Phase A — Define the job and constraints

**Step 1. Write the job to be done.**
Input: the decision and the data.
Output: a one-sentence job statement with the input, the output, the latency, and the scale.
Rationale: the job determines the layer and the tool.
Branch: if the job is a one-off analysis, a notebook and a file may be enough.
Edge case: a one-off can become a recurring pipeline; design for the likely path, not the imagined one.

**Step 2. State the scale and latency.**
Input: the data volume, the growth rate, and the decision cadence.
Output: gigabytes, rows, requests per second, and the latency budget.
Rationale: scale determines whether a single-node tool is sufficient.
Branch: if the data fit in memory on one machine, prefer a single-node tool.
Edge case: "big data" is often a few gigabytes with a large narrative.

**Step 3. State the team and the operating model.**
Input: the skills, the on-call model, and the budget.
Output: a team profile and an operations constraint.
Rationale: a tool the team cannot operate is a liability.
Branch: if the team has no distributed-systems experience, a managed service or a single node is safer.
Edge case: a managed service moves the operational burden to the vendor and adds lock-in.

**Step 4. State the governance constraints.**
Input: the data classification, the licence, the residency, and the audit requirements.
Output: a constraint list.
Rationale: governance can eliminate a tool before the comparison.
Branch: if the data cannot leave a region, a managed service in another region is out.
Edge case: a tool's telemetry can violate a residency requirement.

### Build versus buy

| Factor | Build | Buy | Decision rule |
|---|---|---|---|
| Differentiation | Core to the product | Commodity | Build only the differentiator |
| Time to value | Slow | Fast | Buy when the timeline is short |
| Control | Full | Vendor-defined | Build when control is required |
| Cost | Staff-heavy | Fee-heavy | Compare first-year and three-year |
| Lock-in | Low | High | Price the exit |
| Compliance | Team-owned | Vendor-owned | Check the vendor's evidence |

The default should be buy for commodity functions and build for the differentiator. A team that builds its own orchestrator is spending its differentiation budget on plumbing.

### Reproducibility checklist

| Item | Evidence | Failure if missing |
|---|---|---|
| Code version | Commit hash | Cannot rerun |
| Environment | Container digest or lock file | Dependency drift |
| Data version | Hash or snapshot | Data drift |
| Parameters | Config file | Silent change |
| Random seeds | Recorded | Non-determinism |
| Hardware | CPU/GPU and driver | Performance difference |
| Outputs | Hash or artifact | Cannot verify |

A reproducible stack makes each row cheap to produce. A stack that makes them expensive will not be used.

### Supply-chain and security checks

| Check | Method | Frequency |
|---|---|---|
| Dependency inventory | SBOM generation | Each build |
| Vulnerability scan | Scanner on the image | Each build |
| Licence scan | SPDX or scanner | Each release |
| Maintainer activity | Release cadence | Each adoption |
| Signed artefacts | Signature verification | Each release |
| Pinned digests | Image digest | Each deployment |
| Secret scanning | Repository scan | Continuous |

The supply chain is part of the stack. A tool with an unmaintained dependency is a risk even if its own code is sound.

### Reference stacks by job

| Job | Minimal stack | Scale-up path | Avoid |
|---|---|---|---|
| Tabular analysis under 10 GB | Python, pandas, Parquet | DuckDB, then warehouse | Spark |
| Claims analytics under 200 GB | DuckDB, Parquet, dbt | PostgreSQL warehouse | A lakehouse for one team |
| Imaging research | pydicom, SimpleITK, MONAI | GPU cluster | A custom DICOM parser |
| Whole-slide pathology | OpenSlide, PyTorch | Tile feature store | Loading full slides into memory |
| Clinical NLP | Transformers, spaCy, vLLM | GPU serving | A custom tokenizer |
| Signal processing | WFDB, MNE, NeuroKit2 | Parallel batch | A custom file format |
| Survival analysis | lifelines, statsmodels | R survival | A deep model without a baseline |
| Causal analysis | DoWhy, EconML | Target trial emulation | A regression with no graph |
| Deployment | Docker, FastAPI, ONNX | Triton, Kubernetes | Kubernetes for one endpoint |
| Governance | Git, DVC, MLflow | A data catalog | A platform with no owner |

The minimal stack is the default. The scale-up path is the evidence that the minimal stack has an exit.

### Stack layers and their boundaries

| Boundary | What crosses it | Common failure |
|---|---|---|
| Source to storage | Schema and encoding | Silent type coercion |
| Storage to transformation | File format and partitioning | Small-file problem |
| Transformation to model | Feature definitions | Train/serve skew |
| Model to serving | Serialisation and preprocessing | Version mismatch |
| Serving to monitoring | Predictions and inputs | Missing ground truth |
| Monitoring to governance | Alerts and lineage | No owner |

Most production failures occur at a boundary, not inside a tool. The bake-off should test the boundaries, not only the components.

A boundary test is often more informative than a component benchmark. It reveals the schema mismatch, the serialisation cost, and the version skew that the demo never shows.

### Phase B — Compare candidates

**Step 5. Inventory candidates by function.**
Input: the job and the constraints.
Output: two or three candidates per layer.
Rationale: a single candidate is a justification, not a decision.
Branch: include the simplest credible option.
Edge case: a tool can cover several layers, which changes the comparison.

**Step 6. Score fit.**
Input: the scale, latency, and data model.
Output: a fit score with evidence.
Rationale: fit is the first filter.
Branch: if no candidate fits the latency, the architecture changes.
Edge case: a tool that fits today may not fit the growth path.

**Step 7. Score reproducibility.**
Input: the environment, the data versioning, and the test story.
Output: a reproducibility score.
Rationale: a result that cannot be rerun is an anecdote.
Branch: if the tool cannot pin its environment, it is a risk in a regulated setting.
Edge case: a managed service can change under the team without a version change.

**Step 8. Score licence.**
Input: the licence and the distribution model.
Output: a licence class and a legal question.
Rationale: the distribution model determines whether copyleft matters.
Branch: use the licence decision table below.
Edge case: a tool can have a permissive core and a proprietary enterprise edition.

**Step 9. Score team skill.**
Input: the team profile.
Output: a skill score and a training estimate.
Rationale: the best tool for the job is the one the team can run.
Branch: if the skill gap is large and the timeline short, choose the familiar tool.
Edge case: a skill gap can be closed with a contractor, at a cost.

**Step 10. Score cost and operations.**
Input: the licence fee, infrastructure, and staff time.
Output: a first-year cost and an operating estimate.
Rationale: the total cost includes operations, not only the licence.
Branch: use the cost model below.
Edge case: a free tool with high operations can be more expensive than a paid tool.

### Phase C — Prototype and measure

**Step 11. Build the same small analysis in both stacks.**
Input: a representative task and data.
Output: two working implementations.
Rationale: a prototype exposes the friction that a slide cannot.
Branch: if one stack cannot complete the task, it is out.
Edge case: a prototype on toy data can hide scale problems.

**Step 12. Measure setup time.**
Input: a stopwatch and a clean machine.
Output: hours to a working environment.
Rationale: setup time is a real cost and a reproducibility signal.
Branch: if setup exceeds a day, the onboarding burden is high.
Edge case: a container can reduce setup time and hide a complex build.

**Step 13. Measure runtime and memory.**
Input: the same data and hardware.
Output: elapsed time, peak memory, and cost.
Rationale: performance claims need measurement.
Branch: if the simple stack is within the latency budget, it wins.
Edge case: a benchmark on a laptop does not predict a cluster.

**Step 14. Test the failure modes.**
Input: a broken input, a missing dependency, a failed job.
Output: the error message, the recovery path, and the operator effort.
Rationale: the failure path determines the operating cost.
Branch: if recovery requires a specialist, the tool is a single point of failure.
Edge case: a silent failure is worse than a loud one.

**Step 15. Test the upgrade path.**
Input: a version bump and a dependency conflict.
Output: the effort to upgrade and the breakage.
Rationale: every tool will be upgraded.
Branch: if the upgrade breaks the pipeline, the pinning strategy is inadequate.
Edge case: a major version can change the file format.

### Phase D — Decide and document

**Step 16. Decide with the evidence.**
Input: the scores and the measurements.
Output: a selected stack and a rejected alternative.
Rationale: the decision must be reviewable.
Branch: if the scores are close, choose the simpler stack.
Edge case: a tie can be broken by the exit cost.

**Step 17. Write the decision record.**
Input: the evidence and the constraints.
Output: a record with the context, the options, the decision, and the consequences.
Rationale: the record is the artefact that survives turnover.
Branch: include the switching trigger and the owner.
Edge case: a decision record without a date is not a record.

**Step 18. Re-evaluate on scale, licence, or team change.**
Input: the growth, the licence, and the team.
Output: an updated record.
Rationale: stacks age.
Branch: re-run when the data doubles, the licence changes, or the team changes.
Edge case: a vendor can change its pricing or terms without a version change.

### Function-by-function table

| Function | Tool | Good for | Licence class | Characteristic failure |
|---|---|---|---|---|
| Data access | DuckDB | Local analytics on Parquet and CSV | MIT | Single-writer concurrency |
| Data access | PostgreSQL | Transactional and multi-user | PostgreSQL licence | Analytics at scale |
| Data access | pandas | Small data and exploration | BSD-3 | Memory limits |
| Data access | Polars | Fast single-node dataframes | MIT | Smaller ecosystem |
| Storage | Parquet on object storage | Columnar, cheap, portable | Apache-2.0 | No transactions |
| Storage | Delta Lake or Iceberg | Table semantics on a lake | Apache-2.0 | Operational complexity |
| Transformation | dbt | Versioned SQL and tests | Apache-2.0 | Not an orchestrator |
| Transformation | Spark | Distributed processing | Apache-2.0 | Overhead for small data |
| Orchestration | Airflow | Scheduled batch workflows | Apache-2.0 | Operator burden |
| Orchestration | Dagster | Asset-oriented pipelines | Apache-2.0 | Learning curve |
| Orchestration | Prefect | Python-native workflows | Apache-2.0 | Smaller ecosystem |
| Orchestration | Snakemake | Reproducible scientific workflows | MIT | Not a service |
| Modelling | scikit-learn | Classical models and pipelines | BSD-3 | Deep learning |
| Modelling | XGBoost | Tabular accuracy | Apache-2.0 | Serving and calibration |
| Modelling | PyTorch | Deep learning research | BSD-3 | Serving and operations |
| Imaging | pydicom | DICOM reading and writing | MIT | No pipeline |
| Imaging | SimpleITK | Image processing | Apache-2.0 | Not a model |
| Imaging | MONAI | Medical imaging models | Apache-2.0 | Not clinical validation |
| Imaging | OpenSlide | Whole-slide reading | LGPL-2.1 | Not a model |
| NLP | Transformers | Pretrained language models | Apache-2.0 | Memory and latency |
| NLP | spaCy and medspaCy | Clinical NLP pipelines | MIT | Not a generator |
| NLP | Presidio | PII detection | MIT | Not clinical de-identification |
| NLP | vLLM | High-throughput LLM serving | Apache-2.0 | GPU dependency |
| Signals | WFDB | ECG and waveform IO | MIT | Not a model |
| Signals | MNE | EEG and MEG analysis | BSD-3 | Learning curve |
| Signals | NeuroKit2 | Physiological signal processing | MIT | Method choices |
| Statistics | statsmodels | Inference and diagnostics | BSD-3 | Scale |
| Statistics | lifelines | Survival analysis | MIT | Complex designs |
| Statistics | DoWhy | Causal inference | MIT | Assumptions |
| Statistics | EconML | Heterogeneous effects | MIT | Assumptions |
| Deployment | Docker | Reproducible environments | Apache-2.0 | Image size |
| Deployment | BentoML | Model serving | Apache-2.0 | Operational scope |
| Deployment | Triton | High-performance inference | BSD-3 | GPU complexity |
| Deployment | ONNX Runtime | Portable inference | MIT | Operator coverage |
| Governance | MLflow | Tracking and registry | Apache-2.0 | Not a data catalog |
| Governance | DVC | Data and model versioning | Apache-2.0 | Not a backup |
| Governance | Great Expectations | Data quality checks | Apache-2.0 | Alert fatigue |
| Governance | Evidently | Drift monitoring | Apache-2.0 | Drift is not harm |
| Governance | Fairlearn | Fairness metrics | MIT | Metrics conflict |
| Governance | OpenDP | Differential privacy | MIT | Privacy budget |
| Governance | OpenLineage | Lineage metadata | Apache-2.0 | Adoption |

The table is a starting point. The licence class is a legal question, and the failure mode is the one to test in the bake-off.

### Licence decision table

| Licence class | Commercial use | Distribution obligation | Internal use | Legal review |
|---|---|---|---|---|
| MIT, BSD, Apache-2.0 | Permitted | Attribution and notices | Permitted | Light |
| LGPL | Permitted | Relinkable library | Permitted | Moderate |
| GPL | Permitted | Derivative distribution under GPL | Permitted | High |
| AGPL | Permitted | Network use can trigger | Permitted | High |
| Proprietary | Contract | Contract | Contract | High |
| Source-available | Usually restricted | Usually restricted | Usually restricted | High |

The distribution model decides whether copyleft matters. A model served inside a company is different from a model distributed as software.

### When a simpler tool wins

| Situation | Over-engineered choice | Simpler choice | Why |
|---|---|---|---|
| Data under 50 GB | Spark cluster | DuckDB or Polars | Single node is faster to operate |
| Three scheduled jobs | Airflow cluster | Cron or a workflow file | Less infrastructure |
| One model endpoint | Kubernetes | A container and a queue | Lower operations |
| A one-off analysis | A pipeline platform | Notebook and a script | Speed |
| A small team | A data catalog | A README and a schema | Adoption |
| A stable dataset | A lakehouse | Parquet files | Portability |

The simpler tool is not always right. It is right when the scale, the team, and the latency allow it.

### Edge cases that change the procedure

**Licence forbids commercial use.** A copyleft or source-available tool can block a distributed product. Legal review is required before adoption, not after.

**Project abandoned.** A tool can lose its maintainer. Check the release cadence and the governance before adopting.

**Credentialing takes months.** A managed service can require procurement and security review. A local tool can start sooner.

**Version drift.** An unpinned dependency can change behaviour. Pin versions and container digests.

**Label errors discovered.** A data quality tool can miss semantic errors. Validation is a separate step.

**Benchmark contamination.** A tool benchmark can be tuned to the benchmark. Measure on your own data.

## 6. Worked example

### The case

A 20-person analytics team supports a payer's care-management programme. The data are about 200 GB of claims and enrolment files, refreshed monthly. The team needs cohort construction, feature engineering, a readmission model, and a monthly scoring job. It has one data engineer and no dedicated site-reliability engineer.

**Job statement.** "Produce a monthly cohort and feature table from 200 GB of claims, train and recalibrate a readmission model, and score the eligible population within three business days of the data refresh."

**Constraints.** The team knows Python and SQL, has no Spark experience, and cannot staff an on-call rotation. The data arePHI and must remain in the approved cloud region.

### Candidate stacks

| Stack | Components | Why it is on the list |
|---|---|---|
| A | DuckDB, Python, Parquet, cron | Simplest stack that fits the scale |
| B | PostgreSQL, dbt, Airflow | Familiar warehouse pattern |
| C | Spark, Delta Lake, Airflow | Distributed path for growth |

### Score table

| Criterion | Weight | A | B | C |
|---|---:|---:|---:|---:|
| Fit | 0.30 | 4 | 5 | 5 |
| Reproducibility | 0.20 | 4 | 5 | 4 |
| Licence | 0.20 | 5 | 5 | 5 |
| Team skill | 0.15 | 5 | 3 | 2 |
| Cost | 0.10 | 5 | 3 | 2 |
| Operations | 0.05 | 4 | 3 | 2 |
| Weighted total | 1.00 | 4.45 | 4.40 | 3.90 |

**Fit.** Stack A handles 200 GB on a single node with DuckDB spilling to disk. Stack B handles it comfortably and supports concurrent users. Stack C handles it and scales further. A scores 4 because the single-node ceiling is closer; B and C score 5.

**Reproducibility.** Stack A uses a Python lock file, Parquet snapshots, and a cron script. Stack B uses dbt models, a database, and Airflow. Stack C uses notebooks and a cluster. B scores 5, A and C score 4.

**Licence.** All components are permissive or Apache-2.0. All score 5. The licence does not decide this case.

**Team skill.** The team knows Python and SQL and has no Spark experience. A scores 5, B scores 3, C scores 2.

**Cost.** The first-year cost model gives A about $16,000, B about $41,400, and C about $75,900 in staff and infrastructure. A scores 5, B 3, C 2.

**Operations.** A needs a cron job and a disk monitor. B needs a database administrator and an Airflow deployment. C needs a cluster and an on-call rotation. A scores 4, B 3, C 2.

### The two diagnostics

**Diagnostic 1: the benchmark does not scale linearly.** The team prototyped on a 1 GB sample. Stack A completed the cohort in 40 seconds; Stack C completed it in 90 seconds because of cluster overhead. On 200 GB, Stack A took 38 minutes with disk spilling, and Stack C took 22 minutes. The ranking reversed. A benchmark on a small sample measures startup overhead, not throughput at scale.

**Diagnostic 2: the operations burden is not in the benchmark.** Stack B's Airflow deployment required a metadata database, a scheduler, a webserver, and an upgrade path. The team had no one to own it. The model ran, and the pipeline became a single point of failure. The diagnostic is the on-call question: who fixes it at 2 a.m.?

A third diagnostic is the exit cost. Stack A can move to B or C by rewriting SQL and file paths. Stack C's Delta tables and Spark jobs are harder to move. The simpler stack has the lower exit cost.

### The decision

Stack A is selected. The margin over B is 0.05, which is within the noise of the scoring. The tie-break is the operations burden and the exit cost, not the weighted total.

| Track | Steps | Owner | Duration | Planning cost |
|---|---|---|---|---|
| Prototype | Cohort and model on a sample | Data engineer | 2 weeks | Staff |
| Scale test | 200 GB run and memory profile | Data engineer | 2 weeks | Compute |
| Environment | Lock file, container, snapshots | Data engineer | 1 week | Staff |
| Pipeline | Cron, retries, alerts | Data engineer | 2 weeks | Staff |
| Handover | Runbook and training | Team lead | 1 week | Staff |

### Time and cost

| Item | A | B | C |
|---|---:|---:|---:|
| Setup hours | 40 | 160 | 240 |
| Run hours per month | 10 | 17 | 25 |
| Infrastructure per month | $200 | $900 | $2,500 |
| First-year staff and infrastructure | $16,000 | $41,400 | $75,900 |

The numbers are planning estimates, not quotes. The structural point is that the simplest stack is also the cheapest to operate at this scale.

### Substitute plan if the data grow to 2 TB

If the data grow tenfold, Stack A's single-node run exceeds the three-day window. The plan changes in three ways:

1. Move the transformation layer to Stack B or C, keeping the Python model code unchanged.
2. Partition Parquet by year and site to reduce the scanned data.
3. Re-run the bake-off at the new scale before committing to a cluster.

The exit is designed, not improvised. That is the value of choosing the simpler stack first.

### Vary one assumption: the team already knows Spark

Suppose the team has two Spark engineers and an on-call rotation. Stack C's team-skill score rises from 2 to 5 and its operations score from 2 to 4. The weighted total rises above Stack A, and the decision changes. The tool did not change; the team did.

### Conditions under which the recommendation changes

| Condition | Effect on the scores | New decision |
|---|---|---|
| Data grows to 2 TB | A fit falls | Move to B or C |
| Team hires Spark engineers | C team and ops rise | Re-run the bake-off |
| Central Airflow is available | B ops rise | B becomes competitive |
| Real-time scoring required | A fit falls | Add a serving layer |
| A copyleft dependency is added | Licence score falls | Legal review |
| The vendor changes pricing | Cost scores change | Re-run the cost model |

The recommendation is conditional on the context. A decision record that does not state the conditions is a preference.

### Gate evidence log

| Gate | Stack A | Stack B | Stack C |
|---|---|---|---|
| Fit | Single node, 200 GB | Warehouse, concurrent | Cluster, growth |
| Reproducibility | Lock file and snapshots | dbt and database | Cluster config |
| Licence | Permissive | Permissive | Permissive |
| Team skill | Python and SQL | SQL plus Airflow | Spark required |
| Cost | Lowest | Middle | Highest |
| Operations | Cron and disk | DBA and Airflow | Cluster and on-call |

### Decision timeline

| Month | Bake-off | Environment | Handover | Deliverable |
|---:|---|---|---|---|
| 1 | Prototype on sample | Lock file | Runbook draft | Working prototype |
| 2 | Scale test at 200 GB | Container | Training | Performance report |
| 3 | Decision record | Snapshot process | Review | Selected stack |
| 4 | Pipeline build | Alerts | On-call plan | Monthly job |
| 5 | Monitoring | Drift checks | Sign-off | Production |

### What the case demonstrates

- The simplest stack won at 200 GB, and the margin was small. The tie-break was operations and exit cost.
- The small-sample benchmark reversed at scale. A benchmark on 1 GB measured startup overhead.
- The operations burden was invisible in the benchmark and decisive in the decision.
- The licence did not decide the case because all candidates were permissive.
- The substitute plan preserved the model code and changed the transformation layer.
- The team skill, not the tool capability, was the largest score difference.

### Sensitivity of the decision

| Change | Effect | Plan response |
|---|---|---|
| Data grows to 2 TB | Single node exceeds the window | Move transformation to B or C |
| Team hires Spark engineers | C becomes competitive | Re-run the bake-off |
| A managed warehouse is procured | B's operations fall | Re-score B |
| Airflow is already operated centrally | B's operations rise | Re-score B |
| The model moves to real-time | Batch stack insufficient | Add a serving layer |
| A copyleft dependency is added | Licence score falls | Legal review |

### Conditions under which the case would fail

- **The data exceed the single-node ceiling.** The runtime misses the window, and the transformation layer moves.
- **The team loses its only data engineer.** The runbook is insufficient, and the pipeline stalls.
- **The licence changes.** A dependency moves to a restrictive licence, and legal review blocks distribution.
- **The cloud region changes.** The data cannot move, and the stack must be redeployed.
- **A managed service is mandated.** The build option is removed, and the comparison restarts.
- **The model needs real-time scoring.** The batch stack is insufficient, and a serving layer is added.

The decision is reversible at a known cost, which is the main reason to choose the simpler stack first. A reversible decision can be corrected; an irreversible one cannot.

The recommendation is a starting point, and the conditions above are the contract for revisiting it. A stack that is right for 200 GB and a 20-person team is not right for 2 TB and a 200-person organisation.

A stack decision is a hypothesis about the next two years. Write the hypothesis and the test that would falsify it.

The falsification test is the scale test and the operations review.

If either fails, the decision reopens.

## 7. Code, query, or calculation

The three blocks below use the Python standard library only. They implement the weighted decision matrix, the first-year cost model, and a licence-class check.

### 7.1 Weighted decision matrix

```python
CRITERIA = {
    "fit": 0.30,
    "reproducibility": 0.20,
    "licence": 0.20,
    "team_skill": 0.15,
    "cost": 0.10,
    "operations": 0.05,
}
STACKS = {
    "DuckDB + Python + Parquet": {"fit": 4, "reproducibility": 4, "licence": 5, "team_skill": 5, "cost": 5, "operations": 4},
    "PostgreSQL + dbt + Airflow": {"fit": 5, "reproducibility": 5, "licence": 5, "team_skill": 3, "cost": 3, "operations": 3},
    "Spark + Delta Lake": {"fit": 5, "reproducibility": 4, "licence": 5, "team_skill": 2, "cost": 2, "operations": 2},
}

print(f"{'Stack':<32}{'Weighted score':>16}")
for name, scores in STACKS.items():
    total = sum(CRITERIA[k] * scores[k] for k in CRITERIA)
    print(f"{name:<32}{total:>16.2f}")
print()
print("Weights sum:", sum(CRITERIA.values()))
print("Scores are 1 to 5, higher is better.")
```

Expected output:

```text
Stack                             Weighted score
DuckDB + Python + Parquet                   4.45
PostgreSQL + dbt + Airflow                  4.40
Spark + Delta Lake                          3.90

Weights sum: 1.0
Scores are 1 to 5, higher is better.
```

The margin between the top two is small. A weighted score is a communication device; the decision needs the tie-break criteria.

### 7.2 First-year cost model

```python
def first_year_cost(setup_hours, run_hours, infra_month, staff_rate, licence_fee=0):
    staff = (setup_hours + run_hours) * staff_rate
    infra = infra_month * 12
    return staff + infra + licence_fee

stacks = {
    "DuckDB + Python": (40, 120, 200, 85, 0),
    "Postgres + dbt + Airflow": (160, 200, 900, 85, 0),
    "Spark + Delta": (240, 300, 2500, 85, 0),
}
print(f"{'Stack':<28}{'Staff':>10}{'Infra':>10}{'Total':>10}")
for name, args in stacks.items():
    total = first_year_cost(*args)
    staff = (args[0] + args[1]) * args[3]
    infra = args[2] * 12
    print(f"{name:<28}{staff:>10,}{infra:>10,}{total:>10,}")
print()
print("Setup time is a one-off; run time and infrastructure recur.")
```

Expected output:

```text
Stack                            Staff     Infra     Total
DuckDB + Python                 13,600     2,400    16,000
Postgres + dbt + Airflow        30,600    10,800    41,400
Spark + Delta                   45,900    30,000    75,900

Setup time is a one-off; run time and infrastructure recur.
```

The model is deliberately simple. It omits hiring, training, and the cost of an outage, all of which favour the simpler stack at this scale.

### 7.3 Licence-class check

```python
LICENCE_CLASSES = {
    "MIT/Apache-2.0": {"commercial": True, "copyleft": False, "patent_grant": True},
    "GPL-3.0": {"commercial": True, "copyleft": True, "patent_grant": True},
    "AGPL-3.0": {"commercial": True, "copyleft": True, "patent_grant": True},
    "Proprietary": {"commercial": "contract", "copyleft": False, "patent_grant": "contract"},
}
print(f"{'Licence':<18}{'Commercial':>12}{'Copyleft':>10}{'Patent':>10}")
for name, terms in LICENCE_CLASSES.items():
    print(f"{name:<18}{str(terms['commercial']):>12}{str(terms['copyleft']):>10}{str(terms['patent_grant']):>10}")
print()
print("Copyleft affects distribution of derivative software, not internal use.")
print("AGPL can be triggered by network use; legal review is required.")
```

Expected output:

```text
Licence             Commercial  Copyleft    Patent
MIT/Apache-2.0            True     False      True
GPL-3.0                   True      True      True
AGPL-3.0                  True      True      True
Proprietary           contract     False  contract

Copyleft affects distribution of derivative software, not internal use.
AGPL can be triggered by network use; legal review is required.
```

The table is a screening device. A legal review decides whether a specific distribution triggers a copyleft obligation.

## 8. Data, standards, and regulatory context

Tool choices are constrained by software standards, security frameworks, and medical-device regulation. The table separates binding requirements from conventions.

| Instrument | Status | What it constrains | What it does not do |
|---|---|---|---|
| SPDX | Standard | Licence and SBOM representation | Provide legal advice |
| SemVer | Convention | Version numbering | Guarantee compatibility |
| CycloneDX | Standard | SBOM format | Guarantee security |
| SLSA | Framework | Build provenance and integrity | Certify a product |
| OpenSSF Scorecard | Framework | Open-source security signals | Replace review |
| ISO/IEC 27001 | Certifiable standard | Information security management | Certify a model |
| HIPAA Security Rule | Binding in the US | Safeguards for PHI | Prescribe a tool |
| GDPR | Binding in the EU | Personal data processing | Prescribe a stack |
| IEC 62304 | Voluntary, often contractual | Medical device software lifecycle | Validate a model |
| FDA device guidance | Binding when applicable | Software in a device | Certify a framework |

**A permissive licence is not a security review.** A tool can be MIT-licensed and unmaintained. The licence and the supply chain are separate checks.

**A framework is not a product.** A model trained in PyTorch is not a medical device. The device evidence is separate from the training stack.

**A managed service is a supply-chain dependency.** The vendor's uptime, terms, and subprocessors become part of the stack.

### Security and compliance checks by layer

| Layer | Check | Evidence | Failure if skipped |
|---|---|---|---|
| Storage | Encryption at rest | Configuration | Data exposure |
| Storage | Access logging | Log export | No audit trail |
| Transformation | Secret handling | Secret manager | Credential leak |
| Orchestration | Least privilege | Role definition | Lateral movement |
| Modelling | Dependency scan | SBOM | Vulnerable package |
| Serving | Authentication | Gateway config | Open endpoint |
| Monitoring | PII in logs | Log review | Privacy incident |
| Governance | Retention | Policy record | Over-retention |

The checks are layer-specific. A secure storage layer does not protect a serving endpoint that logs prompts.

### Reproducibility standards

| Standard | What it gives | What it does not give |
|---|---|---|
| Lock file | Exact dependency versions | Data version |
| Container digest | Exact environment | Hardware |
| SBOM | Dependency inventory | Vulnerability status |
| Signed artefact | Provenance | Correctness |
| Data hash | Dataset identity | Data quality |
| Config file | Parameter record | Parameter correctness |

A reproducible stack combines all six. A container alone is not reproducibility.

### Regulatory artefacts for a stack

- **Software lifecycle file** describes the development, verification, and maintenance process.
- **SBOM** lists the components and licences.
- **Security review** covers vulnerabilities and access.
- **Validation record** covers the intended use and the tests.
- **Change control** covers upgrades and their impact.
- **Monitoring** covers runtime behaviour and incidents.

## 9. Industry practice and case evidence

### What organisations actually do

- **Choose the simplest stack that meets the latency and scale.** Mature teams resist distributed systems for small data.
- **Measure the bake-off.** Setup time, runtime, memory, and lines of code are recorded.
- **Pin and scan.** Lock files, image digests, SBOMs, and vulnerability scans are standard.
- **Own the operations.** Every tool has a named owner and a runbook.
- **Price the exit.** The switching cost is part of the decision.
- **Review licences before adoption.** Legal review is not an afterthought.
- **Retire tools.** A stack accumulates; mature teams remove components.

### Case evidence

| Case | Documented fact | Inference |
|---|---|---|
| DuckDB | An in-process analytical database distributed under the MIT licence. | A single-node engine can replace a cluster for many analytical workloads. |
| PostgreSQL | An open-source relational database under the PostgreSQL licence. | A general-purpose database is a strong default for multi-user workloads. |
| dbt | A SQL transformation framework under Apache-2.0. | Transformation logic can be versioned and tested without a custom framework. |
| Airflow | A workflow orchestrator under Apache-2.0. | Orchestration is powerful and operationally demanding. |
| MONAI | A medical imaging toolkit under Apache-2.0. | Domain tooling reduces boilerplate and does not validate a model. |
| MLflow | A tracking and registry platform under Apache-2.0. | Experiment tracking is a commodity and should not be built in-house. |

The rows are stable facts about the projects. The inference is a planning judgement for a specific team.

### Tool retirement

| Signal | Action | Risk if ignored |
|---|---|---|
| No release for 24 months | Evaluate alternatives | Security gaps |
| Maintainer departure | Check governance | Abandonment |
| Repeated vulnerabilities | Patch or replace | Exploitation |
| Licence change | Legal review | Distribution block |
| Cost increase | Re-run the cost model | Budget overrun |
| Team skill loss | Retrain or replace | Single owner |

A stack accumulates tools. A retirement review is as important as an adoption review.

### What good looks like

- The job, scale, team, and constraints are written before the tool list.
- At least two candidates are compared per layer.
- The bake-off measures setup, runtime, memory, and failure.
- The licence is reviewed against the distribution model.
- Every tool has an owner and a runbook.
- The exit cost is stated.
- The decision record has a date and a switching trigger.

### Ownership register

| Component | Owner | Runbook | Review date |
|---|---|---|---|
| Ingestion | Data engineer | Yes | Quarterly |
| Storage | Platform | Yes | Quarterly |
| Transformation | Analytics | Yes | Quarterly |
| Orchestration | Data engineer | Yes | Quarterly |
| Model training | Data science | Yes | Per release |
| Serving | Platform | Yes | Per release |
| Monitoring | Data science | Yes | Monthly |

A component without an owner is a component without a review. The register is the minimum governance artefact for a stack.

### Why stack choices mislead teams

A tool demo shows the happy path on a small dataset. The production path includes setup, failure, upgrade, and operations. A team that chooses on the demo chooses on the least representative part of the evidence. The bake-off exists to measure the rest.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Over-engineering | A cluster for 20 GB | Fashion or résumé-driven choice | Operations burden | Start simple |
| Unpinned environment | Ranges in requirements | Dependency drift | Irreproducible result | Lock files and digests |
| Copyleft surprise | No licence review | Distribution trigger | Release blocked | Licence scan |
| Single owner | One person knows the pipeline | Key-person risk | Stalled work | Runbook and pairing |
| Benchmark on toy data | Small-sample timing | Startup dominates | Wrong choice | Scale test |
| Hidden managed dependency | Vendor telemetry | Residency or cost | Compliance issue | Data-flow review |
| Alert fatigue | Thousands of warnings | Poor thresholds | Ignored alerts | Triage and severity |
| Upgrade breakage | Major version bump | Format or API change | Pipeline failure | Staged upgrades |
| Train-serve skew | Features differ in serving | Separate code paths | Wrong predictions | Shared feature code |
| Orphaned tool | No owner | Team change | Unmaintained | Ownership register |
| Licence change | Project relicensed | Supply-chain risk | Legal exposure | Monitor releases |
| Data format lock-in | Proprietary table format | Migration cost | Vendor dependence | Open formats |
| Secret in image | Credentials baked in | Build practice | Security incident | Secret scanning |
| Unmonitored drift | No baseline | Silent change | Degraded model | Monitoring |

Red flags that should stop a stack decision:

- The only evidence is a vendor demo.
- No one can name the owner.
- The licence has not been reviewed.
- The environment is not pinned.
- The benchmark used toy data.
- The exit cost is unknown.
- The operations model is "we will figure it out."
- The stack has more components than the team has people.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The job is stable | The stack follows the job | The product changes | Re-run the decision |
| The scale estimate is right | Fit depends on it | Growth | Measure and forecast |
| The team can operate it | Operations depend on it | Turnover | Runbook and on-call |
| The licence permits the use | Distribution depends on it | Relicensing | Licence scan |
| The benchmark is representative | The choice depends on it | Toy data | Scale test |
| The vendor terms are stable | Cost and residency depend on them | Price or region change | Contract review |
| The exit is affordable | Reversibility depends on it | Proprietary formats | Migration plan |
| The dependencies are maintained | Security depends on it | Abandonment | Release cadence |
| The environment is reproducible | Reruns depend on it | Unpinned versions | Clean-machine test |
| The monitoring is meaningful | Drift detection depends on it | No baseline | Backtest |
| The training and serving code match | Predictions depend on it | Separate paths | Shared code |
| The owner is named | Continuity depends on it | Turnover | Ownership register |

The deepest limit is that a stack decision is a bet on the future. The data may grow, the team may change, and the vendor may reprice. The decision record should state the bet and the trigger to revisit it.

## 12. Exercises

**Exercise 1. Map functions to tools.**
Deliverable: a table of eight functions with two candidates each and the trade-off.
Acceptance criteria: eight functions; licence class; one failure mode per candidate.
Time: 40 minutes.

**Exercise 2. Score a stack.**
Deliverable: a weighted decision matrix with evidence per score.
Acceptance criteria: weights justified; scores evidenced; a tie-break rule.
Time: 45 minutes.

**Exercise 3. Classify licences.**
Deliverable: a table of six tools with licence class and distribution consequence.
Acceptance criteria: six tools; commercial and copyleft columns; one legal question.
Time: 35 minutes.

**Exercise 4. Run a bake-off.**
Deliverable: setup time, runtime, memory, and lines of code for two stacks on the same task.
Acceptance criteria: same data and hardware; measured not estimated; a recommendation.
Time: 90 minutes.

**Exercise 5. Identify failure modes.**
Deliverable: a failure-mode table for a proposed stack with detection and mitigation.
Acceptance criteria: at least eight tools; one boundary failure; an owner.
Time: 45 minutes.

**Exercise 6. Decide when simpler wins.**
Deliverable: a one-page memo comparing a simple and a distributed option at a stated scale.
Acceptance criteria: scale evidence; operations estimate; exit cost.
Time: 40 minutes.

**Exercise 7. Design a reproducible environment.**
Deliverable: a lock file, container digest, data hash, and config record.
Acceptance criteria: clean-machine rerun; versions pinned; data identified.
Time: 60 minutes.

**Exercise 8. Write a decision record.**
Deliverable: context, options, decision, consequences, and switching trigger.
Acceptance criteria: rejected option named; exit cost stated; owner and date.
Time: 45 minutes.

## 13. Self-check answers

**Item 1. Why is a single-node tool often enough?**
Many analytical datasets fit in memory or on local disk, and a single-node engine avoids the operational cost of a cluster. The correct question is the data size and latency, not the prestige of the tool.

**Item 2. What does a weighted score not decide?**
It does not decide the tie. A small difference is within scoring noise, and the decision needs the operations, exit cost, and risk.

**Item 3. When does copyleft matter?**
When the software is distributed or, for AGPL, when it is offered as a network service. Internal use is usually unaffected, and legal review is required.

**Item 4. Why is setup time a reproducibility signal?**
A stack that takes days to set up is hard to recreate, and a hard-to-recreate environment produces results that cannot be rerun. The clean-machine test is the evidence.

**Item 5. What is train-serve skew?**
A difference between the features used in training and the features available at serving. It produces wrong predictions even when the model is unchanged.

**Item 6. Why is a toy benchmark misleading?**
At small scale, startup and overhead dominate; at large scale, throughput and memory dominate. The ranking can reverse.

**Item 7. What is the exit cost?**
The effort to replace a tool, including data migration, code changes, and retraining. It is part of the decision.

**Item 8. Why does every tool need an owner?**
Tools decay. Without an owner, upgrades, vulnerabilities, and failures go unaddressed.

**Item 9. What is an SBOM?**
A software bill of materials: the list of components and versions in an artefact. It supports vulnerability and licence review and does not guarantee security.

**Item 10. When is a managed service the right choice?**
When the team cannot operate the tool, the compliance evidence is adequate, and the lock-in is priced. It is not right merely because it is convenient.

**Item 11. Why is a data contract useful between teams?**
It defines the schema, semantics, and ownership of a dataset so that downstream teams can depend on it. It reduces silent breakage and clarifies who fixes a change.

**Item 12. What is the difference between observability and monitoring?**
Monitoring checks known metrics against thresholds; observability asks whether the system's internal state can be inferred from its outputs. A stack needs both.

**Item 13. Why is idempotency important in a pipeline?**
A rerun should produce the same result and not duplicate data. Non-idempotent jobs create duplicates after a retry, which is a common failure in scheduled pipelines.

**Item 14. What is a canary release?**
A small-traffic rollout to detect failures before a full deployment. It reduces the blast radius and requires a way to compare the canary with the baseline.

**Item 15. Why is a feature store not a substitute for a data contract?**
A feature store centralises computation; a contract defines meaning and ownership. A feature store with undefined semantics produces consistent but wrong features.

**Item 16. What is the first check before adopting a new tool?**
The job, the scale, the team, and the licence. A strong answer starts with the job. A weak answer starts with the tool's popularity.

**Item 17. Why is a runbook part of the stack?**
It encodes how to detect, diagnose, and recover from failures. A tool without a runbook depends on the memory of one person, which is a single point of failure.

**Item 18. What is the difference between a data lake and a warehouse?**
A lake stores raw files with schema-on-read; a warehouse stores structured tables with schema-on-write. The choice determines the transformation and governance model.

**Rubric for the design exercises (2, 4, 6, 8).** Full marks require measured evidence, a licence check, an operations estimate, and an exit plan. Partial marks for a complete matrix with estimated numbers. No marks for a vendor demo presented as a bake-off.

## 14. Interview questions

**1. How would you choose a stack for a 200 GB analytics workload?**
Start with the simplest stack that meets the latency, measure the bake-off, check the licence, and price the operations and exit. A strong answer names the scale and the team. A weak answer names a favourite tool.

**2. When is Spark the wrong choice?**
When the data fit on one machine and the team cannot operate a cluster. A strong answer gives a scale threshold. A weak answer says Spark is always for big data.

**3. What does a copyleft licence require?**
Derivative distributions under the same licence, with source availability. A strong answer notes that internal use is different. A weak answer says it forbids commercial use.

**4. How do you make an analysis reproducible?**
Pin code, environment, data, parameters, and seeds; test on a clean machine; record the outputs. A strong answer names the artefacts. A weak answer says "use Docker."

**5. What is train-serve skew and how do you prevent it?**
A difference between training and serving features. Share the feature code and test parity. A strong answer proposes a parity test. A weak answer blames the model.

**6. Why is a bake-off better than a demo?**
It measures setup, runtime, memory, and failure on the same task. A strong answer reports the numbers. A weak answer relies on impressions.

**7. How do you assess a tool's supply chain?**
SBOM, vulnerability scan, maintainer activity, licence scan, signed releases, and pinned digests. A strong answer names the checks. A weak answer checks the stars.

**8. What is the total cost of a free tool?**
Infrastructure, staff time, operations, training, and the cost of failure. A strong answer includes operations. A weak answer says zero.

**9. When would you choose a managed service?**
When the team lacks the operations capacity and the compliance evidence is adequate. A strong answer prices the lock-in. A weak answer chooses convenience.

**10. What belongs in a stack decision record?**
Context, options, evidence, decision, consequences, owner, date, and switching trigger. A strong answer includes the rejected option. A weak answer lists the chosen tools.

**11. How do you handle a major version upgrade?**
Stage it, test in a branch, check the changelog and the data format, and roll back if needed. A strong answer plans the upgrade. A weak answer upgrades in production.

**12. When do you retire a tool?**
When it is unmaintained, insecure, superseded, or no longer owned. A strong answer has a trigger. A weak answer keeps it because it works.

**13. How do you decide between a lake and a warehouse?**
By the data structure, the users, and the governance. A lake suits raw and varied data; a warehouse suits structured analytics. A strong answer notes that most teams need both.

**14. What is the role of a feature store in a regulated product?**
It centralises feature computation and supports parity between training and serving. It does not replace a data contract or a validation of feature meaning.

**15. How would you evaluate a managed service's lock-in?**
By the export format, the migration effort, the pricing, and the contractual exit. A strong answer prices the exit. A weak answer assumes the service is interchangeable.

**16. When is a notebook acceptable in production?**
Rarely. A notebook can be a prototype and should be converted to a tested, scheduled job with pinned dependencies. A strong answer names the conversion. A weak answer schedules the notebook.

## 15. Key terms

- **AGPL** — copyleft licence with a network-use trigger.
- **Airflow** — workflow orchestrator.
- **Apache-2.0** — permissive licence with a patent grant.
- **Bake-off** — controlled comparison of stacks.
- **BentoML** — model serving framework.
- **Copyleft** — licence requiring derivative distribution under the same terms.
- **CycloneDX** — SBOM standard.
- **Dagster** — asset-oriented orchestrator.
- **dbt** — SQL transformation framework.
- **DuckDB** — embedded analytical database.
- **DVC** — data version control.
- **Evidently** — drift monitoring library.
- **Exit cost** — cost of replacing a tool.
- **Fairlearn** — fairness metrics library.
- **GPL** — copyleft licence.
- **Great Expectations** — data quality library.
- **IEC 62304** — medical device software lifecycle standard.
- **LGPL** — weaker copyleft library licence.
- **Lock file** — exact dependency versions.
- **Managed service** — vendor-operated platform.
- **MIT** — permissive licence.
- **MLflow** — tracking and model registry.
- **MONAI** — medical imaging toolkit.
- **ONNX** — model interchange format.
- **OpenDP** — differential privacy library.
- **OpenLineage** — lineage standard.
- **Parquet** — columnar file format.
- **Polars** — single-node dataframe library.
- **Presidio** — PII detection library.
- **Reproducibility** — ability to rerun a result.
- **SBOM** — software bill of materials.
- **SemVer** — semantic versioning.
- **SLSA** — build provenance framework.
- **SPDX** — licence and SBOM standard.
- **Switching cost** — see exit cost.
- **Train-serve skew** — feature mismatch.
- **Triton** — inference server.
- **vLLM** — LLM inference engine.
- **WFDB** — waveform data format and library.
- **XGBoost** — gradient boosting library.
- **Artifact registry** — storage for versioned build artefacts.
- **Blue-green deployment** — two environments with a switch.
- **Canary release** — a small-traffic rollout.
- **Configuration drift** — environments diverging over time.
- **Data contract** — an agreed schema and semantics.
- **Dependency hell** — conflicting version requirements.
- **Deterministic build** — a build that reproduces byte-for-byte.
- **Feature store** — a shared feature computation layer.
- **Golden image** — a validated base image.
- **Idempotency** — a rerun produces the same result.
- **Infrastructure as code** — declarative infrastructure.
- **Lineage** — the recorded path of data through the stack.
- **Observability** — logs, metrics, and traces.
- **Orphan dependency** — an unmaintained package.
- **Pinning** — fixing a version or digest.
- **Provenance** — the origin and build history of an artefact.
- **Rollback** — returning to a previous version.
- **Runbook** — the operating procedure for a failure.
- **Schema registry** — a central schema definition.
- **Secrets management** — controlled storage of credentials.
- **Service level objective** — a target for reliability.
- **Snowflake environment** — an unmanaged, drifting environment.
- **Staging environment** — a pre-production copy.
- **Stateless service** — a service with no local state.
- **Telemetry** — emitted operational data.
- **Vendor lock-in** — dependence on a vendor.
- **Version skew** — components on incompatible versions.
- **Zero-downtime deployment** — deployment without service interruption.
- **Reproducible build** — a build that can be recreated.
- **Immutable artefact** — an artefact that is never modified.
- **Change data capture** — propagating database changes.
- **Batch window** — the time allowed for a batch job.
- **Backfill** — reprocessing historical data.
- **Circuit breaker** — a failure-isolation pattern.
- **Dead letter queue** — storage for failed messages.
- **Rate limit** — a cap on requests.
- **Retry policy** — rules for retrying failures.
- **Time to recovery** — how long a failure takes to fix.
- **Access control** — rules limiting who can use a resource.
- **Audit log** — a record of actions.
- **Backup** — a copy for recovery.
- **Baseline model** — a simple reference model.
- **Batch inference** — scoring in bulk.
- **Columnar storage** — storing data by column.
- **Compute budget** — the allowed spend on compute.
- **Data catalog** — an inventory of datasets.
- **Data lake** — storage for raw data.
- **Data quality rule** — a constraint on data.
- **Data warehouse** — structured analytical storage.
- **Delta table** — a table format with transactions.
- **Embedded database** — a database inside an application.
- **ETL** — extract, transform, load.
- **ELT** — extract, load, transform.
- **Feature engineering** — creating model inputs.
- **Inference latency** — time to score one case.
- **Job scheduler** — a system that runs jobs on a timetable.
- **Model registry** — a store for model versions.
- **Model serving** — exposing a model for predictions.
- **Notebook** — an interactive analysis document.
- **Object storage** — storage for files as objects.
- **Partitioning** — splitting data by a key.
- **Pipeline** — a sequence of data steps.
- **Query engine** — software that executes queries.
- **Replication** — copying data for availability.
- **Schema** — the structure of data.
- **Serverless** — compute managed by a provider.
- **Streaming** — processing data as it arrives.
- **Table format** — a specification for tabular files.
- **Throughput** — work completed per unit time.
- **Transaction** — an atomic unit of work.
- **Uptime** — the share of time a service is available.
- **Vectorised execution** — processing data in batches.
- **Workload isolation** — separating jobs to avoid interference.
- **Zero-copy clone** — a cheap copy of a dataset.
- **Blue-green database** — two databases with a switch.
- **Change log** — a record of changes.
- **Data freshness** — how recent the data are.
- **Access pattern** — how data are read.
- **Batch size** — records processed per step.
- **Benchmark harness** — a repeatable performance test.
- **Cluster** — a group of machines.
- **Container image** — a packaged environment.
- **Data skew** — uneven partition sizes.
- **Distributed join** — a join across machines.
- **Execution plan** — the engine's query plan.
- **File format** — the encoding of stored data.
- **Index** — a structure for fast lookup.
- **Ingestion** — bringing data into the system.
- **Job** — a unit of scheduled work.
- **Materialisation** — storing a computed result.
- **Memory spill** — writing to disk when memory is full.
- **Partition pruning** — skipping irrelevant partitions.
- **Query optimisation** — improving a query plan.
- **Replication factor** — the number of copies.
- **Shuffle** — redistributing data across machines.
- **Single point of failure** — a component whose failure stops the system.
- **Staging table** — an intermediate table.
- **Streaming ingestion** — continuous data intake.
- **Table scan** — reading all rows.
- **Task** — a unit of work in a workflow.
- **Throughput test** — a test of work per unit time.
- **Vectorisation** — batch processing in CPU-friendly form.
- **Workflow** — a set of dependent tasks.
- **Write-ahead log** — a durability mechanism.
- **Zone map** — per-block statistics for pruning.
- **Column pruning** — reading only needed columns.
- **Predicate pushdown** — filtering at the source.
- **Compaction** — merging small files.
- **Small-file problem** — many tiny files slowing scans.
- **Cardinality** — the number of distinct values.
- **Selectivity** — the fraction of rows passing a filter.
- **Skew mitigation** — rebalancing uneven data.
- **Broadcast join** — sending a small table to all nodes.
- **Cache** — fast temporary storage.
- **Cold start** — delay on first use.
- **Data contract test** — a check of schema and semantics.
- **Dependency graph** — the relationships between tasks.
- **Ephemeral environment** — a temporary environment.
- **Feature parity** — equal features in training and serving.
- **Golden dataset** — a fixed reference dataset.
- **Ingestion lag** — delay before data are available.
- **Job dependency** — a task that must finish first.
- **Model version** — an identified model release.
- **Parquet predicate** — a filter pushed to the file scan.
- **Pipeline test** — a test of the end-to-end flow.
- **Query latency** — time to return a query result.
- **Resource quota** — a limit on compute or storage.
- **Schema evolution** — changing a schema over time.
- **Service level agreement** — a contractual reliability commitment.
- **Snapshot** — a point-in-time copy.
- **Staging data** — intermediate data for a load.
- **Storage tier** — a cost and performance class.
- **Table statistics** — metadata for query planning.
- **Task retry** — re-running a failed task.
- **Telemetry pipeline** — collecting operational data.
- **Time travel** — querying a previous table version.
- **Uptime target** — a reliability goal.
- **Workload manager** — software scheduling jobs.
- **Write amplification** — extra writes from updates.
- **Zero-copy** — sharing data without copying.
- **Column store** — storage organised by column.
- **Row store** — storage organised by row.
- **Data mart** — a subset of a warehouse.
- **Semantic layer** — a business definition layer.

## 16. Sources

All URLs accessed 26 October 2026. Licences and project governance change; verify at the project.

1. DuckDB. https://duckdb.org/
2. PostgreSQL. https://www.postgresql.org/
3. dbt. https://www.getdbt.com/
4. Apache Airflow. https://airflow.apache.org/
5. Dagster. https://dagster.io/
6. Prefect. https://www.prefect.io/
7. Apache Spark. https://spark.apache.org/
8. Polars. https://pola.rs/
9. pandas. https://pandas.pydata.org/
10. scikit-learn. https://scikit-learn.org/
11. XGBoost. https://xgboost.readthedocs.io/
12. PyTorch. https://pytorch.org/
13. MONAI. https://monai.io/
14. OpenSlide. https://openslide.org/
15. Hugging Face Transformers. https://huggingface.co/docs/transformers/
16. spaCy. https://spacy.io/
17. WFDB. https://physionet.org/content/wfdb-python/
18. MNE. https://mne.tools/
19. statsmodels. https://www.statsmodels.org/
20. lifelines. https://lifelines.readthedocs.io/
21. DoWhy. https://py-why.github.io/dowhy/
22. MLflow. https://mlflow.org/
23. DVC. https://dvc.org/
24. Great Expectations. https://greatexpectations.io/
25. OpenDP. https://opendp.org/
26. SPDX. https://spdx.dev/
27. SLSA. https://slsa.dev/
28. OpenSSF. https://openssf.org/
29. BentoML. https://www.bentoml.com/
30. Triton Inference Server. https://developer.nvidia.com/triton-inference-server
31. ONNX Runtime. https://onnxruntime.ai/
32. Evidently. https://www.evidentlyai.com/
33. Fairlearn. https://fairlearn.org/
34. Module 21 survey file, section 16. [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)

## 17. Links

- Module map and topic table: [`_index.md`](_index.md)
- Previous topic, pharma and safety datasets: [`10_pharma-and-safety-datasets.md`](10_pharma-and-safety-datasets.md)
- Public health datasets: [`09_public-health-datasets.md`](09_public-health-datasets.md)
- Access and licensing: [`02_access-and-licensing.md`](02_access-and-licensing.md)
- Root survey file: [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)
- Labs, including the tool stack bake-off: [`../../labs/`](../../labs/)

