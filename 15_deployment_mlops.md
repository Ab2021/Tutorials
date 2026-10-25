# Module 15 — Deployment and MLOps

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 28 hours (10 lecture, 4 case clinic, 14 lab)
> **Track fit:** Common core; required for MedTech and product roles
> **Prerequisites:** Modules 03, 06, 14
> **Tools:** Docker, FastAPI, MLflow, DVC, Kubernetes or a managed service, Evidently, Prometheus/Grafana
> **Industry question:** "Can this model run reliably in the real workflow, and will we know when it stops working?"

---

## 1. Why industry cares

Most clinical AI never reaches durable production. The reasons are rarely algorithmic:

- The model is not integrated into the workflow.
- Latency is too high.
- The interface is confusing.
- The model is not monitored.
- A data feed changes.
- A vendor updates a model.
- Nobody owns incidents.
- The business case disappears after the pilot.

MLOps is the discipline of keeping a model useful after the demo.

## 2. Learning outcomes

By the end of this module, you can:

- Choose a deployment pattern that matches the decision and workflow.
- Package a model with reproducible dependencies and versioned artifacts.
- Build a CI/CD pipeline with tests and approvals.
- Design monitoring for data quality, drift, calibration, performance, fairness, and operations.
- Write a runbook and incident response plan.
- Design a shadow-mode and phased rollout.
- Implement rollback and change control.
- Estimate latency, throughput, and cost.
- Align deployment with regulatory change-control obligations.

## 3. Mental model: the production loop

```
Data → Features → Model → Prediction → Workflow → Action → Outcome
  ↑        ↑         ↑          ↑           ↑          ↑
Monitor  Validate  Version   Explain     Integrate   Measure
  └────────┴─────────┴──────────┴───────────┴──────────┘
                     Feedback and retraining
```

## 4. Deployment patterns

For model compression and on-device inference constraints, see Module 14B. This module assumes the model has already met its target-hardware budget.

| Pattern | Description | Latency | Use |
|---|---|---|---|
| Batch | Score population on schedule | Minutes–days | Population health, claims |
| Real-time API | Score on request | Milliseconds–seconds | EHR alert, triage |
| Streaming | Score events continuously | Seconds | ICU monitoring, wearables |
| Edge | Score on device | Milliseconds | Wearables, imaging devices |
| Embedded | Inside EHR/device | Workflow-dependent | Clinical decision support |
| Shadow | Score silently, no action | Any | Pre-deployment validation |
| Human-in-loop | Model recommends, human decides | Any | High-stakes decisions |
| Federated | Model travels to data | Varies | Multi-site privacy |

**Rule:** the deployment pattern must match the decision's latency and the workflow's tolerances.

## 5. Packaging and reproducibility

- Containerize the model and preprocessing.
- Pin dependencies and base images.
- Version model weights, config, and training data hash.
- Store the model card and intended use with the artifact.
- Use a model registry with stage transitions.
- Separate code, config, and secrets.
- Test on a clean environment.

## 6. CI/CD/CT

| Stage | Checks |
|---|---|
| CI | Unit tests, schema tests, linting, security scan |
| CD | Build, sign, deploy to staging |
| CT | Training reproducibility, data validation, model evaluation |
| Approval | Clinical, regulatory, security sign-off |
| Release | Canary or shadow, monitoring, rollback |

**Regulated context:** automated deployment may not be appropriate. Human approval and documented change control are often required.

## 7. Monitoring

### 7.1 Dimensions

| Dimension | Metrics | Alert example |
|---|---|---|
| Availability | Uptime, error rate, latency | p95 latency > 2s |
| Data quality | Missingness, schema, ranges | Critical lab missing > 5% |
| Drift | PSI, KS, Jensen-Shannon | Feature PSI > 0.2 |
| Performance | AUROC, calibration, Brier | Calibration slope < 0.8 |
| Fairness | Subgroup performance | Gap > pre-specified threshold |
| Operations | Alert volume, override rate | Alerts/1,000 patients > 50 |
| Outcomes | Time to treatment, readmission | Unexplained shift |
| Cost | Inference, storage, review time | Budget variance |

### 7.2 Drift types

- **Covariate shift:** input distribution changes.
- **Label shift:** outcome prevalence changes.
- **Concept drift:** relationship between inputs and outcome changes.
- **Upstream data change:** coding, device, or workflow change.

### 7.3 Monitoring without labels

Clinical outcomes arrive late. Use proxy metrics:

- Input drift.
- Prediction distribution.
- Calibration on available outcomes.
- Clinician override rate.
- Alert burden.
- Data completeness.

### 7.4 Drift taxonomy and worked thresholds

| Drift type | What changes | Detection | Response |
|---|---|---|---|
| Covariate | Input distribution | PSI, KS, Jensen-Shannon | Investigate source; recalibrate |
| Label | Outcome prevalence | Label monitoring; audit sample | Recalibrate; reweight |
| Concept | Input–outcome relationship | Performance on delayed labels | Retrain or revise |
| Upstream schema | Coding, units, feed | Schema tests, range checks | Fix pipeline; rollback |
| Device/firmware | Sensor or algorithm version | Device metadata, version mix | Stratify; revalidate |
| Population | Case mix, site, access | Subgroup and site monitoring | Local validation |
| Workflow | Alert use, staffing | Override rate, response time | Human-factors review |

**Thresholds are context-specific.** A PSI above 0.2, a calibration slope below 0.8, or a subgroup gap above a pre-specified limit can trigger review, but none is a universal rule. Pre-specify thresholds and pair them with a response plan.

### 7.5 Delayed labels and proxy monitoring

Clinical outcomes arrive late. Use proxies with known limitations:

- Prediction distribution and score entropy.
- Alert volume and override rate.
- Data completeness and timing.
- Calibration on a scheduled audit sample.
- Complaint and safety-event rates.
- Downstream actions (orders, referrals, escalations).

**Rule:** a proxy is not the outcome. State what it can and cannot detect.

### 7.6 Device fleet and time synchronization

For connected devices, monitor:

- Firmware and model version by device.
- Battery, thermal, and connectivity state.
- Sensor calibration and lot.
- Clock offset and time zone.
- Signal-quality flags.
- OTA update success and rollback.

**Interfaces:** IEEE 11073, IHE PCD, HL7 v2/FHIR `Device` and `DeviceMetric`, Bluetooth GATT, DICOM. Log device time and EHR time separately; NTP/PTP drift can look like clinical change.

## 8. Incident response

A runbook should include:

1. Detection and severity.
2. On-call owner.
3. Triage steps.
4. Rollback procedure.
5. Communication plan.
6. Regulatory notification assessment.
7. Root-cause analysis.
8. Corrective action.
9. Post-incident review.

**Rule:** a model incident is a clinical safety incident until proven otherwise.

### 8.1 Worked post-mortem: a silent false-negative increase

A sepsis model's alert volume fell by 30% over two weeks. The team initially celebrated. A scheduled audit found sensitivity had dropped 12 points after a lab-feed change: a high-frequency lab was now arriving after the model's feature cutoff. Root cause: upstream interface timing, not model drift. Corrective actions: contract test on arrival time, feature-cutoff monitor, backfill and revalidation, clinician communication. The lesson: **monitor inputs and outcomes, not only alerts.**

### 8.2 Rollback drill

Quarterly, restore the previous model, feature set, and configuration in staging; verify predictions, latency, and audit logs; measure time to restore; update the runbook.
## 9. Rollout

- Offline validation.
- Shadow mode.
- Limited site or unit.
- Stepped-wedge or randomized rollout.
- Full deployment.
- Continuous monitoring.

**Stop criteria:** pre-specified performance, safety, or equity thresholds.

### 9.1 Cost model

Total cost includes inference, storage, integration, clinical review, monitoring, on-call, validation per model change, and retirement. Estimate cost per prediction and cost per quality-adjusted outcome where possible. A model that is cheap to train can be expensive to maintain.

### 9.2 PCCP-aligned change control

| Change | PCCP-covered? | Evidence |
|---|---|---|
| Retrain on new data within pre-specified protocol | Possibly | Protocol, impact assessment, acceptance criteria |
| New input feature | Usually not | New submission or supplement |
| New population or indication | No | New clinical evidence |
| Threshold change | Possibly | Risk analysis, local validation |
| Cybersecurity patch | Usually controlled | Security testing, change record |

**Rule:** if the change is not described in the PCCP, assume it needs regulatory review.

## 9A. Reference production stack

The course ships a small production-shaped service in `labs/mlops/`.

```
labs/mlops/
├── app/
│   ├── main.py       # FastAPI: /healthz /readyz /predict /metrics
│   ├── model.py      # deterministic logistic model
│   ├── schemas.py    # Pydantic input/output
│   └── model.json    # versioned coefficients
├── tests/test_api.py
├── Dockerfile
├── requirements.txt
├── train.py
├── monitoring/
│   ├── drift.py
│   └── prometheus-rules.yaml
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── .github/workflows/ci.yml
├── RUNBOOK.md
└── model_card.md
```

### 9A.1 Service contract

| Endpoint | Purpose | Failure behavior |
|---|---|---|
| `/healthz` | Process alive | 200 unless process dead |
| `/readyz` | Model loaded and ready | 503 if model missing |
| `/predict` | Risk score/decision | 422 on invalid input; 503 if not ready |
| `/metrics` | Prometheus text | Counts, status codes, latency sum/count |

The service logs a structured event without PHI, returns the model version, applies a threshold, and separates `routine` from `review`. It is a teaching model, not a clinical device.

### 9A.2 Build and test

```bash
cd labs/mlops
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
docker build -t biomed-risk:0.1.0 .
```

The CI workflow runs tests and builds the image on every push. A real pipeline adds dependency scanning, image signing, SBOM generation, and a registry push with immutable tags.

### 9A.3 Kubernetes deployment

The manifests add non-root execution, read-only root filesystem, probes, resource limits, HPA, and a service. Apply after replacing the image:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl rollout status deploy/risk-service -n biomed
```

**Acceptance:** readiness blocks traffic until the model loads; liveness restarts a hung process; rollout leaves at least one ready pod; HPA scales on CPU.

### 9A.4 Monitoring and drift

`monitoring/prometheus-rules.yaml` includes error-rate, p99-latency, and no-prediction alerts. `monitoring/drift.py` computes PSI for one numeric feature:

```bash
python monitoring/drift.py baseline.csv current.csv systolic_bp
```

A PSI above 0.2 triggers review, not automatic retraining. Pair it with calibration and subgroup checks; drift is not harm.

### 9A.5 Rollback

```bash
kubectl rollout undo deployment/risk-service -n biomed
kubectl rollout status deployment/risk-service -n biomed
```

`RUNBOOK.md` covers triage, communication, rollback, and post-incident review. A model rollback must restore the image and model artifact together; a config-only rollback can create an incompatible pair.

### 9A.6 What is still missing

This is a scaffold, not a complete enterprise platform. It does not include:

- a model registry with signed artifacts;
- a feature store with point-in-time correctness;
- a real database or event bus;
- OIDC, TLS, secret manager, or service mesh;
- multi-region backup and tested RTO/RPO;
- HIPAA/BAA review or clinical validation.

Those are the next production engineering steps, not model-tuning steps.

## 10. Industry case studies

### 10.1 Epic Sepsis Model

The model was widely deployed, but external validation found lower performance and high alert burden.

**Lesson:** deployment without local validation and monitoring is not evidence.

Source: Wong, A. et al., *JAMA Internal Medicine*, 2021. https://doi.org/10.1001/jamainternmed.2021.2626

### 10.2 IDx-DR

IDx-DR was deployed in primary care with a defined workflow and image-quality checks.

**Lesson:** workflow and image-quality controls are part of the product.

Source: FDA De Novo DEN180001. https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf

### 10.3 NHS AI deployment

NHS AI deployments have highlighted the importance of local validation, integration, and workforce support.

**Lesson:** adoption is sociotechnical, not just technical.

Source: NHS England AI guidance. https://www.england.nhs.uk/ai/

### 10.4 Model updates and drift

Post-market model updates can change performance and risk. Regulators expect change control, not silent updates.

**Lesson:** treat model updates as controlled changes.

Source: FDA PCCP guidance. https://www.fda.gov/regulatory-information/search-fda-guidance-documents

## 11. Labs

### Lab 15.1 — Package and serve a model

**Objective:** Containerize a model and expose a prediction API.

**Steps:**

1. Build a Docker image.
2. Add health and prediction endpoints.
3. Validate input schema.
4. Log metadata without PHI.
5. Load-test.

**Deliverable:** container and API.

**Acceptance criteria:** reproducible build; schema validation; no PHI logs.

### Lab 15.2 — CI pipeline

**Objective:** Add tests and security scans.

**Steps:**

1. Unit and schema tests.
2. Lint and type check.
3. Dependency vulnerability scan.
4. Build artifact.
5. Fail on thresholds.

**Deliverable:** CI pipeline.

**Acceptance criteria:** deterministic; fast; catches planted errors.

### Lab 15.3 — Monitoring dashboard

**Objective:** Monitor data quality, drift, and performance.

**Steps:**

1. Compute PSI/KS for key features.
2. Track prediction distribution.
3. Track calibration on available labels.
4. Track alert volume.
5. Add alert thresholds and owners.

**Deliverable:** dashboard and alert rules.

**Acceptance criteria:** actionable; no alert fatigue; runbook linked.

### Lab 15.4 — Shadow deployment

**Objective:** Run a model silently for 2 weeks of simulated data.

**Steps:**

1. Log predictions.
2. Compare with current practice.
3. Measure calibration and alert burden.
4. Analyze subgroup performance.
5. Decide go/no-go.

**Deliverable:** shadow evaluation.

**Acceptance criteria:** pre-specified metrics; no clinical action.

### Lab 15.5 — Incident simulation

**Objective:** Respond to a model outage or drift event.

**Steps:**

1. Detect.
2. Triage.
3. Roll back.
4. Communicate.
5. Assess regulatory reporting.
6. Write post-incident review.

**Deliverable:** incident report.

**Acceptance criteria:** timeline; root cause; corrective action.

### Lab 15.6 — Device fleet and OTA rollout

**Objective:** Plan a signed firmware/model update across a device fleet.

**Steps:**

1. Inventory firmware and model versions.
2. Define canary and rollback criteria.
3. Simulate update failure and power loss.
4. Verify audit logs and version reporting.
5. Write a patient-safety communication plan.

**Deliverable:** OTA rollout plan.

**Acceptance criteria:** signed update; rollback tested; no device bricking.

### Lab 15.7 — Drift taxonomy and response

**Objective:** Diagnose three simulated drift events.

**Steps:**

1. Inspect feature distributions and schema.
2. Check calibration and subgroup performance.
3. Distinguish covariate, label, concept, and upstream drift.
4. Choose a response and owner.
5. Write the monitoring ticket.

**Deliverable:** drift diagnosis.

**Acceptance criteria:** correct taxonomy; response proportional; no retraining reflex.

## 12. Assignment: deployment and monitoring plan

Write a 2,000-word plan:

1. Deployment pattern.
2. Architecture diagram.
3. Reproducibility and versioning.
4. CI/CD/CT.
5. Monitoring metrics and thresholds.
6. Alerting and ownership.
7. Rollout and stop criteria.
8. Incident response.
9. Change control.
10. Cost and capacity.

## 13. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Architecture | 20% | Notebook only | Service with tests | Workflow-integrated and resilient |
| Reproducibility | 20% | Manual steps | Container and versions | Signed, auditable artifacts |
| Monitoring | 25% | Uptime only | Drift and performance | Fairness, outcomes, alert burden |
| Operations | 20% | No runbook | Basic incident plan | Tested rollback and post-incident review |
| Governance | 15% | Ignores change control | Documents changes | PCCP-aligned lifecycle |

## 14. Regulatory, ethical, and commercial lens

- **Change control:** model updates may require regulatory review.
- **Cybersecurity:** production systems are attack surfaces.
- **Audit:** logs must support investigation without exposing PHI.
- **Equity:** monitoring must include subgroup performance.
- **Commercial:** maintenance is a recurring cost; price it.

## 15. Failure modes and red flags

- No monitoring.
- No owner.
- No rollback.
- Silent model updates.
- Logging PHI.
- Ignoring alert burden.
- Training-serving skew.
- Unpinned dependencies.
- No shadow period.
- No stop criteria.

## 16. Interview and discussion questions

1. **What is training-serving skew?**
   Hint: preprocessing or feature computation differs in production.
2. **How do you monitor a model without labels?**
   Hint: input drift, prediction distribution, override rate, proxy outcomes.
3. **What is shadow mode?**
   Hint: silent scoring before clinical use.
4. **How do you roll back a model?**
   Hint: versioned artifacts, feature compatibility, deployment switch.
5. **What is a model registry?**
   Hint: versioned artifacts with stage and metadata.
6. **How do you handle a data feed change?**
   Hint: contract tests, schema validation, canary, rollback.
7. **What is concept drift?**
   Hint: input-output relationship changes.
8. **How do you set alert thresholds?**
   Hint: clinical impact, baseline variability, false alarm cost.
9. **What belongs in a runbook?**
   Hint: detection, triage, rollback, communication, regulatory assessment.
10. **How do you deploy a model in an EHR?**
    Hint: integration point, latency, alert design, permissions, audit.
11. **What is the total cost of ownership?**
    Hint: inference, integration, monitoring, review, maintenance.
12. **When should you stop a deployment?**
    Hint: pre-specified safety, performance, or equity thresholds.

## 17. Key terms

| Term | Definition |
|---|---|
| MLOps | Machine learning operations |
| CI/CD | Continuous integration/delivery |
| CT | Continuous training |
| Model registry | Versioned model artifact store |
| Feature store | Centralized feature definitions |
| Shadow mode | Silent scoring |
| Canary release | Limited early deployment |
| Drift | Change in data or relationships |
| PSI | Population stability index |
| Runbook | Operational incident procedure |
| Rollback | Revert to previous version |
| Training-serving skew | Difference between training and production preprocessing |

## 18. Sources and further reading

- FDA, *PCCP guidance*. https://www.fda.gov/regulatory-information/search-fda-guidance-documents
- FDA, *Cybersecurity*. https://www.fda.gov/medical-devices/digital-health-center-excellence/cybersecurity
- NHS England, *AI*. https://www.england.nhs.uk/ai/
- MLflow: https://mlflow.org/
- Evidently: https://www.evidentlyai.com/
- DVC: https://dvc.org/
- Kubernetes: https://kubernetes.io/
- FastAPI: https://fastapi.tiangolo.com/
- Prometheus: https://prometheus.io/
- Grafana: https://grafana.com/

> **Next:** [Module 16](16_privacy_security_ethics.md) addresses the privacy, security, and ethical controls that make deployment acceptable.
