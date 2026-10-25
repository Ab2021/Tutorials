# Module 26 — Cloud, Kubernetes, and Streaming Production Labs

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 24 hours (8 lecture, 4 case clinic, 12 lab)
> **Track fit:** Health Systems and MedTech advanced elective
> **Prerequisites:** Modules 03, 15, 16
> **Tools:** Docker, Kubernetes, Terraform, Kafka or Redpanda, PostgreSQL/DuckDB, FastAPI, Prometheus/Grafana, a cloud account or local kind/minikube
> **Industry question:** "Can this pipeline run securely, observably, and affordably when the data arrive continuously and the network is not reliable?"

---

## 1. Why this module exists

Most biomedical data science is taught batch-first: a CSV, a notebook, a model, a report. Production is different:

- Data arrive as events, not files.
- Services fail partially.
- Clocks drift.
- Networks drop.
- Models and schemas change.
- Costs grow silently.
- Security reviews block the launch.

This module builds the production environment skills that employers screen for: containers, Kubernetes, infrastructure as code, streaming, observability, security, cost, and disaster recovery.

## 2. Learning outcomes

By the end of this module, you can:

- Design a cloud architecture for batch, real-time, and streaming biomedical workloads.
- Containerize a model or pipeline and deploy it to Kubernetes.
- Provision infrastructure with Terraform.
- Ingest FHIR/HL7 events and device telemetry with a streaming platform.
- Implement point-in-time-correct features.
- Instrument services with metrics, logs, traces, and alerts.
- Design for security, secrets, and least privilege.
- Estimate and control cloud cost.
- Plan backup, disaster recovery, and incident response.
- Run a production readiness review.

## 3. Architecture patterns

| Pattern | Use | Trade-off |
|---|---|---|
| Batch warehouse | Claims, nightly EHR extracts | Stale data |
| Streaming | ICU monitors, wearables, ADT events | Complexity, ordering |
| Event-driven | FHIR subscriptions, device alerts | Idempotency, replay |
| Feature store | Reusable point-in-time features | Operational overhead |
| Model service | Real-time prediction | Latency, availability |
| Data lakehouse | Multimodal storage | Governance |
| Federated | Multi-site privacy | Heterogeneity |
| Edge + cloud | Device inference + cohort analytics | Sync, versioning |

### 3.1 Reference architecture

```
Sources → Ingestion → Raw/landing → Validation → Standardization → Feature store
   → Model service → API/events → EHR/app → Monitoring → Audit
```

### 3.2 Cloud building blocks

| Capability | AWS | Azure | GCP |
|---|---|---|---|
| Object storage | S3 | Blob Storage | Cloud Storage |
| Warehouse | Redshift | Synapse | BigQuery |
| Managed Kafka | MSK | Event Hubs | Pub/Sub |
| Kubernetes | EKS | AKS | GKE |
| Secrets | Secrets Manager | Key Vault | Secret Manager |
| Monitoring | CloudWatch | Monitor | Cloud Monitoring |
| IaC | Terraform/CDK | Bicep/Terraform | Terraform |

**Rule:** use managed services when they reduce operational risk; do not build a distributed system to avoid a managed queue.

## 4. Containers and Kubernetes

### 4.1 Container checklist

- Pin base image and dependencies.
- Run as non-root.
- Read-only filesystem where possible.
- Health/readiness endpoints.
- Resource requests and limits.
- No secrets in the image.
- SBOM and image scanning.
- Graceful shutdown.
- Structured logs without PHI.

### 4.2 Kubernetes objects

| Object | Purpose |
|---|---|
| Deployment | Stateless service |
| StatefulSet | Stateful service |
| Service | Stable network endpoint |
| Ingress | External routing |
| ConfigMap | Non-secret config |
| Secret | Sensitive config |
| Job/CronJob | Batch |
| HPA | Autoscaling |
| PodDisruptionBudget | Availability |
| NetworkPolicy | Segmentation |

### 4.3 Production readiness

- Liveness/readiness probes.
- Resource limits.
- Rolling update and rollback.
- Pod disruption budget.
- Horizontal scaling.
- Persistent storage and backup.
- Secrets rotation.
- Network policy.
- Audit logging.
- Cost labels.

## 5. Infrastructure as code

Terraform pattern:

```hcl
resource "aws_s3_bucket" "raw" {
  bucket = "biomed-raw-${var.env}"
}

resource "aws_kms_key" "phi" {
  description = "PHI encryption"
}
```

**Rules:**

- Remote state with locking.
- Separate environments.
- No secrets in state.
- Plan before apply.
- Review diffs.
- Tag for cost and ownership.
- Destroy safely.

## 6. Streaming

### 6.1 Concepts

- Topic, partition, offset.
- Producer/consumer.
- Consumer group.
- At-least-once vs exactly-once.
- Event time vs processing time.
- Watermark.
- Idempotency.
- Replay.
- Schema registry.
- Dead-letter queue.

### 6.2 Clinical streams

| Stream | Example | Risk |
|---|---|---|
| ADT | Admit/discharge/transfer | Out-of-order, duplicates |
| FHIR subscription | Resource changes | Versioning, delivery gaps |
| Device telemetry | ECG/PPG/pump | Loss, clock drift, battery |
| Labs | Results | Late arrival, units |
| Claims | 837/835 | Batch, lag |
| Alerts | Sepsis/deterioration | Alert fatigue |

### 6.3 Exactly-once is a system property

Exactly-once processing requires idempotent writes, deduplication keys, transactional sinks, and replay-safe logic. A broker setting alone does not guarantee it.

## 7. Observability

| Signal | Use |
|---|---|
| Metrics | Latency, error rate, saturation, throughput |
| Logs | Debugging, audit, security |
| Traces | Cross-service latency |
| Health checks | Availability |
| Data quality metrics | Freshness, completeness, schema |
| Model metrics | Calibration, drift, alert burden |
| Cost metrics | Spend by service/team |

**Rule:** never log PHI. Log identifiers, counts, hashes, and version numbers instead.

## 8. Security

- Least privilege.
- MFA and short-lived credentials.
- Encryption at rest/in transit.
- Private networking.
- Secrets manager.
- Image signing and SBOM.
- Network policies.
- Audit logs.
- Vulnerability scanning.
- Incident response.
- BAA/DPA with cloud vendors.

## 9. Cost and reliability

- Label all resources.
- Set budgets and alerts.
- Right-size requests.
- Autoscale with limits.
- Use spot/preemptible for batch.
- Archive cold data.
- Monitor egress.
- Estimate cost per prediction.
- Define RTO/RPO.
- Test restore.

## 9A. Reference implementation: a working stack

The course ships a runnable teaching stack in `labs/cloud/`. It is deliberately small enough to inspect end to end.

```
labs/cloud/
├── terraform/
│   ├── versions.tf, variables.tf, main.tf, outputs.tf
│   └── modules/
│       ├── network/   # VPC, private/public subnets, NAT
│       ├── storage/   # KMS, S3 raw bucket, ECR
│       └── eks/       # EKS cluster + managed node group
├── k8s/
│   ├── 00-namespace.yaml
│   ├── 01-configmap.yaml
│   ├── 01a-model-configmap.yaml
│   ├── 02-secret.example.yaml
│   ├── 03-deployment.yaml
│   ├── 04-service.yaml
│   ├── 05-ingress.yaml
│   ├── 06-hpa.yaml
│   ├── 07-pdb.yaml
│   ├── 08-networkpolicy.yaml
│   └── 09-serviceaccount.yaml
└── kafka/
    ├── schemas.py
    ├── producer.py
    ├── consumer.py
    ├── events.jsonl
    └── docker-compose.yml
```

### 9A.1 Terraform walkthrough

The root composes three modules:

```hcl
module "network" { source = "./modules/network" ... }
module "storage" { source = "./modules/storage" ... }
module "eks"     { source = "./modules/eks" ... }
```

The storage module creates a KMS key, versioned S3 bucket with public access blocked, and an immutable ECR repository with scan-on-push. The network module creates three AZs, private/public subnets, and a single NAT gateway for a lab. The EKS module creates a managed node group.

Run:

```bash
cd labs/cloud/terraform
terraform init
terraform plan -var="project=biomed" -var="env=dev"
terraform apply
```

**Acceptance:** `terraform output raw_bucket` exists; the bucket blocks public access; the ECR repo has scan-on-push; `kubectl get nodes` works after updating kubeconfig.

**Cost warning:** this creates billable AWS resources. Use a sandbox account and destroy after the lab. The single NAT gateway is a cost compromise, not a production HA design.

### 9A.2 Kubernetes walkthrough

Apply in order:

```bash
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-configmap.yaml
kubectl apply -f k8s/01a-model-configmap.yaml
kubectl apply -f k8s/02-secret.example.yaml   # replace placeholders
kubectl apply -f k8s/03-deployment.yaml
kubectl apply -f k8s/04-service.yaml
kubectl apply -f k8s/05-ingress.yaml
kubectl apply -f k8s/06-hpa.yaml
kubectl apply -f k8s/07-pdb.yaml
kubectl apply -f k8s/08-networkpolicy.yaml
kubectl apply -f k8s/09-serviceaccount.yaml
```

The deployment includes non-root execution, read-only root filesystem, dropped capabilities, readiness/liveness probes, resource requests/limits, preStop drain, topology spread, HPA, PDB, and default-deny network policy with explicit ingress/egress.

**Acceptance:** `kubectl rollout status deploy/risk-service -n biomed`; `/healthz` and `/readyz` return 200; deleting a pod does not drop below one replica; HPA scales under load; the pod cannot reach unapproved egress.

### 9A.3 Kafka walkthrough

The local stack uses Redpanda:

```bash
cd labs/cloud
make up
make produce
make consume
```

The producer validates every event with Pydantic, keys by `patient_id` for per-patient ordering, enables idempotence, uses `acks=all`, and attaches a schema-version header. The consumer commits after processing, deduplicates by `event_id` in SQLite, and routes invalid events to `adt.events.dlq`.

Key consumer logic:

```python
try:
    event = AdtEvent.model_validate_json(raw)
except ValidationError as exc:
    dlq.produce(args.dlq, key=msg.key(), value=raw.encode())
    consumer.commit(message=msg, asynchronous=False)
    continue

try:
    db.execute("INSERT INTO processed(event_id) VALUES (?)", (event.event_id,))
    db.commit()
except sqlite3.IntegrityError:
    consumer.commit(message=msg, asynchronous=False)  # replay-safe
    continue
```

**Acceptance:** produce six events, consume them, replay the same file, and verify no duplicate processing. Send one malformed event and verify it lands in the DLQ.

### 9A.4 What the stack does not prove

It does not prove HIPAA compliance, production availability, multi-region disaster recovery, or clinical safety. It is a lab scaffold: replace example secrets, add TLS/OIDC, use a managed database and schema registry, add real backup/restore tests, and perform security review before any real data.

## 10. Labs

### Lab 27.1 — Containerize and deploy

**Objective:** Deploy a model API to Kubernetes.

**Steps:** build image, non-root, probes, limits, service, ingress, HPA; test rollout/rollback.

**Deliverable:** manifests and load test.

**Acceptance criteria:** reproducible; no secrets; rollback works.

### Lab 27.2 — Terraform infrastructure

**Objective:** Provision storage, a database, and a queue.

**Steps:** modules, state backend, plan/apply, tags, destroy.

**Deliverable:** Terraform repo.

**Acceptance criteria:** no secrets in state; plan reviewed; reproducible.

### Lab 27.3 — Streaming ingestion

**Objective:** Ingest synthetic ADT/device events.

**Steps:** topic, producer, consumer, schema, dead-letter queue, replay.

**Deliverable:** streaming pipeline.

**Acceptance criteria:** idempotent; handles late/duplicate events.

### Lab 27.4 — FHIR subscription pipeline

**Objective:** Process FHIR resource changes.

**Steps:** subscribe, validate, map, store, handle deletions/versions.

**Deliverable:** FHIR event pipeline.

**Acceptance criteria:** no silent drops; versioning; audit.

### Lab 27.5 — Point-in-time feature store

**Objective:** Build features without leakage.

**Steps:** event-time joins, as-of timestamps, offline/online parity, backfill.

**Deliverable:** feature pipeline.

**Acceptance criteria:** no future data; reproducible backfill.

### Lab 27.6 — Observability

**Objective:** Instrument a service.

**Steps:** metrics, logs, traces, dashboards, alerts, runbook.

**Deliverable:** dashboard and alert rules.

**Acceptance criteria:** actionable; no PHI; p99 latency.

### Lab 27.7 — Cost and autoscaling

**Objective:** Control cost under load.

**Steps:** load test, HPA, budget, cost per request, spot for batch.

**Deliverable:** cost report.

**Acceptance criteria:** SLO met; cost bounded.

### Lab 27.8 — Incident and disaster recovery

**Objective:** Recover from failure.

**Steps:** simulate zone failure, restore backup, measure RTO/RPO, post-mortem.

**Deliverable:** DR report.

**Acceptance criteria:** restore tested; data integrity; communication.

## 11. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Architecture | 20% | Single VM | Managed services | Cost, security, resilience |
| Kubernetes | 20% | Pod only | Probes, limits, rollout | HPA, PDB, network policy |
| Streaming | 20% | Batch only | Idempotent consumer | Replay, schema, exactly-once |
| Observability | 15% | Logs only | Metrics and alerts | Traces, SLOs, runbooks |
| Security | 15% | Basic | Least privilege, secrets | Threat model and audit |
| Cost/DR | 10% | Ignores | Budget and backup | Tested RTO/RPO |

## 12. Failure modes

- Secrets in Git.
- `latest` image tags.
- No resource limits.
- No probes.
- No DLQ.
- No idempotency.
- Logging PHI.
- No cost labels.
- No restore test.
- Single-zone production.

## 13. Interview questions

1. **Why is exactly-once hard?**
   Hint: idempotency, transactions, replay.
2. **How do you prevent feature leakage in streaming?**
   Hint: event time, as-of joins.
3. **What is a dead-letter queue?**
   Hint: failed messages for inspection/replay.
4. **How do you secure a Kubernetes cluster?**
   Hint: RBAC, network policy, secrets, image scanning.
5. **What is an SLO?**
   Hint: target reliability/latency.
6. **How do you control cloud cost?**
   Hint: labels, budgets, right-sizing, autoscaling.
7. **What is RTO/RPO?**
   Hint: recovery time/point objectives.
8. **How do you handle late data?**
   Hint: watermarks, replay, reprocessing.
9. **Why log without PHI?**
   Hint: privacy, audit, breach risk.
10. **When would you use batch instead of streaming?**
    Hint: latency requirement, complexity, cost.

## 14. Key terms

| Term | Definition |
|---|---|
| Kubernetes | Container orchestration |
| Terraform | Infrastructure as code |
| Kafka | Distributed event streaming |
| Topic/partition | Stream organization |
| Consumer group | Parallel consumers |
| Idempotency | Safe retry |
| DLQ | Dead-letter queue |
| Watermark | Late-event threshold |
| Feature store | Point-in-time feature service |
| SLO/SLI | Objective/indicator |
| RTO/RPO | Recovery time/point objective |
| HPA | Horizontal pod autoscaler |
| PDB | Pod disruption budget |
| SBOM | Software bill of materials |

## 15. Sources

- Kubernetes. https://kubernetes.io/docs/
- Terraform. https://developer.hashicorp.com/terraform/docs
- Apache Kafka. https://kafka.apache.org/documentation/
- Redpanda. https://docs.redpanda.com/
- FHIR subscriptions. https://hl7.org/fhir/subscription.html
- OpenTelemetry. https://opentelemetry.io/docs/
- Prometheus. https://prometheus.io/docs/
- Grafana. https://grafana.com/docs/
- AWS HIPAA. https://aws.amazon.com/compliance/hipaa-compliance/
- Azure HIPAA. https://learn.microsoft.com/en-us/azure/compliance/offerings/offering-hipaa-us
- GCP HIPAA. https://cloud.google.com/security/compliance/hipaa
- NIST Cybersecurity Framework. https://www.nist.gov/cyberframework

> **Previous:** [`25_trial_operations_hta_global_regulatory.md`](25_trial_operations_hta_global_regulatory.md)
>
> **End of advanced electives.** Return to `README.md` or choose a capstone brief in Module 19.
