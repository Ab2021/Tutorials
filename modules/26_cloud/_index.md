# Module 26 — Cloud, Kubernetes, and Streaming Production Labs

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../26_cloud_streaming_production_labs.md`](../../26_cloud_streaming_production_labs.md)
> **Folder:** `modules/26_cloud/`
> **Version:** 2026.3

---

## 1. Module purpose

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 03, 15 |
| Unlocks | none |
| Assessment | Production readiness review |
| Survey file | `26_cloud_streaming_production_labs.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_architecture-patterns.md`](01_architecture-patterns.md) | Architecture Patterns | Reference architectures for clinical analytics and streaming. | 1,200-1,800 | planned |
| 2 | [`02_containers.md`](02_containers.md) | Containers | Images, layers, security scanning, and reproducible builds. | 1,200-1,800 | planned |
| 3 | [`03_kubernetes.md`](03_kubernetes.md) | Kubernetes | Deployments, services, probes, autoscaling, and network policy. | 1,200-1,800 | planned |
| 4 | [`04_infrastructure-as-code.md`](04_infrastructure-as-code.md) | Infrastructure as Code | Terraform modules, state, and environment promotion. | 1,200-1,800 | planned |
| 5 | [`05_streaming.md`](05_streaming.md) | Streaming | Kafka semantics, schemas, idempotence, deduplication, and DLQs. | 1,200-1,800 | planned |
| 6 | [`06_observability.md`](06_observability.md) | Observability | Metrics, logs, traces, and clinical service SLOs. | 1,200-1,800 | planned |
| 7 | [`07_security.md`](07_security.md) | Security | Secrets, identity, network segmentation, and supply chain. | 1,200-1,800 | planned |
| 8 | [`08_cost-and-reliability.md`](08_cost-and-reliability.md) | Cost and Reliability | SLOs, error budgets, multi-AZ, and cost control. | 1,200-1,800 | planned |
| 9 | [`09_reference-stack-walkthrough.md`](09_reference-stack-walkthrough.md) | Reference Stack Walkthrough | End-to-end deployment of the lab stack with acceptance criteria. | 1,200-1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Architecture Patterns ->
02 Containers ->
03 Kubernetes ->
04 Infrastructure as Code ->
05 Streaming ->
06 Observability ->
07 Security ->
08 Cost and Reliability ->
09 Reference Stack Walkthrough
```

## 5. What this module does not cover

- Hands-on lab execution: see `labs/` and the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.
- Clinical validation: no topic file here validates a product or a model.

## 6. Status

| Status | Count |
|---|---:|
| planned | 9 |
| draft | 0 |
| complete | 0 |

Regenerate this table with `python tools/scaffold_modules.py --status`.

## 7. Sources for this module

See section 17 of the survey file and the per-topic source sections.

