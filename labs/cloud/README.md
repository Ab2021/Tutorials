# Cloud and Streaming Reference Stack

A small, runnable reference for Module 26. It is a teaching stack, not a production deployment.

## What is included

| Path | Contents |
|---|---|
| `terraform/` | AWS VPC, S3/KMS, ECR, and EKS modules |
| `k8s/` | Namespace, deployment, service, ingress, HPA, PDB, network policy |
| `kafka/` | Redpanda compose, producer, consumer, schema, sample events |
| `Makefile` | Local Kafka up/down/produce/consume/test |

## Quick start (local Kafka)

```bash
cd labs/cloud
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
make up
make produce
make consume
```

Open the Redpanda console at `http://localhost:8080`.

## Terraform (AWS)

```bash
cd terraform
terraform init
terraform plan -var="project=biomed" -var="env=dev"
terraform apply
```

This creates billable AWS resources. Use a sandbox account, review the plan, and destroy when finished:

```bash
terraform destroy
```

The S3 backend is an example. Create the state bucket and lock table first, or use local state for a lab.

## Kubernetes

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
```

Replace `ghcr.io/example/biomed-risk:0.1.0` with your image. The manifests assume an NGINX ingress controller and metrics-server; install or remove the ingress/HPA if your cluster lacks them.

## Safety

- Do not put PHI in the sample events.
- Do not commit real secrets. `02-secret.example.yaml` is a template.
- Do not run Terraform against a shared account without review.
- The stack is not HIPAA-compliant by itself.

## Verification status

- Python producer/consumer compile; events JSONL validates.
- YAML manifests parse.
- Terraform was not applied because no AWS credentials/tooling were available in the authoring environment. Run `terraform validate` and review the plan first.
- Kafka was not started because Docker was unavailable in the authoring environment. Use the local compose path or a managed broker.
