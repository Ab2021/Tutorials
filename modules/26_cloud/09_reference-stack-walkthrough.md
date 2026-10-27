# Reference Stack Walkthrough

> Module 26 — Cloud, Kubernetes, and Streaming Production Labs
> Topic 09 of 09
> Estimated study time: 4 hours
> Prerequisites: 01, 02, 03, 04, 05, 06
> Version 2026.3

## 1. Why this topic exists

The previous eight topics explain the parts. This one runs the whole stack from an empty shell to
an accepted deployment, and then takes it apart again. The purpose is to make the interactions
visible: the Terraform output that becomes a kubectl context, the image tag that the Deployment
expects, the topic that the producer creates, the policy that the consumer's dead-letter path
depends on, and the cost that accrues while all of it runs.

The stack is in `labs/cloud/` and is deliberately small. It has three Terraform modules, eleven
Kubernetes manifests, four Python files, a compose file, and a Makefile. It can be read end to end in
an afternoon. That is the point: a stack small enough to understand completely is a better teaching
instrument than a realistic one nobody can hold in their head.

### 1.1 What the walkthrough tests

| Phase | What it proves | What it does not prove |
|---|---|---|
| Local streaming | Producer, consumer, deduplication, dead-letter queue (DLQ) work | Durability, replication, security |
| Terraform | The infrastructure can be described and created | Production hardening, multi-region |
| Kubernetes | The service schedules, serves, and survives a pod deletion | Load capacity, real traffic |
| Verification | The acceptance criteria are checkable | Regulatory compliance |
| Teardown | The environment can be removed cleanly | Data retention |

Each row is a real capability and a real limit. The walkthrough is written so that the second column
is demonstrated and the third is stated.

### 1.2 The teaching-scaffold statement

> **Note:** The lab stack is a teaching scaffold, not a compliant deployment. It does not implement
> encryption in transit between services, authentication at the ingress, an external secret manager,
> image signature verification, audit logging, backup and restore, or multi-region disaster
> recovery. It has not been assessed against Health Insurance Portability and Accountability Act (HIPAA), HITRUST, IEC 62304, or any other framework. Do
> not put protected health information in it.

That statement is repeated at the start of Section 5 and at the end of Section 9 because it is the
single most important thing to carry away from this topic. A working stack is not a compliant
system, and the distance between them is most of the work.

### 1.3 Why order matters

The apply order is not arbitrary. Terraform must create the network and the cluster before kubectl
can address them. The namespace must exist before the ConfigMap. The secret must exist before the
Deployment that references it. The Deployment must be ready before the Service has endpoints. The
Horizontal Pod Autoscaler (HPA) needs the metrics server, which the cluster may not have.

A stack applied out of order fails in ways that look like defects. A Deployment applied before its
Secret produces pods in `CreateContainerConfigError`. A Service applied before its Deployment has no
endpoints and returns connection refused. Neither is a bug in the manifest; both are a sequencing
error.

### 1.4 How to use this walkthrough

There are three ways through it, and they suit different goals.

| Approach | Time | What you get | What you miss |
|---|---|---|---|
| Read only | 60 minutes | The dependency graph and the failure modes | Muscle memory for the commands |
| Local only, no cloud | 90 minutes | Producer, consumer, dedup, DLQ | Infrastructure and Kubernetes |
| Full run | 4 to 5 hours | Every phase, including teardown | Depth on any single topic |

The local path is the highest value per minute. It exercises the semantics that are hardest to
reason about, at zero cloud cost, and it fails in ways that are visible in the console. The cloud
path is worth doing once, with a budget alert and a teardown reminder, to see the sequencing and the
timing. Nobody needs to do it twice.

Whichever path you choose, keep a transcript. The transcript is the deliverable for the exercises,
and it is the artefact that turns "I ran the lab" into evidence.

## 2. Learning outcomes

By the end of this topic you can:

1. **Execute** the stack in the documented order, local streaming first, then infrastructure, then
   workloads.
2. **Verify** each phase against explicit acceptance criteria and distinguish a pass from an
   unverified assumption.
3. **Diagnose** the five most common failures in the walkthrough from their symptoms.
4. **Compute** the cost of a lab session and the cost of leaving it running.
5. **State** at least ten specific differences between the lab and a production deployment.
6. **Tear down** the environment without leaving billable resources or losing required data.
7. **Produce** a production readiness assessment that separates what was demonstrated from what was
   assumed.

Outcomes 1 and 2 are assessed by Exercises 12.1 and 12.2. Outcomes 3 and 6 by 12.4 and 12.7.
Outcomes 4 and 5 by 12.5 and 12.6. Outcome 7 by 12.8.

## 3. The core mental model

The stack has three planes that meet at two interfaces. The local plane runs a broker on a laptop.
The cloud plane runs managed infrastructure. The workload plane runs on the cluster. The interfaces
are a bootstrap address and an image reference.

```text
  local plane                          cloud plane
  ┌──────────────────────┐            ┌───────────────────────────┐
  │ docker compose       │            │ terraform                 │
  │  redpanda :9092      │            │  network -> storage -> eks│
  │  console   :8080     │            │  outputs: bucket, repo,   │
  │ producer.py          │            │           cluster name    │
  │ consumer.py          │            └─────────────┬─────────────┘
  │ events.jsonl         │                          │ kubeconfig
  └──────────────────────┘                          ▼
                                     ┌───────────────────────────┐
                                     │ kubernetes                │
                                     │  ns -> config -> secret ->│
                                     │  deployment -> service -> │
                                     │  ingress -> hpa -> pdb -> │
                                     │  netpol -> serviceaccount │
                                     └───────────────────────────┘
```

The two interfaces are where most confusion lives. `bootstrap.servers` is `localhost:9092` for the
local plane and a different address entirely for a managed broker. The image is
`ghcr.io/example/biomed-risk:0.1.0` in the manifest and a digest in the Elastic Container Registry (ECR) repository that Terraform
creates. The stack does not connect them for you; that is an exercise.

### 3.1 The three kinds of acceptance

| Kind | Question | Evidence | Strength |
|---|---|---|---|
| Structural | Does the object exist | `kubectl get`, `terraform output` | Weakest |
| Behavioural | Does it do what it should | A request, a replay, a deletion | Moderate |
| Adversarial | Does it fail when it should | A blocked connection, a malformed event | Strongest |

The survey file's acceptance criteria mix all three. "`kubectl get nodes` works" is structural.
"Deleting a pod does not drop below one replica" is behavioural. "The pod cannot reach unapproved
egress" is adversarial. A production readiness review needs all three, and most lab exercises stop
at the first.

### 3.2 The gap between running and operable

A stack that runs has demonstrated that the manifests are syntactically valid and the images exist.
An operable stack has runbooks, alerts, backups, an owner, and a tested recovery. The gap is not a
matter of more YAML. It is a matter of process, and Section 9 lists what is missing.

### 3.3 Cost accrues from the first apply

The Terraform apply creates billable resources immediately. The Elastic Kubernetes Service (EKS) control plane bills per hour
from creation, the NAT gateway bills per hour, and the nodes bill per hour. Tearing down is part of
the exercise, not an afterthought. Section 6 quantifies the difference between a four-hour session
and a forgotten environment.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Walkthrough | Ordered execution of a stack with acceptance checks | A demo | Procedure |
| Phase | A group of steps with a shared precondition and acceptance | A step | Stage |
| Acceptance criterion | A checkable condition for completion | A goal | Test |
| Precondition | State required before a step can run | A nice-to-have | Assertion |
| Idempotent apply | Re-running produces no change | Re-running is safe in general | Property |
| Bootstrap address | Broker address given to a client | A cluster endpoint | Host and port |
| Kubeconfig | File describing cluster access | A credential | YAML |
| Context | Named cluster and user in kubeconfig | A namespace | Name |
| Teardown | Removal of created resources | Suspension | Procedure |
| Drift | Difference between declared and actual state | A bug | Attribute diff |
| Smoke test | Minimal check that a system responds | A load test | Test |
| Acceptance score | Fraction of criteria that pass | A quality score | Ratio |
| Production readiness | Evidence that a system can operate under load and failure | A checklist | Review |
| Blast radius | Scope affected by a failure | Any outage | Description |
| Runbook | Documented response to an alert | A wiki page | Document |
| Rollback | Return to a previous known state | A restart | Procedure |
| Replay | Reprocessing retained events | Retry | Offset range |
| Dead-letter queue | Destination for unprocessable records | An error log | Topic |
| Namespace | Logical partition of a cluster | A cluster | Object |
| Resource quota | Aggregate limit in a namespace | A per-pod limit | Object |
| Image digest | Content hash of an image | A tag | `sha256:` hex |
| Service endpoint | Address behind a Service | The Service object | IP and port |
| Readiness | Whether a pod can serve traffic | Whether it is alive | Boolean |
| Lab scaffold | Teaching implementation, not production | A reference architecture | Description |
| Compliance | Demonstrated satisfaction of a framework | Working software | Assessment |

Three distinctions carry the weight.

**A passing check is not a proof of the property.** A `kubectl get nodes` that returns Ready does
not prove the nodes can carry the workload. It proves the kubelet registered.

**A local pass is not a cloud pass.** The compose broker has replication factor one and no transport layer security (TLS). A
consumer that works against it has demonstrated logic, not durability.

**Teardown is part of the procedure.** A walkthrough that leaves resources running has not completed.

## 5. The mechanics, step by step

The procedure has five phases and fourteen steps. Each phase states its preconditions, its commands,
and its acceptance criteria. Do not proceed to the next phase until the current one is accepted or
the failure is understood.

> **Note:** This is a teaching scaffold, not a compliant deployment. It creates billable AWS
> resources. Use a sandbox account, review every plan, and destroy the stack when finished.

### Phase 0 — Prerequisites and safety

#### Step 1 — Establish the account and the budget boundary

**Input.** An AWS account, a region, and a spending limit.

**Output.** A dedicated account or a dedicated role, a budget alert, and a written teardown owner.

**Rationale.** The stack creates an EKS cluster, a NAT gateway, and EC2 instances. In a shared
account, the cost is attributed to nobody and the teardown is somebody else's problem. A budget alert
is the minimum guardrail.

**Acceptance.** The account or role is not used for other work. A budget exists with an alert
threshold. The teardown owner is named.

**Branch point.** If only a shared account is available, use a separate region and a unique project
prefix, and tag every resource. The blast radius is larger and the attribution weaker.

#### Step 2 — Install and verify the toolchain

**Input.** A workstation with Python, Docker, Terraform, AWS CLI, and kubectl.

**Output.** Version output from each tool, recorded.

**Rationale.** Version mismatches produce failures that look like configuration errors. Terraform
below 1.11 does not support `use_lockfile`. A kubectl more than one minor version away from the
cluster can fail on API differences.

```bash
python --version
docker --version
terraform version
aws --version
kubectl version --client
```

**Acceptance.** Every command returns a version. Terraform is 1.11 or later if the example backend
is used. The AWS CLI is authenticated to the intended account:

```bash
aws sts get-caller-identity
```

**Edge case.** A workstation with an expired SSO session fails at the first AWS call with an error
that looks like a permissions problem. Re-authenticate first.

### Phase 1 — The local streaming path

#### Step 3 — Start the broker

**Input.** `labs/cloud/docker-compose.yml`.

**Output.** A running Redpanda container and console.

```bash
cd labs/cloud
make up
docker compose ps
```

**Rationale.** The local path tests the producer and consumer logic without cloud cost or
credentials. The compose file runs one broker with `--smp 1` and `--memory 1G`, and a console on port
8080.

**Acceptance.** `docker compose ps` shows the broker healthy. `http://localhost:8080` loads the
console. The health check command `rpk cluster health` reports healthy.

**Edge case.** Port 9092 may already be in use by another broker. The compose file maps 9092 and
29092; change the host mapping rather than stopping the other service.

#### Step 4 — Produce events

**Input.** `labs/cloud/kafka/events.jsonl` and `producer.py`.

**Output.** A topic `adt.events` with three partitions and six events.

```bash
make produce
```

**Rationale.** `producer.py` validates every line with the Pydantic model in `schemas.py`, creates
the topic if absent, keys by `patient_id`, enables idempotence, uses `acks=all`, and attaches a
`schema_version` header. Each of those choices is a topic in
[`05_streaming.md`](05_streaming.md).

**Acceptance.** The producer reports six events submitted. Delivery callbacks report partition and
offset. The topic exists with three partitions.

**Edge case.** The producer's `ensure_topic` call catches an exception when the topic already
exists; that is expected and printed. A validation failure prints the line number and skips the
record, which is the correct behaviour for a producer that must not block on one bad row.

#### Step 5 — Consume, replay, and dead-letter

**Input.** The running consumer and the same events file.

**Output.** Six processed events, zero duplicates on replay, and one dead-lettered record.

```bash
make consume
make produce      # replay
make consume
```

**Rationale.** The consumer commits after processing, deduplicates by `event_id` in SQLite, and
routes validation failures to `adt.events.dlq`. Replaying the same file must not produce duplicate
processing. Sending a malformed record must produce a dead-letter entry and a committed offset.

**Acceptance.**

```text
first run:  6 processed
replay:     6 duplicates, 0 processed
malformed:  1 dead-lettered, consumer continues
```

**Edge case.** The SQLite file `consumer.db` persists between runs. Delete it to reset the
deduplication state, and note that doing so makes the replay produce six processed events again,
which is exactly the behaviour the file is there to prevent.

### Phase 2 — Infrastructure

#### Step 6 — Review the Terraform configuration

**Input.** `labs/cloud/terraform/`.

**Output.** A written list of the resources to be created and the ones that are risky.

**Rationale.** Reading the configuration before applying it is the review step from
[`04_infrastructure-as-code.md`](04_infrastructure-as-code.md). The modules create a virtual private cloud (VPC) with three
availability zones and one NAT gateway, a key management service (KMS) key, a versioned S3 bucket with public access blocked,
an immutable ECR repository with scan on push, and an EKS cluster with a managed node group.

**Branch point.** `variables.tf` defaults `cluster_version` to `"1.30"`. As of 26 October 2026, EKS
lists 1.34 through 1.37 in standard support and 1.31 through 1.33 in extended support. Version 1.30
is not listed. Change the default to a supported version before applying, or the apply fails or the
cluster is auto-upgraded.

**Edge case.** `backend.tf.example` is an example, not an active backend. Without renaming it to
`backend.tf` and creating the bucket, Terraform uses local state. That is acceptable for a lab and
unacceptable for anything shared.

#### Step 7 — Initialise and plan

**Input.** The configuration and credentials.

**Output.** A plan file and a review.

```bash
cd labs/cloud/terraform
terraform init
terraform plan -var="project=biomed" -var="env=dev" -out=tfplan
```

**Rationale.** `terraform init` downloads the providers and modules pinned in `versions.tf` and the
module blocks. The plan shows every resource and every replacement.

**Acceptance.** `terraform init` completes without a version error. The plan shows the expected
resource count and no unexpected replacements. The plan is saved.

**Edge case.** The EKS module is pinned at `~> 20.0`. The registry's current release line is 21.x. The
plan will use a 20.x release, which is supported but not current. Do not change the constraint
during a lab; upgrade deliberately with the migration guide.

#### Step 8 — Apply

**Input.** The saved plan.

**Output.** Created infrastructure and outputs.

```bash
terraform apply tfplan
terraform output
```

**Rationale.** Applying the saved plan, rather than re-planning, ensures that what was reviewed is
what runs. The outputs include the raw bucket name and the cluster name.

**Acceptance.** The apply completes. `terraform output raw_bucket` exists.
`aws s3api get-public-access-block --bucket <name>` shows all four settings true.
`aws ecr describe-repositories` shows `imageTagMutability: IMMUTABLE` and
`scanOnPush: true`.

**Edge case.** The apply takes 12 to 18 minutes, mostly for the EKS control plane and node group.
A timeout in the terminal does not mean the apply failed; check `terraform state list` and the AWS
console before retrying.

#### Step 9 — Connect kubectl

**Input.** The cluster name and region.

**Output.** A kubeconfig context for the cluster.

```bash
aws eks update-kubeconfig --name biomed-dev --region us-east-2
kubectl config current-context
kubectl get nodes
```

**Rationale.** Terraform creates the cluster but does not modify the local kubeconfig. The AWS CLI
writes the context, and the IAM principal that created the cluster has access because the module sets
`enable_cluster_creator_admin_permissions = true`.

**Acceptance.** The context names the cluster. `kubectl get nodes` shows two nodes in `Ready` state.

**Edge case.** If the nodes are not `Ready`, the node group may still be scaling. Wait five minutes
before investigating. A node that never joins is usually a subnet, IAM, or quota problem.

### Phase 3 — Workloads

#### Step 10 — Apply the namespace, configuration, and secret

**Input.** `00-namespace.yaml`, `01-configmap.yaml`, `01a-model-configmap.yaml`,
`02-secret.example.yaml`.

**Output.** A namespace, two ConfigMaps, and a Secret.

```bash
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-configmap.yaml
kubectl apply -f k8s/01a-model-configmap.yaml
kubectl apply -f k8s/02-secret.example.yaml
```

**Rationale.** The Deployment references the ConfigMap by name and the Secret by key. Applying it
first produces `CreateContainerConfigError`. The example secret contains placeholders and must be
replaced or the application will fail to connect to a database it does not have.

**Acceptance.** The namespace exists. Both ConfigMaps have the expected keys. The Secret exists with
the expected key, even if the value is a placeholder.

**Edge case.** The example secret is a template. Committing a real value is a breach; the file is
named `.example` for that reason.

#### Step 11 — Apply the workload

**Input.** `03-deployment.yaml` through `09-serviceaccount.yaml`.

**Output.** A Deployment, Service, Ingress, HPA, PodDisruptionBudget (PDB), NetworkPolicy, and ServiceAccount.

```bash
kubectl apply -f k8s/03-deployment.yaml
kubectl apply -f k8s/04-service.yaml
kubectl apply -f k8s/05-ingress.yaml
kubectl apply -f k8s/06-hpa.yaml
kubectl apply -f k8s/07-pdb.yaml
kubectl apply -f k8s/08-networkpolicy.yaml
kubectl apply -f k8s/09-serviceaccount.yaml
```

**Rationale.** The order follows the dependency graph: the workload before the Service that selects
it, the Service before the Ingress that routes to it, and the ServiceAccount before the Deployment
that references it. In practice Kubernetes reconciles eventual consistency, so a different order
converges, but the errors during convergence are confusing.

**Branch point.** The image `ghcr.io/example/biomed-risk:0.1.0` may not exist. Replace it with an
image you control, ideally the digest from the ECR repository Terraform created.

**Edge case.** The Ingress requires an NGINX ingress controller and a TLS secret named
`risk-service-tls`. The lab does not install either. Without them, the Ingress has no address and the
TLS handshake fails.

#### Step 12 — Wait for readiness

**Input.** The applied objects.

**Output.** Two ready pods and a populated EndpointSlice.

```bash
kubectl rollout status deploy/risk-service -n biomed --timeout=180s
kubectl get pods -n biomed -o wide
kubectl get endpoints risk-service -n biomed
```

**Rationale.** A Deployment that has been applied is not a Deployment that is serving. The rollout
status blocks until the new ReplicaSet is available, which requires the image to pull and the
readiness probe to pass.

**Acceptance.** `rollout status` reports success. Two pods are `Running` and `1/1 Ready`. The
EndpointSlice lists two addresses.

### Phase 4 — Verification

#### Step 13 — Run the acceptance checks

**Input.** The running stack.

**Output.** A pass or fail for each criterion, with evidence.

**Rationale.** The checks in the survey are a mix of structural, behavioural, and adversarial. Run
all three kinds or the assessment is incomplete.

```bash
kubectl rollout status deploy/risk-service -n biomed
kubectl get endpoints risk-service -n biomed
kubectl delete pod -n biomed -l app=risk-service --wait=false
kubectl get pods -n biomed -w
kubectl run nettest --rm -it --image=curlimages/curl -n default -- \
  curl -m 3 http://risk-service.biomed.svc.cluster.local/healthz
```

**Acceptance.** The rollout completes. The endpoints are non-empty. Deleting a pod leaves at least
one ready. The curl from the default namespace fails.

**Edge case.** The HPA check requires metrics-server. A cluster without it reports `<unknown>` and
does not scale. That is a missing add-on, not a broken HPA.

### Phase 5 — Teardown

#### Step 14 — Destroy in reverse order

**Input.** The applied stack.

**Output.** No billable resources, and a record of what was deleted.

```bash
kubectl delete -f k8s/08-networkpolicy.yaml
kubectl delete namespace biomed
cd labs/cloud/terraform
terraform destroy -var="project=biomed" -var="env=dev"
```

**Rationale.** Terraform destroys what its state knows about. Kubernetes objects are not in that
state, so the namespace is deleted first to avoid orphaned load balancers that block VPC deletion.
The ECR repository may contain images and the S3 bucket may contain objects; both can prevent
deletion unless they are emptied or the resources are configured to force destruction.

**Acceptance.** `terraform destroy` completes. The AWS console shows no EKS cluster, no NAT gateway,
and no running instances. The budget for the next month shows no accrual.

**Edge case.** A load balancer created by the Ingress controller holds an elastic network interface
in the VPC subnets and blocks subnet deletion. Deleting the namespace first usually removes it;
if not, delete the Service of type LoadBalancer and wait for the controller to clean up.

### 5.1 The acceptance matrix

| # | Criterion | Kind | Command | Lab result |
|---:|---|---|---|---|
| 1 | Terraform output exists | Structural | `terraform output raw_bucket` | Pass |
| 2 | Bucket blocks public access | Structural | `aws s3api get-public-access-block` | Pass |
| 3 | ECR scans on push | Structural | `aws ecr describe-repositories` | Pass |
| 4 | Nodes ready | Structural | `kubectl get nodes` | Pass |
| 5 | Rollout completes | Behavioural | `kubectl rollout status` | Pass |
| 6 | Health endpoint returns 200 | Behavioural | `curl /healthz` | Pass |
| 7 | Readiness endpoint returns 200 | Behavioural | `curl /readyz` | Pass |
| 8 | Pod deletion keeps a replica | Behavioural | delete and watch | Pass |
| 9 | HPA scales under load | Behavioural | load test | Fails without metrics-server |
| 10 | Unapproved egress blocked | Adversarial | curl from default namespace | Fails without a policy-enforcing container network interface (CNI) |
| 11 | Replay produces no duplicates | Behavioural | produce twice, consume | Pass |
| 12 | Malformed event reaches DLQ | Adversarial | inject malformed record | Pass |

Ten of twelve pass as shipped. The two failures are not defects in the manifests; they are missing
cluster capabilities. That distinction is the lesson of the matrix: a stack can be correct and still
not demonstrate a property, because the environment does not support the test.

### 5.2 Edge cases that change the answer

| Edge case | Symptom | Cause | Fix |
|---|---|---|---|
| Unsupported cluster version | Apply error naming the version | Default 1.30 past extended support | Raise the default, check the EKS calendar |
| Node capacity | Pod stays Pending | Request exceeds usable node memory | Lower the request or use a larger node |
| Image unavailable | `ImagePullBackOff`, `manifest unknown` | Placeholder or private image | Use an image you control |
| Metrics server absent | HPA target `<unknown>` | Add-on not installed | Install metrics-server |
| CNI without policy support | Blocked curl succeeds | Policy objects ignored | Verify enforcement, change CNI |
| Ingress controller absent | Ingress has no address | NGINX not installed | Install the controller |
| TLS secret absent | Default certificate served | Secret not created | Create `risk-service-tls` |
| State backend absent | Local `terraform.tfstate` | Example backend not activated | Accept for a lab, never for shared work |
| ECR repository not empty | Destroy fails | Images retained | Empty or `force_delete` |
| S3 bucket not empty | Destroy fails | Objects retained | Empty or `force_destroy` |

## 6. Worked example

A learner runs the full walkthrough in a sandbox account in us-east-2 on a weekday evening. The
question is how long it takes, what it costs, and which acceptance checks pass.

### 6.1 Inputs

| Parameter | Value | Source |
|---|---|---|
| Region | us-east-2 (Ohio) | Configuration |
| Currency and month | USD, 730 hours | Convention |
| Node type | `t3.medium`, 2 vCPU, 4 GiB | `variables.tf` default |
| Node count | 2 | `node_desired_size = 2` |
| EKS price | USD 0.10/cluster-hour | AWS Price List, 26 October 2026 |
| Node price | USD 0.0416/hour | AWS Price List, 26 October 2026 |
| NAT price | USD 0.045/hour | AWS VPC pricing example |
| S3 and gp3 | 100 GB each | Estimate |
| Image size | 1.2 GB | Assumed |
| Session length | 4 hours | Exercise |

### 6.2 Arithmetic: monthly and hourly cost

```text
EKS     1 × 0.10   × 730 =  73.00
Nodes   2 × 0.0416 × 730 =  60.74
NAT     1 × 0.045  × 730 =  32.85
S3      100 GB × 0.023   =   2.30
gp3     100 GB × 0.08    =   8.00
                           ------
total                       176.89 USD/month
```

Hourly:

```text
176.89 ÷ 730 = 0.2423 USD/hour
```

A four-hour session:

```text
0.2423 × 4 = 0.97 USD
```

Leaving it running for 30 days:

```text
176.89 USD
```

The difference is a factor of 182. A forgotten lab environment is the most common cost mistake in a
course like this, and it is entirely avoidable with a calendar reminder and a `terraform destroy`.

### 6.3 Arithmetic: capacity on two nodes

Usable per node, with a 700m CPU and 2,560 MiB reservation:

```text
CPU:    2,000 − 700 = 1,300m
Memory: 4,096 − 2,560 = 1,536 MiB
```

Pods that fit at the reference request of 250m and 512 MiB:

```text
min(1,300 ÷ 250, 1,536 ÷ 512) = min(5, 3) = 3 pods per node
```

Two nodes hold six pods. The Deployment needs two, and the HPA can reach six. The cluster is exactly
sized for the HPA maximum and has no room for a surge pod during a rollout, which is why a rollout
with `maxSurge: 1` can stall if the HPA has already scaled to six.

At a request of 1,000m and 2,048 MiB:

```text
min(1,300 ÷ 1,000, 1,536 ÷ 2,048) = min(1, 0) = 0 pods per node
```

The memory request exceeds usable memory, so the pod never schedules. The scheduler event is
`Insufficient memory`, and no amount of CPU adjustment fixes it.

### 6.4 Arithmetic: elapsed time

| Phase | Duration | Notes |
|---|---:|---|
| Toolchain check | 5 minutes | Manual |
| Local compose and produce/consume | 10 minutes | Includes two consumer runs |
| `terraform init` | 2 minutes | Provider download |
| `terraform apply` | 15 minutes | EKS control plane dominates |
| kubectl context and node readiness | 5 minutes | Node group join |
| Apply eleven manifests | 3 minutes | Sequential |
| Image pull and rollout | 3 minutes | Cold nodes, 1.2 GB image |
| Acceptance checks | 10 minutes | Includes a pod deletion |
| Teardown | 12 minutes | Cluster deletion dominates |
| **Total** | **65 minutes** | Excludes reading and troubleshooting |

The Terraform phases are 27 minutes of the 65, and almost all of that is waiting for AWS. The
learning content is in the 38 minutes that remain, which is why the walkthrough is best done once
with attention and repeated later for speed.

### 6.5 Arithmetic: the acceptance score

Twelve criteria, ten pass as shipped:

```text
10 ÷ 12 = 83.3 percent
```

The two failures are the HPA test, which needs metrics-server, and the egress test, which needs a
policy-enforcing CNI. Both are environment capabilities rather than manifest defects, and both are
instructive: the stack cannot demonstrate a property that the cluster does not support.

A readiness review that reports "83 percent" without naming the two failures is worse than one that
reports "ten pass, two unverified, here is why". The number is less useful than the list.

### 6.6 Two diagnostics that reveal a misconfiguration

**Diagnostic 1: the apply fails on the cluster version.**

```text
Error: creating EKS Cluster (biomed-dev): InvalidParameterException:
Unsupported Kubernetes minor version: 1.30
```

The cause is the `cluster_version` default. Confirm the supported versions:

```bash
aws eks describe-cluster-versions \
  --query 'clusterVersions[].[clusterVersion,status]' --output table
```

Fix the variable, re-plan, and apply. Do not edit the cluster in the console; the next apply will
revert it.

**Diagnostic 2: pods are Pending after the manifests are applied.**

```bash
kubectl describe pod -n biomed -l app=risk-service | sed -n '/Events/,$p'
```

The event `0/2 nodes are available: 2 Insufficient memory` means the request exceeds usable memory
on every node. The check is the request in `03-deployment.yaml` against the allocatable in
`kubectl describe node`. A request of 2,048 MiB cannot fit on a `t3.medium` with a 2,560 MiB
reservation, as Section 6.3 shows.

A third diagnostic is a consumer that reprocesses the same six events on every run. The SQLite file
is in a different working directory, so the deduplication table is empty each time. The consumer
prints `processed` rather than `duplicate`, which is the signal.

### 6.7 Vary one assumption

Add a third node to the group (`node_desired_size = 3`). Monthly cost becomes:

```text
3 × 0.0416 × 730 = 91.10
176.89 − 60.74 + 91.10 = 207.25 USD/month
```

The hourly rate becomes `207.25 ÷ 730 = 0.2839`, and a four-hour session costs USD 1.14 instead of
USD 0.97. The capacity becomes nine pods at the reference request, which gives the HPA its maximum
of six plus room for a surge pod. The extra USD 0.17 per session buys a rollout that cannot stall.
That is the same trade as Section 6.3 expressed in a different unit, and it is the reason capacity
and cost belong in one calculation.

## 7. Code, query, or calculation

The following standard-library Python models the lab cost, capacity, rollout time, and acceptance
score. It reproduces Section 6.

```python
"""Reference stack cost, capacity, and acceptance model. Stdlib only."""
from __future__ import annotations

import math

HOURS_MONTH = 730.0
EKS_HOURLY = 0.10
NAT_HOURLY = 0.045
S3_GB_MONTH = 0.023
GP3_GB_MONTH = 0.08
T3_MEDIUM = 0.0416

NODE_ALLOC_CPU = 2000
NODE_ALLOC_MEM = 4096
RESERVE_CPU = 700
RESERVE_MEM = 2560


def monthly(nodes: int = 2) -> dict[str, float]:
    return {
        "eks": EKS_HOURLY * HOURS_MONTH,
        "nodes": nodes * T3_MEDIUM * HOURS_MONTH,
        "nat": NAT_HOURLY * HOURS_MONTH,
        "s3": 100 * S3_GB_MONTH,
        "gp3": 100 * GP3_GB_MONTH,
    }


def pods_per_node(cpu_milli: int, mem_mib: int) -> int:
    usable_cpu = NODE_ALLOC_CPU - RESERVE_CPU
    usable_mem = NODE_ALLOC_MEM - RESERVE_MEM
    return min(usable_cpu // cpu_milli, usable_mem // mem_mib)


def timeline(pull_s: float, startup_s: float, ready_s: float, replicas: int,
             surge: int = 1) -> float:
    waves = math.ceil(replicas / surge)
    return waves * (pull_s + startup_s + ready_s)


def main() -> None:
    parts = monthly(2)
    total = sum(parts.values())
    print("lab environment, us-east-2, 730 hours")
    for key, value in parts.items():
        print(f"  {key:8} {value:8.2f} USD/month")
    print(f"  {'total':8} {total:8.2f} USD/month")
    print(f"  hourly  {total / HOURS_MONTH:8.4f} USD/hour")
    for hours in (2, 4, 8, 24):
        print(f"  {hours:2d}-hour session: {total / HOURS_MONTH * hours:6.2f} USD")
    print(f"  left running for 30 days: {total:.2f} USD")

    print()
    for cpu, mem in ((250, 512), (500, 1024), (1000, 2048)):
        print(f"  request {cpu:4d}m/{mem:4d}Mi -> {pods_per_node(cpu, mem)} pods per t3.medium")
    print(f"  usable per node: {NODE_ALLOC_CPU - RESERVE_CPU}m / "
          f"{NODE_ALLOC_MEM - RESERVE_MEM}Mi")

    print()
    print("rollout timeline, 1.2 GB image, cold nodes")
    for replicas in (2, 6):
        t = timeline(48, 20, 10, replicas)
        print(f"  replicas={replicas}  elapsed={t:5.1f} s")
    print("rollout timeline, 180 MB image, cold nodes")
    for replicas in (2, 6):
        t = timeline(7.2, 20, 10, replicas)
        print(f"  replicas={replicas}  elapsed={t:5.1f} s")

    print()
    checks = [
        ("terraform output raw_bucket exists", True),
        ("bucket blocks public access", True),
        ("ECR scan on push enabled", True),
        ("kubectl get nodes returns Ready", True),
        ("rollout status completes", True),
        ("/healthz returns 200", True),
        ("/readyz returns 200", True),
        ("pod deletion keeps one replica", True),
        ("HPA scales under load", False),
        ("egress to unapproved host blocked", False),
        ("replay produces no duplicates", True),
        ("malformed event reaches DLQ", True),
    ]
    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"  {passed}/{len(checks)} acceptance checks pass in the lab as shipped")


if __name__ == "__main__":
    main()
```

Actual output:

```text
lab environment, us-east-2, 730 hours
  eks         73.00 USD/month
  nodes       60.74 USD/month
  nat         32.85 USD/month
  s3           2.30 USD/month
  gp3          8.00 USD/month
  total      176.89 USD/month
  hourly    0.2423 USD/hour
   2-hour session:   0.48 USD
   4-hour session:   0.97 USD
   8-hour session:   1.94 USD
  24-hour session:   5.82 USD
  left running for 30 days: 176.89 USD

  request  250m/ 512Mi -> 3 pods per t3.medium
  request  500m/1024Mi -> 1 pods per t3.medium
  request 1000m/2048Mi -> 0 pods per t3.medium
  usable per node: 1300m / 1536Mi

rollout timeline, 1.2 GB image, cold nodes
  replicas=2  elapsed=156.0 s
  replicas=6  elapsed=468.0 s
rollout timeline, 180 MB image, cold nodes
  replicas=2  elapsed= 74.4 s
  replicas=6  elapsed=223.2 s

  PASS  terraform output raw_bucket exists
  PASS  bucket blocks public access
  PASS  ECR scan on push enabled
  PASS  kubectl get nodes returns Ready
  PASS  rollout status completes
  PASS  /healthz returns 200
  PASS  /readyz returns 200
  PASS  pod deletion keeps one replica
  FAIL  HPA scales under load
  FAIL  egress to unapproved host blocked
  PASS  replay produces no duplicates
  PASS  malformed event reaches DLQ
  10/12 acceptance checks pass in the lab as shipped
```

The rollout model assumes every pod pulls the image. In practice the first pod warms the node cache
and later pods on the same node start faster, so 156 seconds is an upper bound for two replicas on
two cold nodes.

The following Makefile target sequence is the local half of the walkthrough. It is validated by
parsing, not executed here.

```makefile
# labs/cloud/Makefile (excerpt, for reference; the real file uses tabs)
up:
	docker compose up -d

produce:
	python kafka/producer.py --file kafka/events.jsonl

consume:
	python kafka/consumer.py --max-messages 10

down:
	docker compose down -v
```

A useful addition to the real Makefile is a verification target that runs the local acceptance
checks in order and fails on the first mismatch:

```bash
verify: up
	python kafka/producer.py --file kafka/events.jsonl
	python kafka/consumer.py --max-messages 6
	python kafka/producer.py --file kafka/events.jsonl
	python kafka/consumer.py --max-messages 6
```

The second consumer run should print six duplicates and process zero. If it processes six, the
deduplication store is not persisting, which is the most common local failure.

## 8. Data, standards, and regulatory context

### 8.1 What the walkthrough demonstrates

The stack demonstrates that a container image can be built and run with a restricted security
context, that infrastructure can be described and recreated, that a streaming path can be
idempotent, and that a Kubernetes service can be exposed and verified. Those are real engineering
capabilities and they are the prerequisites for compliance.

### 8.2 What it does not demonstrate

| Requirement | Framework | Lab status | What is missing |
|---|---|---|---|
| Risk analysis | HIPAA 164.308(a)(1)(ii)(A) | Absent | Documented analysis |
| Audit controls | HIPAA 164.312(b) | Absent | Audit pipeline and retention |
| Transmission security | HIPAA 164.312(e)(1) | Partial | TLS inside the cluster |
| Access control | HIPAA 164.312(a)(1) | Partial | Authentication, least-privilege IAM |
| Contingency plan | HIPAA 164.308(a)(7) | Absent | Backup, restore test, RTO and RPO |
| Configuration management | IEC 62304 clause 8 | Partial | Reviewed change records, baselines |
| SBOM in premarket submission | FD&C Act 524B | Absent | SBOM generation and retention |
| Incident response | HIPAA 164.308(a)(6) | Absent | Plan, roles, exercise |

The table is the honest answer to "is this production ready". Eight rows, two partial, six absent.

### 8.3 The role of a reference architecture

A reference architecture demonstrates a pattern. It is not a compliance artefact, and vendors that
present one as compliant are eliding the difference between "can be configured to meet" and "has
been assessed as meeting". The distinction matters in procurement, where a reference architecture is
often the only technical document a buyer sees.

### 8.4 Data handling in the lab

The sample events in `labs/cloud/kafka/events.jsonl` use synthetic identifiers (`synth-1001` and
similar). They are not protected health information (PHI). The secret template contains placeholders. The S3 bucket is empty unless
the learner puts something in it. Those are deliberate choices that make the lab safe to run in a
sandbox account.

If you adapt the lab to real data, every row of the table above becomes a blocker, not a task.

## 9. Industry practice and case evidence

### 9.1 Documented practice

**Reference stacks are common and rarely production-ready.** AWS publishes reference architectures
through the Well-Architected programme and the Healthcare Industry Lens; the Kubernetes project
publishes example deployments; vendors publish quick starts. Each carries a disclaimer that it is a
starting point. The disclaimer is the substantive part.

**The survey file's acceptance criteria are a fair model.** The course's own criteria mix structural,
behavioural, and adversarial checks, and explicitly warn that the stack does not prove HIPAA
compliance, production availability, multi-region disaster recovery, or clinical safety. That
statement is unusually direct for teaching material and is worth preserving.

**Managed Kubernetes version lifecycles are short.** EKS supports a version for 14 months of standard
support plus 12 months of extended support. A lab pinned to an older version becomes an upgrade
exercise within a year, which is itself a useful lesson in maintenance.

### 9.2 Case evidence

1. **The Kubernetes project's own example deployments** (the guestbook and the microservices demo)
   are widely used as references and are explicitly not production configurations. They lack
   resource limits, network policies, and pod disruption budgets in their simplest forms.
2. **The AWS EKS quick start** provisions a cluster with sensible defaults and a documented
   responsibility matrix. The matrix is the useful artefact: it states what the customer must add.
3. **The 2021 S3 us-east-1 outage** affected services that had followed a reference architecture
   within a single region. The lesson is not that the reference was wrong but that its availability
   assumptions were regional.

### 9.3 Inference

The following are inferences, labelled as such:

- Learners who run the stack once remember the apply order; learners who read it do not. The
  sequencing errors are the most durable lesson.
- The most common adaptation failure is replacing the placeholder image without checking its
  architecture against the node type. `t3.medium` is x86-64; an arm64 image fails with a platform
  error.
- Teams that adopt a reference stack without a teardown procedure leave billable resources behind at
  a rate that is invisible until the invoice arrives.

A final inference concerns the value of the two failed acceptance checks. They are the most useful
part of the exercise, because they teach the difference between a configuration and a capability. A
NetworkPolicy object is a configuration; enforcement is a capability of the CNI. An HPA object is a
configuration; metrics are a capability of the cluster. A readiness review that cannot tell those
apart will mark both as done and discover the gap during an incident.

The habit worth building is to ask, for every check, "what would have to be true for this to fail
silently". For the HPA it is a missing metrics API. For the egress test it is a CNI that accepts
policy objects and ignores them. For the replay test it is a deduplication store in a different
working directory. Each silent failure has a cheap detection, and the detection belongs in the
acceptance criteria.

A completed walkthrough is not the end of the learning. The useful next step is to break the stack
deliberately, one property at a time, and confirm that the acceptance check catches it: remove the
CPU request and watch the HPA go unknown, change the Secret key name and watch the pods fail to
start, delete the ConfigMap and watch the rollout stall. Each break is a rehearsal for an incident,
and the stack is small enough that the rehearsal costs minutes rather than days.

Record each break and its detection in a short table. The table becomes the runbook for the service,
and it is more useful than a list of components because every row has been observed rather than
assumed.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Unsupported cluster version | Apply error naming the version | Default past end of support | No cluster or auto-upgrade | Check the EKS calendar |
| Pod unschedulable | `Insufficient memory` event | Request exceeds usable capacity | No rollout | Size requests to node |
| Image pull failure | `ImagePullBackOff` | Wrong tag, private registry, arch mismatch | No pods | Verify image and architecture |
| Missing ConfigMap | `CreateContainerConfigError` | Deployment applied before config | Crash loop | Apply in order |
| Missing secret | Same | Secret not created | Crash loop | Create before deployment |
| Ingress without controller | No address on the Ingress | Controller not installed | No external access | Install or remove Ingress |
| Missing TLS secret | Default certificate served | Secret not created | Wrong certificate | Create the secret |
| HPA without metrics | `<unknown>` target | metrics-server absent | No autoscaling | Install metrics-server |
| NetworkPolicy ignored | Adversarial test succeeds | CNI lacks policy support | False segmentation | Verify enforcement |
| Duplicate processing on replay | Consumer prints `processed` twice | Dedup store not persisted | Duplicate output | Check the database path |
| Malformed event stalls partition | Lag grows, same record retried | No DLQ path | Partition stalls | Dead-letter and commit |
| State left behind | `terraform destroy` fails | Non-empty bucket or repository | Ongoing cost | Empty or force-destroy |
| Orphaned load balancer | Subnet deletion fails | Ingress service not deleted | VPC cannot be destroyed | Delete namespace first |
| Forgotten environment | Bill continues after the session | No teardown | Cost accrual | Calendar reminder, budget alert |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Sandbox account with no other workloads | Cost and blast radius | Shared account | `aws ec2 describe-instances` |
| Terraform 1.11 or later | `use_lockfile` support | Older Terraform | `terraform version` |
| Two `t3.medium` nodes are enough | Capacity arithmetic | HPA at maximum, surge pod | `kubectl describe node` |
| The placeholder image exists | Rollout | Private or deleted image | `docker manifest inspect` |
| The ingress controller is installed | External access | No NGINX | `kubectl get ingressclass` |
| metrics-server is installed | HPA test | Minimal cluster | `kubectl top pods` |
| The CNI enforces network policy | Adversarial test | Plugin without policy | Curl from a blocked namespace |
| The local broker has no TLS | Local test simplicity | Production expectations | Inspect the listener configuration |
| A four-hour session is enough | Cost estimate | Slow first run | Time the phases |
| Teardown removes all billable resources | Cost control | Orphaned load balancer | Check the console after destroy |

## 12. Exercises

### 12.1 Execute the local phase

**Deliverable.** A transcript of `make up`, `make produce`, `make consume`, a replay, and a malformed
event, with the duplicate and dead-letter outputs highlighted.

**Acceptance criteria.** Six processed on the first run, six duplicates on the replay, one
dead-lettered record. No PHI in any output.

**Time.** 45 minutes.

### 12.2 Execute the cloud phase

**Deliverable.** A transcript of `terraform init`, `plan`, `apply`, the kubeconfig update, and the
manifest applications, with the acceptance checks and their results.

**Acceptance criteria.** Plan reviewed before apply. Outputs recorded. Ten of twelve checks run,
with the two failures explained.

**Time.** 90 minutes.

### 12.3 Fix the unsupported version

**Deliverable.** A diff that raises `cluster_version` to a supported version, with the source for
the supported list and the date accessed.

**Acceptance criteria.** Version is in standard support on the access date. Re-plan shows no
unexpected replacement.

**Time.** 20 minutes.

### 12.4 Diagnose a pending pod

**Deliverable.** A written diagnosis of `Insufficient memory` for a 2,048 MiB request on
`t3.medium`, with the arithmetic and three possible fixes.

**Acceptance criteria.** Shows usable memory after reservation. Each fix states its cost.

**Time.** 30 minutes.

### 12.5 Cost the session

**Deliverable.** The cost of a 4-hour session and the cost of leaving the stack running for 30 days,
at us-east-2 in USD, with the price sources.

**Acceptance criteria.** Every price sourced and dated. The ratio stated.

**Time.** 25 minutes.

### 12.6 List the production gaps

**Deliverable.** A table of at least ten differences between the lab and a production deployment,
each with the file affected and the change required.

**Acceptance criteria.** At least three gaps are in Terraform. At least one cannot be fixed in a
manifest. Each change is specific.

**Time.** 50 minutes.

### 12.7 Tear down cleanly

**Deliverable.** A teardown transcript and a post-teardown check showing no billable resources.

**Acceptance criteria.** Namespace deleted before Terraform destroy. No EKS cluster, NAT gateway, or
running instance remains. Any retained data is named with a reason.

**Time.** 40 minutes.

### 12.8 Write a readiness assessment

**Deliverable.** A two-page assessment that separates demonstrated capabilities, unverified
assumptions, and absent controls, with evidence for each.

**Acceptance criteria.** Uses the three acceptance kinds. Does not claim compliance. Names the two
failed checks and why they failed.

**Time.** 60 minutes.

## 13. Self-check answers

### 13.1 Local transcript

The first consume processes six events. The replay prints six duplicates and processes zero, because
the `processed` table has a primary key on `event_id`. A malformed record prints a validation error,
is produced to `adt.events.dlq`, and the original offset is committed. If the replay processes six,
the SQLite file was in a different working directory or was deleted.

### 13.2 Cloud transcript

The plan should show the VPC, three private and three public subnets, one NAT gateway, the KMS key,
the S3 bucket and its policy resources, the ECR repository, the EKS cluster, and the node group. The
apply takes 12 to 18 minutes. The kubeconfig update writes a context named for the cluster. Ten of
twelve acceptance checks pass; the HPA and egress checks need cluster capabilities the lab does not
install.

### 13.3 Version fix

Change the default in `variables.tf` to a version in standard support on the access date, for example
1.37 as of 26 October 2026, or pass `-var="cluster_version=1.37"`. Confirm with
`aws eks describe-cluster-versions`. The plan should show an in-place control plane update and a node
group update, not a replacement of the cluster.

### 13.4 Pending pod

Usable memory is `4,096 − 2,560 = 1,536 MiB`, which is less than the 2,048 MiB request, so the pod
cannot fit on any node. Fixes: lower the request to below 1,536 MiB, use a larger node type such as
`m7g.large` with 8 GiB, or reduce the reservation if it is over-provisioned. The first is free and
requires knowing the real working set; the second costs more per hour; the third requires
measurement.

### 13.5 Session cost

`176.89 ÷ 730 = 0.2423 USD/hour`. Four hours is USD 0.97. Thirty days is USD 176.89. The ratio is
182.

### 13.6 Production gaps

Examples: image by tag rather than digest (`03-deployment.yaml`); placeholder secret
(`02-secret.example.yaml`); no TLS inside the cluster (all manifests); broad egress
(`08-networkpolicy.yaml`); public cluster endpoint (`modules/eks/main.tf`); no audit logging (cluster
configuration); no backup (`modules/storage`); single NAT (`modules/network`); unsupported version
(`variables.tf`); no startup probe (`03-deployment.yaml`); no resource quota (namespace). Encryption
in transit inside the cluster cannot be fixed in a single manifest; it requires a mesh or a CNI
feature.

### 13.7 Teardown

Delete the namespace first so the ingress controller removes its load balancer. Then
`terraform destroy`. If the bucket or repository is non-empty, empty it or set force-destroy. Check
the console for an EKS cluster, a NAT gateway, and running instances. The most common leftover is a
load balancer's network interface, which blocks subnet deletion.

### 13.8 Readiness assessment

A strong assessment has three sections. Demonstrated: the image runs with a restricted context, the
service becomes ready and serves, a pod deletion preserves a replica, replay is idempotent, malformed
events are dead-lettered. Unverified: HPA scaling, egress blocking, capacity under load, restore.
Absent: audit, TLS in transit, secret management, backup, incident response, SBOM. It does not claim
compliance and it names the evidence for each row.

## 14. Interview questions

**1. Why does apply order matter?**
A strong answer gives a concrete dependency, such as Secret before Deployment, and the error that
results from reversing it. A weak answer says Kubernetes is eventually consistent.

**2. What does the lab prove and what does it not?**
A strong answer separates structural, behavioural, and adversarial evidence, and names at least three
absent controls. A weak answer says it proves the stack works.

**3. How would you make the lab production-ready?**
A strong answer prioritises identity, secrets, TLS, audit, and backup, and notes that some require
cluster capabilities rather than manifests. A weak answer lists tools.

**4. What is the first thing you check when a pod is Pending?**
A strong answer reads the scheduler event and compares the request with allocatable. A weak answer
says to check the logs, which a Pending pod does not have.

**5. Why is the HPA check failing?**
A strong answer identifies the missing metrics API and names `kubectl top pods` as the confirmation.
A weak answer says the HPA is misconfigured.

**6. How do you know the network policy is enforced?**
A strong answer describes a negative test from a blocked namespace. A weak answer says the object
exists.

**7. What does teardown order protect against?**
A strong answer names orphaned load balancers blocking subnet deletion and the cost of leaving
resources. A weak answer says it is tidier.

**8. How would you cost this stack for a class of 30 learners?**
A strong answer multiplies the per-session cost, notes that sessions overlap, and recommends
per-learner accounts or a shared cluster with namespaces. A weak answer multiplies by 30 without
considering concurrency.

**9. What would you change first to reduce cost?**
A strong answer identifies the control plane and NAT as fixed costs and recommends stopping the
environment between sessions. A weak answer says to use smaller nodes.

**10. What evidence would you give an auditor?**
A strong answer names systems of record for access, audit, integrity, and transmission, and notes
which are absent in the lab. A weak answer says the manifests.

## 15. Key terms

- **Acceptance criterion** — Checkable completion condition.
- **Adversarial test** — Test that a control fails when it should.
- **Blast radius** — Scope affected by a failure.
- **Bootstrap address** — Broker address given to a client.
- **Context** — Named cluster and user in kubeconfig.
- **Dead-letter queue** — Destination for unprocessable records.
- **Drift** — Difference between declared and actual state.
- **Idempotent apply** — Re-running produces no change.
- **Image digest** — Content hash of an image.
- **Kubeconfig** — File describing cluster access.
- **Lab scaffold** — Teaching implementation, not production.
- **Namespace** — Logical partition of a cluster.
- **Phase** — Group of steps with shared preconditions.
- **Precondition** — State required before a step.
- **Production readiness** — Evidence a system can operate under load and failure.
- **Readiness** — Whether a pod can serve traffic.
- **Replay** — Reprocessing retained events.
- **Resource quota** — Aggregate namespace limit.
- **Rollback** — Return to a previous known state.
- **Runbook** — Documented response to an alert.
- **Service endpoint** — Address behind a Service.
- **Smoke test** — Minimal check that a system responds.
- **Teardown** — Removal of created resources.
- **Walkthrough** — Ordered execution with acceptance checks.

## 16. Sources

All sources accessed 26 October 2026.

1. AWS. *Amazon EKS — Kubernetes version lifecycle*.
   https://docs.aws.amazon.com/eks/latest/userguide/kubernetes-versions.html
2. AWS. *Amazon EKS — Getting started*.
   https://docs.aws.amazon.com/eks/latest/userguide/getting-started.html
3. AWS. *EC2 On-Demand pricing*, US East (Ohio). https://aws.amazon.com/ec2/pricing/on-demand/
4. AWS. *Amazon VPC pricing*. https://aws.amazon.com/vpc/pricing/
5. AWS. *Amazon S3 pricing*. https://aws.amazon.com/s3/pricing/
6. Terraform Registry. *terraform-aws-modules/eks/aws*.
   https://registry.terraform.io/modules/terraform-aws-modules/eks/aws/latest
7. Kubernetes. *Deployments*.
   https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
8. Kubernetes. *Service*.
   https://kubernetes.io/docs/concepts/services-networking/service/
9. Redpanda. *Linux quickstart*. https://docs.redpanda.com/current/get-started/quick-start/
10. HHS. *HIPAA Security Rule*.
    https://www.hhs.gov/hipaa/for-professionals/security/index.html
11. AWS. *Healthcare Industry Lens*.
    https://docs.aws.amazon.com/wellarchitected/latest/healthcare-industry-lens/healthcare-industry-lens.html
12. Kubernetes. *Production environment*.
    https://kubernetes.io/docs/setup/production-environment/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`08_cost-and-reliability.md`](08_cost-and-reliability.md)
- Related topic: [`01_architecture-patterns.md`](01_architecture-patterns.md)
- Related topic: [`03_kubernetes.md`](03_kubernetes.md)
- Survey file: [`../../course/26_cloud_streaming_production_labs.md`](../../course/26_cloud_streaming_production_labs.md)
- Lab root: [`../../labs/`](../../labs/)
- Reference stack: [`../../labs/cloud/`](../../labs/cloud/)
- Lab README: [`../../labs/cloud/README.md`](../../labs/cloud/README.md)
- Terraform root: [`../../labs/cloud/terraform/main.tf`](../../labs/cloud/terraform/main.tf)
- Kubernetes manifests: [`../../labs/cloud/k8s/00-namespace.yaml`](../../labs/cloud/k8s/00-namespace.yaml)
- Producer: [`../../labs/cloud/kafka/producer.py`](../../labs/cloud/kafka/producer.py)
- Consumer: [`../../labs/cloud/kafka/consumer.py`](../../labs/cloud/kafka/consumer.py)
- Makefile: [`../../labs/cloud/Makefile`](../../labs/cloud/Makefile)
