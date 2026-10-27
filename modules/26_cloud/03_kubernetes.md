# Kubernetes

> Module 26 — Cloud, Kubernetes, and Streaming Production Labs
> Topic 03 of 09
> Estimated study time: 4 hours
> Prerequisites: 02
> Version 2026.3

## 1. Why this topic exists

Kubernetes is a scheduling and reconciliation system. You declare a desired state, and a control
loop works to make the observed state match it. That is the whole idea. Everything else, including
the parts that cause incidents, follows from the gap between the desired state and the observed
state: a pod that cannot be scheduled because no node has enough unrequested CPU, a readiness probe
that never passes because the application is waiting on a database, a rollout that stalls because a
pod disruption budget will not allow an eviction, or a service that receives no traffic because a
network policy silently drops it.

For a biomedical data team, Kubernetes is usually inherited rather than chosen. A platform team
provides a cluster, and the data team owns a namespace. That arrangement works only if the data team
understands four things: how requests become scheduling decisions, how probes become traffic
decisions, how budgets become availability decisions, and how policies become connectivity
decisions. This topic covers those four, using the manifests in `labs/cloud/k8s/` as the worked
example.

### 1.1 The problems that appear as something else

| Symptom reported | Actual mechanism | Where this topic covers it |
|---|---|---|
| "The service is down" | Readiness probe failing, endpoints removed | Section 5, Step 4 |
| "The new version never rolled out" | `maxUnavailable: 0` plus insufficient spare capacity | Section 5, Step 5 |
| "Some pods are pending forever" | Sum of requests exceeds allocatable CPU | Section 5, Step 2 |
| "The pod keeps restarting" | Liveness probe too aggressive during startup | Section 5, Step 4 |
| "Node maintenance is stuck" | PodDisruptionBudget (PDB) cannot be satisfied, eviction returns 429 | Section 5, Step 9 |
| "One zone has all the replicas" | No topology spread constraint | Section 5, Step 10 |
| "The service cannot reach the database" | Default-deny egress without an allow rule | Section 5, Step 11 |
| "Autoscaling does nothing" | Metrics server absent, or no CPU requests | Section 5, Step 8 |

Every row is a configuration property, not an application bug. That is why this topic is about
manifests.

### 1.2 The version problem in the reference stack

`labs/cloud/terraform/variables.tf` defaults `cluster_version` to `"1.30"`. As of 26 October 2026,
Amazon Elastic Kubernetes Service (EKS) lists 1.34, 1.35, 1.36, and 1.37 in standard support, and 1.31, 1.32, and 1.33 in extended
support. Version 1.30 is past the end of extended support and is not listed at all. A cluster
created with the default would either be rejected or auto-upgraded. Raise the default before
running the lab; the manifest content is unaffected, but the version claim in the lab is stale.

This is a useful lesson in itself. Version pinning is necessary, and a pin that nobody revisits
becomes a liability. The Terraform module reference also matters: the module pins
`terraform-aws-modules/eks/aws` at `~> 20.0`, while the registry's current release line is 21.x.
Section 4 of [`04_infrastructure-as-code.md`](04_infrastructure-as-code.md) covers the upgrade.

> **Note:** The lab stack is a teaching scaffold. It is not a compliant deployment and does not
> provide encryption in transit, authentication at the ingress, or a production secrets path.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compute** whether a set of pods fits on a set of nodes, using requests, allocatable capacity,
   and DaemonSet overhead.
2. **Design** startup, readiness, and liveness probes that distinguish "not started", "not ready",
   and "wedged", and state the failure consequence of each.
3. **Predict** the horizontal pod autoscaler's replica count for a given utilisation, target, and
   tolerance, and explain why it does not react when metrics are missing.
4. **Explain** what a pod disruption budget protects and what it does not, including the difference
   between a voluntary eviction and an involuntary failure.
5. **Write** a default-deny network policy with explicit DNS and application allowances, and state
   the container network interface requirement.
6. **Diagnose** a pending pod, a crash loop, a stalled rollout, and a blocked node drain from
   `kubectl` output.
7. **Distinguish** a lab manifest from a production manifest and name the specific changes.

Outcomes 1 and 3 are assessed by Exercises 12.2 and 12.4. Outcomes 2 and 6 by 12.3 and 12.7.
Outcomes 4, 5, and 7 by 12.5, 12.6, and 12.8.

## 3. The core mental model

Kubernetes has three independent control loops that matter for a service: scheduling, endpoint
management, and horizontal scaling. They read different inputs and fail in different ways.

```text
                    ┌──────────────────────────────┐
   pod spec ───────▶│ kube-scheduler               │
   (requests)       │ fits requests to allocatable │──▶ node assignment
                    └──────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ kubelet                      │
                    │ starts container, runs probes│
                    └──────────────────────────────┘
                          │                  │
                 ready?   │                  │  alive?
                          ▼                  ▼
              ┌────────────────────┐  ┌──────────────────┐
              │ EndpointSlice      │  │ restart container│
              │ controller         │  │ per restartPolicy│
              └────────────────────┘  └──────────────────┘
                          │
                          ▼
                    Service VIP / ingress

   separate loop:
   ┌───────────────────────────────────────────────────────┐
   │ metrics-server ──▶ HPA controller ──▶ scale subresource│
   └───────────────────────────────────────────────────────┘
```

The scheduler sees requests, never actual usage. The kubelet sees actual usage but does not move
pods. The HPA sees usage relative to requests, which means a pod without requests is invisible to
it. The endpoint controller sees readiness, not liveness. Keep these four inputs separate and most
confusing behaviour becomes predictable.

### 3.1 Requests are a scheduling currency

A request is a claim on a node's allocatable resources. The scheduler sums the requests of all pods
already assigned to a node and admits a new pod only if the remaining capacity covers its request.
Actual usage is irrelevant to this decision.

```text
node allocatable: 4000m CPU, 16384 MiB
daemonsets:        700m CPU,  2560 MiB
usable:           3300m CPU, 13824 MiB

pod requests:      250m CPU,   512 MiB
pods that fit:     min(3300/250, 13824/512) = min(13, 27) = 13
```

The node is CPU-bound at thirteen pods even though it has memory for twenty-seven. Adding memory
does not help. Reducing the CPU request does.

### 3.2 Limits are a runtime currency

A CPU limit is enforced by the kernel scheduler as throttling. A memory limit is enforced by the
kernel out-of-memory killer. They behave differently, and the difference matters:

| Resource | Limit exceeded | Observable symptom | Latency effect |
|---|---|---|---|
| CPU | Process throttled for the remainder of the period | `container_cpu_cfs_throttled_periods_total` rises | Increased tail latency, no error |
| Memory | Process killed | `OOMKilled` in container state | Restart, connection reset |

A CPU limit that equals the request gives predictable latency and no burst. A CPU limit much higher
than the request allows bursting but makes capacity planning harder. The reference deployment sets
requests of 250m/512Mi and limits of 1 CPU/1Gi, which is a four-to-one CPU overcommit ratio.

### 3.3 Probes are traffic and lifecycle decisions

Each probe answers a different question:

- **Startup**: has the process finished initialising. Until it succeeds, the other probes do not
  run.
- **Readiness**: should this pod receive traffic right now. Failure removes the pod from
  EndpointSlices; it does not restart the container.
- **Liveness**: is this container wedged in a way that only a restart fixes. Failure restarts the
  container.

The defaults are `periodSeconds: 10`, `timeoutSeconds: 1`, `failureThreshold: 3`,
`successThreshold: 1`, and `initialDelaySeconds: 0`. A liveness probe with defaults therefore
restarts a container after roughly 30 seconds of failure. If the application takes 60 seconds to
start, and the liveness probe has no startup probe in front of it, the container is killed during
startup, restarts, and repeats. The log shows a normal startup followed by a kill, which is the
signature of this mistake.

### 3.4 The reconciliation gap

Every Kubernetes failure is a gap between desired and observed state. The gap is visible in
`status`, and the reason is usually in an event. The discipline is to read the condition, not to
guess:

```text
kubectl get pods -n biomed
kubectl describe pod <pod> -n biomed        # events, last state, exit code
kubectl get events -n biomed --sort-by=.lastTimestamp
kubectl describe node <node>                # Allocated resources, Conditions
```

Most incident time is spent guessing without these four commands.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Pod | Smallest schedulable unit: one or more containers sharing a network namespace | A container | Object |
| Deployment | Controller managing replicated stateless pods and rollouts | A pod | Object |
| StatefulSet | Controller with stable identity and storage per replica | A Deployment for databases | Object |
| Service | Stable virtual IP and DNS name for a set of pods | A load balancer | Object |
| Ingress | HTTP routing rule from outside to a Service | A Service | Object |
| ConfigMap | Non-secret configuration data | A secret store | Object |
| Secret | Base64-encoded sensitive data, optionally encrypted at rest | An encrypted vault | Object |
| Request | Resources reserved for scheduling | Actual usage | Millicores, MiB |
| Limit | Maximum resources a container may use | A request | Millicores, MiB |
| Allocatable | Resources a node offers to pods after system reservation | Capacity | Millicores, MiB |
| Quality of service | Classification from requests and limits: Guaranteed, Burstable, BestEffort | A priority class | Enum |
| Liveness probe | Restart signal for a wedged container | A health dashboard | Probe |
| Readiness probe | Traffic admission signal | A liveness check | Probe |
| Startup probe | Initialisation gate that disables other probes | A long initial delay | Probe |
| EndpointSlice | The set of pod addresses a Service routes to | The Service itself | Object |
| HPA | Horizontal Pod Autoscaler | Cluster autoscaler | Object |
| Cluster autoscaler | Adds or removes nodes | HPA | Component |
| PDB | PodDisruptionBudget, limiting voluntary evictions | An availability guarantee | Object |
| Eviction API | API used by drain to respect PDBs | Pod deletion | API |
| Topology spread constraint | Rule for distributing pods across zones or nodes | Affinity | Object |
| NetworkPolicy | L3/L4 allow rules enforced by the container network interface (CNI) | A firewall for the cluster edge | Object |
| CNI | Container Network Interface plugin | The cluster network | Plugin |
| ServiceAccount | Identity for pods | A user account | Object |
| RBAC | Role-based access control | Authentication | Object |
| Pod Security Standards | Privileged, baseline, restricted profiles | A policy engine | Admission label |
| Namespace | Logical partition with quota and policy | A cluster | Object |
| ResourceQuota | Aggregate limit on requests and object counts in a namespace | A per-pod limit | Object |
| LimitRange | Defaults and bounds for pod resources in a namespace | A quota | Object |
| Termination grace period | Time between SIGTERM and SIGKILL | A probe timeout | Seconds |
| ReplicaSet | Controller ensuring a number of identical pods | A Deployment | Object |

Three distinctions matter more than the rest.

**Request versus limit.** The scheduler uses requests; the kernel uses limits. A pod with no
request is scheduled as if it needed nothing and can then starve its neighbours.

**Liveness versus readiness.** A readiness failure removes traffic. A liveness failure restarts
the process. Using a liveness probe to check a dependency causes restart storms when the dependency
is down.

**PDB versus replication.** A PDB does not create replicas. It only limits how many pods may be
evicted voluntarily. With one replica and `minAvailable: 1`, a drain can never proceed.

## 5. The mechanics, step by step

The procedure takes a container image and produces a service that schedules, serves, scales, and
survives maintenance. The order follows the manifests in `labs/cloud/k8s/`, which are numbered for
this reason.

### Step 1 — Create the namespace and the labels

**Input.** A service name and an owner.

**Output.** A namespace with labels used by policy and cost reporting.

**Rationale.** The namespace is the unit of quota, policy, and access control.
`labs/cloud/k8s/00-namespace.yaml` creates `biomed` with
`app.kubernetes.io/part-of: biomed-reference`. Labels are not decoration: network policies select
namespaces by label, Pod Security admission selects namespaces by label, and cost tools group by
label.

**Branch point.** If multiple teams share a cluster, one namespace per team plus a `ResourceQuota`
per namespace. If a single team owns the cluster, namespaces still separate environments.

### Step 2 — Set resource requests from measurement

**Input.** Observed CPU and memory use at p95, and the node's allocatable capacity.

**Output.** A request per container, recorded in the Deployment.

**Rationale.** Requests drive scheduling and HPA utilisation. Setting them too low overcommits the
node and causes throttling and eviction under load; setting them too high wastes capacity and
blocks rollouts.

**A defensible method.** Measure the container over a representative week. Set the CPU request to
the p95 usage rounded up to the nearest 50 millicores, and the memory request to the p95 working set
rounded up to the nearest 64 MiB. Do not set the memory request below the observed peak; memory is
not compressible.

**Branch point.** If the workload is bursty and latency-tolerant, a lower request with a higher
limit is acceptable. If the workload is latency-sensitive, keep the CPU limit at or near the
request to avoid throttling.

### Step 3 — Set limits and understand the quality-of-service class

**Input.** The requests and the application's tolerance for throttling and kill.

**Output.** Limits, and a known QoS class.

**Rationale.** The QoS class determines eviction order under node pressure. BestEffort pods are
evicted first, Burstable next, Guaranteed last. A streaming consumer with no limits is BestEffort
and will be the first casualty of a memory-pressured node, exactly when it is most needed.

The classes are assigned as follows:

| Class | Condition | Eviction order |
|---|---|---|
| Guaranteed | Every container has CPU and memory requests equal to limits | Last |
| Burstable | At least one container has a request or limit, but not Guaranteed | Middle |
| BestEffort | No container has any CPU or memory request or limit | First |

A common misunderstanding: setting a limit without a request copies the limit into the request and
can produce Guaranteed, not Burstable.

### Step 4 — Design the three probes

**Input.** The application's startup behaviour, dependency checks, and failure semantics.

**Output.** A startup probe, a readiness probe, and a liveness probe with explicit budgets.

**Rationale.** The three probes answer three different questions and must not be copies of each
other.

```yaml
startupProbe:
  httpGet: { path: /healthz, port: http }
  periodSeconds: 5
  failureThreshold: 30        # up to 150 s to start
readinessProbe:
  httpGet: { path: /readyz, port: http }
  periodSeconds: 10
  failureThreshold: 3         # 30 s before removal from endpoints
livenessProbe:
  httpGet: { path: /healthz, port: http }
  periodSeconds: 20
  failureThreshold: 3         # 60 s before restart
```

The reference manifest omits a startup probe and sets `initialDelaySeconds: 15` on liveness. That
is adequate for a fast-starting API and inadequate for a service that loads a large model. The
production change is a startup probe with a budget derived from the measured load time.

**Branch point.** If the readiness endpoint checks a downstream dependency, a downstream outage
removes every replica from service at once. That may be correct, if the service genuinely cannot
serve, or catastrophic, if it can serve degraded. Decide explicitly.

**Edge case.** An `exec` probe requires a shell in the image. Distroless images have none, so the
probe fails with `executable file not found`. Use `httpGet` or `grpc`.

### Step 5 — Choose the rollout strategy

**Input.** The replica count and the available spare capacity.

**Output.** A strategy with `maxSurge` and `maxUnavailable`, and a `minReadySeconds` if needed.

**Rationale.** `maxUnavailable: 0` with `maxSurge: 1` guarantees that capacity never drops during a
rollout, at the cost of needing room for one extra pod. If the cluster has no spare capacity, the
new pod stays Pending and the rollout stalls with the old pods still serving. That is safe but not
progressing, and the error is capacity, not the strategy.

**Branch point.** With a tight capacity budget, `maxSurge: 0` and `maxUnavailable: 1` allows the
rollout to proceed by taking capacity down first. It is faster and less available. The choice
depends on the service level objective (SLO), not on preference.

**Edge case.** A `preStop` sleep longer than the remaining grace period causes the kubelet to
SIGKILL the process mid-drain. Keep `preStop` plus application shutdown below
`terminationGracePeriodSeconds`.

### Step 6 — Expose the service

**Input.** The container port and the intended clients.

**Output.** A ClusterIP Service with a named target port.

**Rationale.** A Service selects pods by label and routes to the named container port. The named
port indirection means the Service and the probes refer to a name, so a port change is one edit.
`labs/cloud/k8s/04-service.yaml` maps port 80 to `targetPort: http`.

**Branch point.** If the client is outside the cluster, an Ingress or a load balancer is needed. An
Ingress requires an ingress controller; the lab assumes NGINX and does not install it.

### Step 7 — Configure ingress and transport layer security (TLS)

**Input.** The hostname, the TLS certificate source, and the ingress class.

**Output.** An Ingress with a TLS secret reference and a redirect annotation.

**Rationale.** `labs/cloud/k8s/05-ingress.yaml` sets `ssl-redirect: "true"` and references
`risk-service-tls`. The secret must exist; the manifest does not create it. A missing secret
produces a default self-signed certificate in many controllers, which is worse than a clear
failure because it looks like it works.

**Edge case.** Certificate expiry is a credential-expiry failure with a fixed date. Automate
renewal and alert at 21 days remaining.

### Step 8 — Configure horizontal autoscaling

**Input.** The target metric, the minimum and maximum replicas, and the scale-down behaviour.

**Output.** An HPA with a resource metric and a behaviour block.

**Rationale.** The HPA computes:

```text
desiredReplicas = ceil(currentReplicas × currentMetricValue / desiredMetricValue)
```

subject to a default tolerance of 10 percent, the min and max, and the behaviour policies. The
controller syncs every 15 seconds by default. Utilisation is a percentage of the request, so a pod
without a request makes the metric undefined and the HPA does nothing.

`labs/cloud/k8s/06-hpa.yaml` sets `minReplicas: 2`, `maxReplicas: 6`, a CPU target of 70 percent,
and a 300-second scale-down stabilisation window. That is a sensible demo configuration. A
production configuration also sets scale-up policies, because the default allows a doubling every
15 seconds, which can overwhelm a downstream dependency.

**Branch point.** If the bottleneck is queue depth rather than CPU, a resource metric is the wrong
signal. Use a custom or external metric, for example consumer lag per replica, and accept the extra
operational dependency on the metrics adapter.

**Edge case.** Metrics server absent produces `<unknown>` in `kubectl get hpa`. The HPA does not
scale down on missing metrics, which is the safe behaviour but looks like a stuck autoscaler.

### Step 9 — Set a pod disruption budget

**Input.** The replica count and the availability target.

**Output.** A PDB with `minAvailable` or `maxUnavailable`.

**Rationale.** A PDB limits voluntary disruptions through the Eviction API. It does not prevent
involuntary failures, and it does not create pods.
`labs/cloud/k8s/07-pdb.yaml` sets `minAvailable: 1` with a selector for `app: risk-service`.

With two replicas, one eviction is allowed at a time. With one replica, no eviction is allowed and a
node drain blocks indefinitely, returning 429 to the eviction request. The PDB status shows
`disruptionsAllowed: 0`.

**Branch point.** `minAvailable` as a percentage rounds up, and `maxUnavailable` rounds down. With
three replicas, `minAvailable: 50%` protects two pods, not one and a half.

**Edge case.** Direct pod deletion bypasses the PDB. A `kubectl delete pod` is not an eviction, so
the budget does not apply. Only `kubectl drain` and other Eviction API clients are constrained.

### Step 10 — Spread across zones

**Input.** The node topology labels and the replica count.

**Output.** A topology spread constraint with the right `whenUnsatisfiable` policy.

**Rationale.** The scheduler does not spread pods across zones by default unless asked. The
reference manifest sets `topologyKey: topology.kubernetes.io/zone`, `maxSkew: 1`, and
`whenUnsatisfiable: ScheduleAnyway`. The last setting means the constraint is a preference, not a
requirement, so a rollout proceeds even if perfect spreading is impossible.

**Branch point.** `DoNotSchedule` enforces the constraint and can leave pods Pending when the
cluster has fewer zones than the replica count. Use it only when the availability requirement
justifies the scheduling failure.

**Edge case.** Spreading across zones does not help if the Service has no pod disruption budget and
a node drain removes both replicas in one zone at once. The controls compose.

### Step 11 — Apply default-deny network policy

**Input.** The allowed ingress sources and egress destinations.

**Output.** A default-deny policy for the namespace and explicit allow policies per workload.

**Rationale.** Kubernetes network policy is additive and default-allow. Until a pod is selected by
a policy, all traffic is permitted. `labs/cloud/k8s/08-networkpolicy.yaml` creates `default-deny`
with `podSelector: {}` and both policy types, then allows ingress from the ingress-nginx and
monitoring namespaces on port 8000, DNS egress to kube-dns, and HTTPS egress to the internet.

The DNS allowance is mandatory. A default-deny egress policy without it breaks name resolution, and
the symptom is every outbound call failing with a timeout while the policy looks correct.

**Edge case.** Network policy is enforced by the CNI, not by the API server. A cluster whose plugin
does not implement policy accepts the objects and ignores them. Confirm enforcement with a test
connection from a pod that should be blocked.

**Edge case.** The reference egress rule allows `0.0.0.0/0` on port 443. That is broad. A production
policy names the database endpoint or uses a private endpoint and a narrower rule.

### Step 12 — Bind a service account with no API token

**Input.** The workload's cloud identity requirement.

**Output.** A ServiceAccount, a workload identity association, and
`automountServiceAccountToken: false`.

**Rationale.** A mounted service account token lets any process in the container call the
Kubernetes API with the pod's RBAC. Most application pods do not need it.
`labs/cloud/k8s/09-serviceaccount.yaml` sets the flag to false. The deployment references the
account by name.

Cloud identity is separate. On EKS, either IAM Roles for Service Accounts or EKS Pod Identity
provides AWS credentials. The lab Terraform sets `enable_irsa = true` but the Kubernetes manifests
do not annotate the service account with a role ARN, so the pod has no AWS permissions. That is
correct for the demo and must change before the service reads from S3.

### Step 13 — Verify the rollout and the failure behaviour

**Input.** The applied manifests.

**Output.** Evidence that the service is ready, that a pod deletion does not drop below the
minimum, and that the policy blocks what it should.

**Rationale.** Applying a manifest is not deploying a service. The acceptance checks in the survey
file are the right ones:

```bash
kubectl rollout status deploy/risk-service -n biomed
kubectl get endpoints risk-service -n biomed
kubectl delete pod -n biomed -l app=risk-service --wait=false
kubectl get pods -n biomed -w
kubectl run nettest --rm -it --image=curlimages/curl -n default -- \
  curl -m 3 http://risk-service.biomed.svc.cluster.local/healthz
```

The last command should fail from an unapproved namespace. If it succeeds, the network policy is
not enforced.

### 5.1 Edge cases that change the answer

**Node pressure.** Memory pressure triggers kubelet eviction. The eviction order is BestEffort,
then Burstable pods exceeding their requests, then Guaranteed. A service with requests far below
its usage is the first to go, even though its manifest looks reasonable.

**Zone failure.** Losing a zone removes every pod on those nodes at once. A topology spread
constraint with `maxSkew: 1` and three zones keeps the loss to roughly one third. Without it, the
scheduler may have placed all replicas in one zone because that zone had capacity.

**Quota exhaustion.** A `ResourceQuota` on the namespace caps the sum of requests. A rollout that
needs one extra pod can fail with `exceeded quota` even though the cluster has free capacity. The
error appears on the ReplicaSet, not on the node.

**Credential expiry.** An image pull secret or a projected service account token expires. New pods
fail while running pods continue. The failure is delayed and correlates with a rollout, not with
the credential change.

**Probe flapping.** A readiness probe with a one-second timeout against a service whose p99 is 1.2
seconds removes healthy pods intermittently. The endpoint count oscillates, and the remaining pods
take more load, which increases latency, which removes more pods. This is a self-reinforcing
failure and is fixed by raising the timeout or reducing the check's cost.

**HPA and cluster autoscaler disagreement.** The HPA requests more replicas, but no node has room,
so pods stay Pending. The cluster autoscaler adds a node if the node group maximum allows it. If
the maximum is reached, the HPA keeps requesting and nothing happens. Alert on pending pods, not
only on replica count.

### 5.2 A decision table for probe design

| Situation | Startup | Readiness | Liveness | Consequence of getting it wrong |
|---|---|---|---|---|
| Fast stateless API | Optional, 5 s budget | Dependency-free check | Process check | Restart loop if liveness checks a dependency |
| Model loading at start | Required, measured budget | Model loaded and warm | Process check | Kill during load, endless restart |
| Consumer with no HTTP server | Optional | Not applicable | Heartbeat file or exec | No traffic concept; liveness only |
| Service with a degraded mode | Optional | Ready if it can serve degraded | Process check | Removing all pods during a dependency outage |
| Batch Job | Not applicable | Not applicable | Usually none | Restarting a job that should fail |

## 6. Worked example

A namespace runs three workloads: `risk-service` at six replicas with 250m CPU and 512 MiB memory
requests, `feature-api` at four replicas with 500m and 1024 MiB, and `adt-consumer` at three
replicas with 1000m and 2048 MiB. The platform team offers `m7g.xlarge` nodes. The question is how
many nodes are needed, what the HPA will do under load, and whether maintenance can proceed.

### 6.1 Inputs

| Parameter | Value | Source |
|---|---|---|
| Node type | `m7g.xlarge`, 4 vCPU, 16 GiB | Candidate |
| Node price | USD 0.1632/hour | AWS Price List, us-east-2, October 2026 |
| System and DaemonSet reservation | 700m CPU, 2560 MiB per node | Planning assumption |
| Total requests, CPU | 6500m | 6×250 + 4×500 + 3×1000 |
| Total requests, memory | 13,312 MiB | 6×512 + 4×1024 + 3×2048 |
| HPA | min 2, max 6, target 70 percent CPU | `labs/cloud/k8s/06-hpa.yaml` |
| PDB | `minAvailable: 1` | `labs/cloud/k8s/07-pdb.yaml` |
| Availability service level objective (SLO) | 99.9 percent | Proposed |

All costs are USD, us-east-2, for 730 hours.

### 6.2 Arithmetic: usable capacity per node

```text
allocatable CPU    = 4000m
reservation        =  700m
usable CPU         = 3300m

allocatable memory = 16384 MiB
reservation        =  2560 MiB
usable memory      = 13824 MiB
```

Pods that fit, per workload, on one node:

```text
risk-service:  min(3300 ÷ 250, 13824 ÷ 512)  = min(13, 27) = 13
feature-api:   min(3300 ÷ 500, 13824 ÷ 1024) = min(6, 13)  = 6
adt-consumer:  min(3300 ÷ 1000, 13824 ÷ 2048)= min(3, 6)   = 3
```

The node is CPU-bound for every workload. Memory is not the constraint at these ratios.

### 6.3 Arithmetic: nodes required

Aggregate method:

```text
CPU:    ceil(6500 ÷ 3300) = ceil(1.97) = 2 nodes
Memory: ceil(13312 ÷ 13824) = 1 node
```

Two nodes fit the requests on paper, at 6500 ÷ 6600 = 98.5 percent of usable CPU. That leaves no
room for a surge pod, no room for a node failure, and no room for the scheduler's imperfect
packing. The practical answer is three nodes:

```text
usable CPU across 3 nodes = 9900m
utilisation               = 6500 ÷ 9900 = 65.7 percent
```

Cost:

```text
3 × USD 0.1632/hour × 730 hours = USD 357.41 per month
```

Two nodes would cost USD 238.27 per month and would fail the first rollout that used
`maxSurge: 1`.

### 6.4 Arithmetic: the HPA under load

At two replicas and 80 percent CPU utilisation against a 70 percent target:

```text
ratio = 80 ÷ 70 = 1.143
```

The ratio exceeds the 10 percent tolerance, so the controller acts:

```text
desired = ceil(2 × 80 ÷ 70) = ceil(2.286) = 3 replicas
```

At 140 percent, which is possible because the CPU limit is four times the request:

```text
desired = ceil(2 × 140 ÷ 70) = 4 replicas
```

At 200 percent, the maximum caps it at 6. The HPA can request six replicas while the Deployment
starts at two, so the namespace must have room for six. Six risk-service pods at 250m add 1000m,
which fits in the third node's spare capacity.

### 6.5 Arithmetic: the availability budget

A 99.9 percent SLO over 30 days:

```text
30 × 24 × 60 = 43,200 minutes
43,200 × 0.001 = 43.2 minutes of allowed unavailability
```

A node drain that takes one of two replicas out of service for five minutes does not breach the
budget if the remaining replica absorbs the load. It breaches the budget if the remaining replica
cannot, which is a capacity question, not a PDB question. The PDB guarantees that one replica
remains; it does not guarantee that one replica is enough.

### 6.6 Two diagnostics that reveal a misconfiguration

**Diagnostic 1: the HPA shows `<unknown>` for its target.**

```bash
kubectl get hpa risk-service -n biomed
kubectl describe hpa risk-service -n biomed
```

Output includes `failed to get cpu utilization: unable to get metrics for resource cpu`. The cause
is a missing metrics server or a container without a CPU request. The HPA is not broken; its input
is absent. Confirm with `kubectl top pods -n biomed`, which uses the same metrics API.

**Diagnostic 2: a pod is Pending with an insufficient CPU event.**

```bash
kubectl describe pod <pod> -n biomed | sed -n '/Events/,$p'
```

Output includes `0/3 nodes are available: 3 Insufficient cpu`. The scheduler is reporting that the
sum of requests on every node plus this pod's request exceeds allocatable. The fix is to add a node,
lower the request, or remove a workload. Raising the limit does nothing, because the scheduler never
reads limits.

A third diagnostic worth knowing: a rollout that reports `Waiting for deployment ... rollout to
finish: 1 old replicas are pending termination`. With `maxUnavailable: 0` and no spare capacity,
the new pod is Pending and the old pod cannot be removed. The Deployment is healthy and stuck. The
event on the new pod is the same `Insufficient cpu`.

### 6.7 Vary one assumption

Change the `risk-service` CPU request from 250m to 500m, matching its observed p95 rather than a
rounded-down guess. Total CPU requests become:

```text
6 × 500 + 4 × 500 + 3 × 1000 = 3000 + 2000 + 3000 = 8000m
```

Nodes required:

```text
ceil(8000 ÷ 3300) = ceil(2.42) = 3 nodes
```

Three nodes still fit, at 8000 ÷ 9900 = 80.8 percent utilisation. The headroom for a surge pod
falls from 3400m to 1900m, which is still more than one 500m pod but less than two. The change is
affordable here because the workload was under-requesting. If the request had been raised on a
larger workload, the node count would have crossed a boundary and the monthly cost would have
jumped by USD 119.14 for the fourth node.

A second variation exposes the cost of the availability requirement. Drop the topology spread
constraint and the scheduler may place all three nodes in one zone if that zone has capacity. The
monthly cost is unchanged, and the blast radius of a zone failure rises from one third of the
service to all of it. The spread constraint costs nothing in this cluster because three nodes in
three zones satisfy it; in a cluster where it forces a fourth node, the cost is the node.

## 7. Code, query, or calculation

The following standard-library Python model computes node fit, node count, cost, and HPA response.
It reproduces every number in Section 6.

```python
"""Capacity and autoscaling model for a small clinical service set. Stdlib only."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    name: str
    vcpu_milli: int
    memory_mib: int
    hourly_usd: float


@dataclass(frozen=True)
class Workload:
    name: str
    replicas: int
    cpu_milli: int
    memory_mib: int
    cpu_limit_milli: int
    memory_limit_mib: int


NODES = {
    "t3.medium": Node("t3.medium", 2000, 4096, 0.0416),
    "m7g.large": Node("m7g.large", 2000, 8192, 0.0816),
    "m7g.xlarge": Node("m7g.xlarge", 4000, 16384, 0.1632),
    "m7i.xlarge": Node("m7i.xlarge", 4000, 16384, 0.2016),
}

# Planning reservation for kubelet, system, and DaemonSets (CNI, kube-proxy,
# CoreDNS is a Deployment but counted here for simplicity, pod identity agent).
RESERVE_CPU_MILLI = 700
RESERVE_MEMORY_MIB = 2560

WORKLOADS = [
    Workload("risk-service", 6, 250, 512, 1000, 1024),
    Workload("feature-api", 4, 500, 1024, 1000, 2048),
    Workload("adt-consumer", 3, 1000, 2048, 2000, 4096),
]


def usable(node: Node) -> tuple[int, int]:
    return (node.vcpu_milli - RESERVE_CPU_MILLI, node.memory_mib - RESERVE_MEMORY_MIB)


def fit_count(node: Node, pod: Workload) -> int:
    cpu, mem = usable(node)
    return min(cpu // pod.cpu_milli, mem // pod.memory_mib)


def nodes_needed(node: Node) -> int:
    cpu = sum(w.replicas * w.cpu_milli for w in WORKLOADS)
    mem = sum(w.replicas * w.memory_mib for w in WORKLOADS)
    ucpu, umem = usable(node)
    return max(math.ceil(cpu / ucpu), math.ceil(mem / umem))


def utilisation(node: Node, count: int) -> tuple[float, float]:
    cpu = sum(w.replicas * w.cpu_milli for w in WORKLOADS) / (count * node.vcpu_milli)
    mem = sum(w.replicas * w.memory_mib for w in WORKLOADS) / (count * node.memory_mib)
    return cpu, mem


def hpa_desired(current: int, utilisation_pct: float, target_pct: float,
                tolerance: float = 0.10, minimum: int = 2, maximum: int = 6) -> int:
    ratio = utilisation_pct / target_pct
    if abs(ratio - 1.0) <= tolerance:
        return current
    return max(minimum, min(maximum, math.ceil(current * ratio)))


def main() -> None:
    print("node          allocatable(cpu/mem)  pods risk  pods feature  pods consumer")
    for node in NODES.values():
        cpu, mem = usable(node)
        print(f"{node.name:13} {cpu:5d}m / {mem:6d}Mi     "
              f"{fit_count(node, WORKLOADS[0]):3d}        "
              f"{fit_count(node, WORKLOADS[1]):3d}            "
              f"{fit_count(node, WORKLOADS[2]):3d}")

    print()
    for name in ("m7g.xlarge", "m7i.xlarge", "t3.medium"):
        node = NODES[name]
        n = nodes_needed(node)
        cpu, mem = utilisation(node, n)
        cost = n * node.hourly_usd * 730
        print(f"{name:12} nodes={n}  cpu_util={cpu:5.1%}  mem_util={mem:5.1%}  "
              f"cost={cost:8.2f} USD/month")

    print()
    print("HPA response, risk-service at 2 replicas")
    for util in (60, 70, 75, 80, 140, 200):
        print(f"  current={util:3d}% target=70%  desired={hpa_desired(2, util, 70)}")

    print()
    print("HPA response with target 50%")
    for util in (60, 75, 100, 150):
        print(f"  current={util:3d}% target=50%  desired={hpa_desired(2, util, 50)}")


if __name__ == "__main__":
    main()
```

Actual output:

```text
node          allocatable(cpu/mem)  pods risk  pods feature  pods consumer
t3.medium      1300m /   1536Mi       3          1              0
m7g.large      1300m /   5632Mi       5          2              1
m7g.xlarge     3300m /  13824Mi      13          6              3
m7i.xlarge     3300m /  13824Mi      13          6              3

m7g.xlarge   nodes=2  cpu_util=81.2%  mem_util=40.6%  cost=  238.27 USD/month
m7i.xlarge   nodes=2  cpu_util=81.2%  mem_util=40.6%  cost=  294.34 USD/month
t3.medium    nodes=9  cpu_util=36.1%  mem_util=36.1%  cost=  273.31 USD/month

HPA response, risk-service at 2 replicas
  current= 60% target=70%  desired=2
  current= 70% target=70%  desired=2
  current= 75% target=70%  desired=2
  current= 80% target=70%  desired=3
  current=140% target=70%  desired=4
  current=200% target=70%  desired=6

HPA response with target 50%
  current= 60% target=50%  desired=3
  current= 75% target=50%  desired=3
  current=100% target=50%  desired=4
  current=150% target=50%  desired=6
```

The model's `cpu_util` divides by total node vCPU, not usable vCPU, which is why it reports 81.2
percent for two nodes rather than 98.5 percent. Both are true statements about different
denominators. State the denominator whenever you quote a utilisation figure.

The following manifest fragment shows the production changes to the reference deployment. It is
validated by parsing, not applied.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: risk-service
  namespace: biomed
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0
      maxSurge: 1
  template:
    spec:
      terminationGracePeriodSeconds: 45
      containers:
        - name: api
          image: 123456789012.dkr.ecr.us-east-2.amazonaws.com/biomed-app@sha256:0000000000000000000000000000000000000000000000000000000000000000
          resources:
            requests:
              cpu: "250m"
              memory: "512Mi"
            limits:
              cpu: "1"
              memory: "1Gi"
          startupProbe:
            httpGet:
              path: /healthz
              port: http
            periodSeconds: 5
            failureThreshold: 30
          readinessProbe:
            httpGet:
              path: /readyz
              port: http
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
            periodSeconds: 20
            timeoutSeconds: 3
            failureThreshold: 3
```

## 8. Data, standards, and regulatory context

### 8.1 Kubernetes is not a compliance control

No clause of the Health Insurance Portability and Accountability Act (HIPAA) Security Rule names Kubernetes. The Rule constrains outcomes, and a cluster
can help or hinder them:

| Clause | Constraint | Kubernetes mechanism | Limit |
|---|---|---|---|
| 164.312(a)(1) Access control | Limit access to electronic protected health information (PHI) | RBAC, NetworkPolicy, Pod Security | RBAC controls the API, not the data path |
| 164.312(a)(2)(i) Unique user identification | Identify each user | ServiceAccount per workload | A service account is not a human user |
| 164.312(b) Audit controls | Record activity | API audit log, application logs | Audit log retention is a platform decision |
| 164.312(c)(1) Integrity | Protect against alteration | Read-only root filesystem, immutable images | Does not cover data at rest |
| 164.312(e)(1) Transmission security | Protect PHI in transit | NetworkPolicy, service mesh mutual transport layer security (mTLS) | NetworkPolicy is not encryption |

The distinction between a control that contributes and a control that satisfies is the one
auditors test.

### 8.2 Pod Security Standards

Kubernetes defines three Pod Security Standards profiles: Privileged, Baseline, and Restricted.
The Restricted profile requires non-root execution, a seccomp profile, dropped capabilities, and
disallows privilege escalation. The reference deployment in `labs/cloud/k8s/03-deployment.yaml`
satisfies most of Restricted:

- `runAsNonRoot: true`
- `seccompProfile.type: RuntimeDefault`
- `allowPrivilegeEscalation: false`
- `readOnlyRootFilesystem: true`
- `capabilities.drop: ["ALL"]`

It does not set a numeric `runAsUser`, which some policy engines require. Enforce the profile with
a namespace label:

```yaml
metadata:
  labels:
    pod-security.kubernetes.io/enforce: restricted
```

Labelling an existing namespace with `enforce: restricted` can block future pods. Test with `warn`
and `audit` first.

### 8.3 NIST SP 800-207 and network location

The second tenet of zero trust is that all communication is secured regardless of network
location. A Kubernetes cluster is a network location. Pod-to-pod traffic inside it is not
trustworthy by virtue of being inside. NetworkPolicy is a segmentation control, not an
authentication control; it limits reachability but does not establish identity. A production design
adds mutual TLS with workload identity, which the lab stack does not.

### 8.4 CIS Kubernetes Benchmark

The Center for Internet Security publishes a benchmark for Kubernetes with recommendations on
control plane configuration, kubelet settings, and RBAC. EKS shares responsibility: AWS secures the
control plane, and the customer secures nodes, workloads, and identity. A CIS scan of a managed
cluster therefore reports on the customer's share, not on AWS's.

### 8.5 What none of these settle

They do not tell you the replica count, the probe timings, or the CPU request. Those are engineering
decisions constrained by an SLO and a budget. A cluster can be fully compliant with every framework
and still drop every request during a rollout.

## 9. Industry practice and case evidence

### 9.1 Documented behaviour

**HPA semantics are documented precisely.** The controller syncs every 15 seconds by default,
applies a 10 percent tolerance, and uses a 300-second scale-down stabilisation window by default.
Utilisation is computed against the request, and a pod without a request makes the metric
undefined. These are not implementation details; they determine whether autoscaling works.

**PDBs protect evictions, not pods.** The Kubernetes documentation states that deleting a
Deployment or a pod directly bypasses the PDB, and that an unsatisfiable PDB causes the eviction to
be rejected with HTTP 429. A drain retries until it times out, which is why a `minAvailable` equal
to the replica count blocks maintenance.

**NetworkPolicy requires CNI support.** The Kubernetes documentation states that creating a
NetworkPolicy without a plugin that implements it has no effect. Calico's documentation makes the
same point more bluntly. This is the most common false sense of security in a Kubernetes
deployment.

**EKS does not upgrade nodes with the control plane.** The EKS version lifecycle documentation
states that managed node groups are not automatically upgraded when the control plane is upgraded,
and recommends keeping control plane and nodes on the same version. A cluster that is upgraded
without a node group update runs a version skew that Kubernetes supports for up to three minors,
not indefinitely.

### 9.2 Public case evidence

Kubernetes incident write-ups from healthcare organisations are rare. The public record is mostly
from platform vendors and from the Kubernetes project itself:

1. **The 2018 Kubernetes API server outage during a etcd quota event** is documented in the
   project's own postmortem. It illustrates that the control plane is a distributed system with its
   own failure modes, independent of workloads.
2. **The 2020 Argo CD and Kubernetes privilege escalation advisories** (CVE-2020-8554 and related)
   illustrate that multi-tenant clusters need policy, not only RBAC.
3. **The Kubernetes 1.25 removal of PodSecurityPolicy** replaced a built-in admission control with
   the Pod Security Standards and an external policy engine. Clusters that relied on the removed
   API had to migrate; the lesson is that cluster-level policy has a lifecycle too.

### 9.3 Inference

The following are inferences, labelled as such:

- Most stalled rollouts in production are capacity problems disguised as deployment problems. The
  Deployment controller is behaving correctly; the cluster has no room.
- Teams that set `maxUnavailable: 0` without adding spare capacity create a system that is safe
  during rollout and unable to roll out. The two settings must be chosen together.
- The most common NetworkPolicy mistake is forgetting DNS egress, because the resulting failure is a
  timeout that looks like a service problem.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Requests below usage | Node CPU throttling under load | Scheduler overcommits | Latency spikes, eviction | Measure p95, set request from it |
| Requests above usage | Low node utilisation, pending pods | Over-reservation | Wasted capacity, stalled rollout | Right-size with usage data |
| No CPU request | HPA reports `<unknown>` | Utilisation undefined | No autoscaling | Always set requests |
| Liveness checks a dependency | Restart storm during dependency outage | Dependency failure restarts pods | Cascading outage | Liveness checks the process only |
| No startup probe | Container killed during load | Liveness budget shorter than startup | Restart loop | Startup probe with measured budget |
| Readiness always true | Traffic to unready pods | Probe does not reflect ability to serve | 5xx during startup | Check dependencies and warm-up |
| `maxUnavailable: 0` with no spare capacity | Rollout stuck at old replicas | New pod Pending | Cannot deploy | Add capacity or allow one unavailable |
| PDB equals replica count | Drain hangs, eviction 429 | No allowed disruptions | Maintenance blocked | `minAvailable` below replicas |
| No topology spread | All replicas in one zone | Scheduler packs by capacity | Zone loss removes service | Spread constraint per zone |
| Default-deny egress without DNS | All outbound calls time out | Name resolution blocked | Service appears broken | Explicit DNS allow rule |
| NetworkPolicy without CNI support | Policy objects exist, traffic flows | No enforcement | False security | Test a blocked connection |
| Service account token mounted | Token readable in the container | Default automount | API access from a compromised pod | `automountServiceAccountToken: false` |
| Image pull secret expiry | New pods fail, old pods run | Static credential expires | Rollout failure | Node identity, short-lived credentials |
| Unsupported cluster version | EKS console warning, extended support billing | Version past standard support | Cost increase, eventual auto-upgrade | Track the EKS release calendar |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Node reservation of 700m and 2560 MiB | Capacity arithmetic | Different CNI, more DaemonSets, larger kubelet reservation | Read `kubectl describe node` allocated resources |
| Pods are packed perfectly | Node-count lower bound | Fragmentation, anti-affinity, spread constraints | Compare with actual scheduling |
| Requests reflect p95 usage | Right-sizing | A new workload with no history | Measure for a week before setting |
| The CNI enforces NetworkPolicy | Segmentation claim | A plugin without policy support | Attempt a blocked connection |
| Metrics server is installed | HPA operation | Minimal cluster, missing add-on | `kubectl top pods` |
| The ingress controller exists | External access | No NGINX installed | `kubectl get ingressclass` |
| Nodes are in three zones | Zone-failure arithmetic | Two-zone region, single-zone node group | `kubectl get nodes -L topology.kubernetes.io/zone` |
| The cluster autoscaler can add nodes | HPA scale-out | Node group at maximum, quota exhausted | Scale test to the maximum |
| The application handles SIGTERM | Graceful rollout | No signal handler | Delete a pod under load and watch for 5xx |
| Probe timeouts exceed p99 latency | Stable endpoints | Slow dependency, cold cache | Compare probe timeout with measured latency |

## 12. Exercises

### 12.1 Read a pending pod

**Deliverable.** Given the event `0/3 nodes are available: 3 Insufficient cpu`, a list of the five
possible causes in order of likelihood, and the command that distinguishes each.

**Acceptance criteria.** Distinguishes request too high, node full, quota, affinity, and taint.
States why raising the limit does not help.

**Time.** 25 minutes.

### 12.2 Compute node count

**Deliverable.** For the three workloads in Section 6, compute node count and monthly cost on
`m7g.large` and `m7i.xlarge`. State the reservation assumption.

**Acceptance criteria.** Shows usable capacity, aggregate requests, and the ceiling division.
Explains any difference between the two instance types.

**Time.** 40 minutes.

### 12.3 Design probes

**Deliverable.** A probe design for a service that loads a 2 GB model in 45 seconds, depends on a
database for reads, and can serve cached predictions during a database outage.

**Acceptance criteria.** Startup budget exceeds 45 seconds. Readiness reflects degraded capability.
Liveness does not restart on database failure. Each choice is justified.

**Time.** 45 minutes.

### 12.4 Predict HPA behaviour

**Deliverable.** A table of desired replicas for current utilisation of 50, 65, 70, 78, 90, and 150
percent, at a target of 70 percent, a tolerance of 10 percent, and a range of 2 to 6.

**Acceptance criteria.** Applies the tolerance correctly. States what happens when metrics are
missing.

**Time.** 30 minutes.

### 12.5 PDB and maintenance

**Deliverable.** A written procedure for draining a node that hosts one of two `risk-service`
replicas, with `minAvailable: 1`. Include the expected eviction response and the rollback if the
drain blocks.

**Acceptance criteria.** Correctly states that one eviction is allowed. Names the 429 response when
the budget is exhausted. Does not confuse PDB with replication.

**Time.** 30 minutes.

### 12.6 Write a network policy

**Deliverable.** A default-deny policy plus an allow policy for a service that must reach a
PostgreSQL database at a known CIDR on port 5432, resolve DNS, and receive traffic from the ingress
namespace only.

**Acceptance criteria.** Both policy types declared. DNS allowed. No `0.0.0.0/0` rule. Parses as
YAML.

**Time.** 45 minutes.

### 12.7 Diagnose a crash loop

**Deliverable.** A triage sequence for a pod in `CrashLoopBackOff` that distinguishes application
error, read-only filesystem, OOM kill, failed liveness probe, and missing configuration.

**Acceptance criteria.** Names the `kubectl` command and the field or log line for each cause. Does
not recommend disabling probes as a fix.

**Time.** 35 minutes.

### 12.8 Lab to production

**Deliverable.** A diff-style list of every change needed to take `labs/cloud/k8s/` from a teaching
scaffold to a production baseline, covering image reference, secrets, TLS, network egress, service
account identity, and pod security.

**Acceptance criteria.** At least eight specific changes, each tied to a file and a line. One item
must be "this cannot be fixed in the manifest alone".

**Time.** 60 minutes.

## 13. Self-check answers

### 13.1 Pending pod causes

In order of likelihood: the pod's request exceeds the largest node's allocatable capacity; every
node's remaining allocatable is below the request; a ResourceQuota on the namespace blocks the
ReplicaSet from creating the pod; a node affinity or selector matches no node; a taint has no
tolerations. `kubectl describe pod` shows the scheduler's message, which names the first three.

### 13.2 Node count on m7g.large

Usable CPU per node: 2000 − 700 = 1300m. Memory: 8192 − 2560 = 5632 MiB. Aggregate requests are
6500m and 13,312 MiB.

```text
CPU:    ceil(6500 ÷ 1300) = ceil(5.0) = 5 nodes
Memory: ceil(13312 ÷ 5632) = ceil(2.36) = 3 nodes
```

Five nodes at USD 0.0816/hour:

```text
5 × 0.0816 × 730 = USD 297.84 per month
```

On `m7i.xlarge`, usable CPU is 3300m and memory 13,824 MiB, giving two nodes by the aggregate
formula and USD 294.34 per month. The near-equality is a coincidence of pricing and capacity, not
a general rule.

### 13.3 Probe design

Startup probe: `periodSeconds: 5`, `failureThreshold: 20`, giving 100 seconds, which exceeds the
45-second load. Readiness: checks that the model is loaded and that either the database is reachable
or the cache is warm. Liveness: checks that the HTTP server responds, with no database check, so a
database outage does not restart pods. This is the design that survives a dependency outage while
still removing genuinely unready pods from service.

### 13.4 HPA table

| Utilisation | Ratio to 70 | Within 10 percent | Desired |
|---:|---:|---|---:|
| 50 | 0.714 | no | 2 (floor) |
| 65 | 0.929 | yes | 2 |
| 70 | 1.000 | yes | 2 |
| 78 | 1.114 | no | 3 |
| 90 | 1.286 | no | 3 |
| 150 | 2.143 | no | 5 |

At 150 percent: `ceil(2 × 2.143) = ceil(4.286) = 5`. At 200 percent it would reach 6, the maximum.
When metrics are missing, the HPA takes no action and reports a `ScalingActive` condition with
`FailedGetResourceMetric`.

### 13.5 Drain procedure

`kubectl drain <node> --ignore-daemonsets --delete-emptydir-data` issues evictions. With two
replicas and `minAvailable: 1`, one eviction is allowed and the other pod continues. The evicted pod
is rescheduled elsewhere if capacity exists. If the drain blocks with a 429, the budget is exhausted;
the rollback is `kubectl uncordon <node>` and restore capacity before retrying.

### 13.6 Network policy

The policy must include `policyTypes: ["Ingress", "Egress"]`, an egress rule to the DNS namespace on
UDP 53, an egress rule to the database CIDR on TCP 5432, and an ingress rule from the ingress
namespace on the application port. A correct answer does not include a broad internet egress rule.

### 13.7 Crash loop triage

Application error: `kubectl logs --previous` shows a stack trace or a configuration error.
Read-only filesystem: the log contains `Read-only file system` and the pod spec shows
`readOnlyRootFilesystem: true`. OOM kill: `kubectl describe pod` shows `Last State: Terminated,
Reason: OOMKilled` and the exit code is 137. Failed liveness: events show `Liveness probe failed`
followed by `Killing container`. Missing configuration: the log shows a missing environment variable
or a failed ConfigMap mount.

### 13.8 Lab to production

The changes include: replace the `ghcr.io` tag with an Elastic Container Registry (ECR) digest; replace the example secret with a
secrets-manager integration or an external secrets operator; add TLS issuance and renewal; narrow
the `0.0.0.0/0` egress rule; add a Pod Identity or IAM Roles for Service Accounts (IRSA) association to the service account; add a
startup probe; set a numeric `runAsUser`; raise the cluster version from 1.30; add a ResourceQuota;
and add a policy engine for admission. The item that cannot be fixed in the manifest alone is
encryption in transit inside the cluster, which requires a service mesh or a CNI with transparent
encryption.

## 14. Interview questions

**1. What is the difference between a request and a limit?**
A strong answer says requests drive scheduling and HPA utilisation while limits are enforced at
runtime, with CPU throttling and memory kill as different consequences. A weak answer says requests
are the minimum and limits are the maximum, without the scheduler consequence.

**2. Why would a pod stay Pending when the node has free memory?**
A strong answer explains that the scheduler reads requests, that the node may be CPU-bound, and that
memory availability is irrelevant if CPU requests are exhausted. A weak answer says "the cluster is
full".

**3. What does a readiness probe do that a liveness probe does not?**
A strong answer explains endpoint removal versus container restart. A weak answer says both check
health.

**4. When does the HPA not scale?**
A strong answer lists missing metrics, missing requests, tolerance, min and max bounds, and
stabilisation windows. A weak answer says "when the cluster is full".

**5. What does a PDB guarantee?**
A strong answer says it limits voluntary evictions through the Eviction API and does not prevent
involuntary failures or create replicas. A weak answer says it guarantees availability.

**6. Why is default-deny egress risky to apply without a DNS rule?**
A strong answer explains that name resolution fails and every outbound call times out, producing a
failure that looks like an application problem. A weak answer does not mention DNS.

**7. How do you know a NetworkPolicy is enforced?**
A strong answer says you test a connection that should be blocked and confirm the CNI implements
policy. A weak answer says you check that the object exists.

**8. What happens during a node drain with `minAvailable` equal to the replica count?**
A strong answer says no eviction is allowed, the API returns 429, and the drain retries until it
times out. A weak answer says the pods are rescheduled.

**9. Why does a rollout stall with `maxUnavailable: 0`?**
A strong answer connects it to the need for one surge pod and the absence of spare capacity. A weak
answer blames the image.

**10. How would you distinguish a CPU problem from a memory problem in a running pod?**
A strong answer names throttling metrics and `OOMKilled` state, and explains that CPU degrades
latency while memory kills the process. A weak answer says "check the metrics".

## 15. Key terms

- **Allocatable** — Node resources available to pods after system reservation.
- **CNI** — Container Network Interface plugin.
- **ConfigMap** — Non-secret configuration object.
- **Deployment** — Controller for replicated stateless pods.
- **EndpointSlice** — Set of pod addresses behind a Service.
- **Eviction API** — API that respects pod disruption budgets.
- **HPA** — Horizontal Pod Autoscaler.
- **Ingress** — HTTP routing from outside the cluster.
- **kubelet** — Node agent that runs containers and probes.
- **Limit** — Maximum resource a container may use.
- **LimitRange** — Namespace defaults and bounds for resources.
- **Namespace** — Logical partition for quota and policy.
- **NetworkPolicy** — L3/L4 allow rules enforced by the CNI.
- **Node pressure** — Condition triggering pod eviction.
- **PDB** — PodDisruptionBudget.
- **Pod** — Smallest schedulable unit.
- **Pod Security Standards** — Privileged, baseline, restricted profiles.
- **Probe** — Startup, readiness, or liveness check.
- **QoS class** — Guaranteed, Burstable, BestEffort.
- **RBAC** — Role-based access control.
- **Request** — Resource reserved for scheduling.
- **ResourceQuota** — Namespace aggregate limit.
- **Rolling update** — Gradual replacement of pods.
- **Service** — Stable virtual IP and DNS name.
- **ServiceAccount** — Pod identity for API and cloud access.
- **StatefulSet** — Controller with stable identity and storage.
- **Topology spread constraint** — Rule for distributing pods across topology domains.
- **Toleration** — Permission to schedule on a tainted node.

## 16. Sources

All sources accessed 26 October 2026.

1. Kubernetes. *Pod Lifecycle*.
   https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/
2. Kubernetes. *Configure Liveness, Readiness and Startup Probes*.
   https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/
3. Kubernetes. *Horizontal Pod Autoscaling*.
   https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/
4. Kubernetes. *Disruptions*.
   https://kubernetes.io/docs/concepts/workloads/pods/disruptions/
5. Kubernetes. *Network Policies*.
   https://kubernetes.io/docs/concepts/services-networking/network-policies/
6. Kubernetes. *Pod Quality of Service Classes*.
   https://kubernetes.io/docs/concepts/workloads/pods/pod-qos/
7. Kubernetes. *Pod Security Standards*.
   https://kubernetes.io/docs/concepts/security/pod-security-standards/
8. AWS. *Amazon EKS — Kubernetes version lifecycle*.
   https://docs.aws.amazon.com/eks/latest/userguide/kubernetes-versions.html
9. AWS. *Amazon EKS — IAM roles for service accounts*.
   https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html
10. AWS. *EC2 On-Demand pricing*, US East (Ohio). https://aws.amazon.com/ec2/pricing/on-demand/
11. NIST. *SP 800-207: Zero Trust Architecture*. https://csrc.nist.gov/pubs/sp/800/207/final
12. Center for Internet Security. *CIS Kubernetes Benchmark*.
    https://www.cisecurity.org/benchmark/kubernetes
13. Tigera. *Get started with Kubernetes network policy*.
    https://docs.tigera.io/calico/latest/network-policy/get-started/kubernetes-policy/kubernetes-network-policy
14. HHS. *HIPAA Security Rule*.
    https://www.hhs.gov/hipaa/for-professionals/security/index.html

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`02_containers.md`](02_containers.md)
- Next topic: [`04_infrastructure-as-code.md`](04_infrastructure-as-code.md)
- Related topic: [`08_cost-and-reliability.md`](08_cost-and-reliability.md)
- Survey file: [`../../course/26_cloud_streaming_production_labs.md`](../../course/26_cloud_streaming_production_labs.md)
- Lab root: [`../../labs/`](../../labs/)
- Reference stack: [`../../labs/cloud/`](../../labs/cloud/)
- Namespace: [`../../labs/cloud/k8s/00-namespace.yaml`](../../labs/cloud/k8s/00-namespace.yaml)
- Deployment: [`../../labs/cloud/k8s/03-deployment.yaml`](../../labs/cloud/k8s/03-deployment.yaml)
- HPA: [`../../labs/cloud/k8s/06-hpa.yaml`](../../labs/cloud/k8s/06-hpa.yaml)
- PDB: [`../../labs/cloud/k8s/07-pdb.yaml`](../../labs/cloud/k8s/07-pdb.yaml)
- Network policy: [`../../labs/cloud/k8s/08-networkpolicy.yaml`](../../labs/cloud/k8s/08-networkpolicy.yaml)
- Service account: [`../../labs/cloud/k8s/09-serviceaccount.yaml`](../../labs/cloud/k8s/09-serviceaccount.yaml)
