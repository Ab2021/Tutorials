# Kubernetes and Terraform

> Module 30 — Simulation, Optimization, and Real-Time Production Labs  
> Topic 09 of 10  
> Estimated study time: 3–4 hours  
> Prerequisites: topics 01 and 08; Python 3, shell, containers, basic HTTP and Git  
> Version 2026.5

## 1. Why this topic exists

A supply-chain planner may need a forecast service throughout a planning cycle.
A service that works only on its author's laptop is not a dependable shared capability.
The operations team must know what is running, how to replace it, and how to undo it.
A different team must be able to reproduce the declared environment and inspect its changes.
These are deployment questions, not evidence that the forecast itself is accurate.

Kubernetes reconciles requested workload state with observed cluster state.
Terraform records and changes infrastructure through provider APIs.
Combining the two can make an environment reviewable, but neither tool makes a deployment safe by itself.
Configuration mistakes, mutable images, insufficient capacity, and insecure access remain possible.
The owner needs an explicit change process and a cleanup path as well as working commands.

This lab deploys a deliberately tiny, synthetic supply-chain response service on a local kind cluster.
A Kubernetes Deployment runs the service, a ClusterIP Service provides an internal endpoint, and Terraform manages both.
The lab scales replica count declaratively, observes it, tests the response, and destroys the resources.
The Python application in section 7 provides a deterministic fixture; it is not a trained model.
There is no business result, latency guarantee, production deployment, or cloud savings claim.

A forecast estimates uncertain demand; a plan specifies a selected action.
A target is desired performance, while a commitment is an authorized obligation.
The fixture returns a fixed illustrative forecast, not a plan, target, or commitment.
No live order, customer, employee, or supplier records belong in this lab.

## 2. Learning outcomes

After this topic you can:

1. Explain how Terraform state and Kubernetes reconciliation differ.
2. Create an isolated local cluster and identify the context before a change.
3. Apply and inspect a declared Deployment, ConfigMap, and Service.
4. Test a deterministic HTTP fixture and distinguish readiness from availability.
5. Scale by changing the declaration and explain capacity and replica limitations.
6. Identify sensitive state, image provenance, exposure, and cost risks.
7. Destroy the managed resources and independently delete the local cluster.
8. Critique a production rollout proposal without mistaking a lab for evidence.

Exercises 1–8 assess these outcomes in order.
The commands use Bash; on Windows use an approved WSL 2 distribution with Docker integration.
Other shells require appropriate quoting and environment-variable syntax.

## 3. The core mental model

```text
human-reviewed files + provider lock
                 |
          terraform plan
                 |
          terraform apply <---- local state snapshot
                 |
         Kubernetes API server
           /        |         \
     ConfigMap   Deployment   ClusterIP Service
         |           |               |
    Python file   ReplicaSet       selector
                      |               |
                   Pods <-------------+
                      |
           localhost port-forward
                      |
               fixture client
```

Terraform compares declared resources with its state and refreshed provider observations.
A plan is a proposed change; it does not reserve capacity or guarantee a successful apply.
The state file associates Terraform addresses with remote object identities.
State is not the Kubernetes API and can be stale or contain secrets.

The Kubernetes Deployment controller seeks the declared number of matching Pods.
The scheduler finds nodes for Pods subject to requests, constraints, and capacity.
A Pod may exist without being ready to receive service traffic.
The Service selects ready endpoints by labels; the Service does not create replicas.
The API is the source of Kubernetes desired state, not proof that traffic is reaching a healthy Pod.

Terraform and Kubernetes operate at different timescales.
Terraform acts when invoked; Kubernetes controllers continuously reconcile within their scope.
Changing replicas with `kubectl scale` may work temporarily, but the next Terraform apply can reverse it.
Choose one authoritative owner per field and document exceptions.

This exercise creates the cluster outside Terraform to avoid a provider-bootstrap dependency.
Terraform owns only namespaced workload objects after the cluster exists.
`kind delete cluster` removes the separate local cluster; `terraform destroy` alone does not.
The isolation boundary is practical, not a security guarantee against hostile code.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Kubernetes | Container orchestration system and API | Assuming it makes applications correct |
| kind | Kubernetes in Docker, a local test cluster | Treating it as a production cluster |
| Cluster | Control plane and worker nodes | Equating a local cluster with multiple failure domains |
| Context | Named cluster, user, and namespace selection in kubeconfig | Assuming current context always targets a sandbox |
| Namespace | Logical scope for namespaced Kubernetes resources | Treating it as a complete security boundary |
| Node | Machine or container hosting Kubernetes components | Assuming each Pod is on a separate physical machine |
| Pod | Kubernetes scheduling unit containing one or more containers | Treating a Pod as durable storage |
| Deployment | Controller declaration for a replicated stateless workload | Assuming ready replicas are guaranteed |
| ReplicaSet | Controller maintaining a number of identical Pods | Editing its replicas instead of the Deployment |
| Replica | One requested Pod copy, measured in Pods | Equating replicas with independent failure zones |
| Service | Stable Kubernetes network abstraction | Assuming it is the application process |
| ClusterIP | Service address reachable inside the cluster | Assuming it is a public address |
| Selector | Label criteria for associating objects | Assuming names match automatically |
| EndpointSlice | API representation of Service endpoints | Ignoring readiness when diagnosing no endpoints |
| ConfigMap | Non-secret configuration object | Putting credentials or large datasets into it |
| Readiness probe | Signal used to determine whether a Pod should receive Service traffic | Treating it as liveness or end-to-end validation |
| Liveness probe | Signal used to restart an unhealthy container | Using it for a transient dependency failure |
| Request | Scheduler resource reservation, in CPU or bytes | Calling it a measured utilization |
| Limit | Container resource ceiling, in CPU or bytes | Assuming an overly low limit is harmless |
| CPU millicore | One thousandth of one logical CPU | Reading 100m as 100 CPU cores |
| Image | Packaged container filesystem and metadata | Assuming a tag is immutable |
| Digest | Content-derived image identifier | Treating a floating tag as the same guarantee |
| Terraform | Declarative provisioning tool and state engine | Assuming all providers behave identically |
| Provider | Plugin implementing resource API operations | Confusing provider version with Terraform version |
| State | Terraform's resource identity and attribute record | Publishing it because configuration has no passwords |
| Plan | Proposed Terraform action set | Calling it a guarantee of successful apply |
| Apply | Execution of a Terraform plan or configuration | Assuming partial failure automatically rolls back |
| Destroy | Requested deletion of Terraform-managed resources | Assuming it removes separately created clusters |
| Drift | Difference between desired and actual configuration | Assuming every difference is visible to every provider |
| Rollout | Controlled transition between workload revisions | Treating a created Pod as proof of successful rollout |
| Port-forward | Client-mediated TCP tunnel to a Kubernetes object | Treating it as a production ingress architecture |
| Autoscaling | Controller-driven adjustment based on a signal | Assuming scaling fixes a broken dependency |
| Synthetic fixture | Deliberately invented test input or response | Treating its result as operational evidence |

## 5. The mechanics, step by step

### 5.1 Establish the sandbox boundary

Use a personal, disposable local machine or an approved isolated training environment.
Do not use a shared production Docker daemon, kubeconfig, namespace, or Terraform state.
The instructions require Docker, kind, kubectl, Terraform, Python 3, curl, and Bash.
The Docker daemon must be running; kind creates node containers on that daemon.
Internet access is ordinarily needed for the kind node image, Terraform provider, and Python image.
Install the tools from their official sources and follow your organization's approval process.
Record their versions with `docker version`, `kind version`, `kubectl version --client`,
`terraform version`, and `python3 --version` before beginning.
The example pins the Terraform Kubernetes provider to 2.33.0 and constrains Terraform to 1.5 or later, below 2.0.
A provider pin is not a complete supply-chain verification or a pin of the container image.

The local lab does not provision billable cloud infrastructure by itself.
However, an approved remote workstation, metered data transfer, paid Docker service, or cloud-hosted Docker daemon can cost money.
An existing machine consumes CPU, memory, disk, electricity, and network bandwidth.
Review your own environment and pricing before starting; there is no universal USD amount to promise.
Do not create cloud load balancers or persistent volumes as part of this lab.
Stop if you cannot positively identify your Docker daemon and cluster as disposable and authorized.

### 5.2 Create a working directory and cluster

Run the following from an empty directory reserved for this lab.
It uses a private kubeconfig and refuses to reuse a cluster with the same name.
Do not replace the safety check with deletion of a cluster you do not own.

```bash
mkdir -p sc-k8s-lab
cd sc-k8s-lab
mkdir -p .lab
if kind get clusters | grep -Fxq sc-lab; then
  echo 'Cluster sc-lab already exists; stop and inspect ownership.' >&2
  exit 1
fi
kind create cluster --name sc-lab --kubeconfig "$PWD/.lab/kubeconfig"
export KUBECONFIG="$PWD/.lab/kubeconfig"
kubectl config current-context
kubectl get nodes
```

Expected context is `kind-sc-lab`; wait until the single node is `Ready` before continuing.
If the node cannot start, inspect Docker resources and network policy rather than selecting another context.
The default kind node image depends on the installed kind release; record its identity for reproducibility.
This lab intentionally uses one local node and cannot demonstrate node-level high availability.

### 5.3 Create the fixture

Copy the complete Python program in section 7 into `app.py` in this working directory.
It serves `/healthz` and `/forecast` on port 8080 using only the Python standard library.
Both the local fixture test and the Kubernetes container execute that same file.
The forecast represents a synthetic 137 units for one SKU and one day.
The HTTP response has no dependence on external credentials, databases, or model artifacts.

### 5.4 Declare the managed objects

Create `main.tf` with the exact content below.
The provider is explicitly pointed at the isolated kubeconfig and the expected context.
The namespace is created separately from the cluster and is owned by this Terraform configuration.
The three-Pod scaling target appears later; the initial declaration requests two Pods.

```hcl
terraform {
  required_version = ">= 1.5.0, < 2.0.0"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "= 2.33.0"
    }
  }
}

provider "kubernetes" {
  config_path    = "${path.module}/.lab/kubeconfig"
  config_context = "kind-sc-lab"
}

resource "kubernetes_namespace_v1" "lab" {
  metadata {
    name = "sc-lab"
  }
}

resource "kubernetes_config_map_v1" "app" {
  metadata {
    name      = "forecast-app"
    namespace = kubernetes_namespace_v1.lab.metadata[0].name
  }
  data = {
    "app.py" = file("${path.module}/app.py")
  }
}

resource "kubernetes_deployment_v1" "app" {
  metadata {
    name      = "forecast"
    namespace = kubernetes_namespace_v1.lab.metadata[0].name
  }
  spec {
    replicas = 2
    selector {
      match_labels = { app = "forecast" }
    }
    template {
      metadata {
        labels = { app = "forecast" }
        annotations = {
          "lab.example/app-sha256" = filesha256("${path.module}/app.py")
        }
      }
      spec {
        container {
          name    = "forecast"
          image   = "python:3.12-slim"
          command = ["python", "/app/app.py"]
          port {
            container_port = 8080
          }
          resources {
            requests = { cpu = "50m", memory = "64Mi" }
            limits   = { cpu = "250m", memory = "128Mi" }
          }
          readiness_probe {
            http_get {
              path = "/healthz"
              port = 8080
            }
            initial_delay_seconds = 1
            period_seconds        = 5
          }
          liveness_probe {
            http_get {
              path = "/healthz"
              port = 8080
            }
            initial_delay_seconds = 5
            period_seconds        = 10
          }
          volume_mount {
            name       = "app"
            mount_path = "/app"
            read_only  = true
          }
        }
        volume {
          name = "app"
          config_map {
            name = kubernetes_config_map_v1.app.metadata[0].name
          }
        }
      }
    }
  }
  wait_for_rollout = true
}

resource "kubernetes_service_v1" "app" {
  metadata {
    name      = "forecast"
    namespace = kubernetes_namespace_v1.lab.metadata[0].name
  }
  spec {
    selector = { app = "forecast" }
    port {
      port        = 80
      target_port = 8080
    }
    type = "ClusterIP"
  }
}
```

The file hash annotation changes the Pod template when `app.py` changes.
That creates a new Deployment revision rather than assuming the running Python process reloads a mounted file.
The ConfigMap volume is read-only inside the container, but the ConfigMap is not secret storage.
The tag is convenient for a sandbox; it is mutable and must not be considered a production image pin.
A production image needs an approved, verified immutable digest and organization-specific controls.
The lab's 50m CPU and 64Mi memory requests are illustrative scheduling inputs, not measured capacity.
The limits are ceilings, not promises that a node has spare resources.

### 5.5 Initialize, inspect, and apply

First confirm `kubectl config current-context` still prints `kind-sc-lab`.
The provider itself uses the private file and context even if your normal kubeconfig differs.
Do not copy the provider configuration into a directory with a production kubeconfig.

```bash
terraform init
terraform validate
terraform plan -out=lab.tfplan
```

Review the plan: it should create the namespace, ConfigMap, Deployment, and Service.
A saved plan can contain sensitive information; keep it local and do not commit it.
If the proposed resources or target are not expected, stop instead of applying.
Then apply the reviewed plan and inspect observed state:

```bash
terraform apply lab.tfplan
kubectl -n sc-lab rollout status deployment/forecast --timeout=120s
kubectl -n sc-lab get deployment,pods,service
kubectl -n sc-lab get endpointslice -l kubernetes.io/service-name=forecast
```

Expected: Deployment desired and available counts reach two, two Pods become Ready,
and a ClusterIP Service points to ready endpoints on port 8080.
Actual names, addresses, startup time, and ordering vary.
Image pulls can take longer or fail under restricted registry access.
The rollout command reports an unsuccessful wait if Pods do not become ready in time.
A successful Terraform apply or rollout does not validate the forecast's business quality.

### 5.6 Exercise the endpoint

In one terminal, with the same working directory and `KUBECONFIG` exported:

```bash
kubectl -n sc-lab port-forward --address 127.0.0.1 service/forecast 18080:80
```

Keep that terminal running. In a second terminal run:

```bash
curl --fail --silent --show-error http://127.0.0.1:18080/healthz
curl --fail --silent --show-error http://127.0.0.1:18080/forecast
```

Expected JSON values are `{"status":"ok"}` and a forecast object containing SKU `SKU-17`,
date `2026-01-15`, and `forecast_units` equal to 137.
The server also emits a newline after each JSON response.
Port-forward is bound to loopback; it is a local diagnostic tunnel, not a production ingress.
It forwards through one selected Pod and is not a load-balancing or failover test.
If it exits on a Pod replacement, restart it before repeating the checks.
Stop it with Ctrl-C when finished.

### 5.7 Scale through the declared source

Edit `main.tf` and change only `replicas = 2` to `replicas = 3`.
Run `terraform plan -out=scale.tfplan`, inspect the proposed change, and apply it with
`terraform apply scale.tfplan`.
Then run `kubectl -n sc-lab rollout status deployment/forecast --timeout=120s` and
`kubectl -n sc-lab get deployment,pods`.
Expected: the Deployment reaches three available replicas, subject to local capacity and image availability.
The `forecast_units` response remains 137; scaling replicas does not change model output.
Do not use a direct `kubectl scale` as the durable source of truth for this exercise.

The scheduler considers three requests of 50m CPU and 64Mi memory: 150m CPU and 192Mi memory in total.
Limits total 750m CPU and 384Mi memory, but actual usage can differ.
The node must also support its own system workloads and scheduling overhead.
Those totals do not guarantee that the Pods will schedule or serve three times the traffic.
To scale back, change the same declaration to two, plan, review, and apply.

### 5.8 Diagnose rather than conceal

If Terraform cannot connect, inspect the path and context in the private kubeconfig.
If a Pod remains Pending, inspect `kubectl -n sc-lab describe pod POD_NAME` and node capacity.
If it is in `ImagePullBackOff`, inspect registry reachability and image policy.
If it is running but not Ready, inspect the probe, the container port, and logs with
`kubectl -n sc-lab logs deployment/forecast --all-containers=true`.
If the Service has no endpoints, compare its selector with the Pod labels and readiness.
If the local port is in use, choose another local port and update the curl URL consistently.
Never respond to a failure by disabling security controls on an unrelated cluster.

### 5.9 Destroy and independently clean up

Stop all port-forward processes first.
From the same directory and with the private kubeconfig still present, review and execute:

```bash
terraform plan -destroy -out=destroy.tfplan
terraform apply destroy.tfplan
kubectl get namespace sc-lab --ignore-not-found
kind delete cluster --name sc-lab
kind get clusters
```

Confirm that the namespace no longer appears and that `sc-lab` no longer appears in the cluster list.
The first operation removes Terraform-managed objects; the second removes the separately created kind cluster.
If destroy fails, retain the kubeconfig and state, diagnose the error, and retry before deleting the cluster.
If the cluster has already been deleted, Terraform cannot necessarily destroy through its provider;
verify the cluster is gone and handle the now-orphaned local state as sensitive local data.
After successful deletion, remove the lab directory, `.terraform`, plans, state, backups, and kubeconfig
according to local policy; these files may include sensitive attributes in non-synthetic variants.
Container images may remain in Docker's cache; remove only images you own and no longer need.
Do not run broad Docker prune commands on a shared daemon.
A cloud or remotely hosted variation requires separate inspection of its provider console and billing;
local cluster deletion is not proof that cloud resources have been deleted.

## 6. Worked example

The supply-chain team wants to let two planners inspect a synthetic forecast at the same time.
The requested decision is deployment configuration, not the number of units to order.
The fixed fixture returns 137 units for SKU-17 on 2026-01-15.
That value is illustrative and is not a measured forecast or a replenishment commitment.

Initial declaration: two Pods, each requesting 50 millicores of CPU and 64 MiB of memory.
Initial aggregate requests are 2 × 50m = 100m CPU and 2 × 64 MiB = 128 MiB memory.
Each Pod has a CPU limit of 250m and a memory limit of 128 MiB.
Aggregate configured ceilings are 2 × 250m = 500m CPU and 2 × 128 MiB = 256 MiB memory.
This is arithmetic over settings, not measured usage or a capacity guarantee.

The team changes the declared replica count from two to three and reviews the Terraform plan.
Aggregate requests become 3 × 50m = 150m CPU and 3 × 64 MiB = 192 MiB memory.
The difference from the initial declaration is 50m CPU and 64 MiB requested.
Aggregate ceilings become 750m CPU and 384 MiB memory.
Assuming all three Pods schedule and become Ready, the Deployment reports three available Pods.
The same fixture still returns 137 units; replica count does not add forecasts together.

A single local node hosts all replicas in this sandbox.
Losing that node can remove all three Pods at once.
Three replicas on one node therefore do not establish availability across independent failures.
The example does not provide throughput data: no load generation or latency measurement has been performed.
A valid capacity decision requires representative requests, saturation tests, headroom, and dependency analysis.
An operator should define a traffic distribution and service-level objective before interpreting scaling tests.

The ClusterIP Service selects Pods with `app=forecast` and maps Service port 80 to container port 8080.
Readiness checks the synthetic process at `/healthz`.
It does not establish that a future real model has current input data or authorized users.
A port-forward demonstrates one local HTTP path, not external routing or balanced traffic.
If one Pod is unready, an EndpointSlice should exclude it from ready Service endpoints.
Inspect actual EndpointSlice conditions rather than inferring endpoint health from Pod count alone.

The plan is reviewed before applying because apply can create or change real objects.
After the scenario, Terraform deletes the namespaced objects and kind deletes the independent cluster.
The fixture, provider installation, state, saved plans, and cached images are separate local artifacts.
Clean each according to its owner and retention policy rather than assuming one destroy command removes all of them.

## 7. Code, query, or calculation

Save the following complete program as `app.py` before running Terraform.
It binds port 8080 in the container and returns synthetic JSON.
Only Python 3 standard-library modules are required.
Do not run the local and container test on the same host port without isolating them.

```python
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FORECAST = {
    "sku": "SKU-17",
    "date": "2026-01-15",
    "forecast_units": 137,
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            status, payload = 200, {"status": "ok"}
        elif self.path == "/forecast":
            status, payload = 200, FORECAST
        else:
            status, payload = 404, {"error": "not found"}
        body = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
```

For a local-only fixture check, run `python3 app.py` in a separate terminal.
It listens on all local interfaces, so use only an isolated trusted machine and stop it immediately after testing.
In another terminal run the following assertions; terminate the server with Ctrl-C afterward.

```bash
python3 - <<'PY'
import json
from urllib.error import HTTPError
from urllib.request import urlopen

base = "http://127.0.0.1:8080"
with urlopen(base + "/healthz") as response:
    assert response.status == 200
    assert json.load(response) == {"status": "ok"}
with urlopen(base + "/forecast") as response:
    assert response.status == 200
    assert json.load(response) == {
        "sku": "SKU-17", "date": "2026-01-15", "forecast_units": 137
    }
try:
    urlopen(base + "/missing")
except HTTPError as exc:
    assert exc.code == 404
else:
    raise AssertionError("missing route must return 404")
print("fixture checks passed")
PY
```

Expected output is `fixture checks passed` if all three assertions succeed.
These tests check response shape and status; they do not test load, authentication, or forecast quality.
To test the deployed service instead, retain the port-forward and change `base` to
`http://127.0.0.1:18080` in a temporary copy of the assertion script.
The use of a mutable Python image and a ConfigMap-mounted script is intentionally educational.
For production, build and attest an image, pin its digest, and apply your platform's hardening standards.

## 8. Data, standards, and regulatory context

The Kubernetes API schema and provider schema are technical interfaces, not regulatory approval.
The manifests define a workload; they do not establish data rights or lawful processing.
This lab's synthetic SKU, date, and quantity contain no actual customer or supplier records.
A real forecast service may contain sensitive commercial information even if it contains no personal data.
Classify the model, inputs, predictions, logs, and state before deployment.

Terraform state can include values that providers mark sensitive, depending on resource and provider behavior.
Marking an output sensitive changes presentation and is not equivalent to encrypting the state file.
Saved plans can also contain sensitive values; restrict access and retention.
Remote state in a real system needs encryption, access control, locking, recovery, and audit policy.
Those controls must be selected for the actual backend; this local lab uses no remote backend.
The dependency lock file records selected provider packages and checksums; review its provenance before committing it.

The Kubernetes documentation identifies ConfigMaps as non-confidential configuration.
A Secret is not automatically a complete secret-management system; use platform encryption and authorization controls.
Avoid copying credentials into images, environment files, command history, or diagnostic bundles.
Use least-privilege identities, restricted network paths, and auditable access in a real deployment.
A namespace scopes objects but does not by itself guarantee tenant isolation.
Role-based access control and network policy require correct cluster support and configuration.

The NIST Cybersecurity Framework 2.0 is voluntary guidance, not a universal legal requirement.
It helps structure governance, protection, detection, response, and recovery decisions.
Whether GDPR, contractual confidentiality, sector-specific obligations, or other rules apply depends on actual data and jurisdiction.
This educational example makes no compliance determination and is not legal advice.
Record the governing policy, data classification, retention period, and responsible owner before a real release.

## 9. Industry practice and case evidence

Kubernetes documentation describes Deployments, Services, readiness probes, and resource management as general mechanisms.
HashiCorp documents Terraform plans, state, and the Kubernetes provider as provisioning mechanisms.
These sources support the technical behaviors discussed here, not a claim of any company's achieved savings.

Spotify has publicly described its migration to Kubernetes in a CNCF case study.
That named case is evidence that an organization used the platform, not proof that this lab reproduces its results.
Its environment, organizational constraints, scale, and operational controls differ from a one-node kind sandbox.
Do not infer the same availability or cost from adopting a similar technology name.

In a supply-chain deployment, model owners, platform engineers, security, and business owners have different responsibilities.
A model owner validates the prediction and monitors data assumptions.
A platform engineer maintains the platform and deployment controls.
A business owner decides whether the forecast is suitable for a particular planning action.
Security and privacy owners assess data access, identity, and retention under applicable policy.
Those roles may be combined in a small organization, but accountability still needs an explicit owner.

A realistic release process reviews an immutable artifact, environment changes, expected traffic, and rollback criteria.
A real rollout records the release identity and monitors both technical health and business-data quality.
A deployment with healthy probes can still return stale or nonsensical forecasts.
Conversely, a correct model cannot serve users reliably if its platform is unavailable.
Neither condition can be inferred from the synthetic test in this lab.

### Operational review cards

#### 1. Cluster identity

- Check: Read the Docker context and the private kubeconfig before starting.
- Reason: A correct Kubernetes context does not prove that the Docker daemon is local.
- Decision: Record both ownership and purpose in the run notes.

#### 2. Namespace ownership

- Check: Create a namespace reserved for this exercise.
- Reason: A shared namespace can contain resources outside Terraform state.
- Decision: Do not delete a namespace unless its ownership is established.

#### 3. Provider bootstrap

- Check: Create the kind cluster before initializing the Kubernetes provider.
- Reason: The provider needs an API endpoint and credentials to manage workload resources.
- Decision: Keep the cluster lifecycle separate from namespaced resource lifecycle.

#### 4. Provider version

- Check: Use the declared exact provider version and review the generated lock file.
- Reason: Provider schema or behavior may vary across versions.
- Decision: The lock file supplements, rather than replaces, source and checksum review.

#### 5. Node image

- Check: Record the kind version and node image used in the run.
- Reason: A different Kubernetes version can change available API behavior.
- Decision: A local kind result is evidence only for the environment actually used.

#### 6. Application image

- Check: Record the resolved image digest when reproducibility matters.
- Reason: A tag may later point to different bytes.
- Decision: Use an approved immutable digest for a controlled release.

#### 7. Configuration change

- Check: Change the application file before generating the plan.
- Reason: The file hash in the Pod template is evaluated during planning.
- Decision: Review both the ConfigMap and rollout changes together.

#### 8. Service selector

- Check: Compare the Service selector to the Pod template labels.
- Reason: An otherwise healthy Pod is not selected when the labels differ.
- Decision: Inspect EndpointSlices and readiness after the rollout.

#### 9. Readiness design

- Check: Keep the lab readiness endpoint deterministic and inexpensive.
- Reason: An endpoint that always succeeds says nothing about future data dependencies.
- Decision: Design real readiness checks around actual serving requirements.

#### 10. Liveness design

- Check: Use liveness only for failures that a restart can plausibly repair.
- Reason: An external dependency outage can otherwise cause restart storms.
- Decision: Evaluate startup and failure behavior before changing probe thresholds.

#### 11. Resource requests

- Check: Use requests as scheduler inputs, not as observed CPU use.
- Reason: The scheduler also accounts for other Pods and node allocatable capacity.
- Decision: Inspect scheduling events if the third replica remains Pending.

#### 12. Resource limits

- Check: Understand that a memory limit may lead to a container termination.
- Reason: A CPU limit may throttle the process under contention.
- Decision: Choose real limits from measurements and platform policy.

#### 13. Scaling authority

- Check: Make the Terraform declaration the owner of replica count in this lab.
- Reason: A horizontal autoscaler or manual scale can conflict with Terraform.
- Decision: Assign one owner and explicitly design any production handoff.

#### 14. Rollout observation

- Check: Wait for the Deployment and inspect the Pods separately.
- Reason: An API object can exist before its containers are ready.
- Decision: Capture observed counts and any failure messages, not only the plan.

#### 15. Local routing

- Check: Use loopback-only port-forward for diagnostic access.
- Reason: The tunnel chooses a Pod and can stop when that Pod is replaced.
- Decision: Restart the tunnel after a rollout if it exits.

#### 16. API contract

- Check: Check status codes and JSON fields as well as connectivity.
- Reason: A listening port can return the wrong response.
- Decision: The synthetic contract does not validate a production model.

#### 17. Traffic distribution

- Check: Do not infer balanced traffic from a single port-forward request.
- Reason: One forwarded connection does not exercise Service routing under normal load.
- Decision: Use an approved workload generator and realistic traffic model for performance work.

#### 18. Data freshness

- Check: Separate service health from forecast freshness.
- Reason: The sample has a fixed date and no upstream data source.
- Decision: A real service needs explicit freshness thresholds and ownership.

#### 19. State confidentiality

- Check: Protect Terraform state and saved plans as potentially sensitive files.
- Reason: A provider may store resource attributes even when output is redacted.
- Decision: Apply access control, retention, and approved storage practices.

#### 20. Plan review

- Check: Inspect both the target and object changes before applying.
- Reason: A syntactically valid plan can still affect the wrong environment.
- Decision: Stop when the plan differs from the intended small change.

#### 21. Partial apply

- Check: Treat a failed apply as a potentially partial change.
- Reason: Some API operations can complete before a later operation fails.
- Decision: Inspect state and cluster objects before planning a correction.

#### 22. Drift handling

- Check: Investigate unexpected changes before applying a reconciliation plan.
- Reason: Direct API edits may be intentional incident work or unauthorized drift.
- Decision: Decide the authoritative configuration with the responsible owner.

#### 23. Failure domains

- Check: Draw node and infrastructure boundaries before counting replicas.
- Reason: Multiple Pods on one node fail together when that node fails.
- Decision: Production availability requires independent failure domains and recovery testing.

#### 24. Rollback

- Check: Retain the previous reviewed source and approved image identity.
- Reason: Rolling back Kubernetes alone can conflict with Terraform on the next apply.
- Decision: Reconcile the source of truth and verify the restored behavior.

#### 25. Deletion sequence

- Check: Destroy managed objects while the API is still available.
- Reason: Deleting the cluster first prevents normal provider deletion.
- Decision: Confirm deletion before removing kubeconfig and local state.

#### 26. Local costs

- Check: Account for compute, storage, bandwidth, and electricity.
- Reason: A local container can still consume metered remote resources.
- Decision: Check where the daemon runs and who pays for it.

#### 27. Cloud boundaries

- Check: Keep the local Service as ClusterIP for this exercise.
- Reason: A cloud LoadBalancer or volume can create separately billed infrastructure.
- Decision: Review actual cloud resources and invoices independently after cleanup.

#### 28. Artifact hygiene

- Check: Treat plans, state backups, kubeconfig, and downloaded images separately.
- Reason: Deleting a cluster does not remove every local artifact.
- Decision: Remove only owned artifacts according to local policy.

#### 29. Security boundary

- Check: Use no production credentials or confidential data in the fixture.
- Reason: A local container shares the host kernel and is not an isolation guarantee.
- Decision: Use an approved environment for untrusted images and sensitive workloads.

#### 30. Evidence discipline

- Check: Record command output only after actually executing the command.
- Reason: Expected output is a test oracle, not an observed result.
- Decision: Label assumptions and failures instead of reporting invented success.

### Interpreting observed deployment evidence

Use these diagnostic comparisons after a run.
They describe possible observations, not results obtained by the authors.
Record the actual command, target, time, and output before drawing a conclusion.

#### Desired versus available replicas

The Deployment's desired count is a request to the controller.
The available count depends on the controller's availability criteria and Pod health.
If desired is three and available is two, the scale operation is not fully realized.
Inspect the third Pod and its events before deciding that a new node is needed.
An image pull failure, probe failure, scheduling constraint, or termination can produce this gap.

#### Created versus scheduled Pods

A Pod can exist in the API without being bound to a node.
The scheduler evaluates requests, node conditions, affinity, and other constraints.
`kubectl describe pod POD_NAME` includes events that can explain a scheduling failure.
Do not use container logs as the first diagnostic when the container has never started.
The exact event messages vary with Kubernetes version and cluster conditions.

#### Scheduled versus started containers

A scheduled Pod still needs its image and container runtime.
Registry authorization, DNS, an unavailable image tag, or network rules can prevent startup.
Inspect the container waiting reason and events before changing the application code.
An `ImagePullBackOff` is a retry state, not proof that the image does not exist.
Follow the actual registry and organization policy when resolving the cause.

#### Running versus ready

Running describes a Pod lifecycle phase, while readiness is a condition.
The process may run while its readiness endpoint fails.
Inspect probe configuration, application logs, and the path and port in the Pod specification.
In this fixture, `/healthz` must return HTTP 200 from port 8080.
An HTTP 200 from another path does not prove the configured readiness probe works.

#### Ready versus selected

A Ready Pod is not necessarily an endpoint of every Service.
The Service selector must match the Pod's labels in the same namespace.
Inspect the Service and the Pod template if the EndpointSlice has no matching ready endpoints.
An empty EndpointSlice can also reflect that matching Pods have not become Ready.
Do not change selectors blindly without understanding which workload they would include.

#### Selected versus reachable

Endpoint selection is necessary but does not prove every network path works.
An incorrectly configured target port can route traffic to a port where no process listens.
Network policy, a stopped port-forward, or host firewall rules can also affect reachability.
Compare the Service port, target port, container port, and client URL systematically.
The lab maps Service port 80 to container port 8080 and local port 18080 to Service port 80.

#### Endpoint reachability versus response correctness

A successful TCP connection does not establish that JSON has the intended meaning.
Check the response code, content type, field names, and values.
The fixed forecast should identify SKU-17, the specified date, and 137 units.
For a real model, add schema, freshness, feature, and model-version checks.
Avoid inferring statistical quality from a syntactically valid JSON document.

#### One response versus a population

One successful curl request demonstrates only that one request succeeded at one time.
It does not estimate an error rate, a latency distribution, or capacity.
The sample tunnel is particularly unsuitable for measuring Service load distribution.
Choose a realistic request mix and observation period before any performance test.
Report sample size and failure criteria with measured results.

#### Replica count versus workload capacity

Increasing replicas adds potential serving processes, subject to placement and readiness.
It does not add database capacity or fix a saturated downstream dependency.
Python process behavior, node resources, network limits, and request shape all affect throughput.
Use controlled measurements to determine whether a bottleneck actually moves.
Never claim a linear improvement merely from a change in desired replica count.

#### Replication versus fault isolation

Every Pod in this lab may share the same kind node and underlying Docker daemon.
The host, daemon, local disk, and network are therefore common failure points.
Three Ready replicas cannot survive loss of their only shared node.
A production design needs explicit topology and failure-domain assumptions.
Test recovery against the failure scenarios that the business actually requires.

#### Restart versus replacement

Kubernetes may restart a container inside a Pod or replace the Pod itself.
These events have different identities and operational implications.
Inspect restart count, Pod age, and events when investigating repeated failures.
Application state stored only in a container filesystem may be lost on replacement.
The fixture is stateless, so it does not demonstrate durable data recovery.

#### Liveness versus readiness intervention

Liveness failure can lead to a container restart.
Readiness failure removes the Pod from normal ready Service endpoints without necessarily restarting it.
Treat repeated restarts and missing endpoints as different symptoms.
An unavailable external database does not automatically mean restarting the service will help.
Keep dependency-sensitive checks and restart criteria deliberately separated.

#### Readiness versus business fitness

The fixture's readiness probe checks only that its process can answer `/healthz`.
It does not evaluate whether the forecast date or demand pattern is useful.
An operational system may need separate data-quality and model-monitoring signals.
Specify what happens when a prediction is stale but the service is technically reachable.
The business owner, not the probe alone, decides whether output is fit for a planning action.

#### Source versus live configuration

Terraform source describes the intended state for resources under its management.
The Kubernetes API contains the current accepted object specification.
The provider reads observed attributes to help construct a plan.
Compare a reviewed plan with API observations when diagnosing differences.
Do not assume that saving a file has already changed a running Deployment.

#### Live configuration versus state

Terraform state stores resource identities and attributes used by Terraform.
It is not a continuous mirror of every Kubernetes field.
An out-of-band change can occur after a plan is generated.
Saved plans represent the proposal at planning time, not an indefinite safety guarantee.
Minimize delay between review and apply, and replan when circumstances change.

#### State versus backup

A state file is essential to normal Terraform management, but its existence is not a tested backup.
Local disk failure, accidental deletion, and unauthorized disclosure are separate risks.
Real teams should use an approved backend with access control and recovery procedures.
Document who can restore state and how concurrent changes are controlled.
This lab intentionally avoids configuring a remote backend or real credentials.

#### Redacted output versus encrypted storage

Terraform can suppress sensitive values in some displayed output.
That presentation control does not imply that the underlying state or plan is encrypted.
The state, its backups, and saved plan files need their own security controls.
Do not paste their contents into public issues or chat channels.
Use synthetic configuration here to avoid creating a secret-management exercise accidentally.

#### ConfigMap update versus application reload

Kubernetes can update projected ConfigMap data without a process automatically rereading it.
This Python program loads its constants when the process starts.
The template hash in the Terraform declaration changes when `app.py` changes.
That change creates a new Pod template revision and initiates a rollout.
Confirm the actual new revision and response rather than assuming a file update was consumed.

#### Tag versus image identity

The example uses `python:3.12-slim` for convenient local demonstration.
The same tag can resolve to different image content at different times.
Record the resolved digest when comparing runs, and account for platform architecture.
Production pipelines should verify an approved digest and provenance.
A digest identifies content but does not alone certify that the content is safe.

#### Declared requests versus measured usage

CPU and memory requests influence scheduling.
They do not say how much CPU or memory the container actually consumes.
Limits are separate controls and their runtime consequences differ by resource type.
Observe representative workload behavior before choosing production values.
The sample request arithmetic is an exercise, not a sizing recommendation.

#### Local disk versus persistent storage

The lab requires no Kubernetes PersistentVolume.
The application returns fixed data and writes no business state.
An actual forecasting pipeline might need durable artifacts, datasets, or external object storage.
Those resources require their own lifecycle, access control, backup, and cost ownership.
Deleting a namespace should never be assumed to delete every external storage resource.

#### Diagnostic tunnel versus public ingress

The port-forward command explicitly binds to `127.0.0.1`.
It is intended for local inspection, not for sharing a service with planners.
A production entry point needs an approved ingress design, authentication, and transport security.
It also needs traffic policy, observability, and an operational owner.
None of those capabilities is demonstrated by the local tunnel.

#### Declarative scaling versus autoscaling

The exercise changes a fixed replica count in the Terraform configuration.
That is declarative manual scaling, not an autoscaling experiment.
An autoscaler requires a valid metric, a controller, bounds, and an ownership strategy.
It may conflict with Terraform if both continually manage the replica field.
Test scale-up, scale-down, saturation, and failure handling before production use.

#### Scale-up versus scale-down

Adding a replica requires enough schedulable resources and a working image pull.
Removing one requires the service to tolerate termination and in-flight requests.
This small standard-library server is not a complete graceful-shutdown design.
Real services should define termination grace, draining, and client retry behavior.
Validate that scaling down does not interrupt work or cause duplicate side effects.

#### Plan success versus apply success

Planning can validate a proposed graph without guaranteeing API operations will succeed.
Permissions, admission policy, network access, and resource availability can change.
Apply may succeed for some objects before another operation fails.
Review the actual outcome and any errors rather than assuming transactional rollback.
Keep the state and kubeconfig until the partial result has been understood.

#### Apply success versus independent reproducibility

A successful apply says the declared provider operations completed under that run's conditions.
It does not guarantee the same image bytes, node version, or network conditions later.
Record the tool versions, provider lock, image digest, and node image for meaningful comparison.
Pinning and recording are complementary: one constrains selection, the other documents it.
Compare observations in a clean, authorized environment before claiming repeatability.

#### Destroy versus decommission

Terraform destroy requests removal of resources tracked by the configuration and state.
The separate kind cluster, local files, and Docker image cache remain separate concerns.
External systems may retain logs, artifacts, or billable resources in real deployments.
Define an owner and verification step for each category of resource.
A local destroy is not evidence that a remote environment is cost-free.

#### Exercise evidence versus production evidence

The fixture can establish whether a narrowly defined HTTP response is reachable.
It cannot establish demand accuracy, planner adoption, or avoided stockouts.
It also cannot establish a recovery time objective or sustained availability.
Business evidence requires real authorized data, representative tests, and a decision baseline.
Label this sandbox result as a teaching demonstration rather than a production outcome.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong context | Unexpected cluster name | CLI or provider selects another kubeconfig | Unintended changes | Explicit isolated path and context; stop on mismatch |
| Shared cluster name | Existing `sc-lab` | Lab would reuse another owner's cluster | Disruption or data exposure | Refuse reuse; inspect ownership |
| Mutable image | Same tag pulls new bytes | Tag is not a content identifier | Different behavior on restart | Pin verified digest for production |
| No endpoints | Service exists, endpoint list empty | Labels mismatch or Pods unready | Requests fail | Inspect selector, labels, and readiness |
| Pending Pods | Desired replicas exceed ready replicas | Insufficient schedulable resources | Reduced capacity | Inspect scheduler events and node allocatable |
| Image pull failure | `ImagePullBackOff` | Registry, network, authorization, or tag problem | Rollout stalls | Inspect events and approved registry access |
| Probe failure | Restarts or unready Pods | Incorrect path, port, timing, or process | No serving endpoints | Inspect logs and probe behavior |
| Secret in state | Sensitive value appears in plan or state | Provider records attributes | Credential disclosure | Avoid secrets in inputs; protect state and plans |
| Manual drift | Next plan proposes reversal | Direct API edit conflicts with declaration | Surprising rollback | One field owner; review plan |
| App edit not reloaded | ConfigMap changed but old behavior persists | Process does not reread mounted file | Stale response | Template hash triggers rollout |
| One-node failure | Every replica resides on one node | Shared failure domain | Complete local outage | Real multi-zone design and tested recovery |
| Overstated scale | More replicas but latency unchanged | Shared bottleneck or poor concurrency | Wasted resources | Measure workload and dependencies |
| Premature cleanup | Cluster removed before destroy | Provider loses API access | Orphaned state and uncertain deletion | Destroy while cluster exists; retain evidence |
| Open diagnostic port | Port-forward bound beyond loopback | Unintended remote reachability | Exposure | Explicit loopback binding and stop tunnel |
| Namespace assumed secure | Unrestricted cross-namespace access | Namespace is logical scope | Unauthorized access | Apply validated identity and network controls |
| Unreviewed destroy | Wrong plan or target | Destructive changes not inspected | Loss of unrelated resources | Review destroy plan and target ownership |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Disposable authorized Docker daemon | kind creates containers | Shared or production host | Inspect Docker context and ownership |
| Private kubeconfig | Prevents accidental target selection | Reusing another cluster's credentials | Check path and current context |
| Images and providers reachable | Startup needs downloads | Offline or blocked registry | Inspect pull and initialization errors |
| Node has headroom | Scheduler must place replicas | CPU or memory exhaustion | Inspect node allocatable and events |
| Local port available | Port-forward needs a listener | Another process owns 18080 | Observe bind error and use an unused port |
| Synthetic fixture sufficient | Demonstrates API wiring | Real workload needs data and models | Compare interface and operational requirements |
| ConfigMap is non-secret | API-visible configuration | Added credentials or proprietary payload | Review contents and state access |
| Terraform owns replicas | Declarative changes remain authoritative | Another controller edits same field | Inspect drift and ownership design |
| One-node scope accepted | Keeps lab self-contained | Availability across failure domains is required | Map Pod placement to failure domains |
| Health endpoint adequate for lab | Enables basic readiness check | Dependencies or stale forecasts matter | Design dependency and freshness checks |
| Cluster survives through destroy | Provider must delete objects | Cluster removed earlier | Confirm API access before destroy |
| Tag mutability accepted for lab | Keeps image example simple | Repeat runs require byte-identical image | Record digest and compare pulls |

## 12. Exercises

Each exercise is based on the sandbox above; do not use real business data.

### Exercise 1 — Ownership boundaries

Deliverable: a diagram of Terraform, the Kubernetes API, controllers, and Pods.
Acceptance: identify the state file, the action trigger, and the reconciliation loop.
Time: 10 minutes.

### Exercise 2 — Target verification

Deliverable: a short pre-apply checklist for the kubeconfig, context, and Docker daemon.
Acceptance: include a stop condition for an existing cluster of unknown ownership.
Time: 10 minutes.

### Exercise 3 — Resource math

Deliverable: calculate requests and limits for four replicas with this lab's per-Pod settings.
Acceptance: report CPU in millicores and memory in MiB; distinguish requests from usage.
Time: 10 minutes.

### Exercise 4 — Fixture assertions

Deliverable: execute the section 7 assertions locally or through port-forward.
Acceptance: the health, forecast, and missing route checks succeed and the server is stopped afterward.
Time: 15 minutes.

### Exercise 5 — Scale and inspect

Deliverable: a reviewed Terraform plan for three replicas and the observed Deployment counts.
Acceptance: declare the replica change in source, report any unscheduled Pod, and explain the unchanged fixture value.
Time: 20 minutes.

### Exercise 6 — Security review

Deliverable: identify four risks before replacing the fixture with a real forecast API.
Acceptance: cover identity, state, image provenance, and network exposure with a mitigation for each.
Time: 15 minutes.

### Exercise 7 — Cleanup inventory

Deliverable: list the order of destroying Terraform resources, deleting the kind cluster, and removing local artifacts.
Acceptance: include what to do if destroy fails and identify cached images as a separate concern.
Time: 10 minutes.

### Exercise 8 — Production design critique

Deliverable: one page of rollout and recovery criteria for an approved real environment.
Acceptance: cover failure domains, monitoring, traffic testing, rollback, model quality, costs, and ownership.
Time: 25 minutes.

## 13. Self-check answers

### 1. Ownership boundaries

Terraform uses the configuration and state when a plan or apply is invoked.
The Kubernetes API stores the declared workload, and controllers continuously reconcile it.
The Deployment controller manages ReplicaSets and Pods; the Service selects endpoints.
A state file is a Terraform mapping, not a substitute for observing Kubernetes.

### 2. Target verification

Use the private `.lab/kubeconfig`, check `kind-sc-lab`, and confirm the Docker daemon belongs to the approved sandbox.
Stop when `kind get clusters` already lists `sc-lab` and ownership is uncertain.
Do not delete or reuse it without authorization.

### 3. Resource math

Four replicas request 4 × 50m = 200m CPU and 4 × 64 MiB = 256 MiB memory.
Their configured ceilings total 4 × 250m = 1,000m CPU and 4 × 128 MiB = 512 MiB memory.
Neither total is a measurement of runtime consumption.

### 4. Fixture assertions

The health and forecast routes return status 200 with the specified JSON objects.
The missing route returns status 404.
The program prints `fixture checks passed` only if all assertions pass.
Actual execution evidence must come from the learner's own run.

### 5. Scale and inspect

The planned change should alter the desired replica count from two to three.
Actual availability depends on scheduling, pull, and readiness; inspect rather than assume.
Each identical service instance returns 137 units, not 411 units.

### 6. Security review

A strong response requires authenticated, authorized callers and least-privilege workload identities.
Protect and restrict state and plans, including backups.
Verify an immutable image digest and provenance rather than trusting a floating tag.
Restrict ingress and egress according to approved policy, and avoid exposing diagnostic ports.

### 7. Cleanup inventory

Stop the tunnel, review and apply the destroy plan, confirm namespaced objects are gone,
then delete the kind cluster and confirm it no longer appears.
If destroy fails, preserve the kubeconfig and state while diagnosing and retrying.
Remove local plans, state, kubeconfig, and cached images only under the relevant ownership policy.

### 8. Production design rubric

A strong proposal names an accountable owner and measurable technical and business criteria.
It separates node, zone, and dependency failure domains and tests real recovery.
It includes realistic load and cost estimates rather than extrapolating from Pod count.
It specifies rollback authority, artifact identity, data freshness, and forecast-quality monitoring.
It identifies access controls, retention, and an independent cleanup or decommissioning process.

## 14. Interview questions

| Question | Strong answer contains | Weak answer reveals |
|---|---|---|
| How do Terraform and Kubernetes differ? | Invocation-driven provisioning versus controller reconciliation | Treats state as the running cluster |
| Why inspect context? | Explicit target selection and independent provider configuration | Assumes current shell context controls everything |
| What does readiness mean? | Whether a Pod is eligible for Service traffic under its probe | Equates it with model accuracy |
| Why can replicas remain Pending? | Scheduling constraints, resource requests, and events | Assumes desired equals available |
| How is Service routing selected? | Labels and ready endpoints, not matching names | Believes Service creates Pods |
| What does a request reserve? | Scheduler input in CPU or memory units | Calls it measured utilization |
| Why protect Terraform state? | Resource attributes and sensitive values can be recorded | Assumes sensitive marking encrypts it |
| How would you scale? | Single owner, measured bottlenecks, capacity, and plan | Multiplies traffic by replica count without evidence |
| Is one-node redundancy high availability? | Shared failure domain and recovery requirements | Counts Pods as independent nodes |
| What does destroy remove? | Managed objects, subject to provider/API access | Assumes all clusters, state, and caches vanish |

## 15. Key terms

- **Apply:** execute changes; see [mechanics](#5-the-mechanics-step-by-step).
- **Cluster:** control plane and nodes; see [mental model](#3-the-core-mental-model).
- **ClusterIP:** internal Service address; see [mechanics](#5-the-mechanics-step-by-step).
- **ConfigMap:** non-confidential configuration; see [data context](#8-data-standards-and-regulatory-context).
- **Context:** named Kubernetes connection selection; see [mechanics](#5-the-mechanics-step-by-step).
- **Controller:** reconciles API state; see [mental model](#3-the-core-mental-model).
- **Deployment:** manages replicated workload; see [mechanics](#5-the-mechanics-step-by-step).
- **Destroy:** delete managed objects; see [cleanup](#59-destroy-and-independently-clean-up).
- **Digest:** content-addressed image identity; see [failure modes](#10-failure-modes-and-red-flags).
- **Drift:** divergence from declared state; see [failure modes](#10-failure-modes-and-red-flags).
- **EndpointSlice:** Service endpoint representation; see [worked example](#6-worked-example).
- **Fixture:** synthetic fixed response; see [code](#7-code-query-or-calculation).
- **Image:** packaged runtime artifact; see [mechanics](#5-the-mechanics-step-by-step).
- **kind:** local Kubernetes in Docker; see [mechanics](#5-the-mechanics-step-by-step).
- **Limit:** resource ceiling; see [worked example](#6-worked-example).
- **Namespace:** logical object scope; see [definitions](#4-definitions-and-vocabulary).
- **Plan:** proposed Terraform actions; see [mechanics](#5-the-mechanics-step-by-step).
- **Pod:** Kubernetes scheduling unit; see [mental model](#3-the-core-mental-model).
- **Port-forward:** diagnostic TCP tunnel; see [mechanics](#5-the-mechanics-step-by-step).
- **Provider:** Terraform plugin; see [definitions](#4-definitions-and-vocabulary).
- **Readiness:** eligibility signal for traffic; see [worked example](#6-worked-example).
- **Replica:** requested workload copy; see [worked example](#6-worked-example).
- **Request:** scheduler resource input; see [worked example](#6-worked-example).
- **Rollout:** workload revision transition; see [mechanics](#5-the-mechanics-step-by-step).
- **Selector:** label match; see [definitions](#4-definitions-and-vocabulary).
- **Service:** stable network abstraction; see [mental model](#3-the-core-mental-model).
- **State:** Terraform resource mapping; see [data context](#8-data-standards-and-regulatory-context).

## 16. Sources

- [Kubernetes: Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) — controller and rollout behavior.
- [Kubernetes: Services](https://kubernetes.io/docs/concepts/services-networking/service/) — service discovery and endpoint selection.
- [Kubernetes: probes](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/) — liveness, readiness, and startup semantics.
- [Kubernetes: resource management](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/) — requests and limits.
- [Kubernetes: ConfigMaps](https://kubernetes.io/docs/concepts/configuration/configmap/) — non-confidential configuration.
- [Kubernetes: Secrets](https://kubernetes.io/docs/concepts/configuration/secret/) — security cautions for sensitive values.
- [kind quick start](https://kind.sigs.k8s.io/docs/user/quick-start/) — local cluster creation and deletion.
- [Terraform state](https://developer.hashicorp.com/terraform/language/state) — state purpose and management.
- [Terraform sensitive data](https://developer.hashicorp.com/terraform/language/manage-sensitive-data) — sensitive values and state risks.
- [Terraform plan](https://developer.hashicorp.com/terraform/cli/commands/plan) — plan and saved-plan behavior.
- [Terraform Kubernetes provider](https://registry.terraform.io/providers/hashicorp/kubernetes/2.33.0/docs) — provider and resources used here.
- [NIST Cybersecurity Framework 2.0](https://www.nist.gov/cyberframework) — voluntary cybersecurity risk-management framework.
- [CNCF Spotify case study](https://www.cncf.io/case-studies/spotify/) — public organizational adoption context, not this lab's measured result.

## 17. Links

- [Module topic index](_index.md).
- [Environment and repository lab](01_environment-repo.md).
- [Demand forecast pipeline lab](02_demand-forecast-pipeline.md).
- [Warehouse simulation lab](06_warehouse-des.md).
- [Module 30 survey and lab map](../../../course/30_simulation_optimization_production_labs.md).
