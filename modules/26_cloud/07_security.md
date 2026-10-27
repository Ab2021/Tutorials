# Security

> Module 26 — Cloud, Kubernetes, and Streaming Production Labs
> Topic 07 of 09
> Estimated study time: 4 hours
> Prerequisites: 02, 03
> Version 2026.3

## 1. Why this topic exists

A clinical data platform holds data that is valuable to an attacker, regulated by law, and
expensive to lose. The security question is not whether the platform is "secure" but which specific
controls prevent which specific failures, and what evidence exists that they are working.

The reference stack is a useful study because it is honest about its gaps. The good parts are real:
`labs/cloud/k8s/03-deployment.yaml` drops all capabilities, disables privilege escalation, runs
read-only, and sets a seccomp profile; `labs/cloud/k8s/09-serviceaccount.yaml` disables the mounted
API token; `labs/cloud/k8s/08-networkpolicy.yaml` starts from default-deny; and
`labs/cloud/terraform/modules/storage/main.tf` blocks public access on the bucket, encrypts with a
customer managed key, and marks the registry repository immutable. The gaps are equally real: the
deployment references an image by tag rather than digest, `02-secret.example.yaml` is a template with
placeholder values, there is no transport layer security (TLS) between pods, the network policy allows HTTPS to any address,
and `cluster_endpoint_public_access = true` in `labs/cloud/terraform/modules/eks/main.tf`.

The purpose of this topic is to let you tell those apart, in a review, with evidence.

### 1.1 The threats that matter for a clinical platform

| Threat | Concrete form | Primary control |
|---|---|---|
| Credential theft | Long-lived cloud key in a CI variable | Workload identity, short-lived tokens |
| Lateral movement | Compromised pod reaches the database | Network segmentation, mutual transport layer security (mTLS) |
| Supply-chain compromise | Malicious dependency in a base image | software bill of materials (SBOM), signing, scanning |
| Data exfiltration | Public bucket, over-broad egress | Bucket policy, egress restriction |
| Insider access | Engineer reads protected health information (PHI) without a business need | Least privilege, audit |
| Secret leakage | Token in a manifest committed to git | External secret store, scanning |
| Container escape | Root container with capabilities | Non-root, dropped caps, seccomp |
| Denial of service | Unbounded request rate | Rate limits, quotas |

Each row is a control that can be tested. "We follow best practices" is not a test.

### 1.2 Why a checklist is not enough

A team can satisfy a checklist and remain exposed. The common pattern is controls that exist but are
not enforced: a NetworkPolicy object in a cluster whose container network interface (CNI) ignores it, an image scan that reports
findings nobody gates on, a secret rotated on a schedule but copied into three environments, or an
IAM role with the right name and `Action: "*"`.

The discipline is to pair every control with a verification. Section 5 states the verification for
each step, and Section 6 shows two diagnostics that reveal controls that look present and are not.

### 1.3 The regulatory frame

The Health Insurance Portability and Accountability Act (HIPAA) Security Rule at 45 CFR Part 164 Subpart C requires administrative, physical, and technical
safeguards for electronic PHI. The technical safeguards at 164.312 are the ones most relevant to a
cloud platform, but the administrative safeguards at 164.308 are what make them operate: risk
analysis, sanction policy, access management, and incident response.

NIST SP 800-207, published August 2020, supplies the architectural principle: no implicit trust
based on network location. That principle is why a pod-to-pod call inside a cluster needs
authentication, not only a network policy.

> **Note:** The lab stack is a teaching scaffold. It is not a compliant deployment. It has no TLS
> between services, no external secret manager, no image signature verification, and no audit
> pipeline. Do not put real PHI in it.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** a clinical data path to trust boundaries and name the control and the verification at
   each.
2. **Configure** workload identity so that a pod receives short-lived cloud credentials scoped to a
   role, and explain why node-level credentials are worse.
3. **Design** a secret path from a secret manager to a process without the secret entering an image,
   a manifest, or Terraform state.
4. **Apply** default-deny network policy and verify enforcement rather than assuming it.
5. **Generate** an SBOM, sign an image, and verify the signature at admission.
6. **Compute** the exposure window of a static credential versus a short-lived token at a stated
   request rate.
7. **Distinguish** a control that exists from a control that is enforced, and produce evidence for
   the second.

Outcomes 1 and 7 are assessed by Exercises 12.1 and 12.8. Outcomes 2 and 3 by 12.2 and 12.3.
Outcomes 4 and 5 by 12.4 and 12.5. Outcome 6 by 12.6.

## 3. The core mental model

Security in a cloud-native platform is a chain of identity assertions. Each hop asserts who is
calling, and each hop must be verified by the receiver. A break anywhere in the chain means the
later assertions are unverified claims.

```text
  human            workload             cloud API              data
  ┌──────┐  SSO   ┌──────────┐  token  ┌──────────┐  policy  ┌────────┐
  │ eng  │───────▶│ CI job   │────────▶│ IAM role │─────────▶│ S3/KMS │
  └──────┘  MFA   └──────────┘  OIDC   └──────────┘          └────────┘
                                             ▲
  ┌──────┐  OIDC  ┌──────────┐  projected  │
  │ user │───────▶│ ingress  │  token      │
  └──────┘        └────┬─────┘             │
                       ▼                   │
                  ┌──────────┐  mTLS   ┌──┴───────┐
                  │ pod A    │────────▶│ pod B    │
                  └──────────┘         └──────────┘
                       │
                       ▼
                  ┌──────────┐
                  │ audit log│
                  └──────────┘
```

The chain has four properties worth naming. Each link is authenticated with a credential whose
lifetime is short. Each credential is scoped to the minimum permission. Each use is logged with the
identity. Each identity can be revoked without rotating a shared secret.

### 3.1 Network location is not identity

NIST SP 800-207's second tenet states that all communication is secured regardless of network
location. A request from inside the cluster is not more trustworthy than one from outside, because
an attacker who compromises a pod is inside. Network policy reduces reachability; it does not
establish identity. Mutual TLS does.

### 3.2 The three secret states

A secret exists in three states, and each needs a different control:

| State | Example | Control |
|---|---|---|
| At rest | Database password in a secret manager | Encryption, access policy, rotation |
| In transit | Password sent to a pod | TLS, projected volume, no env logging |
| In use | Password in process memory | Least privilege, no core dumps, no logs |

Most breaches happen in the second or third state, not the first. A secret manager with excellent
encryption and a process that logs its environment is not protected.

### 3.3 The principle of enforceable controls

Every control should have a test that fails when the control is absent. The test is the difference
between a policy and a practice.

```text
control:   default-deny network policy
test:      a curl from an unapproved namespace must time out
failure:   the curl succeeds, so the CNI does not enforce policy

control:   image signature verification
test:      apply a pod with an unsigned image; admission must reject it
failure:   the pod starts, so verification is advisory
```

### 3.4 Blast radius is a design parameter

The number of paths an attacker can take after compromising one pod is a property of the network
design, not of the attacker. A flat network with forty pods has 1,560 directed pairs. Segmented into
tiers with explicit edges, it has far fewer. Section 6 computes the difference.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Workload identity | A mechanism that gives a workload a cloud identity without static keys | A service account | Federation |
| IRSA | IAM Roles for Service Accounts on EKS | Any pod identity | OIDC federation |
| Pod Identity | EKS mechanism using an agent and an association API | IRSA | Association |
| OIDC | OpenID Connect, an identity federation protocol | OAuth | Specification |
| STS | AWS Security Token Service | IAM user | Service |
| Short-lived credential | Token valid for minutes or hours | A rotated password | Duration |
| Secret manager | Service storing secrets with access control and audit | A Kubernetes Secret | Service |
| Sealed secret | Encrypted secret safe to commit | A plain Secret manifest | Tool-specific |
| Projected volume | Volume populated from API objects or identity tokens | A ConfigMap | Volume type |
| mTLS | Mutual TLS, both parties present certificates | One-way TLS | Protocol |
| SPIFFE | Identity framework for workloads | A certificate authority | Standard |
| NetworkPolicy | L3/L4 allow rules enforced by the CNI | A firewall | Object |
| CNI | Container Network Interface plugin | The cluster network | Plugin |
| Default deny | Policy that permits nothing until allowed | A blocking rule | Posture |
| Egress | Outbound traffic | Ingress | Direction |
| Pod Security Standards | Privileged, baseline, restricted profiles | A scanner | Admission |
| Admission controller | Component accepting or rejecting objects | A linter | Webhook |
| SBOM | Software bill of materials | A vulnerability report | SPDX or CycloneDX |
| Provenance | Record of how an artifact was built | A build log | in-toto |
| Sigstore | Signing and transparency project | A certificate authority | Project |
| Cosign | Container signing tool | A scanner | Tool |
| SLSA | Supply-chain levels for software artifacts | A certification | Level |
| Least privilege | Minimum permissions for the task | No permissions | Principle |
| Separation of duties | No single identity can complete a sensitive action alone | Job rotation | Principle |
| Break-glass | Emergency access path with extra logging | A shared admin account | Procedure |
| Audit log | Append-only record of access and actions | Debug log | Event record |
| Data classification | Label determining handling requirements | A tag | Schema |
| Egress allowlist | Named destinations permitted outbound | A proxy | Policy |
| Secret scanning | Detection of credentials in repositories | A linter | Tool |
| Key rotation | Replacing a cryptographic key | Changing a password | Procedure |
| BAA | Business Associate Agreement | A security certification | Contract |
| PHI | Protected health information | Any health data | Regulatory category |

Three distinctions carry the weight.

**Authentication versus authorization.** Authentication establishes who is calling. Authorization
decides what they may do. A valid token with broad permissions is authenticated and over-privileged.

**Encryption at rest versus access control.** Encryption protects a stolen disk. It does not protect
against an application with the key. Access control is what limits the application.

**Vulnerability versus exploitability.** A scanner reports the presence of a vulnerable component.
Whether it can be reached determines the risk. Both matter, and neither substitutes for the other.

## 5. The mechanics, step by step

The procedure builds the identity and segmentation chain, then attaches evidence. It follows the
reference stack and marks where the lab stops.

### Step 1 — Classify the data and the assets

**Input.** The data flows and the stores.

**Output.** A classification per store and per flow, and a list of assets that grant access.

**Rationale.** Controls follow classification. PHI needs encryption, audit, and least privilege. A
model artefact needs integrity. A public dataset needs availability. Without a classification, every
control is applied everywhere, which is expensive and usually incomplete.

**Branch point.** If the classification is ambiguous, treat the data as the higher class until
proven otherwise. The cost of over-classifying is operational; the cost of under-classifying is a
breach.

### Step 2 — Map trust boundaries

**Input.** The classification and the architecture diagram.

**Output.** A boundary diagram with the control and the verification at each crossing.

**Rationale.** A trust boundary is where an identity claim is checked. Typical boundaries in this
stack: internet to ingress, ingress to service, service to service, pod to cloud API, pod to
database, human to cluster, CI to registry.

**Branch point.** If a boundary has no control, it is not a boundary. It is an opening. Record it as
a finding rather than drawing a line on the diagram.

### Step 3 — Establish workload identity

**Input.** The cloud resources the workload must reach.

**Output.** An IAM role, a Kubernetes service account, and a binding between them.

**Rationale.** A workload needs cloud permissions to read a bucket or write to a queue. The wrong way
is to give the node a role and let every pod inherit it. The right way is to give each workload its
own role, scoped to its resources.

EKS offers two mechanisms. IAM Roles for Service Accounts (IRSA) uses the cluster's OIDC provider and
an annotated service account. EKS Pod Identity uses an agent and an association API, with a single
trust policy for the `pods.eks.amazonaws.com` service principal. AWS recommends Pod Identity for new
workloads; IRSA remains supported and is required for clusters below 1.24 and for some non-EKS
environments.

The lab sets `enable_irsa = true` in `labs/cloud/terraform/modules/eks/main.tf` but the service
account in `labs/cloud/k8s/09-serviceaccount.yaml` has no role annotation. The result is a pod with
no AWS permissions, which is safe and not yet useful.

**Edge case.** A pod that can reach the instance metadata service can obtain the node role's
credentials even without IRSA. Block pod access to `169.254.169.254` unless it is needed, and prefer
Pod Identity, which does not rely on the metadata endpoint in the same way.

**A test for the binding.** The only convincing evidence is a call from inside the pod:

```bash
kubectl exec -n biomed deploy/risk-service -- \
  aws sts get-caller-identity --query Arn --output text
```

The ARN should name the workload role, not `instance-profile/...`. If it names an instance profile,
the binding is missing and the pod is using the node's identity.

### Step 4 — Issue short-lived credentials

**Input.** The identity from Step 3 and the credential lifetime policy.

**Output.** Tokens with a bounded lifetime and automatic refresh.

**Rationale.** A static credential is valid until someone rotates it. A short-lived token is valid
for minutes or hours and is refreshed by the platform. The difference in exposure is proportional to
the ratio of the lifetimes, which Section 6 quantifies.

**Branch point.** If a batch job runs longer than the token lifetime, the job must refresh. A job
that caches the token in an environment variable at start will fail partway.

### Step 5 — Move secrets out of images and manifests

**Input.** The secrets the workload needs.

**Output.** A path from a secret manager to a mounted file or an injected environment variable, with
no secret in git, an image layer, or Terraform state.

**Rationale.** `labs/cloud/k8s/02-secret.example.yaml` is a template with placeholder values. It is
committed to demonstrate the shape, and committing a real value would be a breach. A Kubernetes
Secret is base64-encoded, not encrypted by default; encryption at rest depends on the cluster's
encryption configuration.

**Branch point.** Three common patterns, with different trade-offs:

| Pattern | Secret in git | Rotation | Operational cost |
|---|---|---|---|
| Kubernetes Secret manifest | Yes, unless sealed | Manual | Low |
| External Secrets Operator | No | Automatic from the manager | Medium |
| Secrets Store CSI driver | No | Automatic, mounted as a file | Medium |
| Init container fetch | No | Automatic on restart | Low, but the secret is in process memory |

**Edge case.** A secret mounted as a file is not updated in the running container unless the
application watches the file. An environment variable is never updated. Rotation without a restart
therefore requires a file-watching application or a rollout.

### Step 6 — Encrypt in transit everywhere

**Input.** The service graph.

**Output.** TLS at the ingress and mTLS between services.

**Rationale.** `labs/cloud/k8s/05-ingress.yaml` terminates TLS at the ingress with a secret named
`risk-service-tls`. That covers the client-to-edge hop. Inside the cluster, traffic between the
ingress controller and the pod, and between pods, is plaintext in the lab.

**Branch point.** A service mesh provides mTLS with workload certificates and policy, at the cost of
a sidecar or an eBPF data plane and a new operational surface. A CNI with transparent encryption
provides less granular identity. Choose based on whether you need per-service authorization or only
confidentiality.

### Step 7 — Segment the network

**Input.** The allowed traffic matrix.

**Output.** A default-deny policy and explicit allow policies per workload.

**Rationale.** `labs/cloud/k8s/08-networkpolicy.yaml` is a good shape: a `default-deny` with
`podSelector: {}` and both policy types, then an allow policy that permits ingress from
`ingress-nginx` and `monitoring` on port 8000, DNS egress on UDP 53, and HTTPS egress to
`0.0.0.0/0`.

Two weaknesses are worth naming. The HTTPS rule allows any destination, which permits exfiltration
to an arbitrary host on port 443. And the policy is enforced only if the CNI implements it.

**Branch point.** If the cluster uses a CNI without policy support, the objects are accepted and
ignored. Verify with a test connection from a namespace that should be blocked.

**Edge case.** DNS is the most common omission. Default-deny egress without a DNS allowance produces
timeouts on every outbound call, and the failure looks like an application problem.

**A traffic matrix first.** Before writing policy, list the edges. A minimal matrix for this stack:

| Source | Destination | Port | Purpose | Lab rule |
|---|---|---|---|---|
| ingress-nginx | risk-service | 8000 | Client requests | Present |
| monitoring | risk-service | 8000 | Metrics scrape | Present |
| risk-service | kube-dns | 53 | Name resolution | Present |
| risk-service | S3 endpoint | 443 | Model and data | Broad internet rule |
| risk-service | PostgreSQL | 5432 | Patient data | Absent |
| Anything else | risk-service | any | None | Denied |

The absent database edge is a finding: the application cannot reach its database under the current
policy, which means either the database is not deployed or the policy has not been updated. Both are
worth knowing before an incident.

### Step 8 — Harden the workload

**Input.** The pod specification.

**Output.** A pod that satisfies the Restricted Pod Security Standard.

**Rationale.** The reference deployment already does most of this: non-root, seccomp
`RuntimeDefault`, no privilege escalation, read-only root, all capabilities dropped, and no
automounted token. The remaining gap is a numeric `runAsUser`, which some policy engines require.

**Branch point.** If a workload genuinely needs a capability, add it explicitly and document why. A
blanket `privileged: true` for a workload that needs one capability is a common and unnecessary
expansion.

### Step 9 — Sign images and verify at admission

**Input.** The built image and a signing identity.

**Output.** A signature over the digest, and an admission policy that rejects unsigned images.

**Rationale.** A signature proves that a named identity approved a specific digest. Verification at
admission is what makes the signature a control. A signature that nobody checks is a record, not a
gate.

**Branch point.** If admission control is unavailable, a weaker substitute is a policy that rejects
tags and requires digests. It prevents accidental drift, not a determined attacker.

### Step 10 — Generate and attach an SBOM

**Input.** The image digest.

**Output.** An SBOM in SPDX or CycloneDX, attached to the image.

**Rationale.** The SBOM answers "which images contain library X" during an incident. In a
medical-device premarket submission, a software bill of materials is expected for cyber devices.
Generating it at build time and attaching it to the digest keeps it findable.

**Edge case.** An SBOM that is generated but not stored with the digest is effectively lost. Attach
it as an attestation or store it under a path that includes the digest.

### Step 11 — Scan and gate

**Input.** The image and the registry scanning configuration.

**Output.** A gate that blocks on a stated severity, with a documented waiver path.

**Rationale.** `labs/cloud/terraform/modules/storage/main.tf` enables `scan_on_push` on the Elastic Container Registry (ECR)
repository. That is ECR basic scanning, which covers operating-system packages and runs at most once
per image per 24 hours. Enhanced scanning, through Amazon Inspector, covers language packages and
can rescan continuously.

**Branch point.** A gate on critical findings only is easier to operate and misses exploitable
medium findings. A gate on all findings blocks delivery permanently. The workable compromise is
critical blocks, high requires a waiver with an expiry, medium is tracked.

### Step 12 — Audit and detect

**Input.** The access paths and the control plane.

**Output.** CloudTrail, Kubernetes audit logs, application audit records, and alerts on anomalies.

**Rationale.** HIPAA requires audit controls at 45 CFR 164.312(b). Detection requires more than
retention: it requires a baseline and an alert. A login from a new location, a role assumption from
an unexpected principal, or a sudden increase in S3 `GetObject` calls are detectable signals.

**Edge case.** Audit logs that can be deleted by the same administrator who is being audited are not
a control. Ship them to a separate account with restricted write access.

### Step 13 — Rotate, revoke, and rehearse

**Input.** The credential inventory.

**Output.** A rotation schedule, a revocation procedure, and a rehearsal record.

**Rationale.** A credential that has never been rotated is a credential whose rotation procedure has
never been tested. The first rotation during an incident is the wrong time to discover that the
application caches the old value.

**Branch point.** If rotation requires downtime, the procedure must state the window and the
approval. Zero-downtime rotation requires the application to accept both old and new credentials
during a transition.

### 5.1 Edge cases that change the answer

**Credential expiry mid-job.** A long batch job outlives its token. The job fails with an
authentication error that looks like a permissions problem. Implement refresh before expiry and
distinguish authentication failures from authorization failures.

**Image pull secret expiry.** A registry credential stored as a Kubernetes Secret expires. New pods
fail with `ErrImagePull` while running pods continue. Prefer node identity or a registry that
supports workload identity.

**Token audience mismatch.** A projected service account token is issued for one audience and
presented to another. The receiver rejects it. The error is usually a 401 with no detail, and the fix
is in the token projection, not the application.

**Node metadata access.** A pod can reach the instance metadata service and obtain the node role.
Block it with a network policy to `169.254.169.254/32` unless the workload needs it.

**Policy bypass through a new namespace.** A default-deny policy scoped to the `biomed` namespace
does not protect a new namespace created by a developer. Use a cluster-wide policy or a policy
engine that applies to all namespaces.

**Key deletion.** A KMS key scheduled for deletion makes every object encrypted with it unreadable.
The deletion window is a grace period, not a backup. Restrict `kms:ScheduleKeyDeletion` to a
break-glass role.

**Cluster endpoint exposure.** `cluster_endpoint_public_access = true` exposes the Kubernetes API to
the internet. Authentication still applies, but the attack surface includes every unpatched
client-side vulnerability. Use private access with a bastion, or restrict the public endpoint to
known CIDRs.

**Secret in Terraform state.** A database resource writes its generated password into state. State
read access is secret read access. Encrypt the state bucket and restrict its policy.

### 5.2 A decision table for secret delivery

| Requirement | K8s Secret | Sealed Secret | External Secrets Operator | CSI driver |
|---|---|---|---|---|
| No secret in git | No | Yes | Yes | Yes |
| Automatic rotation | No | No | Yes | Yes |
| Works without a controller | Yes | Yes | No | No |
| Audit of access | Cluster audit only | Cluster audit only | Manager audit | Manager audit |
| Mounted as a file | Yes | Yes | Yes | Yes |
| Simplest to operate | Yes | Yes | Medium | Medium |

### 5.3 A control-to-evidence table

Every control in a review should have a named artefact that proves it operates. The table is the
backbone of an evidence pack.

| Control | Evidence | System of record | Review cadence |
|---|---|---|---|
| Workload identity bound | `sts get-caller-identity` output | Ticket attachment or CI log | Quarterly |
| Least privilege role | IAM policy document and access analyzer findings | Terraform and IAM Access Analyzer | Quarterly |
| Secret not in git | Secret scanning result | Repository security tab | Per commit |
| Secret rotation | Rotation log with timestamps | Secret manager audit | Monthly |
| Encryption at rest | Bucket encryption configuration | Terraform state and AWS Config | Per change |
| Encryption in transit | TLS policy and certificate inventory | Load balancer config | Quarterly |
| Network segmentation | Negative connection test result | Test log | Per change |
| Pod hardening | Admission policy and a rejected-pod test | Policy engine log | Per change |
| Image signature | Signature verification log | Admission controller | Per deploy |
| Vulnerability gate | Scan result and gate decision | Registry and CI | Per build |
| Audit logging | Trail status and log destination | CloudTrail | Monthly |
| Access review | Reviewed user and role list | Identity provider | Quarterly |

The cadence column is the part teams forget. A control verified once at go-live decays. The cadence
is what turns a project into a process.

## 6. Worked example

A team runs a 36-pod clinical platform across three tiers: ingress, application, and data access.
A code review finds a database password in a CI variable that has been unchanged for 90 days. The
security lead asks how much exposure that represents and whether segmentation would contain a
compromise.

### 6.1 Inputs

| Parameter | Value | Source |
|---|---|---|
| Request rate | 20 requests/s | Load test |
| Static credential age | 90 days | CI variable creation date |
| Mean time to detection | 45 days | Assumption: half the credential lifetime |
| Proposed token lifetime | 1 hour | Workload identity default |
| Alternative token lifetime | 15 minutes | Stricter policy |
| Pods per tier | 12 | Deployment inventory |
| Tiers | 3 | Ingress, application, data |
| Cross-tier edges | 2 | Ingress to application, application to data |

### 6.2 Arithmetic: exposure window

Requests served during the static credential's exposure window:

```text
45 days × 86,400 s/day × 20 requests/s = 77,760,000 requests
```

With a one-hour token:

```text
3,600 s × 20 requests/s = 72,000 requests
```

With a fifteen-minute token:

```text
900 s × 20 requests/s = 18,000 requests
```

The reduction from static to one-hour is:

```text
77,760,000 ÷ 72,000 = 1,080
```

Three orders of magnitude. That ratio is the entire argument for workload identity, and it does not
depend on the request rate.

### 6.3 Arithmetic: blast radius

Fully connected, the number of directed pod-to-pod paths is:

```text
36 × 35 = 1,260
```

With three tiers of twelve pods, allowing full mesh within a tier and two cross-tier edges:

```text
within tiers: 3 × 12 × 11 = 396
cross tiers:  2 × 12 × 12 = 288
total:        396 + 288 = 684
reduction:    1 − 684/1,260 = 45.7 percent
```

The reduction is modest because intra-tier meshes are still large. Requiring every call to pass
through an authenticated service, rather than allowing a mesh, removes the intra-tier edges
entirely:

```text
cross tiers only: 2 × 12 × 12 = 288
reduction: 1 − 288/1,260 = 77.1 percent
```

That is the difference between segmentation and authorization. Network policy alone gives the first
number; mTLS with per-service authorization gives the second.

### 6.4 Arithmetic: control coverage

The pod posture check in Section 7 evaluates eight controls. The reference deployment passes seven
and fails one: the image is referenced by tag rather than digest.

```text
coverage = 7 ÷ 8 = 87.5 percent
```

That number is misleading in isolation, because the failing control is the one that ties the running
container to a reviewed artefact. A coverage percentage without weights hides which control failed.
Weight the controls by consequence, or report the failures by name, which is what the check does.

### 6.5 Arithmetic: secret rotation load

With 40 secrets and a 90-day rotation policy:

```text
40 ÷ 90 = 0.44 rotations per day
```

That is manageable manually. With a 30-day policy it is 1.33 per day, and with a 7-day policy it is
5.7 per day. The operational load scales inversely with the period, which is why rotation is
automated or not done.

### 6.6 Two diagnostics that reveal a misconfiguration

**Diagnostic 1: a pod assumes the node role.**

```bash
kubectl exec -n biomed deploy/risk-service -- \
  aws sts get-caller-identity --query Arn --output text
```

If the output names an instance role rather than a workload role, the service account is not bound
to an IAM role and the pod is inheriting the node's permissions. The cause is a missing annotation or
association, not a missing policy. The lab would produce exactly this result, because
`enable_irsa = true` creates the OIDC provider but the service account has no role ARN.

**Diagnostic 2: the network policy is present and not enforced.**

```bash
kubectl run nettest --rm -it --image=curlimages/curl -n default -- \
  curl -m 3 http://risk-service.biomed.svc.cluster.local/healthz
```

If the request succeeds from a namespace that the policy does not allow, the CNI is not enforcing
policy. The objects exist, the API accepted them, and the traffic flows. Confirm the CNI's policy
support before relying on any network policy.

A third diagnostic: `kubectl auth can-i --list --as=system:serviceaccount:biomed:risk-service` shows
what the service account can do in the API. If it lists more than discovery verbs, the RBAC binding
is broader than the workload needs.

A fourth is a scan of the rendered manifests for secrets:

```bash
grep -rniE '(password|secret|token|api[_-]?key)\s*[:=]\s*[^"$<{ ]' labs/cloud/k8s/
```

The reference `02-secret.example.yaml` will match, which is expected because it is an example. The
check is useful on rendered charts and environment overlays, where a real value can hide among
placeholders. A match is a prompt for review, not proof of a leak.

### 6.7 Vary one assumption

Reduce the token lifetime from one hour to fifteen minutes. Exposure falls from 72,000 requests to
18,000, a factor of four. The cost is four times as many token refreshes:

```text
pods × refreshes per hour
36 pods × 4 per hour = 144 token requests per hour
```

At 36 pods that is negligible. At 3,600 pods it is 14,400 per hour, and the identity provider becomes
a dependency with its own availability requirement. The correct lifetime is the longest that keeps
the exposure acceptable, not the shortest technically possible.

A second variation is the detection time. If monitoring improves from a 45-day mean to a 1-day mean,
the static credential's exposure falls to:

```text
1 day × 86,400 s × 20 requests/s = 1,728,000 requests
```

That is still 24 times the one-hour token's exposure. Detection time matters, but it cannot
compensate for a credential that never expires. The two controls multiply rather than substitute.

## 7. Code, query, or calculation

The following standard-library Python evaluates a pod specification against a subset of the
Restricted Pod Security Standard, and computes credential exposure and network blast radius. The pod
specification is the reference deployment translated to JSON.

```python
"""Workload security posture check and exposure-window model. Stdlib only."""
from __future__ import annotations

import json


LAB_POD = {
    "metadata": {"name": "risk-service", "namespace": "biomed"},
    "spec": {
        "automountServiceAccountToken": False,
        "securityContext": {"runAsNonRoot": True, "seccompProfile": {"type": "RuntimeDefault"}},
        "containers": [{
            "name": "api",
            "image": "ghcr.io/example/biomed-risk:0.1.0",
            "securityContext": {
                "allowPrivilegeEscalation": False,
                "readOnlyRootFilesystem": True,
                "capabilities": {"drop": ["ALL"]},
            },
            "env": [{"name": "LOG_LEVEL", "value": "INFO"}],
        }],
    },
}


def check_pod(pod: dict) -> list[tuple[str, str, str]]:
    findings: list[tuple[str, str, str]] = []
    spec = pod["spec"]
    pod_sc = spec.get("securityContext", {})
    if not spec.get("automountServiceAccountToken", True):
        findings.append(("PASS", "IDENTITY-1", "service account token not mounted"))
    else:
        findings.append(("FAIL", "IDENTITY-1", "service account token is mounted by default"))
    if pod_sc.get("runAsNonRoot"):
        findings.append(("PASS", "PSS-1", "pod runs as non-root"))
    else:
        findings.append(("FAIL", "PSS-1", "pod may run as root"))
    if pod_sc.get("seccompProfile", {}).get("type") in {"RuntimeDefault", "Localhost"}:
        findings.append(("PASS", "PSS-2", "seccomp profile set"))
    else:
        findings.append(("FAIL", "PSS-2", "no seccomp profile"))
    if spec.get("hostNetwork"):
        findings.append(("FAIL", "PSS-3", "hostNetwork enabled"))
    else:
        findings.append(("PASS", "PSS-3", "host network not used"))

    for container in spec["containers"]:
        name = container["name"]
        sc = container.get("securityContext", {})
        image = container.get("image", "")
        if "@sha256:" in image:
            findings.append(("PASS", "IMG-1", f"{name}: image pinned by digest"))
        else:
            findings.append(("FAIL", "IMG-1", f"{name}: image is not pinned by digest"))
        if sc.get("allowPrivilegeEscalation") is False:
            findings.append(("PASS", "PSS-4", f"{name}: privilege escalation disabled"))
        else:
            findings.append(("FAIL", "PSS-4", f"{name}: privilege escalation allowed"))
        if sc.get("readOnlyRootFilesystem"):
            findings.append(("PASS", "PSS-5", f"{name}: root filesystem read-only"))
        else:
            findings.append(("FAIL", "PSS-5", f"{name}: root filesystem writable"))
        drops = sc.get("capabilities", {}).get("drop", [])
        if "ALL" in [d.upper() for d in drops]:
            findings.append(("PASS", "PSS-6", f"{name}: all capabilities dropped"))
        else:
            findings.append(("FAIL", "PSS-6", f"{name}: capabilities not fully dropped"))
        if sc.get("privileged"):
            findings.append(("FAIL", "PSS-7", f"{name}: privileged container"))
    return findings


def exposure_requests(ttl_seconds: int, rps: float,
                      mean_detection_days: float | None = None) -> float:
    window = mean_detection_days * 86_400 if mean_detection_days is not None else ttl_seconds
    return window * rps


def reachable_pairs(tiers: int, per_tier: int, cross_edges: int) -> tuple[int, int]:
    pods = tiers * per_tier
    flat = pods * (pods - 1)
    segmented = tiers * per_tier * (per_tier - 1) + cross_edges * per_tier * per_tier
    return flat, segmented


def main() -> None:
    print(f"pod posture: {LAB_POD['metadata']['namespace']}/{LAB_POD['metadata']['name']}")
    for status, rule, message in check_pod(LAB_POD):
        print(f"  {status:4} {rule:11} {message}")

    print()
    print("credential exposure, 20 requests/s")
    static = exposure_requests(0, 20, mean_detection_days=45)
    hourly = exposure_requests(3600, 20)
    quarter = exposure_requests(900, 20)
    print(f"  static 90-day secret, 45-day mean detection: {static:,.0f} requests")
    print(f"  one-hour STS token:                          {hourly:,.0f} requests")
    print(f"  fifteen-minute STS token:                    {quarter:,.0f} requests")
    print(f"  reduction, static -> one hour:               {static/hourly:,.0f}x")

    print()
    print("reachable directed pod pairs")
    for tiers, per_tier, edges in ((3, 12, 2), (4, 10, 3), (5, 8, 4)):
        flat, seg = reachable_pairs(tiers, per_tier, edges)
        print(f"  tiers={tiers} pods/tier={per_tier:2d} edges={edges}  "
              f"flat={flat:5d}  segmented={seg:5d}  reduction={1-seg/flat:5.1%}")

    print()
    pod = json.loads(json.dumps(LAB_POD))
    pod["spec"]["containers"][0]["securityContext"]["readOnlyRootFilesystem"] = False
    pod["spec"]["automountServiceAccountToken"] = True
    changed = check_pod(pod)
    fails = [f for f in changed if f[0] == "FAIL"]
    print(f"after two regressions: {len(fails)} failures")
    for status, rule, message in fails:
        print(f"  {status:4} {rule:11} {message}")


if __name__ == "__main__":
    main()
```

Actual output:

```text
pod posture: biomed/risk-service
  PASS IDENTITY-1  service account token not mounted
  PASS PSS-1       pod runs as non-root
  PASS PSS-2       seccomp profile set
  PASS PSS-3       host network not used
  FAIL IMG-1       api: image is not pinned by digest
  PASS PSS-4       api: privilege escalation disabled
  PASS PSS-5       api: root filesystem read-only
  PASS PSS-6       api: all capabilities dropped

credential exposure, 20 requests/s
  static 90-day secret, 45-day mean detection: 77,760,000 requests
  one-hour STS token:                          72,000 requests
  fifteen-minute STS token:                    18,000 requests
  reduction, static -> one hour:               1,080x

reachable directed pod pairs
  tiers=3 pods/tier=12 edges=2  flat= 1260  segmented=  684  reduction=45.7%
  tiers=4 pods/tier=10 edges=3  flat= 1560  segmented=  660  reduction=57.7%
  tiers=5 pods/tier= 8 edges=4  flat= 1560  segmented=  536  reduction=65.6%

after two regressions: 3 failures
  FAIL IDENTITY-1  service account token is mounted by default
  FAIL IMG-1       api: image is not pinned by digest
  FAIL PSS-5       api: root filesystem writable
```

The check is deliberately narrow. It does not inspect the image contents, verify signatures, or
read the network policy. It answers one question: does the pod spec satisfy a stated subset of the
Restricted profile. That is a useful gate in a pull request and not a substitute for a scanner.

The following policy is an EKS Pod Identity association expressed in HCL. It is validated by parsing,
not applied.

```hcl
resource "aws_eks_pod_identity_association" "risk_service" {
  cluster_name    = module.eks.cluster_name
  namespace       = "biomed"
  service_account = "risk-service"
  role_arn        = aws_iam_role.risk_service.arn
}

resource "aws_iam_role" "risk_service" {
  name = "biomed-risk-service"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "pods.eks.amazonaws.com" }
      Action = ["sts:AssumeRole", "sts:TagSession"]
    }]
  })
}
```

The single trust policy replaces a per-cluster OIDC provider entry. That is the operational advantage
of Pod Identity: adding a cluster does not require editing every role's trust policy.

## 8. Data, standards, and regulatory context

### 8.1 HIPAA technical safeguards

45 CFR 164.312 contains five standards. Each maps to a control in this topic:

| Clause | Standard | Control here | Lab status |
|---|---|---|---|
| 164.312(a)(1) | Access control | RBAC, workload identity, network policy | Partial |
| 164.312(a)(2)(i) | Unique user identification (required) | Per-workload service accounts | Present |
| 164.312(a)(2)(iii) | Automatic logoff (addressable) | Token expiry | Present |
| 164.312(a)(2)(iv) | Encryption and decryption (addressable) | KMS at rest, TLS in transit | Partial |
| 164.312(b) | Audit controls | CloudTrail, Kubernetes audit | Absent |
| 164.312(c)(1) | Integrity | Immutable tags, versioning | Present |
| 164.312(d) | Person or entity authentication | SSO with MFA | Out of scope |
| 164.312(e)(1) | Transmission security | TLS at ingress | Partial |

"Addressable" requires an assessment, not a decision to skip. A covered entity that does not
implement encryption must document why it is not reasonable and appropriate and what equivalent
measure is in place.

### 8.2 Administrative safeguards

164.308 requires a risk analysis, a risk management plan, a sanction policy, access authorization,
and incident response. These are process controls, and no amount of Kubernetes configuration
substitutes for them. An audit will ask for the risk analysis before it asks for the network
policy.

### 8.3 NIST SP 800-207

The seven tenets, published August 2020, are:

1. All data sources and computing services are considered resources.
2. All communication is secured regardless of network location.
3. Access to individual enterprise resources is granted on a per-session basis.
4. Access to resources is determined by dynamic policy.
5. The enterprise monitors and measures the integrity and security posture of all owned and
   associated assets.
6. All resource authentication and authorization are dynamic and strictly enforced before access.
7. The enterprise collects as much information as possible about the current state of assets,
   network infrastructure, and communications and uses it to improve its security posture.

Tenets 2, 3, and 6 are the ones a Kubernetes platform most often fails, because the cluster's flat
network and long-lived service account tokens encourage implicit trust.

### 8.4 Medical device cybersecurity

Section 524B of the Federal Food, Drug, and Cosmetic Act requires premarket submissions for cyber
devices to include a software bill of materials and a plan for monitoring and patching
vulnerabilities. NIST SP 800-218 provides a secure development framework. Both apply to device
software regardless of whether it runs in a container.

### 8.5 Business Associate Agreements

A covered entity must have a BAA with each business associate that handles PHI. The major cloud
providers offer BAAs that cover a defined list of services. Using a service that is not on the
HIPAA-eligible list, even inside a covered account, is outside the BAA. The AWS Healthcare Industry
Lens directs architects to verify eligibility per service.

### 8.6 What the standards do not settle

They do not prescribe Pod Identity over IRSA, a service mesh, a signing tool, or a secret manager.
They constrain the outcomes and the evidence. A platform can satisfy every clause with different
tools.

## 9. Industry practice and case evidence

### 9.1 Documented practice

**Pod Identity is the recommended default for new EKS workloads.** AWS documentation and the
launch material describe Pod Identity as removing the per-cluster OIDC provider requirement, with a
single trust policy for `pods.eks.amazonaws.com`, support for session tags and attribute-based access
control, and an association API for inventory. IRSA remains supported and is required outside EKS
and on older clusters.

**IRSA has documented scaling limits.** The launch comparison cites a limit of 100 OIDC providers
per account and a 2 KB trust policy size limit, which constrains how many role-to-service-account
mappings fit in one policy. Those limits are why Pod Identity was introduced.

**Network policy requires CNI support.** The Kubernetes documentation states that a NetworkPolicy
without a plugin that implements it has no effect. Calico's documentation repeats the point. This is
the most common false sense of security in Kubernetes security reviews.

**ECR scanning has two modes with different coverage.** Basic scanning covers operating-system
packages, once per image per 24 hours, up to 100,000 images per 24 hours per registry. Enhanced
scanning uses Amazon Inspector, covers operating-system and language packages, and supports
continuous rescanning.

### 9.2 Case evidence

1. **The 2023 MOVEit Transfer vulnerability (CVE-2023-34362)** affected healthcare organisations
   disproportionately because the software was widely used for file transfer of PHI. It is a
   documented example of a supply-chain vulnerability with sector-specific impact.
2. **The 2024 Change Healthcare incident** disrupted claims processing across the US for weeks. The
   public reporting describes a compromised remote access pathway and a lack of multi-factor
   authentication on a critical service. The lesson for this topic is that identity controls on
   remote access matter more than container hardening.
3. **The 2020 SolarWinds Orion compromise** demonstrated that a trusted build pipeline can
   distribute malicious code. It is the reason provenance and signature verification appear in
   supply-chain frameworks.

These are public events, not healthcare-platform post-mortems. The specific lessons drawn are
inferences.

The common thread is access, not exploitation. In each case an identity or a trust relationship was
used as intended by an attacker, and the controls that would have limited the damage were
segmentation, least privilege, and detection. Container hardening would not have prevented any of
them, which is why Section 5 puts identity and segmentation before image controls.

### 9.3 Inference

The following are inferences, labelled as such:

- Most Kubernetes security findings in clinical environments are identity findings, not
  container-escape findings. Over-broad roles and long-lived credentials are more common than
  privileged pods.
- Teams that enable default-deny before they have an inventory of required traffic spend a week
  debugging DNS. The order matters: inventory, then deny, then allow.
- Signature verification is rare in practice compared with signing. The gap means the signature is
  evidence after an incident, not a preventive control.

One further inference concerns the order of adoption. Teams that start with network segmentation
before they have an inventory of dependencies spend their first week restoring DNS and their second
week restoring a database connection. Teams that start with identity get a faster first win: a
workload role is a bounded change, it removes a static credential, and it is easy to verify. The
practical order is identity, then secrets, then segmentation, then supply chain. That ordering also
matches the exposure arithmetic in Section 6, where the credential lifetime dominates.

The order has an exception. If the platform already has a compromised-credential incident behind it,
segmentation moves first, because it limits what a stolen credential can reach. The general rule is
about building capability; the exception is about containing a known problem. State which situation
you are in before recommending an order, because the two lead to different first steps.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Node role inherited by pods | `sts get-caller-identity` names an instance role | No workload identity binding | Every pod has node permissions | IRSA or Pod Identity per workload |
| Long-lived access key | Key older than 90 days in CI | Static credential | Large exposure window | Short-lived tokens via OIDC |
| Secret in git | Secret scanning alert | Committed manifest | Credential leak | External secret store, history purge |
| Secret in image layer | `docker history` shows a value | `ENV` or `ARG` | Leak to anyone with pull | Build secret mount |
| Unencrypted pod traffic | Plaintext capture succeeds | No mTLS | PHI exposure inside cluster | Service mesh or CNI encryption |
| Network policy not enforced | Curl succeeds from a blocked namespace | CNI lacks policy support | False segmentation | Verify with a test |
| Egress to 0.0.0.0/0 | Any destination on 443 allowed | Broad allow rule | Exfiltration path | Named endpoints or proxy |
| Public cluster endpoint | API reachable from the internet | `public_access = true` | Brute force, client exploits | Private endpoint |
| Over-broad IAM role | `Action: "*"` in policy | Convenience | Full account compromise | Least privilege, access analyzer |
| Unsigned image | No admission policy | Signature not verified | Supply-chain risk | Sign and verify |
| Scan without gate | Findings dashboard only | No enforcement | Vulnerable image ships | Severity gate |
| Audit log deletable | Same admin can delete logs | No separation | Evidence lost | Separate log account |
| KMS key deletable | No break-glass restriction | Broad `kms:*` | Data unrecoverable | Restrict deletion, enable rotation |
| Token in logs | Authorization header logged | Debug logging | Credential leak | Header redaction |
| Unpinned image in production | Tag in the pod spec | Registry tag repointed | Unreviewed code runs | Digest plus admission policy |
| Over-broad namespace policy | One policy for all pods | Convenience | No real segmentation | Per-workload policies |
| Audit destination writable | Administrators can delete records | No separation of duties | Evidence destroyed | Append-only store in a separate account |
| Egress endpoint unprotected | Traffic reaches an internal service directly | No private endpoint policy | Bypass of the intended path | Endpoint policy and security group |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Mean detection is half the credential lifetime | Exposure arithmetic | Monitoring improves or degrades | Measure detection time |
| Request rate is constant | Exposure in requests | Burst traffic | Compare hourly rates |
| The CNI enforces network policy | Segmentation claim | Plugin without policy | Attempt a blocked connection |
| The identity provider is available | Short-lived credentials | Outage | Check provider SLA |
| Secrets are not cached in environment variables | Rotation claim | Application reads once at start | Rotate and observe |
| The image registry is trusted | Supply chain | Compromised registry | Verify signatures |
| Audit logs are immutable | Evidence claim | Same admin can delete | Attempt deletion |
| KMS key access is restricted | Data protection | Broad key policy | Review key policy |
| The pod cannot reach metadata | Node role isolation | No egress rule for 169.254.169.254 | Exec and curl the metadata endpoint |
| Human access uses SSO with MFA | Authentication claim | Local cluster users | List cluster auth mappings |
| The registry is private | Supply-chain claim | Public image reference | `docker pull` without credentials |
| Policy applies to all namespaces | No bypass | Cluster-scoped policy absent | Create a test namespace and apply a pod |
| The audit destination is append-only | Evidence integrity | Write access for administrators | Attempt to delete an audit object |
| The private endpoint policy is scoped | No bypass of the intended path | Open endpoint policy | Describe the endpoint policy |

## 12. Exercises

### 12.1 Map trust boundaries

**Deliverable.** A diagram of the lab stack from ingress to S3, with every trust boundary, the
control at each, and whether the control is present in the lab.

**Acceptance criteria.** At least six boundaries. Each control is a named mechanism. At least three
are marked absent.

**Time.** 45 minutes.

### 12.2 Configure workload identity

**Deliverable.** The Terraform and Kubernetes changes needed to give `risk-service` read access to
one S3 prefix using Pod Identity, with the role policy and the association.

**Acceptance criteria.** Role scoped to the prefix. Trust policy names the Pod Identity service
principal. Service account name matches. No node role reuse.

**Time.** 60 minutes.

### 12.3 Design a secret path

**Deliverable.** A design for delivering a database password from a secret manager to the pod without
the value entering git, an image, or Terraform state, including rotation.

**Acceptance criteria.** Names the delivery mechanism. Explains how rotation reaches the process.
States what happens during the transition.

**Time.** 45 minutes.

### 12.4 Verify network policy

**Deliverable.** A test procedure that proves whether the CNI enforces the policies in
`labs/cloud/k8s/08-networkpolicy.yaml`, with expected pass and fail outputs.

**Acceptance criteria.** Includes a positive test (allowed traffic succeeds) and a negative test
(blocked traffic fails). Names the CNI check.

**Time.** 30 minutes.

### 12.5 Sign and verify

**Deliverable.** A pipeline stage that signs an image digest and an admission policy that rejects
unsigned images, with the verification identity named.

**Acceptance criteria.** Signature is over the digest, not the tag. Verification happens at
admission. The failure mode when the verifier is unavailable is stated.

**Time.** 60 minutes.

### 12.6 Compute exposure

**Deliverable.** The exposure window in requests for a static credential aged 180 days with a
60-day mean detection time, at 5 requests per second, compared with a 30-minute token.

**Acceptance criteria.** Correct arithmetic. Ratio stated. Assumption about detection time made
explicit.

**Time.** 25 minutes.

### 12.7 Review the lab manifests

**Deliverable.** A findings list for `labs/cloud/k8s/` and `labs/cloud/terraform/`, each finding with
severity, evidence, and a fix.

**Acceptance criteria.** At least eight findings. At least two are in Terraform. Each cites a file
and a line.

**Time.** 50 minutes.

### 12.8 Design an evidence pack

**Deliverable.** A list of the evidence an auditor would request for the technical safeguards, and
where each piece is produced.

**Acceptance criteria.** Covers access control, audit, integrity, authentication, and transmission.
Each item names a system of record.

**Time.** 40 minutes.

## 13. Self-check answers

### 13.1 Trust boundaries

The boundaries are: internet to ingress (TLS, WAF), ingress to service (network policy, authentication),
service to service (mTLS, absent), pod to AWS API (workload identity, absent), pod to database
(credentials, TLS, absent), CI to registry (OIDC, signature, absent), human to cluster (SSO, RBAC,
absent in the lab). The lab provides the ingress TLS and the network policy; the rest are gaps.

### 13.2 Workload identity

Create an IAM role with a policy allowing `s3:GetObject` on
`arn:aws:s3:::biomed-dev-raw-*/models/*`. Create an EKS Pod Identity association for cluster
`biomed-dev`, namespace `biomed`, service account `risk-service`. Remove any node role permissions
that the pod no longer needs.

### 13.3 Secret path

Use the Secrets Store CSI driver with the AWS provider, mounting the secret as a file at
`/var/run/secrets/db/password`. The pod's service account is bound to a role that can read that
secret. Rotation updates the secret in Secrets Manager; the CSI driver refreshes the mounted file on
its rotation poll interval; the application watches the file and reconnects. During the transition,
the database accepts both the old and new password.

### 13.4 Network policy verification

Positive test: a curl from the `ingress-nginx` namespace to `risk-service:8000` returns 200. Negative
test: a curl from the `default` namespace times out. If the negative test succeeds, the CNI does not
enforce policy. Confirm the CNI by checking the cluster's network plugin.

### 13.5 Signing

Sign the digest with a workload identity in CI, store the signature in the registry, and configure
an admission policy that verifies the signature against the expected identity and rejects anything
else. If the verifier is unavailable, the policy should fail closed, which means a verifier outage
blocks deploys.

### 13.6 Exposure

```text
60 days × 86,400 × 5 = 25,920,000 requests
30 minutes = 1,800 s × 5 = 9,000 requests
ratio = 2,880
```

### 13.7 Lab findings

Examples: image referenced by tag in `03-deployment.yaml`; placeholder secret in
`02-secret.example.yaml`; public cluster endpoint in `modules/eks/main.tf`; broad HTTPS egress in
`08-networkpolicy.yaml`; no startup probe; no numeric `runAsUser`; cluster version 1.30 unsupported;
no TLS secret creation for the ingress; no audit log configuration; bucket lifecycle deletes
noncurrent versions after 30 days, which may conflict with retention.

### 13.8 Evidence pack

Access control: IAM policies and RBAC bindings from Terraform and the cluster. Audit: CloudTrail
trails, Kubernetes audit policy and log destination, application audit table. Integrity: image
digests, signatures, S3 versioning configuration. Authentication: identity provider configuration and
MFA policy. Transmission: TLS certificate configuration and a scan result showing no plaintext
listeners.

## 14. Interview questions

**1. Why is a node IAM role worse than a workload identity?**
A strong answer says every pod on the node inherits the role, so the blast radius is the node, and
there is no per-workload audit. A weak answer says it is less secure without saying why.

**2. What does a NetworkPolicy guarantee?**
A strong answer says it restricts L3/L4 traffic only if the CNI enforces it, and that it is not
authentication. A weak answer says it isolates the pod.

**3. Where should a secret live?**
A strong answer describes a secret manager, a delivery mechanism, rotation, and the three states of
a secret. A weak answer says "not in git".

**4. What does image signing prove?**
A strong answer says a named identity approved a specific digest, and that it proves nothing unless
verified at admission. A weak answer says the image is safe.

**5. How do you compute the risk of a long-lived credential?**
A strong answer multiplies the exposure window by the request rate and compares with a short-lived
token, and states the detection-time assumption. A weak answer says long-lived credentials are bad.

**6. What is the difference between IRSA and Pod Identity?**
A strong answer covers OIDC provider per cluster versus a service principal trust, the association
API, and the environments each supports. A weak answer says both give pods AWS permissions.

**7. How would you prove segmentation works?**
A strong answer describes a negative test from a namespace that should be blocked. A weak answer
says to check that the policy exists.

**8. What is an SBOM for?**
A strong answer says inventory during an incident and a regulatory submission artefact, and that it
is not a vulnerability report. A weak answer says it lists dependencies.

**9. What happens if the KMS key is deleted?**
A strong answer says the data becomes unreadable permanently after the deletion window, and that the
window is a grace period, not a backup. A weak answer says the data is encrypted.

**10. How do you handle break-glass access?**
A strong answer describes a separate identity, strong authentication, session recording, alerting,
and a post-use review. A weak answer says "an admin account".

## 15. Key terms

- **Admission controller** — API server component accepting or rejecting objects.
- **BAA** — Business Associate Agreement.
- **Blast radius** — Scope affected by a compromise.
- **Break-glass** — Emergency access path with extra logging.
- **CNI** — Container Network Interface plugin.
- **Cosign** — Container signing tool.
- **Default deny** — Policy permitting nothing until allowed.
- **Egress** — Outbound traffic.
- **IRSA** — IAM Roles for Service Accounts.
- **KMS** — Key Management Service.
- **Least privilege** — Minimum permissions for the task.
- **mTLS** — Mutual TLS.
- **OIDC** — OpenID Connect.
- **PHI** — Protected health information.
- **Pod Identity** — EKS workload identity mechanism.
- **Pod Security Standards** — Privileged, baseline, restricted profiles.
- **Provenance** — Record of how an artifact was built.
- **RBAC** — Role-based access control.
- **SBOM** — Software bill of materials.
- **Secret manager** — Service storing secrets with access control and audit.
- **Separation of duties** — No single identity completes a sensitive action alone.
- **Short-lived credential** — Token valid for minutes or hours.
- **Sigstore** — Signing and transparency project.
- **SLSA** — Supply-chain levels for software artifacts.
- **SPIFFE** — Workload identity framework.
- **STS** — Security Token Service.
- **Threat model** — Structured description of threats and controls.
- **Trust boundary** — Point where identity or policy is enforced.
- **Workload identity** — Cloud identity for a workload without static keys.

## 16. Sources

All sources accessed 26 October 2026.

1. NIST. *SP 800-207: Zero Trust Architecture*. August 2020.
   https://csrc.nist.gov/pubs/sp/800/207/final
2. HHS. *HIPAA Security Rule*.
   https://www.hhs.gov/hipaa/for-professionals/security/index.html
3. Cornell Legal Information Institute. *45 CFR 164.312*.
   https://www.law.cornell.edu/cfr/text/45/164.312
4. Cornell Legal Information Institute. *45 CFR 164.308*.
   https://www.law.cornell.edu/cfr/text/45/164.308
5. AWS. *Amazon EKS — IAM roles for service accounts*.
   https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html
6. AWS. *Amazon EKS — Access control*.
   https://docs.aws.amazon.com/eks/latest/userguide/cluster-auth.html
7. AWS Containers Blog. *Amazon EKS Pod Identity*.
   https://aws.amazon.com/blogs/containers/amazon-eks-pod-identity-a-new-way-for-applications-on-eks-to-obtain-iam-credentials/
8. AWS. *Scan images for software vulnerabilities in Amazon ECR*.
   https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-scanning.html
9. Kubernetes. *Network Policies*.
   https://kubernetes.io/docs/concepts/services-networking/network-policies/
10. Kubernetes. *Pod Security Standards*.
    https://kubernetes.io/docs/concepts/security/pod-security-standards/
11. NIST. *SP 800-218: Secure Software Development Framework*.
    https://csrc.nist.gov/pubs/sp/800/218/final
12. SLSA. *Supply-chain Levels for Software Artifacts*. https://slsa.dev/spec/v1.0/
13. AWS. *Healthcare Industry Lens*.
    https://docs.aws.amazon.com/wellarchitected/latest/healthcare-industry-lens/healthcare-industry-lens.html
14. CISA. *Known Exploited Vulnerabilities Catalog*. https://www.cisa.gov/known-exploited-vulnerabilities-catalog

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`06_observability.md`](06_observability.md)
- Next topic: [`08_cost-and-reliability.md`](08_cost-and-reliability.md)
- Related topic: [`02_containers.md`](02_containers.md)
- Survey file: [`../../course/26_cloud_streaming_production_labs.md`](../../course/26_cloud_streaming_production_labs.md)
- Lab root: [`../../labs/`](../../labs/)
- Reference stack: [`../../labs/cloud/`](../../labs/cloud/)
- Service account: [`../../labs/cloud/k8s/09-serviceaccount.yaml`](../../labs/cloud/k8s/09-serviceaccount.yaml)
- Secret template: [`../../labs/cloud/k8s/02-secret.example.yaml`](../../labs/cloud/k8s/02-secret.example.yaml)
- Network policy: [`../../labs/cloud/k8s/08-networkpolicy.yaml`](../../labs/cloud/k8s/08-networkpolicy.yaml)
- Storage and ECR: [`../../labs/cloud/terraform/modules/storage/main.tf`](../../labs/cloud/terraform/modules/storage/main.tf)
- EKS module: [`../../labs/cloud/terraform/modules/eks/main.tf`](../../labs/cloud/terraform/modules/eks/main.tf)
