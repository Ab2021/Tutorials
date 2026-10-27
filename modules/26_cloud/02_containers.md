# Containers

> Module 26 — Cloud, Kubernetes, and Streaming Production Labs
> Topic 02 of 09
> Estimated study time: 3 hours
> Prerequisites: 01
> Version 2026.3

## 1. Why this topic exists

A container image is the unit of deployment in every cloud-native biomedical stack. It is also the
unit of supply-chain risk, the unit of reproducibility, and the unit that determines how fast a
rollout completes and how much disk a node needs. Teams that treat the Dockerfile as an
afterthought discover the consequences later: a 2 GB image that takes four minutes to pull on every
node, a container running as root that a security review rejects, a build that cannot be reproduced
because the base image tag moved, or a secret baked into a layer and pushed to a registry.

The biomedical context sharpens each of these. A model service carries a model artifact whose
identity must be recorded. A clinical API handles protected health information (PHI) and therefore
must not log it, must not bake credentials into the image, and must run with the least privilege the
runtime allows. A regulated device must be able to show what software was running on a given date,
which means the image digest, the software bill of materials (SBOM), and the build provenance must
all exist and be linked.

The reference stack makes the target concrete. `labs/cloud/terraform/modules/storage/main.tf`
creates an Elastic Container Registry (ECR) repository with `image_tag_mutability = "IMMUTABLE"` and
`scan_on_push = true`. `labs/cloud/k8s/03-deployment.yaml` runs a non-root container with a
read-only root filesystem, all Linux capabilities dropped, and a tmpfs mount for the one directory
that must be writable. Those two files are the specification this topic teaches you to satisfy.

### 1.1 The four failures containers are supposed to prevent

| Failure | Without containers | With containers and discipline |
|---|---|---|
| "Works on my machine" | Dependency drift between laptop and server | Image digest identifies the exact filesystem |
| Silent dependency change | `pip install` resolves differently six months later | Lockfile with hashes, pinned base digest |
| Privilege sprawl | Service runs as a system account with broad rights | Non-root user, dropped capabilities, read-only root |
| Unknowable contents | Nobody can list the installed packages | SBOM generated at build, stored with the digest |

Containers alone deliver only the first row. The other three require the build discipline in
Section 5.

### 1.2 Why this is an architecture topic and not a packaging chore

Image size determines rollout time, which determines how quickly a bad release can be replaced and
how much a node's ephemeral storage is consumed by cached layers. Image contents determine the
attack surface that a scanner reports and a security reviewer argues about. Image identity
determines whether an incident review can answer "which model produced this score". These are
architecture properties, and they are set in the Dockerfile.

### 1.3 What the lab stack does and does not do

`labs/cloud/k8s/03-deployment.yaml` pins `ghcr.io/example/biomed-risk:0.1.0` and sets
`imagePullPolicy: IfNotPresent`. The tag is a version, not a digest, and it is hosted on a registry
that the Terraform stack does not create. A production configuration uses the ECR repository from
`labs/cloud/terraform/modules/storage/main.tf`, references the image by digest, and uses an
image pull secret or node role scoped to that repository. The lab's container security context is
sound and worth copying; its registry wiring is not.

> **Note:** The lab stack is a teaching scaffold. It does not implement image signing, admission
> verification, or a vulnerability gate, and it is not a compliant deployment.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** the difference between an image, a tag, a digest, and a manifest, and state which
   one a deployment should reference.
2. **Write** a multi-stage Dockerfile that produces a non-root, read-only-compatible runtime image
   with a pinned base and hashed dependencies.
3. **Compute** image pull time and node ephemeral-storage consumption for a stated image size,
   node type, and bandwidth.
4. **Generate** an SBOM and a provenance record, and state what each one proves and does not prove.
5. **Configure** registry scanning and interpret the difference between operating-system and
   language-package findings.
6. **Diagnose** a `CrashLoopBackOff` caused by a read-only filesystem and an `ImagePullBackOff`
   caused by an architecture mismatch.
7. **Decide** which of four base-image strategies fits a given service, and defend the trade-off
   between image size and debugging capability.

Outcomes 1 and 6 are assessed by Exercises 12.1 and 12.7. Outcomes 2 and 3 by 12.2 and 12.3.
Outcomes 4 and 5 by 12.4 and 12.5. Outcomes 7 by 12.6 and 12.8.

## 3. The core mental model

An Open Container Initiative (OCI) image is a stack of content-addressed layers plus a small
configuration document. The image digest is computed over the manifest, which lists the layer
digests and the configuration digest. Change one byte in one layer and the digest changes.

```text
        manifest (application/vnd.oci.image.manifest.v1+json)
        ┌──────────────────────────────────────────────┐
        │ config digest ─────────────┐                 │
        │ layer 1 digest ──────────┐ │                 │
        │ layer 2 digest ────────┐ │ │                 │
        │ layer 3 digest ──────┐ │ │ │                 │
        └──────────────────────┼─┼─┼─┼─────────────────┘
                               │ │ │ │
                               ▼ ▼ ▼ ▼
                          ┌─────────────────┐
                          │  image digest   │  sha256:...
                          └─────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
        registry tag          SBOM (SPDX or       provenance
        (mutable)             CycloneDX)          attestation
```

The tag is a pointer into the registry. The digest is the identity. A Deployment that references a
tag is asking the registry "whatever you currently call 0.1.0"; a Deployment that references a
digest is asking for one specific filesystem.

### 3.1 Layers are the cache, and the cache is the build model

Each instruction that changes the filesystem creates a layer. The builder reuses a cached layer if
the instruction and all preceding layers are byte-identical. This is why dependency installation
comes before source copy: copying source first invalidates the dependency layer on every commit.

```text
  COPY requirements.txt .     ← changes rarely  ─┐
  RUN pip install ...          ← cached          ─┤ stable layers
  COPY app ./app               ← changes often   ─┘ rebuilt layer
```

A build that copies the repository first and installs dependencies second rebuilds everything on
every commit. The symptom is a slow continuous integration (CI) pipeline, and the fix is ordering,
not a bigger runner.

### 3.2 The runtime contract

The image must satisfy a contract that the orchestrator depends on:

| Contract term | Consequence if violated |
|---|---|
| Listens on a declared port | Service routing fails, probes fail |
| Terminates on SIGTERM within the grace period | Requests dropped during rollout |
| Responds to a readiness endpoint only when able to serve | Traffic sent to a starting pod |
| Writes only to declared paths | Crash under a read-only root filesystem |
| Runs as a non-root user | Security context rejection, or a root-owned file surprise |
| Has no runtime dependency on a shell | Exec probes and init containers fail |

The reference deployment encodes several of these: `containerPort: 8000`, a `preStop` sleep, a
readiness probe on `/readyz`, a liveness probe on `/healthz`, `readOnlyRootFilesystem: true`, and an
`emptyDir` mounted at `/tmp`. A container that writes to `/app/cache` instead of `/tmp` will crash
under that manifest, and the crash will look like an application bug.

### 3.3 Size is a latency and a storage parameter

Image size affects three things that appear in different parts of the system:

1. **Cold start.** A node that has never pulled the image must download it before the container
   starts. On a 500 Mbit/s link, 1.2 GB takes about 19 seconds at full rate and considerably longer
   in practice.
2. **Node ephemeral storage.** Cached layers count against the node's disk. Kubernetes evicts pods
   when the image filesystem is nearly full.
3. **Registry and network cost.** Every node pulls independently unless a pull-through cache is
   used.

A distroless runtime of 180 MB and a slim Python runtime of 1.2 GB differ by a factor of six in all
three.

### 3.4 Trust is a chain, not a property of the image

An image is trustworthy if you can establish four things: what is inside it (SBOM), how it was
built (provenance), who approved it (signature), and that it has not changed since approval
(digest). None of these is provided by building the image. Each requires a separate step and a
verifier that checks it at deployment time.

```text
  source ──▶ build ──▶ image digest ──▶ registry ──▶ admission ──▶ node
              │             │              │             │
              ├─ SBOM ──────┤              │             │
              ├─ provenance ┤              │             │
              └─ signature ─┴──────────────┴─────────────┘
                                      verified here
```

An attestation that is generated but never verified is documentation. A signature that admission
does not check is a sticker.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| OCI | Open Container Initiative, which publishes the image and runtime specifications | Docker-specific format | Spec version |
| Image | Content-addressed filesystem layers plus configuration | The running process | Digest |
| Container | A running instance of an image with its own namespaces | The image | Process |
| Layer | A filesystem diff produced by one build instruction | A package | Digest |
| Tag | Mutable name pointing to an image | A version | String |
| Digest | Content hash of a manifest | A tag | `sha256:` hex |
| Manifest | Document listing layer and config digests | The Dockerfile | JSON media type |
| Manifest list | A multi-architecture index pointing to per-platform manifests | A manifest | JSON media type |
| Base image | The image a build starts from | The operating system | Digest |
| Multi-stage build | A build with separate stages for compilation and runtime | A large Dockerfile | Stage count |
| Distroless | A runtime image with the language runtime and no shell or package manager | A minimal Linux distribution | Image family |
| Lockfile | Resolved dependency list with exact versions and hashes | A requirements file with ranges | File hash |
| SBOM | Software bill of materials: machine-readable component inventory | A list of top-level packages | SPDX or CycloneDX |
| Provenance | Record of how an artifact was built | A build log | in-toto attestation |
| Attestation | Signed statement about an artifact | A signature over the image | DSSE envelope |
| Reproducible build | Build that can be recreated from recorded inputs | A build that succeeds twice | Digest or tolerance |
| Entrypoint | Executable that runs when the container starts | The command line | Exec form array |
| Signal handling | Process behaviour on SIGTERM and SIGINT | Automatic | Exit code |
| Graceful shutdown | Draining work before exit | A sleep | Seconds |
| Readiness | Whether the container can serve traffic now | Whether it is alive | Boolean |
| Ephemeral storage | Node disk used by layers, logs, and writable layers | Persistent volume | GiB |
| Image pull policy | When kubelet fetches the image | A security control | Enum |
| Image pull secret | Credential used to authenticate to a registry | A service account | Secret reference |
| Registry | Store of images addressed by repository and digest | A tag namespace | Repository URI |
| Immutable tag | Registry policy preventing a tag from being moved | A convention | Registry policy |
| Scan on push | Vulnerability scan triggered by an image push | Continuous scanning | Registry setting |
| SBOM attestation | SBOM attached to the image in the registry | A separate file on a wiki | Attestation manifest |
| Admission controller | Component that accepts or rejects objects at the API server | A scanner | Webhook |
| Supply chain level | SLSA maturity level for build provenance | A vendor claim | SLSA Build level |

Three distinctions carry the operational weight.

**An image is not a virtual machine.** It has no init system, no cron daemon, and no package
manager in the runtime stage. A container that needs to start three processes needs a supervisor or
a redesign.

**A tag is not an identity.** The lab's `ghcr.io/example/biomed-risk:0.1.0` can be repointed by
anyone with push rights. The digest cannot.

**A vulnerability scan is not a risk assessment.** A finding with a critical severity in a package
that is never loaded may be less urgent than a medium finding in the request parser. Scanners
report presence, not exploitability.

## 5. The mechanics, step by step

The procedure builds a runtime image and the evidence needed to deploy it. Steps 1 to 4 establish
the build. Steps 5 to 7 make the runtime safe. Steps 8 to 12 attach identity and enforce it.

### Step 1 — Declare the runtime contract

**Input.** The service code, its model or data dependencies, and the platform it will run on.

**Output.** A written contract: language runtime version, CPU architecture, system libraries,
listening port, startup time, shutdown grace period, writable paths, and outbound network
dependencies.

**Rationale.** The contract is the specification the image must satisfy and the checklist the
Deployment must match. Most crash loops in Kubernetes are contract violations, not code defects: a
process that binds `127.0.0.1` instead of `0.0.0.0`, a startup that exceeds the probe budget, a
write to a read-only path.

**Branch point.** If the service loads a model at startup, state the model's size and load time. A
1.5 GB model that takes 40 seconds to memory-map needs a startup probe with a budget longer than
the liveness probe's.

### Step 2 — Choose a base image strategy

**Input.** The runtime contract and the team's debugging requirements.

**Output.** A base image family and a pinning strategy.

**Rationale.** The base image determines size, attack surface, and the availability of a shell for
debugging. There is no universally correct choice; there is a correct choice per service.

**Decision table.** More than two outcomes, so the choice is tabulated.

| Strategy | Typical size | Shell | Package manager | Debuggability | Best fit |
|---|---|---|---|---|---|
| Distroless | 20–180 MB | No | No | Poor, needs ephemeral debug container | Production services with mature observability |
| Alpine-based | 50–200 MB | Yes (busybox) | apk | Moderate | Small services, simple native dependencies |
| Slim Debian | 120–400 MB | Yes | apt | Good | Python and Node services with native wheels |
| Full distribution | 700 MB–1.5 GB | Yes | apt or dnf | Best | Build stages, not runtime |

The size ranges depend on the language runtime and are illustrative; measure the actual image with
`docker image inspect`.

**Branch point.** If the service needs a shell for an entrypoint script, distroless is not viable
without rewriting the entrypoint as a binary. If the service is a regulated device component whose
investigation procedure depends on shell access, a slim distribution may be the defensible choice
even at a size cost.

### Step 3 — Pin the base image by digest

**Input.** The chosen base image and a registry mirror policy.

**Output.** A `FROM` line with a digest, recorded in the build metadata.

**Rationale.** A tag can move. A digest cannot. Pinning by digest makes the base layer immutable and
makes the build reproducible while the registry retains the manifest.

**Branch point.** If the base image is rebuilt upstream with a security patch, the digest changes
and the team must update it deliberately. Pinning without a patch cadence trades reproducibility
for unpatched vulnerabilities. The correct process is a scheduled rebuild that updates the digest,
runs the tests, and records the new SBOM.

**Edge case.** Some registries garbage-collect old digests. Keep an internal mirror of approved base
images, and record the mirror digest rather than the upstream one.

### Step 4 — Separate build and runtime stages

**Input.** The lockfile, the source, and the runtime contract.

**Output.** A multi-stage Dockerfile where the runtime stage contains only the application and its
runtime dependencies.

**Rationale.** Compilers, headers, test frameworks, and package caches belong in the build stage.
Copying them into the runtime inflates the image, enlarges the SBOM, and adds components that must
be scanned and patched.

**Branch point.** If a native extension links against a shared library at runtime that is present
only in the build image, install that library explicitly in the runtime stage and record it. Do not
copy the whole build filesystem as a shortcut.

**Measure, do not estimate.** The compressed size is what the node pulls; the uncompressed size is
what the node stores. Both matter, and they differ by roughly a factor of two to three. Use
`docker image inspect --format '{{.Size}}'` for the uncompressed size and the registry's reported
manifest size for the compressed size. A build that is optimized against the wrong number will be
optimized against the wrong constraint.

### Step 5 — Lock dependencies with hashes

**Input.** Direct dependencies and the target Python version.

**Output.** A lockfile with exact versions, transitive dependencies, and hashes, installed with
`--require-hashes` or the equivalent.

**Rationale.** Ranges resolve differently over time. Hashes prevent a substituted package from
being installed silently. The lock must be generated on the target platform; a lock generated on
macOS may include wheels that do not exist for Linux.

**Edge case.** A transitive dependency without a Linux wheel forces a source build, which requires
build tools. That is a reason for a multi-stage build, not a reason to loosen the lock.

**What the lock does not cover.** System packages installed with `apt-get` are not in the Python
lockfile. Pin them by version or, better, rebuild from a base image digest that already contains the
approved system packages. A lockfile that pins Python packages and leaves the operating system
floating is half a lock.

### Step 6 — Create a non-root user and set ownership

**Input.** The runtime stage and the writable paths from the contract.

**Output.** A numeric user identifier, file ownership that user needs, and a `USER` instruction.

**Rationale.** Root in a container is root on the node if a container escape occurs, and it is
root for any file written to a mounted volume. Kubernetes can enforce `runAsNonRoot`, but the image
should not depend on the cluster to get this right.

Use a numeric identifier rather than a name. Kubernetes cannot verify that a named user is
non-root, and the `runAsNonRoot` check fails closed on a name it cannot resolve.

**Branch point.** If the application writes to a mounted volume owned by another identifier, a
fixed numeric user causes permission errors. Set the volume's `fsGroup` in the pod security context,
or use an init container to fix ownership.

### Step 7 — Make the root filesystem read-only compatible

**Input.** The application's write paths.

**Output.** An image that runs with `readOnlyRootFilesystem: true` and mounts `emptyDir` or a
persistent volume at each writable path.

**Rationale.** A read-only root filesystem prevents an attacker from modifying binaries and makes
the container's state explicit. It also catches accidental writes to the image at development time
rather than in production.

**Edge case.** Python writes bytecode caches, matplotlib writes a font cache, and many libraries
write to a home directory. Set `PYTHONDONTWRITEBYTECODE=1`, `MPLCONFIGDIR=/tmp/matplotlib`, and a
writable `HOME` under `/tmp`, or the container will fail in ways that look unrelated.

### Step 8 — Keep secrets out of every layer

**Input.** The build arguments, environment variables, and configuration files.

**Output.** A build that contains no credential, and a runtime that reads secrets from a mounted
file or the environment at start.

**Rationale.** A secret in an `ENV` or `ARG` instruction is recorded in the image history and
remains recoverable even if a later layer deletes the file. Registry access control does not help:
anyone who can pull the image can read it.

**Branch point.** If a build genuinely needs a credential, for example to fetch a private package,
use a build secret mount that is not persisted in a layer. Confirm with `docker history` that no
value appears.

### Step 9 — Build for the target architecture

**Input.** The node instance families in the cluster.

**Output.** A multi-architecture image or one image per architecture, with the correct manifest
list.

**Rationale.** Graviton instances such as `m7g` are arm64; `t3` and `m7i` are x86-64. An image built
on a developer's arm64 laptop and pushed without a manifest list fails on an x86 node with
`no match for platform in manifest`. The lab's default node type is `t3.medium`, which is x86-64;
moving to `m7g` requires arm64 images.

**Edge case.** Native extensions and some Python wheels are architecture-specific. A multi-arch
build must run the dependency installation separately per platform, not copy a wheel across.

### Step 10 — Generate the SBOM and provenance

**Input.** The built image digest.

**Output.** An SBOM in SPDX or CycloneDX format, a provenance attestation, and both attached to the
image in the registry.

**Rationale.** The SBOM answers "what is inside" and the provenance answers "how was it built".
Both are needed for a supply-chain review and for the software bill of materials expected in
medical-device premarket cybersecurity submissions. Attaching them to the digest keeps them
findable.

**Branch point.** If the build platform cannot produce signed provenance, record that fact in the
risk register rather than generating an unsigned file and calling it provenance.

### Step 11 — Scan and apply a severity gate

**Input.** The image and the registry scanning configuration.

**Output.** A scan result, a documented threshold, and a build outcome.

**Rationale.** Scanning without a gate produces a dashboard nobody acts on. The gate must state
which severities block, which require a waiver, and who can approve a waiver.

ECR offers two modes. Basic scanning reports operating-system package vulnerabilities, runs at most
once per image per 24 hours, and supports scan on push or manual. Enhanced scanning, powered by
Amazon Inspector, covers operating-system and programming-language packages and can rescan
continuously as new vulnerabilities are published. The lab's
`image_scanning_configuration { scan_on_push = true }` is basic scanning.

**Edge case.** A blocking gate on a base image with an unpatched critical vulnerability can halt all
delivery. The waiver path must exist before it is needed.

### Step 12 — Tag immutably, record the digest, and verify at admission

**Input.** The scanned image and the deployment manifest.

**Output.** An immutable tag in ECR, a digest recorded in the deployment record, and an admission
policy that accepts only signed images from the approved registry.

**Rationale.** `image_tag_mutability = "IMMUTABLE"` prevents a tag from being repointed after review.
Recording the digest ties the deployment to one artifact. Admission verification closes the loop;
without it, a cluster can run any image a pod spec names.

**Branch point.** If admission control is not available, a policy that forbids `:latest` and
requires a digest in the pod spec is a weak but real substitute. It prevents the most common
mistake, not a determined attacker.

### 5.1 Edge cases that change the answer

**Image pull secret expiry.** A registry credential stored as a Kubernetes secret expires, and new
pods fail with `ErrImagePull` while running pods continue. The failure appears at the worst time,
during a rollout. Prefer node-level identity or a credential helper over a long-lived pull secret.

**Registry quota and rate limits.** Docker Hub applies pull rate limits to anonymous and free
accounts. A cluster that scales up and pulls the same image on twenty nodes can hit them. Mirror
approved images into ECR.

**Layer cache poisoning.** A cache shared across branches can serve a layer built from a different
branch's source. Scope caches per branch, and never share a cache between trust levels.

**Base image CVE published after deployment.** The image was clean at build time. Continuous
scanning exists because this is normal, not exceptional. Without it, the finding is discovered at the
next rebuild, which may be months later.

**Node disk pressure.** Cached layers and container logs consume ephemeral storage. When the
image filesystem crosses the kubelet eviction threshold, pods are evicted. A large image with many
retained versions is a common cause.

**Architecture mismatch after a node group change.** A team migrates from `t3.medium` to `m7g.large`
for cost and finds that half the images have no arm64 manifest. The migration plan must include an
image audit.

**Credential expiry during a long build.** A build that fetches dependencies from a private index
can fail midway when a token expires. The failure is a build error, but the cause is a credential
policy, and the fix belongs in the build platform.

**Signal handling and the preStop hook.** The reference deployment sleeps five seconds in
`preStop` and then sends SIGTERM. If the application ignores SIGTERM, the kubelet kills it after the
grace period and in-flight requests fail. Test the shutdown path, not only the startup path.

**The image and the manifest disagree on the port.** The Dockerfile exposes 8000, the Service
targets a named port `http` that resolves to 8000, and the probe uses the same name. If someone
changes the application to listen on 8080 without changing the manifest, the container starts,
fails readiness, and never receives traffic. The pod is running, the rollout stalls, and the
Deployment reports `MinimumReplicasUnavailable`. Port names are part of the contract.

**A read-only filesystem that is read-only in the wrong place.** Mounting an `emptyDir` at `/tmp`
does not help if the application writes to `/var/tmp`. Both paths must be declared, or the
application must be configured to use one.

### 5.2 A decision table for the runtime security context

| Requirement | Security context setting | Consequence if omitted |
|---|---|---|
| No root | `runAsNonRoot: true`, numeric `runAsUser` | Container runs as root on the node |
| No privilege escalation | `allowPrivilegeEscalation: false` | Setuid binaries can gain privileges |
| Minimal capabilities | `capabilities.drop: ["ALL"]` | Default capability set retained |
| Immutable filesystem | `readOnlyRootFilesystem: true` | Runtime tampering possible |
| Syscall filtering | `seccompProfile.type: RuntimeDefault` | Broader kernel attack surface |
| No API token | `automountServiceAccountToken: false` | Pod can call the Kubernetes API |

The reference deployment in `labs/cloud/k8s/03-deployment.yaml` sets five of these six at the pod or
container level, and `labs/cloud/k8s/09-serviceaccount.yaml` sets the sixth. That combination is a
reasonable production baseline.

### 5.3 Gate order in a build pipeline

The order of checks matters, because each check is cheaper than the next and a later check should
not run on an artifact that an earlier one would reject.

| Order | Gate | Catches | Cost per run | Fail behaviour |
|---:|---|---|---|---|
| 1 | Dockerfile lint | Root user, floating base, secret in `ENV` | Milliseconds | Block build |
| 2 | Dependency resolution with hashes | Tampered or missing package | Seconds | Block build |
| 3 | Build the image | Compilation and packaging errors | Minutes | Block build |
| 4 | Unit and contract tests inside the image | Runtime contract violations | Minutes | Block build |
| 5 | SBOM generation | Missing inventory | Seconds | Block promotion |
| 6 | Vulnerability scan | Known vulnerable components | Minutes | Block on severity, waiver path |
| 7 | Sign and attach attestations | Unapproved artifact | Seconds | Block promotion |
| 8 | Admission verification at deploy | Anything that bypassed the pipeline | Per admission | Reject pod |

Running the scan before the tests wastes minutes on an image that will not ship. Running the lint
after the build wastes a full build. The order above is the common compromise between speed and
completeness.

### 5.4 What the Dockerfile cannot control

The Dockerfile does not control the node's architecture, the registry's retention policy, the
cluster's admission policy, or the pull policy. A team can produce a perfect image and still deploy
it badly. The deployment manifest is the other half of the contract, and
[`03_kubernetes.md`](03_kubernetes.md) covers it.

## 6. Worked example

A risk-scoring service is deployed to a three-node EKS cluster. The image is 1.2 GB compressed. Each
node is `t3.medium` with 2 vCPU, 4 GiB of memory, and a 20 GiB gp3 root volume, the default for an
EKS managed node group. The Deployment requests two replicas with `maxSurge: 1` and
`maxUnavailable: 0`.

### 6.1 Inputs

| Parameter | Value | Source |
|---|---|---|
| Image size, compressed | 1.2 GB | Measured with `docker image inspect` |
| Node type | `t3.medium`, 2 vCPU, 4 GiB | `labs/cloud/terraform/variables.tf` default |
| Node root volume | 20 GiB gp3 | EKS managed node group default |
| Node count | 3 | `node_desired_size = 2`, `node_max_size = 4`; assume 3 after one scale-out |
| Registry bandwidth per node | 500 Mbit/s | Assumed cluster network |
| Effective pull efficiency | 40 percent of link rate | Assumed, includes TLS and registry overhead |
| Replicas | 2 | `labs/cloud/k8s/03-deployment.yaml` |
| Startup time after pull | 20 s | Assumed, model load included |
| Readiness delay | 10 s | `initialDelaySeconds: 5` plus probe period |
| Node price | USD 0.0416/hour | AWS Price List, us-east-2, October 2026 |
| gp3 price | USD 0.08/GB-month | AWS EBS published example rate |

All costs are USD, us-east-2, for 730 hours.

### 6.2 Arithmetic: pull time

Effective per-node bandwidth:

```text
500 Mbit/s ÷ 8 = 62.5 MB/s
62.5 MB/s × 0.40 = 25 MB/s effective
```

Time to pull one image on a node with no cache:

```text
1,200 MB ÷ 25 MB/s = 48 seconds
```

Time to pull two images concurrently on the same node, assuming the link is shared fairly:

```text
2 × 1,200 MB ÷ 25 MB/s = 96 seconds
```

The first pod on a fresh node pays 48 seconds before the container even starts.

### 6.3 Arithmetic: rollout duration

With `maxUnavailable: 0` and `maxSurge: 1`, Kubernetes creates one new pod, waits for it to become
ready, then terminates one old pod, then repeats.

Per new pod on a node that already has the image cached:

```text
startup 20 s + readiness 10 s = 30 s
```

Per new pod on a node without the cache:

```text
pull 48 s + startup 20 s + readiness 10 s = 78 s
```

With three nodes and two replicas, the scheduler may place the new pod on a node that already has
the image from the previous release, in which case the rollout is 60 seconds for two pods. If the
node group was just scaled out, both new pods land on uncached nodes and the rollout is 156
seconds. The difference is not the application; it is the cache.

### 6.4 Arithmetic: node ephemeral storage

Each node caches every image version it has pulled. With five retained versions of a 1.2 GB image:

```text
5 × 1.2 GB = 6.0 GB of image layers
```

Container writable layers and logs add perhaps 1 GB per node. The kubelet's default eviction
threshold triggers when the image filesystem has less than 15 percent free:

```text
20 GiB × 0.15 = 3 GiB free-space threshold
```

With 6 GB of images, 1 GB of logs, and a 20 GiB volume, free space is approximately 13 GiB before
the operating system. That is comfortable. Raise the retained versions to fifteen, which is
plausible on a busy cluster with `revisionHistoryLimit: 5` per Deployment and several Deployments,
and the images alone reach 18 GB. The node is then evicting pods, and the evicted pod is the one
with the lowest quality-of-service class.

### 6.5 Cost of the compute that performs the pulls

Node compute for one month:

```text
3 nodes × USD 0.0416/hour × 730 hours = USD 91.10
```

This is the cost of the nodes, not of the pulls. There is no separate charge for pulling from a
registry in the same region for ECR; cross-region or internet egress would add cost. The point of
the number is that image size does not appear in the bill directly. It appears as eviction, as
rollout duration, and as the number of nodes needed to hold the cache.

### 6.6 Two diagnostics that reveal a misconfiguration

**Diagnostic 1: `ImagePullBackOff` with `no match for platform in manifest`.** The pod event log
shows the registry returned a manifest with no matching platform. This is an architecture mismatch,
not a credential problem. Confirm with:

```bash
kubectl describe pod <pod> -n biomed | sed -n '/Events/,$p'
docker manifest inspect ghcr.io/example/biomed-risk:0.1.0 | grep -i architecture
```

If the node is `t3.medium` (amd64) and the manifest lists only `arm64`, the fix is a multi-arch
build, not a new pull secret.

**Diagnostic 2: `CrashLoopBackOff` with `Read-only file system` in the previous log.** The
container started and then failed on a write. The reference manifest sets
`readOnlyRootFilesystem: true` and mounts `emptyDir` at `/tmp` and a ConfigMap at `/models`. An
application that writes `/app/.cache` or `$HOME/.cache` fails. Confirm with:

```bash
kubectl logs <pod> -n biomed --previous | grep -i "read-only"
kubectl get pod <pod> -n biomed -o jsonpath='{.spec.containers[0].volumeMounts}'
```

The fix is to redirect the cache to `/tmp` through an environment variable, not to disable the
read-only root filesystem.

A related diagnostic is a pod that is `Running` but never `Ready`, with an exec probe timing out.
In a distroless image there is no shell, so an `exec: ["sh", "-c", ...]` probe always fails. The
pod event shows `exec: "sh": executable file not found in $PATH`. The fix is an HTTP or gRPC probe
against the application's own endpoint.

### 6.7 Vary one assumption

Replace the 1.2 GB slim image with a 180 MB distroless image. Pull time becomes:

```text
180 MB ÷ 25 MB/s = 7.2 seconds
```

Rollout on an uncached node becomes:

```text
7.2 s + 20 s + 10 s = 37.2 seconds
```

Five cached versions occupy 0.9 GB instead of 6.0 GB, and the node's free-space headroom rises by
more than 5 GB. The trade-off is that the image has no shell, so a debugging session requires an
ephemeral container with the same image or a debug variant. That is a real operational cost and
belongs in the decision record.

A second variation is worth costing. Suppose the cluster pre-pulls the image with a DaemonSet
instead of relying on the first scheduled pod. The DaemonSet runs on every node, so a scale-out
event still waits for a pull, but a rollout to an already-warm node starts in 30 seconds rather than
78. The DaemonSet costs one pod per node, negligible in CPU, and it converts a per-rollout delay
into a per-node background task. For a service with frequent releases and a tight error budget,
that is usually the better trade.

## 7. Code, query, or calculation

The following is a container build policy linter in standard-library Python. It reads a Dockerfile
and reports findings by severity. It is the kind of check that belongs in CI before the image is
built, because it catches the mistakes that are cheap to fix at the source and expensive to fix
after a push.

```python
"""Container build policy linter. Standard library only.

Reads a Dockerfile from stdin or a path argument and reports policy findings.
Exit code 1 if any finding is severity HIGH.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass


@dataclass
class Finding:
    line: int
    rule: str
    severity: str
    message: str


SECRET_PATTERN = re.compile(
    r"(?i)(password|passwd|secret|token|api[_-]?key|private[_-]?key)\s*[=:]\s*\S+"
)
PINNED_PIP = re.compile(r"^[A-Za-z0-9_.\-]+==[0-9]")


def logical_lines(text: str):
    """Yield (starting_line_number, joined_instruction) handling backslash continuations."""
    buffer: list[str] = []
    start = 0
    for number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not buffer:
            start = number
        if stripped.endswith("\\"):
            buffer.append(stripped[:-1].rstrip())
            continue
        buffer.append(stripped)
        joined = " ".join(part for part in buffer if part)
        if joined:
            yield start, joined
        buffer = []


def lint(text: str) -> list[Finding]:
    findings: list[Finding] = []
    instructions = list(logical_lines(text))
    from_lines = [(n, i) for n, i in instructions if i.upper().startswith("FROM ")]
    stages = len(from_lines)
    has_user = False
    user_value = ""
    workdir = ""

    for number, inst in instructions:
        upper = inst.upper()

        if upper.startswith("FROM "):
            image = inst.split()[1]
            if image.lower() == "scratch":
                continue
            if "@sha256:" not in image:
                findings.append(Finding(number, "BASE-001", "HIGH",
                    f"base image '{image}' is not pinned by digest"))
            if image.endswith(":latest") or ":" not in image.rsplit("/", 1)[-1]:
                findings.append(Finding(number, "BASE-002", "HIGH",
                    f"base image '{image}' uses a floating tag"))

        if upper.startswith("USER "):
            has_user = True
            user_value = inst.split(maxsplit=1)[1].strip()

        if upper.startswith("WORKDIR "):
            workdir = inst.split(maxsplit=1)[1].strip()

        if upper.startswith("ADD "):
            findings.append(Finding(number, "LAYER-001", "MEDIUM",
                "ADD can fetch remote content and auto-extract; prefer COPY"))

        if upper.startswith(("ENV ", "ARG ")):
            if SECRET_PATTERN.search(inst):
                findings.append(Finding(number, "SEC-001", "HIGH",
                    "possible secret in ENV or ARG; build args are visible in image history"))

        if upper.startswith("RUN ") and "pip install" in inst:
            packages = inst.split("pip install", 1)[1].split()
            skip_next = False
            for package in packages:
                if skip_next:
                    skip_next = False
                    continue
                if package in {"-r", "--requirement", "-c", "--constraint"}:
                    skip_next = True
                    continue
                if package.startswith("-"):
                    continue
                if not PINNED_PIP.match(package):
                    findings.append(Finding(number, "DEP-001", "MEDIUM",
                        f"package '{package}' is not pinned with =="))
                    break

        if upper.startswith("RUN ") and "apt-get install" in inst:
            if "--no-install-recommends" not in inst:
                findings.append(Finding(number, "DEP-002", "LOW",
                    "apt-get install without --no-install-recommends"))
            if "rm -rf /var/lib/apt/lists" not in inst:
                findings.append(Finding(number, "DEP-003", "LOW",
                    "apt lists not cleaned in the same layer"))

        if upper.startswith("RUN ") and re.search(r"chmod\s+777", inst):
            findings.append(Finding(number, "SEC-002", "HIGH",
                "world-writable permissions granted"))

        if upper.startswith("EXPOSE ") and "22" in inst.split()[1:]:
            findings.append(Finding(number, "SEC-003", "HIGH",
                "SSH port exposed; containers should not run sshd"))

    if stages == 1:
        findings.append(Finding(0, "BUILD-001", "MEDIUM",
            "single-stage build; build tooling will ship in the runtime image"))
    if not has_user:
        findings.append(Finding(0, "SEC-004", "HIGH",
            "no USER instruction; the container runs as root by default"))
    elif user_value in {"root", "0", "0:0"}:
        findings.append(Finding(0, "SEC-005", "HIGH",
            f"USER is '{user_value}', which is root"))
    if workdir in {"", "/"}:
        findings.append(Finding(0, "BUILD-002", "LOW",
            "no WORKDIR; files land in the image root"))

    return findings


def main() -> int:
    if len(sys.argv) > 1:
        text = open(sys.argv[1], encoding="utf-8").read()
    else:
        text = sys.stdin.read()
    findings = lint(text)
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    for finding in sorted(findings, key=lambda f: (order[f.severity], f.line, f.rule)):
        location = f"line {finding.line}" if finding.line else "file"
        print(f"{finding.severity:6} {finding.rule:9} {location:9} {finding.message}")
    high = sum(1 for f in findings if f.severity == "HIGH")
    print(f"\n{len(findings)} findings, {high} high severity")
    return 1 if high else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run against a deliberately poor Dockerfile:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
ENV DATABASE_PASSWORD=hunter2
RUN apt-get update && apt-get install -y gcc libpq-dev
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000 22
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

Actual output:

```text
HIGH   SEC-004   file      no USER instruction; the container runs as root by default
HIGH   BASE-001  line 1    base image 'python:3.12-slim' is not pinned by digest
HIGH   SEC-001   line 3    possible secret in ENV or ARG; build args are visible in image history
HIGH   SEC-003   line 8    SSH port exposed; containers should not run sshd
MEDIUM BUILD-001 file      single-stage build; build tooling will ship in the runtime image
LOW    DEP-002   line 4    apt-get install without --no-install-recommends
LOW    DEP-003   line 4    apt lists not cleaned in the same layer

7 findings, 4 high severity
```

The linter exits 1, which in a CI pipeline blocks the build.

A hardened version of the same image, shown here for comparison. The digest is illustrative and
must be replaced with a real one.

```dockerfile
FROM python:3.12.7-slim@sha256:1111111111111111111111111111111111111111111111111111111111111111 AS build
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12.7-slim@sha256:1111111111111111111111111111111111111111111111111111111111111111
WORKDIR /app
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY app ./app
RUN useradd --uid 10001 --no-create-home appuser
USER 10001
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

The linter reports zero findings for this version. Note that the linter does not check for a
`HEALTHCHECK`, does not verify that the digest exists, and does not inspect the resulting image. It
is a source-level policy check, not a substitute for scanning.

The following Kubernetes admission policy is expressed as a YAML document for illustration. It is
validated by parsing, not applied to a cluster.

```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: require-image-digest
spec:
  failurePolicy: Fail
  matchConstraints:
    resourceRules:
      - apiGroups: ["apps"]
        apiVersions: ["v1"]
        operations: ["CREATE", "UPDATE"]
        resources: ["deployments"]
  validations:
    - expression: >-
        object.spec.template.spec.containers.all(c,
          c.image.contains('@sha256:'))
      message: "container images must be referenced by digest"
```

## 8. Data, standards, and regulatory context

### 8.1 The OCI specifications

The Open Container Initiative publishes the image specification, the runtime specification, and the
distribution specification. They define what a layer is, how a manifest lists layers, how a runtime
executes a container, and how a registry serves blobs. The relevant consequence for architects is
that image identity is a content hash defined by the specification, not a vendor convention. A
digest is portable across conforming registries.

What the specification does not define: signing, scanning, admission policy, or provenance. Those
are supplied by other projects and by the registry.

### 8.2 HIPAA Security Rule

The technical safeguards at 45 CFR 164.312 constrain containers indirectly:

| Clause | Constraint | Container consequence |
|---|---|---|
| 164.312(a)(1) Access control | Limit access to electronic PHI | Non-root user, no shared credentials in the image |
| 164.312(a)(2)(i) Unique user identification | Identify each user | Service identity, not a shared account baked into an image |
| 164.312(b) Audit controls | Record access and activity | No PHI in image logs; ship logs to an audited sink |
| 164.312(c)(1) Integrity | Protect against improper alteration | Immutable tags, digest pinning, read-only root filesystem |
| 164.312(e)(1) Transmission security | Protect PHI in transit | TLS between services; the lab does not provide this |

Encryption and decryption at 164.312(a)(2)(iv) is an addressable specification. Addressable means
the covered entity must assess it, implement it if reasonable and appropriate, and document the
decision otherwise. It does not mean optional.

### 8.3 Medical device software

IEC 62304 defines software lifecycle processes for medical device software, including configuration
identification and change control. A container image used in a device must have its configuration
items identified; the digest, SBOM, and provenance are the natural configuration records.

Section 524B of the US Federal Food, Drug, and Cosmetic Act, added by the Consolidated
Appropriations Act of 2023, requires premarket submissions for cyber devices to include a software
bill of materials. The FDA's guidance on cybersecurity in premarket submissions describes the
expectation. A team that cannot produce an SBOM for a shipped image cannot meet that requirement.

### 8.4 Supply-chain frameworks

The NIST Secure Software Development Framework, SP 800-218, describes practices for producing
software with evidence of its provenance. SLSA defines build levels from L1 (provenance exists) to
L3 (provenance is unforgeable and the build is isolated). Both are voluntary frameworks in most
jurisdictions; both are increasingly required by procurement.

What none of them do: tell you which base image to use, how large an image may be, or which
vulnerabilities to accept. Those remain engineering and risk decisions.

### 8.5 What the standards leave open

The standards do not require containers. A virtual machine deployment can satisfy every clause
above. They also do not require a specific registry, scanner, or signing tool. The architecture
choice is yours; the evidence obligations are not.

> **Note:** The reference stack uses a placeholder image hosted outside the Terraform-managed
> registry and does not sign or verify it. It demonstrates the security context and the manifest
> structure, not a compliant supply chain.

## 9. Industry practice and case evidence

### 9.1 Documented practice

**Scanning modes are differentiated in public documentation.** Amazon ECR basic scanning covers
operating-system packages and supports scan on push or manual, with one scan per image per 24 hours
and a limit of 100,000 images per 24 hours per registry. Enhanced scanning uses Amazon Inspector to
cover both operating-system and programming-language packages, and supports continuous rescanning
as new vulnerabilities are published. The cost model differs: basic scanning is billed through ECR,
enhanced through Inspector.

**Immutable tags are a registry feature, not a convention.** ECR's `IMMUTABLE` setting rejects a
push that would move an existing tag. The lab's storage module sets it.

**Multi-architecture images are mainstream.** The OCI image index, formerly a manifest list,
allows one tag to resolve to different manifests per platform. This is how a single deployment can
run on both Graviton and x86 nodes.

### 9.2 Case evidence

Two public supply-chain incidents illustrate why the build stage matters:

1. **Log4Shell (CVE-2021-44228, December 2021).** A vulnerability in a widely used Java logging
   library required organisations to find every image containing it. Teams with an SBOM could
   query; teams without one rebuilt everything and hoped. The lesson is inventory, not patching
   speed.
2. **xz-utils backdoor (CVE-2024-3094, disclosed March 2024).** A malicious contributor introduced
   a backdoor into a compression library used by many Linux distributions. The affected versions
   never reached most stable release channels, but the incident demonstrated that dependency
   provenance and reproducible builds are defensive controls, not compliance theatre.

Neither incident is specific to healthcare. Both affected healthcare software through shared base
images and language packages.

### 9.3 Inference

The following are inferences, labelled as such:

- Most clinical teams that adopt distroless do so after an incident or a security review, not at
  first deployment, because the loss of a shell is felt immediately in debugging.
- Image size is usually controlled by removing build tooling and caches, not by changing the base
  distribution. A slim base with a multi-stage build is often within 20 percent of a distroless
  image.
- The most common cause of stale base images in production is not negligence but the absence of a
  scheduled rebuild. Nobody owns the digest update.

### 9.4 What is not public

There are few published, detailed post-mortems from health systems describing container-related
outages. Vendor case studies describe success. Treat any claim that a specific hospital avoided an
outage through containers as marketing unless a named report supports it.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Root user in the container | Security review finding; `runAsNonRoot` rejection at admission | Default user is root | Container escape grants node root | Numeric non-root `USER`, pod security context |
| Floating base tag | Builds differ without a commit | Tag repointed upstream | Unreproducible artifact | Pin by digest, scheduled rebuild |
| Secret in a layer | `docker history` shows a value | `ENV` or `ARG` persisted | Credential leak to anyone with pull rights | Runtime secret mount, build secret |
| Single-stage build | Image above 1 GB | Compilers and caches shipped | Slow pulls, larger attack surface | Multi-stage build |
| Missing architecture | `no match for platform` | amd64 image on arm64 node | Pods never start | Multi-arch build and manifest audit |
| No read-only root | Container writes to image path | Writable layer | Runtime tampering, hidden state | Read-only root plus explicit mounts |
| Cache redirected nowhere | `Read-only file system` in previous log | Library writes to `$HOME` | Crash loop | `HOME`, `MPLCONFIGDIR`, bytecode settings |
| No SBOM | Cannot answer "which images contain library X" | No component inventory | Slow incident response | Generate and attest SBOM |
| Scan without a gate | Findings dashboard grows | No enforcement | Known critical ships | Severity gate with waiver process |
| Mutable tag in production | Same tag, different digest | Tag repointed | Rollback deploys the wrong code | Immutable tags, digest in manifest |
| Ignored SIGTERM | 5xx spike during rollout | Process killed after grace period | Dropped requests | Signal handler, `preStop` drain |
| Image pull secret expiry | `ErrImagePull` only on new pods | Static credential expires | Rollout failure | Node identity, short-lived credentials |
| Unbounded image retention | Node disk pressure evictions | Layers accumulate | Pod eviction under load | Image garbage collection, registry lifecycle |
| Probe against a shell | Exec probe fails in distroless | No `/bin/sh` | Pod never ready | HTTP or gRPC probes |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| 40 percent of link rate is achievable for pulls | Pull-time arithmetic | Congested node, small images with high overhead | Measure a real pull with `crictl` timings |
| Node root volume is 20 GiB | Ephemeral storage arithmetic | Custom launch template with larger volume | Inspect the node group's block device mapping |
| Startup takes 20 seconds | Rollout arithmetic | Large model load, slow imports | Measure container start to ready |
| The registry is reachable from the cluster | Deployment | Private endpoint misconfiguration, egress policy | Pull from a debug pod |
| The base image digest remains available | Reproducibility | Registry garbage collection | Rebuild from the recorded digest quarterly |
| The application tolerates a read-only root | Security context | Writes to image paths | Run with the flag in CI |
| The scanner's database is current | Vulnerability gate | Offline scanner, stale feeds | Check the scan timestamp |
| A single image serves all architectures | Deployment simplicity | Native extensions without arm64 wheels | `docker manifest inspect` |
| Registry storage cost is not material | Cost model | Thousands of retained versions | Query registry storage metrics |
| The team can debug without a shell | Distroless choice | Incident requires interactive inspection | Run an ephemeral debug container drill |

## 12. Exercises

### 12.1 Identify image identity

**Deliverable.** For the image reference `ghcr.io/example/biomed-risk:0.1.0`, state the repository,
the tag, the registry, and what must be added to make it an immutable reference. Then explain what
happens to a running pod if the tag is repointed and the pod restarts.

**Acceptance criteria.** Correctly distinguishes tag from digest. States that a running pod is not
affected until it restarts, and that a restart may then pull different code.

**Time.** 20 minutes.

### 12.2 Write a hardened Dockerfile

**Deliverable.** A multi-stage Dockerfile for a Python FastAPI service that satisfies every rule in
the Section 7 linter, runs as a numeric non-root user, and works with a read-only root filesystem.

**Acceptance criteria.** Linter reports zero findings. The container starts with
`--read-only --tmpfs /tmp`. The image is under 400 MB.

**Time.** 60 minutes.

### 12.3 Compute pull and rollout time

**Deliverable.** A calculation of pull time and rollout duration for a 900 MB image, a 1 Gbit/s
node link at 50 percent efficiency, 4 replicas, `maxSurge: 1`, `maxUnavailable: 0`, and a 25-second
startup.

**Acceptance criteria.** Shows effective bandwidth, per-image time, and the number of sequential
pod replacements. States the assumption about cache hits.

**Time.** 30 minutes.

### 12.4 Generate an SBOM

**Deliverable.** An SBOM for a small image in SPDX or CycloneDX JSON, plus a one-paragraph statement
of what it proves and what it does not.

**Acceptance criteria.** SBOM parses as JSON and lists at least the base image packages. The
statement distinguishes inventory from vulnerability status.

**Time.** 45 minutes.

### 12.5 Configure scanning

**Deliverable.** A written comparison of ECR basic and enhanced scanning for a team that pushes 40
images per day, with a recommendation and a cost consideration.

**Acceptance criteria.** States coverage difference, rescan behaviour, and which AWS service bills
each mode. Recommendation follows from the team's release cadence.

**Time.** 30 minutes.

### 12.6 Choose a base image

**Deliverable.** A decision memo choosing between distroless, Alpine, and slim Debian for two
services: a public-facing API and an internal batch job that is debugged interactively twice a
month.

**Acceptance criteria.** Uses the Section 5 decision table. Names the debugging consequence for each
choice. Does not recommend the same base for both without justification.

**Time.** 40 minutes.

### 12.7 Diagnose two failures

**Deliverable.** For each of the two diagnostics in Section 6, write the commands you would run,
the evidence you would look for, and the fix. Do not run them against a live cluster; write the
procedure.

**Acceptance criteria.** Commands are correct for Kubernetes 1.34 and later. The fixes address the
cause, not the symptom.

**Time.** 30 minutes.

### 12.8 Design the supply chain

**Deliverable.** A diagram and a one-page description of a build pipeline from commit to admission,
including SBOM generation, signing, scanning gate, digest recording, and admission verification.

**Acceptance criteria.** Every artifact is tied to a digest. The verifier is named. The waiver path
for a blocked vulnerability is described.

**Time.** 75 minutes.

## 13. Self-check answers

### 13.1 Image identity

Repository: `example/biomed-risk` under the `ghcr.io` registry. Tag: `0.1.0`. To make it immutable,
reference `ghcr.io/example/biomed-risk@sha256:<digest>`, and enable tag immutability at the registry
if it supports it. A running pod keeps its local image until the container restarts. On restart,
the kubelet resolves the tag again if `imagePullPolicy` allows, which can pull different code.

### 13.2 Linter expectations

A correct Dockerfile will have two `FROM` lines with digests, a `USER` with a numeric identifier
greater than zero, a `WORKDIR` that is not `/`, no `ENV` containing a secret-like key with a value,
and pinned packages. If the linter reports `DEP-001` on a `-r requirements.txt` line, that is a
false positive in an older version of the linter; the version in Section 7 handles it.

### 13.3 Pull and rollout

Effective bandwidth: 1 Gbit/s ÷ 8 = 125 MB/s, times 0.5 = 62.5 MB/s. One image: 900 ÷ 62.5 = 14.4
seconds. With four replicas, `maxSurge: 1` and `maxUnavailable: 0`, Kubernetes replaces one pod at
a time, so there are four sequential new pods. On an uncached node each takes 14.4 + 25 = 39.4
seconds. Total approximately 158 seconds, plus scheduling and readiness jitter. If all nodes have
the image cached from a previous release, the pull term is zero and the rollout is approximately
100 seconds.

### 13.4 SBOM statement

The SBOM proves which components and versions are present in the image at the time it was
generated. It does not prove that the image is free of vulnerabilities, that the components are
unmodified, or that the build was honest. Those require scanning, signature verification, and
provenance respectively.

### 13.5 Scanning recommendation

At 40 images per day, basic scanning's once-per-24-hours-per-image limit is not binding, because
each image is scanned once on push. The coverage difference is decisive: basic scanning misses
language-package vulnerabilities, which are the majority of findings in a Python service. Enhanced
scanning is the recommendation, with continuous rescanning enabled for images in production. The
cost moves from ECR to Inspector and must be budgeted.

### 13.6 Base image memo

The public API should use distroless: the attack surface is exposed to the internet, the service
should have mature observability, and debugging can use ephemeral containers. The internal batch
job, debugged interactively twice a month, should use slim Debian: the operational cost of losing a
shell exceeds the size and attack-surface benefit for an internal workload. Recommending distroless
for both without addressing the debugging workflow is the common weak answer.

### 13.7 Diagnostic procedures

For the architecture mismatch: `kubectl describe pod` shows `Failed to pull image` with a platform
message; `docker manifest inspect` shows the available architectures; the fix is a multi-arch build.
For the read-only failure: `kubectl logs --previous` shows `Read-only file system`; inspect
`volumeMounts`; the fix is to redirect the cache to `/tmp` with an environment variable.

### 13.8 Supply chain rubric

A strong design has: a build that produces a digest, an SBOM attestation attached to that digest, a
signature over the digest by a workload identity, a scan gate before the push is promoted, and an
admission policy that verifies the signature and the registry. A weak design lists the tools
without saying what is verified or where.

## 14. Interview questions

**1. What is the difference between an image tag and an image digest?**
A strong answer explains that a tag is a mutable pointer and a digest is a content hash, and that
production manifests should use the digest. A weak answer treats them as two names for a version.

**2. Why is a multi-stage build better than a single-stage build?**
A strong answer separates build-time and runtime dependencies, and connects image size to pull time,
attack surface, and SBOM contents. A weak answer says "it is smaller".

**3. How do you keep secrets out of an image?**
A strong answer explains that `ENV` and `ARG` values persist in history, that deleting a file in a
later layer does not remove it, and that runtime mounts or a secret manager are the fix. A weak
answer says "use a `.dockerignore`".

**4. What does a read-only root filesystem break?**
A strong answer names bytecode caches, font caches, home directories, and temporary files, and
describes redirecting them to a mounted `emptyDir`. A weak answer says "nothing, it is best
practice".

**5. What is the difference between basic and enhanced ECR scanning?**
A strong answer covers operating-system versus language packages, rescan frequency, and which
service bills each. A weak answer says one is better.

**6. Why does an image that works locally fail in the cluster with a platform error?**
A strong answer explains manifest lists, node architecture, and native dependencies. A weak answer
blames the registry.

**7. What does an SBOM prove?**
A strong answer says it is an inventory at a point in time and does not prove absence of
vulnerabilities or build integrity. A weak answer says it proves the image is secure.

**8. How does image size affect reliability?**
A strong answer connects size to pull time, rollout duration, node ephemeral storage, eviction, and
scale-out latency. A weak answer mentions only storage cost.

**9. Where should signature verification happen?**
A strong answer says at admission, with the verifier named, and notes that verification at build
time only is insufficient. A weak answer says "in the registry".

**10. How would you handle a critical vulnerability in a base image on a Friday?**
A strong answer describes the assessment (is the vulnerable path reachable), the waiver or patch
decision, the rebuild path, and the communication. A weak answer says "rebuild and redeploy" without
addressing the gate or the risk acceptance.

## 15. Key terms

- **Admission controller** — API server component that accepts or rejects objects.
- **Attestation** — Signed statement about an artifact.
- **Base image** — Image a build starts from.
- **BuildKit** — Container build engine used by modern Docker and CI systems.
- **CycloneDX** — SBOM format.
- **Digest** — Content hash identifying an image.
- **Distroless** — Runtime image without a shell or package manager.
- **ECR** — Elastic Container Registry.
- **Entrypoint** — Executable started when the container runs.
- **Ephemeral storage** — Node disk consumed by layers, logs, and writable layers.
- **Image** — Content-addressed layers plus configuration.
- **Immutable tag** — Registry policy preventing tag movement.
- **Layer** — Filesystem diff from one build instruction.
- **Lockfile** — Resolved dependency list with hashes.
- **Manifest** — Document listing layer and config digests.
- **Manifest list** — Multi-architecture index.
- **Multi-stage build** — Build with separate compilation and runtime stages.
- **OCI** — Open Container Initiative.
- **Provenance** — Record of how an artifact was built.
- **Read-only root filesystem** — Runtime mount policy preventing writes to the image.
- **Reproducible build** — Build recreatable from recorded inputs.
- **SBOM** — Software bill of materials.
- **Scan on push** — Vulnerability scan triggered by an image push.
- **Seccomp** — System call filtering.
- **SLSA** — Supply-chain levels for software artifacts.
- **SPDX** — SBOM format.
- **Supply chain** — The set of inputs and processes that produce an artifact.

## 16. Sources

All sources accessed 26 October 2026.

1. Open Container Initiative. *Image Specification*.
   https://github.com/opencontainers/image-spec
2. Kubernetes. *Images*.
   https://kubernetes.io/docs/concepts/containers/images/
3. Kubernetes. *Configure a Security Context for a Pod or Container*.
   https://kubernetes.io/docs/tasks/configure-pod-container/security-context/
4. AWS. *Scan images for software vulnerabilities in Amazon ECR*.
   https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-scanning.html
5. AWS. *Scanning Amazon ECR container images with Amazon Inspector*.
   https://docs.aws.amazon.com/inspector/latest/user/scanning-ecr.html
6. HHS. *HIPAA Security Rule*.
   https://www.hhs.gov/hipaa/for-professionals/security/index.html
7. Cornell Legal Information Institute. *45 CFR 164.312*.
   https://www.law.cornell.edu/cfr/text/45/164.312
8. NIST. *SP 800-218: Secure Software Development Framework*.
   https://csrc.nist.gov/pubs/sp/800/218/final
9. SLSA. *Supply-chain Levels for Software Artifacts*.
   https://slsa.dev/spec/v1.0/
10. FDA. *Cybersecurity in Medical Devices: Quality System Considerations and Content of Premarket
    Submissions*. https://www.fda.gov/regulatory-information/search-fda-guidance-documents
11. AWS. *EC2 On-Demand pricing*, US East (Ohio). https://aws.amazon.com/ec2/pricing/on-demand/
12. AWS. *Amazon EBS pricing*. https://aws.amazon.com/ebs/pricing/
13. Docker. *Multi-stage builds*. https://docs.docker.com/build/building/multi-stage/
14. Red Hat. *What is a distroless container image*. https://www.redhat.com/en/topics/cloud-native-apps/what-is-a-distroless-container

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`01_architecture-patterns.md`](01_architecture-patterns.md)
- Next topic: [`03_kubernetes.md`](03_kubernetes.md)
- Related topic: [`07_security.md`](07_security.md)
- Survey file: [`../../course/26_cloud_streaming_production_labs.md`](../../course/26_cloud_streaming_production_labs.md)
- Lab root: [`../../labs/`](../../labs/)
- Reference stack: [`../../labs/cloud/`](../../labs/cloud/)
- Deployment manifest: [`../../labs/cloud/k8s/03-deployment.yaml`](../../labs/cloud/k8s/03-deployment.yaml)
- Registry module: [`../../labs/cloud/terraform/modules/storage/main.tf`](../../labs/cloud/terraform/modules/storage/main.tf)
