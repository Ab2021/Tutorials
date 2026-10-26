# Packaging and Reproducibility

> Module 15 — Deployment and Machine Learning Operations (MLOps)
> Topic 03 of 10
> Estimated study time: 3 hours
> Prerequisites: 01
> Version 2026.3

## 1. Why this topic exists

A model artifact is not a file. It is a claim that a specific set of inputs, transformed by a
specific set of code and dependencies, produces a specific output. If any element of that set can
change without being recorded, the claim cannot be verified, and a deployment cannot be defended
in an incident review, an audit, or a recall.

Reproducibility is usually taught as a research virtue: can another scientist obtain the same
result. In production it is an operational requirement with three concrete uses. It lets an
engineer rebuild the exact artifact that produced a suspect prediction. It lets a regulator or
auditor confirm what software was running on a given date. It lets a team roll back to a known
state without guessing which combination of image, model, and configuration was live.

The failure mode is quiet. A `requirements.txt` with `>=` ranges installs a different version of a
numerical library six months later. A base image tagged `python:3.12-slim` is rebuilt upstream with
a new operating system patch. A model file is mounted from a network share rather than baked into
the image, so the image digest no longer identifies the model. A configuration value is stored in a
Kubernetes ConfigMap that is edited in place. Each change is small, individually defensible, and
collectively fatal to reproducibility.

### 1.1 What breaks without packaging discipline

| Question asked after an incident | Answer without discipline | Answer with discipline |
|---|---|---|
| Which model produced this score | "The production model" | Digest, version, and training run identifier |
| Which preprocessing was applied | "The usual pipeline" | Feature definitions and code digest in the artifact |
| Which dependency versions were present | "Whatever pip installed" | Lockfile with hashes, recorded in the software bill of materials (SBOM) |
| Can we rebuild the same artifact | "Probably, if the registry still has the base image" | Rebuild from the recorded inputs and compare digests |
| Was this artifact approved | "It was deployed, so yes" | Signature, attestation, and approval record |
| What else is in the image | "Python and our code" | SBOM with every package and version |

The table is the business case. Each row is a question that will be asked, not a hypothetical.

### 1.2 The specific cost in a regulated setting

In a medical device quality system, configuration management is a required process. IEC 62304
clause 8 requires identification, control, and status accounting of configuration items, and
clause 8.2 requires change control. A model service whose dependencies are unpinned cannot
demonstrate configuration status accounting, because the configuration is not knowable after the
fact.

Cybersecurity regulation adds a second layer. The US Food and Drug Administration's authority over
cyber devices includes a requirement for a software bill of materials in premarket submissions for
devices that meet the definition. A team that cannot produce an SBOM cannot meet that requirement,
regardless of how good the model is.

The commercial cost is subtler. A vendor that cannot reproduce a customer's deployed version
cannot diagnose a customer-specific defect. The support conversation degenerates into "send us your
logs" and "we cannot reproduce it," which is the beginning of a lost account.

### 1.3 Why containers are necessary but not sufficient

Containers solve the operating-system and runtime part of reproducibility. They do not solve the
model, the data, the configuration, or the build process. A container image built from a floating
tag, with dependencies resolved at build time, and a model downloaded at startup is not
reproducible even though it is containerized.

The mental model in Section 3 separates the layers. The container is one layer. The discipline is
the whole stack.

## 2. Learning outcomes

By the end of this topic you can:

1. **Identify** every layer that must be pinned for an artifact to be reproducible, and state what
   each layer contributes.
2. **Distinguish** bit-for-bit reproducibility from statistical equivalence, and choose the level a
   use case requires.
3. **Write** a build specification with a pinned base image, a hashed lockfile, and a multi-stage
   build.
4. **Generate** and interpret an SBOM, a provenance attestation, and a signature.
5. **Diagnose** environment drift from a failed rebuild or a changed prediction.
6. **Design** a clean-room rebuild test and a periodic rebuild drill.
7. **Map** packaging decisions to configuration management and cybersecurity obligations.

Outcomes 1 to 3 are assessed by Exercises 12.1 to 12.3; outcomes 4 and 5 by Exercises
12.4 to 12.6; outcomes 6 and 7 by Exercises 12.7 and 12.8. A learner who can pass 12.3 and 12.5
has the working skill this topic is for: the ability to build a sealed artifact and to explain a
rebuild failure from evidence.

## 3. The core mental model

Think of the deployable artifact as a sealed stack of layers. Each layer has an identity, and the
identity of the whole is a digest computed over the identities of the parts.

```text
        ┌──────────────────────────────────────────────┐
        │  Model card, intended use, validation report │  metadata
        ├──────────────────────────────────────────────┤
        │  Configuration: threshold, feature list      │  config
        ├──────────────────────────────────────────────┤
        │  Model weights / coefficients (content hash) │  model
        ├──────────────────────────────────────────────┤
        │  Application code (git commit)               │  code
        ├──────────────────────────────────────────────┤
        │  Python packages (lockfile + hashes)         │  dependencies
        ├──────────────────────────────────────────────┤
        │  System libraries (base image digest)        │  OS
        └──────────────────────────────────────────────┘
                       │
                       ▼
              OCI image digest  ──▶ registry ──▶ deployment
                       │
                       ├── SBOM (what is inside)
                       ├── provenance (how it was built)
                       └── signature (who approved it)
```

The digest at the bottom identifies the image, but only if every layer above it is inside the
image or referenced by content. A model mounted at runtime is outside the digest. A ConfigMap
edited after deployment is outside the digest. A secret injected at runtime is outside the digest
by design, which is why secrets must be rotated and recorded separately.

### 3.1 The three identities

Reproducibility requires three identities to be recorded together:

1. **Build identity**: the base image digest, the lockfile hash, the source commit, and the build
   platform.
2. **Data identity**: the content hash of the training snapshot and the feature definitions used.
3. **Runtime identity**: the deployed image digest, the configuration version, and the model
   version reported by the service.

A common failure is recording the first two and not the third. The training run is reproducible,
but the deployed artifact cannot be tied to it because the deployment records only a tag.

### 3.2 Tags move; digests do not

A tag is a pointer. `python:3.12-slim` resolves to whatever the publisher last pushed under that
name. A digest is a content address: `python@sha256:...` identifies one image, permanently. Pinning
by digest is the single highest-value packaging change a team can make, because it removes the
largest source of silent drift.

The same logic applies to model artifacts. A file named `model.json` on a shared drive can be
overwritten. A file whose SHA-256 is recorded in the image label and verified at startup cannot be
changed without detection.

### 3.3 What "reproducible" means at three levels

| Level | Definition | Evidence | Typical use |
|---|---|---|---|
| Recorded | Every input is recorded, but a rebuild may differ | Lockfile, data hash, build log | Internal analytics |
| Rebuildable | A rebuild produces an artifact that behaves identically within a stated tolerance | Rebuild drill with output comparison | Clinical model in service |
| Bit-for-bit | A rebuild produces an identical digest | Independent rebuild, matching digest | Regulated device, high-assurance supply chain |

Bit-for-bit reproducibility is difficult with GPU kernels, timestamps embedded in images, and
non-deterministic library builds. It is achievable with care and is not always necessary. The
important discipline is to state which level the artifact claims, and to test that claim.

### 3.4 The provenance chain

Provenance answers "how did this artifact come to exist." A complete chain links the source
commit, the build platform, the build inputs, and the output digest. SLSA, the supply-chain levels
for software artifacts, defines levels from L1 (provenance exists) to L3 (provenance is
unforgeable and the build is isolated).

```text
source commit ──▶ build definition ──▶ builder ──▶ output digest
      │                  │                 │             │
      └────────── provenance attestation ──┴─────────────┘
                              │
                              ▼
                    signature over attestation
```

The chain is only useful if a verifier checks it at deployment time. An attestation that is
generated but never verified is documentation, not a control.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Artifact | The complete deployable package: image, model, config, metadata | The model weights alone | Digest |
| Image | An OCI-compatible filesystem snapshot plus metadata | Used to mean the running service | Digest |
| Tag | A mutable name pointing to an image | Treated as a version | String |
| Digest | A content hash identifying an image or file | Confused with a tag | `sha256:` hex |
| Base image | The image a build starts from | Referenced by a floating tag | Digest |
| Lockfile | A resolved dependency list with exact versions and hashes | A requirements file with ranges | File hash |
| SBOM | Software bill of materials: a machine-readable inventory of components | A text list of top-level packages | SPDX or CycloneDX |
| Provenance | A record of how an artifact was built | A build log | in-toto attestation |
| Attestation | A signed statement about an artifact | A signature over the artifact itself | DSSE envelope |
| Signature | A cryptographic proof that a named identity approved an artifact | A checksum | Cosign or equivalent |
| Reproducible build | A build that can be recreated from recorded inputs | A build that succeeds twice | Digest or tolerance |
| Deterministic | Producing the same output from the same input | Producing a similar output | Bit-for-bit |
| Environment drift | A change in the runtime environment after the artifact was built | Any code change | Version delta |
| Content addressing | Identifying data by its hash | Identifying data by a path | SHA-256 |
| OCI | Open Container Initiative image and runtime specifications | Docker-specific format | Spec version |
| Multi-stage build | A build that uses separate stages for compilation and runtime | A single large image | Stage count |
| Layer cache | Reuse of unchanged image layers across builds | Assumed to be correct | Cache hit rate |
| Build argument | A value passed into a build, often a version | A secret | Key-value |
| Secret mount | Injecting a secret at build or runtime without baking it in | An environment variable in the image | Mount path |
| Immutable tag | A registry policy that prevents a tag from being moved | A tag that is rarely changed | Registry policy |
| Model registry | A versioned store of model artifacts and metadata | A shared folder | Registry URI |
| Model card | A structured summary of intended use, data, performance, and limits | A README | Document version |
| Data version | A content hash or version identifier for a training snapshot | A file path | Hash or DVC revision |
| Feature definition | A versioned specification of how a feature is computed | A SQL snippet | Version identifier |
| Runtime | The environment in which the artifact executes | The model | Image digest |
| Dependency confusion | An attack that exploits ambiguous package resolution | A typo in a package name | Attack technique |
| Hermetic build | A build that uses no network and only declared inputs | A build with a cache | Build property |
| Rebuild drill | A scheduled test that rebuilds and compares an artifact | A backup restore test | Drill record |

Two distinctions carry most of the operational weight.

**Reproducible is not the same as deterministic.** A build can be reproducible at the level of
behaviour while producing a different digest, because of embedded timestamps or nondeterministic
compilation. A build can be deterministic on one machine and not another, because of hardware or
library differences. State the claim precisely.

**An SBOM is not a vulnerability report.** An SBOM lists components. A vulnerability report maps
components to known vulnerabilities at a point in time. The SBOM is necessary for the report and
insufficient for it.

## 5. The mechanics, step by step

The procedure produces a sealed artifact and the evidence needed to defend it. Steps 1 to 4 build
the runtime. Steps 5 to 7 seal the contents. Steps 8 to 10 attach evidence and enforce it. Steps
11 to 13 test the claim.

### Step 1 — Declare the runtime contract

**Input.** The service code, the model, and the environment it needs.

**Output.** A written runtime contract: Python version, system libraries, CPU architecture, memory,
filesystem assumptions, network dependencies, and startup time.

**Rationale.** The runtime contract is the specification the artifact must satisfy. Without it,
the build is a sequence of implicit assumptions. "It needs Python 3.12" is incomplete; the artifact
may also need glibc, a timezone database, a specific CPU instruction set, and a writable temporary
directory.

The contract must distinguish build-time dependencies from runtime dependencies. A compiler is
needed to build a wheel and not needed to run the service. Keeping both in the runtime image
inflates the attack surface and the image size.

Branch point: if the service requires a GPU or a specific instruction set, the build must target
that architecture explicitly. An image built on x86 and deployed on arm64 will fail at startup with
an exec format error, often after the readiness probe has already marked it unhealthy.

### Step 2 — Pin the base image by digest

**Input.** The runtime contract and an approved base image.

**Output.** A base image reference with a digest, recorded in the Dockerfile and the provenance
record.

**Rationale.** A tag is a pointer that can move. Pinning by digest makes the base layer immutable
and auditable. It also makes the build reproducible across time, provided the registry retains the
image.

The base image should be minimal and maintained. A slim distribution reduces the number of
packages that need vulnerability scanning. A maintained image receives security patches, which
means the team must have a process to update the digest, rebuild, and redeploy. Pinning without a
patch process trades reproducibility for unpatched vulnerabilities.

Branch point: if the base image digest is no longer available from the registry, the build cannot
be reproduced. Keep an internal mirror of approved base images, and record the mirror digest.

### Step 3 — Resolve and lock dependencies with hashes

**Input.** The direct dependency list and the target Python version.

**Output.** A lockfile with exact versions, transitive dependencies, and cryptographic hashes.

**Rationale.** A requirements file with ranges resolves differently over time. A lockfile fixes the
resolution. Hashes prevent a compromised or substituted package from being installed silently.

The lock must be generated in the same environment as the build. A lock generated on macOS may
include platform-specific wheels that do not exist for Linux. The lock should be regenerated when
dependencies change, reviewed like code, and stored with the artifact.

Branch point: if a transitive dependency has no wheel for the target platform, the build must
compile it, which requires build tools in the build stage and a longer build. That is a reason to
use a multi-stage build, not a reason to loosen the lock.

### Step 4 — Separate build and runtime stages

**Input.** The lockfile, the source, and the runtime contract.

**Output.** A multi-stage Dockerfile with a build stage and a minimal runtime stage.

**Rationale.** The build stage needs compilers, headers, and package managers. The runtime stage
needs the application and its libraries. Copying only the built artifacts from the build stage to
the runtime stage reduces size and attack surface, and it makes the runtime image's contents
easier to inventory.

A single-stage image often contains pip caches, compilers, and test dependencies. These are not
needed at runtime and they appear in the SBOM as components that must be scanned and patched.

Branch point: if the application needs a system library at runtime that is only available in the
build image, install it explicitly in the runtime stage and record it. Do not copy the whole build
filesystem as a shortcut.

### Step 5 — Bake the model and its metadata into the artifact

**Input.** The model file, the feature definitions, the threshold configuration, and the model
card.

**Output.** An image containing all of them, with content hashes recorded in labels and verified at
startup.

**Rationale.** A model mounted at runtime is outside the image digest. The deployment then has two
identities that can drift independently: the image and the model. Baking the model in makes the
image digest identify the model. For large models, a content-addressed volume is acceptable, but
the volume's digest must be recorded and verified, and the deployment must refuse to start on a
mismatch.

The metadata belongs with the model. A model card stored in a wiki is not available to the service,
the auditor, or the incident responder at the moment they need it.

Branch point: if the model is too large for the image registry, use a content-addressed artifact
store and record the digest in the image label. The startup check must fail closed if the digest
does not match.

### Step 6 — Externalize configuration with a version

**Input.** Thresholds, feature flags, endpoint URLs, and other runtime settings.

**Output.** A versioned configuration object with a schema, separated from secrets.

**Rationale.** Configuration that is edited in place defeats reproducibility. Configuration that is
baked into the image makes every change a rebuild. The workable middle is a versioned configuration
object, deployed with the image, recorded by version, and validated against a schema at startup.

Secrets are a separate category. They must not be baked into the image, and they must be rotated
on a schedule. The artifact should reference a secret by name, not by value.

Branch point: if a configuration change alters model behaviour, such as a threshold, it is a model
change for change-control purposes even though the image digest is unchanged. Record it in the
change log and, where applicable, in the predetermined change control plan (PCCP) impact assessment.

### Step 7 — Make the build hermetic and record inputs

**Input.** The Dockerfile, lockfile, source commit, base digest, and build arguments.

**Output.** A build that uses only declared inputs and a record of those inputs.

**Rationale.** A hermetic build does not fetch arbitrary code from the network at build time. Every
input is declared, resolved from a controlled source, and recorded. This is what makes a rebuild
meaningful: the same declared inputs should produce the same artifact.

Full hermeticity is hard with package managers and base images. The practical target is to
minimize undeclared network access, pin what is fetched, and record what remains.

Branch point: if the build must access a private package index, record the index URL and the
credentials' rotation policy, and ensure the credentials are not embedded in a layer.

### Step 8 — Generate a software bill of materials

**Input.** The built image and the dependency metadata.

**Output.** An SBOM in SPDX or CycloneDX format, attached to the image.

**Rationale.** The SBOM answers "what is inside" without running the image. It is the basis for
vulnerability scanning, license review, and incident response when a new vulnerability is
announced. Generating it at build time is more reliable than reconstructing it later.

The SBOM must cover operating-system packages as well as language packages. A Python-only SBOM
misses the OpenSSL version in the base image, which is often the component with the most serious
vulnerabilities.

Branch point: if the build produces multiple artifacts, generate an SBOM per artifact and record
the relationship. An SBOM for a parent image that does not identify child artifacts is incomplete
for a multi-service deployment.

### Step 9 — Sign and attest

**Input.** The image digest, the SBOM, the provenance, and an identity.

**Output.** A signature over the digest and an attestation over the provenance and SBOM.

**Rationale.** A signature binds an identity to an artifact. An attestation binds evidence to an
artifact. Together they allow a deployer to answer "who built this, from what, and did the
approved identity sign it" without trusting the registry.

Keyless signing, as supported by Sigstore's cosign, binds a workload identity through OpenID
Connect and records the signing event in a transparency log. Key-based signing is also supported
and is often required in air-gapped environments.

Branch point: if the organization cannot use a public transparency log, use an internal log or a
key-based scheme with a documented key management process. The control is verification, not the
particular signing infrastructure.

### Step 10 — Verify at admission or deployment

**Input.** The signed artifact, the policy, and the verifier's trust root.

**Output.** An admission decision: allow or deny.

**Rationale.** A signature that is never verified is not a control. The verification should happen
at the point of deployment, using a policy that names the allowed builder identity and the required
attestations. Kubernetes admission controllers and registry policies can enforce this.

The policy must fail closed. A missing signature, an unknown builder, or a digest mismatch should
block deployment, not produce a warning that is ignored.

Branch point: if the deployment platform cannot verify signatures, verify in the CI pipeline and
record the result, then restrict registry write access so that only the pipeline can publish. This
is weaker than admission control but better than no verification.

### Step 11 — Test in a clean room

**Input.** The artifact and a fresh environment with no build cache.

**Output.** A test report: startup time, dependency resolution, model load, prediction parity, and
error behavior.

**Rationale.** Building and testing on the same machine hides missing dependencies and cached
state. A clean-room test pulls the artifact into a fresh environment and runs the acceptance tests.
This is where a missing system library or a model path assumption appears.

Prediction parity is the key test. Run a fixed set of inputs through the artifact and compare the
outputs with the reference implementation, within a stated tolerance.

Branch point: if the clean-room environment has no network, the test also proves that the service
does not depend on an undeclared external call at startup. That is a stronger and more useful test
than a networked one.

### Step 12 — Publish immutably and record provenance

**Input.** The verified artifact and its evidence.

**Output.** An immutable registry entry, a provenance record, and a deployment manifest.

**Rationale.** An immutable tag or a digest-only reference prevents a later build from overwriting
the artifact that was approved. The provenance record links the artifact to the source, the data,
and the build. The deployment manifest states which digest and which configuration version are
live.

Branch point: if the registry does not support immutable tags, use digest references everywhere and
add a registry policy check. A tag that can move is not a version.

### Step 13 — Run a rebuild drill

**Input.** The recorded inputs and a scheduled date.

**Output.** A drill report: whether the rebuild succeeded, whether the digest matched, whether the
predictions matched, and what differed.

**Rationale.** Reproducibility is a claim, and claims decay. A drill tests the claim before an
incident does. The drill should be quarterly for a clinical service and after any base-image or
dependency change.

The drill must use the recorded inputs, not the current repository state. Rebuilding from `main`
tests the current code, not the deployed artifact.

Branch point: if the rebuild fails because a dependency or base image is gone, the artifact is no
longer reproducible. Record the gap, mirror the missing inputs, and decide whether the deployed
artifact can remain in service without a rebuild path.

### 5.14 Edge cases the procedure must handle

**Dependency resolution drift.** A lockfile generated without hashes can resolve to a different
patch version. Regenerate and review locks on a schedule, and fail the build if the lock is out of
date relative to the manifest.

**Base image removal.** Registries can delete old images. Mirror approved base images internally
and record the mirror digest.

**Private index outage.** The build cannot fetch dependencies. Use an internal package mirror and
a build cache that is content-addressed, not a mutable cache.

**Large model artifacts.** The image registry has size limits. Use a content-addressed artifact
store, record the digest, and verify at startup.

**Secrets in layers.** A secret copied into an image remains in the layer history even if deleted
in a later layer. Never copy secrets into the build context; use build secrets or runtime
injection.

**Architecture mismatch.** The image is built for the wrong CPU architecture. Build multi-arch
images or pin the target platform explicitly.

**Locale and timezone.** The runtime image lacks the timezone database or a locale, and timestamps
are wrong. Install the required data and test a daylight-saving transition.

**Build cache poisoning.** A shared cache serves a stale or malicious layer. Use cache keys tied to
content and trust boundaries, and rebuild from scratch in the drill.

**Registry outage.** The deployment cannot pull the image. Mirror the artifact in the target
region and record the mirror.

### 5.15 Required reproducibility level by context

| Context | Minimum level | Evidence required | Rebuild cadence |
|---|---|---|---|
| Internal analytics | Recorded | Lockfile, data hash, build log | On request |
| Clinical service, non-device | Rebuildable | Recorded inputs plus quarterly drill | Quarterly |
| Regulated device | Bit-for-bit or verified rebuildable | Independent rebuild or verified reproducible build | Per release and annually |
| Third-party supplied model | Rebuildable or verified provenance | Supplier SBOM, signature, and provenance | Per supplier release |
| Research publication | Recorded with container | Image digest and data snapshot | On request |

### 5.16 The build specification, annotated

The reference service's `labs/mlops/Dockerfile` is a single-stage build that installs dependencies,
copies the application, and runs as a non-root user. It is a reasonable teaching starting point and
a useful contrast with a sealed build.

```dockerfile
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN groupadd -r app && useradd -r -g app app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

The line-by-line reading:

| Line | What it does | Reproducibility gap |
|---|---|---|
| `FROM python:3.12-slim` | Selects a base image by tag | The tag can move; use a digest |
| `COPY requirements.txt` | Copies a range-based dependency list | Resolution is not fixed; use a hash-locked file |
| `pip install` | Installs at build time from the network | The index can change; hashes and a mirror are needed |
| `COPY app ./app` | Copies code and model | The model digest is not recorded or verified |
| `USER app` | Drops root privileges | Good practice; keep it |
| `CMD` | Starts the service | Startup does not verify the model digest |

A sealed version of the same build would pin the base digest, install from a hashed lockfile,
use a build stage to compile any platform-specific wheels, copy only the runtime artifacts, add
labels for the model digest and source commit, and run a startup check that verifies the model
file against the recorded digest before the readiness probe can succeed.

The contrast is the lesson. The lab's Dockerfile is not wrong for a teaching service with no
patient data and no regulatory claim. It would be insufficient for a service that must be rebuilt
after an incident, because it does not record enough to rebuild the same thing.

## 6. Worked example

A hospital runs a risk service built in February 2025. In April 2026, a clinician reports that a
patient with a documented creatinine of 3.1 mg/dL received a low risk score. The incident review
asks the team to rebuild the exact artifact that was running in February 2025 and reproduce the
score.

The image was built with a floating base tag, a range-based requirements file, and a model file
copied from a shared build context. The configuration threshold lived in a Kubernetes ConfigMap.

### 6.1 The build economics

The team builds 42 images per month across the service. The release pipeline's success objective is
95 percent of builds succeeding within 30 minutes, measured over a 30-day window.

```text
allowed_failures = 42 × 0.05 = 2.1 per month
observed_rate    = 0.06
observed_failures = 42 × 0.06 = 2.52 per month
burn_rate        = 0.06 / 0.05 = 1.2
```

A burn rate of 1.2 is a slow leak, not an emergency. The cost of the leak is:

```text
engineer_time   = 3 hours at 95 USD per hour = 285 USD
compute_time    = 12 USD per failed build
cost_per_failure = 297 USD
monthly_cost     = 2.52 × 297 = 748 USD
```

The remediation is 16 hours of engineering to pin the base image, generate a hashed lockfile, and
add a startup digest check:

```text
remediation_cost = 16 × 95 = 1,520 USD
payback_months   = 1,520 / 748 = 2.03 months
```

The payback is two months, before counting the value of a successful rebuild during an incident.
That value is not a monthly saving; it is the difference between a two-hour investigation and a
two-week one.

### 6.2 What the rebuild actually does

The team checks out the source commit recorded in the deployment manifest and runs the build. The
build resolves `python:3.12-slim` to a digest published eleven months after the original build.
The base layer grows from 780 MB to 960 MB because the upstream image added packages.

The requirements file resolves `fastapi>=0.111` to a newer minor version and `pydantic>=2.7` to a
newer minor version. The application code is unchanged. The model file is copied from the build
context, but the build context is a directory that the training pipeline writes to, and the file
there was overwritten in January 2026 by a retraining run that was never promoted.

The rebuilt image therefore differs in three ways:

| Layer | Original | Rebuild | Consequence |
|---|---|---|---|
| Base OS | Digest A, 780 MB | Digest B, 960 MB | New packages, new vulnerabilities, different libc |
| Python dependencies | fastapi 0.111, pydantic 2.7 | fastapi 0.115, pydantic 2.9 | Stricter validation, changed error behavior |
| Model file | SHA-256 `a1...` | SHA-256 `7c...` | Different coefficients, different scores |

The image digest differs, which is expected. The problem is that the prediction differences are
not explainable from the record, because the record does not say which model file was in the
original image.

### 6.3 Diagnostic one: prediction parity fails

The team replays 10,000 de-identified requests through the rebuilt image and compares the outputs
with the archived outputs from the original deployment. Of the 10,000, 180 requests, or 1.8
percent, produce a different decision at the same threshold.

The first diagnostic is a parity diff by feature. The differing requests cluster among patients
with creatinine values in a narrow range, which points to the model coefficients rather than to
the dependency versions. The second diagnostic is a hash comparison between the model file in the
image and the hash recorded in the training run that produced the deployed model. They do not
match.

The conclusion is that the rebuild used a different model. The image was never a complete
identity for the deployed system; the model was outside the recorded digest.

### 6.4 Diagnostic two: validation behavior changes

The rebuilt image also rejects 4.7 percent of replayed requests with a 422 that the original
accepted. The requests contain `prior_admissions: null`. The original pydantic version coerced
null to zero in this field; the newer version rejects it.

This is a dependency-drift failure, not a model failure. It would have been caught by a clean-room
test that replayed a fixed request corpus and compared status codes, but the pipeline only tested
the happy path.

The two diagnostics together separate the causes: the model mismatch explains the score changes,
and the dependency change explains the validation changes. Neither is visible from the image
digest alone.

### 6.5 Vary one assumption: pin everything

Suppose the team had pinned the base digest, generated a hashed lockfile, baked the model with a
recorded digest, and added a startup check. The rebuild would produce:

```text
prediction_differences = 0 of 10,000
validation_differences = 0 of 10,000
digest_match           = app layer matches; full image may differ on timestamps
build_time_increase    = 2 minutes
image_size_increase    = 0 MB (the pinned base is the same size)
```

The full image digest may still differ because of embedded timestamps and build metadata. That is
why the claim should be "rebuildable and behaviourally identical", not "bit-for-bit", unless the
team has done the additional work to make the build deterministic.

If the build failure rate were 12 percent instead of 6 percent, the burn rate would be 2.4 and the
monthly cost 1,497 USD. The payback on the 1,520 USD remediation would fall to about one month,
and the decision to fix immediately rather than next quarter would be unambiguous. This is the
general shape of a reliability investment: the payback depends on the failure rate, and the failure
rate is usually unknown until it is measured.

### 6.6 The storage and pull arithmetic

The image is 1.42 GB. The registry retains 24 versions.

```text
storage = 24 × 1.42 GB = 34.08 GB
cost    = 34.08 × 0.10 USD per GB-month = 3.41 USD per month
pull_time = 1.42 GB / 100 MB per second = 14.2 seconds
```

Storage is negligible. Pull time matters at scale: a rolling deployment of six replicas pulls 8.5
GB, and a cold node pulling at 100 MB per second takes 85 seconds before the first pod is ready.
That is a capacity and rollout consideration (Topic 08), not a storage cost consideration.

The SBOM contains 212 packages: 96 operating-system packages and 116 Python packages. A scan at
rebuild time reports 14 known vulnerabilities, 3 of them critical, all in the base image. The
original image, built fourteen months earlier, had 4 findings at the time and would have far more
if scanned today. Pinning the base image without a patch cadence would preserve the old
vulnerabilities; the correct policy is to pin, scan, and update on a schedule.

### 6.7 What the example establishes

The image digest is necessary but not sufficient. The deployed system's identity is the image
digest plus the model digest plus the configuration version. A packaging process that records only
the first cannot answer the question an incident review asks.

The example also shows that the cost of poor packaging is not mainly storage or compute. It is
engineer time on failed rebuilds and the much larger cost of an investigation that cannot
reproduce the system under review.

### 6.8 Reconstructing the original score by hand

Because the image is not a complete identity, the team reconstructs the original environment
manually. It finds the training run identifier in the experiment tracker, retrieves the model file
from the run's artifact store, and confirms the file hash against the run record. It finds the
base image digest in the container registry's access log for the original build, which happens to
retain digest history even though the tag moved. It finds the dependency versions in the build log,
which was archived.

The reconstruction takes 19 hours across three engineers. It succeeds, and the original score is
reproduced to four decimal places. The team can now answer the clinical question: the score was
low because the model used the correct creatinine value but the feature pipeline queried the
laboratory system with a lookback window that excluded the result by eleven minutes.

The 19 hours are the cost of missing provenance. With a complete artifact identity, the same
reconstruction would have taken about two hours: pull the digest, verify the model hash, replay
the request. The difference is not storage or compute. It is the engineering time required to
reconstruct a system that was never fully identified.

The incident review records two corrective actions. First, bake the model into the image and
record its digest in an image label. Second, add a provenance attestation that names the training
run and the data snapshot, so that the next reconstruction starts from a record rather than from
an archaeology project.

## 7. Code, query, or calculation

The script demonstrates two things: how an artifact digest changes when any layer changes, and the
build economics from Section 6. It creates a temporary directory, hashes five synthetic layers,
rebuilds with two changed layers, and reports the difference. It uses only the standard library.

```python
"""Artifact identity and build economics. Standard library only."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

LAYERS = ["base", "dependencies", "code", "model", "config"]


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_manifest(root: Path) -> dict[str, str]:
    manifest = {}
    for layer in LAYERS:
        path = root / f"{layer}.bin"
        manifest[layer] = digest_bytes(path.read_bytes())
    return manifest


def artifact_digest(manifest: dict[str, str]) -> str:
    payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
    return digest_bytes(payload)


def changed_layers(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(k for k in before if before[k] != after[k])


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for layer in LAYERS:
            (root / f"{layer}.bin").write_bytes(f"{layer}-v1".encode())
        original = build_manifest(root)
        original_digest = artifact_digest(original)

        (root / "base.bin").write_bytes(b"base-v2-with-extra-packages")
        (root / "model.bin").write_bytes(b"model-retrained-january")
        rebuilt = build_manifest(root)
        rebuilt_digest = artifact_digest(rebuilt)

        differences = changed_layers(original, rebuilt)
        print(f"original_digest      {original_digest[:16]}")
        print(f"rebuilt_digest       {rebuilt_digest[:16]}")
        print(f"digest_match         {original_digest == rebuilt_digest}")
        print(f"changed_layers       {','.join(differences)}")

    builds_per_month = 42
    success_slo = 0.95
    observed_failure = 0.06
    allowed = builds_per_month * (1 - success_slo)
    failures = builds_per_month * observed_failure
    burn = observed_failure / (1 - success_slo)
    cost_per_failure = 3 * 95 + 12
    monthly_cost = failures * cost_per_failure
    remediation = 16 * 95
    payback = remediation / monthly_cost

    print(f"allowed_failures     {allowed:.2f}")
    print(f"observed_failures    {failures:.2f}")
    print(f"burn_rate            {burn:.2f}")
    print(f"cost_per_failure     {cost_per_failure} USD")
    print(f"monthly_cost         {monthly_cost:.0f} USD")
    print(f"payback_months       {payback:.2f}")

    image_gb = 1.42
    versions = 24
    storage_gb = image_gb * versions
    storage_cost = storage_gb * 0.10
    pull_seconds = image_gb * 1000 / 100
    rollout_gb = image_gb * 6
    rollout_seconds = rollout_gb * 1000 / 100

    print(f"storage_gb           {storage_gb:.2f}")
    print(f"storage_cost         {storage_cost:.2f} USD")
    print(f"pull_seconds         {pull_seconds:.1f}")
    print(f"rollout_seconds      {rollout_seconds:.1f}")

    assert differences == ["base", "model"]
    assert abs(burn - 1.2) < 1e-9
    assert abs(payback - 2.03) < 0.01


if __name__ == "__main__":
    main()
```

Expected output, confirmed on Python 3.11:

```text
original_digest      80df1613b698cdad
rebuilt_digest       e000c8e1bd1be35c
digest_match         False
changed_layers       base,model
allowed_failures     2.10
observed_failures    2.52
burn_rate            1.20
cost_per_failure     297 USD
monthly_cost         748 USD
payback_months       2.03
storage_gb           34.08
storage_cost         3.41 USD
pull_seconds         14.2
rollout_seconds      85.2
```

The digest values are deterministic for this input and will differ if the layer content changes.
That property is the point: the digest is a function of content, not of time or build number.

The script hashes synthetic layer files, not real container layers. A real implementation would
hash the files that make up each layer and store the per-layer digests in an attestation. The OCI
image manifest does this natively; the script reproduces the logic so that the failure mode is
visible.

The reference service's `labs/mlops/Dockerfile` does not record a manifest of this kind. Its image
digest identifies the filesystem, but the model file inside it is not separately hashed, and the
deployment does not verify it at startup. Adding a `model.sha256` file and a startup check would
close that gap without changing the build system.

## 8. Data, standards, and regulatory context

Packaging is governed by a mixture of open specifications, procurement requirements, and medical
device regulation. The table separates the binding from the conventional.

### 8.1 The instruments

| Instrument | Status | What it constrains | What it does not constrain |
|---|---|---|---|
| OCI Image Specification | Open specification; binding through tooling | Image manifest, layers, configuration, digests | Build process or dependency management |
| OCI Distribution Specification | Open specification | Registry application programming interface (API), content addressing, tag behavior | Retention policy or signing |
| SLSA v1.2 | Voluntary supply-chain framework | Build provenance levels, builder isolation, attestation format | Whether a model is clinically valid |
| in-toto attestation framework | Open specification | Statement and predicate format, DSSE envelope | The policy that consumes the attestation |
| SPDX and CycloneDX | Open SBOM formats | Component inventory structure and identifiers | Vulnerability status or exploitability |
| Sigstore and cosign | Open tooling and services | Keyless signing, transparency log, attestation storage | Key management policy in air-gapped settings |
| NIST SP 800-218, Secure Software Development Framework | Voluntary for most organizations; required in some federal procurement | Secure build practices, provenance, vulnerability response | Medical device-specific validation |
| FDA cybersecurity premarket guidance, 2023 | Final guidance for devices with cyber device status | Security risk management, SBOM, vulnerability monitoring | Model performance or clinical evidence |
| FD&C Act section 524B | Binding statute for cyber devices | SBOM and cybersecurity plan in premarket submissions | The format of the SBOM, beyond accepted standards |
| IEC 62304 clause 8 | Binding when invoked by the quality system | Configuration identification, control, status accounting, change control | Specific tooling or registry choice |
| ISO 13485 | Binding when invoked | Document and record control, traceability | Software build reproducibility specifically |
| 21 CFR Part 11 | Binding where predicate rules apply | Audit trails for electronic records, record retention | Container image contents |
| HIPAA Security Rule | Binding for covered entities and business associates | Audit controls, access control, integrity | Build provenance |

### 8.2 What the SBOM requirement actually asks

Section 524B of the Food, Drug, and Cosmetic Act, added by the Consolidated Appropriations Act of
2023, requires premarket submissions for cyber devices to include a software bill of materials.
The FDA's 2023 cybersecurity guidance describes the expectation in more detail, including a
machine-readable SBOM and a process for monitoring and responding to vulnerabilities.

The requirement is about the device's software, which includes the model service and its
dependencies. A Python SBOM that omits the base operating system is incomplete for this purpose.
A CycloneDX or SPDX document that includes operating-system packages, language packages, and their
versions is the practical target.

The SBOM does not prove the absence of vulnerabilities. It makes the question answerable. When a
new vulnerability is announced, the SBOM is what allows a manufacturer to determine, within hours,
which deployed devices contain the affected component.

### 8.3 What IEC 62304 configuration management requires

Clause 8.1 requires a configuration management plan that identifies configuration items, defines
the naming and versioning scheme, and controls changes. For a model service, the configuration
items include the source code, the build files, the model artifact, the feature definitions, the
configuration schema, and the documentation.

Clause 8.2 requires that changes to configuration items be controlled: identified, reviewed,
approved, and verified. A tag that can be moved is a configuration item without control. A model
file on a shared drive is a configuration item without identification.

### 8.4 What provenance does and does not establish

SLSA provenance establishes how an artifact was built: which builder, which inputs, which source.
It does not establish that the source is correct, that the model is valid, or that the build was
authorized. Those are separate controls: code review, validation, and approval.

A common overclaim is "we have SLSA Level 3, so our supply chain is secure." Level 3 establishes
that the provenance is unforgeable and the build is isolated. It does not establish that the
dependencies are free of vulnerabilities, that the source is trustworthy, or that the artifact
was deployed as intended.

### 8.5 What is convention

- Docker as the container runtime. OCI images can be built and run by several tools.
- A particular registry vendor. The requirement is content addressing and access control.
- Keyless signing. Key-based signing is acceptable and sometimes required.
- A specific SBOM format. SPDX and CycloneDX are both accepted in practice; the choice should
  follow the consumer's tooling.
- Semantic versioning of images. Digest pinning is stronger; a version tag is a convenience.

> **Note:** A signed artifact from an approved builder is not automatically safe to deploy. The
> deployment policy must also check the vulnerability scan result, the model approval record, and
> the configuration version. Signing establishes identity, not acceptability.

## 9. Industry practice and case evidence

### 9.1 Log4Shell and the value of a component inventory

In December 2021, a critical vulnerability was published in the Log4j library (CVE-2021-44228),
affecting a very large number of Java applications. Organizations that maintained a component
inventory could determine exposure quickly; organizations that did not had to search their
systems manually.

For machine learning services, the lesson transfers to Python and operating-system packages. A
model service may depend on a numerical library with a transitive dependency that is vulnerable.
Without an SBOM, the question "are we exposed" is answered by rebuilding and scanning each image,
which is slow and incomplete.

The inference for packaging is that the SBOM's value is realized during an incident, not at build
time. A team that generates an SBOM and never stores it with the artifact has created a document,
not a capability.

### 9.2 The Codecov uploader compromise

In 2021, attackers modified a Bash uploader script distributed by Codecov, a code coverage service.
The script was executed in customers' continuous integration environments and could export
environment variables, including credentials. The compromise was disclosed by Codecov and
investigated by customers.

The documented lesson is that build-time scripts are part of the supply chain. A packaging process
that pins application dependencies but downloads an unpinned shell script from a vendor has a gap.
The inference is that the build should fetch only declared, verified inputs, including scripts and
base images.

### 9.3 SolarWinds and signed build systems

The SolarWinds compromise involved malicious code inserted into a build system and distributed in a
signed update. The signature was valid because the build system was compromised, not because the
signature was forged.

The lesson for packaging is that signing is necessary and not sufficient. Provenance that names the
builder is only as trustworthy as the builder. This is why SLSA emphasizes isolated, ephemeral
build environments and why verification must check the builder identity, not only the presence of a
signature.

### 9.4 FDA SBOM expectations in practice

The FDA's 2023 cybersecurity guidance and section 524B changed the premarket expectations for
connected devices. Manufacturers now need a machine-readable SBOM and a vulnerability monitoring
process. Public FDA guidance does not prescribe a single toolchain, and practice varies.

The inference, clearly labelled, is that many clinical AI teams will inherit an SBOM requirement
from the device classification even when their model is delivered as software. A team that builds
the SBOM at image build time and stores it with the artifact is better positioned than one that
assembles it manually before a submission.

### 9.5 What the evidence does not show

There is little public evidence on how often clinical model deployments fail a rebuild drill, or on
the distribution of rebuild failures by cause. Vendor documentation describes provenance and
signing capabilities, not operational outcomes. A team should measure its own rebuild success rate
and treat that as the primary evidence for investment.

A second gap is that public case reports describe dramatic supply-chain failures and rarely
describe the base rate of ordinary rebuild failures. The dramatic cases justify the control; the
base rate determines whether the control pays for itself. Measure the base rate locally before
quoting a payback period to a budget owner.

The common thread in these cases is that the failure was not in the model. It was in
the identity, the inputs, or the build environment. That is the argument for treating packaging
as a control system with evidence, not as a build script that happens to work.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Floating base tag | Rebuild produces a different base layer | The tag resolves to a newer image | Unrecorded OS and library changes | Pin by digest; mirror approved images |
| Unpinned dependencies | Two builds in the same week differ | Resolution picks newer patch versions | Behaviour changes without a code change | Hashed lockfile, reviewed on change |
| Model outside the image | Image digest is stable but scores change | Model mounted from a mutable path | The artifact does not identify the model | Bake the model or record and verify its digest |
| Mutable configuration | Threshold differs between replicas | ConfigMap edited in place | Inconsistent decisions from the same image | Versioned config, schema validation, restart on change |
| Secret in a layer | Image history contains a token | Secret copied during build | Credential exposure through the registry | Build secrets or runtime injection |
| Incomplete SBOM | Scan finds a package the SBOM omits | Tool covers Python but not OS packages | Exposure assessment is wrong | Generate at build time from the full image |
| Signature never verified | Unsigned images deploy successfully | Admission policy is permissive | The control exists only on paper | Fail-closed admission policy |
| Tag reuse | A deployed tag points to a new digest | Registry allows mutable tags | Rollback restores the wrong artifact | Immutable tags or digest-only references |
| Non-hermetic build | Build fails when the network is restricted | Undeclared download at build time | Unreproducible and unauditable build | Declare and mirror all inputs |
| Missing platform wheel | Build compiles from source or fails | Lock generated on another OS | Long builds or no artifact | Lock per platform; build stage with toolchain |
| Architecture mismatch | Pods crash with exec format error | Image built for x86, deployed on arm64 | Service unavailable after rollout | Multi-arch build; explicit platform |
| Oversized image | Pull time exceeds the rollout budget | Model and build tools in one image | Slow scale-out and failed readiness | Multi-stage build; artifact store for large models |
| Registry retention | Old digest cannot be pulled | Cleanup policy deleted the image | Rollback and rebuild both fail | Retention policy aligned with audit period |
| Timestamps in build | Digest differs on every build | Build embeds time or random IDs | Bit-for-bit claim is false | Deterministic build flags; state the claim |
| Dependency confusion | An internal package name resolves publicly | Package index priority is misconfigured | Malicious package installed | Scoped names, private index first, hashes |

Red flags: a Dockerfile whose `FROM` line has no digest; a requirements file whose entries use
`>=`; a deployment manifest that references a tag rather than a digest; an SBOM generated after
deployment rather than at build; a signature step in CI with no verification step at admission.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The registry retains the digests in use | Rollback and rebuild depend on them | Retention policy, vendor migration | Pull every deployed digest monthly |
| The lockfile resolves on the target platform | The build needs installable artifacts | Platform-specific wheels, compiler differences | Build in a clean container of the target platform |
| The model fits the image or artifact store | The artifact must contain or reference it | Model size growth | Check size limits; test the large-model path |
| The build platform matches the runtime architecture | The image must execute | Mixed x86 and arm64 fleets | Multi-arch build and a deployment test |
| The SBOM tool covers every ecosystem | Exposure assessment depends on completeness | OS packages, vendored libraries, native extensions | Compare SBOM against a filesystem scan |
| Signatures bind a meaningful identity | Verification depends on it | Shared keys, compromised CI | Review the trust root and builder policy |
| Build logs share a time source | Ordering and correlation depend on it | Clock skew across runners | Record timestamps with offsets |
| Configuration is deployed with the image | Behaviour is reproducible | Separate manual edits | Compare live config hash to the manifest |
| The drill uses recorded inputs | It tests the deployed artifact | Rebuilding from `main` | Audit the drill procedure |
| Patching the base image preserves compatibility | Security updates are required | ABI or library changes | Rebuild and run the full test suite |
| The artifact store is durable | Rebuilds need the model and data | Vendor deletion, account closure | Mirror critical artifacts |

## 12. Exercises

### Exercise 12.1 — Inventory the layers

Choose a model service you have built or the reference service in `labs/mlops/`. List every layer
from Section 3, state where each layer currently lives, and mark whether it is inside the image
digest, referenced by a recorded digest, or unrecorded.

**Deliverable.** A six-row table with location and identity status.

**Acceptance criteria.** All layers covered; the model and configuration are assessed separately
from the image; at least one unrecorded layer is identified for the reference service.

**Time.** 30 minutes.

### Exercise 12.2 — Compute rebuild economics

A team builds 60 images per month. The success objective is 97 percent. The observed failure rate
is 9 percent. A failed build costs 400 USD in engineer and compute time. Remediation requires 20
hours at 110 USD per hour. Compute allowed failures, observed failures, burn rate, monthly cost,
and payback.

**Deliverable.** Five numbers with the arithmetic.

**Acceptance criteria.** Burn computed against the failure budget, not the success rate; payback
computed from the monthly cost; units and currency stated.

**Time.** 25 minutes.

### Exercise 12.3 — Write a sealed Dockerfile

Rewrite the reference service's Dockerfile with a pinned base digest, a multi-stage build, a
hashed lockfile, a non-root user, and a startup digest check for the model file. Annotate each
line with the layer it controls.

**Deliverable.** A Dockerfile and a line-by-line annotation table.

**Acceptance criteria.** Base image pinned by digest; build and runtime stages separated; no
secrets in layers; model digest recorded and verified; the image still runs the existing tests.

**Time.** 60 minutes.

### Exercise 12.4 — Compare two manifests

Using the script in Section 7, change only the configuration layer and rerun. Report which layers
differ, whether the artifact digest changes, and what that implies for a deployment that pins the
image but not the configuration.

**Deliverable.** The modified script, its output, and a three-sentence interpretation.

**Acceptance criteria.** Correct layer identified; digest change observed; interpretation
distinguishes image identity from configuration identity.

**Time.** 25 minutes.

### Exercise 12.5 — Diagnose a rebuild mismatch

A rebuild produces the same application code, a newer base image, a newer pydantic version, and a
model file with a different hash. Replay shows 2.1 percent different decisions and 5.3 percent new
422 responses. Diagnose each difference, name the layer responsible, and state the fix.

**Deliverable.** A 350-word diagnosis with a table of symptom, layer, and fix.

**Acceptance criteria.** Separates model differences from dependency differences; does not blame
the model for the validation errors; proposes a parity test and a lockfile.

**Time.** 35 minutes.

### Exercise 12.6 — Design the evidence pipeline

Design a build pipeline that produces an image, an SBOM, a provenance attestation, and a signature,
then verifies all four at deployment. State the failure behavior for each missing artifact.

**Deliverable.** A pipeline diagram and a verification policy table.

**Acceptance criteria.** Every artifact has a producer and a verifier; admission fails closed;
the builder identity is checked, not only the signature; the policy is testable.

**Time.** 50 minutes.

### Exercise 12.7 — Plan a rebuild drill

Write the procedure for a quarterly rebuild drill for a clinical model service. Include which
inputs are used, how the digest is compared, how predictions are compared, what counts as failure,
and who signs the report.

**Deliverable.** A two-page drill procedure with a report template.

**Acceptance criteria.** Uses the recorded artifact inputs, not `main`; states a numeric parity
tolerance; defines failure and escalation; names an accountable signer.

**Time.** 45 minutes.

### Exercise 12.8 — Map to configuration management

For the same service, map each packaging artifact to the IEC 62304 clause 8 configuration
management activities: identification, control, status accounting, and audit. Identify one gap.

**Deliverable.** A four-row table plus a gap statement.

**Acceptance criteria.** Each activity has a concrete artifact and record; the gap is specific and
remediable; the answer distinguishes a tool from a control.

**Time.** 40 minutes.

## 13. Self-check answers

### 13.1 Exercise 12.1

For the reference service, the base image is pinned by tag, not digest, so the OS layer is
unrecorded. Dependencies are in a range-based requirements file, so the dependency layer is
unrecorded. The model file is inside the image but has no separate hash or startup check, so its
identity is implicit rather than recorded. Configuration is the model's own threshold inside
`model.json`, which is inside the image, so it is recorded but not independently versioned. The
model card is in the repository, not in the image.

### 13.2 Exercise 12.2

```text
allowed_failures  = 60 × (1 − 0.97) = 1.8
observed_failures = 60 × 0.09 = 5.4
burn_rate         = 0.09 / 0.03 = 3.0
monthly_cost      = 5.4 × 400 = 2,160 USD
payback_months    = (20 × 110) / 2,160 = 2,200 / 2,160 = 1.02
```

The burn rate of 3.0 means the failure budget is consumed three times faster than the window
allows. The remediation pays back in about one month, so the decision is not close.

### 13.3 Exercise 12.4

Only the configuration layer differs. The artifact digest changes because the digest is computed
over the manifest of all layers, including configuration. This is the desired behaviour: a
configuration change that alters model behaviour should change the artifact identity.

A deployment that pins the image digest but reads configuration from a mutable source has an
identity gap. The image digest identifies the code and the default configuration; the live
configuration may be different. The fix is to version the configuration object and record its
hash in the deployment manifest, or to bake validated configuration into the image and treat any
change as a rebuild.

### 13.4 Exercise 12.5

The 2.1 percent decision differences are consistent with a different model file. The responsible
layer is the model, and the fix is to bake the model with a recorded digest and verify it at
startup.

The 5.3 percent new 422 responses are consistent with stricter validation in a newer pydantic
version. The responsible layer is dependencies, and the fix is a hashed lockfile plus a replay
test that compares status codes, not only successful predictions.

The two causes are independent. A team that fixes only the model will still have the validation
regression; a team that fixes only the lockfile will still have different scores.

### 13.5 Rubric for the design exercises

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Completeness | 25% | Pins one layer | Pins image, deps, model | Records and verifies every layer |
| Evidence | 25% | Generates an SBOM | Signs and attests | Verifies at admission with a policy |
| Failure handling | 20% | Assumes the build works | Adds a clean-room test | Tests reproducibility on a schedule |
| Traceability | 20% | Names a tool | Maps artifacts to records | Maps to IEC 62304 activities |
| Operability | 10% | One-off process | Documented pipeline | Measured build success and payback |

### 13.6 Exercise 12.7

A correct drill uses the deployment manifest's recorded digest and build inputs, rebuilds in a
clean environment, and compares both the digest and the predictions. The failure criteria should
be numeric: a digest mismatch with identical predictions is a warning; a prediction mismatch
above the stated tolerance is a failure. The report is signed by the service owner and retained
with the quality record.

## 14. Interview questions

**1. What makes an artifact reproducible?**
Strong answers name the layers: base image, dependencies, code, model, config, and metadata, and
say that each must be identified. Weak answers say "use Docker."

**2. Why pin a base image by digest rather than tag?**
Strong answers explain that tags move and digests are content addresses. Weak answers say digests
look more professional.

**3. What is the difference between an SBOM and a vulnerability scan?**
Strong answers say the SBOM is an inventory and the scan is a point-in-time mapping to known
vulnerabilities. Weak answers treat them as the same artifact.

**4. What does a signature prove?**
Strong answers say it binds an identity to a digest, and that it does not prove safety or
correctness. Weak answers say it proves the artifact is trusted.

**5. How would you test reproducibility?**
Strong answers describe a clean-room rebuild from recorded inputs and a prediction parity check.
Weak answers say "build it twice."

**6. What is provenance and why does it matter?**
Strong answers describe the build inputs, builder identity, and the chain to the output. Weak
answers describe a build log.

**7. How do you handle a model too large for the image?**
Strong answers use a content-addressed artifact store, record the digest, and verify at startup.
Weak answers say to compress it.

**8. What is configuration drift and how do you prevent it?**
Strong answers version configuration, validate a schema, and record the deployed hash. Weak
answers say to use environment variables.

**9. What would you do if a base image digest disappeared?**
Strong answers describe an internal mirror, a retention policy, and a documented gap. Weak
answers say to use the latest tag.

**10. How does packaging interact with change control?**
Strong answers say a dependency or threshold change can alter output and is therefore a change
requiring assessment. Weak answers say only model weights matter.

## 15. Key terms

- **Artifact** — the complete deployable package of image, model, configuration, and metadata.
- **Attestation** — a signed statement about how an artifact was built or what it contains.
- **Base image** — the starting image for a build, ideally pinned by digest.
- **Bit-for-bit reproducibility** — a rebuild produces an identical digest.
- **Build argument** — a value passed into a build, distinct from a secret.
- **Content addressing** — identifying data by its hash rather than its path.
- **Dependency confusion** — an attack exploiting ambiguous package resolution.
- **Deterministic build** — a build that produces identical output from identical input.
- **Digest** — a content hash that identifies an image or file.
- **DSSE** — Dead Simple Signing Envelope, a format for signed attestations.
- **Environment drift** — a change in the runtime environment after the build.
- **Hermetic build** — a build that uses only declared inputs and no arbitrary network access.
- **Immutable tag** — a registry policy preventing a tag from being reassigned.
- **in-toto** — a framework for supply-chain attestations.
- **Layer** — a filesystem change set in a container image.
- **Lockfile** — a resolved dependency list with exact versions and hashes.
- **Model card** — a structured summary of a model's intended use, data, and limits.
- **Model registry** — a versioned store of model artifacts and metadata.
- **Multi-stage build** — a build with separate stages for compilation and runtime.
- **OCI** — Open Container Initiative, the image and runtime specification.
- **Provenance** — a record of how an artifact was built.
- **Rebuild drill** — a scheduled test that rebuilds and compares a deployed artifact.
- **Reproducible build** — a build that can be recreated from recorded inputs.
- **SBOM** — software bill of materials, a machine-readable component inventory.
- **Secret mount** — injecting a secret at build or runtime without baking it in.
- **SLSA** — Supply-chain Levels for Software Artifacts, a provenance framework.
- **Tag** — a mutable name pointing to an image digest.

## 16. Sources

All URLs accessed 26 October 2026.

- Open Container Initiative. *OCI Image Format Specification*. https://github.com/opencontainers/image-spec
- Open Container Initiative. *Distribution Specification*. https://github.com/opencontainers/distribution-spec
- SLSA. *SLSA Specification v1.2*. https://slsa.dev/spec/v1.2/
- SLSA. *Provenance*. https://slsa.dev/spec/v1.0/provenance
- in-toto. *Specification*. https://in-toto.io/
- SPDX. *SPDX Specification*. https://spdx.dev/
- CycloneDX. *Specification*. https://cyclonedx.org/specification/overview/
- Sigstore. *Signing containers with cosign*. https://docs.sigstore.dev/cosign/signing/signing_with_containers/
- NIST. *Secure Software Development Framework, SP 800-218*. https://csrc.nist.gov/pubs/sp/800/218/final
- NIST. *Application Container Security Guide, SP 800-190*. https://csrc.nist.gov/pubs/sp/800/190/final
- US Food and Drug Administration. *Cybersecurity in Medical Devices: Quality System Considerations and Content of Premarket Submissions*, 2023. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/cybersecurity-medical-devices-quality-system-considerations-and-content-premarket-submissions
- US Food and Drug Administration. *Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions*, 2024, reissued 2025. https://www.fda.gov/media/166704/download
- IEC 62304:2006/AMD1:2015. *Medical device software — Software life cycle processes*. https://webstore.iec.ch/en/publication/22794
- ISO 13485:2016. *Medical devices — Quality management systems*. https://www.iso.org/standard/59752.html
- 21 CFR 11.10, Controls for closed systems. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11/subpart-B/section-11.10
- 45 CFR 164.312, Technical safeguards. https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C/section-164.312
- MITRE. *CVE-2021-44228 (Log4Shell)*. https://nvd.nist.gov/vuln/detail/CVE-2021-44228
- Codecov. *Post-mortem on the Bash Uploader compromise*, 2021. https://about.codecov.io/apr-2021-post-mortem/
- CISA. *SolarWinds Orion supply chain compromise*. https://www.cisa.gov/news-events/cybersecurity-advisories/aa21-008a

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/15_deployment_mlops.md`](../../course/15_deployment_mlops.md)
- Previous topic: [`02_deployment-patterns.md`](02_deployment-patterns.md)
- Next topic: [`04_ci-cd-ct.md`](04_ci-cd-ct.md)
- Related topic: [`09_model-governance.md`](09_model-governance.md)
- Labs overview: [`../../labs/`](../../labs/)
- Reference service: [`../../labs/mlops/`](../../labs/mlops/)
- Dockerfile: [`../../labs/mlops/Dockerfile`](../../labs/mlops/Dockerfile)
- Requirements: [`../../labs/mlops/requirements.txt`](../../labs/mlops/requirements.txt)
- Model artifact: [`../../labs/mlops/app/model.json`](../../labs/mlops/app/model.json)
