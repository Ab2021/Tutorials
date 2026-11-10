---
collection: "python-sql-linux-engineering-basics"
file_id: "04"
topic: "Docker and containers"
level: 2
status: "draft; unreviewed"
version: "2.0"
created: "2026-11-07"
reviewed: null
review_trigger: "Base image family, storage driver, runtime cgroup version, orchestrator probe semantics, or workload shape changes; recheck pinned spec and engine versions before quoting any behaviour"
notation_style: "plain-text"
prerequisites: ["01_Python_Async_and_Concurrency.md", "Working knowledge of Linux processes, signals, and filesystems"]
pattern_tags: ["immutable-artifact", "layer-cache", "capacity-sizing", "supply-chain", "runtime-contract"]
source_status: "OPEN — mapped source title only; original body unavailable; spec semantics described from public OCI/Docker documentation, version specifics unverified"
owner_role: "platform engineer with application owner and security reviewer"
source_index: "INDEX.md"
---

# Docker and Containers

## 1. File Header

> **Purpose:** Treat a container image as a versioned, content-addressed artifact and size the runtime that executes it. The decisions this sheet supports are: which base to inherit, how to order build stages so the cache survives ordinary edits, where secrets and durable state may live, how many workers fit inside a memory limit, how many replicas a latency budget actually requires, and what evidence proves the deployed bytes are the reviewed bytes.
> **Prerequisites:** [Python async and concurrency](01_Python_Async_and_Concurrency.md) for the worker-pool and backpressure vocabulary; comfortable reading a Dockerfile; ability to read `ps`, `kill`, and `df` output; basic arithmetic with MiB versus MB.
> **Key outcome:** Given a Dockerfile, a memory limit, a CPU limit, an arrival rate, a per-request CPU cost, and a latency budget, you can compute the image transfer size, the expected rebuild time under a stated change-rate model, the safe worker count, the replica count that meets the budget, the monthly registry and egress cost, and then name the exact test that would falsify each number.
> **This file does NOT cover:** orchestrator scheduling policy, service-mesh traffic management, image-vulnerability triage workflow, deep kernel hardening (seccomp/LSM profile authoring), or registry product selection. Those are separate handoffs, named in Section 7.
> **Level:** 2 Core Methods.
> **Version / reviewed:** 2.0 / not reviewed.
> **Evidence:** DERIVED arithmetic and invariants; ILLUSTRATIVE workload numbers (every figure below is constructed for teaching, not measured on any system); SOURCED only where a public specification or documented runtime behaviour is named, and then without a pinned version; OPEN where a version-specific limit would need a live check.
> **Retrieval cue:** If a container exits with code 137, if an image "works locally" and fails after deploy, if a rebuild takes four minutes for a one-line change, or if a pod is killed while `docker stats` looks comfortable, start at Section 8.

**Source boundary.** The mapped source entry is `Agents_Stack_Infra/Module_00_Prerequisites/04_docker_and_containers.md`, listed in [`directory.txt`](../../../../Templates/directory.txt). Its body was not available while authoring. The file name therefore establishes a topic, not content: it tells us a lesson about Docker and containers was intended, and nothing about which Dockerfile, registry, cluster, or incident that lesson described. This sheet is a constructed teaching treatment of the subject, and no example below describes a real company, a real repository, or an observed production failure. Where a number looks like a benchmark, it is arithmetic performed on inputs chosen for roundness and labelled ILLUSTRATIVE at the point of use.

The one structural claim inherited from the source map is scope: this is a prerequisites-level engineering topic, not a distributed-systems elective. That is why the emphasis falls on the artifact contract, the layer cache, and capacity arithmetic, and why orchestration is explicitly excluded.

### 1.1 What the reader is actually deciding

Container teaching material tends to collapse into a command catalogue: how to build, how to run, how to copy a file in. A catalogue is easy to write and nearly useless under pressure, because the interesting questions are not "which flag" but "which claim am I entitled to make". Five claims recur in every container review, and each is falsifiable:

1. *The bytes I tested are the bytes I deployed.* Falsifiable by comparing the digest recorded at build time with the digest the runtime reports, not by comparing tag strings.
2. *A clean checkout can reproduce this image.* Falsifiable by building from a fresh clone in a network-restricted environment with the recorded inputs.
3. *No credential is recoverable from the artifact.* Falsifiable by scanning layer contents and build history, not by scanning the Dockerfile text.
4. *The process will fit its limit under peak, not average, load.* Falsifiable by driving peak concurrency with the production limit applied and watching for OOM kills and cgroup throttling.
5. *Traffic reaches the container only when the application can serve it.* Falsifiable by starting the container with a slow dependency and observing whether the load balancer sends requests before readiness.

Everything in Sections 4 through 10 exists to make one of these five claims computable or testable. The formulas are not decoration; each one converts a vague worry ("is it too big?", "will it OOM?", "do we have enough replicas?") into a number with assumptions you can attack.

### 1.2 Why the topic is a prerequisite and not an appendix

A Python or SQL lesson assumes an environment. If that environment is "whatever laptop the reader has", every downstream lesson inherits an unstated dependency set: interpreter patch version, C library, locale, time zone database, installed wheels, and the shell that invoked them. Containers matter in a foundations curriculum because they are the cheapest mechanism that turns an implicit environment into an explicit, hashable one. The cost is a new class of failure: the artifact can now disagree with the source tree, the artifact can carry a secret into a registry, and the runtime limit can kill a process that would have run fine on a larger host.

That trade is the entire subject. A container buys reproducibility and portability by introducing an artifact lifecycle with its own integrity, security, and sizing questions. Readers who skip the lifecycle and learn only `docker run` end up debugging layer caches and OOM kills without a model, which is why Section 2 comes before any formula.

### 1.3 How to read the numbers in this sheet

Three conventions apply throughout, and they are worth stating once so that no table has to repeat them.

- **MiB is 1,048,576 bytes; MB is 1,000,000 bytes.** Image sizes are quoted in MiB because registries and `docker images` report binary units under a decimal-looking label. A "164 MB" image in a UI is usually 164 MiB, and mixing the two conventions produces a 4.86% error that compounds across pull-time and egress calculations.
- **Compressed versus uncompressed.** A registry transfers compressed layers; the local store and the container filesystem hold uncompressed content. The ratio depends entirely on content (text compresses, already-compressed wheels do not). The illustrative model in this sheet assumes the quoted size is the transferred size and says so; treating compressed and uncompressed bytes as interchangeable is a common source of wrong capacity estimates.
- **Limits are ceilings, not reservations.** A 2 GiB memory limit means the kernel kills at 2 GiB; it does not mean 2 GiB is available from other tenants, nor that the container will use 2 GiB. A 1.5 vCPU limit means a quota over a period, so a container can burst within the period and be throttled afterwards. Every sizing formula in Section 4 therefore needs a utilization target separate from the limit.

**Takeaway:** The reader's job is not to remember flags but to keep five falsifiable claims alive — same bytes, reproducible build, no recoverable credential, fits at peak, traffic only when ready — and every later section is machinery for one of those claims.

## 2. Concept Map / Mental Model

### 2.1 The two contracts

A container deployment is governed by two contracts that are frequently confused. The **build contract** says: given this context directory, this Dockerfile, these pinned base digests, and this dependency lock, produce exactly one image manifest, and that manifest has a digest. The **runtime contract** says: given this image digest, this command, this environment schema, these mounts, these limits, and this network attachment, produce one process tree with defined lifecycle, signal, and health semantics.

The failure mode created by confusing them is the phrase "it works locally". Locally, both contracts were satisfied on one machine at one moment. A deploy re-executes the runtime contract on a different kernel, architecture, cgroup version, and configuration, and may execute a *different* image because someone re-tagged. The build contract is not implicated at all. Separating the two tells you which evidence to collect: for a "works locally" failure you diff runtime inputs and image digests; you do not re-read the Dockerfile.

```text
BUILD CONTRACT                                  RUNTIME CONTRACT
context dir ─┐                                  image digest ─┐
Dockerfile ──┼─> build ─> manifest ─> digest    command ──────┤
base digests ┤              │                   env schema ───┼─> container
lockfile ────┘              │                   mounts ───────┤     │
                            v                   limits ───────┘     │
                       registry (tags are                            v
                       mutable pointers)                     namespaces (view)
                            │                                cgroups   (share)
                            └──── pull by digest ──────────> LSM/seccomp (filter)
                                                                   │
                                                                   v
                                                            health signals
                                                            (live / ready)
```

The left column is a pure function of hashed inputs; the right column is a process under constraint. The registry sits between them and is the only mutable element in the picture, which is why the digest — not the tag — crosses the boundary.

### 2.2 Lifecycle map

The following map is the one to keep in mind when debugging. Each arrow is a distinct place where a claim can break, and each place needs a different observation.

```text
[source tree]
     |  (1) context tar: .dockerignore decides what is even visible
     v
[build steps] --(2) cache key per instruction--> [layer blobs]
     |                                                |
     |  (3) config + manifest assembled               |
     v                                                v
[image manifest] ------------------------------> [content-addressed store]
     |  (4) tag assignment (mutable, human-meaningful)
     v
[registry] --(5) pull: bytes already local are skipped
     |
     v
[unpack layers read-only] --(6) overlay mount + writable upper dir
     |
     v
[process started] --(7) namespaces, cgroups, seccomp, capabilities applied
     |
     +--> (8) probe traffic: is it serving?  --> no: stay out of endpoints
     |
     +--> (9) SIGTERM on shutdown: drain, exit 0 within grace, else SIGKILL
     |
     v
[container removed: writable upper dir discarded unless it was a volume]
```

Nine arrows, nine distinct evidence sources: build log, layer digests, manifest, tag resolution, registry transfer log, mount table, cgroup files, probe status, and exit code. A debugging session that does not name which arrow failed will wander.

### 2.3 Where identity lives, and what each identifier is good for

There are four identifiers in common use and they are not interchangeable:

| Identifier | What it names | Stable? | Correct use | Wrong use |
|---|---|---|---|---|
| Tag (`app:1.4.2`, `app:latest`) | A human-chosen pointer into a registry | No — can be moved to any manifest | Discovery, bookkeeping, release notes | Proving what ran; pinning a deploy |
| Digests (`sha256:...`) | Exact manifest or blob bytes | Yes, by construction | Deploy pin, audit trail, cache key | Reading by a human |
| Image config ID | The config JSON (env, cmd, entrypoint, labels) | Yes | Inspecting declared settings | Distinguishing two builds with identical config |
| Container ID | One running instance | Unique per instance | Logs, exec, metrics | Anything about the artifact |

The practical consequence: a release record must contain the digest, and a dashboard that shows only tags cannot answer "which build produced this error rate". A registry that allows a tag to be overwritten silently converts an audit trail into a rumour.

### 2.4 Central invariant

The invariant that makes the whole model work is **content addressability**: an object's name is a cryptographic hash of its bytes, so identical bytes have identical names and modified bytes cannot keep the old name. Layer deduplication, local pull skipping, and build-cache validity all descend from this one property. When you see "the pull was instant because layers were cached", you are seeing content addressability; when you see "the tag pointed at a different image than we thought", you are seeing the one place the property was not used.

A second invariant governs the runtime side: **the container's root filesystem is a stack of read-only layers plus one writable layer, and nothing in a read-only layer can change while the container runs.** Writes go to the writable layer (or to a mounted volume, which is not part of the stack). This is why "state the container wrote" disappears when the container is removed, and why writing a large file that already exists in a lower layer costs a copy of the whole file before the first byte is modified.

### 2.5 What changes between variants, and the most important boundary

The variants in Section 7 are variations on four knobs: what the base provides, how many build stages exist, what the runtime mounts, and how the process is supervised. Changing the base changes the C library, the shell, the package manager, and the attack surface simultaneously — which is why "just switch to a smaller base" is not a safe refactor. Changing the number of stages changes image size and the availability of build tools at build time. Changing mounts changes durability and performance. Changing supervision changes signal delivery and zombie reaping.

The most important boundary is the one the marketing language obscures: **a container is not a security boundary the way a virtual machine is.** It is a process on the host kernel with a restricted view (namespaces), a resource envelope (cgroups), and syscall/capability filtering. All containers on a host share one kernel. A kernel vulnerability reachable from a container is reachable from the host. This does not make containers unsafe; it makes the threat model different, and it makes claims like "we isolate untrusted code in a container" incomplete unless the kernel attack surface is also reduced.

**Takeaway:** Model the system as two contracts joined by a content-addressed artifact, debug by naming which of the nine lifecycle arrows failed, and never let a mutable tag stand in for the identity of the bytes you shipped.

## 3. Foundational Definitions

Definitions below are ordered by dependency, not alphabetically: you need *image*, *layer*, and *digest* before *cache*, and *namespace* and *cgroup* before *limit* and *probe* can be stated precisely.

### 3.1 Core terms

| Term | Formal meaning | Units / shape | Valid domain | Common confusion | Where it bites |
|---|---|---|---|---|---|
| Image | An ordered set of content-addressed filesystem layers plus a JSON config plus a manifest describing both | Manifest (JSON) + `n` layer blobs | Immutable once built | Called a "VM image" | Identity of a release |
| Layer | One tar archive of filesystem changes, named by the digest of its compressed content | Bytes, unique per digest | Read-only when mounted | Confused with a Dockerfile line | Size, cache, dedup |
| Container | One process tree executing with a chosen root filesystem, namespaces, cgroups, and security profile | Live kernel object | Exists only while running (or stopped-but-present) | Used as a synonym for image | State loss, exit codes |
| Digest | Lowercase algorithm prefix plus hex hash of exact bytes, e.g. `sha256:` followed by 64 hex characters | 32 bytes / 64 hex chars (sha256) | Any content-addressed object | Treated as a version number | Pinning, cache keys |
| Tag | A registry-scoped, mutable mapping from a string to a manifest | String → manifest | May be reassigned | Treated as an identity | Silent drift |
| Build context | The file tree sent to the builder, filtered by ignore rules | Bytes / file count | Must be reproducible | Assumed to be the repository root | Leaks, slow builds |
| Instruction | One Dockerfile step producing zero or one layer plus config metadata | Line / stage | Deterministic if inputs pinned | Assumed 1:1 with layers | Cache granularity |
| Cache key | The hash of an instruction plus resolved inputs (parent layer, copied file contents, arguments) | Hash | Invalidated on any input change | Assumed time-based | Rebuild cost |
| Writable layer | The single per-container read-write layer in the overlay stack | Bytes, grows at runtime | Discarded on removal | Assumed durable | Data loss, copy-up cost |
| Volume | Storage mounted into the container from outside the overlay stack, with an independent lifecycle | Bytes/IOPS | Survives container removal | Confused with writable layer | Durability, permissions |
| Bind mount | A host path made visible inside the container, sharing the host's inode and ownership | Path | Host-dependent | Assumed portable | UID mismatch |
| Namespace | A kernel view-isolation object (PID, mount, network, UTS, IPC, user, cgroup, time) | Kernel object | Per-container or shared | Confused with a security boundary | "Works locally", DNS |
| Cgroup | A kernel resource-accounting and limiting hierarchy (CPU, memory, IO, PIDs) | Hierarchy + controller files | cgroup v1 or v2 | Confused with a quota of reserved capacity | OOM, throttling |
| Memory limit | A hard cgroup ceiling whose breach triggers reclaim then OOM kill | Bytes | Kernel-enforced | Read as "guaranteed memory" | 137 exits |
| CPU limit | A quota over a period plus a weight for contention, not a pinned core | Quota/period (cores) | Kernel-enforced | Read as "reserved core" | Throttling, latency tails |
| Entrypoint | The executable that always runs; arguments from `CMD` or the run command are appended | argv prefix | Exec form recommended | Confused with `CMD` | Signals, arg handling |
| CMD | Default argument vector (or default command if no entrypoint) | argv | Overridable at run time | Assumed to be the command | Silent override |
| Readiness | Whether the instance should receive traffic now | Boolean signal | Dynamic | Merged with liveness | Traffic to a booting app |
| Liveness | Whether the instance should be restarted | Boolean signal | Dynamic | Merged with readiness | Restart loops |
| Registry | A content-addressable blob store with tag resolution and access control | Service | External dependency | Assumed to be a backup | Availability, egress cost |
| OCI | Vendor-neutral specifications for image format and runtime behaviour | Spec | — | Assumed to define security | Scope overreach |

### 3.2 The distinctions that actually decide arguments

Six pairs are routinely conflated in review, and each conflation has a concrete consequence.

**Image versus container.** The image is bytes on disk; the container is a running process tree whose root filesystem is that image plus a writable layer. Deleting the image does not stop a running container (the layers are in use); stopping a container does not modify the image. Consequence: "restart the container to fix it" restores the image's state but discards writable-layer state, which either fixes the problem (state was corrupt) or destroys evidence (state was the only record of what happened).

**Tag versus digest.** A tag is a name a human typed; a digest is a claim about bytes. Consequence: a rollback instruction phrased as "redeploy `app:previous`" is not a rollback unless `previous` is immutable or the digest was recorded.

**Volume versus writable layer.** Both are writable at run time; only the volume has a lifecycle independent of the container, and only the volume can be shared between containers. Consequence: a database whose data directory is in the writable layer loses everything on container replacement — and the symptom, "the data vanished after a routine deploy", looks like a bug in the database.

**`ENTRYPOINT` versus `CMD`.** With the exec form, `ENTRYPOINT` is the fixed executable and `CMD` supplies default arguments. With shell form, the shell becomes PID 1 and may swallow signals. Consequence: a Java or Python service that takes 30 seconds to shut down and always hits `SIGKILL` usually has a shell wrapping problem, not a slow shutdown problem.

**Readiness versus liveness.** Readiness gates traffic; liveness triggers restart. Consequence: wiring liveness to a dependency check converts a dependency outage into a cluster-wide restart storm, because every instance fails liveness at once and restarts into the same outage.

**Image size versus memory footprint.** Compressed image bytes and resident memory are different quantities with different optimisers. Consequence: a team shrinks an image by 60% and confidently lowers the memory limit, then OOMs, because page cache and interpreter overhead never lived in the image.

### 3.3 Definitions that are load-bearing for the formulas

Section 4 uses a small symbol set. Defining it here prevents later ambiguity:

```text
DEFINITIONS
  s_d        bytes of layer blob d (compressed, as transferred)   [bytes]
  D(I)       set of layer digests referenced by image I           [set]
  beta       effective pull bandwidth for one image               [MiB/s]
  m          fixed pull overhead: auth, manifest, unpack setup    [s]
  t_start    process start to serving traffic                     [s]
  q_i        probability that stage i's inputs differ from cache  [dimensionless]
  c_i        duration of stage i when it must execute             [s]
  L          container memory limit                               [MiB]
  b          baseline resident memory before serving load         [MiB]
  rho        fractional reserve held back from the limit          [dimensionless]
  w          per-worker steady-state peak memory                  [MiB]
  lambda     arrival rate                                         [requests/s]
  c_cpu      CPU service time per request                         [core-seconds]
  C          CPU limit per replica                                [cores]
  u_star     target CPU utilization                               [dimensionless]
  S          mean service time observed by the queue              [s]
  D_budget   end-to-end latency budget                            [s]
```

Every formula block in Section 4 restates these symbols with units, per the course standard, because a copied formula must not lose its units.

### 3.4 Evidence vocabulary used in this sheet

`DERIVED` marks a result that follows from stated premises by algebra or by a named theorem — the queueing and sizing results are of this kind, and their premises are stated each time. `ILLUSTRATIVE` marks an input chosen for teaching; the arithmetic on it may still be independently checked, and the checks are shown. `SOURCED` marks a statement attributed to a public specification or documented runtime behaviour without re-verification here; the document is named, the version is not pinned, and the reader is told to recheck. `HEURISTIC` marks an engineering rule of thumb with a rationale and a known limitation. `OPEN` marks a gap that would need a live environment to close, such as a specific kernel's lower-layer limit or an engine version's default storage driver.

The distinction matters because a number can be arithmetically correct and empirically meaningless. `164 MiB / 40 MiB/s = 4.1 s` is verified arithmetic on an illustrative bandwidth; it is not a measurement of any registry. Reporting it as "pull time" without the label is the error this vocabulary exists to prevent.

**Takeaway:** Almost every container argument is a collision between a name and an identity (tag/digest), a view and a limit (namespace/cgroup), or a byte count and a memory count (image size/footprint); define which one you mean before computing.

## 4. Core Formulas with Full Derivation

The formulas below are grouped by the claim they support. 4.1 derives the three base quantities (transfer, rebuild, workers) from first principles with the simplest possible model. 4.2 generalises to shared registries, multi-replica queues, and deduplicated stores. 4.3 covers variants that change one assumption at a time. 4.4 shows the boundaries where each formula stops being usable.

### 4.1 Base case: one image, one host, one container

#### 4.1.1 Image transfer time

**Validity before use:** The image is not already present in the local store, the registry serves layers at a roughly constant effective bandwidth, layers are transferred sequentially or the bandwidth is the aggregate, and per-layer round-trip overhead is constant. Invalid if the client already has some layers (see the general case), if the network is highly variable, or if decompression is the bottleneck rather than transfer.

**Derivation.** Content addressability means the set of blobs an image needs is exactly the set of digests its manifest references. Let `D(I)` be that digest set and `s_d` the transferred size of blob `d`. Transferring all of them at effective bandwidth `β` takes `Σ s_d / β`. Add a fixed cost `m` for authentication, manifest fetch, and connection setup, and a per-layer unpack cost `t_unpack` that the runtime pays after bytes arrive. Summing gives total time to a mounted root filesystem. The model is additive, not overlapping, which makes it conservative for pipelines but accurate for a naive client that unpacks as it goes.

```text
FORMULA [DC-01]: T_pull = (Σ_{d ∈ D(I)} s_d) / β + m + Σ_{d ∈ D(I)} t_d

WHERE:  D(I)      set of layer digests referenced by the image manifest       [set]
        s_d       transferred (compressed) size of blob d                     [MiB]
        β         effective sustained pull bandwidth for this client          [MiB/s]
        m         fixed overhead: auth, manifest, TLS, connection setup       [s]
        t_d       unpack cost for layer d (write + extract)                   [s]

CONSTRAINT: cold local store; β approximately constant; sequential transfer;
            s_d measured as transferred bytes, not uncompressed bytes; no
            concurrent pulls competing for the same link.

NUMERICAL EXAMPLE:
        s = 164 MiB total across 5 layers; β = 40 MiB/s; m = 0.15 s;
        t_d = 0.05 s each → Σ t_d = 0.25 s.
        T_pull = 164/40 + 0.15 + 0.25 = 4.10 + 0.40 = 4.50 s.

INTERPRETATION: On a 40 MiB/s link this image costs 4.5 s before any process
        starts. Shrinking the image to 100 MiB saves 64/40 = 1.6 s of pull,
        which is a startup-latency argument, not a memory argument.
```

**Check.** Dimensions: MiB ÷ (MiB/s) = s, and the two overhead terms are already seconds, so the sum is homogeneous. Limiting behaviour: as `β → 0`, `T_pull → ∞` and the image is unusable regardless of size; as `s → 0`, `T_pull → m + Σ t_d`, which is the floor a "distroless" image cannot beat.

#### 4.1.2 Expected rebuild time under a change-rate model

**Validity before use:** Each build stage's inputs change independently with probability `q_i` per build; a stage re-executes if and only if its own inputs changed or any earlier stage re-executed; stages execute sequentially; `c_i` is the measured cost of executing stage `i` when it does run. Invalid under parallel stage execution, remote shared caches that skip based on content without ordering, or when a stage's cost depends on which earlier stages reran.

**Derivation.** The event "stage `i` executes" is the complement of "stage `i` and every stage before it hit the cache". Because the inputs are independent across stages, `P(stage 1..i all hit) = Π_{j≤i} (1 − q_j)`. Therefore `P(stage i executes) = 1 − Π_{j≤i}(1 − q_j)`. Cost contribution is that probability times `c_i`, and linearity of expectation sums the contributions. The critical structural consequence is that the product term is *cumulative*: a volatile early stage poisons every later stage, which is why instruction order matters more than instruction count.

```text
FORMULA [DC-02]: E[T_rebuild] = Σ_{i=1..n} [ 1 − Π_{j=1..i} (1 − q_j) ] · c_i

WHERE:  n     number of build stages (instructions that can cache)           [count]
        q_i   probability that stage i's inputs differ from the cached value  [0..1]
        c_i   wall-clock cost of executing stage i when it runs               [s]
        Π     product over stages 1..i                                        [dimensionless]

CONSTRAINT: independent per-stage change events; strict ordering; a stage runs
            iff its own inputs or any predecessor's inputs changed; c_i constant.
            Cache misses that propagate backwards (e.g. base digest rotation)
            must be modelled by setting q_base = 1 for that build.

NUMERICAL EXAMPLE (three stages: dependency install, wheel install, copy source):
        Order B (bad): q = (0.01, 0.05, 0.90), c = (60, 180, 5) s
          stage 1: 1−0.99              = 0.0100 → 0.60 s
          stage 2: 1−(0.99·0.95)=0.0595 → 10.71 s
          stage 3: 1−(0.99·0.95·0.10)=0.90595 → 4.53 s
          E[T] = 0.60 + 10.71 + 4.53 = 15.84 s
        Order A (good): q = (0.90, 0.05, 0.01), c = (5, 180, 60) s
          stage 1: 0.900 → 4.50 s
          stage 2: 1−(0.10·0.95)=0.905 → 162.90 s
          stage 3: 1−(0.10·0.95·0.99)=0.90595 → 54.36 s
          E[T] = 4.50 + 162.90 + 54.36 = 221.76 s

INTERPRETATION: The same three stages cost 15.8 s in one order and 221.8 s in
        the other — a 14× difference produced purely by ordering. Copy the
        volatile source last; install dependencies first.
```

**Check.** Set all `q_i = q`. Then `P(i executes) = 1 − (1−q)^i`, which is increasing in `i` and tends to 1: with any nonzero change rate, later stages eventually always run. Verified for `n = 3, q = 0.5`: `Σ [1 − 0.5^i] = 0.5 + 0.75 + 0.875 = 2.125`, and the closed form `n − (1−q)(1−(1−q)^n)/q = 3 − 0.5(1−0.125)/0.5 = 3 − 0.875 = 2.125`. The two agree.

**Boundary.** If `q_1 = 1` (for example, an unpinned base that is re-resolved every build), every stage executes every time and `E[T] = Σ c_i`; ordering becomes irrelevant and the correct fix is pinning, not reordering.

#### 4.1.3 Safe worker count inside a memory limit

**Validity before use:** Worker memory is roughly constant at steady state, the baseline is additive with the workers, the limit is hard (OOM kill at breach), and the reserve is expressed as a fraction of the limit. Invalid when a single request can allocate an unbounded buffer, when memory grows with concurrency nonlinearly, or when the kernel's page cache for that cgroup is charged against the same limit (it usually is, and reclaimable, but under pressure it still affects latency).

**Derivation.** Let `L` be the cgroup limit, `b` the baseline footprint (interpreter, libraries, allocator arenas, connection pools) measured with zero in-flight work, and `w` the marginal peak memory per worker. Two things must be subtracted before dividing: the baseline, because it is not optional, and a reserve `ρL`, because steady-state measurement under-reports the transient peak (garbage that has not been collected, a burst of larger payloads, allocator fragmentation). The remaining budget divided by `w` and floored gives the count that cannot exceed the limit *under the measurement conditions*. The floor is essential: a fractional worker does not exist, and rounding up silently spends reserve you just set aside.

```text
FORMULA [DC-03]: N_max = floor( (L·(1 − ρ) − b) / w )

WHERE:  L    container memory limit (cgroup hard limit)                     [MiB]
        ρ    fractional reserve for transients and fragmentation           [0..1]
        b    baseline RSS with zero in-flight work                         [MiB]
        w    marginal peak RSS per additional worker                       [MiB]
        floor  round down to an integer worker count                       [count]

CONSTRAINT: worker memory independent and additive; limit is hard; reserve
            chosen before measurement and not tuned to make the answer nice;
            w measured at the payload's upper decile, not its mean.

NUMERICAL EXAMPLE: L = 2048 MiB, ρ = 0.15, b = 1080 MiB, w = 130 MiB.
        reserve      = 0.15 × 2048      = 307.2 MiB
        budget       = 2048 − 307.2 − 1080 = 660.8 MiB
        N_max        = floor(660.8 / 130) = floor(5.083) = 5 workers
        memory used  = 1080 + 5×130 = 1730 MiB → 84.5% of limit.
        Naive answer = floor(2048 / 130) = 15 workers, which would need
        1080 + 15×130 = 3030 MiB, i.e. 982 MiB more than the limit allows.

INTERPRETATION: The naive division over-provisions by 3× because it treats the
        baseline as free and the limit as usable capacity. Five workers with
        318 MiB of headroom is the defensible number; 15 is an OOM kill.
```

**Check.** Units: MiB ÷ MiB = dimensionless count. Limits: if `b ≥ L(1−ρ)`, the formula returns 0 or negative, which correctly means the baseline alone does not fit — a real condition worth checking before tuning workers. If `ρ = 0` and `b = 0`, it reduces to `floor(L/w)`, the naive form, confirming the naive form is the special case that ignores both corrections.

### 4.2 General case: shared registry, many replicas, deduplicated store

#### 4.2.1 Warm-start pull with partial layer presence

**Validity before use:** The local store retains some layers from previous images; layer identity is by digest so a retained layer is byte-identical; the transfer is only the missing subset.

**Derivation.** Because the pull set is the digest set difference `D(I) \ D(local)`, and because content addressing guarantees that a digest present locally has exactly the needed bytes, the transfer cost is the sum over the difference only. This is the derivation of "the second deploy was fast": nothing changed except `D(local)`. The general form also explains a counter-intuitive measurement — after a base-image security update, *every* layer above the base changes, because the base's digest changed and each subsequent layer's parent changed with it.

```text
FORMULA [DC-04]: T_warm = (Σ_{d ∈ D(I) \ D(local)} s_d) / β + m + Σ_{d ∈ D(I) \ D(local)} t_d

WHERE:  D(local)  digests already present in the local content store          [set]
        \         set difference (layers still needed)                        [set]
        other symbols as DC-01                                                —

CONSTRAINT: local store trusted (assume it is — a corrupted store is a distinct
            failure, see F-05); digests comparable across registries; the
            manifest itself is always fetched, so m > 0 even at zero misses.

NUMERICAL EXAMPLE: same 164 MiB image, but the app layer (4 MiB) changed and
        the dependency layer (96 MiB) did not:
        missing = 4 MiB → T_warm = 4/40 + 0.15 + 0.05 = 0.10 + 0.20 = 0.30 s.
        Base-image rotation: all 5 layers missing → back to 4.50 s.

INTERPRETATION: Rebuild-and-redeploy time is dominated by which layer the change
        lands in. Put application code in the last, smallest layer; put a
        security-patched base at the bottom, accepting the full re-pull.
```

#### 4.2.2 Replica count from a CPU budget

**Validity before use:** Requests are CPU-bound in steady state, `c_cpu` is a measured mean at the target payload mix, the target utilisation `u*` is below the knee where queueing explodes, and each replica's CPU limit `C` is enforced. Invalid for IO-bound services (then the bottleneck is elsewhere), for pathological payloads that dominate the mean, and when `u*` is chosen so high that queueing, not capacity, determines latency.

**Derivation.** Total CPU demand per second is `λ·c_cpu` core-seconds per second, i.e. cores. Usable capacity per replica is `C·u*` cores, because a replica held at 100% of its quota has no burst room and will be throttled. Dividing demand by usable capacity gives the number of replicas; the ceiling is required because replicas are integral and a fractional replica cannot absorb the remaining demand.

```text
FORMULA [DC-05]: n = ceil( λ · c_cpu / (C · u*) )

WHERE:  λ       arrival rate                                               [req/s]
        c_cpu   CPU service time per request, mean at target mix        [core·s/req]
        C       CPU limit per replica                                   [cores]
        u*      target utilisation of that limit                        [0..1]
        n       replica count                                           [count]

CONSTRAINT: CPU is the binding resource; steady state; c_cpu measured, not
            estimated from code reading; u* < 1 with the gap reserved for
            burst and for throttling avoidance. Not a latency guarantee —
            use DC-06 for that.

NUMERICAL EXAMPLE: λ = 45 req/s, c_cpu = 0.09 core·s, C = 1.5 cores, u* = 0.7.
        demand   = 45 × 0.09 = 4.05 cores
        usable   = 1.5 × 0.7 = 1.05 cores per replica
        n        = ceil(4.05 / 1.05) = ceil(3.857) = 4 replicas
        resulting utilisation = 4.05 / (4 × 1.5) = 67.5%
        spare capacity        = 4 × 1.5 / 0.09 × 0.7 = 46.67 req/s

INTERPRETATION: Four replicas leave 1.67 req/s of headroom above 45 req/s at the
        target utilisation — about 3.7%. Any traffic increase beyond that needs
        a fifth replica or a lower c_cpu; the arithmetic says the margin is thin.
```

**Check.** Dimensions: (req/s)·(core·s/req) = cores; cores ÷ cores = 1. Limits: as `u* → 1`, `n → ceil(λc/C)`, the theoretical minimum, achievable only with no queueing variance; as `c_cpu → 0`, `n → 1`, and the formula stops describing anything because a single replica serving infinite throughput is not a real regime.

#### 4.2.3 Latency from a multi-server queue (Erlang C)

**Validity before use:** Poisson arrivals, exponential service times, `c` identical servers, one shared queue, no abandonment, no retries, infinite buffer, and steady state with `λ < cμ`. Invalid for bursty or self-similar traffic, for services whose latency is dominated by a downstream dependency, and whenever utilisation exceeds roughly 85% — because the model's own variance assumptions then understate real tails. It is a *capacity-planning* model, not a tail-latency predictor.

**Derivation.** The M/M/c system is a birth–death chain whose stationary distribution exists when the offered load `a = λ/μ` is below `c`. From the stationary probabilities, the probability that an arrival finds all servers busy is the Erlang C function. Conditioning on that event gives the expected wait as `C(c,a)/(cμ − λ)`: when all servers are busy, the queue drains at the surplus rate `cμ − λ`, and the mean number of waiting customers divided by the arrival rate yields the mean wait by Little's law. The formula is a two-step composition — `C` for the probability of queueing, then the drain rate for the duration.

```text
FORMULA [DC-06]: W_q = C(c, a) / (c·μ − λ),      W = W_q + 1/μ
                 C(c, a) = [ a^c/(c!)]·[c/(c−a)] / ( Σ_{k=0..c−1} a^k/k! + [a^c/(c!)]·[c/(c−a)] )

WHERE:  λ    arrival rate                                                 [req/s]
        μ    per-server service rate = 1/S                               [req/s]
        a    offered load = λ/μ                                          [erlang]
        c    number of identical servers                                 [count]
        C    Erlang C probability that an arrival queues                 [0..1]
        W_q  mean queueing delay                                          [s]
        W    mean sojourn time (wait + service)                           [s]

CONSTRAINT: Poisson arrivals, exponential service, homogeneous servers,
            λ < cμ (else no stationary distribution), infinite buffer,
            work-conserving discipline. Violations bias W downward.

NUMERICAL EXAMPLE: λ = 45 req/s, S = 1/16.667 = 0.06 s → μ = 16.667 req/s,
        c = 4 → a = 2.7, ρ = a/c = 0.675.
        partial sums: 1 + 2.7 + 3.645 + 3.2805 = 10.6255   (k = 0..3)
        tail term:    2.7^4/4! × 4/(4−2.7) = 2.21434 × 3.07692 = 6.8128
        C(4, 2.7)  = 6.8128 / (10.6255 + 6.8128) = 6.8128/17.4383 = 0.3907
        W_q = 0.3907 / (4×16.667 − 45) = 0.3907 / 21.667 = 0.01803 s = 18.0 ms
        W   = 18.0 + 60.0 = 78.0 ms.
        Same λ with c = 3 (ρ = 0.9):  W_q = 163.4 ms, W = 223.4 ms.

INTERPRETATION: Dropping from four replicas to three keeps utilisation "legal"
        at 90% but multiplies mean sojourn time by 2.86× (223.4/78.0). The
        fourth replica buys latency stability, not throughput.
```

**Check.** Little's law independently: `L = λW = 45 × 0.078 = 3.51` requests in system, of which `45 × 0.06 = 2.7` are in service (matching `a`) and `0.81` are waiting. Both are consistent with `L_q = λW_q = 45 × 0.01803 = 0.811`. The numbers cross-validate. Limit: as `ρ → 1`, `W_q → ∞`; at `ρ = 0.9` with `c = 4` the model already reports 118 ms, and real systems with any burstiness report more.

#### 4.2.4 Deduplicated store size across a tag set

**Validity before use:** Layers are identified by digest, the registry counts unique blobs once, and no per-manifest overhead dominates.

**Derivation.** Each manifest references a set of digests. The union over manifests is what must be stored; the sum over manifests is what would be stored without deduplication. Their difference is the saving, and it equals the total size of blobs referenced by more than one manifest, counted `(k−1)` times for `k` references. This is why "we have forty images" usually means far less than forty image-sizes.

```text
FORMULA [DC-07]: S_stored = Σ_{d ∈ ∪_m D(m)} s_d,     S_naive = Σ_m Σ_{d ∈ D(m)} s_d
                 S_saved  = S_naive − S_stored

WHERE:  m          image manifest (one per tag)                                [index]
        D(m)       its layer digest set                                        [set]
        ∪_m        union across manifests                                      [set]
        s_d        blob size                                                   [MiB]

CONSTRAINT: digest-addressed store, no re-compression per manifest, blob
            retention not enforced separately. Registry-specific metadata and
            index objects excluded.

NUMERICAL EXAMPLE: three service images sharing a common base and runtime:
        api-1: 30 + 12 + 22 + 96 + 4.0  = 164.0 MiB
        api-2: 30 + 12 + 22 + 96 + 4.2  = 164.2 MiB
        worker:30 + 12 + 22 + 88 + 3.0  = 155.0 MiB
        S_naive  = 483.2 MiB
        S_stored = 30 + 12 + 22 + 96 + 88 + 4.0 + 4.2 + 3.0 = 259.2 MiB
        S_saved  = 224.0 MiB = 46.4% of naive.

INTERPRETATION: Deduplication turns three near-copies into roughly 1.6 image
        equivalents. The saving is a property of shared *prefix* layers, so it
        disappears the moment each service builds its own base.
```

### 4.3 Special cases and variants

#### 4.3.1 Copy-up cost when modifying a file from a lower layer

**Validity before use:** Overlay-style stacking, file exists in a read-only lower layer, modification is in place, and the driver performs a full-file copy before the first write.

**Derivation.** A read-only lower layer cannot be modified, so a write triggers a copy of the entire file into the writable layer, after which writes are local. For `k` distinct files of size `s` each rewritten `j` times, cost is `k·s` once (the first copy), not `k·s·j` — subsequent writes hit the copy already in the upper layer. For an append-only log the file is copied once, then grows. The practical consequence is that rewriting a large file that came from the image is expensive in a way that creating a new file is not.

```text
FORMULA [DC-08]: W_copyup = Σ_{f ∈ F} s_f · [j_f ≥ 1]

WHERE:  F     set of files modified at run time                             [set]
        s_f   file size                                                     [MiB]
        j_f   number of modification episodes (only the first copies)       [count]
        [·]   Iverson bracket: 1 if the condition holds, else 0             [0/1]

CONSTRAINT: copy-up happens on first write per mount lifetime; file is fully
            materialised; sparse-file optimisation and reflink support can
            reduce it. Applies to the writable layer only, not volumes.

NUMERICAL EXAMPLE: a 30 MiB SQLite file shipped inside the image, modified 200
        times by a benchmark loop: W_copyup = 30 × 1 = 30 MiB for the layer,
        plus 200 × 30 MiB = 6000 MiB = 5.86 GiB of *logical* write traffic to
        the writable layer if each episode rewrites the whole file after a
        remount; on a single mount, one copy plus 200 incremental writes.

INTERPRETATION: Never put a hot database file in the image. The first write
        copies it, and any per-request remount pattern converts a 30 MiB file
        into gigabytes of writable-layer churn.
```

#### 4.3.2 Health-probe detection window

**Validity before use:** Probes run on a fixed period with a per-probe timeout and a consecutive-failure threshold; the orchestrator acts only after the threshold is reached.

**Derivation.** A failure that occurs immediately after a successful probe is detected at the earliest after one period (next probe), and at the latest after `threshold` periods plus one timeout if each probe must time out before counting. So the detection interval is bounded below by `period + timeout` and above by `threshold·period + timeout`. This window is a *design parameter*: it must be longer than the worst legitimate slow response and shorter than the harm you can absorb while serving from a broken instance.

```text
FORMULA [DC-09]: T_detect ∈ [ P + τ , k·P + τ ]

WHERE:  P   probe period                                                   [s]
        τ   probe timeout                                                  [s]
        k   consecutive failures required                                  [count]

CONSTRAINT: probe result independent per attempt; failures persistent (a
            flapping dependency yields a longer effective window); action
            taken by the orchestrator, not by the application.

NUMERICAL EXAMPLE: P = 10 s, τ = 3 s, k = 3 → T_detect ∈ [13 s, 33 s].
        At 45 req/s, up to 33 × 45 = 1485 requests can be served by an
        instance that is already failing liveness.

INTERPRETATION: Tighten P for faster detection, but P must exceed the p99
        probe latency or healthy instances restart. Detection cost is measured
        in requests served by a dead instance, so state it in those units.
```

#### 4.3.3 CPU throttling from bursty demand against a quota

**Validity before use:** CFS-style quota over a fixed period, demand concentrated within a single period, no burst credit available, single-threaded or few-threaded work that cannot spread.

**Derivation.** A CPU limit `C` is enforced as `quota/period`. If `d` requests each needing `c_cpu` arrive within one period, demand in that period is `d·c_cpu`; the fraction of the period the cgroup is runnable but throttled is `max(0, d·c_cpu − C·period) / (d·c_cpu)`. Averaged over many periods with uniform arrivals this vanishes, which is why throttling shows up as tail latency rather than reduced throughput.

```text
FORMULA [DC-10]: f_throttled = max(0, d·c_cpu − C·P) / (d·c_cpu)

WHERE:  d        requests arriving in one period                           [count]
        c_cpu    CPU time per request                                   [core·s]
        C        CPU quota in cores                                         [cores]
        P        quota period                                               [s]
        f        fraction of demanded CPU time that was throttled        [0..1]

CONSTRAINT: single period, no carry-over burst credit, demand known; the
            average over long windows will look fine while tails suffer.

NUMERICAL EXAMPLE: P = 0.1 s, C = 0.2 cores (quota 20 ms), 50 requests of
        4 ms CPU arrive in one 100 ms window:
        demand = 50 × 0.004 = 0.200 core·s; allowance = 0.2 × 0.1 = 0.020 core·s
        f = (0.200 − 0.020)/0.200 = 0.90 → 90% of that window's CPU demand was
        throttled, adding up to 180 ms of delay to the burst.

INTERPRETATION: A container averaging 20% CPU can still be throttled 90% of a
        burst window. Average utilisation is not evidence of adequate quota.
```

### 4.4 Asymptotic behaviour and counterexamples

| Result | Required assumption | Diagnostic that exposes violation | Violating example | Correct alternative |
|---|---|---|---|---|
| DC-03 gives a safe worker count | Additive, bounded per-worker memory | Run at `N_max` with upper-decile payloads; watch `memory.events` `oom_kill` and RSS | A batch job whose per-item buffer scales with input size | Bound the buffer; process in chunks; cap `N` by token budget instead |
| DC-05 gives adequate replicas | CPU is binding | Compare CPU utilisation against queue depth: low CPU with rising latency means CPU is not the bottleneck | Service waiting on a database with a connection-pool cap | Size the downstream pool, then re-derive |
| DC-06 predicts latency | Poisson arrivals, exponential service | Measure the coefficient of variation of inter-arrival times; `CV² ≫ 1` invalidates | Retry storms creating correlated bursts | Measure empirically; add headroom; use simulation |
| DC-02 predicts rebuild cost | Independent change events, strict ordering | Check actual cache-hit logs per stage over 20 builds | Monorepo where a lockfile changes on every commit | Split lockfiles per service; pin more aggressively |
| DC-07 predicts storage | Digest-addressed dedup | List blobs per manifest; count shared digests | Each service builds its own base from `ubuntu:latest` | Standardise a base; rebuild services from it on a schedule |
| DC-09 bounds detection | Persistent failure, independent probes | Inject a real failure and measure time to endpoint removal | Probe depends on the same broken dependency as traffic | Split liveness (process) from readiness (dependencies) |

**The `ρ → 1` counterexample in full.** A team runs four replicas at 67.5% utilisation and observes 78 ms mean latency. Traffic grows 33% to 60 req/s; utilisation becomes 90% and the model reports 178 ms. Traffic grows another 11% to 66.7 req/s and utilisation reaches 100%, at which point DC-06 has no stationary distribution — the queue grows without bound and the only correct statements are about admission control and shedding. The counterexample matters because it shows that a formula valid at 67% is not merely "less accurate" at 99%; it is *undefined*, and any tool that prints a number there is extrapolating beyond its domain.

**The zero-baseline counterexample.** DC-03 with `b = 0` reproduces the naive division. A container that reports `b = 0` in `docker stats` at idle is usually reporting a rounded or cached value, or the process has not yet faulted in its libraries. Consequence: measuring `b` too early over-states `N_max`. The discriminating check is to measure after one full request cycle and one garbage-collection cycle, then measure again at the 95th percentile of load.

**Takeaway:** Every sizing formula here is a product of a rate and a duration or a quotient of two quantities with the same units; when a number surprises you, check the units first and the utilisation second, because the formulas are exact algebra on assumptions that are usually the thing that broke.

## 5. Worked Numerical Example

**Example title and evidence label:** "Release 4.2 of an illustrative order-status API: from Dockerfile to capacity plan." All workload inputs are ILLUSTRATIVE. The arithmetic has been independently checked (Section 5.6). Nothing here is a measurement of a real service, registry, or cluster.

### 5.1 The scenario and the decision

A team owns a small HTTP service that looks up order status. It runs 45 requests per second at the daily mean, with a peak of 90 for about two hours. Each request costs 90 ms of CPU. The service must answer within a 150 ms budget at the mean, must survive three replica failures without violating that budget, must not lose order-state writes, and must be deployable fifteen times a week without the build taking more than two minutes. The team must decide: image structure, memory limit, worker count, replica count, state location, and probe settings.

The decision is not "use Docker". That is given. The decision is the *set of numbers and placements* that make the five claims in Section 1.1 true for this service, and the evidence that would falsify each one.

### 5.2 The Dockerfile under review

The following is the illustrative artifact. It is deliberately written the wrong way first, because the wrong way is what most repositories contain, and the correction is the lesson.

```text
# ILLUSTRATIVE — deliberately suboptimal, do not copy
FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENV API_TOKEN=placeholder-not-a-real-secret
EXPOSE 8000
CMD python -m app.server --port 8000
```

Five defects, each mapping to a later section:

1. The base is a floating tag, so the C library, Python patch version, and OS packages change silently between builds.
2. `COPY . .` precedes dependency installation, so every source edit invalidates the dependency layer. With DC-02 this is the 221.8 s ordering, not the 15.8 s one.
3. `pip install` is unpinned, so two builds of the same commit can produce different site-packages.
4. The token is in `ENV`, so it appears in the image config and in `docker history` output forever.
5. `CMD` uses shell form, so PID 1 is `/bin/sh`, which does not forward `SIGTERM` to Python; shutdown becomes a 10-second grace period followed by `SIGKILL`.

### 5.3 The corrected structure, and why each line moved

```text
# ILLUSTRATIVE — corrected structure
FROM python:3.12-slim@sha256:<pinned-digest>          # (1) pinned digest
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.lock .                              # (2) manifests first
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
COPY src/ ./src/                                      # (3) volatile last
RUN useradd --uid 10001 --create-home appuser         # (4) non-root
USER 10001
EXPOSE 8000
ENTRYPOINT ["python", "-m", "app.server"]             # (5) exec form
CMD ["--port", "8000"]
```

Each change is falsifiable. Pinning is verified by rebuilding from a fresh clone and comparing manifests. Manifest-first ordering is verified by editing one line of `src/` and confirming the dependency layer reports `CACHED`. Non-root is verified with `id` inside the running container. Exec form is verified by sending `SIGTERM` and observing a clean exit within the grace period. The token moved to runtime injection, verified by scanning the image config and layer contents for the string.

### 5.4 Image and startup arithmetic

The corrected image has five layers: base 30 MiB, OS packages 12 MiB, Python dependencies 22 MiB, wheel payload 96 MiB, application 4 MiB. Total transferred size 164 MiB.

- Cold pull: `164/40 + 0.15 + 5×0.05 = 4.10 + 0.40 = 4.50 s`.
- Warm pull after an app-only change: `4/40 + 0.15 + 0.05 = 0.30 s`.
- Full startup: pull (4.50) + unpack-and-mount (1.20) + interpreter boot (2.40) + readiness warm-up such as connection pool and first query (3.00) = **11.10 s** cold, **6.90 s** warm.

The 11.1 s cold start matters for two decisions: the orchestrator's initial-delay setting must exceed it, and the readiness probe must not count a listening socket as readiness, because the socket opens around 5.7 s while the first successful query is at 11.1 s.

### 5.5 Memory, workers, and replicas

**Workers.** `L = 2048 MiB`, `ρ = 0.15`, baseline `b = 1080 MiB`, per-worker peak `w = 130 MiB`:

```text
reserve = 0.15 × 2048      = 307.2 MiB
budget  = 2048 − 307.2 − 1080 = 660.8 MiB
N_max   = floor(660.8/130) = 5
RSS     = 1080 + 5×130     = 1730 MiB  (84.5% of limit)
```

Five workers. The naive fifteen would need 3030 MiB and would be OOM-killed on the first burst. The configuration therefore holds `workers = 5`. That pool caps one replica at `5/0.06 ≈ 83.3 req/s`, while the mean assigns only `45/4 = 11.25 req/s` to each of four replicas — so the worker pool is not binding at the mean, and the replica count is what carries the peak. The pool does bound the damage of a single misbehaving client: it can occupy at most five workers, so the sixth request waits rather than allocating a sixth copy of the per-worker peak.

**Replicas at mean traffic.** `λ = 45`, `c_cpu = 0.09`, `C = 1.5`, `u* = 0.7`:

```text
demand = 45 × 0.09 = 4.05 cores
usable = 1.5 × 0.7 = 1.05 cores/replica
n      = ceil(4.05/1.05) = 4
```

**Replicas at peak, with failure tolerance.** At 90 req/s demand is `90 × 0.09 = 8.10` cores, so `ceil(8.10/1.05) = 8` replicas would satisfy capacity alone. Failure tolerance is a separate constraint: after three failures, `n − 3` replicas must still carry the demand at the target utilisation. Solving `(n − 3) × 1.05 ≥ 8.10` gives `n − 3 ≥ 7.71`, so `n − 3 = 8` and `n = 11`. The check is `8 × 1.05 = 8.40 ≥ 8.10`, leaving 0.30 cores spare — 3.7% headroom, thin enough to be worth stating explicitly rather than assuming. Eleven replicas at peak, four at mean, with autoscaling between them.

**Latency at mean traffic, four replicas.** `μ = 1/0.06 = 16.667`, `a = 2.7`, `ρ = 0.675`, `C(4, 2.7) ≈ 0.3907`:

```text
W_q = 0.3907/(4×16.667 − 45) = 0.3907/21.667 = 0.01803 s = 18.03 ms
W   = 18.03 + 60 = 78.03 ms     (budget 150 ms → 71.97 ms margin)
```

At the 90 req/s peak with the 11 replicas derived above: `a = 5.4`, `ρ = 0.491`, and by the same recursion `C(11, 5.4) ≈ 0.0252`, `W_q = 0.0252/(11×16.667 − 90) = 0.27 ms`, `W ≈ 60.3 ms` — better per request than at the mean, because eleven servers at 49.1% each queue far less than four at 67.5%. The danger is not the peak itself but the transition: if autoscaling lags by 60 s, requests during the ramp see the four-replica queue at 90 req/s, where `a = 5.4` against `c = 4` gives `ρ = 1.35 > 1` and the queue has no stationary distribution.

### 5.6 Independent checks

| Check | Method | Result |
|---|---|---|
| Pull arithmetic | `164/40 = 4.1`; `4.1 + 0.15 + 0.25` | 4.50 s, consistent |
| Unit homogeneity in DC-03 | MiB ÷ MiB, integer floor | 5, not 5.083 |
| Worker memory feasibility | `1080 + 5×130 = 1730 ≤ 2048 − 0` | 318 MiB headroom |
| Replica arithmetic | `45×0.09 = 4.05`; `4.05/1.05 = 3.857` | 4 replicas |
| Little's law cross-check | `λW = 45×0.07803 = 3.51`; `a + λW_q = 2.7 + 0.81` | 3.51 both ways |
| Failure-tolerance algebra | 11 replicas − 3 = 8; `8×1.05 = 8.40 ≥ 8.10` | Feasible with 0.30 cores spare |
| Cache-order difference | `15.84` vs `221.76` from DC-02 | Factor 14.0 |

### 5.7 The wrong path, in detail

The plausible mistake is to answer the memory question with `docker stats` under light load. At 5 requests per second the container shows 1.18 GiB, so an engineer computes `2048/1180 ≈ 1.7` and concludes "one worker per 2 GiB, we need a 4 GiB limit". This is wrong three ways: the reading excludes the baseline split (the 1.18 GiB *is* mostly baseline plus two workers), it ignores the reserve, and it treats an idle measurement as peak. The corrected procedure measures baseline with zero in-flight requests (1080 MiB), marginal cost by stepping concurrency 1→8 (130 MiB per worker at the 95th percentile of payload size), and then applies DC-03.

The misleading result is a limit increase. The correction is a measurement protocol, and the verification is to run at `N_max` with upper-decile payloads for thirty minutes and confirm `oom_kill` stays zero while the RSS high-water mark stays below `L(1−ρ)`.

A second wrong path: sizing replicas from CPU utilisation observed in production (67.5%) rather than from demand and per-replica capacity. Observed utilisation tells you what *was* used, not what must be available; at 45 req/s it yields the same 4 replicas by luck, but at a 2× traffic forecast it returns 8 when the failure-tolerance requirement demands 11.

### 5.8 Production-scale interpretation

The numbers that change under production conditions are the ones attached to variance, not to the mean. Pull bandwidth is shared across a rolling deploy, so 11 replicas pulling 164 MiB simultaneously is 1.8 GiB of transfer, not 11 × 4.5 s of independent latency. Image-layer locality across nodes determines whether that is 45 s or 5 s. Startup at 11.1 s multiplied by a rolling update with `maxSurge = 1` means the deploy takes 11 rounds × 11.1 s ≈ 122 s before the new version is fully live, during which two versions coexist — which requires the API change to be backward compatible.

### 5.9 What the example does not establish

It does not establish that 90 ms of CPU per request is realistic for any particular service; it is chosen for arithmetic clarity. It does not establish that M/M/c describes this or any traffic — the model's assumption of exponential service is likely violated by a cache-heavy lookup, and a real tail would be worse. It does not establish registry pricing, which is illustrative and dated. It establishes that the six decisions above are computable from stated inputs, that the arithmetic is self-consistent, and that the wrong measurement procedure produces a specific, identifiable wrong answer.

**Takeaway:** The worked example's value is the chain, not the constants — every number is derived from an input you can challenge, and each challenge changes exactly one decision.

## 6. Visual / Geometric Intuition

### 6.1 The overlay stack, drawn to scale by consequence

The diagram below is schematic; the widths represent *writable* cost, not bytes. The key point is that only the top band is writable and that a write to a lower band first copies the whole file upward.

```text
container root filesystem (mount namespace view)
+-------------------------------------------------------------+
| writable layer  (per container, discarded on removal)       | <-- writes land here
+-------------------------------------------------------------+
| layer n : application code            4 MiB   read-only     |     ^
+-------------------------------------------------------------+     | copy-up
| layer n-1: dependency wheels         96 MiB   read-only     |     | on first
+-------------------------------------------------------------+     | write to
| layer n-2: python packages           22 MiB   read-only     |     | an existing
+-------------------------------------------------------------+     | file
| layer n-3: OS packages               12 MiB   read-only     |
+-------------------------------------------------------------+
| layer 1 : base image digest          30 MiB   read-only     |
+-------------------------------------------------------------+
        |                                    |
        |                                    +--> shared by every container
        |                                         from the same image: one
        v                                         copy on disk, N mounts
   volume mount (if declared)  <-- NOT part of the stack; survives the container
```

Reading the diagram: the five read-only bands are shared. Eleven replicas of the same image occupy one copy of those 164 MiB plus eleven writable layers, not eleven copies of 164 MiB. The writable layer is where `docker diff` looks, where copy-up cost accumulates, and where state dies. A volume is drawn outside the stack because it is a different mount, with a different lifecycle and different permission semantics.

The vertical arrow labelled copy-up is the mechanism behind DC-08: a write to a file that lives in the 30 MiB base triggers a copy of that file into the top band before the first byte changes. Reading is unaffected; only modification pays.

### 6.2 The memory budget as a bar

DC-03 is easier to trust when seen as a divided bar. The bar below is the 2048 MiB limit from Section 5, drawn in proportion.

```text
0 MiB                                                    2048 MiB
|----------|----------------|--------------------|---------|
|<--1080-->|<----307.2----->|<------660.8------->|
  baseline     reserve          worker budget
  (b)          (rho*L)         (L(1-rho)-b)
                                   |
                                   +-- 5 workers x 130 = 650 MiB used
                                   +-- 10.8 MiB unallocated slack

naive view (wrong): |<------------------- 2048 ------------------->|
                    / 130 = 15 workers  -> needs 3030 MiB, dies at ~2048
```

The two bars are the same physical limit. The top bar subtracts what cannot be spent; the bottom bar treats the limit as spendable capacity. Every OOM incident in a container with an obviously "sufficient" limit is the bottom bar being used as if it were the top one.

Note where the reserve sits: between baseline and worker budget, not at the end. That placement matters, because fragmentation and transient allocation peaks grow with concurrency and land in the middle of the budget, not against the ceiling. Placing the reserve at the end (dividing first, then subtracting) systematically overstates the safe count.

### 6.3 Latency against utilisation: the knee

```text
mean sojourn time W (ms), 4 servers, S = 60 ms
200 |                                                    *
    |                                                  *
150 |                                              *
    |                                          *
100 |                                    *
    |                            *
 50 |                 *
    |      *
  0 +---*----*----*----*----*----*----*----*----*----*----> rho
      0.1  0.2  0.3  0.4  0.5  0.6  0.675 0.8  0.9  0.95  1.0
                    ^                    ^
                 W = 61 ms            W = 178 ms
                 (c=6 curve)          (c=4, rho=0.9)

curve shape:   W = C(a)/(c*mu - lambda) + 1/mu
               the C/(...) term is the queue, and it diverges as lambda -> c*mu
```

Three labelled points: at `ρ = 0.45` with six servers the queue contributes about 1.2 ms; at `ρ = 0.675` with four servers it contributes 18.0 ms; at `ρ = 0.90` with four servers it contributes 163.4 ms. The curve is flat then vertical — there is no comfortable operating region near 1.0, which is the geometric reason `u*` is set conservatively rather than optimistically.

The dashed asymptote at `ρ = 1.0` is not "very slow"; it is *no stationary distribution*. Requests arrive faster than they complete and the queue grows without bound. Any capacity tool that reports a finite latency at `ρ ≥ 1` is extrapolating outside the model.

### 6.4 The cache-order timeline

DC-02 is a statement about a product of hit probabilities, which is easiest to see as a timeline of "what reruns".

```text
stage costs (s):  base 25 | os pkgs 40 | deps 60 | wheels 120 | src 5

build 1  [base][os pkgs][deps][wheels][src]   nothing cached -> 25+40+60+120+5 = 250 s
build 2  [base][os pkgs][deps][wheels][src]   src edited    -> 5 s
         ----------------------------------> only the last stage reruns
build 3  [base][os pkgs][deps][wheels][src]   lockfile      -> 60+120+5 = 185 s
         --------------------->                 deps and everything above rerun
build 4  [base][os pkgs][deps][wheels][src]   base rotated  -> 250 s
         ----------------------------->        everything reruns

wrong order (src copied first):
build 5  [src][base][deps][wheels]            any src edit  -> 250 s
         ---------------------->               the volatile stage poisons all above
```

The arrows mark the boundary of the cache hit: everything to the right of the arrow reruns. The lesson: cost is determined by the *position of the most volatile input*, and that position is a choice the author makes, not a property of the project.

### 6.5 Two-probe state machine

Readiness and liveness as a state machine, with the transition conditions that distinguish them.

```text
      start
        |
        v
   +----------+  readiness fails  +-----------+
   | STARTING |------------------>| NOT READY |  (no traffic, no restart)
   +----------+  (dependency)     +-----------+
        |                               |
        | readiness passes              | readiness passes
        v                               v
   +----------+   liveness fails  +-----------+
   |  READY   |<------------------|  SERVING  |
   +----------+   k times         +-----------+
        |                               |
        | liveness fails k times        | SIGTERM
        v                               v
   +----------+                   +-----------+
   | RESTART  |                   | DRAINING  |--> exit 0 within grace
   +----------+                   +-----------+    else SIGKILL (137)
```

The dangerous edge is the left one: a dependency outage that fails *liveness* converts `NOT READY` into `RESTART` for every instance simultaneously, and each restart re-enters the outage. The only correct wiring is readiness = "can serve now, including dependencies", liveness = "process is alive and not deadlocked".

### 6.6 Where the bytes go on a deploy

```text
build machine      registry           node A            node B
  |                   |                 |                 |
  |--push 164 MiB---->|                 |                 |
  |                   |<--pull 164 MiB--|                 |
  |                   |<--pull 164 MiB--------------------|
  |                   |                 |                 |
  total registry egress = 328 MiB for 2 nodes
  total node disk       = 164 MiB per node (layers shared with other images)
  total pull wall time  = max(4.5 s, contention) not sum(4.5, 4.5)
```

Two facts hide in this picture. First, egress scales with *nodes*, not with image count, so a fan-out deploy is an egress event. Second, wall time does not add across parallel pulls, but bandwidth does subtract: eleven simultaneous pulls over one node's uplink is one 40 MiB/s link shared eleven ways, not eleven 40 MiB/s links.

**Takeaway:** The four pictures share one structure — a shared read-only region, a small private writable region, and a curve that is flat until it is vertical — and the vertical parts are where container incidents live.

## 7. Comparison / Selection Table

### 7.1 Base image selection

| Option | Preconditions | Quality/guarantee | Size / surface | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Full distribution base (e.g. a general-purpose `python:3.x`) | Build tools needed at run time; shell debugging expected | Complete userland, familiar tooling | Largest; widest package surface | Early development, legacy builds needing compilers at run time | Runtime that only needs the interpreter | DERIVED from contents; sizes ILLUSTRATIVE |
| `-slim` variant | No compilers at run time; glibc-compatible wheels | Same libc family, trimmed docs and tooling | Moderate; fewer OS packages | Most Python services | Native builds requiring headers at run time | DERIVED; version specifics OPEN |
| Alpine / musl-based | Pure-Python dependencies or musl wheels available | Small image; different libc | Smallest of the general options; musl ABI | Small services with no native wheel dependency | Projects depending on glibc-only wheels or DNS edge cases | DERIVED; compatibility OPEN per dependency |
| Distroless / no shell | Static or fully bundled runtime; debugging via ephemeral containers | No shell or package manager to abuse | Small; minimal surface | Hardened production runtimes | Teams without out-of-band debugging tooling | DERIVED; operational cost real |
| Scratch | Single static binary, no libc dependency | Nothing but your binary | Minimal | Go/Rust static binaries | Anything dynamic | DERIVED |
| Vendor-hardened base | Compliance requirement; vendor patch SLA | Policy-backed patch cadence | Large | Regulated environments | You cannot consume the update cadence | OPEN unless a contract exists |

The selection rule is not "smallest wins". Every step down the table removes tools that somebody will eventually need at 02:00, so the choice pairs a size decision with a debugging-capability decision. The defensible practice is the smallest base that still permits the team's *actual* incident response procedure, tested once before adoption.

### 7.2 Build strategy selection

| Strategy | Preconditions | Cache behaviour | Build time | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Naive single-stage, `COPY . .` first | None | Invalidated by any source change | Slow on every edit | Throwaway experiments | Any repeated build | DERIVED (DC-02: 221.8 s case) |
| Manifest-first single stage | Lockfile separated from source | Dependency layer survives source edits | Fast on source edits | Simple services | Multi-language repos | DERIVED (DC-02: 15.8 s case) |
| Multi-stage with builder + runtime | Build tools disposable | Same as above, plus smaller final | Fast builds, small artifact | Compiled languages, native wheels | Runtime needs the toolchain | DERIVED |
| Cache mount for package manager | BuildKit-style cache mounts available | Package cache persists across builds | Fastest dependency installs | Large dependency trees | Reproducibility requires a clean cache proof | DERIVED; feature support version-specific, OPEN |
| Remote/shared cache | CI with shared cache backend | Cross-machine hits | Very fast on unchanged inputs | Many CI runners | Cache poisoning risk unmanaged | DERIVED; security boundary OPEN |
| Fully hermetic offline build | Vendored sources, no network at build | Deterministic | Slow first, then stable | Audited releases | Dependency freshness matters | DERIVED |

```text
IF a build is repeated more than a few times a day AND source edits dominate changes
    -> manifest-first ordering, then multi-stage for the runtime artifact.
ELSE IF the same dependency set is rebuilt across many CI machines
    -> add a shared cache, with cache keys scoped so untrusted branches cannot poison.
ELSE IF the artifact must be provably reproducible offline
    -> hermetic build with vendored inputs; accept slower first builds.
If conditions conflict -> reproducibility beats speed when the artifact is a release;
    speed beats reproducibility when the artifact is a preview.
```

### 7.3 Process supervision selection

| Option | Preconditions | Signal behaviour | Zombie reaping | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Shell form `CMD python app.py` | None | Shell is PID 1; may not forward signals | No | Nothing production-facing | Graceful shutdown required | DERIVED from shell semantics |
| Exec form `ENTRYPOINT ["python", ...]` | Application handles `SIGTERM` | Direct delivery to the app | App must reap its own children | Single-process services | App spawns unsupervised children | DERIVED |
| Tiny init (`--init` or an init binary) | Init binary available in image | Forwards signals, reaps orphans | Yes | Apps that fork, or that misbehave on signals | Trivial single-process cases | DERIVED |
| Full supervisor in container | Policy requires process supervision | Supervisor decides restart | Yes | Multi-process legacy apps | Orchestrator already owns restart policy | DERIVED; anti-pattern risk |
| Orchestrator-managed restart | Restart policy configured | Orchestrator sends `SIGTERM`, then `SIGKILL` after grace | N/A (host reaps) | Most services | Supervised restart is the requirement | DERIVED |

The distinction that decides most cases: a container should have **one process with a clear lifecycle**. The moment two processes must both stay alive, the container boundary is being used for something it does not model well, and the failure modes (one dies, the other keeps the container "running") become invisible to the orchestrator.

### 7.4 State placement selection

| Option | Preconditions | Durability | Performance | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Writable layer | None | None — lost on removal | Copy-up on modify | Scratch space, caches that can be rebuilt | Anything that must survive | DERIVED (DC-08) |
| Named volume | Volume driver available | Survives container; lifecycle explicit | Near-native | Single-writer databases in dev/small prod | Multi-node access needed | DERIVED; driver specifics OPEN |
| Bind mount | Host path exists with correct ownership | Host-dependent | Native | Config injection, development iteration | Portability required | DERIVED |
| Object storage | Network + credentials at run time | Durable by contract | Higher latency | Artifacts, uploads, logs | POSIX semantics required | DERIVED |
| Managed database | External service available | Contractual | Network-bound | Production relational state | Offline/air-gapped requirement | DERIVED; provider specifics OPEN |

The rule that resolves nearly every case: if losing the data on container replacement is unacceptable, the data must not be in the writable layer; if two replicas must read the same file simultaneously with POSIX semantics, a single-node volume is the wrong primitive.

### 7.5 Prioritised selection guide

```text
IF a hard constraint fails (no credential in image, non-root required, digest pinned)
    -> disqualify the option regardless of size or speed.
ELSE IF the workload is latency-sensitive and CPU-bound
    -> size replicas from DC-05 at u* <= 0.7, verify with DC-06, and reject any
       plan whose utilisation exceeds 1.0 after N-3 failures.
ELSE IF the workload is memory-bound with bounded per-request allocation
    -> size workers from DC-03 with a stated reserve; measure b and w, not guess.
ELSE IF the artifact must be auditable
    -> pin digests, record build inputs, and treat tags as labels only.
ELSE IF the team is small and the service is internal
    -> manifest-first single stage, slim base, exec-form entrypoint, named volume
       for state; defer multi-stage and distroless until a requirement appears.
If two conditions conflict -> the security and auditability constraints win over
    build speed, because their failures are not recoverable by rebuilding.
```

**Takeaway:** Selection is a sequence of disqualifications, not a search for a best base image; once the hard constraints are applied, the remaining choices differ by cost and debuggability, and those are team decisions, not technical ones.

## 8. Failure Modes & Edge Cases

Ten entries below, each with *competing causes* because the whole point of a discriminating check is that the obvious explanation is one of several. Confidence in a cause is stated where it is not established. Provenance: `HYPOTHETICAL` unless marked otherwise; these are constructed from documented mechanism, not observed incidents.

### F-01 — Exit code 137 with a "sufficient" memory limit

**Trigger/symptom.** A container restarts periodically; `docker inspect` shows `ExitCode: 137`; the memory limit was set from a measurement that looked comfortable.

**Competing causes.** (a) The container exceeded its cgroup memory ceiling and the kernel OOM-killed it — 137 is 128 + 9, the shell convention for "terminated by SIGKILL", and an OOM kill is one way to receive SIGKILL; (b) the orchestrator or an operator sent `SIGKILL` after a grace period expired, which produces the same 137 with a completely different cause; (c) the node itself was under memory pressure and a system-level killer chose the largest process, not the cgroup limit; (d) the process called `exit(137)` itself, which is unusual but happens in wrappers.

**Discriminating check.** Read `memory.events` (cgroup v2) or `memory.failcnt`/`memory.oom_control` (v1) for the container's cgroup: `oom_kill` counter > 0 establishes cause (a) directly. If `oom_kill` is 0, check `State.FinishedAt` minus `State.StartedAt` against the grace period — a stop at exactly the grace boundary indicates (b). Check node-level `dmesg`/kernel logs for a global OOM message to distinguish (c). Cause (d) is confirmed by the application log.

**Correct action.** For (a): apply DC-03 with the measured baseline and the reserve, then reduce concurrency or raise the limit. For (b): fix signal handling (F-02) so shutdown finishes within grace. For (c): stop overcommitting the node; the container's own limit is not the constraint.

**Prevention/verification.** Alert on `oom_kill` rate as a first-class metric, not on restart count. Verify by driving upper-decile payloads at `N_max` for thirty minutes and confirming `oom_kill = 0` while the RSS high-water mark stays under `L(1−ρ)`.

**Provenance.** HYPOTHETICAL; the 137 = 128+9 convention and the cgroup counters are documented kernel/runtime behaviour (SOURCED, version-specific paths OPEN).

### F-02 — Every shutdown takes exactly the grace period and ends in SIGKILL

**Trigger/symptom.** Rolling deploys are slow; every pod takes the full termination grace period (often 30 s) to stop; logs show no graceful-shutdown message; the app's in-flight requests are cut.

**Competing causes.** (a) PID 1 is a shell because `CMD` used shell form, and the shell does not forward `SIGTERM` — a documented and extremely common shell behaviour; (b) the application ignores `SIGTERM` because its handler was never written; (c) the application received `SIGTERM` and is genuinely draining a long-lived request; (d) the orchestrator sends `SIGKILL` immediately because the grace period is set to 0; (e) the entrypoint is a wrapper script that execs the app in a subshell.

**Discriminating check.** Inspect PID 1 inside a running container: if it is `/bin/sh`, cause (a) is established. Send `SIGTERM` manually and observe the exit code and elapsed time — an app that ignores the signal exits by `SIGKILL` at the grace boundary with nothing in its log. Check the configured grace period value for (d). For (c), look for an in-flight request counter that stays non-zero.

**Correct action.** Convert to exec form (`ENTRYPOINT ["python", "-m", "app"]`); if the app genuinely needs a shell, use a tiny init. Add an explicit handler that stops accepting new work, waits for in-flight work with a bounded timeout, and exits 0. Set grace period to the observed drain time plus margin, and verify the sum is less than the orchestrator's own patience.

**Prevention/verification.** A shutdown test in CI: start the container, send `SIGTERM`, assert exit code 0 within `T_drain_max`. A deploy-time check that asserts PID 1 is not a shell.

**Provenance.** HYPOTHETICAL; shell signal-forwarding behaviour is SOURCED from POSIX shell semantics.

### F-03 — Credential recoverable from the image

**Trigger/symptom.** A scanner flags a secret; or a credential is found by inspecting an image that was pushed months ago.

**Competing causes.** (a) The secret is in `ENV` or `ARG`, so it is in the image config; (b) it was `COPY`ed in and later deleted in the same layer, so it is still in the layer; (c) it was `COPY`ed in and deleted in a *later* layer, so it is in the earlier layer and recoverable; (d) it is in build history as an argument value, even if never written to a layer; (e) it is fetched at build time and baked into a compiled artifact.

**Discriminating check.** Three separate inspections are needed because they catch different cases: read the image config (env, cmd, labels) for (a); list layer contents and search each layer's tar for (b)/(c); read build history for (d) where the engine records argument usage. Case (e) requires scanning the artifact itself, not the layer listing.

**Correct action.** Revoke and rotate the credential first — the image may be public or mirrored, and removal does not un-leak it. Then move the secret to runtime injection (environment from a secret store, or a file mounted at run time), rebuild from a context that never contained it, and verify by scanning the *new* image, not by reading the Dockerfile.

**Prevention/verification.** Pre-commit scanning of the build context, a `.dockerignore` that excludes local credential files, and a CI gate that scans every built image. The gate must scan the image, because a clean Dockerfile can still produce a dirty artifact.

**Provenance.** HYPOTHETICAL; layer persistence of deleted files is inherent to the layer model (DERIVED).

### F-04 — Data disappears after a routine redeploy

**Trigger/symptom.** A service that writes files works, then after a deploy "loses" its database, uploads, or cache; the data existed for days.

**Competing causes.** (a) The data directory is in the writable layer, which is discarded with the container; (b) the data is in a volume, but the redeploy created a *new* volume because the volume name or mount declaration changed; (c) the data is on a bind mount to a host path that differs between nodes, so the replica moved and found an empty directory; (d) an init script wipes the directory on start (common with "reset" defaults); (e) file ownership changed with a new `USER`, so the app cannot write and silently falls back to a temp directory.

**Discriminating check.** `docker diff` on a live container shows writable-layer changes — if the data path appears there, cause (a). Inspect the mount table inside the container to confirm the declared volume is actually mounted at the expected path (b). Compare the mount source across two replicas (c). Read the start logs for truncation/reset messages (d). Check `id` inside the container against the directory's owner (e).

**Correct action.** Move durable state to a declared volume or an external store; make the volume name deterministic; pin the mount path in the container spec; remove destructive init defaults; align the runtime UID with the data-path owner. For anything that must survive node loss, the writable layer and a single-node volume are both wrong — the store must be external.

**Prevention/verification.** A backup-and-restore test that creates data, destroys the container, recreates it from the same spec, and asserts the data is present. This test is the only thing that distinguishes "we have volumes" from "we have durability".

**Provenance.** HYPOTHETICAL; the writable-layer lifecycle is DERIVED from the image/container model.

### F-05 — "Works locally, fails after deploy" with identical tags

**Trigger/symptom.** The same tag behaves differently in two environments; logs show a dependency version, a locale, a time zone, or an architecture-specific error.

**Competing causes.** (a) The tag was rebuilt and repointed, so the two environments pulled different manifests under the same name; (b) an architecture difference — an amd64 image on an arm64 host runs under emulation or fails to start; (c) environment variables or config files differ (time zone, locale, feature flags), and the application reads them; (d) a mounted file or directory differs between environments; (e) the host kernel or cgroup version differs, changing a limit's enforcement; (f) the application resolved a dependency at *run* time (a plugin download, a model fetch) and got different bytes.

**Discriminating check.** Compare digests first — this single comparison eliminates (a) or establishes it, and it takes seconds. If digests match, compare `docker image inspect` output for architecture and config, then diff the runtime environment schema, then compare mounts. Cause (f) is visible only in the application's own network log, which is why "the image is identical" can be true and irrelevant simultaneously.

**Correct action.** Deploy by digest, not tag. Fix the architecture by building for the target platform explicitly. Externalise configuration through a declared, versioned schema. Remove run-time dependency resolution or pin it with the same rigour as build-time dependencies.

**Prevention/verification.** A deploy gate that records and compares digests, plus a startup log line that prints the digest, the architecture, and the config schema version. The log line is the cheapest possible diagnostic and is usually absent.

**Provenance.** HYPOTHETICAL; multi-architecture manifest behaviour is SOURCED from OCI image-index semantics (version specifics OPEN).

### F-06 — Build takes four minutes for a one-line change

**Trigger/symptom.** CI rebuilds dominate feedback time; a comment change triggers a full dependency reinstall.

**Competing causes.** (a) `COPY . .` before dependency installation, so the source layer sits below the dependencies (the DC-02 ordering failure); (b) an unpinned base tag re-resolved on every build, invalidating everything; (c) `.dockerignore` missing, so a build context includes `.git`, virtualenvs, or dataset files, changing the context hash and inflating transfer; (d) a lockfile regenerated on every CI run (timestamps or a non-deterministic resolver), so its bytes change; (e) a build argument that changes per commit (a version string, a timestamp) placed in a layer that precedes dependencies.

**Discriminating check.** Read the build log's per-step `CACHED` markers to find the *first* non-cached step; every step above it also reruns. Then check that step's inputs: which file changed? Compare the build context size reported by the client against the repository size (c). Diff the lockfile across two runs of the same commit (d).

**Correct action.** Order by volatility (manifests → dependencies → source). Pin the base digest. Add a `.dockerignore`. Make the lockfile deterministic. Move volatile build arguments into the last stage, or into labels that do not affect layer content.

**Prevention/verification.** Track a per-stage cache-hit rate over the last twenty builds; a stage with a hit rate near zero is either genuinely volatile (acceptable) or misordered (not). Verify by editing only `src/` and asserting the dependency step reports `CACHED` and total time is within the DC-02 prediction.

**Provenance.** HYPOTHETICAL; DERIVED from DC-02.

### F-07 — Readiness probe passes while the app cannot serve

**Trigger/symptom.** Traffic is routed to a new instance and users see 500s or timeouts for the first several seconds; or worse, indefinitely.

**Competing causes.** (a) The probe checks that a port is open, and the framework opens the socket before warm-up completes; (b) the probe checks a static path that does not touch the database, so a broken database is invisible; (c) the probe's success criterion is a 200 from a health endpoint that the app serves from a cache even when its dependency is down (a deliberate "always healthy" handler); (d) the initial delay is shorter than the cold start (11.1 s in Section 5), so the probe fails during boot and the orchestrator restarts a container that would have become healthy; (e) the probe timeout is shorter than the p99 legitimate latency, so healthy instances flunk randomly.

**Discriminating check.** Compare the timestamp of the first successful probe against the timestamp of the first successful *business* request in the logs — a gap proves (a) or (d). Call the health endpoint while deliberately stopping the dependency: if it still returns 200, (b)/(c) is established. Compare probe timeout against the endpoint's measured p99.

**Correct action.** Make readiness reflect dependency-usable state, and make liveness reflect only process health. Set the initial delay above the measured cold start with margin. Set the probe timeout above the p99 and the period above the timeout. Add a startup probe where the orchestrator supports one, so boot slowness does not consume the liveness budget.

**Prevention/verification.** A test that starts the app with the dependency deliberately unavailable and asserts (i) readiness stays false, (ii) liveness stays true, (iii) no restart occurs. This is the test that separates the two probes, and it is rarely written.

**Provenance.** HYPOTHETICAL; probe semantics are SOURCED from orchestrator documentation (version OPEN).

### F-08 — Registry availability becomes a deploy outage

**Trigger/symptom.** A rollback or scale-up fails; new nodes cannot start containers; the registry returns errors or is rate-limiting.

**Competing causes.** (a) The registry is a single external dependency with no local cache, so a registry incident stops all new starts; (b) pull rate limits are hit because every node pulls independently on every scale event; (c) credentials expired, so pulls fail with an authentication error that looks like a network error in some clients; (d) the image was deleted by a retention policy that counted days but not the running dependency; (e) the tag moved and the previous digest was never recorded, so rollback has no target.

**Discriminating check.** Attempt a pull by digest from a node and read the exact status: 401/403 indicates (c), 429 indicates (b), 404 indicates (d), and connection errors indicate (a). Confirm (e) by checking whether a digest was recorded in the release metadata at all — if the record only contains a tag, rollback capability is unproven regardless of what the registry holds.

**Correct action.** Cache images on nodes or in a pull-through cache for availability. Stagger or batch pulls across nodes to respect limits. Use credentials with a defined rotation and alert on auth failures separately from network failures. Pin retention to "keep every digest referenced by a running or recently-running deployment" rather than to a day count. Record digests at deploy time.

**Prevention/verification.** A rollback drill: deploy a new digest, then roll back to the recorded previous digest, and measure the time. A drill that has never been run cannot be counted as a capability.

**Provenance.** HYPOTHETICAL; rate-limit and retention behaviour is registry-specific and OPEN.

### F-09 — CPU throttling with low average utilisation

**Trigger/symptom.** p99 latency is much worse than the mean suggests; CPU utilisation averages 20%; users report periodic stalls; the service "feels" bursty.

**Competing causes.** (a) The quota is smaller than a single thread's requirement, so a multi-threaded app is throttled every period (DC-10); (b) garbage collection or a batch job creates periodic CPU bursts that exceed the quota within one period; (c) the limit is fine but the *node* is oversubscribed, so the container waits for physical CPU despite having quota; (d) an I/O stall is being mistaken for CPU throttling because both show up as latency; (e) the latency comes from a downstream dependency, and local CPU is irrelevant.

**Discriminating check.** Read the cgroup's throttling counters (`cpu.stat` in v2: `nr_throttled`, `throttled_usec`). Non-zero `nr_throttled` with growing `throttled_usec` establishes (a)/(b). If throttling counters are zero, the cause is (c), (d), or (e) — compare node-level CPU pressure and the downstream's own latency.

**Correct action.** For (a)/(b): raise the quota, or reduce period-burstiness, or reduce concurrency so demand spreads. For (c): stop oversubscribing nodes. For (d)/(e): fix the actual bottleneck; raising the CPU limit will change nothing and will hide the diagnosis.

**Prevention/verification.** Alert on throttle ratio as a distinct signal from utilisation. Verify by load-testing a burst pattern that matches production and confirming `nr_throttled` stays at zero (or at a stated acceptable rate) while p99 stays within budget.

**Provenance.** HYPOTHETICAL; cgroup CPU accounting is SOURCED (version-specific file paths OPEN).

### F-10 — Image grows without anyone deciding it should

**Trigger/symptom.** Pull times creep from 4.5 s to 40 s over a year; disk pressure on nodes; nobody remembers approving it.

**Competing causes.** (a) Build tools and caches left in the final layer (no multi-stage); (b) package-manager caches not cleaned in the same layer that created them; (c) test fixtures, datasets, or model weights committed into the context and copied in; (d) the base image itself grew because it floats on a tag; (e) multiple services converged on a shared base but each added its own large dependencies, so the shared prefix stopped dominating (DC-07 saving erodes).

**Discriminating check.** List layers sorted by size and attribute each to an instruction. If the largest layer is the one that installed packages, (a)/(b). If the largest is the application layer, (c). Compare the base layer size across two months (d). Recompute the dedup ratio across the image set (e).

**Correct action.** Multi-stage build so compilers stay out of the runtime; clean package caches in the same `RUN` instruction that populated them; exclude data from the context; pin and schedule base updates deliberately; measure the dedup ratio as a tracked metric rather than assuming it.

**Prevention/verification.** A CI size budget per image with a documented process for raising it. A monthly layer-attribution report. The budget is what converts "the image grew" from a discovery into a decision.

**Provenance.** HYPOTHETICAL; DERIVED from the layer and dedup models.

### 8.11 Edge cases worth stating explicitly

| Edge case | What happens | Why it surprises | Correct handling |
|---|---|---|---|
| Container with no init and a forking child | Orphaned children are re-parented to PID 1 in that namespace, which is the app, which may not reap them | Zombie accumulation with a "healthy" app | Use an init, or do not fork |
| `SIGTERM` during a long request with a 30 s grace | Response is cut; client retries; duplicate side effect | Retries look like new traffic | Idempotency keys plus bounded drain |
| Read-only root filesystem | Writes to the writable layer fail immediately with EROFS | Common assumption that the layer is always writable | Declare writable paths as volumes or tmpfs |
| `tmpfs` mount for scratch | Uses memory, counted against the limit | Scratch space silently consumes the memory budget | Size the tmpfs and subtract it from the budget in DC-03 |
| Non-root UID not present in `/etc/passwd` | Some tools fail on user lookup; some warn only | Intermittent permission and lookup errors | Use a numeric UID consistently; add an entry if a tool requires it |
| Two containers sharing a PID namespace | One sees the other's processes | Debugging tools show unexpected processes | Intentional sharing only, and named |
| Time zone inside the container | Defaults to UTC, not the host's zone | Timestamps shift by hours after containerization | Set `TZ` explicitly; store UTC, render locally |
| Locale not set | Sorting and case-folding differ from the developer's machine | "Works locally" on text processing | Set `LANG`/`LC_ALL` explicitly |
| Hosts file and DNS in the container | The container has its own `/etc/hosts` and resolver configuration | Name resolution differs from the host | Use explicit service names; configure resolv settings deliberately |
| Clock skew after host suspend | Monotonic clock is fine; wall clock jumps | Token expiry and cache TTL misbehave | Depend on monotonic time for durations |

**Takeaway:** Every failure in this section is a case where two plausible causes produce the same symptom; the diagnostic that separates them is always a specific counter, config, or timestamp, never "check the logs more carefully".

## 9. Complexity & Cost Analysis

### 9a. Computational complexity of the build and pull

Container operations are cheap in algorithm and expensive in bytes. The table defines the size variables once: `F` = files in the build context, `B` = bytes in the context, `n` = Dockerfile instructions, `L` = total layer bytes in the image, `k` = layers, `N` = nodes pulling, `M` = manifests (tags) in the registry.

| Operation | Input sizes | Time | Space | I/O / network | Assumptions / dominant bottleneck |
|---|---|---|---|---|---|
| Context hashing for cache lookup | `F`, `B` | O(B) hash | O(1) streaming | Local disk read of context | Hash dominates for large contexts; `.dockerignore` reduces `B` |
| Context upload to builder | `B` | O(B) | O(B) temporary | Network when builder is remote | Dominant cost in remote-builder setups |
| Cache lookup per instruction | `n` | O(n) | O(n) digests | Registry/local metadata | Cheap; the cost is what a miss triggers |
| Layer creation (a `RUN` step) | step's own writes | proportional to filesystem changes | O(step output) | Disk writes, then tar+compress | Compression is CPU-bound; `tar` is IO-bound |
| Layer compression | step output bytes | O(size · compression cost) | O(window) | CPU | Usually the largest CPU term after package installs |
| Image pull | missing bytes `L_miss` | O(L_miss / β) | O(L_miss) disk | Network | Bandwidth or unpack, whichever is slower |
| Layer unpack | `L_miss` | O(L_miss) extract | O(L_miss) disk | Disk write | Decompression CPU can bind on small files |
| Container start | image present | O(cold start) = `t_start` | O(writable churn) | Almost none | Interpreter boot and warm-up dominate |
| Container stop | in-flight work | O(`T_drain`) | O(1) | Network for draining | Grace period is the ceiling |
| Registry storage | unique blobs `|∪D|` | — | O(unique bytes) | — | Dedup determines whether this scales with `M` |
| Dedup effectiveness | `M` manifests | O(M·k) digest comparisons | O(unique digests) | Metadata only | Cheap to compute, easy to ignore |

Two complexity facts deserve naming because they are counter-intuitive. First, **build time is not monotone in image size**: a build that produces a 2 GiB compressed artifact may be fast if it is mostly one large already-compressed file, while a build producing 200 MiB of many small text files can be slower because compression and per-file metadata dominate. Second, **pull time is not linear in image count on a node**: the second image sharing a base costs only its delta (DC-04), so the marginal cost of an image falls as the shared prefix grows.

### 9b. Worked sizing: build, registry, egress, and unit cost

The workload below continues the Section 5 scenario for a single service, with four environments (dev, staging, two production regions) and fifteen deploys per week. All prices are ILLUSTRATIVE and stated in USD for arithmetic only; the reader must substitute their own rates and date.

```text
WORKLOAD
  Deploys per week                    15
  Builds per deploy (CI + retries)    1.3  -> 19.5 builds/week
  Mean build wall time (corrected)    4.2 min
  Build machine allocation            2 cores
  Images per deploy (4 environments)  4  (one per environment, same digest)
  Image transferred size              164 MiB
  Nodes per production region         11 at peak
  Production regions                  2
  Registry price (illustrative)       $0.10 per GiB-month stored
  Egress price (illustrative)         $0.09 per GiB transferred
  Compute price (illustrative)        $0.05 per core-hour
```

**Build compute.** Per week: `19.5 builds × 4.2 min × 2 cores = 163.8 core-minutes = 2.73 core-hours`. Per year: `2.73 × 52 = 141.96 core-hours ≈ $7.10`. Build compute is trivially cheap; what it consumes is *engineer attention*, which is why the 14× ordering difference from DC-02 matters more than the dollars.

**Registry storage.** Unique blobs across the service's images (DC-07 arithmetic): 259.2 MiB. With fifteen deploys per week each adding a new application layer of 4 MiB and occasionally a new dependency layer of 96 MiB, a year of history is roughly:

```text
app layers:   15 × 52 × 4 MiB      = 3120 MiB
dep layers:    4 × 52 × 96 MiB     = 19968 MiB   (dependencies change ~4x/year)
base layers:   1 × 30 MiB          =    30 MiB
os layers:     1 × 12 + 1 × 22     =    34 MiB
total                              = 23152 MiB = 22.61 GiB
storage cost @ $0.10/GiB-month     = 22.61 × 0.10 = $2.26/month
```

**Egress per deploy.** Each of the 11 nodes per region pulls 164 MiB, two regions: `2 × 11 × 164 = 3608 MiB = 3.52 GiB` per deploy. At fifteen deploys per week: `3.523 × 15 = 52.85 GiB/week`, `× 52 = 2748.3 GiB/year`. At $0.09/GiB that is `2748.3 × 0.09 = $247.35/year` — still small, and the reason egress cost is usually an *availability and latency* problem (F-08) rather than a budget problem at this scale. The arithmetic becomes material at a hundred services or a thousand nodes, where it grows linearly in both.

**Unit cost per request.** Four replicas at 1.5 cores: `6 cores × $0.05 = $0.30/hour`. At 45 req/s that is `162,000 requests/hour`, so `$0.30 / 162,000 = $1.85 per million requests`. At the 90 req/s peak with 11 replicas: `16.5 cores × $0.05 = $0.825/hour` for `324,000 requests/hour` = `$2.55 per million`. The unit cost *rises* at peak because capacity is provisioned for the peak's utilisation target, not because the service is inefficient — a distinction that matters when someone proposes "save money by running at 95% utilisation" (which DC-06 shows converts into a latency cliff).

### 9c. Bottleneck analysis and what measurement would confirm it

| Suspected bottleneck | Signature | Measurement that confirms | Measurement that refutes |
|---|---|---|---|
| Network bandwidth on pull | Pull time scales with image size; CPU idle during pull | Byte counters vs elapsed time during a cold pull | High unpack CPU with low network utilisation |
| Decompression on pull | High CPU during pull, network finishes early | CPU profile during unpack | Network saturated |
| Disk write on pull | I/O wait, slow unpack with free CPU and network | Disk throughput and queue depth | CPU-bound unpack |
| Dependency resolution on build | Build time dominated by the package step even when cached inputs unchanged | Step-level timings with `CACHED` markers | Cached step completing in milliseconds |
| Compression on build | High CPU at layer export | CPU profile during export | Export fast, subsequent step slow |
| Cache invalidation | Repeated full rebuilds with unchanged inputs | Per-step cache-hit log | Cache hit but slow later step |
| CPU quota throttling | p99 latency with low utilisation | `nr_throttled` / `throttled_usec` counters | Counters zero |
| Memory pressure | OOM kills or high reclaim | `oom_kill`, `memory.events` `high`/`max` | Zero counters, latency from elsewhere |
| Registry latency | Long pause before first byte transfers | Time-to-first-byte separate from transfer time | Bytes flow immediately |

The table's purpose is to stop the most common analytical error: attributing a slow pull to "the image is too big" when the image is fine and the unpack is CPU-bound on one small node, or the node's disk is saturated by an unrelated workload.

### 9d. Sensitivity: what moves the answer

```text
Sensitivity of deploy wall time (rolling update, maxSurge 1, 11 replicas):
  T_deploy ≈ replicas × (T_start + probe_readiness_lag)
           = 11 × (11.1 s + 5 s) = 177.1 s   at 11.1 s cold start
  halve cold start to 5.6 s: 11 × 10.6 = 116.6 s   (-60.5 s, -34%)
  image 164 -> 64 MiB:      cold start 11.1 -> 8.6 s, T_deploy 149.6 s (-27.5 s)
  probe period 10 -> 5 s:   lag ~5 -> ~2.5 s, T_deploy 149.6 s (-27.5 s)

Sensitivity of memory safety margin (L = 2048 MiB, b = 1080, w = 130):
  rho = 0.05 -> N = 6 -> RSS 1860 MiB (90.8%)  headroom 188 MiB
  rho = 0.15 -> N = 5 -> RSS 1730 MiB (84.5%)  headroom 318 MiB
  rho = 0.20 -> N = 4 -> RSS 1600 MiB (78.1%)  headroom 448 MiB
  w = 150 (payload grew 15%) with rho = 0.15 -> N = 4 -> RSS 1680 MiB

Sensitivity of replica count to utilisation target (demand 4.05 cores, C = 1.5):
  u* = 0.5 -> n = 6 ; u* = 0.7 -> n = 4 ; u* = 0.9 -> n = 3
  u* = 0.9 gives W = 223.4 ms at c = 3, exceeding a 150 ms budget -> infeasible.
```

The last line is the important one: the cheap configuration is not a trade-off, it is a violation of the stated latency requirement. Sensitivity analysis exists to find these infeasible corners, not to produce a range of equally acceptable options.

**Takeaway:** Container economics are dominated by bytes-per-node and by the utilisation target, not by build compute; the sensitivity that matters is the one that crosses a requirement boundary, and those crossings are visible in the arithmetic before they are visible in production.

## 10. Key Theorems / Results Reference

The results below are what makes the formulas usable outside the specific numbers in this sheet. Each is stated precisely enough to be applied and qualified enough to be rejected when its hypotheses fail.

| Result | Correct statement | Essential hypotheses | Practical implication | What it does not imply | Primary source |
|---|---|---|---|---|---|
| Content addressability | If a name is the cryptographic hash of an object's bytes, two objects with the same name are identical with overwhelming probability, and any modification produces a different name | Collision resistance of the hash function; no name reuse outside the hash | Layers, manifests, and blobs can be deduplicated and cached purely by name; a digest is a sufficient pin | That the name is *authentic* — a hash proves integrity, not provenance; signing is a separate property | OCI image specification, content descriptors (SOURCED, version OPEN) |
| Little's law | For a stable system in steady state, the long-run average number in the system equals the arrival rate times the long-run average time in the system: `L = λW` | Steady state; conservation of arrivals and departures; averages taken over the same window | Converts a concurrency observation into a latency claim and vice versa; allows cross-checking queueing results (`L_q = λW_q`) | Nothing about distributions or tails; it constrains means only | Little (1961), *Operations Research* (SOURCED) |
| Erlang C | For M/M/c with `c` identical servers, infinite buffer, and `λ < cμ`, the stationary probability that an arrival must wait is `C(c, a)` as defined in DC-06, and the mean wait is `C(c,a)/(cμ − λ)` | Poisson arrivals, exponential service, homogeneous servers, work-conserving FCFS, no abandonment, no retries, `λ < cμ` | Justifies capacity planning with a target utilisation and predicts the knee shape in Section 6.3 | Not a tail-latency model; not valid for bursty, self-similar, or heavy-tailed service; gives no bound when `λ ≥ cμ` | Erlang (1917); standard queueing texts (SOURCED, derivation in DC-06) |
| Burke's theorem / M/M/c output | In steady state, the departure process of an M/M/c queue with Poisson input is Poisson with the same rate | Same M/M/c hypotheses, stable system | Justifies chaining queues (service → downstream) as independent stages in a first-order model | Does not hold for general service distributions; do not chain arbitrary queues this way | Burke (1956) (SOURCED) |
| PASTA (Poisson arrivals see time averages) | For a Poisson arrival process, the distribution of the state seen by an arrival equals the time-stationary distribution of the state | Poisson arrivals, independent of system state | Justifies using stationary formulas to describe what arriving requests experience | Fails for non-Poisson or state-dependent arrival processes (e.g. retry storms, feedback) | Wolff (1982) (SOURCED) |
| Union bound for failure tolerance | The probability that at least one of `n` independent replica failures occurs is at most the sum of individual failure probabilities | Independent failures; any failure is a failure of the whole | A cheap upper bound on availability loss from replica count | Not a tight bound; ignores correlated failures, which dominate in practice | Elementary probability (DERIVED) |
| Amdahl-type bound on pull parallelisation | If a fraction `f` of pull work is serial (auth, manifest, index resolution), the speedup from `k`-way parallelism is at most `1/(f + (1−f)/k)` | Fixed problem size; fixed serial fraction; ideal parallel scaling of the rest | Explains why many parallel layer pulls do not scale linearly and why manifest round-trips matter | Does not model bandwidth contention, which is often the real limiter | Amdahl (1967) (SOURCED) |
| Copy-on-write identity | In an overlay with read-only lower layers, a file's contents in the lower layer are immutable for the container's lifetime; a modification creates a copy in the writable layer | Overlay semantics with copy-up; no shared writable layer between containers | Justifies DC-08 and the "state in the writable layer is disposable" invariant | Does not describe volume semantics, which follow the volume's own driver contract | Filesystem semantics (DERIVED from the model; driver specifics OPEN) |
| Cgroup limit enforcement | A memory limit kill occurs when the cgroup's charged memory cannot be reclaimed; a CPU quota delays runnable tasks once the period's quota is consumed | Kernel cgroup controller enabled; chargeable memory accounting includes page cache | Explains 137 exits and throttling as *accounting* events, not as process bugs | Does not say *which* allocation caused it; attribution requires per-process or per-allocation evidence | Kernel cgroup documentation (SOURCED, v1/v2 differences OPEN) |
| Signal disposition | `SIGKILL` and `SIGSTOP` cannot be caught, blocked, or ignored; other signals can be caught, and default dispositions vary | POSIX process semantics | Justifies grace-period design and explains why 137 cannot be handled by the application | Does not say whether a *given* shell forwards signals — that is shell-specific behaviour | POSIX (SOURCED) |
| PID 1 signal semantics in containers | A process running as PID 1 in a PID namespace has default signal dispositions altered, so handlers must be installed explicitly | Linux PID namespace; process is namespace init | Explains "my app ignores SIGTERM as PID 1 but not as PID 2" | Does not apply outside a PID namespace | Kernel namespace documentation (SOURCED, version OPEN) |
| Dedup bound on registry growth | Storage for `M` manifests is bounded by the union of their layer digests, so growth is `O(unique blobs)` and not `O(M × image size)` | Digest-addressed storage; no per-manifest re-compression | Justifies DC-07 and retention policy based on referenced digests | Does not bound *metadata* growth or the cost of garbage collection scans | DERIVED from content addressability |

Two entries deserve a warning because they are the ones most often over-claimed.

**Erlang C is not a p99 model.** A team that reads "78 ms mean" and assumes the p99 is "about 100 ms" is using a mean-value formula to make a tail claim. The queue's variance determines the tail, and for real traffic the tail is worse than the model's. The correct use is capacity: choose `c` so that utilisation leaves room, then *measure* the tail.

**Little's law is not a performance model.** It is an identity that holds in any stable system, so it can never be refuted by measurement and can never explain anything by itself. Its value is as a *consistency check*: if `L` and `λW` disagree in a measurement, the measurement window or the definitions are wrong. That is precisely how it was used in Section 5.6.

**Takeaway:** Use Little's law to check, Erlang C to size, content addressability to deduplicate, and cgroup semantics to explain kills — and never use a mean-value theorem to make a tail guarantee.

## 11. Interview / Exam Questions

Fifteen questions across recall (Q1–Q3), derivation/application (Q4–Q9), and judgement/design/debug (Q10–Q15). Each has a checked answer, the reasoning that earns credit, a follow-up, an acceptable alternative, a red flag, and scoring.

### Q1 — Recall: what does a container share with the host, and what does it not?

**Answer.** It shares the host kernel — one kernel, one scheduler, one memory manager, one network stack implementation. It does not share the filesystem view (mount namespace), the process view (PID namespace), the network interface set (network namespace), or the resource envelope (cgroups). It also does not get a separate kernel, which is the VM distinction.

**Reasoning.** The sharing model determines the threat model: a kernel vulnerability reachable from container code is a host vulnerability, which is why containers alone are not a security boundary for untrusted code.

**Follow-up.** Which additional mechanisms reduce the reachable kernel surface? (Seccomp syscall filtering, capability dropping, LSM policy, user namespaces.)

**Acceptable alternative.** Framing it as "OS-level virtualisation" and then enumerating namespaces and cgroups explicitly.

**Red flag.** Calling a container a lightweight VM. That answer predicts incorrect behaviour for kernel exploits, for `sysctl` changes, and for memory accounting.

**Scoring.** 0: "it's like a VM"; 1: mentions sharing the kernel but no mechanisms; 2: names namespaces and cgroups; 3: adds the security consequence and the mitigations.

### Q2 — Recall: why is a tag insufficient to identify a release?

**Answer.** A tag is a mutable pointer that a registry can reassign to a different manifest. Two deploys that both say `app:1.4.2` can run different bytes if the tag was rebuilt or moved. A digest names the manifest's exact bytes, so it cannot be silently reassigned without changing the name.

**Reasoning.** Identity requires a name that is a function of content. Tags are human bookkeeping; digests are identity.

**Follow-up.** What must a release record contain so that a rollback is possible? (The digest, plus the build inputs and the runtime configuration.)

**Acceptable alternative.** Noting that some registries support immutable tags, which makes a tag a *proxy* for identity — but only if immutability is enforced and verified, not merely configured.

**Red flag.** "We always tag with the commit SHA, so it is unique." A commit SHA tag can still be overwritten by a rebuild of the same commit with different dependency resolution.

**Scoring.** 0: tags are unique; 1: mutable but no consequence; 2: digest explanation; 3: adds rollback/audit consequence and mutable-tag caveat.

### Q3 — Recall: state the difference between readiness and liveness, and the failure each prevents.

**Answer.** Readiness answers "should this instance receive traffic now?" and its failure removes the instance from the endpoint set without restarting it. Liveness answers "is this process healthy enough to keep?" and its failure restarts the container. Readiness prevents traffic to a booting or dependency-blocked instance; liveness prevents a wedged process from lingering forever.

**Reasoning.** The two decisions have different costs: removing traffic is cheap and reversible; restarting destroys state and cold-starts the process.

**Follow-up.** Why does wiring liveness to a dependency check cause a restart storm?

**Acceptable alternative.** Describing startup probes as a third category that protects the liveness budget during boot.

**Red flag.** "Both check `/health`, so one endpoint is enough." One endpoint returning 200 cannot express two different decisions.

**Scoring.** 0: conflates them; 1: defines both without the failure each prevents; 2: both plus mechanisms; 3: adds the restart-storm mechanism and startup-probe nuance.

### Q4 — Apply: compute the safe worker count.

**Given.** Limit 3 GiB, baseline RSS 1400 MiB, per-worker peak 210 MiB, reserve 12%.

**Answer.**

```text
L        = 3 × 1024 = 3072 MiB
reserve  = 0.12 × 3072 = 368.64 MiB
budget   = 3072 − 368.64 − 1400 = 1303.36 MiB
N_max    = floor(1303.36 / 210) = floor(6.206) = 6 workers
RSS      = 1400 + 6×210 = 2660 MiB = 86.6% of limit; headroom 412 MiB
```

**Reasoning.** Two subtractions before the division: the baseline is not optional and the reserve covers transients. The floor is required because a fractional worker does not exist and rounding up spends the reserve.

**Sanity check.** `6 × 210 = 1260 ≤ 1303.36`, and `7 × 210 = 1470 > 1303.36`, so 6 is maximal.

**Follow-up.** Per-worker memory rises to 260 MiB after a payload change. New answer: `floor(1303.36/260) = 5`, and the lost concurrency must be recovered with a replica.

**Acceptable alternative.** Choosing 5 for extra safety, provided the reduction is stated as a policy choice rather than presented as the formula's output.

**Red flag.** `floor(3072/210) = 14`, which ignores both the baseline and the reserve and would need 4340 MiB.

**Scoring.** 0: naive division; 1: subtracts baseline only; 2: correct with reserve and floor; 3: adds the check and the payload-change follow-up.

### Q5 — Apply: derive the replica count including failure tolerance.

**Given.** 72 req/s, 0.08 core-seconds per request, 2 cores per replica, target utilisation 0.65, must survive 2 replica failures.

**Answer.**

```text
demand       = 72 × 0.08 = 5.76 cores
usable/rep   = 2 × 0.65 = 1.30 cores
capacity-only: ceil(5.76/1.30) = ceil(4.43) = 5 replicas
failure rule: (n − 2) × 1.30 ≥ 5.76  ->  n − 2 ≥ 4.43  ->  n − 2 = 5  ->  n = 7
check: 5 × 1.30 = 6.50 ≥ 5.76, spare 0.74 cores (12.8% headroom)
```

**Reasoning.** Capacity and tolerance are separate constraints, and the binding one is the maximum of the two answers. Solving the tolerance constraint directly is cleaner than adding replicas iteratively.

**Follow-up.** If the failure domain is a whole availability zone holding three replicas, the "2 failures" model is wrong and the zone topology must be modelled instead.

**Acceptable alternative.** Computing 5, then noting "plus 2 for tolerance = 7", which is valid only when the per-replica capacity is unchanged and utilisation is recomputed.

**Red flag.** Answering 5 and calling it resilient because "the orchestrator will restart them" — a restart takes the cold-start time, during which demand is unserved.

**Scoring.** 0: no failure consideration; 1: adds 2 blindly; 2: solves the tolerance constraint; 3: adds the zone/failure-domain caveat.

### Q6 — Apply: recompute the cache ordering result.

**Given.** Two stages: `COPY requirements.txt` + `RUN pip install` (cost 90 s, changes 5% of builds), and `COPY src/` (cost 6 s, changes 95% of builds).

**Answer.** Manifest-first: `P(stage 1 runs) = 0.05 → 0.05×90 = 4.5 s`; `P(stage 2 runs) = 1 − 0.95×0.05 = 1 − 0.0475 = 0.9525 → 0.9525×6 = 5.715 s`; `E[T] = 4.5 + 5.72 = 10.22 s`. Source-first: `P(stage 1 runs) = 0.95 → 0.95×90 = 85.5 s`; `P(stage 2 runs) = 1 − 0.05×0.95 = 0.9525 → 5.715 s`; `E[T] = 85.5 + 5.72 = 91.22 s`. Manifest-first is `91.22/10.22 = 8.9×` faster.

**Reasoning.** The cumulative product term is what makes ordering dominant: a volatile early stage forces every later stage to rerun.

**Sanity check.** In source-first, the 90 s dependency install reruns on 95% of builds, dominating the expectation — which matches the stated change rates.

**Follow-up.** A base digest is pinned *and* rotated weekly on a schedule. Model that as a third stage below both with `q = probability a build falls after a rotation`, and note it invalidates everything above it.

**Acceptable alternative.** Computing expected time per 100 builds (`1022 s` vs `9143 s`) and dividing; the ratio must agree at 8.9×.

**Red flag.** "Both totals are around 96 seconds of stage cost, so ordering does not matter." That answer sums `c_i` and ignores the probabilities, which is the entire mechanism.

**Scoring.** 0: sums costs; 1: one ordering computed; 2: both, correct; 3: adds the base-rotation generalisation.

### Q7 — Apply: Little's law consistency check.

**Given.** A service reports 42 req/s, mean latency 120 ms, and an average of 3.1 requests in flight measured at the load balancer. Is the measurement self-consistent?

**Answer.** Predicted `L = λW = 42 × 0.120 = 5.04`. Observed 3.1. The discrepancy is 1.63×, so either the window, the population, or the definition differs. Candidate causes: latency measured only for successful requests (excludes timeouts), or in-flight counted at the balancer while some requests queue before it, or the averages taken over different windows.

**Reasoning.** Little's law is an identity, so a violation is never a physical anomaly — it is a measurement-definition error. That makes it a powerful audit tool.

**Follow-up.** How would you reconcile? Count all admitted requests including failures, align windows, and count in-flight from admission to completion.

**Acceptable alternative.** Noting that a non-stationary window (traffic ramping) can produce apparent violations without a definition error, which is why the check is only valid in steady state.

**Red flag.** "The system is faster than the model predicts, so the model is wrong." Little's law has no adjustable parameters; the measurement is what is inconsistent.

**Scoring.** 0: accepts the numbers; 1: notes a discrepancy without a cause; 2: computes and names definition error; 3: adds the non-stationarity caveat.

### Q8 — Apply: is four replicas enough at the peak?

**Given.** Peak 88 req/s, service time 55 ms, four replicas, latency budget 150 ms p50.

**Answer.**

```text
mu = 1/0.055 = 18.18 req/s per replica
a  = 88/18.18 = 4.84 erlangs;  rho = 4.84/4 = 1.21 > 1
```

Utilisation exceeds 1, so the queue has **no stationary distribution**. The correct answer is "no finite steady-state result exists at this arrival rate"; the queue grows without bound until arrivals fall. Four replicas are insufficient; the arithmetic minimum is `ceil(4.84) = 5`, but at `ρ = 0.968` the Erlang C mean wait is already 316.6 ms (sojourn 371.6 ms), far outside a 150 ms budget. The defensible number is 7: `ρ = 0.691`, `C(7, 4.84) ≈ 0.2872`, `W_q = 0.2872/(7×18.18 − 88) = 7.31 ms`, `W = 62.3 ms`.

**Reasoning.** Recognising `ρ > 1` is the whole question. Producing a latency number here is the error; naming the undefined regime is the answer.

**Follow-up.** Why is 5 replicas technically feasible but operationally wrong? Because `cμ − λ = 90.9 − 88 = 2.9 req/s` of surplus: the queue is stable in theory but drains so slowly that any burst of a few seconds leaves a backlog that takes minutes to clear.

**Acceptable alternative.** Simulation instead of a closed form, provided the simulation's stationarity check (queue not growing) is reported.

**Red flag.** Plugging into Erlang C with a negative denominator or reporting "about 500 ms". Both are extrapolations outside the domain.

**Scoring.** 0: a latency number; 1: notices high utilisation; 2: identifies `ρ ≥ 1`; 3: adds the minimum and defensible counts.

### Q9 — Apply: compute pull time for a rolling deploy on a shared uplink.

**Given.** 8 nodes, each pulling a 200 MiB image missing all layers, shared 100 MiB/s uplink at the registry, 0.2 s fixed overhead per client, 0.05 s unpack per layer (6 layers).

**Answer.** Aggregate bytes `8 × 200 = 1600 MiB`; aggregate transfer `1600/100 = 16.0 s` if bandwidth is shared fairly and pulls overlap. Per-client view: each sees roughly `200/(100/8) = 16 s` of transfer plus `0.2 + 6×0.05 = 0.5 s`, so approximately 16.5 s, all finishing near the same time. If the pulls are serialised by a deploy controller instead, wall time is `8 × (2.0 + 0.5) = 20 s`.

**Reasoning.** Bandwidth is a shared resource, so the per-client bandwidth is the aggregate divided by concurrency. Adding per-client times as if independent is the classic error; the aggregate byte count over the shared rate is the invariant.

**Follow-up.** With a pull-through cache on the node network, the registry sees one 200 MiB transfer and the nodes see local transfers — the arithmetic changes from `N × S/β_registry` to `S/β_registry` plus local fan-out.

**Acceptable alternative.** Reporting the aggregate-time bound and noting that real completion times differ by layer scheduling.

**Red flag.** `8 × 2.0 = 16 s` presented as "each node takes 2 s, so the deploy takes 2 s".

**Scoring.** 0: per-client times summed or minimised; 1: aggregate bytes but no time; 2: aggregate time; 3: adds serialisation and cache scenarios.

### Q10 — Judge: production CPU is 20% but p99 is bad. Diagnose.

**Answer.** First discriminate: read `nr_throttled` and `throttled_usec` for the cgroup. Non-zero with growing `throttled_usec` establishes quota throttling (`DC-10`): a burst consumed the period's quota and the rest waited. If zero, the cause is elsewhere — node-level CPU contention, an I/O stall, a downstream dependency, or GC pauses. Only after excluding those should the limit be raised.

**Reasoning.** Average utilisation and quota enforcement are different quantities. A container can average 20% and be throttled 90% of a single 100 ms window.

**Follow-up.** What limit change is justified if throttling is confirmed? Either raise the quota or reduce period-burstiness by reducing concurrency, since concurrency is what converts a burst into a throttled period.

**Acceptable alternative.** Profiling the process first to find whether GC or a batch loop creates the burst; both are legitimate starting points if the counters are checked.

**Red flag.** Raising the memory limit, or raising CPU without reading the counters. The first fixes nothing; the second may hide a downstream problem.

**Scoring.** 0: "raise the limit"; 1: names throttling as a possibility; 2: names the counters and the discrimination; 3: adds the non-CPU alternatives and the concurrency fix.

### Q11 — Judge: an image passes its vulnerability scan but carries a live credential. How?

**Answer.** The scanner inspects packages, not arbitrary files, so a credential in an environment variable, a config file, or a layer that was later deleted is invisible to package scanning. Detection requires three distinct checks: image config inspection (env/labels), layer content scanning per layer, and build-history inspection. The credential must be revoked first, because the image may already be mirrored or public; deletion from the registry does not un-leak it.

**Reasoning.** "Clean scan" answers a specific question ("are there known-vulnerable packages?") and is routinely read as a broader assurance ("this image is safe"). The gap between the two is where credentials live.

**Follow-up.** Why does deleting the file in a later `RUN` not remove it? Because layers are immutable; the earlier layer still contains the bytes and is still pulled.

**Acceptable alternative.** Noting that some scanners do include secret rules, in which case the question becomes whether the rules covered config and history, not just files.

**Red flag.** "The scan passed, so we are fine" or "we deleted the file in the next line".

**Scoring.** 0: scan is sufficient; 1: knows deletion is incomplete; 2: names the three inspections; 3: adds revoke-first and mirror exposure.

### Q12 — Judge: the same digest behaves differently in two environments. How?

**Answer.** If the digests are truly identical, the difference is in the runtime contract: environment variables, mounted config, secret versions, DNS/resolver configuration, time zone, architecture (a digest can name a multi-arch index that resolves differently per platform), kernel or cgroup version, or a run-time dependency fetched by the application. Start by confirming the digest *and* the resolved platform, then diff the runtime configuration schema, then check whether the app fetches anything at run time.

**Reasoning.** Content addressability makes the artifact identical, which forces the explanation into the runtime inputs — including the run-time network, which is the input people forget.

**Follow-up.** What log line would have made this diagnosis trivial? One printing digest, platform, config schema version, and dependency endpoints at startup.

**Acceptable alternative.** Treating a multi-arch index tag as "the same image" and finding the platform difference first — valid when the deployments use different architectures.

**Red flag.** "Impossible — same digest means same behaviour." Same bytes, different environment, different behaviour is the normal case.

**Scoring.** 0: dismisses the premise; 1: names env vars only; 2: enumerates runtime inputs; 3: adds platform resolution and run-time fetches.

### Q13 — Judge: a team wants to run a database in a container with its data in the writable layer "for simplicity".

**Answer.** Reject it, with a specific reason: the writable layer is destroyed with the container, so any replacement — deploy, reschedule, crash, node drain — loses the data. If a single-node database must run in a container, the data directory must be a named volume or bind mount with a backup and restore procedure that has been executed at least once. If the data must survive node loss, the correct answer is an external managed store, because a single-node volume does not survive the node.

**Reasoning.** The claim "simpler" is true only until the first replacement. Durability is not a configuration toggle; it is a property of where the bytes live.

**Follow-up.** What test proves durability? Create data, destroy and recreate the container from the same spec, assert the data is present; then repeat with a different node.

**Acceptable alternative.** Accepting a containerised database for disposable test data, explicitly labelled ephemeral, with no durability claim.

**Red flag.** "We take snapshots" without a restore test — an untested backup is a hypothesis.

**Scoring.** 0: accepts; 1: says "use a volume" without lifecycle; 2: names the failure and the volume; 3: adds restore testing and node-loss distinction.

### Q14 — Judge: what breaks first if traffic doubles overnight?

**Answer.** Several things break in a specific order, and naming the order is the answer. First, replica CPU: at 90 req/s demand is 8.10 cores against 6.0 provisioned, so utilisation exceeds 1.0 and latency grows without bound. Second, workers: five workers at 60 ms cap at ~83 req/s per replica, so queues form inside the container even with adequate CPU. Third, connections and file descriptors, then downstream connection pools, then memory as queues grow. The correct response is not one change: replicas must increase to at least 11 (with failure tolerance, more), and the worker pool must be re-derived from a new memory limit, or requests must be shed.

**Reasoning.** Each resource has its own capacity formula and its own knee. Doubling traffic that is already near one knee crosses several simultaneously.

**Follow-up.** Which change is cheapest? Replicas, because they scale horizontally without touching per-container arithmetic — but only until the downstream pool saturates.

**Acceptable alternative.** Shedding load at the edge as the first action, with capacity added in parallel. Defensible when the queue would otherwise collapse.

**Red flag.** "Add replicas" as a complete answer, which leaves the worker cap and downstream pools untouched.

**Scoring.** 0: one resource; 1: two resources unordered; 2: ordered with arithmetic; 3: adds the response plan and the downstream limit.

### Q15 — Judge: how do you verify the five claims before a release?

**Answer.** Each claim has one test. (1) Same bytes: compare the recorded build digest with the digest the runtime reports, per environment. (2) Reproducible: build from a fresh clone, network-restricted, with the recorded inputs, and compare manifests; a differing manifest is a dependency or timestamp leak. (3) No credential: scan config, every layer, and build history of the *built artifact*. (4) Fits at peak: drive upper-decile payloads at `N_max` under the production limit for thirty minutes; assert `oom_kill = 0`, throttle ratio within a stated bound, and RSS below `L(1−ρ)`. (5) Traffic only when ready: start with the dependency unavailable; assert readiness false, liveness true, no restart.

**Reasoning.** Each claim is empirical, so each needs an executed test rather than a review of the Dockerfile. A release gate that runs these five is a stronger guarantee than any amount of static inspection.

**Follow-up.** Which test is most often skipped? The dependency-unavailable probe test and the restore test, both because they require deliberately breaking something.

**Acceptable alternative.** Running the five as a pre-release checklist with recorded evidence per release, which is what most teams can sustain.

**Red flag.** "We reviewed the Dockerfile." Review is not execution, and four of the five claims are about runtime behaviour.

**Scoring.** 0: no tests; 1: one or two claims; 2: all five with tests; 3: adds the skipped-test insight and evidence retention.

**Takeaway:** Strong answers compute with the right formula inside its domain, name the counter that discriminates between causes, and say what test would change their mind.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Treating a container image as a pinned artifact and sizing the runtime that executes it: build ordering, layer economics, secret handling, worker and replica counts, probe wiring, cost |
| Core formulas | `T_pull = Σs/β + m + Σt` · `E[T_build] = Σ[1 − Π(1−q_j)]c_i` · `N = floor((L(1−ρ) − b)/w)` · `n = ceil(λc_cpu/(C u*))` · `W_q = C(c,a)/(cμ−λ)` · `S_stored = Σ_{∪D} s_d` |
| Key assumption | Per-worker memory is additive and bounded, and traffic is steady enough that a utilisation target means something; both are routinely false under bursts and unbounded payloads |
| Complexity | Build `O(context bytes)` hashing plus per-step work; pull `O(missing layer bytes / β)`; dedup `O(M·k)` metadata; runtime cost dominated by bytes-per-node and the utilisation target, not build compute |
| Use when | Deciding a base image, ordering build stages, choosing where state lives, setting a memory limit or CPU quota, computing replicas under a latency budget, or costing a registry |
| Avoid when | The requirement is a *tail* guarantee — use measured percentiles, not Erlang C means; or the workload is IO-bound, where CPU arithmetic is irrelevant; or failures are correlated, where the union bound overstates tolerance |
| Diagnose | 137 → read cgroup `oom_kill` first, then grace-period timing; slow rebuild → first `CACHED` marker; throttling → `nr_throttled`; works-locally → digest, then platform, then env/mount diff |
| Go deeper | [Python async and concurrency](01_Python_Async_and_Concurrency.md) for worker-pool and backpressure mechanics; [REST APIs and WebSockets](02_REST_APIs_and_WebSockets.md) for the idempotency that makes retry-safe draining possible; orchestration and service-mesh policy are explicitly out of scope for this collection |
