---
collection: "ai-platforms-data-systems-production-engineering"
file_id: "01"
topic: "Distributed Training Checkpoint and Recovery"
level: 3
status: "draft"
version: "0.1"
created: "2026-11-07"
reviewed: null
review_trigger: "Source availability, framework or storage semantics change"
notation_style: "plain-text"
prerequisites: ["Distributed training basics", "Optimizer and data-loader fundamentals"]
source_status: "OPEN"
owner_role: "Training infrastructure maintainer"
source_index: "INDEX.md"
---
# 1. File Header

**Outcome:** Design, size, restore, and diagnose a durable distributed training checkpoint without confusing a weight export with a restartable training state.
**Audience:** Training and platform engineers operating data-, tensor-, pipeline-, or optimizer-sharded training.
**Scope:** Checkpoint state, a consistent cut, durable publication, resume, elastic membership, storage economics, and fault diagnosis.
**Evidence:** All configurations, incidents, and numerical values here are illustrative; none are measurements of a real deployment.
**Source limitation:** The directory inventory names `finetuning_course/Module_19_Distributed_Training_Infrastructure/19.5_Fault_Tolerance_and_Elastic_Training.md` and `finetuning_course/Module_14_MLOps_Practical_Workflows/14.3_Checkpointing_and_Fault_Tolerance.md`, but their contents are not present in this workspace.
Consequently this draft is informed by the topic names and the references below, not a verified summary of those files.
Do not treat framework-specific behavior as guaranteed without checking the installed release and testing the storage implementation.
**Prerequisites:** Optimizer steps, gradient accumulation, pseudo-random generators, distributed ranks, and durable storage basics.
**Related course home:** `../24_distributed_training_data_pipelines_and_ml_compute/`.
**External references:** [PyTorch distributed checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html), [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [PyTorch data loading](https://docs.pytorch.org/docs/stable/data.html), [torchrun elastic launch](https://docs.pytorch.org/docs/stable/elastic/run.html), and [PyTorch serialization security](https://docs.pytorch.org/docs/stable/notes/serialization.html).
These are pointers for release-specific verification, not evidence that the illustrative examples were executed.
**Takeaway:** A restart contract includes both computational state and a trustworthy publication protocol.

# 2. Concept Map / Mental Model

```text
training step -> consistent safe point -> immutable shard writes
                                      -> completeness validation
                                      -> committed manifest / pointer
failure -> latest eligible committed manifest -> validate -> restore -> train
```
The arrows denote dependencies and data flow, not claims that each operation is atomic.
The optimizer update is one logical transition; a saved state must identify which updates are included.
At a step boundary, model parameters, optimizer buffers, counters, scheduler, and input position describe the same prefix of training.
A set of files alone is not a checkpoint: only a validated, published manifest identifies a recoverable generation.
A durable generation may exist before it becomes the selected latest generation.
An older complete generation remains the fallback when publication of the next one fails.
An elastic restart may restore equivalent tensors while changing data order and numerical trajectory.
**Takeaway:** Separate logical consistency, physical durability, and discoverability.

# 3. Foundational Definitions

| Term | Meaning and unit/type | Boundary or confusion |
|---|---|---|
| Generation | Immutable logical snapshot identified by a unique ID | Not merely a directory listing |
| Global optimizer step | Integer number of completed parameter updates | Not microbatch count |
| Microbatch | Input subset used for one forward/backward pass | May not trigger an update |
| Accumulation phase | Number of completed microbatches in the current optimizer update | Must be saved with gradients or reset at a safe boundary |
| Rank | Process identity within a worker group | Rank can change across elastic restarts |
| World size | Number of participating processes in the relevant group | Different parallel groups may have different sizes |
| Manifest | Immutable metadata naming expected shards, sizes, hashes, state and schema | Not a substitute for validating the referenced objects |
| Commit | Publication of a generation as eligible for recovery under a documented atomicity protocol | A successful shard write alone is not a commit |
| RPO | Recovery point objective: acceptable lost progress, expressed in time, samples, or steps | It is an objective, not necessarily achieved worst-case loss |
| RTO | Recovery time objective: acceptable elapsed outage until resumed useful work | Excludes no phases unless the objective explicitly says so |
| RNG | Pseudorandom generator state, including CPU, accelerator and worker streams when relevant | Seed alone does not capture current stream position |
| Global sample cursor | Logical next training position in a versioned sampling plan | A worker prefetch cursor is not necessarily consumed progress |
| Shard | Physical partition of model or optimizer state | A rank's shard is not necessarily reusable by the same rank after resize |
| GiB | 2^30 bytes | Not the decimal GB of 10^9 bytes |
| Effective bandwidth | Measured payload bytes completed per elapsed second for the stated workload | Link peak or sum of unshared rank rates is not a measured effective rate |

Checkpointing is a consistency problem as much as a serialization problem.
A weight-only artifact is useful for inference or initialization but normally does not reproduce an optimizer trajectory.
A fully restartable checkpoint includes every mutable input to the next logical update or declares an explicit loss of continuity.
**Takeaway:** Name the unit, scope, and recovery guarantee of every counter.

# 4. Core Formulas with Full Derivation

## 4.1 Simplest / Base Case

**Mechanism and validity:** One writer, a durable store, immutable generation objects, and an atomic or conditional publication primitive are assumed.
Input is the state immediately after a completed optimizer update and the identifier of the previously committed generation.
Output is either the old committed generation or a new complete generation; never a published partial generation.
Transition: write a unique generation; flush or complete upload according to the storage durability contract; verify all objects; write its manifest; atomically publish its identifier.
The old pointer remains readable before publication; after publication, the new pointer names only a validated manifest.
If the writer dies before publication, orphan objects are not selected and can later be garbage-collected.
If the writer dies after successful publication, recovery must select the new generation only if it passes the manifest and content checks.
Atomic rename is relevant only when the actual filesystem provides the required same-filesystem semantics and durability; a rename alone is not universally durable.
On object storage, use provider-documented conditional creation or compare-and-swap with a fencing token; do not assume directory rename is atomic.
Concurrent coordinators need a lease or monotonic fencing mechanism enforced at publication; a stale writer must not overwrite a newer pointer.
Termination requires a bounded timeout and an abort path; unavailable storage can make publishing impossible.
Boundary: if clients can observe a pointer before referenced objects are durable and readable, the protocol does not establish recoverability.

**Validity for interval model:** Checkpoints take a constant non-overlapped time C, failures follow a Poisson process of rate lambda, and rare failures make repeated-loss interactions negligible.
Derivation: periodic checkpoint overhead is approximately C/T per unit useful time; uniformly distributed failure time within an interval loses on average T/2 useful time, at rate lambda.
Differentiate C/T + lambda*T/2 and set its derivative to zero.
```text
FORMULA [I-1]: T* = sqrt(2*C/lambda)
WHERE: T* useful-work interval [s]; C checkpoint pause [s]; lambda failure rate [1/s].
CONSTRAINT: stationary Poisson failures, rare failures, fixed C, no overlap; approximation, not an RPO guarantee.
NUMERICAL EXAMPLE: C=60 s; lambda=1/86400 per s; T*=sqrt(10368000) ≈ 3220 s ≈ 53.7 min.
INTERPRETATION: use as an initial candidate; verify its worst-case progress loss against the RPO.
```
Boundary: lambda=0 makes the formula undefined; increasing checkpoint duration, time-varying hazards, correlated outages, or asynchronous writes require a different model.

## 4.2 General Case

Multiple ranks first agree on a logical cut and a unique generation ID.
Each rank materializes the state assigned to its rank and reports completion, size, and digest to a coordinator.
The expected shard set and layout are determined from a schema and parallelism description, not inferred from whichever files happened to arrive.
The coordinator validates all required shard acknowledgements and fails the generation if any are missing or inconsistent.
The manifest identifies schema, topology, step, tensor metadata, object identities, sizes, and digests.
Publication must fence older coordinators and preserve at least one earlier committed generation.
A barrier may align process progress but does not itself establish durable storage or verify bytes.
The write may involve collectives, so all required ranks must call them in the same order.
Asynchronous serialization requires an immutable snapshot or copy-on-write mechanism: writing live buffers during later updates may mix logical steps.
Cancellation of a writer does not necessarily cancel already-issued storage operations; abandoned generation IDs must not be reused.
A generation ID plus per-object digest permits detecting a stale or mixed upload, assuming digest verification and suitable collision resistance.

**Validity for throughput sizing:** S is total payload bytes, B is sustained effective aggregate payload bandwidth, and overheads are not included.
Derivation: rate equals data divided by elapsed time; rearrange to obtain elapsed payload time.
```text
FORMULA [IO-1]: t_payload = S / B
WHERE: S [GiB], B [GiB/s], t_payload [s].
CONSTRAINT: constant effective aggregate rate, no competing traffic, no compression or fixed overhead.
NUMERICAL EXAMPLE: S=240 GiB, B=4 GiB/s; t_payload=60 s.
INTERPRETATION: 60 s is a payload-only estimate, not a completion-time guarantee.
```
Boundary: zero throughput makes completion unbounded; shared bottlenecks and small-object request latency can dominate.

## 4.3 Key Special Cases / Variants

| Variant | State or mechanism | Appropriate when | Disqualifier |
|---|---|---|---|
| Replicated model | One complete model copy may be saved if synchronized | Replicas are known identical at the cut | Optimizer or RNG state remains rank-local |
| Sharded model and optimizer | Save all shards plus layout and metadata | State does not fit on one process | Missing shard or unsupported resharding |
| Synchronous save | Training pauses until save completes | Simplicity and deterministic cut matter | Pause violates availability or throughput target |
| Asynchronous save | Snapshot first, then write in background | Sufficient memory and I/O isolation exist | Mutable buffers leak later updates into snapshot |
| Full snapshot | Every generation is self-contained | Simple recovery and retention are priorities | Repeated full I/O exceeds budget |
| Incremental snapshot | Changed content plus dependency chain | Low change rate and chain-aware tooling | Missing base or unbounded chain makes restore fragile |
| Preemption save | Save in response to a warning | Notice is long enough for a durable commit | Signal arrives too late or shutdown kills writers early |
| Elastic restart | Load with a different allocation | Checkpoint format and optimizer support resharding | Exact data trajectory or fixed pipeline topology is required |

## 4.4 Asymptotic / Limiting Behaviour and Counterexample

A short interval reduces uncheckpointed work but eventually makes checkpoint overhead dominant.
A very long interval reduces write frequency but can violate the maximum acceptable loss of progress.
If checkpoint data grows without a corresponding increase in effective bandwidth, save time grows linearly under IO-1.
Adding writers cannot overcome a fixed shared backend limit and may worsen request contention.
An asynchronous checkpoint can overlap compute but cannot guarantee that a requested recovery point is already committed.
A single completed shard is a counterexample to the claim that a checkpoint exists: the remaining shards may be absent.
A changed world size is a counterexample to blindly restoring rank-numbered sampler offsets: rank identity and partition sizes may change.
**Takeaway:** Pick intervals and layouts from failure, consistency, and storage constraints together.

# 5. Worked Numerical Example

**Evidence:** Entirely illustrative; no benchmark or production incident is claimed.
A job has eight ranks, each emitting 30 GiB of nonoverlapping checkpoint payload.
It produces a full snapshot every 40 minutes of useful training.
The aggregate storage service sustains 4 GiB/s for this workload, assumed constant.
All ranks reach a safe update boundary before snapshot creation.
The commit protocol requires every rank's object, a validated manifest, and conditional latest-pointer publication.
Total payload = 8 ranks × 30 GiB/rank = 240 GiB.
IO-1 gives 240 GiB / 4 GiB/s = 60 s of payload transfer.
If synchronization, hashing, metadata, and commit take an additional 15 s without overlap, total pause = 75 s.
With a 2,400 s useful-work interval, modeled pause fraction of elapsed time = 75 / (2400 + 75) ≈ 3.03%.
That fraction assumes one successful checkpoint per cycle and no overlapping work.
At a uniformly random failure time during useful work, average lost useful work is 1,200 s = 20 min under the simple model.
The useful-work loss immediately before a new successful checkpoint is near 40 min, plus any other uncommitted work or pause depending on the RPO definition.
A failure during checkpoint writing must fall back to the previous committed generation, even if seven of eight shards are complete.
If the backend is only 2 GiB/s, payload transfer rises to 120 s, and the same 15 s overhead yields 135 s pause.
Then modeled pause fraction is 135 / (2400 + 135) ≈ 5.33%.
An independent check is that halving bandwidth doubles the payload component, not necessarily total latency.
A wrong calculation would multiply 240 GiB by eight again, although 240 GiB already includes all ranks.
Another wrong calculation would call 60 s a measured p99; it is an assumed deterministic payload-only estimate.
Measure actual bytes, latency distributions, backend contention, hash overhead, and commit failures before capacity decisions.
**Takeaway:** Sum unique shards once and include the entire commit path when estimating interruption.

# 6. Visual / Geometric Intuition

```text
step:       1000                 1040                  1080
training:   [committed A]--------work--------[write B]--work
storage:    A complete                         B partial X
recovery:   <---------------- select A, replay lost work
```
The time line shows a failure before B publishes; the pointer remains A.
After B publishes, recovery selects B only after validation.
The width of the uncommitted interval represents potentially lost useful work, not guaranteed wall-clock outage.
For sharded state, imagine a jigsaw whose manifest lists every required piece and its content digest.
A jigsaw with all pieces from different steps is not a consistent picture.
The same logical tensor can be repartitioned into a different number of pieces only if metadata and restore tooling support that mapping.
**Takeaway:** A consistent snapshot is a cut across all participating state, not a timestamp attached to one file.

# 7. Comparison / Selection Table

| Decision | Choose when | Avoid when | Fallback |
|---|---|---|---|
| Step boundary | Gradients can be discarded after an update | Checkpoint trigger demands arbitrary interruption | Recompute from the last boundary |
| Mid-accumulation | Gradients and phase are fully captured consistently | Gradient buffers cannot be snapshotted reliably | Finish or abandon the accumulation window |
| Rank-zero full save | Full materialization fits and synchronization is verified | Sharded optimizer cannot be gathered safely | Distributed shard save |
| Distributed checkpoint | Backend and framework support shard metadata and restore | Required resharding is untested | Restore same topology or offline convert |
| Local fast tier plus durable tier | Node loss is addressed by a separate durable copy | Local copy is mistaken for durable recovery | Publish only after durable tier completion |
| Weight export | Inference or warm start is intended | Continuation of optimizer trajectory is required | Full training-state snapshot |
| Latest pointer | Atomic conditional update is available | Split-brain writers cannot be fenced | External coordinator with explicit generation selection |
| Elastic change | Supported topology and data semantics are acceptable | Exact numerical replay is mandatory | Restart with original configuration |

Prioritize recoverability over the lowest theoretical save time.
Use a documented downgrade from exact replay to equivalent continuation when elasticity changes data order or floating-point reductions.
**Takeaway:** Choose by required recovery semantics, not only peak throughput.

# 8. Failure Modes & Edge Cases

Each example below is illustrative; diagnostics describe expected signals, not observed incidents.

### 8.1 Missing model shard

- **Primary mechanism:** A writer crashes before upload finishes.
- **Expected signal:** A manifest lists a shard that cannot be read.
- **Competing explanation:** A listing omits an object because discovery is delayed.
- **Discriminator:** Read each exact manifest key and compare size and hash.
- **Bounded response:** Reject this generation and restore its retained predecessor.
- **Prevention:** Alert on incomplete generations and test worker termination during write.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.2 Mixed generation shards

- **Primary mechanism:** A stale rank writes to a reused object key.
- **Expected signal:** Shard generation IDs or checksums differ from the manifest.
- **Competing explanation:** Object corruption can also cause a digest mismatch.
- **Discriminator:** Compare embedded generation and step metadata with immutable object identities.
- **Bounded response:** Reject the entire generation; do not patch it with a different step.
- **Prevention:** Use unique generation paths and fenced writers.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.3 Stale latest pointer

- **Primary mechanism:** The new generation was never published.
- **Expected signal:** The pointer still names the earlier complete generation.
- **Competing explanation:** A client cache may instead be returning an old value.
- **Discriminator:** Read the pointer with the storage provider’s documented consistency behavior.
- **Bounded response:** Resume from the latest verified eligible generation, not the largest directory name.
- **Prevention:** Monitor publication success independently of shard upload.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.4 Pointer to partial snapshot

- **Primary mechanism:** Publication occurred before all required writes completed.
- **Expected signal:** The pointer is readable but a required shard fails validation.
- **Competing explanation:** A previously valid object may also have been deleted or corrupted.
- **Discriminator:** Inspect publication timestamp, object versions, and deletion audit.
- **Bounded response:** Reject the candidate and fall back; quarantine the bad pointer.
- **Prevention:** Gate publication on complete validation and protect retained objects.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.5 Concurrent coordinators

- **Primary mechanism:** An old coordinator publishes after a newer run takes ownership.
- **Expected signal:** Pointer revision or epoch regresses despite a newer committed step.
- **Competing explanation:** A legitimate intentional rollback may have the same step ordering.
- **Discriminator:** Compare lease epochs, conditional update results, and operator approvals.
- **Bounded response:** Fence the old writer and explicitly select a verified generation.
- **Prevention:** Enforce monotonic fencing at the publication authority.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.6 Interrupted multipart upload

- **Primary mechanism:** A process dies before completing object assembly.
- **Expected signal:** An upload session exists but the final object is absent.
- **Competing explanation:** Authentication or quota denial can also block completion.
- **Discriminator:** Inspect upload status and storage error codes.
- **Bounded response:** Abort or expire orphan multipart uploads without deleting committed objects.
- **Prevention:** Set lifecycle rules and monitor orphan bytes.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.7 Disk full during write

- **Primary mechanism:** The destination cannot allocate remaining blocks.
- **Expected signal:** Writes or flushes fail with capacity errors.
- **Competing explanation:** Quota exhaustion can appear similar despite free filesystem space.
- **Discriminator:** Compare free space, quotas, inode availability, and write errors.
- **Bounded response:** Free approved scratch space or expand capacity; retain the previous generation.
- **Prevention:** Budget staging plus retained generations and reserve headroom.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.8 Manifest corruption

- **Primary mechanism:** A manifest is truncated or modified.
- **Expected signal:** Parsing, schema validation, or manifest digest verification fails.
- **Competing explanation:** A software version mismatch may instead reject a valid schema.
- **Discriminator:** Inspect the exact bytes and schema version before modifying anything.
- **Bounded response:** Select another verified generation and preserve corrupt evidence.
- **Prevention:** Protect manifests and validate them before pointer publication.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.9 Optimizer omitted

- **Primary mechanism:** Only model weights were exported.
- **Expected signal:** Optimizer state keys or expected moment tensors are absent.
- **Competing explanation:** A stateless optimizer may legitimately have no moment tensors.
- **Discriminator:** Inspect optimizer type, expected state, and step count.
- **Bounded response:** Warm-start explicitly or restore a full snapshot; do not claim exact continuation.
- **Prevention:** Require optimizer completeness in the manifest and restore contract.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.10 Optimizer attached to wrong parameters

- **Primary mechanism:** Parameter ordering or grouping changed.
- **Expected signal:** State tensors load but names, shapes, or group metadata differ.
- **Competing explanation:** An intentional architecture migration can produce similar differences.
- **Discriminator:** Compare stable parameter names and group configurations.
- **Bounded response:** Stop strict resume and perform a reviewed migration or warm start.
- **Prevention:** Record canonical parameter mapping and optimizer configuration.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.11 Learning-rate discontinuity

- **Primary mechanism:** Scheduler state or call ordering was not restored.
- **Expected signal:** The resumed learning rate differs from the saved next-step expectation.
- **Competing explanation:** A deliberately changed schedule can also cause a discontinuity.
- **Discriminator:** Compare saved scheduler state, optimizer group rates, and next-step policy.
- **Bounded response:** Restore scheduler and ordering or document an intentional fork.
- **Prevention:** Log effective learning rate and scheduler position at each commit.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.12 Loss-scale reset

- **Primary mechanism:** Mixed-precision scaler state is missing.
- **Expected signal:** Scale and overflow history restart from initial settings.
- **Competing explanation:** New hardware or different batches may also cause overflows.
- **Discriminator:** Compare saved scaler state and overflow counters.
- **Bounded response:** Restore scaler if supported; otherwise declare changed numerical continuity.
- **Prevention:** Include precision mode and scaler in the checkpoint schema.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.13 Accumulation window lost

- **Primary mechanism:** The job saved between microbatches without gradients.
- **Expected signal:** Restored accumulation phase is nonzero but gradient buffers are absent.
- **Competing explanation:** The step counter may instead refer to the next update.
- **Discriminator:** Inspect phase convention and gradient state at the cut.
- **Bounded response:** Roll back to a prior completed update; do not apply a partial update.
- **Prevention:** Save only at safe boundaries unless partial gradients are captured.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.14 RNG stream reset

- **Primary mechanism:** Only initial seeds were saved.
- **Expected signal:** The restored stochastic sequence restarts instead of continuing.
- **Competing explanation:** Nondeterministic kernels can diverge even with restored streams.
- **Discriminator:** Compare generator states and a controlled next-draw trace.
- **Bounded response:** Restore each relevant generator before its next consumer.
- **Prevention:** Record generator ownership and avoid untracked random draws on restore.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.15 Worker RNG not restored

- **Primary mechanism:** Loader worker streams restart with initial seeds.
- **Expected signal:** Augmentation differs after restart even with matching main-process RNG.
- **Competing explanation:** Nondeterministic transforms or changed worker count are competing causes.
- **Discriminator:** Compare worker seed policy, consumed sample IDs, and augmentation seeds.
- **Bounded response:** Recreate deterministic per-sample randomness or restore worker state.
- **Prevention:** Derive augmentation seeds from stable sample identity when feasible.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.16 Prefetch mistaken for consumption

- **Primary mechanism:** The loader cursor advances when work is queued.
- **Expected signal:** After resume some queued but unconsumed samples are skipped.
- **Competing explanation:** Dataset sharding changes can also alter visible sample IDs.
- **Discriminator:** Compare delivered batches to the snapshot cursor and outstanding queue.
- **Bounded response:** Resume from acknowledged consumption or replay idempotently.
- **Prevention:** Define the commit cursor at the optimizer boundary rather than enqueue time.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.17 Sampler epoch lost

- **Primary mechanism:** Sampler resumes with the wrong shuffle epoch.
- **Expected signal:** Sample order repeats an earlier epoch or jumps to another permutation.
- **Competing explanation:** Changing seed or dataset version can look similar.
- **Discriminator:** Compare epoch, seed, data version, and sampler configuration.
- **Bounded response:** Restore the recorded epoch and logical position.
- **Prevention:** Persist sampler state and a stable data-version identifier.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.18 Dataset version changed

- **Primary mechanism:** The same logical index resolves to different records.
- **Expected signal:** Dataset fingerprint or immutable snapshot identifier differs.
- **Competing explanation:** A preprocessing-code change may also change model inputs.
- **Discriminator:** Compare dataset snapshot, transformation version, and tokenizer hash.
- **Bounded response:** Stop exact resume or deliberately fork with a documented data migration.
- **Prevention:** Pin immutable training inputs and transformations.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.19 World size changed

- **Primary mechanism:** Old rank offsets are reused in a new partitioning.
- **Expected signal:** Global sample IDs repeat or disappear at the repartition boundary.
- **Competing explanation:** Drop-last policy and nondivisible dataset size can cause similar symptoms.
- **Discriminator:** Reconstruct the global consumed set and new rank assignments.
- **Bounded response:** Use a supported global cursor and explicit repartition policy.
- **Prevention:** Test both expansion and contraction on nondivisible input sizes.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.20 Unsupported optimizer reshard

- **Primary mechanism:** New topology cannot map saved optimizer state.
- **Expected signal:** Restore fails on partition metadata or tensor shape.
- **Competing explanation:** Corruption can also yield missing or malformed shard errors.
- **Discriminator:** Check framework support matrix and inspect manifest layout.
- **Bounded response:** Restore with the old topology or perform a validated conversion.
- **Prevention:** Exercise restore at each planned target topology before production.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.21 Topology-compatible but trajectory-different

- **Primary mechanism:** Changed reduction order changes floating-point rounding.
- **Expected signal:** Loss or parameter hashes diverge after a successful elastic restart.
- **Competing explanation:** Unrestored RNG or different sample order can cause a larger divergence.
- **Discriminator:** Compare input order and per-step counters before investigating arithmetic.
- **Bounded response:** Accept bounded non-bitwise continuation only if the contract permits it.
- **Prevention:** Separate correctness, statistical equivalence, and bitwise replay criteria.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.22 Pipeline stage mismatch

- **Primary mechanism:** Saved tensors assume a different pipeline partition.
- **Expected signal:** Layer ownership or optimizer shard metadata does not map.
- **Competing explanation:** A changed architecture can yield a similar mismatch.
- **Discriminator:** Compare layer-to-stage mapping and tensor names against the manifest.
- **Bounded response:** Restore the supported layout or convert through a verified representation.
- **Prevention:** Record pipeline partition and virtual-stage settings.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.23 Stuck collective

- **Primary mechanism:** A rank exits or calls checkpoint collectives in a different order.
- **Expected signal:** Other ranks wait until a collective timeout.
- **Competing explanation:** Backend network failure can also hang a healthy rank.
- **Discriminator:** Correlate rank logs, collective sequence, and failure timestamps.
- **Bounded response:** Abort the generation and restart the worker group under orchestration.
- **Prevention:** Ensure uniform collective order and bounded timeouts.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.24 Overlapping live-buffer write

- **Primary mechanism:** Training mutates tensors after asynchronous saving begins.
- **Expected signal:** Saved tensors combine values from different logical updates.
- **Competing explanation:** A nondeterministic replay can also yield a different model hash.
- **Discriminator:** Pause mutations or compare a frozen copy against the saved object.
- **Bounded response:** Disable unsafe asynchronous mode and restore a known-good snapshot.
- **Prevention:** Use an implementation with documented snapshot isolation.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.25 Insufficient snapshot memory

- **Primary mechanism:** Asynchronous copy or staging exceeds available memory.
- **Expected signal:** Host or device memory rises at save time and the process is killed.
- **Competing explanation:** A separate training-memory leak may have a similar trajectory.
- **Discriminator:** Correlate allocations and memory high-water mark with save start.
- **Bounded response:** Use synchronous or chunked capture if the implementation guarantees consistency.
- **Prevention:** Budget duplicate buffers, pinned memory, and concurrent generations.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.26 Storage throttling

- **Primary mechanism:** Concurrent writers saturate a shared service.
- **Expected signal:** Checkpoint p95 grows with throttling responses and queue depth.
- **Competing explanation:** Network congestion or checksum CPU can also limit rate.
- **Discriminator:** Measure bytes, errors, per-stage timing, and service-side throttles.
- **Bounded response:** Reduce write concurrency or checkpoint rate within RPO limits.
- **Prevention:** Capacity-plan shared clients and add admission control.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.27 Restore exceeds RTO

- **Primary mechanism:** Reading and reconstructing state takes longer than allowed.
- **Expected signal:** Download, deserialization, or resharding dominates the recovery timeline.
- **Competing explanation:** Scheduler placement and container startup may dominate instead.
- **Discriminator:** Time each recovery phase from failure detection to useful work.
- **Bounded response:** Improve the measured bottleneck or change the RTO with stakeholders.
- **Prevention:** Run periodic full restore drills at representative scale.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.28 Local checkpoint lost with node

- **Primary mechanism:** The only copy was on ephemeral local storage.
- **Expected signal:** Node replacement removes all files for the latest generation.
- **Competing explanation:** A mount configuration error can also make durable storage appear empty.
- **Discriminator:** Inspect volume identity, object inventory, and orchestration events.
- **Bounded response:** Restore from an earlier durable copy.
- **Prevention:** Do not report a local staging write as a durable commit.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.29 Retention deletes a live dependency

- **Primary mechanism:** Garbage collection removes a base of an incremental chain.
- **Expected signal:** The selected manifest refers to a deleted parent or chunk.
- **Competing explanation:** A bad manifest can refer to an object that never existed.
- **Discriminator:** Trace references and object deletion audit logs.
- **Bounded response:** Restore an intact independent generation; stop unsafe garbage collection.
- **Prevention:** Mark reachable dependencies from every retained root before sweeping.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.30 Checksum mismatch

- **Primary mechanism:** Stored bytes do not match the expected digest.
- **Expected signal:** Recomputed digest differs for the exact object version.
- **Competing explanation:** Wrong manifest, stale cache, and upload corruption are competing explanations.
- **Discriminator:** Verify object identity and recompute digest from an independent read.
- **Bounded response:** Reject the generation and retrieve a verified replica or predecessor.
- **Prevention:** Store digests in an authenticated manifest and scrub retained objects.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.31 Preemption notice too short

- **Primary mechanism:** The grace period ends before publication.
- **Expected signal:** Termination occurs while a new generation is still uncommitted.
- **Competing explanation:** A missed signal handler or deadlocked collective can also prevent commit.
- **Discriminator:** Compare warning timestamp, save phases, and kill timestamp.
- **Bounded response:** Recover from the previous committed generation.
- **Prevention:** Use periodic checkpoints and treat signal-triggered saves as opportunistic.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.32 Security compromise through checkpoint

- **Primary mechanism:** An untrusted serialized payload executes code or is replaced.
- **Expected signal:** Source provenance or signature is missing or invalid.
- **Competing explanation:** Ordinary corruption can also invalidate a checksum.
- **Discriminator:** Verify origin, access policy, signature, and safe loading mode for the release.
- **Bounded response:** Quarantine the artifact and restore a trusted generation.
- **Prevention:** Restrict writers and avoid loading untrusted pickle-based artifacts.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.33 Checkpoint carries secrets

- **Primary mechanism:** Configuration, logs, or metadata contain credentials.
- **Expected signal:** Sensitive tokens are present in saved non-tensor state.
- **Competing explanation:** A secret scanner may generate false positives on random bytes.
- **Discriminator:** Inspect the specific finding under approved access controls.
- **Bounded response:** Revoke exposed credentials and remove affected artifacts per policy.
- **Prevention:** Exclude secrets, encrypt data, and enforce retention and access boundaries.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.34 Wrong step convention

- **Primary mechanism:** The saved counter is interpreted as the next step rather than last completed.
- **Expected signal:** An update or scheduler transition is duplicated or skipped.
- **Competing explanation:** A data cursor off-by-one produces a similar divergence.
- **Discriminator:** Trace the last applied update, counter increment, and next batch in a small run.
- **Bounded response:** Restore the documented convention or convert the metadata explicitly.
- **Prevention:** Specify counter semantics and test a crash on both sides of the update.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.35 Wrong batch scale after resize

- **Primary mechanism:** World size changes without recalculating effective global batch.
- **Expected signal:** Tokens or examples per optimizer update change unexpectedly.
- **Competing explanation:** A changed accumulation factor or packing policy can also change throughput.
- **Discriminator:** Compute per-rank batch × ranks × accumulation under actual padding policy.
- **Bounded response:** Adjust accumulation or explicitly adopt and evaluate the new training regime.
- **Prevention:** Record batch construction and schedule units with topology.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

### 8.36 Partial restore accepted

- **Primary mechanism:** A permissive loader silently ignores missing keys.
- **Expected signal:** Reported success coexists with randomly initialized or stale state.
- **Competing explanation:** An intentional adapter-only warm start may also omit base-model keys.
- **Discriminator:** Inspect missing and unexpected keys and compare against the allowed schema.
- **Bounded response:** Fail strict resume; use a separately labeled warm-start workflow.
- **Prevention:** Enforce schema validation and publish restore completeness metrics.
- **Verification:** Re-run the specific discriminator after recovery and verify a complete committed generation before retiring the prior one.

**Takeaway:** Distinguish candidate causes before altering the checkpoint or declaring a recovery successful.

# 9. Complexity & Cost Analysis

## 9.1 State inventory for a real restart

The following inventory is a contract to adapt to the actual framework, not a claim that any API automatically saves all entries.
Capture each item at the same logical cut or state precisely why it is reconstructed without changing the promised semantics.

| State | Capture and restore rule | Consequence of omission |
|---|---|---|
| Model parameters | Save every trainable and frozen parameter required by the forward pass, using canonical names and shard metadata | Missing or wrongly mapped weights alter predictions |
| Persistent model buffers | Save running statistics and other persistent buffers alongside parameters | Model behavior may differ despite identical weights |
| Optimizer moments | Save first and second moments or other optimizer-specific history with matching parameter identities | Resuming changes subsequent updates |
| Optimizer step counters | Save per-parameter or global counters where the implementation uses them | Bias correction or update schedule may differ |
| Optimizer parameter groups | Save group membership, hyperparameters and any mutable group state | Different decay or learning rates may apply |
| Master weights | Save full-precision master copies if the training implementation owns them | Reconstructing from reduced precision can lose information |
| Scheduler | Save scheduler internal state, configuration and order relative to optimizer step | Next learning rate can jump |
| Mixed-precision scaler | Save dynamic loss scale and growth/backoff tracking when used | Overflow behavior and skipped steps may differ |
| Global step | Define whether it is completed updates or next update and persist it | Off-by-one causes repeated or skipped transitions |
| Microbatch phase | At a boundary record zero; otherwise capture phase and all accumulated gradients | A partial update cannot be reconstructed from weights alone |
| Gradients | Save only if mid-accumulation recovery is supported; include partition and scaling state | A resumed accumulation produces an incorrect update |
| EMA or target weights | Capture exponential moving average or target network and update counter where used | Evaluation or target behavior changes |
| Training controller | Persist early-stop patience, adaptive curriculum and other mutable controller state | Control decisions may reset |
| Gradient clipping | Record configuration and any adaptive clipping history | Effective optimizer update may change |
| Python RNG | Save the Python generator state when used | Python-side random choices may restart |
| Numerical-library RNG | Save each relevant library generator, including independent generator objects | Augmentation or sampling streams can shift |
| CPU framework RNG | Capture each relevant CPU generator, not only its initial seed | Dropout or input transforms may differ |
| Accelerator RNG | Capture generator states for all relevant devices and streams according to framework semantics | Accelerator-side stochastic operations diverge |
| Worker RNG | Save or deterministically reconstruct data-loader worker random streams | Augmentation differs even if the main process was restored |
| Sampler | Persist sampler type, seed, epoch, permutation definition, and consumed logical position | Sample order may repeat or skip |
| Dataset version | Record immutable snapshot, split and input manifest identity | An index may refer to different content |
| Preprocessing | Record tokenizer, transform version and configuration, including randomization policy | Identical sample IDs may produce different tensors |
| Shuffling buffer | Persist buffer contents and cursor for stateful stream shuffling | Restart cannot reconstruct the same next item from a seed alone |
| Stream offsets | Save source partition offsets at the acknowledged training boundary | Records can be lost or replayed |
| Prefetch state | Drain, snapshot, or reconstruct outstanding queued items and acknowledge only consumed items | Advancing to queued position silently skips unconsumed data |
| Sequence packing | Save packer leftovers and boundaries or define deterministic reconstruction | Example and token boundaries can change |
| Batch policy | Record batch size, accumulation, padding, drop-last and token limits | Effective update size can change after restart |
| Parallel layout | Record world size and data, tensor, pipeline and optimizer parallel dimensions | Restorer may interpret shards incorrectly |
| Tensor schema | Record names, dtypes, shapes, offsets, shards and layout version | Shape-compatible but semantically wrong loading may go unnoticed |
| Framework and code | Record code revision, dependencies, framework and checkpoint-format version | Serialization and optimizer behavior can change |
| Run identity | Record run ID, generation ID, parent generation, wall-clock time and logical step | A stale checkpoint may be mistaken for the intended run |
| Manifest | Record all object IDs, lengths, hashes and expected membership | Partial generations can appear complete |
| Commit authority | Record fencing epoch and conditional publication outcome | Competing writers may overwrite progress |
| Data acknowledgements | Align logical source cursor and externally committed offsets with checkpoint publication | External consumers may advance beyond recoverable training state |
| Configuration | Capture model, optimizer, scheduler, data and precision configurations without secrets | Recreated processes can behave differently |
| Reproducibility policy | Record deterministic settings and documented hardware/software constraints | A restore is incorrectly advertised as bitwise reproducible |

A map-style dataset can reconstruct its permutation from a seed and epoch only when the algorithm, dataset identity, position and all relevant configuration remain stable.
A streaming dataset may require durable offsets, shuffle-buffer contents and a source retention window.
With worker prefetch, the main process must distinguish assigned or fetched data from data included in completed updates.
Some framework data loaders do not expose every worker's live iterator state; do not promise exact replay without a supported capture design.
A robust alternative is a global deterministic sample plan and per-sample augmentation randomness derived from stable identifiers.
That alternative still needs an explicit policy for repeated records, packing, retries, and changed dataset versions.
When external offsets cannot be committed atomically with the snapshot, design for replay and define deduplication or accept at-least-once sample processing.
Exactly-once effects across separate systems require a transaction or equivalent end-to-end protocol; saving a model does not create one.

## 9.2 Restore sequence

1. Stop or fence earlier worker groups and choose a committed candidate for the correct run.
2. Read its manifest by exact identifier and validate schema, generation, topology, expected objects and trusted provenance.
3. Check compatibility of code, model, optimizer, data, precision and planned target topology before mutating the live job.
4. Verify each object against its declared identity, size and digest under the storage system's actual guarantees.
5. If any required object or dependency is invalid, reject the whole candidate and try an older retained generation.
6. Instantiate the model and optimizer using the saved configuration and supported framework loading order.
7. Map model tensors, buffers, optimizer slots and parameter groups to the new rank layout using a supported resharding path.
8. Restore counters, scheduler, loss scaler, controller and any other mutable state with documented step semantics.
9. Restore sampler, data source offsets, packing and acknowledged cursor without confusing prefetched items with consumed items.
10. Restore each random generator at the point required to make the next random consumer receive the intended state.
11. Validate next learning rate, global step, effective batch, data position and rank participation before the first update.
12. Emit the generation ID, saved and resumed topology, state completeness, and any declared reproducibility downgrade.
13. Run the next update; monitor for NaNs, overflow, anomalous loss and unexpected sample IDs.
14. Keep the prior verified checkpoint until the new job has produced another valid committed generation under the retention policy.

Loader construction itself can consume random numbers; actual ordering must follow the framework's documented behavior and be tested for the implementation.
When a rank is replaced, old rank number does not imply the same process, node, device, or data assignment.
A strict exact replay objective may require unchanged world size, kernel determinism, software versions and hardware in addition to complete RNG state.
A changed all-reduce tree can change rounding even if each restored tensor is initially identical.

## 9.3 Sizing and economics

For a full snapshot of P trainable parameters with 2-byte model weights, 4-byte master weights and two 4-byte optimizer moments, the named components total 14P bytes.
This is illustrative only: precision choices, optimizer design, frozen parameters, buffers, metadata, alignment, sharding and compression can change the actual size.
For P=1 billion, these four components use 14 billion bytes ≈ 13.04 GiB; this is not device memory per rank.
The decimal-to-binary conversion is 14,000,000,000 / 1,073,741,824 ≈ 13.04.
An optimizer storing only 8-bit moments, or a run without separate master weights, needs a different component sum.
A run with frozen base parameters still needs their recoverable values or an immutable, verified external model reference.
If 240 GiB is stored per full generation and three independent generations are retained, payload is 720 GiB before staging, replication, metadata and incremental dependencies.
For a two-copy durability scheme, if both copies contain the same 720 GiB and overhead is neglected, raw payload is 1,440 GiB.
A temporary 240 GiB staging generation raises that example's raw requirement further if staging occupies the same capacity pool.
These are capacity estimates, not bills: storage pricing may use decimal units, replication, requests, egress and tier-specific retention rules.
At one 240 GiB full save every 40 minutes of useful work, ignoring pauses gives an idealized upper-rate estimate of 36 saves/day and 8,640 GiB/day.
With the earlier 75 s non-overlapped pause per save, the long-run rate is 86,400 / (2,400 + 75) ≈ 34.91 saves/day, or approximately 8,378 GiB/day before retries and replication; actual whole saves in a calendar day depend on the starting phase.
The physical write volume can be larger because of replication and failed attempts or smaller with effective compression and deduplication.
Assume 240 GiB restore data at 3 GiB/s effective aggregate read bandwidth: payload transfer alone is 80 s.
With 20 s discovery and placement, 25 s object validation, and 35 s deserialization/resharding, a nonoverlapping model totals 160 s.
If phases overlap, adding their full durations overstates the critical path; if placement is slower, the actual RTO grows.
RTO starts at the contractually defined failure time and ends at resumed useful work, not merely when bytes finish downloading.
Model serialization time is at least proportional to the number of bytes actually visited; manifest verification also scales with object count and bytes hashed.
Large numbers of small shards may be dominated by request latency and metadata operations even if aggregate byte volume is modest.
**Takeaway:** Budget both retained storage and the end-to-end critical path, including recovery and control-plane delays.

## 9.4 Fault-injection and recovery drill catalog

Each drill is a proposed isolated test, not an executed result.
Run with synthetic data, an explicit cleanup plan, and a retained known-good baseline.
The expected observations are acceptance criteria; storage and framework versions determine exact instrumentation.

### Drill 01: Writer before first byte
- **Setup:** Committed A; candidate B reserved.
- **Intervention:** Kill writer before first object write.
- **Expected result:** A remains selected; B is not eligible.
- **Discriminating evidence:** Compare pointer version with B allocation log.
- **Safety / interpretation:** Clean B only after its lease expires.

### Drill 02: Writer halfway through shard
- **Setup:** A committed; B shard upload started.
- **Intervention:** Terminate a writer mid-object.
- **Expected result:** No partial object is accepted as B shard.
- **Discriminating evidence:** Read exact object version and inspect completion marker.
- **Safety / interpretation:** Abort abandoned multipart upload.

### Drill 03: Last rank after upload
- **Setup:** All B payloads written; one rank has not acknowledged.
- **Intervention:** Kill that rank before acknowledgement.
- **Expected result:** B remains uncommitted unless a documented verifier proves completeness.
- **Discriminating evidence:** Compare acknowledgements with expected rank set.
- **Safety / interpretation:** Do not manufacture an acknowledgement.

### Drill 04: Coordinator before manifest
- **Setup:** All shard acknowledgements arrived.
- **Intervention:** Kill coordinator before manifest creation.
- **Expected result:** Latest remains A.
- **Discriminating evidence:** Check absence of committed B manifest.
- **Safety / interpretation:** Preserve A and use a fresh generation ID.

### Drill 05: Coordinator after manifest
- **Setup:** B manifest written but pointer still names A.
- **Intervention:** Kill coordinator before publish.
- **Expected result:** Readers still select A.
- **Discriminating evidence:** Inspect immutable manifest and pointer history.
- **Safety / interpretation:** An authorized new coordinator may validate and publish only under the fencing policy.

### Drill 06: Coordinator after publish
- **Setup:** B validated and pointer update succeeded.
- **Intervention:** Kill coordinator immediately after commit.
- **Expected result:** Recovery can select B after validation.
- **Discriminating evidence:** Read pointer and all referenced object versions.
- **Safety / interpretation:** Keep A for fallback.

### Drill 07: Stale writer
- **Setup:** Coordinator epoch 7 is superseded by epoch 8.
- **Intervention:** Let epoch 7 attempt publication.
- **Expected result:** Publication authority rejects the stale epoch.
- **Discriminating evidence:** Inspect conditional write response and current fencing token.
- **Safety / interpretation:** Prevent the old process from retrying with new credentials.

### Drill 08: Duplicate publication
- **Setup:** B already committed.
- **Intervention:** Retry the same publication request.
- **Expected result:** Result is idempotent or rejected according to the contract.
- **Discriminating evidence:** Compare pointer version and generation ID.
- **Safety / interpretation:** Do not create two logical commits for one update.

### Drill 09: Pointer conflict
- **Setup:** Two coordinators attempt different candidates.
- **Intervention:** Schedule both pointer writes concurrently.
- **Expected result:** At most one conditional update succeeds for the expected version.
- **Discriminating evidence:** Inspect provider revision and both responses.
- **Safety / interpretation:** Validate and explicitly resolve the losing candidate.

### Drill 10: Object corruption
- **Setup:** B is committed in an isolated test store.
- **Intervention:** Flip one byte of one shard.
- **Expected result:** Digest validation rejects B.
- **Discriminating evidence:** Verify exact object version and independent digest.
- **Safety / interpretation:** Restore A rather than applying a partial state.

### Drill 11: Manifest corruption
- **Setup:** B manifest stored in a test bucket.
- **Intervention:** Truncate manifest bytes.
- **Expected result:** Parser or manifest integrity validation rejects B.
- **Discriminating evidence:** Check authenticated manifest digest and schema error.
- **Safety / interpretation:** Retain corrupt bytes for incident analysis.

### Drill 12: Missing expected shard
- **Setup:** Manifest expects eight shard IDs.
- **Intervention:** Remove one test object.
- **Expected result:** Restore fails closed for B.
- **Discriminating evidence:** Read every manifest key rather than relying on listing.
- **Safety / interpretation:** Fallback to A and alert.

### Drill 13: Extra stray object
- **Setup:** B contains all eight required shards.
- **Intervention:** Add an unrelated ninth object.
- **Expected result:** Restore uses the manifest set and does not adopt the stray.
- **Discriminating evidence:** Compare object names with manifest membership.
- **Safety / interpretation:** Garbage-collect the stray only after reachability analysis.

### Drill 14: Wrong object generation
- **Setup:** A and B have compatible tensor shapes.
- **Intervention:** Substitute an A object for its B counterpart.
- **Expected result:** Generation identity or hash check rejects the substitution.
- **Discriminating evidence:** Compare object version, generation metadata and digest.
- **Safety / interpretation:** Never repair B by mixing objects from steps.

### Drill 15: Missing optimizer
- **Setup:** Model weights exist; Adam moments absent.
- **Intervention:** Attempt strict training resume.
- **Expected result:** Completeness validation blocks continuation.
- **Discriminating evidence:** Inspect optimizer slot schema and parameter mapping.
- **Safety / interpretation:** A deliberate warm start needs a separate run identity.

### Drill 16: Wrong optimizer group
- **Setup:** Save two groups with distinct learning rates.
- **Intervention:** Swap the group-to-parameter assignment on restore.
- **Expected result:** Compatibility check detects different mapping.
- **Discriminating evidence:** Compare stable parameter names and group metadata.
- **Safety / interpretation:** Use a reviewed mapping migration, not positional guessing.

### Drill 17: Missing scheduler
- **Setup:** Training uses a stateful learning-rate scheduler.
- **Intervention:** Remove scheduler state from test snapshot.
- **Expected result:** Strict resume rejects missing required state.
- **Discriminating evidence:** Compare next effective rate against the saved expected value.
- **Safety / interpretation:** Restore a complete generation or explicitly fork schedule.

### Drill 18: Scaler overflow history
- **Setup:** Use a scaler with noninitial growth tracker.
- **Intervention:** Restart after saving its state.
- **Expected result:** Restored scale and tracker match the checkpoint.
- **Discriminating evidence:** Compare scaler state before and after loading.
- **Safety / interpretation:** Also check whether the last optimizer update was skipped.

### Drill 19: Counter boundary
- **Setup:** Save immediately after optimizer update 42.
- **Intervention:** Restart before update 43.
- **Expected result:** First resumed update is 43, not 42 or 44.
- **Discriminating evidence:** Trace update, scheduler and counter order.
- **Safety / interpretation:** Define whether saved step is completed or next.

### Drill 20: Skipped update
- **Setup:** Inject overflow at a mixed-precision step.
- **Intervention:** Save after the scaler skips the update.
- **Expected result:** Counter and scheduler follow the declared skipped-step policy.
- **Discriminating evidence:** Compare actual parameter change, scaler and counter logs.
- **Safety / interpretation:** Do not infer an update from microbatch count.

### Drill 21: Mid-accumulation
- **Setup:** Accumulate three of four microbatches.
- **Intervention:** Interrupt before the fourth microbatch.
- **Expected result:** Either gradients and phase restore together or recovery rolls back.
- **Discriminating evidence:** Compare gradient buffers and phase to baseline.
- **Safety / interpretation:** Do not step on a partial gradient sum.

### Drill 22: Accumulation boundary
- **Setup:** Finish all microbatches and optimizer step.
- **Intervention:** Save after gradient reset.
- **Expected result:** Resume begins a fresh accumulation window.
- **Discriminating evidence:** Check phase zero and next sample ID.
- **Safety / interpretation:** Record completed update and committed sample boundary together.

### Drill 23: Python RNG
- **Setup:** Draw values from Python random before saving.
- **Intervention:** Resume and draw the next sequence.
- **Expected result:** Values match a controlled same-environment baseline.
- **Discriminating evidence:** Compare full generator state and next draws.
- **Safety / interpretation:** Avoid random draws during restore before state is installed.

### Drill 24: Library RNG
- **Setup:** Use an independent numerical-library generator.
- **Intervention:** Save and resume after several draws.
- **Expected result:** Its next values match the baseline.
- **Discriminating evidence:** Compare this generator object, not merely a global seed.
- **Safety / interpretation:** Inventory every explicitly created generator.

### Drill 25: Device RNG
- **Setup:** Use stochastic accelerator operations.
- **Intervention:** Resume with the same topology and deterministic settings.
- **Expected result:** Controlled next-draw test matches where supported.
- **Discriminating evidence:** Compare per-device state and device mapping.
- **Safety / interpretation:** Do not claim cross-hardware bitwise equivalence.

### Drill 26: Worker augmentation
- **Setup:** Apply stochastic augmentation in loader workers.
- **Intervention:** Restart with identical worker configuration.
- **Expected result:** Expected per-sample augmentation matches the chosen contract.
- **Discriminating evidence:** Log sample ID and augmentation seed rather than just batch number.
- **Safety / interpretation:** Account for worker initialization consuming RNG.

### Drill 27: Prefetch queue
- **Setup:** Allow workers to fetch beyond the last completed update.
- **Intervention:** Interrupt while prefetched items are pending.
- **Expected result:** Resume begins at acknowledged rather than fetched position.
- **Discriminating evidence:** Compare committed sample IDs and queued IDs.
- **Safety / interpretation:** Permit replay only under an explicit duplicate policy.

### Drill 28: Sampler epoch
- **Setup:** Use a shuffled map dataset for two epochs.
- **Intervention:** Save partway through epoch two.
- **Expected result:** Next sample follows the recorded epoch permutation.
- **Discriminating evidence:** Compare epoch, seed, position and dataset fingerprint.
- **Safety / interpretation:** Do not recreate the sampler from seed alone.

### Drill 29: Dataset drift
- **Setup:** Snapshot references immutable dataset version V1.
- **Intervention:** Configure restore with V2.
- **Expected result:** Strict resume detects the version mismatch.
- **Discriminating evidence:** Compare dataset manifest hashes and transform versions.
- **Safety / interpretation:** Treat V2 as a documented new training branch.

### Drill 30: Stream offset
- **Setup:** Consume records from two source partitions.
- **Intervention:** Crash after checkpoint commit but before external offset commit.
- **Expected result:** Recovery may replay but must not silently skip records.
- **Discriminating evidence:** Compare checkpoint cursor and source offset per partition.
- **Safety / interpretation:** Use idempotence or an actual coordinated transaction.

### Drill 31: Shuffle buffer
- **Setup:** Use a streaming shuffle buffer with known contents.
- **Intervention:** Restart with only its seed and source cursor.
- **Expected result:** Exact replay should be rejected if buffer state is missing.
- **Discriminating evidence:** Compare the next sequence against a no-failure baseline.
- **Safety / interpretation:** Save buffer contents or declare a weaker continuity contract.

### Drill 32: Packing leftovers
- **Setup:** Pack variable-length examples into fixed token blocks.
- **Intervention:** Interrupt with unused tokens buffered.
- **Expected result:** Next block uses the same leftovers under strict replay.
- **Discriminating evidence:** Inspect packer buffer and example boundaries.
- **Safety / interpretation:** Reconstruct from a safe cursor only if deterministic.

### Drill 33: Nondivisible dataset
- **Setup:** Use 11 samples across four ranks.
- **Intervention:** Restart with three ranks.
- **Expected result:** Documented drop or padding policy accounts for each sample.
- **Discriminating evidence:** Enumerate IDs assigned before and after restart.
- **Safety / interpretation:** Distinguish intentional padding from accidental duplicates.

### Drill 34: Elastic expansion
- **Setup:** Save with four data-parallel ranks.
- **Intervention:** Restore with eight supported ranks.
- **Expected result:** Tensor values and optimizer state map to the supported new layout.
- **Discriminating evidence:** Check global tensor reconstruction and sample coverage.
- **Safety / interpretation:** Recalculate effective batch and schedule units.

### Drill 35: Elastic contraction
- **Setup:** Save with eight ranks.
- **Intervention:** Restore with four supported ranks.
- **Expected result:** Every required shard contributes to the reconstructed state.
- **Discriminating evidence:** Compare global tensor checksums and optimizer slots.
- **Safety / interpretation:** Abort if target rank memory cannot hold the new partitions.

### Drill 36: Unsupported reshard
- **Setup:** Choose an unsupported source-to-target parallel layout.
- **Intervention:** Attempt restore.
- **Expected result:** Compatibility check fails before training resumes.
- **Discriminating evidence:** Inspect source layout and implementation support.
- **Safety / interpretation:** Use original topology or a verified offline conversion.

### Drill 37: Pipeline remapping
- **Setup:** Split layers across two pipeline stages.
- **Intervention:** Change to three stages at restart.
- **Expected result:** Restore occurs only if layer and optimizer remapping is supported.
- **Discriminating evidence:** Compare layer ownership and tensor keys.
- **Safety / interpretation:** Prevent silent shape-compatible misassignment.

### Drill 38: Effective batch change
- **Setup:** Run four ranks with two microbatches per update.
- **Intervention:** Double ranks without changing other factors.
- **Expected result:** Effective example batch doubles for fixed per-rank batch.
- **Discriminating evidence:** Compute rank count × microbatch size × accumulation.
- **Safety / interpretation:** Retune only as an intentional training-policy change.

### Drill 39: Collective mismatch
- **Setup:** Make one rank skip the checkpoint collective.
- **Intervention:** Trigger save across the group.
- **Expected result:** The job times out and rejects the candidate.
- **Discriminating evidence:** Compare rank logs and collective sequence numbers.
- **Safety / interpretation:** Use bounded timeouts and fail the whole generation.

### Drill 40: Slow rank
- **Setup:** Throttle one writer while others finish.
- **Intervention:** Trigger distributed save.
- **Expected result:** Commit waits for the required slow shard or aborts.
- **Discriminating evidence:** Compare rank completion times and backend request metrics.
- **Safety / interpretation:** Avoid excluding the rank without a redundancy protocol.

### Drill 41: Async isolation
- **Setup:** Capture a checkpoint asynchronously.
- **Intervention:** Perform further optimizer updates during writes.
- **Expected result:** Restored snapshot represents the original cut, not mixed steps.
- **Discriminating evidence:** Compare saved tensors with a frozen reference snapshot.
- **Safety / interpretation:** Disable async if snapshot isolation cannot be established.

### Drill 42: Staging memory
- **Setup:** Use a constrained host-memory budget.
- **Intervention:** Start concurrent snapshot copies.
- **Expected result:** Memory pressure is surfaced before worker termination.
- **Discriminating evidence:** Measure peak staging allocation and in-flight copies.
- **Safety / interpretation:** Limit concurrency or use a proven chunked snapshot design.

### Drill 43: Shared bandwidth
- **Setup:** Run two jobs writing to one test backend.
- **Intervention:** Trigger both checkpoints together.
- **Expected result:** Effective aggregate rates reflect contention rather than link peak.
- **Discriminating evidence:** Measure each job and backend throughput over the same window.
- **Safety / interpretation:** Coordinate checkpoint admission without violating RPO.

### Drill 44: Read throttling
- **Setup:** Constrain restore bandwidth in a test environment.
- **Intervention:** Recover a full checkpoint.
- **Expected result:** Observed RTO increases by the read critical-path delay.
- **Discriminating evidence:** Time placement, read, validation and deserialize separately.
- **Safety / interpretation:** Size recovery capacity for concurrent restarts.

### Drill 45: Local tier loss
- **Setup:** Stage the newest snapshot only on one node.
- **Intervention:** Remove the node before durable upload.
- **Expected result:** Recovery falls back to the prior durable generation.
- **Discriminating evidence:** Inspect durable manifest instead of local status.
- **Safety / interpretation:** Never promote staging completion to durable commit.

### Drill 46: Preemption notice
- **Setup:** Configure a warning shorter than full checkpoint time.
- **Intervention:** Send warning then terminate at deadline.
- **Expected result:** The system retains and restores the last committed generation.
- **Discriminating evidence:** Compare signal, commit and kill timestamps.
- **Safety / interpretation:** Use periodic commits rather than depending on final save.

### Drill 47: Incremental base deletion
- **Setup:** Create a base and a dependent delta in a test store.
- **Intervention:** Attempt to delete the base while delta is retained.
- **Expected result:** Reachability policy blocks deletion.
- **Discriminating evidence:** Traverse the delta manifest to its root.
- **Safety / interpretation:** Never apply a retention count only to delta names.

### Drill 48: Garbage collection race
- **Setup:** Start mark-and-sweep while a generation is publishing.
- **Intervention:** Delay marking until a new root appears.
- **Expected result:** No object referenced by an eligible root is swept.
- **Discriminating evidence:** Compare root snapshot time with fencing and grace period.
- **Safety / interpretation:** Use a publication-aware collection protocol.

### Drill 49: Encryption key unavailable
- **Setup:** Store an encrypted generation in a test environment.
- **Intervention:** Remove restore-time key authorization.
- **Expected result:** Restore fails clearly before claiming success.
- **Discriminating evidence:** Inspect key-service authorization and object read errors.
- **Safety / interpretation:** Preserve old key versions for the approved retention period.

### Drill 50: Untrusted serialization
- **Setup:** Provide a checkpoint from an untrusted source.
- **Intervention:** Attempt to load it in the training environment.
- **Expected result:** Security policy prevents unsafe deserialization.
- **Discriminating evidence:** Verify provenance, signatures and framework safe-loading settings.
- **Safety / interpretation:** Quarantine the artifact rather than bypassing controls.

### Drill 51: Old checkpoint schema
- **Setup:** Restore a generation written by an older supported release.
- **Intervention:** Run its documented migration path.
- **Expected result:** Resulting state matches a verified baseline.
- **Discriminating evidence:** Compare migrated names, dtypes and optimizer groups.
- **Safety / interpretation:** Reject unknown schemas rather than guessing.

### Drill 52: Partial load warning
- **Setup:** Remove a required model key from a test file.
- **Intervention:** Invoke the intended strict restore path.
- **Expected result:** Resume fails instead of reporting success.
- **Discriminating evidence:** Inspect missing and unexpected key reports.
- **Safety / interpretation:** Keep warm-start logic separate from restart logic.

### Drill 53: Wrong run selected
- **Setup:** Place two run manifests in the same storage namespace.
- **Intervention:** Point recovery at the other run.
- **Expected result:** Run identity check rejects the candidate.
- **Discriminating evidence:** Compare run ID, model configuration and data version.
- **Safety / interpretation:** Namespace generation IDs and use authorized pointer updates.

### Drill 54: Retention pressure
- **Setup:** Keep three complete full snapshots and begin a fourth.
- **Intervention:** Exhaust available staging capacity.
- **Expected result:** A failed new write does not delete all known-good snapshots.
- **Discriminating evidence:** Measure available capacity and each generation state.
- **Safety / interpretation:** Reserve staging headroom before starting publication.

### Drill 55: Restore time budget
- **Setup:** Define a 5-minute end-to-end recovery objective.
- **Intervention:** Run an isolated representative restore drill.
- **Expected result:** Measure elapsed failure-to-useful-work time against 300 seconds.
- **Discriminating evidence:** Break down detection, placement, transfer, load and first update.
- **Safety / interpretation:** Do not call a payload-only estimate an achieved RTO.

### Drill 56: Integrity versus correctness
- **Setup:** Create a valid hash over a deliberately wrong tensor.
- **Intervention:** Restore the otherwise complete generation.
- **Expected result:** Hash verification succeeds but semantic checks detect the wrong cut.
- **Discriminating evidence:** Compare step identity, model state invariants and reference update.
- **Safety / interpretation:** Authenticate the producer and validate logical consistency.

### Drill 57: RNG with topology change
- **Setup:** Save per-rank RNG streams at four ranks.
- **Intervention:** Restore at eight ranks.
- **Expected result:** The procedure declares how new streams are assigned.
- **Discriminating evidence:** Compare sample-based randomness and rank mapping.
- **Safety / interpretation:** Avoid promising the original bitwise trajectory.

### Drill 58: Controller state
- **Setup:** Use early stopping with patience nearly exhausted.
- **Intervention:** Restart after a checkpoint.
- **Expected result:** The patience counter and best metric continue from saved values.
- **Discriminating evidence:** Compare controller state and validation history.
- **Safety / interpretation:** A reset is a policy change, not an exact restart.

### Drill 59: EMA weights
- **Setup:** Train while maintaining an exponential moving average.
- **Intervention:** Resume and evaluate the EMA copy.
- **Expected result:** EMA parameters and update count match the saved snapshot.
- **Discriminating evidence:** Compare EMA and live parameter states separately.
- **Safety / interpretation:** Do not reconstruct EMA from the latest model alone.

### Drill 60: Frozen model reference
- **Setup:** Store adapter weights while referencing a frozen base model.
- **Intervention:** Change the base model artifact before restore.
- **Expected result:** Reference hash validation rejects the mismatch.
- **Discriminating evidence:** Compare immutable base identity and adapter metadata.
- **Safety / interpretation:** Retain the referenced base for the entire checkpoint lifetime.

### Drill 61: Checkpoint authentication
- **Setup:** Sign a manifest with a test signing key.
- **Intervention:** Replace both an object and its plain digest.
- **Expected result:** Signature validation rejects the altered manifest.
- **Discriminating evidence:** Verify signature against trusted key and exact manifest bytes.
- **Safety / interpretation:** A plain digest alone does not identify an authorized writer.

**Takeaway:** A recovery claim is credible only after the relevant failure boundary is exercised against the actual stack.

## 9.5 Operational evidence and decision record

The following fields form a practical checklist for a checkpoint service review.
They are distinct questions: recording a value does not establish that it was measured or validated.
- Record the run identifier as a stable identifier for the intended training lineage.
- Record the parent generation so a reviewer can reconstruct the recovery lineage.
- Record the generation identifier separately from the logical optimizer step.
- Record the last completed optimizer update, including whether skipped updates increment the counter.
- Record the number of tokens consumed when the learning-rate schedule is token-based.
- Record the number of examples consumed when the input contract is example-based.
- Record the accumulation phase and whether gradient buffers belong to the snapshot.
- Record the precision policy for weights, gradients, optimizer slots and master parameters.
- Record whether dynamic loss scaling is active and how skipped steps affect scheduler calls.
- Record the model architecture and immutable configuration identity.
- Record the code revision and dependency lock or container image digest.
- Record the framework release and checkpoint serialization schema.
- Record the optimizer implementation and configuration, including fused or sharded variants.
- Record the optimizer parameter-group identities and parameter-name mappings.
- Record scheduler type, state, schedule unit and call order.
- Record the data source snapshot and split identities.
- Record preprocessing, tokenization, packing and augmentation versions.
- Record sampler algorithm, epoch, seed and acknowledged cursor semantics.
- Record stream partition offsets separately from externally committed offsets.
- Record how buffered, prefetched and in-flight samples are handled on restart.
- Record the number of loader workers and their random-seed derivation policy.
- Record CPU, device and library generator ownership and restore order.
- Record deterministic algorithm settings and known nondeterministic operations.
- Record the number and type of accelerators involved in the saved topology.
- Record data-, tensor-, pipeline- and optimizer-parallel group dimensions.
- Record the logical tensor-to-shard mapping and its schema version.
- Record whether a target topology has been tested for restore and resharding.
- Record the expected object count and the exact identifiers of all required objects.
- Record object lengths and cryptographic digests for end-to-end integrity checks.
- Record manifest authentication and the identity authorized to publish it.
- Record storage service, region, namespace, and durability assumptions.
- Record whether staged files reside on ephemeral or durable storage.
- Record the publication primitive and its documented atomicity scope.
- Record the writer lease, fencing epoch and conditional pointer revision.
- Record the timestamp when the cut was created and when publication succeeded.
- Record failures and retries separately from successful checkpoint latency.
- Record per-rank snapshot capture time and write completion time.
- Record coordinator validation time and manifest publication time.
- Record the amount of temporary host and accelerator memory used by staging.
- Record logical payload bytes separately from physical replicated bytes.
- Record object counts and request rates to expose small-object overhead.
- Record effective aggregate write bandwidth over the actual checkpoint window.
- Record effective aggregate restore bandwidth over the actual recovery window.
- Record relevant competing workloads and storage throttling signals.
- Record the most recent successful restore drill and the stack version used.
- Record the most recent failed restore drill and its unresolved cause.
- Record recovery point objective in explicitly named steps, samples, tokens or time.
- Record recovery time objective with explicit start and end events.
- Record the age of the latest validated durable generation.
- Record the latest generation whose full restoration was actually exercised.
- Record whether acceptance requires exact replay or a weaker continuation guarantee.
- Record exact reasons any replay guarantee was downgraded after elasticity.
- Record the observed count of missing, duplicated or intentionally padded samples.
- Record the observed next learning rate before the first resumed update.
- Record scaler state and the next update's overflow or skip outcome.
- Record the sampled loss and numerical-health checks after resumption.
- Record error codes for failed reads, writes, checksum checks and conditional commits.
- Record which older generation remains available as a fallback.
- Record retention policy and all incremental ancestry needed by retained roots.
- Record the time and authorization under which garbage collection may run.
- Record key identifiers and access requirements without saving secret material.
- Record who can approve an intentional rollback or non-exact warm start.
- Record data retention and deletion obligations for checkpoint contents.
- Record the operator and incident identifiers for any emergency restore.
- Record whether the restored run has since created a new validated checkpoint.
- Record unresolved compatibility and source-documentation questions as open issues.

**Takeaway:** The evidence record must explain both what was saved and why a particular recovery was allowed.

# 10. Key Theorems / Results Reference

| Result | Validity / reasoning | Failure boundary |
|---|---|---|
| Complete-state restart | The next transition is determined by all mutable inputs and environment under the selected replay contract | Hidden state, nondeterministic operations, or changed environment invalidate exact replay |
| Atomic publication | A pointer update exposes either the old or new identifier if the actual primitive is atomic | It does not make previously written data durable |
| Conditional fencing | A stale writer cannot publish if the authority compares a monotonic epoch and rejects stale epochs | A token recorded but not enforced provides no fencing |
| Average periodic loss | Uniform failure location within a useful-work interval gives mean lost work T/2 | Nonstationary and correlated failures need another model |
| IO-1 | Bytes divided by effective rate gives payload-only transfer time | Rate changes, fixed latency and contention add uncertainty |
| Retention reachability | Incremental data is safe to delete only when no retained root references it | Concurrent publication or an untracked dependency can race the collector |
| Hash check | Matching a strong digest gives evidence that bytes match the manifest | It does not prove the manifest is trusted or state is logically correct |
| Elastic equivalence | Supported resharding can preserve tensor values and optimizer slots | It does not guarantee identical future batch order or floating-point trajectory |

Check the linked upstream documentation for the installed version before relying on specific distributed checkpoint, loader, or elastic APIs.
No source has been consulted here to establish the contents of the two unavailable course modules.
**Takeaway:** Mechanical integrity, semantic consistency, and training equivalence are separate claims requiring separate evidence.

# 11. Interview / Exam Questions

### 11.1 Why are weights alone insufficient for training continuation?

- **Answer:** An adaptive optimizer also needs moment tensors, step counters, groups and possibly master weights; scheduler, scaler, RNG and data position matter too.
- **Reasoning:** With moments reset, the same next gradient generally produces a different update.
- **Boundary / follow-up:** A weight-only load is valid as an explicitly labeled warm start, not an exact resume.

### 11.2 What makes a distributed checkpoint committed?

- **Answer:** Every required object must be durable and validated, then a manifest must be published by a fenced atomic or conditional commit.
- **Reasoning:** A process barrier establishes participation but is not a durable write or atomic publication.
- **Boundary / follow-up:** If the chosen storage cannot provide the required publication semantics, use a coordinator or a different backend.

### 11.3 A rank fails after seven of eight shards are written. What should recovery do?

- **Answer:** Reject the incomplete candidate and recover the latest earlier committed generation.
- **Reasoning:** The seven successful writes are orphan candidates, not proof that the logical cut is complete.
- **Boundary / follow-up:** Preserve evidence and later garbage-collect only after ensuring no retained manifest references those objects.

### 11.4 What changes if a checkpoint is taken halfway through gradient accumulation?

- **Answer:** The accumulation phase and every relevant gradient buffer must join the snapshot, including scaling and partition metadata.
- **Reasoning:** Without these, the next optimizer update combines an unknown subset of microbatches.
- **Boundary / follow-up:** The simpler alternative is rollback to a completed optimizer-update boundary.

### 11.5 Why does a saved seed not ensure replay?

- **Answer:** A seed specifies an initial stream, not the current generator position or every independent worker and device stream.
- **Reasoning:** Even complete RNG states do not fix nondeterministic kernels or changed sample order.
- **Boundary / follow-up:** Restore generator states and control relevant software, hardware and input behavior for the desired replay level.

### 11.6 How do prefetch and acknowledged data position differ?

- **Answer:** A worker may have fetched records that the trainer has not consumed in a completed update.
- **Reasoning:** Saving its fetched position would skip those records if queued work disappears at restart.
- **Boundary / follow-up:** Track the acknowledged logical training cursor, snapshot the queue, or deterministically replay from a safe earlier boundary.

### 11.7 Estimate write time for 240 GiB at 4 GiB/s and describe the missing terms.

- **Answer:** The payload-only time is 240 / 4 = 60 seconds.
- **Reasoning:** The GiB units cancel and the answer is not an observed latency percentile.
- **Boundary / follow-up:** Add measured synchronization, hashing, metadata and publication time; account for overlap only where it actually occurs.

### 11.8 A 40-minute interval has uniformly distributed failures. How much work is lost on average?

- **Answer:** Under that model the mean is half the interval, or 20 minutes of useful work.
- **Reasoning:** This is neither a maximum nor the full end-to-end recovery time.
- **Boundary / follow-up:** Frequent failure during long save phases, nonuniform hazard, and repeated retries require a more detailed model.

### 11.9 What changes when the world size doubles?

- **Answer:** The system must map model and optimizer shards to the new topology and reassign the remaining data without unintended gaps or duplicates.
- **Reasoning:** Effective global batch may change unless per-rank batch or accumulation is adjusted.
- **Boundary / follow-up:** Even a successful supported reshard can change reduction order and exact training trajectory.

### 11.10 What is the difference between RPO and RTO?

- **Answer:** RPO bounds acceptable lost progress; RTO bounds elapsed time from the defined failure event until resumed useful work.
- **Reasoning:** A one-minute save can still have a long recovery if scheduling or deserialization is slow.
- **Boundary / follow-up:** Measure both against explicit population, clock boundaries and failure scenarios.

### 11.11 An object digest matches. Is the checkpoint safe to resume?

- **Answer:** Not necessarily: the manifest may be untrusted, another shard may be missing, or the bytes may be semantically from an incompatible run.
- **Reasoning:** A hash verifies byte agreement with the stated digest, not checkpoint completeness or provenance.
- **Boundary / follow-up:** Validate every required object, trusted manifest, layout, logical cut and framework compatibility.

### 11.12 A latest pointer moves backward after failover. What would you investigate?

- **Answer:** Compare writer epochs, conditional-update outcomes, pointer object versions and intentional rollback records.
- **Reasoning:** A stale coordinator may have overwritten a newer generation if fencing was absent.
- **Boundary / follow-up:** Enforce a publication authority that rejects stale epochs, then explicitly select a validated generation.

**Takeaway:** Explain the invariant, a competing explanation, and what evidence would change the decision.

# 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Distributed training checkpoint consistency, durability and recovery |
| Core formula | Payload transfer time = unique checkpoint bytes / effective aggregate throughput |
| Key assumption | The published, fenced manifest names all consistent, validated and durable state; restore semantics are tested |
| Complexity | Payload storage and transfer scale with serialized state size; metadata, resharding, replication and contention add costs |
| Use when | Restarting or sizing a distributed training job with explicit recovery objectives |
| Avoid when | A weight-only export or partially written shard set is mistaken for a restartable committed checkpoint |
| Go deeper | [Collection index](INDEX.md); the installed framework and storage provider's documentation |
| Diagnostic | Compare the manifest, object identities, cursor, counters and optimizer state before repair |
| Evidence | Illustrative arithmetic; source files unavailable, operational performance unmeasured |
