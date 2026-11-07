---
case_id: "CASE-2026-FT01"
case_title: "Should support draft generation use a domain adapter?"
case_version: "1.0"
created: "2026-11-07"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "llm-agentic"
domain: "Customer support language-model adaptation"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["data-leakage", "atomic-release-bundle"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-07"
owner_role: "Support operations lead"
source_index: "INDEX.md"
review_trigger: "New policy, model, data rights, rubric, or serving stack"
---

# Should support draft generation use a domain adapter?

## 1. The One-Line Framing

Given a support ticket and authorized policy excerpt at draft time,
recommend a human-reviewed reply for an English support agent,
subject to policy accuracy, privacy and latency constraints,
compared with a manual response and a prompted frozen base model.
The proposed decision is whether an adapter merits a controlled pilot,
not whether an autonomous model should issue refunds.
The case is an illustrative design exercise, with invented values and no real organization,
run, deployment, measured impact or claim of a passed gate.
The reusable pattern is evaluation-mismatch: a better offline proxy need not improve work.
Data-leakage and atomic-release-bundle are supporting patterns.
The primary success condition is a materially higher independently reviewed acceptable-draft rate.
Acceptable language alone cannot compensate for wrong policy or unauthorized disclosure.

## 2. Context & Background

The fictional product is a subscription service with a staffed English-language email queue.
An agent reads a customer ticket, checks the current policy, and writes a reply.
A supervisor can audit decisions and resolve exceptions.
A proposed assistant would prepare a draft only; the agent remains responsible for sending it.
The agent can reject the draft and write manually at any point.
The support lead owns the business decision; ML engineering owns the model artifacts.
Data governance owns access and deletion; safety reviewers own adverse-content assessment.
The security team owns identity, tenant isolation and incident escalation.
The service owner owns runtime availability and rollback.
An assumed workload is 12,000 eligible drafts per business day.
Only English email tickets in the fictional jurisdiction are eligible for the first test.
No historical intervention or prior attempt is claimed.
The manual baseline persists because it gives agents control and policy context.
Its actual time and accuracy are unknown, not established weaknesses.
Policy may change after a ticket is created, so a memorized answer can become stale.
No legal applicability is asserted; counsel must determine applicable obligations.
The time, geography, volume and budget are scenario assumptions, not observed facts.

## 3. The Business Question

Illustrative stakeholder question: should a domain-adapted model enter a limited agent-only pilot?
The decision owner can retain manual drafting, pilot a prompted frozen model,
pilot a LoRA adapter, or defer while collecting evidence.
A QLoRA candidate is a memory-constrained alternative, not an assumed winner.
The deadline is a hypothetical quarterly planning review, not an actual commitment.
The intended value is lower agent effort without sacrificing safe policy adherence.
The downside is customer harm, privacy exposure, rework, and incorrect entitlements.
The model proposes text; it cannot send, refund, change an account, or call tools.
A draft-quality proxy can improve while agents spend longer correcting it.
Therefore a subsequent randomized pilot must measure agent effort and customer outcomes.
Until then no business benefit is established.
A no-change decision remains valid if uncertainty or risk is unacceptable.

## 4. Assumptions & Constraints

| ID | Scenario assumption or constraint | Verification and owner | Consequence and fallback |
|---|---|---|---|
| A01 | Training examples may be used for adaptation | Governance checks consent, purpose and retention | Exclude unauthorized records; defer if coverage fails |
| A02 | Historical agent replies are imperfect labels | Operations audits a stratified sample | Correct or discard; do not train blind |
| A03 | Current policy text is retrievable at response time | Policy owner tests dated access | Abstain and draft manually on missing policy |
| A04 | A suitable base checkpoint and license exist | Legal and ML review actual model terms | Choose an authorized checkpoint or no model |
| A05 | Compute can run the selected recipe | Engineering profiles actual peak memory | Reduce sequence length or use QLoRA, subject to quality gate |
| A06 | Agents have capacity to review every draft | Operations measures pilot throughput | Reduce exposure or use manual workflow |
| C01 | No autonomous customer action | Product control and authorization test | Reject any release that bypasses review |
| C02 | No protected raw customer data in ordinary logs | Security reviews telemetry | Suppress logs and investigate exposure |
| C03 | Release safety and quality gates are mandatory | Named reviewers sign versioned record | Hold rollout on failure or insufficient evidence |
| C04 | Proposed p95 end-to-end draft latency at most 8 seconds | Measure at user boundary under representative load | Return manual workflow when deadline expires |
| C05 | Proposed service availability at least 99.5% monthly | Eligible requests receiving a draft or explicit safe fallback | Pause expansion and investigate error budget |

C04 and C05 are fictional targets, not verified service obligations.
A01, A03 and A06 are the most likely assumptions to reverse this design.
Absent source access or reliable policy retrieval, the manual baseline wins by default.
Changing the policy source, access right or model revision triggers revalidation.

## 5. Data, Tools & Method

### 5a. Data Specification

The row grain is a single ticket and the agent-approved final reply.
The group key is a pseudonymous customer ID; ticket and thread IDs remain separate.
Source rows have ticket creation time, customer ID, thread ID, channel, language,
customer message, policy version effective at response time, final response and labels.
The response timestamp is required to reconstruct the information available to the agent.
Resolution code is a delayed outcome, not a model input.
Internal notes, billing identifiers, payment details and free-text attachments are excluded.
Assume 100,000 candidate ticket rows before cleaning, not observed data.
Assume cleaning leaves 80,000 eligible rows; this is an input assumption, not a finding.
An actual pipeline must publish exclusion counts by reason and segment.
Only final replies independently checked against policy can serve as target text.
Approval by an agent alone is not proof of correctness.

Split by chronological customer cohort and thread, not individual message.
Illustrative planned proportions after eligibility are 64,000 train, 8,000 validation,
and 8,000 final test rows; these counts require construction and reconciliation.
Training uses records with decision time before 2025-10-01 UTC.
Validation uses 2025-10-01 through 2025-11-30 UTC.
Test uses 2025-12-01 through 2026-01-31 UTC.
A customer appearing in multiple periods is assigned to its earliest period;
all later tickets for that customer are excluded from validation and test.
All thread and near-duplicate template clusters crossing a boundary are quarantined.
If those removals change the stated counts, actual counts replace these assumptions.
Allocate partitions before text fitting, synthetic augmentation or label correction.
Do not sample the test set to guide prompt, adapter, checkpoint, or threshold selection.
Validation is for model selection; the final test is one locked confirmatory comparison.
Dates are illustrative; no data extraction has occurred.

At each response time, join policy by effective-from <= response time < effective-to.
Require one authorized version for the product and region; quarantine overlaps and gaps.
Never join on a current policy snapshot when reconstructing a historical decision.
Train on the ticket and the then-effective approved policy excerpt, not future policy text.
Hold out newly introduced policies separately to test generalization across revisions.
Use anonymized IDs and train-only deduplication models; keep a private cross-split audit.
Remove exact and semantic duplicates with documented thresholds and human review.
A fixed common policy passage is permitted in multiple examples as shared context,
but an answer template copied from a held-out ticket is not.
The final test stays inaccessible to researchers during selection.

### 5b. Tools & Stack

Proposed stack: Python training job, a version-pinned transformer runtime,
a parameter-efficient fine-tuning library, an approved frozen decoder-only base,
a quantized-weight backend if QLoRA is evaluated, and a restricted artifact registry.
No model, hardware SKU, library release or API compatibility is asserted as selected.
A hypothetical single accelerator is a planning unit; its capacity must be profiled.
An independent evaluation runner loads frozen manifests and produces signed reports.
A policy service supplies authorized current excerpts; it is not a learned model.
A human-review interface presents draft and source to the agent without auto-send.
Security-approved storage and identity boundaries apply to all proposed components.

### 5c. Method Summary

| Candidate | Status | Advantage | Boundary |
|---|---|---|---|
| Manual drafting | Considered, not measured | Human judgment and no model dependency | Labor cost and quality distribution unknown |
| Frozen base with same prompt and policy | Required comparator, not tested | No training or adapter operations | May miss domain format or conventions |
| LoRA supervised adaptation | Proposed candidate, not tested | Train a small low-rank update | Can overfit style or stale information |
| QLoRA supervised adaptation | Proposed candidate, not tested | Frozen quantized base can reduce base-weight memory | Quantization quality and kernel compatibility require testing |
| Full-parameter tuning | Considered, not tested | Maximum adjustment flexibility | More trainable state and operational cost |

The frozen baseline and adapters receive the same task prompt,
policy snapshot, allowed output budget and review rubric.
Train LoRA only on approved assistant response tokens;
mask user, policy, padding and system tokens from the supervised loss.
Check shifting and padding against the actual model's tokenization contract.
The objective is next-token negative log likelihood averaged over supervised tokens,
not an objective directly measuring customer outcomes.
The optimizer updates adapter parameters only; the base remains frozen.
Select checkpoint on validation quality and safety; never use test labels to select.
Start with rank 8, scaling alpha 16, dropout 0.05 and fixed seed as hypotheses.
These values are candidate settings, not optimal hyperparameters or tested results.
Limit comparison to a preregistered small configuration set and equal compute accounting.

For a frozen weight W with d_out rows and d_in columns,
LoRA adds delta W = (alpha/r) B A, where A is r by d_in,
and B is d_out by r; the forward weight is W + delta W.
The base is frozen; only A and B receive optimizer updates.
The rank of the update is at most r, not necessarily equal to r.
With QLoRA, frozen base weights are stored in a supported low-bit representation,
then dequantized for computation while trainable adapters use a supported higher precision.
Quantization metadata and compute dtype matter; 4 bits is not total training memory.
Adapter capacity does not give authorization, factual freshness or safety guarantees.

### 5d. Provenance, quality, exclusions and leakage controls

The source dataset, permission evidence and snapshots do not exist for this exercise.
A future manifest must record extraction time, owner, rights, source versions and retention.
Pseudonymization is not anonymization; re-identification risk needs access controls.
Remove direct identifiers before training and audit residual personal information.
Keep raw mapping in a separate controlled system with a deletion workflow.
Audit language, product, reason, ticket age, policy version and agent distributions.
A future dataset may underrepresent rare escalations and customers who never contact support.
Missing policy references, broken timestamps or unresolved deletion requests cause quarantine.
Use a separate reviewer from the response author for gold-label sampling.
Training preprocessing statistics, deduplication models and learned filters use training only.
Record every discarded group, its reason and any representativeness consequences.
Do not include a future refund outcome or final agent decision in a draft-time prompt.
The policy text is evidence for current facts; model weights are not an authoritative policy store.

### 5e. Reproducibility and method-selection ledger

| ID | Source or artifact | Scope and availability | Reproducibility boundary |
|---|---|---|---|
| SRC-01 | [Course directory](../../../../Templates/directory.txt) | Inventory identifies 17.3 and 17.8 | Names are not contents or measurements |
| SRC-02 | [Focused source coverage](../../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-playbook-training) | Maps adapter implementation to module 21 | Planned instructional scope only |
| SRC-03 | [Evaluation coverage](../../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-playbook-operations) | Maps evaluation implementation to module 26 | Not evidence of executed evaluation |
| SRC-04 | [LoRA paper](https://arxiv.org/abs/2106.09685) | Original low-rank adaptation method | Not evidence for this support task |
| SRC-05 | [QLoRA paper](https://arxiv.org/abs/2305.14314) | Quantized frozen model plus adapter approach | No deployment or performance claim here |
| RUN-01 | No run artifact exists | Proposed training and evaluation only | Cannot reproduce a nonexistent run |
| DATA-01 | No authorized dataset exists | Hypothetical cohort | Counts and quality unverified |

The original course files are named in the directory inventory but unavailable at the expected path.
Their internal text has not been inspected and is not paraphrased as sourced evidence.
The papers identify methods; no current framework API, price or hardware claim relies on them.
A future run manifest records code commit, dependency lock, container digest,
base revision, tokenizer, policy snapshot, split IDs, seed, masks and target modules.
It must include optimizer, learning schedule, checkpoint, precision and training hardware.
Evaluation manifests include prompt, rubric, decoding, judge revision, sample and outputs.
Record access restrictions and retention rather than publishing customer examples.
Seeds aid reproduction but do not guarantee bitwise determinism across platforms.

## 6. Analysis Walkthrough

The defensible current result is a decision procedure, not a measured model advantage.
The candidate can proceed only if it beats the frozen baseline and respects all guardrails.
No approach was actually trained or rejected; all comparisons below are proposed.

### 6.1 Adapter parameter sizing

Illustrative model geometry: 32 layers, two targeted square projection matrices per layer,
each 4096 by 4096, rank r = 8, with no biases or additional trainable modules.
The number of adapter parameters for one matrix follows from the two factors:

```text
FORMULA [Q1]: P = L * M * r * (d_in + d_out)
WHERE: P parameters; L layers; M target matrices per layer;
       r rank; d_in and d_out matrix dimensions, all counts.
CONSTRAINT: each targeted matrix has this geometry; biases excluded.
NUMERICAL EXAMPLE: P = 32 * 2 * 8 * (4096 + 4096)
                   = 4,194,304 parameters.
INTERPRETATION: this is adapter count, not total model or runtime memory.
```

Independent check: 65,536 adapter parameters per matrix times 64 matrices
is 4,194,304; a full matrix has 16,777,216 parameters,
and 64 full matrices have 1,073,741,824 parameters.
Thus the adapter is 1/256 of those full matrices, not of the entire model.
At rank 16 the adapter count doubles to 8,388,608;
that adds capacity and trainable state but has no guaranteed quality improvement.
Actual grouped-query attention dimensions may differ; calculate each target separately.

### 6.2 Memory estimate

Assume, illustratively, a 7-billion-parameter base stored in two-byte precision.
Its raw base weights alone are 14 billion bytes, approximately 13.04 GiB.
At a nominal four bits per base parameter, raw weights are 3.5 billion bytes,
approximately 3.26 GiB, before scales, metadata, unquantized modules and alignment.
For the adapter, assume 2-byte weights, 2-byte gradients,
and two 4-byte optimizer moments per trainable parameter.

```text
FORMULA [Q2]: S = P * (b_weight + b_gradient + b_m1 + b_m2)
WHERE: S bytes; P trainable parameters; each b is bytes per parameter.
CONSTRAINT: stated storage layout only; no master copy or temporary buffers.
NUMERICAL EXAMPLE: S = 4,194,304 * (2 + 2 + 4 + 4)
                   = 50,331,648 bytes = 48 MiB.
INTERPRETATION: adapter state is small relative to base, but not peak memory.
```

Cross-check 4,194,304 parameters = 4 Mi parameters, times 12 bytes = 48 MiB.
Activations, optimizer implementation, embeddings, temporary buffers, fragmentation,
quantization scales, context length and checkpointing can dominate the remainder.
Do not subtract this raw estimate from GPU capacity and claim the run will fit.
At rank 16 this illustrative adapter state is 96 MiB with identical storage assumptions.
A full-finetuning comparison needs gradients and optimizer states for all unfrozen weights;
the simple raw-base comparison alone is not a full-training memory estimate.

### 6.3 Training and evaluation workload

With microbatch 2 sequences per device, accumulation 8, and four data-parallel devices:

```text
FORMULA [Q3]: B = microbatch * accumulation * data_parallel_degree
WHERE: B is sequences per optimizer update, assuming full equal microbatches.
CONSTRAINT: no dropped or partial batches and no sequence packing adjustment.
NUMERICAL EXAMPLE: B = 2 * 8 * 4 = 64 sequences/update.
INTERPRETATION: normalize loss by actual supervised tokens when lengths vary.
```

If each sequence has exactly 1,024 supervised answer tokens,
64 * 1,024 = 65,536 supervised tokens per update.
The independent reverse check is 65,536 / 1,024 = 64 sequences.
This is a hypothetical distributed sizing example, not a promise of four available GPUs.
If a final microbatch is partial, the true count and denominator must be used.
Packing changes the number of sequences represented, so count actual labeled tokens.

For 500 validation prompts, three checkpoints and two decoding seeds,
500 * 3 * 2 = 3,000 planned generations per candidate family.
At an assumed 0.02 fictional currency units per generation, the arithmetic estimate
is 60 units; independent check: 500 * 0.02 * 6 = 60 units.
This excludes baseline, annotation, retries, storage, failures, and infrastructure.
Doubling seed count doubles generation cost under this simple constant-unit-cost model.
Real costs require a dated quote and token-level accounting; no price is claimed here.

### 6.4 Decision trace and robustness

1. Create an authorized, time-correct dataset and quarantine cross-split clusters.
2. Obtain human-rated frozen-base and manual comparators on matched eligible examples.
3. Train a small LoRA configuration; optionally train QLoRA under the same protocol.
4. Verify frozen weights, loss masks, checkpoint identity and intended modules.
5. Select only on validation; lock the final candidate and rubric before test access.
6. Blind reviewers to model identity and evaluate the same customer-grouped test set.
7. Check safety, privacy, latency, subgroup uncertainty and operational recovery.
8. If any hard gate fails or cannot be measured, do not pilot.
9. If all gates pass, seek owner approval for a randomized, reversible agent-only pilot.

Sensitivity: doubling rank doubles adapter count under Q1 but may worsen overfit.
If duplicate removal shrinks test groups, confidence widens and the pilot decision may defer.
If quantization changes outputs, independently evaluate the exact serving export.
A numerical arithmetic check is not a benchmark, significance test, or evidence of benefit.

## 7. The Finding

Do not authorize a pilot on the available evidence: the case has no dataset, run or measurements.
A LoRA candidate is reasonable to evaluate against the frozen prompted base and manual process,
but its relative quality and any business impact are unknown (CLM-01; Sections 6, 9 and 11).
The primary boundary is that policy accuracy, safety and operational feasibility are untested.
The next action is to obtain authorized data and execute the locked comparative evaluation.

## 8. System/Solution Design

### 8a. Offline / Training / Batch Path

```text
Authorized ticket archive + historical policy versions
  -> rights, redaction, time join and group split -> quarantine on ambiguity
  -> curated train -> base + adapter training -> checkpoint + immutable manifest
  -> validation selection -> locked test -> independent safety and quality gates
  -> signed release bundle in registry -> release approval or manual-only decision
```

The governance owner signs the dataset manifest before access is granted.
The training service can read only the training partition and approved validation material.
An independent evaluation service controls the locked final test.
The registry binds base, adapter, tokenizer, quantization configuration,
prompt template, policy interface, evaluation report and runtime revision.
Failed lineage, missing signatures or mismatched components block promotion.
Checkpoint retention and deletion propagate through registry references and backups.
Dataset changes require a new version; never overwrite a release artifact in place.

### 8b. Online / Serving / Real-Time Path

```text
Agent request -> identity and ticket authorization -> fetch current policy
  -> validate policy scope/freshness -> pinned model bundle -> output safety checks
  -> draft + policy references -> agent review -> optional send by agent
  -> restricted audit and feedback; failures -> explicit manual workflow
```

The request includes a request ID, ticket ID and authorized agent identity.
The service resolves ticket and policy under the caller's scope; caller-supplied policy is untrusted.
No customer message, secret, or policy instruction can override system permissions.
The draft cites only retrieved policy identifiers actually supplied to the model.
The agent sees source policy and remains responsible for checking applicability.
No request may be replayed as a customer send; generation has no send permission.
An expired policy, authorization failure, unavailable model or timeout returns no draft.
It records a reason code without embedding sensitive ticket contents in routine logs.
At cold start, readiness must verify bundle identity and policy-service connectivity.
At overload, admission control rejects excess requests into manual handling.
Retries are bounded by the same request deadline; no unbounded generation retry.

Illustrative successful trace: an authorized agent opens an eligible ticket;
the service resolves the current policy, generates a draft and supplies its policy ID;
the agent checks the source, edits the answer, and sends through the existing workflow.
Illustrative failed trace: policy lookup returns an expired version;
the service returns `POLICY_UNAVAILABLE`, emits a redacted event,
and the agent drafts manually; the policy owner investigates the source freshness.
Neither trace has been executed.

### 8c. Key Design Decisions

| ID | Choice | Alternative and tradeoff | Revisit trigger |
|---|---|---|---|
| DEC-01 | Agent-only drafts | Auto-send might save effort but increases harm | Independent evidence and a new approval process |
| DEC-02 | Current policy retrieval | Memorized policy is cheaper but becomes stale | Policy source cannot meet freshness requirements |
| DEC-03 | LoRA candidate | Full tuning allows broader changes at higher state cost | Adapter cannot meet quality gate |
| DEC-04 | Frozen baseline | Comparing only to manual hides adaptation value | Baseline cannot use the same policy context |
| DEC-05 | Atomic release bundle | Loose components permit accidental mismatches | Runtime or tokenizer changes |
| DEC-06 | Hold on unknown safety | Assuming pass increases exposure | Sufficient adjudicated evidence exists |

### 8d. Architecture artifacts and operational readiness

An idempotency key deduplicates generation requests within a bounded retention period;
it does not authorize sending or make generation deterministic.
Access is least privilege, with short-lived credentials and isolated customer scopes.
Secrets are stored outside model prompts and logs.
Service SLI numerator: eligible requests yielding an in-deadline draft or explicit safe fallback.
Denominator: all eligible authorized draft requests during a calendar month.
The illustrative availability target is 99.5%; fallback rate is monitored separately,
because a service returning only fallback can satisfy availability without delivering value.
The latency metric is p95 from agent submission to draft display,
measured directly on successful draft responses; do not sum component percentiles.
Report timeout and fallback rates alongside latency to avoid survivorship bias.
Owners define on-call routes and a monthly error-budget response before pilot approval.

Capacity: 12,000 daily eligible requests over eight hours implies
12,000 / (8 * 3,600) = 0.4167 requests per second average.
At a hypothetical 5x peak multiplier, provision for 2.083 requests per second,
subject to load test and measured burst distribution.
If mean generation service time is assumed to be 3 seconds,
Little's law gives about 2.083 * 3 = 6.25 concurrently serviced requests at that rate,
assuming a stable system and averaging compatible quantities.
This does not bound queueing, memory, p95 latency or concurrency under bursts.
Doubling the peak multiplier from 5 to 10 implies 4.167 requests per second;
the same mean duration implies 12.5 average in-service requests.
Measure token throughput, queue delay, utilization and real output lengths before sizing replicas.

Rollout requires offline gates, dependency tests, shadow evaluation, explicit sign-off,
then a randomized agent-only canary with a manual fallback and holdback.
Rollback selects the last compatible signed bundle or disables generation entirely.
Keep policy schemas backward-compatible or roll back service and model together.
The operator can stop traffic immediately without deleting audit records.
A restore drill must verify the chosen bundle, permissions and policy compatibility.
No rollout, canary or restore drill has occurred.

## 9. Evaluation & Validation

All following protocols and thresholds are proposed; none has been executed.
The primary metric is acceptable-draft rate: number of test tickets rated acceptable
without substantive factual, policy, safety or required-content correction,
divided by all eligible test tickets assigned to that model, including failures.
The unit is percent of tickets in the locked time-window test population.
Timeouts, missing outputs and refusals on benign eligible requests count as not acceptable.
Compare LoRA to the frozen base on the same tickets and same policy context.
The proposed practical gate is at least 5 percentage points improvement,
with a lower 95% cluster-bootstrap confidence bound above zero.
This gate is chosen for the exercise, not from measured power or a stakeholder commitment.
Report the absolute difference and baseline rate; do not call percentage points relative percent.
Cluster bootstrap resamples customers, retaining all their paired ticket outcomes;
its assumptions include representative independent customer clusters.
A fixed sample may be too small for a rare subgroup or a small effect; then defer.

Safety guardrail: no increase in adjudicated severe policy or privacy violations;
use a prespecified risk limit and interval, not merely a zero observed count.
Define severe examples and the acceptable upper confidence bound with safety owners before testing.
Zero observed violations is not proof of zero risk; rare-event sample requirements may be large.
Other gates include no material degradation by reason and language subgroup,
no unauthorized personal-data disclosure in targeted extraction tests,
and p95 latency and fallback targets from Section 8 measured under realistic load.
No single aggregate score can override a severe-violation gate.
Compare QLoRA and LoRA separately; use the exact exported serving bundle for final tests.
A multiple-comparison plan must be fixed before selecting among candidates.

Sample construction must include common reasons plus targeted rare, adversarial,
out-of-distribution, policy-change and privacy scenarios; report sampling weights.
A stress set deliberately enriched for attacks is not an estimate of population prevalence.
Two trained reviewers independently score blinded, randomized paired outputs;
calibrate with a separate training set, measure agreement, adjudicate disagreements,
and preserve the original scores and adjudication rationale.
Reviewers must see the authorized policy version and a rubric, not the model identity.
An automated judge may triage, but must be versioned and calibrated against human labels;
it cannot replace independent human review of the severe-violation gate without validation.
Review artifacts include dataset version, sample IDs, rubric, model bundle and decodes.
No decision is passed or failed because no test or report exists.

Pilot proposal: randomly assign eligible agents by stable agent cluster to draft or control,
include all eligible tickets, and record assignment, exposure and non-use.
Agent-level assignment reduces within-agent contamination but creates clustering;
measure sample-ratio mismatch and use cluster-aware analysis.
Control is the existing manual process; isolate the adapter's added value separately offline.
Measure handling time from ticket opening to disposition, rework, escalations,
quality, customer complaints, adoption and adverse outcomes over a prespecified window.
Intention-to-treat analysis includes assigned-but-unused tickets; report usage separately.
Supervisor overrides, agent learning and case-mix changes can distort interpretation.
No inferred causality or business saving is available until a properly analyzed pilot exists.

## 10. Action Taken

No action taken as of 2026-11-07; this is a proposed instructional exercise.
No organization has approved a pilot or executed a training job in this case.
The proposed support owner would first commission data-rights and policy-source checks.
ML engineering would build the baseline and independent evaluation harness.
Safety and operations would sign the rubric and stop rules before any release.
No customer-facing feature, model, experiment or autonomous action has been deployed.

## 11. Outcome & Results

**Source declaration:** all figures are constructed teaching values, not company measurements.
No measured outcome, run artifact or intervention exists.

| Metric | Baseline | Candidate | Change | Evidence and limitation |
|---|---|---|---|---|
| Acceptable drafts | Not measured | Not measured | Unknown | CLM-01; no test run |
| Severe violations | Not measured | Not measured | Unknown | No safety evaluation |
| Agent handling time | Not measured | Not measured | Unknown | No randomized pilot |
| End-to-end p95 latency | Not measured | Not measured | Unknown | No serving deployment |
| Cost per accepted draft | Not measured | Not measured | Unknown | No observed workload or price |

The proposed measurement owner is support operations with an independent evaluation lead.
The proposed offline window is the locked December–January test cohort in Section 5.
A future pilot window and stopping rule require approval before randomization.
Do not treat Section 6 sizing arithmetic as an outcome or realized cost.

## 12. Failure Modes & What Went Wrong

No incident occurred in this hypothetical case; the following are prospective diagnostics.

### F-01. Cross-period customer leakage

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Test quality is unexpectedly close to training quality.
- Competing explanations: Repeated customer or true generalization.
- Discriminating diagnostic: Join pseudonymous customer IDs across splits; expect zero intersections.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Quarantine all later-period rows for overlapping customers.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Data engineering.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Rebuild split and recount unique customers.
- Limitation: the proposed diagnostic has not been executed.

### F-02. Near-duplicate answer leakage

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Validation score jumps after importing archived replies.
- Competing explanations: Copied templates or genuine improvement.
- Discriminating diagnostic: Search cross-split response clusters and manually inspect similarities.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Quarantine clusters spanning split boundaries.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Data engineering.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Review sampled neighbors and repeat frozen evaluation.
- Limitation: the proposed diagnostic has not been executed.

### F-03. Future policy leakage

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Old tickets receive information from a later policy.
- Competing explanations: Invalid time join or timestamp error.
- Discriminating diagnostic: Compare effective intervals with historical response timestamp.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Reject unmatched or overlapping policy versions.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Policy data owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Inject overlapping versions and require quarantine.
- Limitation: the proposed diagnostic has not been executed.

### F-04. Wrong loss masking

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Training loss falls but response quality does not.
- Competing explanations: Prompt tokens dominate loss or labels are misaligned.
- Discriminating diagnostic: Inspect decoded tokens beside labels and compare shifted loss.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Correct collator masks and rebuild checkpoint.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: ML engineering.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Unit-test masked positions and a tiny known batch.
- Limitation: the proposed diagnostic has not been executed.

### F-05. Frozen adapter

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Training loss stays flat from first updates.
- Competing explanations: Adapter parameters frozen or task intrinsically difficult.
- Discriminating diagnostic: List trainable tensors and inspect adapter gradient norms.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Repair target selection before launching another run.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: ML engineering.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Check optimizer membership and nonzero gradient on a controlled batch.
- Limitation: the proposed diagnostic has not been executed.

### F-06. Unfrozen base

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Memory exceeds budget and checkpoints become unexpectedly large.
- Competing explanations: Base accidentally trainable or optimizer retaining extra states.
- Discriminating diagnostic: Enumerate requires-gradient tensors and optimizer parameter groups.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Stop run; freeze base and recreate optimizer.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: ML engineering.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Compare base tensor hashes before and after a step.
- Limitation: the proposed diagnostic has not been executed.

### F-07. Wrong target modules

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Adapter has far fewer parameters than predicted.
- Competing explanations: Architecture uses different names or target filter is wrong.
- Discriminating diagnostic: Compare targeted module names and shapes to model configuration.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Use explicit reviewed module selectors.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: ML engineering.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Recompute parameter count for every targeted shape.
- Limitation: the proposed diagnostic has not been executed.

### F-08. QLoRA export mismatch

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Offline runtime and serving responses differ materially.
- Competing explanations: Quantization settings, tokenizer or kernel mismatch.
- Discriminating diagnostic: Replay same prompt and decoding configuration on both bundles.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Hold export; align supported settings and rerun gates.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Serving owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Compare tokens and task metrics under documented tolerances.
- Limitation: the proposed diagnostic has not been executed.

### F-09. Quantization instability

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Quantized model produces non-finite scores.
- Competing explanations: Unsupported compute dtype or numerical issue.
- Discriminating diagnostic: Inspect activations, scales and finite-value checks.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Disable affected quantization path and use safe baseline.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: ML engineering.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Repeat finite-value and safety tests on serving hardware.
- Limitation: the proposed diagnostic has not been executed.

### F-10. GPU memory exhaustion

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Run fails during long examples, despite low weight estimate.
- Competing explanations: Activation peak or fragmentation rather than base weights.
- Discriminating diagnostic: Profile allocated and reserved peak by sequence length.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Reduce length or batch while preserving data coverage.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Training owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Run worst-case batch and verify peak headroom.
- Limitation: the proposed diagnostic has not been executed.

### F-11. Unequal token normalization

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Distributed run differs from single-device reference.
- Competing explanations: Per-device averaging weights short batches incorrectly.
- Discriminating diagnostic: Compare summed loss and supervised-token counts across ranks.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Normalize by global supervised-token count.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Training owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Compare gradients on an identical small fixed batch.
- Limitation: the proposed diagnostic has not been executed.

### F-12. Stale checkpoint selection

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Report attributes an old checkpoint to a new run.
- Competing explanations: Mutable tags or manifest mismatch.
- Discriminating diagnostic: Compare checkpoint digest against report and registry.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Block promotion and regenerate report with immutable IDs.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Evaluation owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Inject mismatched digest and check gate rejection.
- Limitation: the proposed diagnostic has not been executed.

### F-13. Wrong tokenizer

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Responses contain malformed tokens or missing stop markers.
- Competing explanations: Tokenizer drift or decoding bug.
- Discriminating diagnostic: Compare tokenizer digest and special-token mapping with training.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Revert to the pinned tokenizer in release bundle.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Serving owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Replay fixed prompts with exact same tokenizer and stop rules.
- Limitation: the proposed diagnostic has not been executed.

### F-14. Policy service stale

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Model confidently cites an expired rule.
- Competing explanations: Policy cache staleness or incorrect effective-time join.
- Discriminating diagnostic: Inspect policy version, TTL and source effective interval.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Stop drafts for affected policy scope and use manual handling.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Policy owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Advance a simulated policy clock and verify expiry behavior.
- Limitation: the proposed diagnostic has not been executed.

### F-15. Prompt injection in ticket

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Customer message instructs the assistant to ignore policy.
- Competing explanations: Untrusted input obeyed or policy retrieval compromised.
- Discriminating diagnostic: Replay malicious ticket with benign authorized policy.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Treat ticket as data; prevent tool or send permissions.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Security owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Add attack regression cases and inspect agent-visible output.
- Limitation: the proposed diagnostic has not been executed.

### F-16. Sensitive data in logs

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: A diagnostic event contains customer text.
- Competing explanations: Overbroad exception logging or telemetry configuration.
- Discriminating diagnostic: Inspect restricted redacted event samples and sink configuration.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Disable sink and follow incident procedure.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Security owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Test exception paths for raw-text and secret leakage.
- Limitation: the proposed diagnostic has not been executed.

### F-17. Unauthorized policy access

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: A ticket receives another customer scope policy.
- Competing explanations: Broken authorization or cache key collision.
- Discriminating diagnostic: Compare caller scope, policy request and cache partition keys.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Disable affected route and revoke unauthorized access.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Security owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Run cross-scope negative tests and audit access logs.
- Limitation: the proposed diagnostic has not been executed.

### F-18. Overconfident unsupported reply

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Draft gives a policy conclusion without supporting excerpt.
- Competing explanations: Hallucination or incomplete retrieval.
- Discriminating diagnostic: Compare generated claim with retrieved evidence and reference IDs.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Return no draft when policy evidence is missing.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Product owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Blindly score unsupported claims after correction.
- Limitation: the proposed diagnostic has not been executed.

### F-19. Invalid automatic safety judge

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Judge scores prefer a verbose but unsafe answer.
- Competing explanations: Judge length bias or rubric mismatch.
- Discriminating diagnostic: Compare blinded judge scores with independently adjudicated labels.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Use humans for severe-risk gate until calibrated.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Evaluation owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Recalibrate on a held-out diverse labeled sample.
- Limitation: the proposed diagnostic has not been executed.

### F-20. Rare slice regression

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Overall acceptable rate rises while a rare complaint type degrades.
- Competing explanations: Aggregation hides subgroup or sample is too small.
- Discriminating diagnostic: Report subgroup numerator, denominator and cluster interval.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Limit scope or defer until subgroup risk understood.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Support lead.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Evaluate a targeted but separately reported slice.
- Limitation: the proposed diagnostic has not been executed.

### F-21. Pilot contamination

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Control agents use copied treatment drafts.
- Competing explanations: Cross-arm sharing or assignment instrumentation error.
- Discriminating diagnostic: Audit assignment and observe workflow sharing with privacy safeguards.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Stop or redesign randomization at a wider cluster.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Experiment owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Verify exposure logs and cluster boundaries before restart.
- Limitation: the proposed diagnostic has not been executed.

### F-22. Missing outcome denominator

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Dashboard quality rises when timeouts disappear from reports.
- Competing explanations: Failures excluded or traffic mix changed.
- Discriminating diagnostic: Reconcile all eligible requests against output and failure events.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Restore intent-to-treat denominator and expose fallback rate.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Analytics owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Inject timeout events and verify denominator includes them.
- Limitation: the proposed diagnostic has not been executed.

### F-23. Review fatigue

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Reviewers approve fluent but incorrect drafts.
- Competing explanations: High workload or misleading presentation.
- Discriminating diagnostic: Blindly rescore a sample and compare with workload and error type.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Reduce review load and require policy evidence.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Operations owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Track adjudicated disagreement across workload bands.
- Limitation: the proposed diagnostic has not been executed.

### F-24. Training on deleted data

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: A deletion request arrives after model training.
- Competing explanations: Data lineage gap or incomplete deletion propagation.
- Discriminating diagnostic: Trace customer group through dataset, checkpoints and backups.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Quarantine affected artifacts and apply approved deletion procedure.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Governance owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Verify lineage and retraining or removal evidence.
- Limitation: the proposed diagnostic has not been executed.

### F-25. Unrepresentative test

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Offline score is high but unfamiliar ticket types fail.
- Competing explanations: Historical support data excludes new product categories.
- Discriminating diagnostic: Compare current intake distribution to test coverage.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Restrict eligibility and collect prospective labeled cases.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Evaluation owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Repeat time-forward evaluation on the new category.
- Limitation: the proposed diagnostic has not been executed.

### F-26. Saturated serving queue

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: User latency spikes at peak while model compute remains stable.
- Competing explanations: Admission limit missing or policy dependency slowdown.
- Discriminating diagnostic: Separate queue, generation and policy latency traces.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Shed excess load into explicit manual fallback.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Service owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Load-test burst with user-boundary latency and failure rates.
- Limitation: the proposed diagnostic has not been executed.

### F-27. Rollback incompatibility

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Old model cannot parse the newer policy schema.
- Competing explanations: Uncoordinated schema and bundle release.
- Discriminating diagnostic: Check compatibility matrix and replay prior bundle on retained fixtures.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Disable drafts or restore a jointly compatible service bundle.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Release owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Practice rollback after a schema change in staging.
- Limitation: the proposed diagnostic has not been executed.

### F-28. Misleading cost calculation

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Adapter appears cheap because failed generations are omitted.
- Competing explanations: Denominator counts only successes or annotation is excluded.
- Discriminating diagnostic: Reconcile compute, retries and human review with all eligible tickets.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Report full cost per accepted draft and manual comparator.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Finance owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Compare ledger totals with service and annotation records.
- Limitation: the proposed diagnostic has not been executed.

### F-29. Model memorization

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Generated answer reproduces a private training passage.
- Competing explanations: Overfitting or previously public repeated content.
- Discriminating diagnostic: Use authorized canary and extraction tests with matched controls.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Stop candidate and restrict access while investigating.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Privacy owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Retest after data and training adjustments without exposing canaries.
- Limitation: the proposed diagnostic has not been executed.

### F-30. Overfitted validation

- Evidence status: hypothetical; no incident or run is asserted.
- Trigger and symptom: Each new checkpoint improves validation but test fails.
- Competing explanations: Repeated tuning on validation or genuine time shift.
- Discriminating diagnostic: Audit selection history and compare untouched later cohort.
- Expected signal: the diagnostic distinguishes the listed hypotheses rather than assuming one.
- Immediate containment: Freeze selection and obtain a new independent test.
- Blast radius: restrict affected drafts and return users to the manual workflow.
- Accountable role: Evaluation owner.
- Preventive control: encode the relevant invariant in the release gate.
- Verification: Record candidate budget and evaluate once on locked data.
- Limitation: the proposed diagnostic has not been executed.

## 13. Transferable Lesson

The hypothesis is that small trainable updates can adapt response behavior cheaply,
while an independent, policy-grounded evaluation decides if that adaptation is useful.
The mechanism is selective adjustment of model weights, not a guarantee of correctness.
Preconditions include lawful data access, stable task boundaries, valid labels,
a sufficiently representative future-facing holdout, and an independent baseline.
This case provides arithmetic and a test design, not empirical support for superiority.
Avoid the approach when current authoritative facts cannot be supplied at inference,
when humans cannot review high-consequence drafts, or when leakage cannot be controlled.
The manual process or frozen base may win if they meet quality needs at lower total cost.
Portability question: can another domain define an acceptable output and an independent oracle?
For regulated advice, add domain-specific authorization and professional review;
do not transfer the fictional thresholds or assert that the same policy applies.
For changing product rules, version the evidence source rather than retraining for each fact.
If the intended benefit is lower handling time, measure time in a randomized workflow.
A model-score increase alone cannot establish that business outcome.

## 14. Connections & Cross-References

### 14a. Related cases, theory, and cheat sheets

- [LoRA parameter-efficient fine-tuning](../Cheat_Sheets/01_LoRA_Parameter_Efficient_Finetuning.md) gives related parameter mechanics.
- [Module 21 learning home](../INDEX.md) provides adaptation and alignment context.
- [Evaluation and lifecycle source coverage](../../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-playbook-operations) identifies the distinct evaluation home.
- [Authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) distinguish evidence from illustrative calculations.
- No related production case or organization outcome is asserted.

### 14b. Open questions

| Question | Owner | Decision dependence |
|---|---|---|
| Are customer data rights sufficient for training? | Governance | Determines whether any dataset can be built |
| Which base checkpoint and license are acceptable? | Legal and ML | Determines actual architecture and cost |
| What severe-risk confidence bound is acceptable? | Safety | Determines sample size and release gate |
| Can policy be reliably retrieved at decision time? | Policy owner | Determines whether drafts are eligible |
| Can reviewers identify hidden policy errors? | Operations | Determines reliability of the quality oracle |
| Does the exact quantized export preserve quality? | ML and serving | Determines whether QLoRA may be used |

### 14c. Interview pack

Each answer is a proposed design judgment, not a report of an executed system.

#### Q01. Frame the decision

- Question: What exactly is being decided, and by whom?
- Answer: The support lead decides whether to run an agent-only pilot of an adapted draft model.
- Reasoning, part 1: The relevant input is one authorized ticket and current policy; the output is an editable draft.
- Reasoning, part 2: Manual drafting and the frozen prompted base are distinct comparators.
- Follow-up: What if sending must be autonomous?
- Follow-up answer: It is a different risk and approval decision, not an extension of this gate.
- Acceptable alternative: A no-model decision is defensible when risk or evidence is inadequate.
- Red flag: Claiming the assistant has already improved support.
- Answer support: Section 3 defines the bounded decision.
- Scoring, full credit: accurate answer with the stated boundary and comparator.
- Scoring, partial credit: correct direction but misses an important limitation.
- Scoring, no credit: relies on the red flag or invents a measured outcome.
- Evidence status: illustrative reasoning, not an executed experiment.

#### Q02. Separate LoRA and QLoRA

- Question: How does QLoRA differ from ordinary LoRA?
- Answer: Both train low-rank adapter factors while freezing base weights.
- Reasoning, part 1: QLoRA stores the frozen base in a supported low-bit representation and computes with dequantized values.
- Reasoning, part 2: Quantization scales, other tensors and activations mean four bits per weight is not total memory.
- Follow-up: What if the export changes answers?
- Follow-up answer: Evaluate the precise export and block it if guardrails fail.
- Acceptable alternative: A full-precision LoRA candidate is acceptable when memory allows.
- Red flag: Saying a quantized base makes adapters or all computation four-bit.
- Answer support: Section 5c describes the method and boundaries.
- Scoring, full credit: accurate answer with the stated boundary and comparator.
- Scoring, partial credit: correct direction but misses an important limitation.
- Scoring, no credit: relies on the red flag or invents a measured outcome.
- Evidence status: illustrative reasoning, not an executed experiment.

#### Q03. Calculate adapter count

- Question: How many adapter parameters are in the specified rank-eight example?
- Answer: There are 4,194,304 trainable parameters under the stated geometry.
- Reasoning, part 1: Each matrix uses 8 times (4096 + 4096), or 65,536 parameters.
- Reasoning, part 2: There are 32 layers times two matrices, so 64 times 65,536 gives the result.
- Follow-up: What if rank doubles?
- Follow-up answer: The count doubles to 8,388,608, all else equal.
- Acceptable alternative: Sum each actual projection separately when matrix dimensions differ.
- Red flag: Comparing adapter parameters to the entire model using only target-matrix counts.
- Answer support: Section 6.1 contains the dimensional check.
- Scoring, full credit: accurate answer with the stated boundary and comparator.
- Scoring, partial credit: correct direction but misses an important limitation.
- Scoring, no credit: relies on the red flag or invents a measured outcome.
- Evidence status: illustrative reasoning, not an executed experiment.

#### Q04. Calculate training state

- Question: What memory does the adapter optimizer-state example require?
- Answer: The stated adapter tensors require 48 MiB, excluding all other memory.
- Reasoning, part 1: Four Mi parameters times 12 bytes each equals 48 MiB.
- Reasoning, part 2: The 12 bytes comprise 2-byte weights, 2-byte gradients and two 4-byte moments.
- Follow-up: Does a 48 MiB adapter ensure the run fits?
- Follow-up answer: No: base weights, activations, temporary buffers and fragmentation remain.
- Acceptable alternative: A different optimizer can change state bytes per parameter.
- Red flag: Using the adapter-only calculation as peak GPU allocation.
- Answer support: Section 6.2 states assumptions and exclusions.
- Scoring, full credit: accurate answer with the stated boundary and comparator.
- Scoring, partial credit: correct direction but misses an important limitation.
- Scoring, no credit: relies on the red flag or invents a measured outcome.
- Evidence status: illustrative reasoning, not an executed experiment.

#### Q05. Guard against leakage

- Question: Why not split tickets randomly?
- Answer: Random row splitting can place the same customer, thread or template in train and test.
- Reasoning, part 1: This creates a shortcut that exaggerates generalization to later unseen customer groups.
- Reasoning, part 2: Use time boundaries, group assignment and cross-split duplicate quarantine.
- Follow-up: What happens to later tickets of training customers?
- Follow-up answer: Exclude them from validation and test under this specified unseen-customer estimand.
- Acceptable alternative: A separate returning-customer evaluation can be useful if labeled as a different estimand.
- Red flag: Using future final replies or post-resolution fields as input.
- Answer support: Section 5a states the split and Section 5d the audit.
- Scoring, full credit: accurate answer with the stated boundary and comparator.
- Scoring, partial credit: correct direction but misses an important limitation.
- Scoring, no credit: relies on the red flag or invents a measured outcome.
- Evidence status: illustrative reasoning, not an executed experiment.

#### Q06. Join time-correct policy

- Question: Which policy version belongs with a historical training ticket?
- Answer: Use the authorized policy effective at the response decision time.
- Reasoning, part 1: The effective interval must contain that timestamp and match product and region.
- Reasoning, part 2: Zero or multiple matches are quarantined rather than silently choosing the latest.
- Follow-up: What if current policy is unavailable in serving?
- Follow-up answer: Return a safe manual fallback; do not rely on model memory as current authority.
- Acceptable alternative: A verified immutable policy snapshot can be used when its validity is established.
- Red flag: Joining every historical ticket to today’s policy.
- Answer support: Section 5a gives the interval rule and Section 8 the failure path.
- Scoring, full credit: accurate answer with the stated boundary and comparator.
- Scoring, partial credit: correct direction but misses an important limitation.
- Scoring, no credit: relies on the red flag or invents a measured outcome.
- Evidence status: illustrative reasoning, not an executed experiment.

#### Q07. Set the baseline

- Question: Why compare with both manual and prompted frozen base?
- Answer: The frozen base isolates the incremental value of adaptation under shared context.
- Reasoning, part 1: The manual comparator represents the existing workflow and supports a later business comparison.
- Reasoning, part 2: Neither comparator can be replaced by a score on training examples.
- Follow-up: What if frozen base wins quality and cost?
- Follow-up answer: Keep it or the manual process, subject to the same safety gates.
- Acceptable alternative: A rule-based template is another credible baseline if it covers the task.
- Red flag: Claiming any improvement over manual proves fine-tuning was necessary.
- Answer support: Section 5c lists considered alternatives and Section 9 defines comparison.
- Scoring, full credit: accurate answer with the stated boundary and comparator.
- Scoring, partial credit: correct direction but misses an important limitation.
- Scoring, no credit: relies on the red flag or invents a measured outcome.
- Evidence status: illustrative reasoning, not an executed experiment.

#### Q08. Design evaluation

- Question: What must the primary quality metric count?
- Answer: Count acceptable drafts over all eligible assigned tickets, including failures.
- Reasoning, part 1: Independent blinded reviewers use a fixed rubric and the matching policy version.
- Reasoning, part 2: Report the paired difference versus the frozen base and cluster-aware uncertainty.
- Follow-up: What if reviewers disagree on a severe violation?
- Follow-up answer: Preserve individual scores and send disagreement to qualified adjudication.
- Acceptable alternative: A calibrated automated judge can triage but is not assumed to replace humans.
- Red flag: Dropping timeouts or unhelpful refusals from the denominator.
- Answer support: Section 9 defines numerator, denominator, window and guardrails.
- Scoring, full credit: accurate answer with the stated boundary and comparator.
- Scoring, partial credit: correct direction but misses an important limitation.
- Scoring, no credit: relies on the red flag or invents a measured outcome.
- Evidence status: illustrative reasoning, not an executed experiment.

#### Q09. Interpret an offline improvement

- Question: If acceptable rate rises from 70% to 75%, what is the change?
- Answer: The absolute difference is five percentage points, under identical denominators.
- Reasoning, part 1: The relative change is 5 divided by 70, approximately 7.14%.
- Reasoning, part 2: This hypothetical arithmetic alone establishes neither significance nor business value.
- Follow-up: What if the lower confidence bound crosses zero?
- Follow-up answer: Under the proposed gate the evidence is inconclusive and pilot authorization defers.
- Acceptable alternative: Report an interval and the paired customer-cluster analysis rather than a point alone.
- Red flag: Calling a five-point change a five-percent relative increase.
- Answer support: Section 9 explains the locked comparison; the numbers here are illustrative only.
- Scoring, full credit: accurate answer with the stated boundary and comparator.
- Scoring, partial credit: correct direction but misses an important limitation.
- Scoring, no credit: relies on the red flag or invents a measured outcome.
- Evidence status: illustrative reasoning, not an executed experiment.

#### Q10. Reason about safety

- Question: Are zero severe incidents in a small sample proof of safety?
- Answer: No: rare risks can remain substantial even when none are observed.
- Reasoning, part 1: A prespecified upper risk bound and adequate sample are needed, with representative coverage.
- Reasoning, part 2: Adversarial stress-set frequency must not be mistaken for population incidence.
- Follow-up: What if the safety sample is too small?
- Follow-up answer: Defer or restrict scope until the agreed risk criterion can be evaluated.
- Acceptable alternative: Use targeted tests to reveal mechanisms alongside a separate prevalence sample.
- Red flag: Declaring safety proven because a small test has no observed violation.
- Answer support: Section 9 defines the safety gate and limitations.
- Scoring, full credit: accurate answer with the stated boundary and comparator.
- Scoring, partial credit: correct direction but misses an important limitation.
- Scoring, no credit: relies on the red flag or invents a measured outcome.
- Evidence status: illustrative reasoning, not an executed experiment.

#### Q11. Diagnose a flat loss

- Question: The adapter loss is flat. What do you do first?
- Answer: Inspect trainable parameter selection, gradients and optimizer membership before changing rank.
- Reasoning, part 1: Alternative causes include label masking, inadequate data signal and an inappropriate learning rate.
- Reasoning, part 2: A controlled tiny batch and decoded label inspection distinguish implementation from data causes.
- Follow-up: What if the base tensors changed?
- Follow-up answer: Stop the run, freeze the intended base and recreate optimizer state.
- Acceptable alternative: Use a one-batch overfit probe when safe and authorized.
- Red flag: Immediately raising rank without checking that any adapter gradients exist.
- Answer support: Failure entries F-04 through F-07 identify targeted tests.
- Scoring, full credit: accurate answer with the stated boundary and comparator.
- Scoring, partial credit: correct direction but misses an important limitation.
- Scoring, no credit: relies on the red flag or invents a measured outcome.
- Evidence status: illustrative reasoning, not an executed experiment.

#### Q12. Calculate evaluation cost

- Question: How many generations and units for the planned validation exercise?
- Answer: Three checkpoints times 500 prompts times two seeds equals 3,000 generations.
- Reasoning, part 1: At 0.02 fictional currency units each, generation-only cost is 60 units.
- Reasoning, part 2: The figure excludes baseline, annotation, retries and other infrastructure.
- Follow-up: What if two more seeds are added?
- Follow-up answer: Four seeds double generation count to 6,000 and the simplified cost to 120 units.
- Acceptable alternative: Token-based pricing is preferable when actual prices and lengths are known.
- Red flag: Calling the fictional unit price a current vendor quotation.
- Answer support: Section 6.3 contains the worked calculation and scope.
- Scoring, full credit: accurate answer with the stated boundary and comparator.
- Scoring, partial credit: correct direction but misses an important limitation.
- Scoring, no credit: relies on the red flag or invents a measured outcome.
- Evidence status: illustrative reasoning, not an executed experiment.

#### Q13. Explain deployment gate

- Question: What artifacts must move together into serving?
- Answer: Bind the base, adapter, tokenizer, quantization, prompt, policy interface and runtime.
- Reasoning, part 1: The evaluation report and signed manifest must reference the exact same bundle.
- Reasoning, part 2: A mismatch blocks promotion because observed offline behavior may not match serving.
- Follow-up: What if an old bundle cannot read the new policy schema?
- Follow-up answer: Disable drafts or restore a compatible paired bundle and policy service.
- Acceptable alternative: A compatible frozen base is a fallback only if it has separately passed gates.
- Red flag: Swapping only the adapter and assuming all other components are interchangeable.
- Answer support: Section 8 defines the bundle, rollback and fallback.
- Scoring, full credit: accurate answer with the stated boundary and comparator.
- Scoring, partial credit: correct direction but misses an important limitation.
- Scoring, no credit: relies on the red flag or invents a measured outcome.
- Evidence status: illustrative reasoning, not an executed experiment.

#### Q14. Change the hardware constraint

- Question: What if the base does not fit in available accelerator memory?
- Answer: Profile actual memory, then consider a compatible QLoRA configuration or smaller base.
- Reasoning, part 1: The raw four-bit weight estimate excludes activations, metadata and temporary buffers.
- Reasoning, part 2: Changing the representation creates a new candidate needing independent quality and safety tests.
- Follow-up: What if quantized inference fails the guardrails?
- Follow-up answer: Do not deploy it; choose a validated alternative or retain manual drafting.
- Acceptable alternative: Sequence-length reduction is possible if it preserves required policy context.
- Red flag: Assuming nominal four-bit compression guarantees both fit and equivalent behavior.
- Answer support: Section 6.2 and failure entries F-08 through F-10 address this tradeoff.
- Scoring, full credit: accurate answer with the stated boundary and comparator.
- Scoring, partial credit: correct direction but misses an important limitation.
- Scoring, no credit: relies on the red flag or invents a measured outcome.
- Evidence status: illustrative reasoning, not an executed experiment.

#### Q15. Design a causal pilot

- Question: How can the team estimate agent-workflow impact?
- Answer: Randomize eligible agents to access or control and analyze all assigned eligible tickets.
- Reasoning, part 1: Agent-level clusters reduce contamination but require cluster-aware inference.
- Reasoning, part 2: Measure handling time, adoption, errors and customer impact under a prespecified window.
- Follow-up: What if control agents copy treatment drafts?
- Follow-up answer: Investigate interference and consider a wider randomization unit before restarting.
- Acceptable alternative: A carefully designed stepped rollout may be possible but needs separate assumptions.
- Red flag: Comparing only agents who chose to use drafts against everyone else.
- Answer support: Section 9 distinguishes intention-to-treat from usage analysis.
- Scoring, full credit: accurate answer with the stated boundary and comparator.
- Scoring, partial credit: correct direction but misses an important limitation.
- Scoring, no credit: relies on the red flag or invents a measured outcome.
- Evidence status: illustrative reasoning, not an executed experiment.

#### Q16. Size serving demand

- Question: What is the assumed peak arrival rate and in-service concurrency?
- Answer: At 12,000 requests over eight hours the average is approximately 0.4167 requests per second.
- Reasoning, part 1: A fivefold peak is approximately 2.083 requests per second.
- Reasoning, part 2: At a three-second mean service time Little’s law gives approximately 6.25 in service.
- Follow-up: Does this guarantee p95 latency?
- Follow-up answer: No: queueing, bursts, service-time distribution and failures require measurement.
- Acceptable alternative: Use a representative load test and admission control with a manual fallback.
- Red flag: Adding component p95 values and labeling their sum end-to-end p95.
- Answer support: Section 8d specifies the assumptions and sensitivity.
- Scoring, full credit: accurate answer with the stated boundary and comparator.
- Scoring, partial credit: correct direction but misses an important limitation.
- Scoring, no credit: relies on the red flag or invents a measured outcome.
- Evidence status: illustrative reasoning, not an executed experiment.

## 15. Quick Reference Card

| Field | Decision-ready summary |
|---|---|
| Domain | Customer support language-model adaptation. |
| Problem type | Decide whether an agent-only support drafting assistant should use an adapter; authorized ticket and policy excerpt produce an editable draft, never automatic send. |
| Decision and baseline | Compare proposed LoRA with frozen prompted base and manual drafting |
| Method | Time-correct grouped split, supervised response-only loss, independent evaluation |
| Primary result | Not measured: no pilot decision is supported until quality, safety and operational evidence exists. |
| Numerical anchor | Rank-eight adapters: 4,194,304 parameters under Section 6 geometry |
| Failure boundary | Stale policy, leakage, unsafe outputs or mismatched release bundle block promotion |
| Evidence and stage | Illustrative sizing; outcome not measured; proposed, not deployed |
| Review | Draft, unreviewed; recheck after model, policy, rights or rubric changes |
| Key lesson | A domain adapter does not replace current policy, independent evaluation or human review. |
| Tags | `evaluation-mismatch`; `data-leakage`; `atomic-release-bundle` |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-2026-FT01 / 1.0 / 2026-11-07 |
| Evidence level | Illustrative sizing; outcome not measured |
| Implementation stage | Proposed; no deployed pilot |
| Baseline / primary metric | Frozen prompted base and manual drafting; independent quality and safety gates to be defined |
| Key limitation | No actual training run, evaluated dataset or deployment |
| Review trigger / owner | Policy, model, data rights, rubric or serving stack change / support operations lead |
