---
lesson_id: "FT-M00-02"
lesson_title: "Modern LLM Training Pipeline"
version: "1.0-draft"
created: "2026-11-07"
reviewed: null
status: "draft"
evidence_basis: "primary papers, documented principles, and constructed examples"
owner_role: "curriculum maintainer"
review_trigger: "Material change to training stages, data governance, artifact contracts, or deployment validation"
---

# 0.2 — Modern LLM Training Pipeline

> **Course location:** Module 19, Transformers, LLMs and Foundation-Model Internals. This lesson maps to `finetuning_course/Module_00_Foundations/0.2_Modern_LLM_Training_Pipeline.md` in [`Train/Templates/directory.txt`](../../../Templates/directory.txt). The original source lesson is not present for inspection; this is a new course draft based on visible scope and cited public sources.
>
> **Status:** Draft. No training run, data license, benchmark, deployment, or model result has been verified by this lesson.

## Learning outcomes

Learners can:

- Draw the main data-to-training-to-evaluation-to-release flow and identify decision gates.
- Distinguish pretraining, continued pretraining, supervised fine-tuning, preference optimization, evaluation, and serving.
- Track which data, code, tokenizer, base checkpoint, adapter, and evaluation artifacts belong to each run.
- Calculate simple token, batch, and storage quantities with explicit units and assumptions.
- Explain why held-out validation and test sets must remain isolated from training and repeated tuning.
- Compare a training intervention with prompt-only and retrieval baselines before spending on weight updates.
- Specify a reproducible run and a release gate that includes quality, safety, privacy, latency, and cost.

## 1. Pipeline overview

A training pipeline converts permitted data and a declared objective into a versioned model artifact, then establishes whether the artifact is acceptable for a particular use. It is not a single command or a guarantee that the output model will be fit for deployment.

A minimal conceptual flow is:

```text
Problem and acceptance criteria
        ↓
Data rights, provenance, and collection
        ↓
Quality checks, deduplication, filtering, and split assignment
        ↓
Tokenizer/config and example formatting
        ↓
Training objective + initialization + optimization
        ↓
Checkpointing and training diagnostics
        ↓
Held-out evaluation, safety/privacy review, regression tests
        ↓
Release decision and immutable artifact manifest
        ↓
Serving integration, staged rollout, monitoring, rollback
```

Each arrow is an interface contract. For example, the data preparation stage must say what an example is, how duplicates are handled, what text is masked from loss, and how the split is assigned. The training stage must record the base checkpoint and exact configuration. Evaluation must operate on an uncontaminated held-out set with a documented scoring procedure. Release must bind the tested model and runtime to the shipped artifact.

No universal pipeline has exactly these stages. A small adapter run may omit pretraining. A pretrained model may be evaluated and deployed without additional gradient updates. Retrieval, prompting, or a conventional software fix may be more appropriate than training. The flow is a control model for making those choices explicit.

## 2. Define the decision before building the pipeline

Start with the operational failure, not a preferred training method. State the user, task, decision owner, acceptable errors, allowed data, baseline system, response-time budget, and evaluation population. Convert vague goals such as “better answers” into measurable claims.

A useful acceptance statement names:

- The input distribution and relevant user or task slices.
- The target behavior, including abstention or escalation where appropriate.
- A baseline such as the current product, base model with a fixed prompt, or retrieval-only system.
- A primary metric and guardrails, each with unit, denominator, direction, and observation window.
- An uncertainty or minimum-evidence requirement.
- A deployment boundary: internal review, shadow mode, limited canary, or broader release.
- Conditions that stop training or trigger rollback.

If the failure is stale information, retrieval and source versioning are likely candidates. If the failure is a malformed schema, constrained decoding or a parser may be sufficient. If the issue is behavior learned from demonstrations, SFT may be relevant. The stage is chosen only after the target defect has been characterized.

## 3. Data governance and provenance

Training data is not merely a collection of strings. It carries rights, origin, sensitivity, collection context, quality, potential contamination, and retention obligations. A dataset manifest should identify source and acquisition date, permitted purpose, exclusions, transformation code, deduplication policy, label origin, license/consent review status, and deletion or retention conditions.

Authorization to read data does not automatically establish permission to train on it. A public URL does not establish that content is lawful, licensed for the intended use, or free of personal or confidential information. The responsible organization must review its actual rights and applicable rules; this lesson is not legal advice.

Data minimization reduces unnecessary exposure. Remove or transform sensitive information only under a documented method, then test whether sensitive content can still be memorized or extracted. Hashing identifiers may not anonymize them when the input space is small or linkable.

### Provenance fields

| Field | Purpose | Example of a useful value |
|---|---|---|
| Dataset ID and immutable version | Identifies exact corpus state | Content-addressed manifest or versioned snapshot |
| Source family and collection timestamp | Records origin and freshness | Approved internal policy export, captured UTC time |
| Permitted purpose and restrictions | Prevents reuse outside authorization | Training permitted for specified internal support scope |
| Transform code revision | Reconstructs filtering and formatting | Git commit plus dependency lock |
| Example key and dedup group | Enables audit and duplicate analysis | Stable privacy-safe identifiers |
| Split assignment and rationale | Prevents leakage and explains evaluation isolation | Entity-grouped and time-aware assignment |
| Sensitive-data review | Makes review state explicit | Reviewer, scope, date, unresolved risks |
| Retention/deletion linkage | Supports removal obligations | Source-to-derived artifact dependency mapping |

A dataset manifest is a lineage record, not proof that every source was valid. Keep unresolved questions visible and block use where the required authorization is missing.

## 4. Quality, filtering, deduplication, and splits

Data curation changes the distribution a model sees. Filtering can remove noise but also remove rare dialects, minority groups, code, or difficult examples. Deduplication may reduce repeated text and benchmark contamination, but approximate matching can incorrectly collapse distinct examples. Every transformation requires a measurable rationale and a retained audit trail.

A sound split is chosen before iterative tuning. Random row splits can leak near-duplicate documents, users, sessions, or future information across partitions. Group by the entity whose leakage would undermine the intended evaluation; use time-aware splits when deployment predicts future data. If one document yields many chunks, assign the parent document to one split before chunking.

Training, validation, and test roles differ:

- **Training set:** contributes gradients or training targets.
- **Validation set:** supports model selection, early stopping, and hyperparameter decisions; repeated use can overfit decisions to it.
- **Test set:** estimates performance after choices are fixed; repeated inspection or tuning turns it into another validation set.

A sealed test set does not solve all distribution mismatch. Describe how representative it is and when it was collected. Monitor deployment data under appropriate privacy and access controls; do not assume a static test remains sufficient as products, users, or policies change.

### Duplicate sensitivity example

Illustrative corpus: 100,000 records contain 4,000 exact duplicates and an additional estimated 3,000 near-duplicate records. If exact duplicates are removed first, 96,000 records remain. If the near-duplicate detector then removes 1,800 records that meet a predeclared threshold, 94,200 remain. This arithmetic says nothing about whether the detector threshold is appropriate. Review false merges and retained rare content on a stratified sample before using the transformed corpus.

## 5. Tokenization and training-example construction

A tokenizer maps text and control markers into token IDs. The exact tokenizer, vocabulary, normalization, special tokens, and chat template form part of the model contract. Changing them can alter sequence length, input semantics, output decoding, and compatibility with embeddings or output layers.

Before training, test representative examples through the full encode/decode path. Inspect Unicode, whitespace, code, multilingual text, long inputs, empty inputs, role boundaries, tool calls, and end markers. Count tokens using the actual tokenizer rather than a word-count proxy when memory, context length, or billing depends on tokens.

Sequence packing concatenates examples to improve utilization. It must preserve the intended attention and loss boundaries: one example should not accidentally attend to another if isolation is required, and padding or prompt tokens should receive the intended loss mask. Truncation must not silently remove the target or critical context. Keep raw-to-formatted lineage sufficient to audit a sample without exposing restricted data unnecessarily.

For a next-token objective, input and target positions must be shifted consistently. For supervised instruction data, specify whether the model is trained on the user message, assistant answer, tool messages, or a selected subset. A wrong mask can optimize the wrong content even when the loss decreases smoothly.

## 6. Objectives and training stages

### Pretraining

Pretraining commonly optimizes a self-supervised objective over a large corpus. In an autoregressive setup, the model predicts the next token from a preceding prefix. The objective supplies a learning signal without requiring a human label for every next token. It does not directly teach a particular application's policies or prove truthfulness.

### Continued pretraining

Continued pretraining continues a language-model objective from an existing checkpoint on additional data. It may adapt terminology or domain distributions. It can also shift general behavior, increase memorization risk, or forget capabilities when data and optimization are imbalanced. Compare with the original checkpoint and preserve broad regression suites.

### Supervised fine-tuning

SFT trains on examples of desired outputs conditioned on inputs. Demonstrations encode response patterns, but their quality and coverage matter. A narrow or noisy demonstration set may improve formatting while harming other tasks. Train/validation splits must respect user, source, time, and near-duplicate dependencies.

### Preference optimization

Preference methods use comparisons, rankings, or preference-derived objectives to change model behavior. The result depends on preference collection, annotator instructions, comparison coverage, and the algorithm's assumptions. A preference signal is not automatically factual ground truth or a safety proof.

### Distillation and adapters

Distillation trains a student using teacher outputs, probabilities, or labels. It transfers selected behavior, not an abstract guarantee of teacher equivalence. Adapters add or modify a subset of parameters while retaining a base checkpoint; adapter identity and compatible base revision must be bound at load time. Merging weights is a distinct operation that can change compatibility and numerical behavior.

These stage definitions are deliberately high-level. Separate lessons are required for objective-specific derivations, optimizer state, parallel training, reward modeling, DPO/GRPO, PEFT, and distillation.

## 7. Illustrative throughput and token accounting

Use one explicit training-step example to show why batch and token units matter. Assume, purely illustratively, 8 devices, 4 sequences per device per microbatch, 2,048 non-padding tokens per sequence, and 8 accumulation steps. If each microbatch has the stated full token count, then:

```text
FORMULA [PIPE-01]: tokens_per_optimizer_update = devices * sequences_per_device * tokens_per_sequence * accumulation_steps
WHERE: devices [count]; sequences_per_device [sequences/device/microbatch];
       tokens_per_sequence [tokens/sequence]; accumulation_steps [microbatches/update].
CONSTRAINT: full unmasked sequence lengths are assumed; padding and ignored labels make the number of scored tokens smaller.
NUMERICAL EXAMPLE: 8 * 4 * 2048 * 8 = 524,288 input token positions per optimizer update.
INTERPRETATION: this is an input-position count under the assumptions, not automatically 524,288 loss-bearing targets or a hardware throughput measurement.
```

If a measured training rate were 50,000 scored tokens/s, then 1,000,000,000 scored tokens would require `1,000,000,000 / 50,000 = 20,000` seconds, or 5.56 hours of idealized active time. This ignores evaluation, checkpointing, data stalls, restart, warmup, validation, cluster contention, and tail effects. Do not present it as wall-clock capacity without measuring the full run.

Distinguish input tokens, non-padding tokens, loss-bearing targets, examples, optimizer updates, and packed sequence slots. A dataset count expressed as “one billion tokens” is incomplete unless the tokenizer and counting rule are known.

## 8. Optimization, checkpoints, and diagnostics

An optimizer update changes parameters according to gradients and optimizer state. Gradient accumulation combines contributions from multiple microbatches before an update; distributed training may further aggregate gradients across devices. Effective batch size depends on the number of devices, per-device examples, accumulation, and masking. If examples have variable lengths, example count can hide a large variation in processed token count.

Training diagnostics should include loss curves, learning rate, gradient norms, skipped/overflowed updates where relevant, data throughput, device utilization, memory peaks, checkpoint status, and validation metrics. A smoothly falling loss is not sufficient: a model can memorize, diverge on specific slices, or optimize an unintended target.

Save checkpoints at recoverable intervals. A checkpoint may require model weights, optimizer and scheduler state, random-number-generator state, data sampler position, training step, configuration, and code/data identifiers for faithful continuation. A weights-only export may be sufficient for inference but insufficient to resume the original training trajectory.

A checkpoint is not released merely because a file exists. Validate its integrity, record a checksum, load it in a clean evaluation environment, and ensure that its tokenizer, architecture configuration, adapters, and inference code match.

## 9. Evaluation and regression gates

Evaluation must correspond to the release claim. Evaluate against a baseline with identical prompts and decoding where possible. Separate an offline model metric from online product outcomes. For generative outputs, combine automated measures with human review when the task requires semantic judgment; document reviewer instructions, disagreement handling, and adjudication.

A release suite should include:

- Core task performance on a held-out dataset with a documented reference standard.
- General capability regressions relevant to the product's users.
- Safety and refusal tests against expected misuse and prompt-injection patterns.
- Privacy probes for memorization or unauthorized disclosure, scoped to the data and threat model.
- Robustness slices for language, format, length, rare classes, and relevant user groups.
- Structured-output validity and tool-argument correctness where applicable.
- Latency, peak memory, concurrency, cost, and availability under an explicit workload.
- Human escalation and failure-recovery paths.

Predeclare acceptance thresholds and who can approve exceptions. If a proposed model passes average performance but violates a critical slice or privacy gate, the average does not cancel that failure. Store evaluation code and dataset versions with the artifact.

## 10. Release, serving, and monitoring

A production release binds model weights to tokenizer, templates, runtime libraries, decoding settings, retrieval indexes, tools, and application code. Any mismatch can change behavior. Serve a staged rollout: offline gate, shadow or replay test where appropriate, constrained canary, monitored expansion, and rollback path. The chosen stages depend on risk and architecture; the labels are not guarantees.

Define observability without logging sensitive prompts by default. Useful telemetry may include version identifiers, latency distribution, token counts, errors, escalation outcomes, and privacy-preserving quality signals. Access to raw request data should be justified, minimized, restricted, and retained only under an approved policy. Redact secrets and identifiers before diagnostic storage where feasible, and assess re-identification risk.

Monitoring detects change; it does not automatically identify the cause. A rising refusal rate may reflect a model, prompt, policy, retrieval, user, or parser change. Associate alerts with a runbook that identifies candidate causes, evidence to inspect, safe fallback, escalation owner, and rollback conditions.

## 11. Pipeline failure diagnostics

| Symptom | Competing causes | Discriminating check | Safe response | Verify |
|---|---|---|---|---|
| Training loss falls but held-out task score falls | Split leakage in tuning; objective mismatch; label/mask shift; overfitting | Audit masks/splits; compare per-slice curves and unchanged baseline | Stop promotion; restore last accepted artifact; investigate data lineage | Re-run frozen suite with corrected split and target mask |
| Loss becomes NaN or spikes | Learning rate; corrupted batch; overflow; invalid labels; optimizer-state corruption | Locate first bad update; replay that batch; inspect gradients and scaler state | Stop run and preserve logs/checkpoint; fix identified cause, not merely hide NaNs | Deterministic short replay plus finite-value assertions |
| Evaluation score appears implausibly high | Training overlap; near-duplicate leakage; prompt reveals answer; scoring bug | Search overlap, inspect examples, test alternative scoring and sealed set | Suspend claim; quarantine contaminated test set | Fresh independently governed evaluation set |
| Adapter loads but behavior is wrong | Base revision mismatch; target modules differ; tokenizer/template mismatch | Compare artifact manifest and activation/output smoke vectors | Disable incompatible adapter and route to known-good base | Compatibility test and held-out regression before re-enable |
| GPU memory fails at optimizer step | Lazy optimizer allocation; longer sequence; gradient state; allocator fragmentation | Measure synchronized peak by phase; compare sequence/token distribution | Reduce approved workload or choose a measured strategy; keep quality gates | Repeat measured run with identical model/data and report throughput |
| Training data removal request cannot be honored | Missing lineage; transformed data not traceable; checkpoints not mapped | Trace source ID through transformations, shards and checkpoints | Follow approved governance response; block unverified reuse | Demonstrate deletion/retraining plan with auditable lineage |
| Production quality drops after rollout | New checkpoint; prompt/retriever/runtime; distribution shift; incident | Segment by deployed versions and request path; compare known-good routing | Roll back or contain exposure according to runbook | Verify metrics recover and preserve incident evidence |

A plausible mitigation is not evidence that the root cause was fixed. Every diagnosis names both a hypothesis and a test that could disconfirm it.

## 12. Final decision card

| Decision field | Minimum record |
|---|---|
| Training problem | Observable defect, user, task, and baseline |
| Data permission | Source, purpose, restrictions, reviewer, unresolved gaps |
| Dataset version | Immutable manifest, transforms, split policy, tokenizer |
| Training objective | Stage, target, masking, initialization, optimization config |
| Resource estimate | Token count, batch assumptions, device-hours, storage and recovery overhead |
| Evaluation | Held-out protocol, metric definitions, slices, uncertainty, guardrails |
| Artifact lineage | Base/checkpoint/adapter hashes, code, configuration, dependencies |
| Release gate | Named approver, thresholds, staged rollout, safe fallback and rollback |
| Monitoring | Privacy-aware signals, alert thresholds, ownership and review cadence |
| Evidence status | Draft; illustrative calculations only; no training or deployment result measured |

## Review questions

1. **Why define the task before selecting an objective?** Because the training loss may not represent the user-visible failure; the objective should address a measured gap.
2. **What is the difference between validation and test data?** Validation informs decisions; test estimates performance after decisions are fixed. Repeated test-driven tuning contaminates the test role.
3. **Why group by parent document before chunking?** Otherwise sibling chunks may cross splits and leak content.
4. **What does token count mean?** It is a count under a specific tokenizer and inclusion rule; input positions and loss-bearing targets can differ.
5. **Why keep optimizer state in a training checkpoint?** It may be required to resume optimizer dynamics faithfully; a weights-only export is not equivalent.
6. **Does continued pretraining guarantee domain accuracy?** No; it changes parameterization under an objective and needs task-specific held-out evaluation.
7. **When is retrieval preferable to another training run?** When the gap concerns current or traceable external evidence, assuming authorized access and suitable permission controls.
8. **How can a run be reproducible?** Record immutable data/code/model/configuration, RNG and sampler state where needed, environment, and evaluation protocol.
9. **What does a successful canary establish?** Evidence for the tested population, duration, and workload only; it does not establish all future behavior.
10. **Why preserve a prompt-only baseline?** It isolates whether changing weights adds value beyond input design and provides a low-cost comparator.
11. **What is the danger of training on user logs?** Rights, consent, sensitive information, feedback selection, leakage, and shifting data distributions require review.
12. **Why not release the checkpoint with the best training loss?** Selection should match held-out task and safety evidence, not training fit alone.

## Sources and boundaries

- Vaswani et al., [“Attention Is All You Need”](https://arxiv.org/abs/1706.03762), 2017: transformer architecture and sequence modeling context.
- Brown et al., [“Language Models are Few-Shot Learners”](https://arxiv.org/abs/2005.14165), 2020: GPT-3 autoregressive pretraining and few-shot evaluation example; it does not establish a universal training pipeline.
- Ouyang et al., [“Training language models to follow instructions with human feedback”](https://arxiv.org/abs/2203.02155), 2022: one documented instruction-tuning and human-feedback pipeline; not a claim that every model uses this sequence.
- [Fine-tuning Module 00 source-coverage plan](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-foundations): maps source titles, exercise, learning outcome and validation boundaries.
- [`Train/Templates/directory.txt`](../../../Templates/directory.txt), lines 857–859: confirms the visible Module 00 filenames only; original lesson contents are unavailable.

This is an instructional overview, not a complete cookbook for any training framework, model family, accelerator, data governance regime, or deployment. Revalidate implementation details against versioned primary documentation before executing a run.
