# Module 21 Cheat Sheets — Fine-Tuning, Alignment and Model Customisation

## 1. One-paragraph overview

This collection helps ML practitioners select, calculate, evaluate and operate model-customisation methods under memory, quality and deployment constraints. It currently contains three **draft, unreviewed** sheets on LoRA, quantized-base adapters, and RoPE position interpolation; other topics remain planned. Readers should know matrix multiplication, gradient descent, transformer projections and token-level loss. Readiness means independently performing the scoped calculations, identifying their limits, designing an uncontaminated comparison, and diagnosing a failure; reading the cards alone is not sufficient. Follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master.

## 2. Topic ladder

| Level | File | Outcome | Prerequisites | Key skill | Status |
|---|---|---|---|---|---|
| 2 — Core | [01 LoRA parameter-efficient fine-tuning](01_LoRA_Parameter_Efficient_Finetuning.md) | Count adapter parameters and reason about deployment. | Matrix multiplication, gradients, transformer projections. | Low-rank update and target-module accounting. | Draft; not reviewed. |
| 3 — Production | [02 QLoRA and quantized-base adapters](02_QLoRA_Quantized_Base_Adapter_Finetuning.md) | Estimate training memory and design a controlled evaluation. | Sheet 01; precision, loss and bytes/bits. | Quantized storage versus complete runtime footprint. | Draft; not reviewed. |
| 3 — Production | [03 RoPE and position interpolation](03_RoPE_and_Position_Interpolation.md) | Derive relative rotary scores and design a measured long-context deployment. | Transformer attention, rotations, token positions, memory units. | Relative rotations, position scaling and cache accounting. | Draft; not reviewed. |
| Planned | Customisation decision tree | Select among customisation alternatives under quality and resource constraints. | Sheets 01–02. | Decision tradeoffs and disqualifiers. | Planned; not written. |
| Planned | Dataset quality rubric | Identify leakage, label and governance defects. | Train/validation/test splitting. | Split and provenance audit. | Planned; not written. |
| Planned | Post-training runbook | Design a reproducible train, evaluate and release workflow. | Sheets 01–02 and operational ML. | Recovery and rollout controls. | Planned; not written. |
| Planned | Common Issues | Diagnose at least 25 distinct issues with documented provenance. | Available sheets. | Symptom-to-evidence-to-action reasoning. | Planned; not written. |

A source topic's appearance in the directory tree does not imply an authored or reviewed sheet.

## 3. Quick topic lookup

| Question / keyword | File and exact section | Use condition | Warning |
|---|---|---|---|
| How do I calculate LoRA parameters? | [01 — core formulas](01_LoRA_Parameter_Efficient_Finetuning.md#4-core-formulas-with-full-derivation) | Known projection shapes and rank. | Include every targeted matrix and other trainable module. |
| What is the full quantized training footprint? | [02 — core formulas](02_QLoRA_Quantized_Base_Adapter_Finetuning.md#4-core-formulas-with-full-derivation) | Disjoint concurrent allocation estimates exist. | Four-bit codes omit scales, gradients, activations and overhead. |
| Why did training fail or regress? | [02 — failures](02_QLoRA_Quantized_Base_Adapter_Finetuning.md#8-failure-modes--edge-cases) | A measurable symptom and environment are available. | Hypothetical playbooks are not observed incidents. |
| How do I compare QLoRA with alternatives? | [02 — selection](02_QLoRA_Quantized_Base_Adapter_Finetuning.md#7-comparison--selection-table) | Matched task, data and evaluation exist. | No universal equivalence or superiority is established. |
| What is the minimal formula reminder? | [02 — card](02_QLoRA_Quantized_Base_Adapter_Finetuning.md#12-quick-reference-card) | Full assumptions are understood. | The card does not replace the derivation or a real fit test. |
| How does RoPE encode relative position and how does interpolation change it? | [03 — core formulas](03_RoPE_and_Position_Interpolation.md#4-core-formulas-with-full-derivation) | Matching rotary conventions and position policies. | A supported input length does not establish long-context quality. |
| How do I diagnose cache mismatches or size long-context inference? | [03 — failures](03_RoPE_and_Position_Interpolation.md#8-failure-modes--edge-cases) and [complexity](03_RoPE_and_Position_Interpolation.md#9-complexity--cost-analysis) | Known model, workload and cache dimensions. | Payload estimates exclude workspace, weights and allocator overhead. |

## 4. Reading paths — three options

- **Beginner:** review matrix multiplication and gradient descent externally, then [01](01_LoRA_Parameter_Efficient_Finetuning.md), then the concept map and definitions in [02](02_QLoRA_Quantized_Base_Adapter_Finetuning.md). Objective: explain a frozen base and trainable low-rank update.
- **Practitioner:** read [01](01_LoRA_Parameter_Efficient_Finetuning.md), then [02](02_QLoRA_Quantized_Base_Adapter_Finetuning.md) through selection, failure analysis, sizing and card. Objective: prepare and evaluate a scoped run.
- **Advanced:** read [01](01_LoRA_Parameter_Efficient_Finetuning.md), [02](02_QLoRA_Quantized_Base_Adapter_Finetuning.md), and [03](03_RoPE_and_Position_Interpolation.md), challenge the assumptions, and reproduce the arithmetic independently. Objective: defend quality, memory, recovery and rollout decisions. Sheet 03 is independently readable after the transformer and rotation prerequisites.

Paths are shorter than the template's default file counts because only three sheets exist; practitioners working on context extension can add [03](03_RoPE_and_Position_Interpolation.md) after reviewing its explicit prerequisites.

## 5. Prerequisite skills

- **Math/theory:** matrix dimensions, rank, matrix products, derivatives, loss functions and unit conversion; [MIT linear algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) is an external refresher.
- **ML/domain:** transformer linear projections, causal language-model supervision, held-out evaluation, leakage and adaptation. [LoRA paper](https://arxiv.org/abs/2106.09685) provides a primary method reference.
- **Systems/tools:** device and host memory, optimizer states, reproducible Python environments and model checkpointing. [PEFT 0.18.0 quantization guide](https://huggingface.co/docs/peft/v0.18.0/en/developer_guides/quantization) documents its specific implementation sequence.
- **Long-context math and systems:** vector rotations, transformer attention, causal masks, token indices and cache sizing; [RoFormer](https://arxiv.org/abs/2104.09864) is the primary mechanism reference.

## 6. What this collection does NOT cover

- A full supervised fine-tuning curriculum, RLHF or safety certification; these are not established by the available sheets.
- The full taxonomy of quantization-aware training, every quantizer, or a guarantee that QLoRA and QAT are identical.
- Verified benchmarks for a particular business workload, accelerator, vendor, model, pricing plan or serving environment.
- A finished collection-wide troubleshooting artifact; it remains planned, not a completed handoff.

## 7. Master symbol glossary

| Symbol / term | Meaning | Units / shape | Domain and overload warning | First appearance |
|---|---|---|---|---|
| W | Base linear weight | m-by-n matrix | Frozen in adapter-only examples; quantized representation may differ from W. | [02 definitions](02_QLoRA_Quantized_Base_Adapter_Finetuning.md#3-foundational-definitions) |
| A, B | Low-rank adapter factors | r-by-n and m-by-r | Conventional factor orientation; other libraries may name factors differently. | [02 definitions](02_QLoRA_Quantized_Base_Adapter_Finetuning.md#3-foundational-definitions) |
| r | Adapter rank / inner dimension | Positive integer | Rank bound on update, not necessarily effective base weight rank. | [02 definitions](02_QLoRA_Quantized_Base_Adapter_Finetuning.md#3-foundational-definitions) |
| alpha | Conventional adapter scaling parameter | Dimensionless scalar | alpha/r is not the only possible LoRA scaling convention. | [02 definitions](02_QLoRA_Quantized_Base_Adapter_Finetuning.md#3-foundational-definitions) |
| Q, D | Quantization and reconstruction maps | Encoded object and decoded matrix | Quantization metadata is essential to decoding. | [02 definitions](02_QLoRA_Quantized_Base_Adapter_Finetuning.md#3-foundational-definitions) |
| P_ad | Distinct trainable adapter elements | Count | Add extra heads or biases; account for shared parameters. | [02 formulas](02_QLoRA_Quantized_Base_Adapter_Finetuning.md#4-core-formulas-with-full-derivation) |
| M_peak | Peak simultaneous device memory | Bytes | Not the sum of independently timed peaks; host paging is separate. | [02 formulas](02_QLoRA_Quantized_Base_Adapter_Finetuning.md#4-core-formulas-with-full-derivation) |
| NF4 | Nonuniform four-bit NormalFloat weight code format | Four-bit code plus scale metadata | Not interchangeable with uniform signed int4. | [02 definitions](02_QLoRA_Quantized_Base_Adapter_Finetuning.md#3-foundational-definitions) |
| m, n | Logical query and key token positions | Integer token indices | Normally zero-based; not physical padded offsets. | [03 definitions](03_RoPE_and_Position_Interpolation.md#3-foundational-definitions) |
| R(p), theta_j | Rotary position matrix and pair frequency | Matrix; radians per position | Frequency, layout and rotary dimension depend on checkpoint. | [03 definitions](03_RoPE_and_Position_Interpolation.md#3-foundational-definitions) |
| L_train, L_target, s | Trained length, target length and interpolation scale | Token counts; dimensionless scale | Target acceptance is not demonstrated utilization; s uses the convention in sheet 03. | [03 formulas](03_RoPE_and_Position_Interpolation.md#4-core-formulas-with-full-derivation) |
| H_kv, d, b | KV head count, head dimension and bytes per cached scalar | Counts; bytes/scalar | Cache payload excludes allocator and other model allocations. | [03 complexity](03_RoPE_and_Position_Interpolation.md#9-complexity--cost-analysis) |

## 8. Quick Reference Card directory

| Card | Core rule | Use condition | Critical assumption | Avoid when |
|---|---|---|---|---|
| [01 card](01_LoRA_Parameter_Efficient_Finetuning.md#12-quick-reference-card) | Count both low-rank factors per target. | Shapes, rank and target list are known. | Trainable components and sharing are accounted for. | Model architecture or target selection is unknown. |
| [02 card](02_QLoRA_Quantized_Base_Adapter_Finetuning.md#12-quick-reference-card) | Sum concurrent base, metadata, adapter, gradient, optimizer, activation and other allocations. | A compatible frozen quantized base and adapter are available. | All live memory categories are disjoint and sufficiently measured. | Quality, safety, backend or capacity requirements fail. |
| [03 card](03_RoPE_and_Position_Interpolation.md#12-quick-reference-card) | Matching rotations yield relative-position scores; PI scales positions, not token count. | Compatible RoPE checkpoint and measured extension need. | Query/key position policies and cache agree. | Quality, safety or capacity fails. |

## 9. Common Issues locator

**Collection-wide Common Issues file: planned, not written; completed issue count: 0.** The existing [02 failure section](02_QLoRA_Quantized_Base_Adapter_Finetuning.md#8-failure-modes--edge-cases) contains hypothetical local diagnostic playbooks, not a substitute for a reviewed final Common Issues artifact.

| Local ID | Category / symptom | Diagnostic starting point | Anchor |
|---|---|---|---|
| F-01 | Scale / OOM at load | Inventory device processes, base and metadata. | [Sheet 02 failures](02_QLoRA_Quantized_Base_Adapter_Finetuning.md#8-failure-modes--edge-cases) |
| F-03 | Scale / OOM at first optimizer step | Inspect lazy state, gradients and peak timeline. | [Sheet 02 failures](02_QLoRA_Quantized_Base_Adapter_Finetuning.md#8-failure-modes--edge-cases) |
| F-05 | Quality / quantized run regresses | Compare paired base, quantized base and adapter controls. | [Sheet 02 failures](02_QLoRA_Quantized_Base_Adapter_Finetuning.md#8-failure-modes--edge-cases) |
| F03 | Concept / cached and uncached decoding diverge | Compare position IDs, cache mapping and per-step logits. | [Sheet 03 failures](03_RoPE_and_Position_Interpolation.md#8-failure-modes--edge-cases) |
| F09 | Scale / generation exhausts memory | Measure retained KV payload and live sequence lengths. | [Sheet 03 failures](03_RoPE_and_Position_Interpolation.md#8-failure-modes--edge-cases) |

## 10. Empirical sources, versions and collection maintenance

| Source / date / version | Supported claim and setup | Reproduced? / limitation | Review trigger |
|---|---|---|---|
| [QLoRA paper](https://arxiv.org/abs/2305.14314), Dettmers et al., 2023-05-23 | Frozen 4-bit base, NF4, double quantization, paged optimizers and experiment-specific findings. | Not reproduced; results do not establish universal parity. | Paper correction or new target task/model. |
| [PEFT quantization guide](https://huggingface.co/docs/peft/v0.18.0/en/developer_guides/quantization), v0.18.0, accessed 2026-11-07 | Documented four-bit configuration, preparation and adapter targeting. | Guide consulted; local code execution and dependency compatibility not verified. | PEFT, Transformers, bitsandbytes, device or driver changes. |
| Sheet 02 worked examples, 2026-11-07 | Constructed arithmetic and sensitivity estimates. | Arithmetic independently checked; workload and performance are illustrative, not measured. | Assumption or arithmetic correction. |
| Visible source tree label `8.3_QLoRA_and_Quantization_Aware_Fine_Tuning… [TRUNCATED/UNVERIFIED]` | Establishes visible topic only. | Original source contents unavailable; missing suffix not reconstructed. | Complete authoritative source becomes available. |
| [RoFormer](https://arxiv.org/abs/2104.09864), Su et al., 2021-04-20 | RoPE uses position-dependent rotations to encode relative attention interactions. | Abstract consulted; algebra derived in sheet 03; experiments not reproduced. | Paper correction or checkpoint rotary-convention change. |
| [Position Interpolation](https://arxiv.org/abs/2306.15595), Chen et al., 2023-06-27 | Reports downscaled positions and model-specific adaptation experiments up to 32768 tokens. | Abstract consulted; experiments not reproduced; no universal quality guarantee. | New model, task, position policy or evaluation evidence. |
| Module 11 directory inventory | Lists five context-length topics, including RoPE and evaluation. | Source contents unavailable; inventory is not empirical evidence. | Original source material becomes available. |

Per-file status: 01 draft/unreviewed; 02 draft/unreviewed; 03 draft/unreviewed. Open issues: independent substantive review, tested environments and empirical task evaluations remain outstanding. No publication or collection-wide source-coverage completion is claimed.

### Reference-informed additions to develop

These are planned writing briefs, not completed cases or measured results. See the [planning catalog](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Brief | Proposed scope and checks | Reference | Status |
|---|---|---|---|
| PLAN-2026-IR004 | Tenant-scoped LoRA lifecycle, cache keys, session pinning, authorization and negative tests. | [REF-IR004](../../../../INTERVIEW_CASES/Agentic_AI_System/Finetuning_LLM/multi_tenant_adapter_platform.md) | Planned. |
| PLAN-2026-IR009 | Licensed corpus splits, scaling proxy sweeps, parallelism, checkpoint recovery and held-out quality. | [REF-IR009](../../../../INTERVIEW_CASES/Agentic_AI_System/LLM_TrainingSCALING/legal_pretraining_scaling.md) | Planned. |
| PLAN-2026-IR010 | Code repository/time splits, authorized retrieval, multilingual sampling and language-specific quality. | [REF-IR010](../../../../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Code_And_Multilingual_Models.md) | Planned. |
| PLAN-2026-IR017 | Speech adaptation, speaker/call/time splits, independent human gold and paired WER uncertainty. | [REF-IR017](../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Whisper_Adaptation_And_Evaluation.md) | Planned. |

## 11. Collection acceptance checklist

- [x] Three available sheets are labeled draft and unreviewed; planned items are not represented as complete.
- [x] Topic ladder, lookup, three reading paths, prerequisites and exclusions are recorded.
- [x] Glossary and card directory link the available material and state essential assumptions.
- [x] Sheet 02 includes 12 sections, checked illustrative arithmetic, sourced claims, diagnostics and interview answers.
- [x] Sheet 03 contains the 12-section outline, illustrative arithmetic, primary-paper references, local diagnostics and answered interviews.
- [ ] Obtain independent substantive technical review and validate the examples against a supported runtime.
- [ ] Write and review the remaining planned sheets and a final Common Issues file with at least 25 distinct complete issues.
- [ ] Complete collection-wide source coverage, empirical evidence, navigation and publication review; this index is not a publication certification.
