# Module 21 Cheat Sheets â€” Fine-Tuning, Alignment and Model Customisation

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This catalog contains planned topics and an unreviewed draft; expand it to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Draft and planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| — | customisation decision tree | Concise patterns, decision rules, assumptions, and common traps. | planned |
| — | dataset quality rubric | Concise patterns, decision rules, assumptions, and common traps. | planned |
| — | post-training runbook | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 01 | [LoRA parameter-efficient fine-tuning](01_LoRA_Parameter_Efficient_Finetuning.md) | Rank and parameter counts, memory caveats, quality evaluation, deployment and failure boundaries. | draft; not reviewed |

The LoRA sheet is a core-methods draft: prerequisites are matrix multiplication, transformer projections, gradient descent and token-level loss. Its [quick reference card](01_LoRA_Parameter_Efficient_Finetuning.md#12-quick-reference-card) summarizes the update, counts, memory and rollout boundaries. Examples are illustrative, not measured production results. Source coverage for other PEFT topics remains planned; this sheet does not complete the broader source module.

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR004 | LoRA lifecycle; tenant-scoped data; adapter cache keys; session pinning; compatible base checkpoints | Cross-tenant negative tests, domain-task quality, adapter-swap races and latency under a measured tenant mix. Boundary: Adapters do not establish isolation or keep changing facts current; tenant counts and SLOs need new requirements. | [REF-IR004](../../../../INTERVIEW_CASES/Agentic_AI_System/Finetuning_LLM/multi_tenant_adapter_platform.md) | planned; not written |
| PLAN-2026-IR009 | Scaling-law proxy sweeps; licensed corpus splits; parallelism; memory accounting; checkpoint recovery | Held-out domain and general quality, full training cost, achieved utilization and restart/reproducibility drills. Boundary: The reference compute and GPU figures are assumptions; an internal legal checkpoint is not validated legal advice. | [REF-IR009](../../../../INTERVIEW_CASES/Agentic_AI_System/LLM_TrainingSCALING/legal_pretraining_scaling.md) | planned; not written |
| PLAN-2026-IR010 | Fill-in-the-middle; repository/time splits; authorized code retrieval; multilingual sampling; language adapters | Sandboxed functional correctness and latency for code; native-reviewed grounding and worst-language regressions for support. Boundary: Split this reference into two independent future narratives; code-test passage does not prove security or language accuracy. | [REF-IR010](../../../../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Code_And_Multilingual_Models.md) | planned; not written |
| PLAN-2026-IR017 | Whisper adaptation; LoRA and general replay; speaker/call/time splits; independent human gold; scoped deployment | Paired WER with uncertainty by domain and code-switch slice, real-time factor and supported-route regression gates. Boundary: Model-generated transcripts are not independent gold; audio resampling does not restore missing frequencies. | [REF-IR017](../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Whisper_Adaptation_And_Evaluation.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
