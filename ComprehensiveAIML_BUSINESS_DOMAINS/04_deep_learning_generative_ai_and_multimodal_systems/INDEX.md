# Part 04 — Deep Learning, Generative AI and Multimodal Systems

> **Folder:** 04_deep_learning_generative_ai_and_multimodal_systems  
> **Purpose:** Representation learning, foundation models, post-training, and multimodal product systems.

## Position in the programme

Parts 01-03; foundation-model work assumes data, ML validation, and systems basics.

## Modules

| # | Module | Focus | Indexed topics |
|---:|---|---|---:|
| 17 | [Deep Learning, Computer Vision and Representation Learning](./17_deep_learning_computer_vision_and_representation_learning/INDEX.md) | Learn modern neural representations for images, video, spatial data, and transfer learning. | 5 |
| 18 | [NLP, Speech and Document Intelligence](./18_nlp_speech_and_document_intelligence/INDEX.md) | Build language, voice, and document systems around retrieval, extraction, transcription, and human verification. | 5 |
| 19 | [Transformers, LLMs and Foundation-Model Internals](./19_transformers_llms_and_foundation_model_internals/INDEX.md) | Understand tokenisation, attention, scaling, context, and inference so model choices are grounded in mechanics. | 5 |
| 20 | [RAG, Context Engineering and Enterprise Knowledge](./20_rag_context_engineering_and_enterprise_knowledge/INDEX.md) | Build grounded knowledge systems using ingestion, retrieval, reranking, citations, permissions, and evaluation. | 5 |
| 21 | [Fine-Tuning, Alignment and Model Customisation](./21_finetuning_alignment_and_model_customisation/INDEX.md) | Choose between prompting, retrieval, fine-tuning, preference optimization, distillation, and adapters based on evidence. | 5 |
| 22 | [Multimodal Models, Video, Voice and World Models](./22_multimodal_models_video_voice_and_world_models/INDEX.md) | Combine visual, textual, audio, temporal, and action data for perception-rich applications while measuring cross-modal reliability. | 5 |

## Part-level collections

- [Case studies](./Case_Studies/INDEX.md) connect multiple modules into cross-cutting system and business narratives.
- [Cheat sheets](./Cheat_Sheets/INDEX.md) consolidate the part's reusable patterns, decision frameworks, and review checklists.

## Completion rule

A learner should be able to explain the design trade-offs, choose an evidence-based approach, and complete at least one case-study brief before moving on. Each module index records its own prerequisites and planned case/cheat-sheet coverage.

## Local coverage ledger

**Inventory basis:** [directory.txt](../../Templates/directory.txt), [exact source assignments](../00_SOURCE_TOPIC_CROSSWALK.json), and the linked focused briefs. The inventory describes visible names, not inspected original lesson bodies. Source module numbers are not course module numbers.

| Course module | Original topic commitments | Assigned source entries | Visible technical entries | Containers | Supporting artifacts | Unresolved source entries | Focused groups |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 5 | 16 | 14 | 0 | 1 | 1 | 1 |
| 18 | 5 | 22 | 19 | 2 | 1 | 0 | 5 |
| 19 | 5 | 32 | 27 | 5 | 0 | 0 | 6 |
| 20 | 5 | 4 | 3 | 1 | 0 | 0 | 2 |
| 21 | 5 | 54 | 38 | 7 | 5 | 4 | 11 |
| 22 | 5 | 24 | 21 | 2 | 1 | 0 | 5 |
| **Total** | **30** | **152** | **122** | **17** | **8** | **5** | **30** |

Counts are source-inventory classifications, not lesson-completion counts. Overlapping source titles can share a teaching artifact only when their distinct methods, assumptions, numerical work and diagnostic boundaries remain explicit. Supporting artifacts and containers are not teaching outcomes.

The five unresolved entries are the clipped biomedical deep-learning directory, hidden `Finetune_LLM`, hidden `Tutorials-overall1`, clipped process-supervision title 7.7, and clipped QLoRA title 8.3. Visible prefixes may be taught; missing suffixes and hidden contents remain unknown.

### Original collection commitments retained

| Module | Original sheet family, retained independently of newer drafts | Original case threads |
|---:|---|---|
| 17 | Vision model selection; augmentation library; detection metrics | Visual quality inspection; medical-image triage evaluation |
| 18 | NLP task matrix; document extraction checklist; ASR metrics | Claims document extraction; call-centre transcription quality |
| 19 | Transformer anatomy; token budgeting; model selection matrix | Choose a model for code assistance; diagnose long-context quality loss |
| 20 | RAG architecture; chunking guide; retrieval metric card | Policy assistant with citations; knowledge-base migration evaluation |
| 21 | Customisation decision tree; dataset quality rubric; post-training runbook | Specialise a support model; compact a domain model for edge |
| 22 | Multimodal evaluation; voice-agent latency budget; VLA risk checklist | Multimodal maintenance copilot; video safety-event review |

These 18 reference-family and 12 case-thread commitments are not replaced by newer evaluation, permission, adapter or source-mapping drafts. Module collection catalogs own the detailed artifact mapping; part collections provide the cross-module reading paths.

### Reference-informed decision commitments

| Planning ID | Homes | Distinct decision retained |
|---|---|---|
| IR003 | 20 | Answer, abstain or escalate an effective-policy question using authorized citations |
| IR004 | 21 | Train, select, pin, version and roll back tenant adapters without cross-tenant leakage |
| IR007 | 19 | Admit and schedule inference requests under cache, latency and isolation limits |
| IR009 | 19, 21 | Choose from-scratch versus continued training using scoped quality and all-in cost |
| IR010 | 18, 21 | Justify repository-aware completion and multilingual adaptation with independent evaluation |
| IR011 | 18, 22 | Extract claim fields with page-region provenance or route the bundle for review |
| IR012 | 20 | Select chunk/index design using all-required-span success and filtered exact baselines |
| IR013 | 20 | Determine source eligibility under effective and recorded-time versions before ranking |
| IR016 | 18 | Aggregate customer objections with verified transcription, speaker roles and evidence spans |
| IR017 | 18, 21 | Promote a speech adapter only to independently evaluated product/language routes |

Cross-listed planning IDs identify one decision family, not duplicate source-completion credit. Editorial review, runtime validation and inaccessible-source inspection remain separate evidence gates.