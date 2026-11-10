# Wave assignment: 04_deep_learning_generative_ai_and_multimodal_systems

## Unit `04_deep_learning_generative_ai_and_multimodal_systems`
- existing cases (0): none
- existing sheets (0): none
- index sections: cases 0/7, sheets 0/11

## Unit `04_deep_learning_generative_ai_and_multimodal_systems/17_deep_learning_computer_vision_and_representation_learning`
- existing cases (1): 01_visual_quality_inspection.md
- existing sheets (5): 01_Vision_Evaluation_and_Error_Taxonomy.md, 02_CNNs_Attention_and_Transfer_Learning.md, 03_Vision_Transformers.md, 04_Detection_and_Segmentation.md, 05_Self_Supervision_and_Contrastive_Learning.md
- promised case rows (1):
  - Medical-image triage evaluation
- index sections: cases 7/7, sheets 11/11

## Unit `04_deep_learning_generative_ai_and_multimodal_systems/18_nlp_speech_and_document_intelligence`
- existing cases (2): 01_claims_document_extraction.md, 02_call_centre_transcription_quality.md
- existing sheets (2): 01_NLP_Task_Contracts_Classification_NER_Embeddings.md, 03_asr-transcription-evaluation.md
- promised case rows (6):
  - Repository-aware completion / language adapter (planned shared brief)
  - VLM/OCR adjuster assistance (planned shared brief)
  - Customer-objection aggregation (planned)
  - Domain ASR adaptation (planned shared brief)
  - ASR quality can vary by language and route.
  - Language-aware systems need independent splits.
- promised sheet rows (4):
  - Fill-in-the-middle; repository/time splits; authorized code retrieval; multilingual sampling; language adapters
  - Page and region provenance; thumbnail/tiling tradeoffs; schema validation; claim-level temporal splits
  - Channel-aware ASR; VAD; code-switching; speaker roles; redaction; evidence spans; cohort aggregation
  - Whisper adaptation; LoRA and general replay; speaker/call/time splits; independent human gold; scoped deployment
- index sections: cases 7/7, sheets 11/11

## Unit `04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals`
- existing cases (1): 01_code_assistance_model_choice.md
- existing sheets (0): none
- promised case rows (4):
  - Interactive inference latency and admitted throughput (planned)
  - Domain-training budget and restart headroom (planned shared brief)
  - Serving choice changes task latency.
  - Quality/cost claims require a complete budget.
- promised sheet rows (2):
  - Continuous batching; chunked prefill; paged KV storage; prefix authorization; speculative-decoding correctness
  - Scaling-law proxy sweeps; licensed corpus splits; parallelism; memory accounting; checkpoint recovery
- index sections: cases 7/7, sheets 0/11

## Unit `04_deep_learning_generative_ai_and_multimodal_systems/20_rag_context_engineering_and_enterprise_knowledge`
- existing cases (0): none
- existing sheets (2): 01_Permission_Aware_Retrieval.md, 02_RAG_Ingestion_Chunking_and_Grounded_Answers.md
- promised case rows (5):
  - Applicable-policy answer/abstain decision (planned)
  - Chunk and index selection (planned)
  - Effective-date source eligibility (planned)
  - Chunking changes evidence recall.
  - Eligibility precedes ranking.
- promised sheet rows (3):
  - Authorized retrieval; effective policy versions; claim-level citations; embedding migrations
  - Structure-aware chunking; actual tokenizer limits; HNSW and exact filtered baselines; CDC; deletion propagation
  - Effective intervals; recorded-time history; jurisdiction rules; hybrid retrieval; reranking and claim verification
- index sections: cases 7/7, sheets 0/11

## Unit `04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation`
- existing cases (1): 01_support_domain_adaptation.md
- existing sheets (3): 01_LoRA_Parameter_Efficient_Finetuning.md, 02_QLoRA_Quantized_Base_Adapter_Finetuning.md, 03_RoPE_and_Position_Interpolation.md
- promised case rows (10):
  - Customize tenant assistants without crossing isolation boundaries
  - Allocate a domain-training budget with restart headroom
  - Compare specialized models with authorized shared-model baselines
  - Adapt domain ASR while preserving supported-domain quality
  - [REF-IR004](../../../../INTERVIEW_CASES/Agentic_AI_System/Finetuning_LLM/multi_tenant_adapter_platform.md)
  - [REF-IR009](../../../../INTERVIEW_CASES/Agentic_AI_System/LLM_TrainingSCALING/legal_pretraining_scaling.md)
  - [REF-IR010](../../../../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Code_And_Multilingual_Models.md)
  - [REF-IR017](../../../../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Whisper_Adaptation_And_Evaluation.md)
  - Related adapter lifecycle
  - Related comparison with a shared baseline
- promised sheet rows (4):
  - Tenant-scoped LoRA lifecycle, cache keys, session pinning, authorization and negative tests.
  - Licensed corpus splits, scaling proxy sweeps, parallelism, checkpoint recovery and held-out quality.
  - Code repository/time splits, authorized retrieval, multilingual sampling and language-specific quality.
  - Speech adaptation, speaker/call/time splits, independent human gold and paired WER uncertainty.
- index sections: cases 7/7, sheets 11/11

## Unit `04_deep_learning_generative_ai_and_multimodal_systems/22_multimodal_models_video_voice_and_world_models`
- existing cases (0): none
- existing sheets (0): none
- promised case rows (2):
  - Document VLM versus OCR for adjuster assistance (planned shared brief)
  - Planning only
- promised sheet rows (1):
  - Planning reference only
- index sections: cases 7/7, sheets 11/11
