# Reference-informed case and cheat-sheet plan

The interview materials are examples used only as references for curriculum planning. They are not copied, adopted as production evidence, or counted as completed cases. Every brief below remains **not-measured / proposed**; no results have been imported.

## 1. Collection metadata

| Field | Value |
|---|---|
| Editorial owner role | Curriculum maintainer; individual authors not yet assigned |
| Reference inventory reviewed | 2026-11-06; planning scope and provenance only |
| Review cadence | At authoring, source changes, and before publication |
| Index version | 1.0 |
| Planning ID format | PLAN-2026-IRNNN; replace or crosswalk to canonical case IDs when authoring |
| Reference ID format | REF-IRNNN; identifies an example file, not an authored case |
| Source of truth | This planning catalog; original examples remain in Train/INTERVIEW_CASES |
| Reference inventory | 36 files; 68 reading units, including overlapping anthology material |
| Completed cases created by this plan | 0 |

The [case master template](../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md), [cheat-sheet master template](../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md), and [course authoring/numerical standards](00_AUTHORING_AND_NUMERICAL_STANDARDS.md) govern future content. Reference notes have not been certified against these templates.

The first destination is the proposed editorial home; others are supporting curricula. A repeated planning ID in a local index is a cross-reference to this one brief. No link to an unwritten case file is fabricated. Multi-case anthologies must be split into independently framed narratives before publication.

## 2. Proposed case catalog

| Planning ID | Proposed title and reference | Editorial home | Case type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Key result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR001 | [Gate retraining on corpus provenance and long-tail quality (example reference only)](../INTERVIEW_CASES/Agentic_AI_System/AI_engg_challenges/synthetic_corpus_contamination.md) | [27_data_quality_observability_and_reliability_engineering](05_ai_platforms_data_systems_and_production_engineering/27_data_quality_observability_and_reliability_engineering/INDEX.md) | hybrid | Which documents should be admitted or quarantined before the next training run? | `distribution-shift` | `data-leakage`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR002 | [Select model routes within hard customer constraints (example reference only)](../INTERVIEW_CASES/Agentic_AI_System/AI_frontier_sys/constrained_provider_routing.md) | [25_model_serving_inference_optimisation_and_cost](05_ai_platforms_data_systems_and_production_engineering/25_model_serving_inference_optimisation_and_cost/INDEX.md) | llm-agentic | Which eligible provider and fallback path satisfy residency, capability, deadline and spend requirements? | `eligibility-before-ranking` | `cost-latency-tradeoff` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR003 | [Answer support questions from the applicable policy version (example reference only)](../INTERVIEW_CASES/Agentic_AI_System/Applied_AI_engineering/grounded_policy_rag.md) | [20_rag_context_engineering_and_enterprise_knowledge](04_deep_learning_generative_ai_and_multimodal_systems/20_rag_context_engineering_and_enterprise_knowledge/INDEX.md) | llm-agentic | When should a support assistant answer, abstain or escalate a policy question? | `evaluation-mismatch` | `authorization-scope`, `materialized-view` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR004 | [Customize tenant assistants without crossing isolation boundaries (example reference only)](../INTERVIEW_CASES/Agentic_AI_System/Finetuning_LLM/multi_tenant_adapter_platform.md) | [21_finetuning_alignment_and_model_customisation](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md) | llm-agentic | How should tenant adapters be trained, selected, versioned and rolled back? | `authorization-scope` | `atomic-release-bundle`, `rollout-regression` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR005 | [Release and recover an agent as a compatible bundle (example reference only)](../INTERVIEW_CASES/Agentic_AI_System/INDEX/Agent_Release_And_Incident_Operations.md) | [55_incident_response_failure_modes_and_technical_leadership](10_architecture_practice_and_portfolio_capstones/55_incident_response_failure_modes_and_technical_leadership/INDEX.md) | hybrid | Which model, prompt, tools, index and policy versions can be promoted and recovered together? | `atomic-release-bundle` | `rollout-regression`, `human-in-the-loop` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR006 | [Recover procurement workflows without replaying approvals (example reference only)](../INTERVIEW_CASES/Agentic_AI_System/Langraph/durable_human_approval_graph.md) | [30_agentic_systems_workflows_and_orchestration](06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md) | llm-agentic | How can a purchase proceed only after approval of its exact version despite retries and restarts? | `idempotent-side-effects` | `human-in-the-loop`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR007 | [Balance interactive inference latency and admitted throughput (example reference only)](../INTERVIEW_CASES/Agentic_AI_System/LLM_Inference/continuous_batching_serving.md) | [25_model_serving_inference_optimisation_and_cost](05_ai_platforms_data_systems_and_production_engineering/25_model_serving_inference_optimisation_and_cost/INDEX.md) | hybrid | Which scheduling and KV-memory policy meets latency and isolation requirements at the admitted load? | `resource-admission` | `cost-latency-tradeoff`, `authorization-scope` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR008 | [Block assistant releases on auditable quality and safety gates (example reference only)](../INTERVIEW_CASES/Agentic_AI_System/LLM_Safety_Evaluation/release_evaluation_platform.md) | [28_ai_evaluation_monitoring_and_continuous_improvement](05_ai_platforms_data_systems_and_production_engineering/28_ai_evaluation_monitoring_and_continuous_improvement/INDEX.md) | llm-agentic | Does a candidate pass the required suites with enough evidence to justify promotion? | `evaluation-mismatch` | `data-leakage`, `rollout-regression` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR009 | [Allocate a domain-training budget with restart headroom (example reference only)](../INTERVIEW_CASES/Agentic_AI_System/LLM_TrainingSCALING/legal_pretraining_scaling.md) | [24_distributed_training_data_pipelines_and_ml_compute](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md) | hybrid | Should a domain model be trained from scratch or continued from an existing checkpoint within an all-in budget? | `cost-latency-tradeoff` | `data-leakage`, `atomic-release-bundle` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR010 | [Compare specialized models with authorized shared-model baselines (example reference only)](../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Code_And_Multilingual_Models.md) | [33_coding_agents_ai_sdcl_and_developer_productivity](06_llm_agents_knowledge_systems_and_ai_augmented_development/33_coding_agents_ai_sdcl_and_developer_productivity/INDEX.md) | llm-agentic | When do repository-aware completion or language adapters justify their added complexity? | `data-leakage` | `authorization-scope`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR011 | [Compare document VLMs with OCR for adjuster assistance (example reference only)](../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Insurance_Document_VLM.md) | [22_multimodal_models_video_voice_and_world_models](04_deep_learning_generative_ai_and_multimodal_systems/22_multimodal_models_video_voice_and_world_models/INDEX.md) | hybrid | Which claim fields can be extracted reliably, and which bundles require adjuster review? | `human-in-the-loop` | `evaluation-mismatch`, `delayed-selective-labels` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR012 | [Preserve answer evidence through ingestion and filtered retrieval (example reference only)](../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Chunking_And_Vector_Index.md) | [20_rag_context_engineering_and_enterprise_knowledge](04_deep_learning_generative_ai_and_multimodal_systems/20_rag_context_engineering_and_enterprise_knowledge/INDEX.md) | llm-agentic | Which chunking and index design retrieves every required authorized span within the service budget? | `materialized-view` | `authorization-scope`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR013 | [Retrieve regulations applicable to the question date and jurisdiction (example reference only)](../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Regulatory_RAG_And_Reranking.md) | [20_rag_context_engineering_and_enterprise_knowledge](04_deep_learning_generative_ai_and_multimodal_systems/20_rag_context_engineering_and_enterprise_knowledge/INDEX.md) | llm-agentic | Which sources are eligible before ranking, and when is legal-owner review required? | `eligibility-before-ranking` | `snapshot-consistency`, `human-in-the-loop` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR014 | [Separate extraction correctness from routing economics (example reference only)](../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Structured_Extraction_And_Cost_Routing.md) | [31_tools_mcp_protocols_and_enterprise_integrations](06_llm_agents_knowledge_systems_and_ai_augmented_development/31_tools_mcp_protocols_and_enterprise_integrations/INDEX.md) | llm-agentic | How should extraction contracts and model routes be accepted without confusing valid syntax with business success? | `evaluation-mismatch` | `idempotent-side-effects`, `cost-latency-tradeoff` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR015 | [Publish reviewed coaching scripts from current product evidence (example reference only)](../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Approved_Objection_Response_Scripts.md) | [34_conversational_ai_voice_agents_and_customer_operations](06_llm_agents_knowledge_systems_and_ai_augmented_development/34_conversational_ai_voice_agents_and_customer_operations/INDEX.md) | llm-agentic | Which generated scripts are fit for an approved, versioned coaching registry? | `human-in-the-loop` | `eligibility-before-ranking`, `atomic-release-bundle` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR016 | [Distinguish speech, speaker and extraction errors in call analytics (example reference only)](../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Multilingual_Call_Objections.md) | [18_nlp_speech_and_document_intelligence](04_deep_learning_generative_ai_and_multimodal_systems/18_nlp_speech_and_document_intelligence/INDEX.md) | hybrid | Which customer objections can be reported with reliable speaker attribution and evidence? | `evaluation-mismatch` | `authorization-scope`, `human-in-the-loop` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR017 | [Adapt domain ASR while preserving supported-domain quality (example reference only)](../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Whisper_Adaptation_And_Evaluation.md) | [21_finetuning_alignment_and_model_customisation](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md) | hybrid | Should a speech adapter be promoted, and to which evaluated product and language domains? | `rollout-regression` | `distribution-shift`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR018 | [Specify correctness contracts before choosing production algorithms (example reference only)](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_01.md) | [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md) | hybrid | Which guarantees are needed for road routing, distributed admission and phrase retrieval? | `correctness-contract` | `snapshot-consistency`, `bounded-work` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR019 | [Keep routing correctness separate from ETA quality (example reference only)](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_02.md) | [04_system_design_distributed_systems_and_cloud](01_foundations_computing_and_problem_solving/04_system_design_distributed_systems_and_cloud/INDEX.md) | hybrid | How should a navigation service publish legal optimal routes while separately evaluating predicted arrival time? | `snapshot-consistency` | `correctness-contract`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR020 | [Translate local algorithm guarantees into distributed service contracts (example reference only)](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_03.md) | [54_ml_and_ai_system_design_interviews](10_architecture_practice_and_portfolio_capstones/54_ml_and_ai_system_design_interviews/INDEX.md) | hybrid | Which correctness and failure guarantees survive sharding, delayed updates and partial outages? | `correctness-contract` | `resource-admission`, `materialized-view` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR021 | [Evolve an authorized near-real-time search index safely (example reference only)](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_04.md) | [14_recommendation_search_ads_and_ranking](03_machine_learning_optimisation_and_intelligent_retrieval/14_recommendation_search_ads_and_ranking/INDEX.md) | hybrid | How should search remain correct across indexing, deletion, ranking and shard failures? | `materialized-view` | `authorization-scope`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR022 | [Bound global API admission with token leases (example reference only)](../INTERVIEW_CASES/DSA/Algo_Design_Problems/Rate_Limiting/distributed_token_buckets.md) | [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md) | hybrid | Which admission policy remains within its global budget during contention or coordinator failure? | `resource-admission` | `correctness-contract`, `bounded-work` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR023 | [Reject stale shortcuts without losing a shortest-path guarantee (example reference only)](../INTERVIEW_CASES/DSA/Algo_Design_Problems/Routing/closure_aware_shortest_paths.md) | [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md) | hybrid | When must a route query fall back to the live base graph after a closure? | `correctness-contract` | `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR024 | [Bound phrase-search memory while preserving exact matches (example reference only)](../INTERVIEW_CASES/DSA/Algo_Design_Problems/Search/phrase_search_postings.md) | [14_recommendation_search_ads_and_ranking](03_machine_learning_optimisation_and_intelligent_retrieval/14_recommendation_search_ads_and_ranking/INDEX.md) | hybrid | How can common-term phrase queries be evaluated without materializing huge posting lists? | `bounded-work` | `correctness-contract`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR025 | [Choose scheduling algorithms from objective and constraints (example reference only)](../INTERVIEW_CASES/DSA/index_basics/Greedy_Scheduling/activity_selection.md) | [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md) | analytical | When is earliest-finish greedy optimal, and when must scheduling become a weighted optimization problem? | `correctness-contract` | None | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR026 | [Select data structures from operation mix and failure semantics (example reference only)](../INTERVIEW_CASES/DSA/index_basics/INTERVIEW_CASE_STUDIES.md) | [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md) | hybrid | Which data structure fits the update/query workload while retaining the required correctness and resource bounds? | `correctness-contract` | `bounded-work`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR027 | [Maintain exact inventory range totals under point updates (example reference only)](../INTERVIEW_CASES/DSA/index_basics/Range_Queries/fenwick_inventory_totals.md) | [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md) | analytical | Which update and snapshot contract makes range totals correct and efficient? | `snapshot-consistency` | `correctness-contract` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR028 | [Bound request amplification before expensive parsing and validation (example reference only)](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Algorithmic_DoS/hash_and_regex_resource_exhaustion.md) | [55_incident_response_failure_modes_and_technical_leadership](10_architecture_practice_and_portfolio_capstones/55_incident_response_failure_modes_and_technical_leadership/INDEX.md) | incident | Which per-request limits and algorithm changes stop CPU exhaustion without rejecting legitimate traffic unnecessarily? | `bounded-work` | `resource-admission`, `incident-postmortem` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR029 | [Return useful graph candidates under explicit traversal budgets (example reference only)](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Graph_Traversal/bounded_bfs_incident.md) | [15_graph_ml_knowledge_graphs_and_network_analytics](03_machine_learning_optimisation_and_intelligent_retrieval/15_graph_ml_knowledge_graphs_and_network_analytics/INDEX.md) | incident | How should a recommendation API respond when graph expansion reaches a node, edge, memory or time limit? | `bounded-work` | `correctness-contract`, `incident-postmortem` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR030 | [Diagnose production failures from violated algorithm invariants (example reference only)](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md) | [55_incident_response_failure_modes_and_technical_leadership](10_architecture_practice_and_portfolio_capstones/55_incident_response_failure_modes_and_technical_leadership/INDEX.md) | incident | Which invariant failed, how can it be contained, and which replay would prevent recurrence? | `incident-postmortem` | `bounded-work`, `correctness-contract` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR031 | [Load authorized hierarchies with bounded database work (example reference only)](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Tree_Queries/policy_hierarchy_n_plus_one.md) | [06_data_modeling_databases_lakehouse_and_governance](02_data_analytics_and_decision_science/06_data_modeling_databases_lakehouse_and_governance/INDEX.md) | incident | Should a subtree API use a recursive CTE or bounded level-order batching for its workload? | `bounded-work` | `authorization-scope`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR032 | [Balance autocomplete freshness, memory and abuse controls (example reference only)](../INTERVIEW_CASES/DSA/Sys_Des_DSA/Autocomplete/frequency_admission_and_freshness.md) | [14_recommendation_search_ads_and_ranking](03_machine_learning_optimisation_and_intelligent_retrieval/14_recommendation_search_ads_and_ranking/INDEX.md) | hybrid | Which terms enter a bounded suggestion index, and how can safe new trends appear promptly? | `materialized-view` | `bounded-work`, `rollout-regression` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR033 | [Rebalance caches without overwhelming the authoritative store (example reference only)](../INTERVIEW_CASES/DSA/Sys_Des_DSA/Cache_Resilience/cold_cache_and_ring_migration.md) | [04_system_design_distributed_systems_and_cloud](01_foundations_computing_and_problem_solving/04_system_design_distributed_systems_and_cloud/INDEX.md) | hybrid | How should cache fills and ring migration preserve freshness while limiting database amplification? | `materialized-view` | `resource-admission`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR034 | [Design stateful serving structures with explicit consistency (example reference only)](../INTERVIEW_CASES/DSA/Sys_Des_DSA/INTERVIEW_SYSTEM_DESIGN_CASES.md) | [04_system_design_distributed_systems_and_cloud](01_foundations_computing_and_problem_solving/04_system_design_distributed_systems_and_cloud/INDEX.md) | hybrid | Which cache, typeahead, leaderboard and deduplication contracts remain valid during refresh and failure? | `correctness-contract` | `materialized-view`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR035 | [Separate live rank approximation from exact competition settlement (example reference only)](../INTERVIEW_CASES/DSA/Sys_Des_DSA/Leaderboards/score_bucket_rank_consistency.md) | [04_system_design_distributed_systems_and_cloud](01_foundations_computing_and_problem_solving/04_system_design_distributed_systems_and_cloud/INDEX.md) | hybrid | How should score events support fast display ranks and a reproducible final prize result? | `snapshot-consistency` | `idempotent-side-effects`, `correctness-contract` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR036 | [Evaluate incremental investigator value from graph fraud signals (example reference only)](../INTERVIEW_CASES/Graph_Rings/Fraud_Ring_Detection/Claims_Fraud_Ring_Case.md) | [15_graph_ml_knowledge_graphs_and_network_analytics](03_machine_learning_optimisation_and_intelligent_retrieval/15_graph_ml_knowledge_graphs_and_network_analytics/INDEX.md) | predictive-ml | Which graph-enriched alerts add confirmed value at a fixed investigation capacity beyond claim-level scoring? | `delayed-selective-labels` | `human-in-the-loop`, `data-leakage` | not-measured | proposed | 2026-11-06 |  |

### Detailed scope and reference inventory

Reading-unit totals are inventory aids, not counts of unique cases or independent evidence. Reference figures, incident names, company assumptions and performance targets must not be reused as facts.

#### REF-IR001: Gate retraining on corpus provenance and long-tail quality

**Proposed brief:** `PLAN-2026-IR001`. **Mechanisms to develop:** Dataset lineage; synthetic-data feedback; deduplication limits; language and domain coverage.

**Evaluation to design:** Held-out long-tail factual accuracy, diversity and per-slice retention; compare filtering with provenance controls.

**Transfer boundary:** A synthetic-text classifier cannot establish authorship; source incident names and costs are unverified.

**Destination indexes:** [27_data_quality_observability_and_reliability_engineering](05_ai_platforms_data_systems_and_production_engineering/27_data_quality_observability_and_reliability_engineering/INDEX.md); [24_distributed_training_data_pipelines_and_ml_compute](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md); [28_ai_evaluation_monitoring_and_continuous_improvement](05_ai_platforms_data_systems_and_production_engineering/28_ai_evaluation_monitoring_and_continuous_improvement/INDEX.md).

**Example reference:** [Agentic_AI_System/AI_engg_challenges/synthetic_corpus_contamination.md](../INTERVIEW_CASES/Agentic_AI_System/AI_engg_challenges/synthetic_corpus_contamination.md). Reading units:

- [Interview case: Detecting synthetic-data feedback in a training corpus](../INTERVIEW_CASES/Agentic_AI_System/AI_engg_challenges/synthetic_corpus_contamination.md#interview-case-detecting-synthetic-data-feedback-in-a-training-corpus)

#### REF-IR002: Select model routes within hard customer constraints

**Proposed brief:** `PLAN-2026-IR002`. **Mechanisms to develop:** Eligibility before scoring; capability adapters; health-aware routing; bounded retry budgets.

**Evaluation to design:** Constraint violations, successful-task cost, deadline misses and fallback correctness on fault replays.

**Transfer boundary:** Revalidate provider capabilities and economics when authoring; source tables are not current service guarantees.

**Destination indexes:** [25_model_serving_inference_optimisation_and_cost](05_ai_platforms_data_systems_and_production_engineering/25_model_serving_inference_optimisation_and_cost/INDEX.md); [29_api_platform_engineering_cloud_native_and_finops](05_ai_platforms_data_systems_and_production_engineering/29_api_platform_engineering_cloud_native_and_finops/INDEX.md); [52_ai_economics_strategy_procurement_and_vendor_management](09_trust_security_governance_and_product_leadership/52_ai_economics_strategy_procurement_and_vendor_management/INDEX.md).

**Example reference:** [Agentic_AI_System/AI_frontier_sys/constrained_provider_routing.md](../INTERVIEW_CASES/Agentic_AI_System/AI_frontier_sys/constrained_provider_routing.md). Reading units:

- [Interview case: Route reasoning requests across model providers](../INTERVIEW_CASES/Agentic_AI_System/AI_frontier_sys/constrained_provider_routing.md#interview-case-route-reasoning-requests-across-model-providers)

#### REF-IR003: Answer support questions from the applicable policy version

**Proposed brief:** `PLAN-2026-IR003`. **Mechanisms to develop:** Authorized retrieval; effective policy versions; claim-level citations; embedding migrations.

**Evaluation to design:** Unsupported-claim rate, eligible-span recall, abstention quality and policy-version errors against expert labels.

**Transfer boundary:** Citations and entailment checks require validation; never transfer the example refund window into a new business.

**Destination indexes:** [20_rag_context_engineering_and_enterprise_knowledge](04_deep_learning_generative_ai_and_multimodal_systems/20_rag_context_engineering_and_enterprise_knowledge/INDEX.md); [03_enterprise_knowledge_support_and_sales](11_data_science_domain_specialisations/14_saas_telecom_and_b2b/03_enterprise_knowledge_support_and_sales/INDEX.md); [02_omnichannel_customer_loyalty_and_fulfilment](11_data_science_domain_specialisations/04_retail_omnichannel_offline_online_stores/02_omnichannel_customer_loyalty_and_fulfilment/INDEX.md).

**Example reference:** [Agentic_AI_System/Applied_AI_engineering/grounded_policy_rag.md](../INTERVIEW_CASES/Agentic_AI_System/Applied_AI_engineering/grounded_policy_rag.md). Reading units:

- [Interview case: Ground a customer-support policy assistant](../INTERVIEW_CASES/Agentic_AI_System/Applied_AI_engineering/grounded_policy_rag.md#interview-case-ground-a-customer-support-policy-assistant)

#### REF-IR004: Customize tenant assistants without crossing isolation boundaries

**Proposed brief:** `PLAN-2026-IR004`. **Mechanisms to develop:** LoRA lifecycle; tenant-scoped data; adapter cache keys; session pinning; compatible base checkpoints.

**Evaluation to design:** Cross-tenant negative tests, domain-task quality, adapter-swap races and latency under a measured tenant mix.

**Transfer boundary:** Adapters do not establish isolation or keep changing facts current; tenant counts and SLOs need new requirements.

**Destination indexes:** [21_finetuning_alignment_and_model_customisation](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md); [25_model_serving_inference_optimisation_and_cost](05_ai_platforms_data_systems_and_production_engineering/25_model_serving_inference_optimisation_and_cost/INDEX.md); [49_ai_security_privacy_safety_and_red_teaming](09_trust_security_governance_and_product_leadership/49_ai_security_privacy_safety_and_red_teaming/INDEX.md); [03_enterprise_knowledge_support_and_sales](11_data_science_domain_specialisations/14_saas_telecom_and_b2b/03_enterprise_knowledge_support_and_sales/INDEX.md).

**Example reference:** [Agentic_AI_System/Finetuning_LLM/multi_tenant_adapter_platform.md](../INTERVIEW_CASES/Agentic_AI_System/Finetuning_LLM/multi_tenant_adapter_platform.md). Reading units:

- [Interview case: Isolated tenant-specific LoRA assistants](../INTERVIEW_CASES/Agentic_AI_System/Finetuning_LLM/multi_tenant_adapter_platform.md#interview-case-isolated-tenant-specific-lora-assistants)

#### REF-IR005: Release and recover an agent as a compatible bundle

**Proposed brief:** `PLAN-2026-IR005`. **Mechanisms to develop:** Immutable release manifests; compatibility gates; side-effect-free shadowing; staged canaries; incident containment.

**Evaluation to design:** Bundle-compatibility tests, rollback drills, guardrail breaches and unresolved side effects after recovery.

**Transfer boundary:** A prompt-only rollback cannot restore all dependencies; example dwell times and canary percentages are not defaults.

**Destination indexes:** [55_incident_response_failure_modes_and_technical_leadership](10_architecture_practice_and_portfolio_capstones/55_incident_response_failure_modes_and_technical_leadership/INDEX.md); [26_mlops_feature_platforms_and_model_lifecycle](05_ai_platforms_data_systems_and_production_engineering/26_mlops_feature_platforms_and_model_lifecycle/INDEX.md); [30_agentic_systems_workflows_and_orchestration](06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md); [02_claims_fraud_customer_and_operations](11_data_science_domain_specialisations/10_insurance_actuarial_and_claims/02_claims_fraud_customer_and_operations/INDEX.md).

**Example reference:** [Agentic_AI_System/INDEX/Agent_Release_And_Incident_Operations.md](../INTERVIEW_CASES/Agentic_AI_System/INDEX/Agent_Release_And_Incident_Operations.md). Reading units:

- [Operating a high-impact claims agent: release and incident response](../INTERVIEW_CASES/Agentic_AI_System/INDEX/Agent_Release_And_Incident_Operations.md#operating-a-high-impact-claims-agent-release-and-incident-response)

#### REF-IR006: Recover procurement workflows without replaying approvals

**Proposed brief:** `PLAN-2026-IR006`. **Mechanisms to develop:** LangGraph checkpoints and interrupts; authenticated resume; approval binding and expiry; idempotent submission.

**Evaluation to design:** Crash-window replay, duplicate webhooks, changed purchase payloads, checkpoint migration and reconciliation tests.

**Transfer boundary:** External APIs may not support exactly-once effects; approval waiting times and run counts are exercise assumptions.

**Destination indexes:** [30_agentic_systems_workflows_and_orchestration](06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md); [31_tools_mcp_protocols_and_enterprise_integrations](06_llm_agents_knowledge_systems_and_ai_augmented_development/31_tools_mcp_protocols_and_enterprise_integrations/INDEX.md); [03_control_towers_events_and_autonomous_operations](11_data_science_domain_specialisations/07_supply_chain_logistics_warehousing_and_fulfillment/03_control_towers_events_and_autonomous_operations/INDEX.md).

**Example reference:** [Agentic_AI_System/Langraph/durable_human_approval_graph.md](../INTERVIEW_CASES/Agentic_AI_System/Langraph/durable_human_approval_graph.md). Reading units:

- [Interview case: Durable human approval in a LangGraph workflow](../INTERVIEW_CASES/Agentic_AI_System/Langraph/durable_human_approval_graph.md#interview-case-durable-human-approval-in-a-langgraph-workflow)

#### REF-IR007: Balance interactive inference latency and admitted throughput

**Proposed brief:** `PLAN-2026-IR007`. **Mechanisms to develop:** Continuous batching; chunked prefill; paged KV storage; prefix authorization; speculative-decoding correctness.

**Evaluation to design:** TTFT, inter-token latency, offered versus accepted load, KV headroom and target-equivalence tests.

**Transfer boundary:** GPU geometry, concurrency limits and speedups require workload measurements; speculative decoding can be slower.

**Destination indexes:** [25_model_serving_inference_optimisation_and_cost](05_ai_platforms_data_systems_and_production_engineering/25_model_serving_inference_optimisation_and_cost/INDEX.md); [23_gpu_architecture_accelerators_and_ai_hardware](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md); [19_transformers_llms_and_foundation_model_internals](04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/INDEX.md).

**Example reference:** [Agentic_AI_System/LLM_Inference/continuous_batching_serving.md](../INTERVIEW_CASES/Agentic_AI_System/LLM_Inference/continuous_batching_serving.md). Reading units:

- [Interview case: High-throughput interactive LLM serving](../INTERVIEW_CASES/Agentic_AI_System/LLM_Inference/continuous_batching_serving.md#interview-case-high-throughput-interactive-llm-serving)

#### REF-IR008: Block assistant releases on auditable quality and safety gates

**Proposed brief:** `PLAN-2026-IR008`. **Mechanisms to develop:** Versioned evaluation manifests; contaminated benchmarks; calibrated judges; over-refusal; sandboxed code scoring.

**Evaluation to design:** Paired quality comparisons, critical-failure review, uncertainty, judge drift and missing-suite failure tests.

**Transfer boundary:** A judge score is not a deployment guarantee; workload volume and evaluation deadlines remain unspecified.

**Destination indexes:** [28_ai_evaluation_monitoring_and_continuous_improvement](05_ai_platforms_data_systems_and_production_engineering/28_ai_evaluation_monitoring_and_continuous_improvement/INDEX.md); [49_ai_security_privacy_safety_and_red_teaming](09_trust_security_governance_and_product_leadership/49_ai_security_privacy_safety_and_red_teaming/INDEX.md); [03_evaluation_reliability_and_incidents](11_data_science_domain_specialisations/03_practical_ml_design_and_interviews/03_evaluation_reliability_and_incidents/INDEX.md).

**Example reference:** [Agentic_AI_System/LLM_Safety_Evaluation/release_evaluation_platform.md](../INTERVIEW_CASES/Agentic_AI_System/LLM_Safety_Evaluation/release_evaluation_platform.md). Reading units:

- [Interview case: Release-gating safety and quality evaluation](../INTERVIEW_CASES/Agentic_AI_System/LLM_Safety_Evaluation/release_evaluation_platform.md#interview-case-release-gating-safety-and-quality-evaluation)

#### REF-IR009: Allocate a domain-training budget with restart headroom

**Proposed brief:** `PLAN-2026-IR009`. **Mechanisms to develop:** Scaling-law proxy sweeps; licensed corpus splits; parallelism; memory accounting; checkpoint recovery.

**Evaluation to design:** Held-out domain and general quality, full training cost, achieved utilization and restart/reproducibility drills.

**Transfer boundary:** The reference compute and GPU figures are assumptions; an internal legal checkpoint is not validated legal advice.

**Destination indexes:** [24_distributed_training_data_pipelines_and_ml_compute](05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute/INDEX.md); [19_transformers_llms_and_foundation_model_internals](04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/INDEX.md); [21_finetuning_alignment_and_model_customisation](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md).

**Example reference:** [Agentic_AI_System/LLM_TrainingSCALING/legal_pretraining_scaling.md](../INTERVIEW_CASES/Agentic_AI_System/LLM_TrainingSCALING/legal_pretraining_scaling.md). Reading units:

- [Interview case: Compute-budgeted legal-domain pretraining](../INTERVIEW_CASES/Agentic_AI_System/LLM_TrainingSCALING/legal_pretraining_scaling.md#interview-case-compute-budgeted-legal-domain-pretraining)

#### REF-IR010: Compare specialized models with authorized shared-model baselines

**Proposed brief:** `PLAN-2026-IR010`. **Mechanisms to develop:** Fill-in-the-middle; repository/time splits; authorized code retrieval; multilingual sampling; language adapters.

**Evaluation to design:** Sandboxed functional correctness and latency for code; native-reviewed grounding and worst-language regressions for support.

**Transfer boundary:** Split this reference into two independent future narratives; code-test passage does not prove security or language accuracy.

**Destination indexes:** [33_coding_agents_ai_sdcl_and_developer_productivity](06_llm_agents_knowledge_systems_and_ai_augmented_development/33_coding_agents_ai_sdcl_and_developer_productivity/INDEX.md); [18_nlp_speech_and_document_intelligence](04_deep_learning_generative_ai_and_multimodal_systems/18_nlp_speech_and_document_intelligence/INDEX.md); [21_finetuning_alignment_and_model_customisation](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md).

**Example reference:** [Agentic_AI_System/Multimodal/Code_And_Multilingual_Models.md](../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Code_And_Multilingual_Models.md). Reading units:

- [Case A: repository-aware code completion](../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Code_And_Multilingual_Models.md#case-a-repository-aware-code-completion)
- [Case B: multilingual insurance support](../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Code_And_Multilingual_Models.md#case-b-multilingual-insurance-support)

#### REF-IR011: Compare document VLMs with OCR for adjuster assistance

**Proposed brief:** `PLAN-2026-IR011`. **Mechanisms to develop:** Page and region provenance; thumbnail/tiling tradeoffs; schema validation; claim-level temporal splits.

**Evaluation to design:** Field accuracy and calibration by template, handwriting and scan quality; reviewer workload, evidence coverage and latency.

**Transfer boundary:** Neither a similarity match nor a VLM fraud suspicion justifies denial; source scale and patch arithmetic need fresh sizing.

**Destination indexes:** [22_multimodal_models_video_voice_and_world_models](04_deep_learning_generative_ai_and_multimodal_systems/22_multimodal_models_video_voice_and_world_models/INDEX.md); [18_nlp_speech_and_document_intelligence](04_deep_learning_generative_ai_and_multimodal_systems/18_nlp_speech_and_document_intelligence/INDEX.md); [02_claims_fraud_customer_and_operations](11_data_science_domain_specialisations/10_insurance_actuarial_and_claims/02_claims_fraud_customer_and_operations/INDEX.md).

**Example reference:** [Agentic_AI_System/Multimodal/Insurance_Document_VLM.md](../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Insurance_Document_VLM.md). Reading units:

- [Insurance claims: vision-language document understanding](../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Insurance_Document_VLM.md#insurance-claims-vision-language-document-understanding)

#### REF-IR012: Preserve answer evidence through ingestion and filtered retrieval

**Proposed brief:** `PLAN-2026-IR012`. **Mechanisms to develop:** Structure-aware chunking; actual tokenizer limits; HNSW and exact filtered baselines; CDC; deletion propagation.

**Evaluation to design:** All-required-span success, filtered recall, duplicate context, ACL violations, freshness lag and per-tenant latency.

**Transfer boundary:** Reranking cannot recover a missing candidate; the source embedding limit and tenant count are not platform requirements.

**Destination indexes:** [20_rag_context_engineering_and_enterprise_knowledge](04_deep_learning_generative_ai_and_multimodal_systems/20_rag_context_engineering_and_enterprise_knowledge/INDEX.md); [06_data_modeling_databases_lakehouse_and_governance](02_data_analytics_and_decision_science/06_data_modeling_databases_lakehouse_and_governance/INDEX.md); [03_enterprise_knowledge_support_and_sales](11_data_science_domain_specialisations/14_saas_telecom_and_b2b/03_enterprise_knowledge_support_and_sales/INDEX.md).

**Example reference:** [Agentic_AI_System/RAG_Infra_Research/Chunking_And_Vector_Index.md](../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Chunking_And_Vector_Index.md). Reading units:

- [Enterprise retrieval: structure-aware chunking and filtered vector search](../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Chunking_And_Vector_Index.md#enterprise-retrieval-structure-aware-chunking-and-filtered-vector-search)

#### REF-IR013: Retrieve regulations applicable to the question date and jurisdiction

**Proposed brief:** `PLAN-2026-IR013`. **Mechanisms to develop:** Effective intervals; recorded-time history; jurisdiction rules; hybrid retrieval; reranking and claim verification.

**Evaluation to design:** Boundary-date and amendment replays, eligible recall, wrong-version citations and expert-reviewed abstentions.

**Transfer boundary:** This is an architecture exercise, not current legal guidance; eligibility depends on maintained authoritative metadata.

**Destination indexes:** [20_rag_context_engineering_and_enterprise_knowledge](04_deep_learning_generative_ai_and_multimodal_systems/20_rag_context_engineering_and_enterprise_knowledge/INDEX.md); [50_responsible_ai_governance_compliance_and_audits](09_trust_security_governance_and_product_leadership/50_responsible_ai_governance_compliance_and_audits/INDEX.md); [03_insurance_mlops_model_risk_and_climate](11_data_science_domain_specialisations/10_insurance_actuarial_and_claims/03_insurance_mlops_model_risk_and_climate/INDEX.md).

**Example reference:** [Agentic_AI_System/RAG_Infra_Research/Regulatory_RAG_And_Reranking.md](../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Regulatory_RAG_And_Reranking.md). Reading units:

- [Regulatory RAG with temporal eligibility and evidence verification](../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Regulatory_RAG_And_Reranking.md#regulatory-rag-with-temporal-eligibility-and-evidence-verification)

#### REF-IR014: Separate extraction correctness from routing economics

**Proposed brief:** `PLAN-2026-IR014`. **Mechanisms to develop:** Schema versioning; semantic checks; dead-letter replay; idempotency; quality-constrained tiered RAG routing.

**Evaluation to design:** Gold field accuracy and reconciliation for extraction; full-path cost and grounded quality by tier for routing.

**Transfer boundary:** Contains two reference problems: author separate narratives; valid JSON and cheap token prices do not establish successful outcomes.

**Destination indexes:** [31_tools_mcp_protocols_and_enterprise_integrations](06_llm_agents_knowledge_systems_and_ai_augmented_development/31_tools_mcp_protocols_and_enterprise_integrations/INDEX.md); [25_model_serving_inference_optimisation_and_cost](05_ai_platforms_data_systems_and_production_engineering/25_model_serving_inference_optimisation_and_cost/INDEX.md); [02_claims_fraud_customer_and_operations](11_data_science_domain_specialisations/10_insurance_actuarial_and_claims/02_claims_fraud_customer_and_operations/INDEX.md).

**Example reference:** [Agentic_AI_System/RAG_Infra_Research/Structured_Extraction_And_Cost_Routing.md](../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Structured_Extraction_And_Cost_Routing.md). Reading units:

- [Case A: claim-note extraction into an operational API](../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Structured_Extraction_And_Cost_Routing.md#case-a-claim-note-extraction-into-an-operational-api)
- [Case B: tiered RAG cost and accuracy](../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Structured_Extraction_And_Cost_Routing.md#case-b-tiered-rag-cost-and-accuracy)

#### REF-IR015: Publish reviewed coaching scripts from current product evidence

**Proposed brief:** `PLAN-2026-IR015`. **Mechanisms to develop:** Product/date eligibility; grounded generation; native-language review; source dependencies; supersession triggers.

**Evaluation to design:** Unsupported product claims, expired evidence, read-aloud time, reviewer agreement and agent usability.

**Transfer boundary:** Human approval and source maintenance remain necessary; the example word target is not a compliance standard.

**Destination indexes:** [34_conversational_ai_voice_agents_and_customer_operations](06_llm_agents_knowledge_systems_and_ai_augmented_development/34_conversational_ai_voice_agents_and_customer_operations/INDEX.md); [03_enterprise_knowledge_support_and_sales](11_data_science_domain_specialisations/14_saas_telecom_and_b2b/03_enterprise_knowledge_support_and_sales/INDEX.md); [02_claims_fraud_customer_and_operations](11_data_science_domain_specialisations/10_insurance_actuarial_and_claims/02_claims_fraud_customer_and_operations/INDEX.md).

**Example reference:** [Agentic_AI_System/Speech_STT/Approved_Objection_Response_Scripts.md](../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Approved_Objection_Response_Scripts.md). Reading units:

- [Grounded and approved objection-response scripts](../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Approved_Objection_Response_Scripts.md#grounded-and-approved-objection-response-scripts)

#### REF-IR016: Distinguish speech, speaker and extraction errors in call analytics

**Proposed brief:** `PLAN-2026-IR016`. **Mechanisms to develop:** Channel-aware ASR; VAD; code-switching; speaker roles; redaction; evidence spans; cohort aggregation.

**Evaluation to design:** Human-gold WER, domain-term WER, role error, objection precision/recall and end-to-end report coverage.

**Transfer boundary:** Transcript matching does not prove audio fidelity; aggregate coaching themes must not become unvalidated individual scores.

**Destination indexes:** [18_nlp_speech_and_document_intelligence](04_deep_learning_generative_ai_and_multimodal_systems/18_nlp_speech_and_document_intelligence/INDEX.md); [34_conversational_ai_voice_agents_and_customer_operations](06_llm_agents_knowledge_systems_and_ai_augmented_development/34_conversational_ai_voice_agents_and_customer_operations/INDEX.md); [02_claims_fraud_customer_and_operations](11_data_science_domain_specialisations/10_insurance_actuarial_and_claims/02_claims_fraud_customer_and_operations/INDEX.md).

**Example reference:** [Agentic_AI_System/Speech_STT/Multilingual_Call_Objections.md](../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Multilingual_Call_Objections.md). Reading units:

- [Multilingual sales-call transcription and objection mining](../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Multilingual_Call_Objections.md#multilingual-sales-call-transcription-and-objection-mining)

#### REF-IR017: Adapt domain ASR while preserving supported-domain quality

**Proposed brief:** `PLAN-2026-IR017`. **Mechanisms to develop:** Whisper adaptation; LoRA and general replay; speaker/call/time splits; independent human gold; scoped deployment.

**Evaluation to design:** Paired WER with uncertainty by domain and code-switch slice, real-time factor and supported-route regression gates.

**Transfer boundary:** Model-generated transcripts are not independent gold; audio resampling does not restore missing frequencies.

**Destination indexes:** [21_finetuning_alignment_and_model_customisation](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md); [18_nlp_speech_and_document_intelligence](04_deep_learning_generative_ai_and_multimodal_systems/18_nlp_speech_and_document_intelligence/INDEX.md); [34_conversational_ai_voice_agents_and_customer_operations](06_llm_agents_knowledge_systems_and_ai_augmented_development/34_conversational_ai_voice_agents_and_customer_operations/INDEX.md).

**Example reference:** [Agentic_AI_System/Speech_STT/Whisper_Adaptation_And_Evaluation.md](../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Whisper_Adaptation_And_Evaluation.md). Reading units:

- [Domain speech adaptation without cross-domain regression](../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Whisper_Adaptation_And_Evaluation.md#domain-speech-adaptation-without-cross-domain-regression)

#### REF-IR018: Specify correctness contracts before choosing production algorithms

**Proposed brief:** `PLAN-2026-IR018`. **Mechanisms to develop:** CH and A* assumptions; rolling-window semantics; clock skew; positional postings; BM25 statistics.

**Evaluation to design:** Reference Dijkstra comparisons, admissible-heuristic checks, concurrent window simulations and brute-force phrase/ranking oracles.

**Transfer boundary:** Seven overlapping teaching units; source measurements are not independently verified and should not be added together.

**Destination indexes:** [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md); [04_system_design_distributed_systems_and_cloud](01_foundations_computing_and_problem_solving/04_system_design_distributed_systems_and_cloud/INDEX.md); [14_recommendation_search_ads_and_ranking](03_machine_learning_optimisation_and_intelligent_retrieval/14_recommendation_search_ads_and_ranking/INDEX.md); [01_mobility_dispatch_eta_and_pricing](11_data_science_domain_specialisations/13_mobility_travel_and_hospitality/01_mobility_dispatch_eta_and_pricing/INDEX.md).

**Example reference:** [DSA/Algo_Design_Problems/CASE_STUDIES_01.md](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_01.md). Reading units:

- [1. Ride-hailing routes under changing road conditions](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_01.md#1-ride-hailing-routes-under-changing-road-conditions)
- [2. A* with a provably safe heuristic](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_01.md#2-a-with-a-provably-safe-heuristic)
- [3. Sliding-window admission at high throughput](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_01.md#3-sliding-window-admission-at-high-throughput)
- [4. Limiting safely when Redis fails](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_01.md#4-limiting-safely-when-redis-fails)
- [5. Clock skew in distributed rate windows](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_01.md#5-clock-skew-in-distributed-rate-windows)
- [6. Phrase search, positional storage and compressed postings](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_01.md#6-phrase-search-positional-storage-and-compressed-postings)
- [7. BM25 relevance under corrupt index statistics](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_01.md#7-bm25-relevance-under-corrupt-index-statistics)

#### REF-IR019: Keep routing correctness separate from ETA quality

**Proposed brief:** `PLAN-2026-IR019`. **Mechanisms to develop:** Coherent traffic snapshots; CH compatibility; fallback Dijkstra; map matching; route versus ETA objectives.

**Evaluation to design:** Base-graph shortest-path oracle, closure and turn-restriction replays, ETA error and routing tail latency.

**Transfer boundary:** The OCR filename and ordinal ranges disagree; arbitrary traffic overlays do not preserve ordinary CH correctness.

**Destination indexes:** [04_system_design_distributed_systems_and_cloud](01_foundations_computing_and_problem_solving/04_system_design_distributed_systems_and_cloud/INDEX.md); [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md); [01_mobility_dispatch_eta_and_pricing](11_data_science_domain_specialisations/13_mobility_travel_and_hospitality/01_mobility_dispatch_eta_and_pricing/INDEX.md); [02_dispatch_eta_and_delivery_operations](11_data_science_domain_specialisations/06_quick_commerce_food_delivery_and_on_demand/02_dispatch_eta_and_delivery_operations/INDEX.md); [02_transport_network_warehouse_and_last_mile](11_data_science_domain_specialisations/07_supply_chain_logistics_warehousing_and_fulfillment/02_transport_network_warehouse_and_last_mile/INDEX.md).

**Example reference:** [DSA/Algo_Design_Problems/CASE_STUDIES_02.md](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_02.md). Reading units:

- [Case study 02 — Dynamic road routing and ETA](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_02.md#case-study-02--dynamic-road-routing-and-eta)

#### REF-IR020: Translate local algorithm guarantees into distributed service contracts

**Proposed brief:** `PLAN-2026-IR020`. **Mechanisms to develop:** Routing production behavior; coordinated rolling counters; distributed positional inverted indexes.

**Evaluation to design:** Coherent-snapshot queries, concurrent admission accounting, shard failure tests and search-reference comparisons.

**Transfer boundary:** Three units continue earlier routing, admission and search topics; this is not a new independent evidence dataset.

**Destination indexes:** [54_ml_and_ai_system_design_interviews](10_architecture_practice_and_portfolio_capstones/54_ml_and_ai_system_design_interviews/INDEX.md); [04_system_design_distributed_systems_and_cloud](01_foundations_computing_and_problem_solving/04_system_design_distributed_systems_and_cloud/INDEX.md); [14_recommendation_search_ads_and_ranking](03_machine_learning_optimisation_and_intelligent_retrieval/14_recommendation_search_ads_and_ranking/INDEX.md).

**Example reference:** [DSA/Algo_Design_Problems/CASE_STUDIES_03.md](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_03.md). Reading units:

- [1. Navigation routing: production behavior around shortest paths](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_03.md#1-navigation-routing-production-behavior-around-shortest-paths)
- [2. Distributed rate limiting with rolling counters](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_03.md#2-distributed-rate-limiting-with-rolling-counters)
- [3. Search over a distributed positional inverted index](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_03.md#3-search-over-a-distributed-positional-inverted-index)

#### REF-IR021: Evolve an authorized near-real-time search index safely

**Proposed brief:** `PLAN-2026-IR021`. **Mechanisms to develop:** WAL and immutable segments; positional phrase queries; field-weighted BM25; exact top-k; ACLs and tombstones.

**Evaluation to design:** Brute-force Boolean/phrase and global top-k oracles, delete visibility, retrieval quality, index lag and merge capacity.

**Transfer boundary:** Exact global top-k needs a valid distributed protocol; figures in the source are scenario claims, not search-cluster benchmarks.

**Destination indexes:** [14_recommendation_search_ads_and_ranking](03_machine_learning_optimisation_and_intelligent_retrieval/14_recommendation_search_ads_and_ranking/INDEX.md); [06_data_modeling_databases_lakehouse_and_governance](02_data_analytics_and_decision_science/06_data_modeling_databases_lakehouse_and_governance/INDEX.md); [01_catalogue_search_discovery_and_conversion](11_data_science_domain_specialisations/05_ecommerce_marketplaces_and_digital_commerce/01_catalogue_search_discovery_and_conversion/INDEX.md).

**Example reference:** [DSA/Algo_Design_Problems/CASE_STUDIES_04.md](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_04.md). Reading units:

- [1. Build a near-real-time inverted index without blocking readers](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_04.md#1-build-a-near-real-time-inverted-index-without-blocking-readers)
- [2. Boolean and phrase search with field-weighted BM25](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_04.md#2-boolean-and-phrase-search-with-field-weighted-bm25)
- [3. Exact distributed top-k with ACLs and deletes](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_04.md#3-exact-distributed-top-k-with-acls-and-deletes)
- [4. Search capacity, observability and safe evolution](../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_04.md#4-search-capacity-observability-and-safe-evolution)

#### REF-IR022: Bound global API admission with token leases

**Proposed brief:** `PLAN-2026-IR022`. **Mechanisms to develop:** Token-bucket invariants; atomic refill and debit; leased budgets; expiry; failure policy.

**Evaluation to design:** Aggregate-admission histories, refill boundaries, coordinator outages, lease races and rejected-request latency.

**Transfer boundary:** Periodic usage reporting alone is not coordinated allocation; token buckets and exact rolling windows have different contracts.

**Destination indexes:** [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md); [29_api_platform_engineering_cloud_native_and_finops](05_ai_platforms_data_systems_and_production_engineering/29_api_platform_engineering_cloud_native_and_finops/INDEX.md); [04_system_design_distributed_systems_and_cloud](01_foundations_computing_and_problem_solving/04_system_design_distributed_systems_and_cloud/INDEX.md).

**Example reference:** [DSA/Algo_Design_Problems/Rate_Limiting/distributed_token_buckets.md](../INTERVIEW_CASES/DSA/Algo_Design_Problems/Rate_Limiting/distributed_token_buckets.md). Reading units:

- [Interview case: Distributed API rate limiting under contention](../INTERVIEW_CASES/DSA/Algo_Design_Problems/Rate_Limiting/distributed_token_buckets.md#interview-case-distributed-api-rate-limiting-under-contention)

#### REF-IR023: Reject stale shortcuts without losing a shortest-path guarantee

**Proposed brief:** `PLAN-2026-IR023`. **Mechanisms to develop:** Shortcut expansion; closure snapshots; CH compatibility; directed nonnegative Dijkstra fallback.

**Evaluation to design:** Nested-shortcut and adversarial-closure queries compared with base Dijkstra, including unreachable pairs.

**Transfer boundary:** Finding a legal route after filtering does not prove completeness or optimality; richer time/turn state needs a separate contract.

**Destination indexes:** [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md); [01_mobility_dispatch_eta_and_pricing](11_data_science_domain_specialisations/13_mobility_travel_and_hospitality/01_mobility_dispatch_eta_and_pricing/INDEX.md); [02_dispatch_eta_and_delivery_operations](11_data_science_domain_specialisations/06_quick_commerce_food_delivery_and_on_demand/02_dispatch_eta_and_delivery_operations/INDEX.md); [02_transport_network_warehouse_and_last_mile](11_data_science_domain_specialisations/07_supply_chain_logistics_warehousing_and_fulfillment/02_transport_network_warehouse_and_last_mile/INDEX.md).

**Example reference:** [DSA/Algo_Design_Problems/Routing/closure_aware_shortest_paths.md](../INTERVIEW_CASES/DSA/Algo_Design_Problems/Routing/closure_aware_shortest_paths.md). Reading units:

- [Interview case: Route around a newly closed road](../INTERVIEW_CASES/DSA/Algo_Design_Problems/Routing/closure_aware_shortest_paths.md#interview-case-route-around-a-newly-closed-road)

#### REF-IR024: Bound phrase-search memory while preserving exact matches

**Proposed brief:** `PLAN-2026-IR024`. **Mechanisms to develop:** Rarest-term anchor; lazy postings; skips; relative positions; repeated terms; matching analyzers.

**Evaluation to design:** Brute-force token-position oracle, peak memory and tail latency over frequent and repeated-term phrases.

**Transfer boundary:** Rarity affects work, not semantic order; deleted documents and analyzer versions must share a consistent snapshot.

**Destination indexes:** [14_recommendation_search_ads_and_ranking](03_machine_learning_optimisation_and_intelligent_retrieval/14_recommendation_search_ads_and_ranking/INDEX.md); [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md); [01_catalogue_search_discovery_and_conversion](11_data_science_domain_specialisations/05_ecommerce_marketplaces_and_digital_commerce/01_catalogue_search_discovery_and_conversion/INDEX.md).

**Example reference:** [DSA/Algo_Design_Problems/Search/phrase_search_postings.md](../INTERVIEW_CASES/DSA/Algo_Design_Problems/Search/phrase_search_postings.md). Reading units:

- [Interview case: Phrase search without loading common-term postings](../INTERVIEW_CASES/DSA/Algo_Design_Problems/Search/phrase_search_postings.md#interview-case-phrase-search-without-loading-common-term-postings)

#### REF-IR025: Choose scheduling algorithms from objective and constraints

**Proposed brief:** `PLAN-2026-IR025`. **Mechanisms to develop:** Half-open intervals; deterministic ties; exchange proof; weighted interval DP boundary.

**Evaluation to design:** Exhaustive small-instance optimum comparisons and counterexamples for weighted or buffered variants.

**Transfer boundary:** The reference proves equal-value single-room cardinality only; warehouse jobs and staffing need their own constraints.

**Destination indexes:** [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md); [13_optimization_operations_research_and_simulation](03_machine_learning_optimisation_and_intelligent_retrieval/13_optimization_operations_research_and_simulation/INDEX.md).

**Example reference:** [DSA/index_basics/Greedy_Scheduling/activity_selection.md](../INTERVIEW_CASES/DSA/index_basics/Greedy_Scheduling/activity_selection.md). Reading units:

- [Interview case: Selecting the most meetings for one room](../INTERVIEW_CASES/DSA/index_basics/Greedy_Scheduling/activity_selection.md#interview-case-selecting-the-most-meetings-for-one-room)

#### REF-IR026: Select data structures from operation mix and failure semantics

**Proposed brief:** `PLAN-2026-IR026`. **Mechanisms to develop:** Cache policies; ranks/range queries; sketches; prefix/nearest-neighbor search; graphs; scheduling; online windows.

**Evaluation to design:** Operation traces, exact small-data oracles, measured memory/latency and stated approximation or distributed guarantees.

**Transfer boundary:** Eight teaching units plus rapid prompts; approximate structures, local proofs and service guarantees must be distinguished.

**Destination indexes:** [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md); [54_ml_and_ai_system_design_interviews](10_architecture_practice_and_portfolio_capstones/54_ml_and_ai_system_design_interviews/INDEX.md); [01_ml_system_design_production](11_data_science_domain_specialisations/03_practical_ml_design_and_interviews/01_ml_system_design_production/INDEX.md).

**Example reference:** [DSA/index_basics/INTERVIEW_CASE_STUDIES.md](../INTERVIEW_CASES/DSA/index_basics/INTERVIEW_CASE_STUDIES.md). Reading units:

- [Case 1 — A catalog cache under changing access patterns](../INTERVIEW_CASES/DSA/index_basics/INTERVIEW_CASE_STUDIES.md#case-1--a-catalog-cache-under-changing-access-patterns)
- [Case 2 — Mutable measurements, ordered searches, and ranks](../INTERVIEW_CASES/DSA/index_basics/INTERVIEW_CASE_STUDIES.md#case-2--mutable-measurements-ordered-searches-and-ranks)
- [Case 3 — Deduplication and approximate streaming analytics](../INTERVIEW_CASES/DSA/index_basics/INTERVIEW_CASE_STUDIES.md#case-3--deduplication-and-approximate-streaming-analytics)
- [Case 4 — Prefix search, text search, and nearest neighbors](../INTERVIEW_CASES/DSA/index_basics/INTERVIEW_CASE_STUDIES.md#case-4--prefix-search-text-search-and-nearest-neighbors)
- [Case 5 — Routing, connectivity, and dependency graphs](../INTERVIEW_CASES/DSA/index_basics/INTERVIEW_CASE_STUDIES.md#case-5--routing-connectivity-and-dependency-graphs)
- [Case 6 — Scheduling: greedy proof versus dynamic programming](../INTERVIEW_CASES/DSA/index_basics/INTERVIEW_CASE_STUDIES.md#case-6--scheduling-greedy-proof-versus-dynamic-programming)
- [Case 7 — Online windows, selection, sorting, and trees](../INTERVIEW_CASES/DSA/index_basics/INTERVIEW_CASE_STUDIES.md#case-7--online-windows-selection-sorting-and-trees)
- [Case 8 — From local algorithm to distributed service](../INTERVIEW_CASES/DSA/index_basics/INTERVIEW_CASE_STUDIES.md#case-8--from-local-algorithm-to-distributed-service)

#### REF-IR027: Maintain exact inventory range totals under point updates

**Proposed brief:** `PLAN-2026-IR027`. **Mechanisms to develop:** Fenwick tree lowbit traversal; inclusive ranges; point deltas; wide arithmetic; atomic visibility.

**Evaluation to design:** Randomized update/query comparisons with a simple array, boundary/overflow checks and update-visibility tests.

**Transfer boundary:** Fixed ordering and one logical writer are reference assumptions; a Fenwick tree alone does not provide distributed stock correctness.

**Destination indexes:** [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md); [01_planning_forecasting_inventory_and_sourcing](11_data_science_domain_specialisations/07_supply_chain_logistics_warehousing_and_fulfillment/01_planning_forecasting_inventory_and_sourcing/INDEX.md); [01_quick_commerce_inventory_and_assortment](11_data_science_domain_specialisations/06_quick_commerce_food_delivery_and_on_demand/01_quick_commerce_inventory_and_assortment/INDEX.md).

**Example reference:** [DSA/index_basics/Range_Queries/fenwick_inventory_totals.md](../INTERVIEW_CASES/DSA/index_basics/Range_Queries/fenwick_inventory_totals.md). Reading units:

- [Interview case: Live inventory range totals](../INTERVIEW_CASES/DSA/index_basics/Range_Queries/fenwick_inventory_totals.md#interview-case-live-inventory-range-totals)

#### REF-IR028: Bound request amplification before expensive parsing and validation

**Proposed brief:** `PLAN-2026-IR028`. **Mechanisms to develop:** Hash collision amplification; catastrophic regex backtracking; pre-parse limits; bounded queues and downstream budgets.

**Evaluation to design:** Adversarial near-match inputs, collision workload, CPU per request, timeout cascades and legitimate-input acceptance.

**Transfer boundary:** The hash and regex scenarios are separate; legacy runtime behavior and example input limits require fresh validation.

**Destination indexes:** [55_incident_response_failure_modes_and_technical_leadership](10_architecture_practice_and_portfolio_capstones/55_incident_response_failure_modes_and_technical_leadership/INDEX.md); [47_cybersecurity_security_operations_and_ai_defense](08_business_domains_and_decision_intelligence/47_cybersecurity_security_operations_and_ai_defense/INDEX.md); [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md).

**Example reference:** [DSA/Production_DSA_Incidents/Algorithmic_DoS/hash_and_regex_resource_exhaustion.md](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Algorithmic_DoS/hash_and_regex_resource_exhaustion.md). Reading units:

- [Case: valid requests saturate application CPU](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Algorithmic_DoS/hash_and_regex_resource_exhaustion.md#case-valid-requests-saturate-application-cpu)

#### REF-IR029: Return useful graph candidates under explicit traversal budgets

**Proposed brief:** `PLAN-2026-IR029`. **Mechanisms to develop:** Enqueue-time deduplication; paged adjacency; independent work budgets; deterministic traversal; explicit partial results.

**Evaluation to design:** High-degree and cyclic graph replays, peak memory, examined edges, output completeness and candidate utility.

**Transfer boundary:** Depth and output-size limits alone do not bound expansion; source product degrees and outage figures are illustrative claims.

**Destination indexes:** [15_graph_ml_knowledge_graphs_and_network_analytics](03_machine_learning_optimisation_and_intelligent_retrieval/15_graph_ml_knowledge_graphs_and_network_analytics/INDEX.md); [14_recommendation_search_ads_and_ranking](03_machine_learning_optimisation_and_intelligent_retrieval/14_recommendation_search_ads_and_ranking/INDEX.md); [55_incident_response_failure_modes_and_technical_leadership](10_architecture_practice_and_portfolio_capstones/55_incident_response_failure_modes_and_technical_leadership/INDEX.md); [01_catalogue_search_discovery_and_conversion](11_data_science_domain_specialisations/05_ecommerce_marketplaces_and_digital_commerce/01_catalogue_search_discovery_and_conversion/INDEX.md).

**Example reference:** [DSA/Production_DSA_Incidents/Graph_Traversal/bounded_bfs_incident.md](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Graph_Traversal/bounded_bfs_incident.md). Reading units:

- [Case: recommendations crash on a high-degree product](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Graph_Traversal/bounded_bfs_incident.md#case-recommendations-crash-on-a-high-degree-product)

#### REF-IR030: Diagnose production failures from violated algorithm invariants

**Proposed brief:** `PLAN-2026-IR030`. **Mechanisms to develop:** Hash flooding; BFS OOM; regex backtracking; recursive DFS depth; ORM N+1; overflow; unstable ties; floating search termination.

**Evaluation to design:** Reproduce each mechanism separately with bounded adversarial traces; track latency, memory, database calls and deterministic outputs.

**Transfer boundary:** Ten reconstructed incident units include recurring mechanisms; reported timings are not measured outcomes of this library.

**Destination indexes:** [55_incident_response_failure_modes_and_technical_leadership](10_architecture_practice_and_portfolio_capstones/55_incident_response_failure_modes_and_technical_leadership/INDEX.md); [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md); [03_evaluation_reliability_and_incidents](11_data_science_domain_specialisations/03_practical_ml_design_and_interviews/03_evaluation_reliability_and_incidents/INDEX.md).

**Example reference:** [DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md). Reading units:

- [1. Hash flooding across services (INC-001)](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md#1-hash-flooding-across-services-inc-001)
- [2. Product graph traversal exhausts pod memory (INC-002)](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md#2-product-graph-traversal-exhausts-pod-memory-inc-002)
- [3. Email validation catastrophic backtracking (INC-003)](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md#3-email-validation-catastrophic-backtracking-inc-003)
- [4. Deep policy JSON breaks recursive DFS (INC-004)](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md#4-deep-policy-json-breaks-recursive-dfs-inc-004)
- [5. ORM tree walk issues N+1 queries (INC-005)](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md#5-orm-tree-walk-issues-n1-queries-inc-005)
- [6. Distance overflow and unit analysis (INC-006)](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md#6-distance-overflow-and-unit-analysis-inc-006)
- [7. Equal-score sort changes premium (INC-007)](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md#7-equal-score-sort-changes-premium-inc-007)
- [8. Float convergence hangs binary search (INC-008)](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md#8-float-convergence-hangs-binary-search-inc-008)
- [9. HTTP hash-collision postmortem: edge to downstream cascade](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md#9-http-hash-collision-postmortem-edge-to-downstream-cascade)
- [10. Social-graph celebrity user triggers recurring BFS OOM](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md#10-social-graph-celebrity-user-triggers-recurring-bfs-oom)

#### REF-IR031: Load authorized hierarchies with bounded database work

**Proposed brief:** `PLAN-2026-IR031`. **Mechanisms to develop:** N+1 detection; adjacency-list indexing; per-step authorization; cycle checks; stable assembly; snapshot reads.

**Evaluation to design:** Reference traversal comparisons, round trips, rows scanned, unauthorized descendants, deep/wide trees and truncation behavior.

**Transfer boundary:** One SQL statement can still do excessive work; an outer LIMIT is not a universal recursion work bound.

**Destination indexes:** [06_data_modeling_databases_lakehouse_and_governance](02_data_analytics_and_decision_science/06_data_modeling_databases_lakehouse_and_governance/INDEX.md); [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md); [02_claims_fraud_customer_and_operations](11_data_science_domain_specialisations/10_insurance_actuarial_and_claims/02_claims_fraud_customer_and_operations/INDEX.md).

**Example reference:** [DSA/Production_DSA_Incidents/Tree_Queries/policy_hierarchy_n_plus_one.md](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Tree_Queries/policy_hierarchy_n_plus_one.md). Reading units:

- [Case: a policy hierarchy turns one request into hundreds of queries](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Tree_Queries/policy_hierarchy_n_plus_one.md#case-a-policy-hierarchy-turns-one-request-into-hundreds-of-queries)

#### REF-IR032: Balance autocomplete freshness, memory and abuse controls

**Proposed brief:** `PLAN-2026-IR032`. **Mechanisms to develop:** Trie admission; time-window sketches; bounded candidate stores; immutable snapshots; real-time overlays; CDN versioning.

**Evaluation to design:** Memory including two generations, suggestion relevance, freshness lag, abusive-query tests and per-locale tail latency.

**Transfer boundary:** Sketches cannot enumerate terms; approximate frequencies and novelty thresholds must be evaluated rather than copied.

**Destination indexes:** [14_recommendation_search_ads_and_ranking](03_machine_learning_optimisation_and_intelligent_retrieval/14_recommendation_search_ads_and_ranking/INDEX.md); [01_catalogue_search_discovery_and_conversion](11_data_science_domain_specialisations/05_ecommerce_marketplaces_and_digital_commerce/01_catalogue_search_discovery_and_conversion/INDEX.md).

**Example reference:** [DSA/Sys_Des_DSA/Autocomplete/frequency_admission_and_freshness.md](../INTERVIEW_CASES/DSA/Sys_Des_DSA/Autocomplete/frequency_admission_and_freshness.md). Reading units:

- [Case: autocomplete grows until it OOMs, then misses breaking news](../INTERVIEW_CASES/DSA/Sys_Des_DSA/Autocomplete/frequency_admission_and_freshness.md#case-autocomplete-grows-until-it-ooms-then-misses-breaking-news)

#### REF-IR033: Rebalance caches without overwhelming the authoritative store

**Proposed brief:** `PLAN-2026-IR033`. **Mechanisms to develop:** Single-flight scope; jitter; fill admission; consistent hashing; old/new ring epochs; version fences and tombstones.

**Evaluation to design:** Cold-fleet and skewed-key load replays, database concurrency, invalidation lag and stale-fill/delete races.

**Transfer boundary:** Local single-flight is not fleet-wide coordination; expected key movement is not the same as request miss share.

**Destination indexes:** [04_system_design_distributed_systems_and_cloud](01_foundations_computing_and_problem_solving/04_system_design_distributed_systems_and_cloud/INDEX.md); [27_data_quality_observability_and_reliability_engineering](05_ai_platforms_data_systems_and_production_engineering/27_data_quality_observability_and_reliability_engineering/INDEX.md); [03_digital_commerce_data_platform_and_operations](11_data_science_domain_specialisations/05_ecommerce_marketplaces_and_digital_commerce/03_digital_commerce_data_platform_and_operations/INDEX.md).

**Example reference:** [DSA/Sys_Des_DSA/Cache_Resilience/cold_cache_and_ring_migration.md](../INTERVIEW_CASES/DSA/Sys_Des_DSA/Cache_Resilience/cold_cache_and_ring_migration.md). Reading units:

- [Case: cache restart and scale-out overload the database](../INTERVIEW_CASES/DSA/Sys_Des_DSA/Cache_Resilience/cold_cache_and_ring_migration.md#case-cache-restart-and-scale-out-overload-the-database)

#### REF-IR034: Design stateful serving structures with explicit consistency

**Proposed brief:** `PLAN-2026-IR034`. **Mechanisms to develop:** Profile caching; safe typeahead publication; ranked score updates; Bloom-filter crawler admission.

**Evaluation to design:** Failure replays and reference histories; freshness, exact versus approximate results and false-positive consequences.

**Transfer boundary:** Four teaching units overlap focused examples; Bloom false positives and stale ranks need product-specific tolerance.

**Destination indexes:** [04_system_design_distributed_systems_and_cloud](01_foundations_computing_and_problem_solving/04_system_design_distributed_systems_and_cloud/INDEX.md); [54_ml_and_ai_system_design_interviews](10_architecture_practice_and_portfolio_capstones/54_ml_and_ai_system_design_interviews/INDEX.md).

**Example reference:** [DSA/Sys_Des_DSA/INTERVIEW_SYSTEM_DESIGN_CASES.md](../INTERVIEW_CASES/DSA/Sys_Des_DSA/INTERVIEW_SYSTEM_DESIGN_CASES.md). Reading units:

- [1. Social profile cache and incident response](../INTERVIEW_CASES/DSA/Sys_Des_DSA/INTERVIEW_SYSTEM_DESIGN_CASES.md#1-social-profile-cache-and-incident-response)
- [2. Search typeahead and safe index refresh](../INTERVIEW_CASES/DSA/Sys_Des_DSA/INTERVIEW_SYSTEM_DESIGN_CASES.md#2-search-typeahead-and-safe-index-refresh)
- [3. Distributed real-time leaderboard](../INTERVIEW_CASES/DSA/Sys_Des_DSA/INTERVIEW_SYSTEM_DESIGN_CASES.md#3-distributed-real-time-leaderboard)
- [4. Crawler URL deduplication with Bloom filters](../INTERVIEW_CASES/DSA/Sys_Des_DSA/INTERVIEW_SYSTEM_DESIGN_CASES.md#4-crawler-url-deduplication-with-bloom-filters)

#### REF-IR035: Separate live rank approximation from exact competition settlement

**Proposed brief:** `PLAN-2026-IR035`. **Mechanisms to develop:** Score buckets; Fenwick count summaries; deterministic ties; deduplicated events; cross-shard moves; closing watermarks.

**Evaluation to design:** Reference sorted snapshots, duplicate/late events, negative scores, bucket crossings and exact final-rank replay.

**Transfer boundary:** A sum of stale counters is not an exact settlement snapshot; tie semantics and late-event policy must be specified.

**Destination indexes:** [04_system_design_distributed_systems_and_cloud](01_foundations_computing_and_problem_solving/04_system_design_distributed_systems_and_cloud/INDEX.md); [03_dsa_and_computational_thinking](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md); [09_data_streaming_event_systems_and_real_time_analytics](02_data_analytics_and_decision_science/09_data_streaming_event_systems_and_real_time_analytics/INDEX.md).

**Example reference:** [DSA/Sys_Des_DSA/Leaderboards/score_bucket_rank_consistency.md](../INTERVIEW_CASES/DSA/Sys_Des_DSA/Leaderboards/score_bucket_rank_consistency.md). Reading units:

- [Case: a large leaderboard needs fast rank and fair prizes](../INTERVIEW_CASES/DSA/Sys_Des_DSA/Leaderboards/score_bucket_rank_consistency.md#case-a-large-leaderboard-needs-fast-rank-and-fair-prizes)

#### REF-IR036: Evaluate incremental investigator value from graph fraud signals

**Proposed brief:** `PLAN-2026-IR036`. **Mechanisms to develop:** Conservative entity resolution; typed claim-actor graphs; motifs and supernodes; point-in-time features; reversible merges.

**Evaluation to design:** Temporal baseline comparison, precision at capacity, mature outcomes, sampled low-risk audits and appeal/false-positive slices.

**Transfer boundary:** Connectivity and shared providers are not proof of fraud; source reference notes are unavailable and labels are delayed/selected.

**Destination indexes:** [15_graph_ml_knowledge_graphs_and_network_analytics](03_machine_learning_optimisation_and_intelligent_retrieval/15_graph_ml_knowledge_graphs_and_network_analytics/INDEX.md); [02_graph_ml](11_data_science_domain_specialisations/01_production_ml_case_library/02_graph_ml/INDEX.md); [01_fraud_modelling](11_data_science_domain_specialisations/01_production_ml_case_library/01_fraud_modelling/INDEX.md); [02_claims_fraud_customer_and_operations](11_data_science_domain_specialisations/10_insurance_actuarial_and_claims/02_claims_fraud_customer_and_operations/INDEX.md); [01_payments_aml_and_financial_crime](11_data_science_domain_specialisations/11_financial_services_fraud_and_risk/01_payments_aml_and_financial_crime/INDEX.md).

**Example reference:** [Graph_Rings/Fraud_Ring_Detection/Claims_Fraud_Ring_Case.md](../INTERVIEW_CASES/Graph_Rings/Fraud_Ring_Detection/Claims_Fraud_Ring_Case.md). Reading units:

- [Case study: Detect coordinated claims fraud with a graph](../INTERVIEW_CASES/Graph_Rings/Fraud_Ring_Detection/Claims_Fraud_Ring_Case.md#case-study-detect-coordinated-claims-fraud-with-a-graph)

## 3. Canonical reusable patterns for these proposed briefs

These 17 mechanism tags apply to this planning subset. Existing collections retain their own vocabularies until an explicit consolidation review. No author should silently introduce synonyms.

| Tag | Definition | Diagnostic signature | Boundary |
|---|---|---|---|
| `distribution-shift` | The operating or retraining population changes | Slice regressions and corpus-mix changes | Do not infer shift from sampling noise alone |
| `data-leakage` | Evaluation sees information unavailable at decision time | Related repositories, calls, documents or future labels cross splits | A legal pre-cutoff feature is not leakage |
| `evaluation-mismatch` | A proxy metric does not establish decision quality | Valid syntax, fluent text or a judge score masks task failure | Use when a concrete decision-level comparator exists |
| `eligibility-before-ranking` | Apply hard constraints before optimizing a score | A high-score candidate violates policy, date or capability constraints | Ranking cannot repair wrong eligibility metadata |
| `cost-latency-tradeoff` | Choose resource policy under quality and latency bounds | Cheaper routes or bigger batches change quality or waiting time | Compare full-path cost at a stated workload |
| `authorization-scope` | Bind data and cached state to verified access rights | A tenant, repository, adapter or prefix is reused across scopes | Content similarity does not grant access |
| `materialized-view` | Treat an index or cache as rebuildable derived state | Deletes, source versions or invalidation lag diverge from truth | An authoritative transaction needs a separate correctness contract |
| `atomic-release-bundle` | Promote compatible model and system versions together | Prompt, tool, adapter or index versions are independently rolled back | Stateless independent components need not share a release |
| `rollout-regression` | A new release harms a supported population or invariant | Canary or per-domain metrics diverge from baseline | A passing aggregate metric is insufficient evidence of safety |
| `human-in-the-loop` | Qualified review controls uncertain consequential decisions | Approval, evidence adjudication or uncertainty handoff is required | Human review must have capacity, authority and a defined timeout |
| `idempotent-side-effects` | Retries must not repeat an external action | Crash windows or duplicate events can replay purchases or updates | An idempotency key needs durable end-to-end enforcement |
| `snapshot-consistency` | Read related state from one declared version | Graph, ranking, inventory or policy parts can come from different times | Explicit best-effort views must disclose weaker guarantees |
| `resource-admission` | Reserve bounded work before accepting more load | Queue, token, KV or database capacity can be exhausted | Utilization alone does not establish a safe admission limit |
| `correctness-contract` | Choose algorithms from explicit semantics and assumptions | A fast heuristic violates the claimed exactness or objective | A complexity claim cannot replace a correctness proof |
| `bounded-work` | Bound processing independently of output size | Fanout, collisions, recursion or backtracking amplifies work | A depth or response-size cap may leave intermediate work unbounded |
| `incident-postmortem` | Trace a failure mechanism to a preventive replay | A violated invariant propagates across a service boundary | These references are exercises, not verified incident records |
| `delayed-selective-labels` | Outcomes arrive late and depend on prior selection | Only investigated claims acquire confirmed labels | Evaluate mature cohorts and audit beyond the selected queue |

Pattern preconditions, contrasts, supporting brief IDs and linked curriculum scope are recorded below. Deprecated synonyms: none introduced in this subset. Planned briefs provide comparison questions, not empirical confirmation of a pattern.

## 4. Cross-brief similarity and contrast map

| Pattern | Supporting proposed briefs | Contrasting reference scope | Curriculum/theory index | Preconditions and boundary to investigate |
|---|---|---|---|---|
| `distribution-shift` | [PLAN-2026-IR001](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir001-gate-retraining-on-corpus-provenance-and-long-tail-quality), [PLAN-2026-IR017](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir017-adapt-domain-asr-while-preserving-supported-domain-quality) | [REF-IR017](../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Whisper_Adaptation_And_Evaluation.md): Speech-domain regression offers a different population shift from synthetic corpus feedback | [Scope index; detailed theory still planned](05_ai_platforms_data_systems_and_production_engineering/27_data_quality_observability_and_reliability_engineering/INDEX.md) | Do not infer shift from sampling noise alone |
| `data-leakage` | [PLAN-2026-IR001](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir001-gate-retraining-on-corpus-provenance-and-long-tail-quality), [PLAN-2026-IR008](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir008-block-assistant-releases-on-auditable-quality-and-safety-gates), [PLAN-2026-IR009](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir009-allocate-a-domain-training-budget-with-restart-headroom) | [REF-IR008](../INTERVIEW_CASES/Agentic_AI_System/LLM_Safety_Evaluation/release_evaluation_platform.md): Safety evaluation also separates benchmark contamination from judge error | [Scope index; detailed theory still planned](05_ai_platforms_data_systems_and_production_engineering/27_data_quality_observability_and_reliability_engineering/INDEX.md) | A legal pre-cutoff feature is not leakage |
| `evaluation-mismatch` | [PLAN-2026-IR001](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir001-gate-retraining-on-corpus-provenance-and-long-tail-quality), [PLAN-2026-IR003](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir003-answer-support-questions-from-the-applicable-policy-version), [PLAN-2026-IR008](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir008-block-assistant-releases-on-auditable-quality-and-safety-gates) | [REF-IR027](../INTERVIEW_CASES/DSA/index_basics/Range_Queries/fenwick_inventory_totals.md): A deterministic range-sum oracle contrasts with human-judged language quality | [Scope index; detailed theory still planned](05_ai_platforms_data_systems_and_production_engineering/27_data_quality_observability_and_reliability_engineering/INDEX.md) | Use when a concrete decision-level comparator exists |
| `eligibility-before-ranking` | [PLAN-2026-IR002](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir002-select-model-routes-within-hard-customer-constraints), [PLAN-2026-IR013](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir013-retrieve-regulations-applicable-to-the-question-date-and-jurisdiction), [PLAN-2026-IR015](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir015-publish-reviewed-coaching-scripts-from-current-product-evidence) | [REF-IR002](../INTERVIEW_CASES/Agentic_AI_System/AI_frontier_sys/constrained_provider_routing.md): Provider eligibility concerns capabilities/residency; regulation eligibility concerns date/jurisdiction | [Scope index; detailed theory still planned](05_ai_platforms_data_systems_and_production_engineering/25_model_serving_inference_optimisation_and_cost/INDEX.md) | Ranking cannot repair wrong eligibility metadata |
| `cost-latency-tradeoff` | [PLAN-2026-IR002](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir002-select-model-routes-within-hard-customer-constraints), [PLAN-2026-IR007](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir007-balance-interactive-inference-latency-and-admitted-throughput), [PLAN-2026-IR009](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir009-allocate-a-domain-training-budget-with-restart-headroom) | [REF-IR007](../INTERVIEW_CASES/Agentic_AI_System/LLM_Inference/continuous_batching_serving.md): GPU scheduling trades queueing and memory, unlike model-price routing | [Scope index; detailed theory still planned](05_ai_platforms_data_systems_and_production_engineering/25_model_serving_inference_optimisation_and_cost/INDEX.md) | Compare full-path cost at a stated workload |
| `authorization-scope` | [PLAN-2026-IR003](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir003-answer-support-questions-from-the-applicable-policy-version), [PLAN-2026-IR004](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir004-customize-tenant-assistants-without-crossing-isolation-boundaries), [PLAN-2026-IR007](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir007-balance-interactive-inference-latency-and-admitted-throughput) | [REF-IR012](../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Chunking_And_Vector_Index.md): Index filtering and tenant adapter selection enforce access at different layers | [Scope index; detailed theory still planned](04_deep_learning_generative_ai_and_multimodal_systems/20_rag_context_engineering_and_enterprise_knowledge/INDEX.md) | Content similarity does not grant access |
| `materialized-view` | [PLAN-2026-IR003](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir003-answer-support-questions-from-the-applicable-policy-version), [PLAN-2026-IR012](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir012-preserve-answer-evidence-through-ingestion-and-filtered-retrieval), [PLAN-2026-IR020](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir020-translate-local-algorithm-guarantees-into-distributed-service-contracts) | [REF-IR033](../INTERVIEW_CASES/DSA/Sys_Des_DSA/Cache_Resilience/cold_cache_and_ring_migration.md): Cache invalidation has different stale-read risks from search index refresh | [Scope index; detailed theory still planned](04_deep_learning_generative_ai_and_multimodal_systems/20_rag_context_engineering_and_enterprise_knowledge/INDEX.md) | An authoritative transaction needs a separate correctness contract |
| `atomic-release-bundle` | [PLAN-2026-IR004](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir004-customize-tenant-assistants-without-crossing-isolation-boundaries), [PLAN-2026-IR005](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir005-release-and-recover-an-agent-as-a-compatible-bundle), [PLAN-2026-IR009](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir009-allocate-a-domain-training-budget-with-restart-headroom) | [REF-IR006](../INTERVIEW_CASES/Agentic_AI_System/Langraph/durable_human_approval_graph.md): Workflow checkpoint compatibility also needs external-action reconciliation | [Scope index; detailed theory still planned](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md) | Stateless independent components need not share a release |
| `rollout-regression` | [PLAN-2026-IR004](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir004-customize-tenant-assistants-without-crossing-isolation-boundaries), [PLAN-2026-IR005](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir005-release-and-recover-an-agent-as-a-compatible-bundle), [PLAN-2026-IR008](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir008-block-assistant-releases-on-auditable-quality-and-safety-gates) | [REF-IR017](../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Whisper_Adaptation_And_Evaluation.md): ASR uses per-domain human gold; general release gates need task-specific suites | [Scope index; detailed theory still planned](04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md) | A passing aggregate metric is insufficient evidence of safety |
| `human-in-the-loop` | [PLAN-2026-IR005](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir005-release-and-recover-an-agent-as-a-compatible-bundle), [PLAN-2026-IR006](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir006-recover-procurement-workflows-without-replaying-approvals), [PLAN-2026-IR011](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir011-compare-document-vlms-with-ocr-for-adjuster-assistance) | [REF-IR016](../INTERVIEW_CASES/Agentic_AI_System/Speech_STT/Multilingual_Call_Objections.md): A verified text quote can still reflect an acoustic or speaker-attribution error | [Scope index; detailed theory still planned](10_architecture_practice_and_portfolio_capstones/55_incident_response_failure_modes_and_technical_leadership/INDEX.md) | Human review must have capacity, authority and a defined timeout |
| `idempotent-side-effects` | [PLAN-2026-IR006](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir006-recover-procurement-workflows-without-replaying-approvals), [PLAN-2026-IR014](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir014-separate-extraction-correctness-from-routing-economics), [PLAN-2026-IR035](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir035-separate-live-rank-approximation-from-exact-competition-settlement) | [REF-IR035](../INTERVIEW_CASES/DSA/Sys_Des_DSA/Leaderboards/score_bucket_rank_consistency.md): Replay of score deltas differs from a purchase API's action guarantees | [Scope index; detailed theory still planned](06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md) | An idempotency key needs durable end-to-end enforcement |
| `snapshot-consistency` | [PLAN-2026-IR006](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir006-recover-procurement-workflows-without-replaying-approvals), [PLAN-2026-IR013](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir013-retrieve-regulations-applicable-to-the-question-date-and-jurisdiction), [PLAN-2026-IR018](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir018-specify-correctness-contracts-before-choosing-production-algorithms) | [REF-IR023](../INTERVIEW_CASES/DSA/Algo_Design_Problems/Routing/closure_aware_shortest_paths.md): Graph closure snapshots and exact final ranks use different state and finalization contracts | [Scope index; detailed theory still planned](06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md) | Explicit best-effort views must disclose weaker guarantees |
| `resource-admission` | [PLAN-2026-IR007](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir007-balance-interactive-inference-latency-and-admitted-throughput), [PLAN-2026-IR020](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir020-translate-local-algorithm-guarantees-into-distributed-service-contracts), [PLAN-2026-IR022](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir022-bound-global-api-admission-with-token-leases) | [REF-IR022](../INTERVIEW_CASES/DSA/Algo_Design_Problems/Rate_Limiting/distributed_token_buckets.md): Global token allocation is different from local GPU memory admission | [Scope index; detailed theory still planned](05_ai_platforms_data_systems_and_production_engineering/25_model_serving_inference_optimisation_and_cost/INDEX.md) | Utilization alone does not establish a safe admission limit |
| `correctness-contract` | [PLAN-2026-IR018](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir018-specify-correctness-contracts-before-choosing-production-algorithms), [PLAN-2026-IR019](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir019-keep-routing-correctness-separate-from-eta-quality), [PLAN-2026-IR020](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir020-translate-local-algorithm-guarantees-into-distributed-service-contracts) | [REF-IR025](../INTERVIEW_CASES/DSA/index_basics/Greedy_Scheduling/activity_selection.md): Earliest-finish greedy proves an optimum only for the equal-value interval objective | [Scope index; detailed theory still planned](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md) | A complexity claim cannot replace a correctness proof |
| `bounded-work` | [PLAN-2026-IR018](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir018-specify-correctness-contracts-before-choosing-production-algorithms), [PLAN-2026-IR022](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir022-bound-global-api-admission-with-token-leases), [PLAN-2026-IR024](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir024-bound-phrase-search-memory-while-preserving-exact-matches) | [REF-IR029](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Graph_Traversal/bounded_bfs_incident.md): BFS edge fanout needs different bounds from regex backtracking or SQL recursion | [Scope index; detailed theory still planned](01_foundations_computing_and_problem_solving/03_dsa_and_computational_thinking/INDEX.md) | A depth or response-size cap may leave intermediate work unbounded |
| `incident-postmortem` | [PLAN-2026-IR028](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir028-bound-request-amplification-before-expensive-parsing-and-validation), [PLAN-2026-IR029](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir029-return-useful-graph-candidates-under-explicit-traversal-budgets), [PLAN-2026-IR030](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir030-diagnose-production-failures-from-violated-algorithm-invariants) | [REF-IR030](../INTERVIEW_CASES/DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md): The anthology contains separate scenarios; do not merge durations into one incident | [Scope index; detailed theory still planned](10_architecture_practice_and_portfolio_capstones/55_incident_response_failure_modes_and_technical_leadership/INDEX.md) | These references are exercises, not verified incident records |
| `delayed-selective-labels` | [PLAN-2026-IR011](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir011-compare-document-vlms-with-ocr-for-adjuster-assistance), [PLAN-2026-IR036](00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir036-evaluate-incremental-investigator-value-from-graph-fraud-signals) | [REF-IR011](../INTERVIEW_CASES/Agentic_AI_System/Multimodal/Insurance_Document_VLM.md): Document-field gold and mature fraud adjudication have distinct label processes | [Scope index; detailed theory still planned](04_deep_learning_generative_ai_and_multimodal_systems/22_multimodal_models_video_voice_and_world_models/INDEX.md) | Evaluate mature cohorts and audit beyond the selected queue |

## 5. Questions and future-case backlog

| Open question | Origin | Needed evidence | Proposed next step | Priority | Owner role | Review trigger |
|---|---|---|---|---|---|---|
| Which source claims can be independently supported? | All 36 example files | Original OCR/images or authoritative replacement sources | Recover provenance before citing source-reported results | High | Research editor | Before drafting any source-dependent claim |
| How should overlapping anthology units become independent narratives? | DSA anthologies and paired LLM examples | Distinct decision, input population, assumptions and evaluation per unit | Split or merge briefs by decision, recording the crosswalk | High | Curriculum editor | Before assigning canonical case IDs |
| Which domain-specific constraints change the design? | Routing, inventory, support and insurance references | Actual business workflow, data contracts and decision owner | Write domain briefs without transplanting the example's scale or SLO | High | Domain author | At case scoping |
| Which company case claims are documented? | Retail, quick commerce, ecommerce and streaming request | Named-company primary sources with dates and stated limits | Research Walmart, Tesco, Target, Zepto, Swiggy, Flipkart, Disney, Warner and Hotstar separately | High | Domain researcher | Before company-specific writing |
| What remains uncovered in edge robotics and low-level systems? | Original directory audit and validation report | Per-source-topic destination crosswalk and new specialist references | Expand embedded, CUDA, robotics, compiler and streaming topic scope independently | High | Technical editor | Next completeness audit |
| How should planned cheats differ from case narratives? | All reference-informed scope blocks | Minimal algorithms/formulas, preconditions, counterexamples and worked checks | Draft compact reusable references only after their assumptions are explicit | Medium | Reference author | Before cheat-sheet publication |

## 6. Review and retirement log

| Record | Date | What changed | Evidence / trigger | Action | Owner role |
|---|---|---|---|---|---|
| All planning briefs | 2026-11-06 | Created reference-only scope; no cases drafted or source outcomes adopted | User instructed that interview materials are examples for reference only | indexed for planning | Curriculum maintainer |

When a brief is split, merged, superseded or retired, preserve its planning ID and record the replacement. A source edit or recovered evidence requires a fresh review; inventory dates are not certification of technical accuracy.

## 7. Pre-publication checks

- [x] Source inventory mapped without modifying the examples.
- [x] Proposed briefs distinguish reference existence from completed-case status.
- [x] Proposed cases use not-measured evidence, proposed stage and empty result fields.
- [x] Decision questions, mechanisms, evaluation plans and transfer limits are explicit.
- [ ] Independently verify claims, version-sensitive mechanisms and recovered primary sources.
- [ ] Split multi-case references and write each case with metadata and all 15 outline sections, using permitted N/A where appropriate.
- [ ] Specify business owners, actual data availability, baseline, assumptions, operational review triggers and ethical/legal boundaries per case.
- [ ] Assign canonical case IDs, create real case files and update links only when those files exist.
- [ ] Draft and review the separate cheat sheets; their index descriptions are currently plans.
- [x] Visible original-audit entries are now reconciled separately in the [source-topic crosswalk](00_SOURCE_TOPIC_CROSSWALK.md); this reference plan alone still does not establish coverage or completed content.

### Source availability limitations

At this review, **64 relative link occurrences** in the original examples point to unavailable files or directories. The examples themselves exist. The originals were not repaired, rewritten or treated as verified source evidence. See the archived [reference-plan validation report](../Templates/archive/INTERVIEW_REFERENCE_PLAN_VALIDATION_2026-11-06.md) for the exact inventory.

`Train/INTERVIEW_CASES/DS_Domain` contains no files in this snapshot. No robotics, CUDA-kernel or named-company evidence has been inferred from that empty directory. Road-network routing examples do not establish robot-navigation, SLAM, control or safety coverage.
