# Comprehensive AI/ML library validation

**Review date:** 2026-11-06  
**Target:** `Train/ComprehensiveAIML_BUSINESS_DOMAINS/`  
**Verdict:** The current scaffold does **not** meet the requested standard of preserving every documented topic at detailed index level. Its broad organisation is present, but there are confirmed omissions and incomplete template adoption. The earlier family-level coverage matrix is insufficient evidence of completeness.

This review changes no curriculum files. It records findings and the remaining acceptance criteria. Counts below describe the curriculum snapshot inspected, excluding this report.

## 1. Sources and method

Sources reviewed:

- [Original directory audit](TRAIN_DIRECTORY_STRUCTURE_AUDIT_56_IMAGES.md).
- [Case Study Collection INDEX Template v2](INDEX_Case_Studies_Template_v2.md).
- [Integrated Case Study Outline Template v2](Case_Study_Outline_Template_v2_Integrated.md).
- All 423 Markdown files in the target library, including all 421 directory indexes.

The source audit's main tree was parsed by indentation and full parent path. The result independently reproduces its published totals: **186 directory records, 1,053 file records, and 8 additional prefix-only observations**. The parser also identifies **59 explicitly collapsed directories**. These are source records, not a count of distinct learning topics or proof about hidden content.

The validation checks directory/index presence, required collection pairs, existing local Markdown links, file text, catalog fields and placeholders, prerequisite dependencies, and explicit source terminology in the index corpus. The 55 original production-ML case titles were compared individually with their corresponding planned catalog rows.

Topic coverage was assessed at the level of explicit index scope. A source topic does not have to retain its old filename or folder, but its mechanism and learning scope must be identifiable in a destination. Generic headings such as "compiler optimisation" or "preference optimisation" cannot establish that every more specific source topic has been retained.

Term searches were used to locate evidence and omissions. Absence of a term alone is not a complete semantic coverage metric; the confirmed examples below combine that absence with inspection of the relevant destination index. The family coverage matrix and README were not allowed to substitute for actual topic entries.

## 2. Structural and content checks

| Check | Result | Interpretation |
|---|---|---|
| Directories, including library root | 421 | Reproduces the previous structural count |
| Markdown files | 423 | 421 indexes, one README, and one family coverage matrix |
| Directory without INDEX.md | 0 | Pass |
| Broken existing relative Markdown file links | 0 | Pass for links actually written; absent links are not validated by this result |
| Files containing NUL characters | 0 | Pass |
| Subject directory missing its two collections | 1 | The DS-domain library root has neither Case_Studies nor Cheat_Sheets |
| Case_Studies indexes | 140 | 58 detailed domain catalogs plus 82 other collection indexes |
| Case indexes with v2-style metadata/catalog sections | 58 | Partial adoption; see Section 5 |
| DS families / child domain folders | 15 / 58 | The earlier claim of 55 child folders was incorrect; 55 is the original production-case count |
| Dedicated company analysis directories | 0 | Company names occur as context examples |
| External-source links anywhere in DS-domain library | 0 | Company research and evidence are still absent |
| Prerequisite cycles | 1 | Module 16 requires 49, and 49 requires 16 |
| Markdown files with garbled punctuation | 199 | UTF-8 text contains mojibake such as `â€”`; NUL checks did not catch this |

The missing collection pair is under [Data Science Domain Specialisations](../ComprehensiveAIML_BUSINESS_DOMAINS/11_data_science_domain_specialisations/INDEX.md). Collection folders themselves are terminal leaves; this check does not require an infinite recursion of collections inside collections.

The prerequisite cycle is explicit in [Module 16](../ComprehensiveAIML_BUSINESS_DOMAINS/03_machine_learning_optimisation_and_intelligent_retrieval/16_privacy_preserving_ml_and_federated_learning/INDEX.md) and [Module 49](../ComprehensiveAIML_BUSINESS_DOMAINS/09_trust_security_governance_and_product_leadership/49_ai_security_privacy_safety_and_red_teaming/INDEX.md), both at line 4.

The absence of separate case-study and cheat-sheet documents is appropriate for the requested indexing phase. It does mean that the 15-section case-writing standard cannot yet be validated against drafted cases.

## 3. Confirmed missing or underspecified source coverage

The following examples are explicitly present in the source audit and have no corresponding explicit scope in the relevant new indexes. This is an evidence-backed gap list, not an exhaustive enumeration of every missing source lesson.

| Source family and source-audit lines | Details not explicitly indexed | Current destination and coverage limit |
|---|---|---|
| CUDA, 344-369 | Thread-block clusters, DSMEM, inline PTX, PTX state spaces, mbarrier lifecycle, tensor-map descriptors, shared-memory swizzling | Modules 23-25 discuss hardware, training and serving broadly |
| CUDA, 371-399 | cp_async_bulk, Hopper/Ampere async distinctions, WGMMA descriptors/groups, sparse WGMMA, FP8 packing/layout | No WGMMA, mbarrier, PTX, or cp_async entries in any curriculum index |
| CUDA, 400-440 | Roofline/arithmetic intensity, warp specialisation, persistent kernels, SM90 code walkthroughs, MPI, PMIx, NCCL, Slurm | General parallelism/cluster topics do not define these learning units |
| Agent Stack, 72-159 | OpenClaw architecture, Hermes self-evolving-agent architecture, Sourcegraph integration, nvidia-smi, MIG, DCGM, NCA AIIO | No OpenClaw, Hermes, Sourcegraph, NCA AIIO, MIG, or DCGM entry in curriculum indexes |
| Claude Architect, 229-249 | CLAUDE.md hierarchy, slash commands/skills, path-specific rules, SDK hooks, few-shot prompting, structured tool errors | Coding agents, schemas and prompts are broad headings; the source-specific mechanisms lack explicit entries |
| LangGraph/LangChain, 1226-1256 | LCEL, graph nodes/edges/control flow, LangSmith observability, framework-specific production/debugging patterns | LangGraph is mentioned only in a family-coverage statement inside the DS root index; there is no actual LangGraph topic entry |
| Streaming systems, 190-216 | Flink CEP/broadcast state, session windows, Debezium CDC, tiered storage, CQRS, hot partitions, checkpoint OOM recovery | Module 09 has "Kafka and stream processing" plus general state/CDC topics; no Flink, Redpanda, Debezium, CEP, CQRS or tiered-storage entry |
| Finetuning, 854-918 | LoRA/QLoRA mechanics, PPO/RLHF, DPO, GRPO, reward modelling, FlashAttention, MoE | Module 21 lists SFT/PEFT and preference optimisation without enumerating these techniques |
| Finetuning, 931-968 | RoPE/position interpolation, long-context training specifics, mechanistic interpretability, KTO/ORPO/SimPO, RLVR, DAPO, Mamba/RWKV/RetNet, curriculum learning | Broad architecture/context/alignment concepts do not preserve this explicit source sequence |
| Distributed finetuning, 998-1009 | DeepSpeed, FSDP2, Megatron-LM, communication overlap, ZeRO variants, elastic-training/debugging detail | Module 24 has five high-level training topics; these implementation branches are not enumerated |
| Speech finetuning, 1011-1024 | Whisper fine-tuning, voice cloning/zero-shot TTS, multilingual and low-resource speech, native-language training | Speech and voice agents are covered broadly, with no explicit Whisper fine-tuning or equivalent detailed sequence |
| Biomedical edge course, 1040-1054 | Pan-Tompkins/QRS detection, time/frequency/nonlinear HRV, PPG/SpO2, FFT/STFT/wavelets, biomedical noise and dataset specifics | Modules 35-42 and DS healthcare indexes mention signals/ECG without this detailed map |
| Deep-learning time series, 1057-1070 | 1D CNN/BiLSTM mechanics, PatchTST, TimesNet, TSMixer, focal-loss/augmentation specifics | No explicit PatchTST, TimesNet, TSMixer, RNN/LSTM/GRU, or focal-loss topic entries |
| Compression and embedded systems, 1073-1123 | AWQ, SmoothQuant, structured/unstructured pruning, hardware-aware NAS, CMSIS, FreeRTOS, STM32 DMA/UART, linker scripts, SWD/JTAG | Module 36 lists PTQ/QAT and compiler optimisation; the detailed compression/firmware curriculum is absent |
| Edge compilers, 1125-1141 | MLIR dialects/passes, TVM lowering, microTVM AOT, IREE, RISC-V vector extensions, compiler debugging/OSS contribution | No MLIR, TVM, IREE, microTVM, or RISC-V entry in curriculum indexes |
| On-device RAG/integration, 1143-1176 | GGUF, llama.cpp, FAISS IVF/PQ/HNSW, Raspberry Pi/STM32 serial integration, hardware-in-loop validation | RAG, edge runtimes and general integration are covered without this source-level implementation map |
| Healthcare/insurance deep dives, 448-489 | UHG/Optum, CVS, Chubb, pharmacy/member FWA, SDOH and value-based-care comparisons | General healthcare, insurance and pharma folders do not retain the named-company deep dives or these explicit subdomains |
| DS NLP cheat sheets, 665-672 | RNN/LSTM/GRU sequence models, BERT encoders, RLHF alignment | Replaced by generic classical-NLP/transformer/fine-tuning rows; no named entries for these source topics |
| DS fraud cheat sheets, 598-602 | NLP/text fraud signals, synthetic identity verification, healthcare fraud/waste/abuse | Fraud is represented, but these source subtopics are not explicitly scoped |
| DS marketing/search, 644 and 799 | RFM segmentation and TF-IDF retrieval | Generic segmentation/classical-retrieval coverage does not name these source techniques |
| DS practical ML, 699-702 | Loss functions/optimisation traps, production architecture traps, additional common-issue collections | Broad system-design and incident folders do not preserve the specific source issue-bank coverage |
| DS forecasting, 839 | Healthcare/epidemiological forecasting | Healthcare and forecasting both exist, but no explicit epidemiological-forecasting topic |
| Insurance, 1210 and 1217 | Survival-model detail and extreme-value theory for catastrophe | Life contingencies and catastrophe are listed broadly; survival modelling and extreme-value theory are not explicitly indexed |
| Inference test clusters, 1189-1203 | The six cluster-specific test plans: tools, path completion, reasoning, data integrity, safety/escalation, multi-agent | Only named in the family matrix, without corresponding cluster-specific evaluation catalogs |

Examples of destination scope inspected directly:

- [Module 23: GPU hardware](../ComprehensiveAIML_BUSINESS_DOMAINS/05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md) has five broad topic rows.
- [Module 21: Fine-tuning](../ComprehensiveAIML_BUSINESS_DOMAINS/04_deep_learning_generative_ai_and_multimodal_systems/21_finetuning_alignment_and_model_customisation/INDEX.md) has five broad topic rows.
- [Module 36: Edge runtimes/compilers](../ComprehensiveAIML_BUSINESS_DOMAINS/07_edge_ai_robotics_and_cyber_physical_systems/36_edge_deployment_compilers_quantisation_and_runtime/INDEX.md) has five broad topic rows.
- [NLP/LLM playbook](../ComprehensiveAIML_BUSINESS_DOMAINS/11_data_science_domain_specialisations/02_data_science_method_playbooks/05_nlp_llm_playbook/INDEX.md) lists general topic names and status, without detailed descriptions of the missing methods.

The source audit's PDF files, JSON fixtures, scripts, workspaces and compiled text are support assets rather than interchangeable curriculum topics. Their inclusion/exclusion also needs a deliberate decision in a complete item-level crosswalk; absence of a replicated PDF is not automatically a missing lesson.

## 4. What is represented successfully

The **55 original production-ML case themes** have corresponding planned rows across all eleven source categories. The category/title comparison covered every source case, five per category:

| Source category | Original case themes | Corresponding planned rows |
|---|---:|---:|
| Fraud modelling | 5 | 5 |
| Graph ML | 5 | 5 |
| KDA pharma | 5 | 5 |
| Marketing analytics | 5 | 5 |
| NLP/LLM | 5 | 5 |
| Pricing | 5 | 5 |
| Product analytics | 5 | 5 |
| Recommendation systems | 5 | 5 |
| Search/AdTech | 5 | 5 |
| Time series/forecasting | 5 | 5 |
| US commercialization decision intelligence | 5 | 5 |

This establishes theme-level representation, not exact preservation of every qualifier, implementation technique, outcome, or source detail. For example, the source KDA launch-brand and loss-of-exclusivity/mature-brand contexts are shortened in the new case titles. A complete scope description must preserve those distinctions.

The ten principal DS cheat-sheet families and the practical-ML/product-sense families also have corresponding broad destination folders. Their **individual topic coverage remains incomplete**, as the examples above demonstrate.

The requested new retail, commerce, delivery, supply-chain and streaming sectors are present at industry level. Edge robotics also has an explicit seven-module pathway covering embedded AI, runtimes, sensors, ROS 2/control, autonomy/VLA, simulation and fleet operations.

## 5. Case-study template adoption

The current state is **partial v2 adoption**, rather than complete compliance.

| Requirement | Observed result |
|---|---|
| Metadata and 11-field case catalog | Present in 58 of 140 Case_Studies indexes |
| Other collection indexes | 82 have an older brief or generic format |
| Planned evidence/stage separation | The 290 detailed-domain rows correctly use not-measured / proposed and leave results blank |
| A concrete business question per planned case | All 290 rows say "Decision to be framed before writing" |
| Case-specific pattern choice | All 290 rows use evaluation-mismatch / model-monitoring, regardless of case content |
| Case-type differentiation | All 290 rows are hybrid |
| Cross-case comparisons | Tables contain generic descriptions; no contrasting-case column or resolved theory links in the 58 detailed catalogs |
| Structured future-question backlog | The template's question/evidence/priority/owner/review-trigger table is absent from all 58 |
| Review and retirement log | Absent from all 58 |
| Pattern vocabulary | Ten generic definitions copied across the detailed catalogs; no documented rationale for domain-specific selection |
| Integrated 15-section case outline | Referenced as a future authoring rule; there are no individual case files to check yet |

Blank result fields and unmeasured/proposed status are appropriate at this phase. However, the user requested detailed index briefs: business questions, intended decisions, data/evaluation boundaries and case-specific patterns should be planned now. Future case files do not have to be drafted to improve those briefs.

The template permits adaptation of its sample tags; an exact copy of all 18 sample tags is not a requirement. The issue is that current tag assignments are generic and uninformative, rather than tailored to the planned cases.

## 6. Requested industry and company analysis

| Requested area | Current scope | Remaining depth gap |
|---|---|---|
| Offline/online stores; Walmart, Tesco, Target | Three retail subdomains, each with topic rows and five planned cases; companies are named as examples | No company-specific research index, business-model comparison, dataset/metric map, or sourced case backlog |
| E-commerce; Flipkart | Three commerce subdomains; Flipkart appears as context | No Flipkart-specific decision/value-chain analysis or comparison with store-led retail |
| Quick commerce; Zepto, Swiggy | Three subdomains for inventory, delivery and economics; company names appear as context | No separate company/workflow analyses or sourced comparisons of grocery and restaurant-delivery operating models |
| Supply chain | Three subdomains for planning, transport/warehouse and control towers | Large topic labels need expansion into detailed decisions, data grains, constraints, metrics, algorithms and case briefs |
| Streaming; Disney, Warner Brothers, Hotstar | Three subdomains for content/rights, audience and ads/live/QoE; names appear as context | No company-specific analyses, public engineering/source catalog, or detailed business-model/rights/technology comparisons |

All nine company names requested are represented in context text, treating "Warner Brothers" as the Warner Bros. naming used in the indexes. **None has its own analysis directory or evidence-backed analysis file.** There are no external-source links in the DS-domain library.

Industry topic tables generally contain a short topic name plus "planned". They need the requested descriptions, intended decisions, data sources/grains, metrics, failure modes and future artifact names. The existing 58 domain folders have no deeper numbered topic subfolders.

Company names here describe the existing scaffold only. This validation makes no claim about a company's current ownership, product naming, technology stack or operating results.

## 7. Source uncertainty and completeness limits

The source itself cannot support an unconditional guarantee about hidden material:

- 59 explicitly collapsed directories have unknown contents.
- Eight HighEnd overview filenames are prefix-only observations.
- Several finetuning lesson filenames and a Claude instructor PDF are clipped.
- A HighEnd module folder name is clipped, and a separate similarly numbered compression module remains collapsed.
- The System_Design_DSA partial subtree has an unknown parent/full folder name.

These uncertainties are separate from the **confirmed missing detail in visible source entries**. They must remain open questions, while verified source topics can be indexed fully without waiting for them.

## 8. Acceptance criteria for a later completion pass

1. Build a source-entry crosswalk for all 1,053 file records, 186 directory records and eight prefix-only observations. Each receives a destination, an explicit support-asset/exclusion rationale, or an unresolved-evidence status.
2. Map every visible teaching topic to an explicit destination topic brief. Renaming/consolidation is acceptable if the scope survives and can be traced.
3. Expand the CUDA, training/alignment, agents/frameworks, biomedical, embedded/compiler, streaming and insurance gaps listed above.
4. Bring every Case_Studies index to the applicable v2 structure, including concrete decision questions, useful pattern assignments, theory links, backlog and review log.
5. Expand the requested industries and company research contexts with detailed decision/data/metric maps, company research indexes and sourced case backlogs.
6. Add the missing DS-root collection pair, resolve the prerequisite cycle, and repair the 199 mojibake files.
7. Re-run structural/link/text checks and audit source-to-destination coverage after the content pass.

No overall coverage percentage is assigned. The existing family matrix does not supply an item-level denominator or mapping, and assigning a percentage from keyword matches would imply precision the evidence does not support.
