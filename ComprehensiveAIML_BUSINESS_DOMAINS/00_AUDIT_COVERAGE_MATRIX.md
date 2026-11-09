# Audit Coverage Matrix

**Scope:** the entire visible TRAIN tree supplied by the user, also preserved in the [archived source audit](../Templates/archive/TRAIN_DIRECTORY_STRUCTURE_AUDIT_56_IMAGES.md). **State:** exhaustive topic planning, not completed course content. The earlier broad family mapping is replaced by exact-node accounting and focused briefs.

## Coverage authority

- [Exact source-topic crosswalk](00_SOURCE_TOPIC_CROSSWALK.md) and [machine-readable assignments](00_SOURCE_TOPIC_CROSSWALK.json): 1,247 entries, each assigned once.
- 941 planned-topic source entries, 106 supporting artifacts, 126 containers and 74 unresolved-source entries. These are inventory roles, **not unique lesson counts**.
- 144 focused teaching groups name every visible technical title and specify teaching outcomes, derivations/mechanisms, numerical/trace tasks, case decisions and validation.
- [Authoring and numerical standards](00_AUTHORING_AND_NUMERICAL_STANDARDS.md): mandatory master-template structures and mathematical depth for every future artifact.

## Source families and detailed additions

| Source family | Detailed planned coverage | Focused plan |
|---|---|---|
| Agents_Stack_Infra | All Modules 00–14: prerequisites; ten-layer boundaries; serving; control/session state; runtimes; context/memory/Self-RAG/Sourcegraph; tools/MCP; patterns; trust; evaluation; OpenClaw; source-specific Claude harness; Hermes; NVIDIA infrastructure; production | [Agents plan](00_SOURCE_COVERAGE_AGENTS.md) |
| Claude_Architect | All five study domains, practice questions, 60-question exam, hands-on exercises, six quick references, six core scenarios and nine industry/cross-industry scenario groups | [Agents plan](00_SOURCE_COVERAGE_AGENTS.md) |
| LangGraph_LangChain_2025 | LCEL; prompts/parsers; graph nodes/edges; tools/multi-agent; persistence; RAG; streaming; LangSmith; deployment; dated patterns; 100-issue source target | [Agents plan](00_SOURCE_COVERAGE_AGENTS.md) |
| inference | All six clusters: mechanics, paths, reasoning, integrity, safety/escalation and multi-agent; tool/mock/manifest/archive/explainer roles | [Agents plan](00_SOURCE_COVERAGE_AGENTS.md) |
| CUDA | All Modules 00–11 and cheat sheets: Hopper hierarchy, PTX/types/layouts, mbarrier, TMA, cp.async.bulk, WGMMA/FP8/sparsity, kernel design, named CUTLASS/flash walkthroughs, P2P/MPI/PMIx/SLURM/NCCL and parallelisms | [CUDA/training plan](00_SOURCE_COVERAGE_CUDA_TRAINING.md) |
| finetuning_course | All Modules 00–20, implementation playbook and cheats: curation/tokenizers, objectives/scaling, memory, multidimensional parallelism, PEFT/merging, modern alignment/RL, evaluation/inference/context, alternative architectures, VLM, distributed infrastructure and speech | [CUDA/training plan](00_SOURCE_COVERAGE_CUDA_TRAINING.md) |
| HighEnd | All eight visible modules and capstone: signals/HRV/filtering, biosignal DL, compression, runtimes, ARM/RTOS/DMA/UART, compilers, local LLM/RAG, integration/UI/HIL, profiling and portfolio/OSS plans | [Edge/industry plan](00_SOURCE_COVERAGE_EDGE_INDUSTRY.md) |
| Case_Studies_Streaming_Kafka_Flink_Redpanda | Framework/HLD/LLD/tradeoffs; nine separate Flink/Kafka/streaming design cases and two production-incident exercises; versioned Redpanda comparison | [Edge/industry plan](00_SOURCE_COVERAGE_EDGE_INDUSTRY.md) |
| deep_dive | UHG/Optum/CVS/Chubb/other-insurer research contexts; pharmacy/member/FWA/SDOH/VBC; autonomous/federated/climate/genomics; speech, P&C/medical CV, multimodal documents; all pharma agentic workflows | [Edge/industry plan](00_SOURCE_COVERAGE_EDGE_INDUSTRY.md) |
| Insurance_Analytical_Modelling | Probability/distributions, survival/life contingencies, pricing/GLM, reserving/reinsurance/risk/EVT/Bayesian credibility, ML/customer analytics, governance/MLOps and issue targets | [Edge/industry plan](00_SOURCE_COVERAGE_EDGE_INDUSTRY.md) |
| DS_Domain | All 55 production cases across 11 families; all ten detailed method-cheat families; commercialization cases/hidden-sheet boundary; practical ML design, traps and product-sense guide | [DS plan](00_SOURCE_COVERAGE_DATA_SCIENCE.md) |
| Graph_ML_2025 | Theory, GNNs, knowledge graphs/ontologies and fraud, with time-correct graph construction and non-graph baselines | [DS plan](00_SOURCE_COVERAGE_DATA_SCIENCE.md) |
| .claude, agents | Collapsed agent/configuration sources; do not infer their contents | [Agents source limits](00_SOURCE_COVERAGE_AGENTS.md#agent-source-artifacts) |
| Finetune_LLM | Collapsed fine-tuning source, distinct from the visible finetuning_course | [Training source limits](00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-source-artifacts) |
| GLCC, learn, raw, speechtotxt | Unexplained/collapsed sources retained as unresolved, not invented lessons | [Unresolved sources](00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#unknown-sources) |

Existing course-only topics, including DSA/system design and robotics/ROS/control/navigation/simulation, remain in their original modules. The archived audit's unplaced System_Design_DSA discussion is not an extra branch in the supplied tree and is not silently included in the 1,247-entry denominator.

## Source uncertainty and completion boundary

All 74 clipped/collapsed entries remain visible in the machine-readable crosswalk and the detailed registers. Known prefixes and visible children are planned; missing filenames, hidden implementations and unavailable content are not reconstructed. Resolve these through an actual filesystem/content export before claiming hidden-topic completeness.

Every collection now references the master-template contract. Local catalogs remain planned; they must be expanded to all required index sections when their content is authored. All cases and sheets still require their own topic-specific mathematics, answers, source verification and publication checks. No outcomes, benchmarks, completed troubleshooting entries or substantive lesson reviews are invented.

## Verification record

Executed on 2026-11-07 against the saved course files:

| Check | Observed result |
|---|---|
| Exact source inventory → JSON assignment comparison | 1,247 / 1,247 entries, original order preserved; zero missing, extra or duplicate assignments |
| Source → focused brief → teaching home → backlink traversal | All assigned source rows, explicit anchors, destination files and backlinks resolve across 144 groups and 61 teaching homes |
| Course Markdown navigation | 430 Markdown files; 2,197 local link occurrences checked; zero missing files or anchors |
| Collection authoring-contract links | 280 / 280 collection indexes linked |
| Root collection structure | Case INDEX: seven numbered sections; cheat-sheet INDEX: 11 numbered sections |
| Shared numerical example smoke | 120,000-event backlog; 122,880,000 payload bytes = 117.1875 MiB; 600-second drain; no-backlog and zero-surplus boundaries checked |

These checks verify the inventory, navigable planning surface, contract wiring and exercised arithmetic only. They do not certify inaccessible source content, future topic-specific equations, benchmark results, completed cases, issue counts or interview answers.

## Current detailed-authoring pass

The requested continuation covers Parts 01–11, every indexed descendant, and the root cross-domain collections. Directory ownership is partitioned between authoring workers; shared source registers and the root index remain integration-owned. Work in progress is not completed coverage.

The initial configured-model workers stopped with provider `usage_limit_reached` errors; their partial files and coverage ledgers are retained, not counted as completed modules. On 2026-11-09, session-only per-call model selection was enabled and thirteen replacement workers were launched with the explicit selector `openai-codex/gpt-5.6-terra:medium`: one for each part and two for root collections. Launch acceptance is not proof of completed authoring or available provider capacity.

Source reconciliation executed for this pass:

| Check | Observed result |
|---|---|
| `directory.txt` tree entries versus JSON assignments | 1,247 ordered entries match after normalization of documented clipping/extension annotations |
| Duplicate normalized source assignments | 0 |
| Missing teaching-home files | 0 |
| Focused brief anchors and source-path occurrences | All 144 groups resolve; every assigned source path occurs in its register |
| Inventory uncertainty | All 74 unresolved entries retained; hidden contents remain unavailable |

This pass's source checks do not establish completed lessons, substantive depth, technical correctness, or publication review. Final content, navigation and numerical checks must follow authoring and integration.