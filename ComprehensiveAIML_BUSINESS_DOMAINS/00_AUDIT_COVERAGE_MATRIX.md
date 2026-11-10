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

### Provider interruption and checkpoint acceptance

The requested Terra workers subsequently returned `usage_limit_reached` errors as well. Observed failures include Foundations, Machine Learning, Edge/Robotics, Business Domains, Trust/Leadership, Domain Specialisations and the root cheat-sheet worker. This is a provider-capacity interruption, not evidence of an automatic task timeout. No substitute model is authorized by the requested model selection.

No returned manuscript has passed the complete substantive-depth acceptance gate in this pass. Physical line counts alone were rejected: returned firmware material contained repeated laboratory prose, the analytics draft repeated generic diagnostic text, and the six root cases repeated generic validation probes. The problem-formulation sheet repeats generic review evidence and decision-consequence prompts and has extra numbered-H2 subsections; the CNN manuscript repeats diagnostic explanations, mitigation risks and interview fields dozens of times. Both are explicitly marked incomplete in their files/indexes despite exceeding 1,000 physical lines. The fraud taxonomy's repeated boilerplate was removed, leaving 732 lines after integration; its appended taxonomy was moved under Section 11 and its source inventory linked. These are authoring defects, not completed topic coverage. Corrected or shortened drafts must remain visibly incomplete until their individual requirements are met.

Integration corrections also restored the charter's File Header, linked source inventories in the charter/clinical/interview manuscripts, removed an unresolved CNN arithmetic placeholder, and linked the GPU roofline to its visible CUDA inventory. The roofline, charter, clinical, interview, fraud, loss and CNN illustrative calculations were independently exercised as documented below. Arithmetic agreement does not approve repeated manuscript padding or validate an operational result.

A second expansion mapped all remaining Part 11 families to sixteen exclusive medium-effort Codex workers and Part 01 modules 02–05 to another worker; all seventeen returned `usage_limit_reached` immediately without reported edits. Five later continuation retries (orientation, machine learning, edge/robotics, production platforms and analytics) failed with the same error. Retained partial files and precise unfinished scope are the resume state; workers cannot be honestly reported as running or completed while the provider rejects them.

Checks actually executed by integration:

| Exercise | Observed result | Limit |
|---|---|---|
| Firmware arithmetic FW-01–05 | 792 bytes/s; 11,520 bytes/s UART payload; 6.875% utilization; 100 ms window; 30 ms base versus 88 ms stressed response; 0.0012408 J and 12.408 mW | Arithmetic only; no board execution, timing guarantee or safety validation |
| SQLite join counterexample | Three input facts and three joined rows; coverage 2/3, matched-fact multiplicity 3/2; naive joined total 140 versus input total 200 | Executed in-memory SQL fixture; not a warehouse deployment |
| Root-case fluid-queue arithmetic | 120 pending decisions; 300-second drain with positive 0.4 decision/s recovery surplus | Constructed constant-rate example; not replay or capacity evidence |
| Root-case scenario arithmetic | Agent 12 attempts; robotics 10 ms slack; review 12 hours; fraud precision 40%; supply 100 units; incident error fraction 1% | Arithmetic only; not domain efficacy, feasibility certification or measured outcomes |
| Boundary-source claims | NIST least-privilege glossary and OWASP LLM01:2025 guidance read directly | General definitions and mitigation guidance only; no architecture certification |
| GPU roofline | 71 named arithmetic checks passed; CPU GEMM returned `[[58,64],[139,154]]`; eight-key aggregation retained four occurrences per key; binary64 reassociation produced `(1e16+1)-1e16=0` versus `(1e16-1e16)+1=1` | Constructed formulas and CPU references; no CUDA kernel, profiler, device timing or benchmark |
| AI use-case charter | 20 checks passed: `$2,040` base value, `192` versus `180` reviews/day, `15.875%` break-even rate, `-$534.60` warranty net and all stated sensitivity results | Constructed economics; no realized benefit, cost estimate validation or deployment authorization |
| Clinical handoff | Four metric/capacity checks and a time/trace fixture passed: sensitivity 90%, PPV 30%, coverage 90%, 20 review-hours; only vital/missing fields admitted at 09:00; two route attempts retained one task and terminal disposition | Toy paper/state oracle; no clinical efficacy, confidence interval, real patient data or policy compliance |
| Interview framework | 12 arithmetic checks passed including `$3.60` EV, 12,000-request backlog, 400-second drain, 75%/81.82% denominators and 60/85-ms latency traces; retry returned one committed `D91/REVIEW` | In-memory ledger/reference behavior; crash-before-outbox raised as an intentional unresolved recovery scenario, not a production atomicity demonstration |
| Fraud taxonomy | Precision 30%, recall 60%, 140 false positives and 3.333 reviews/true positive; availability filter admitted device/excluded closure; 104-day maturity check | Constructed counts/time; no fraud prevalence, prevention, loss saving or legal conclusion |
| Problem formulation and CNN | 15 arithmetic checks passed including `tau=0.20`, queue expectation 1.68, baseline/proposed utility 322/576.8, required 1.725 staff-days, 118,013,952 MACs and 1,605,632 activation bytes | Arithmetic only; repeated appended manuscript material remains rejected and unapproved |

Resume from the earliest incomplete item in each local index, preserving all original catalog and source-crosswalk commitments. An interrupted or undersized draft is not a completed module; source bodies marked unavailable remain unavailable.

### Integration navigation check after interruption

The previous integrated check covered 597 Markdown files and 3,559 local Markdown link occurrences. It initially found 25 missing file targets, including incorrect template-directory depth, tool-only line selectors embedded in Markdown URLs, and cross-module paths. Those references were corrected. A later 2026-11-09 dependency-free check covered **600 Markdown files and 3,380 local link occurrences**, resolving `path#fragment` links before URL parsing: **0 missing file targets and 0 missing Markdown anchors**. This validates local navigation only, not lesson completeness or external-source accuracy.

The previous requested-model run stopped with provider-capacity errors. At the user's renewed request specifying Codex, thirteen fresh workers were launched with `openai-codex/gpt-5.6-terra:medium`: Module 01 orientation, Parts 02–10, Part 11's first fraud-domain module, and the two shared root collections. Launch acceptance does not establish restored capacity or completed content; later Part 01 and Part 11 modules remain in the continuation queue. Coverage ledgers and local indexes are resumption checkpoints, not evidence that the eleven parts or their submodules are finished. Substantive expansion, removal of rejected repeated material, independent content review, and remaining cases/Common Issues artifacts are still required.

### Original-catalog scope retained during renewed authoring

Explicit planned commitments were restored alongside newer drafts rather than silently replaced:

- Module 06: data model notation and warehouse query patterns, in addition to data contracts and grain/join safety.
- Module 11: classification metrics and model selection, in addition to feature engineering, predictor families, calibration and validation.
- Module 17: vision model selection, augmentation library and detection metrics, in addition to architecture, self-supervision and evaluation drafts.

These restored catalog rows are not authored lessons. Module 11's prerequisite handoffs were also corrected to Module 05 mathematical foundations and Module 06 data modelling/governance.

The existing Part 11 baseline ledger records all **15 families, 58 nested domain modules, 219 descendant indexes and 388 assigned source entries**. The first fraud-domain worker is only the initial checkpoint in that larger scope; it does not replace the remaining fourteen families or any other domain module.

### Full recursive Codex continuation request

At the user's renewed request, 25 worker jobs were launched with the exact selector `openai-codex/gpt-5.6-terra:medium`: one exclusive recursive author for each of Parts 01–10, and one for each of Part 11's 15 families. Each assignment includes every descendant module, original catalog commitment, assigned source topic, case, sheet, troubleshooting artifact and collection index. Work proceeds in local prerequisite/index order; no model substitution is authorized. The integration owner retains the shared root registers and Part 11's top-level index and ledger.

The launch-time inventory contains 600 Markdown files: Part 01 has 31, Part 02 has 27, Part 03 has 27, Part 04 has 61, Part 05 has 54, Part 06 has 25, Part 07 has 29, Part 08 has 30, Part 09 has 23, Part 10 has 18, and Part 11 has 251. These are artifact counts, not accepted lesson counts. The existing Part 11 ledger retains all 58 nested modules; assigning its 15 family authors does not narrow that scope.

Executed source-ownership smoke: all **1,247 crosswalk entries** resolve to **exactly one** of the 25 exclusive subtree assignments, with zero orphaned or overlapping destinations. Part 11's source assignment remains **388 entries**, and its retained ledger reports **15 families / 58 nested modules / 219 descendant indexes**. Families without supplied source rows still retain their original local catalog commitments. This checks assignment completeness only, not teaching completeness.

Scheduling acceptance does not establish provider availability, executed authorship or completion. Provider errors and returned verification must be recorded before changing acceptance status. Existing checkpoints are retained; no part is newly certified complete by this launch.

### Quota failure and measured-defect re-dispatch

The earlier 25-job `openai-codex/gpt-5.6-terra:medium` authoring batch ended with provider `usage_limit_reached` errors, including resumed jobs. Three workers first returned catalog/ledger checkpoints that did not satisfy their full authoring assignments. No part was accepted complete. Interrupted jobs may have saved partial manuscript content; a failed job does not imply that its files are unchanged.

An earlier structural screen covered **275** individual manuscripts and flagged **239** for at least one of: fewer than 1,000 physical lines, an incorrect numbered H2 sequence, or long prose repeated four or more times. Earlier counts from smaller snapshots must not be combined with that total. One observed rejection pattern was mechanical line inflation: `05_ecommerce.../Case_Studies/01_Search_Relevance_Recovery.md` had 872 lines but only **66 distinct** long prose lines, with three generic investigation sentences each repeated **100 times**. Length and section order alone do not establish substantive acceptance.

Replacement jobs were dispatched to `ollama-cloud/glm-5.3` with named manuscript defects. That wave was interrupted; no full-subtree authoring acceptance was recorded. Partial writes must be inspected rather than assumed absent.

A subsequent screen found **275 manuscripts**, of which **36 passed** the limited structural screen and **239 failed** it. Passing that screen does not establish correctness, substantive depth, answered-question counts, complete troubleshooting entries or reviewed evidence. Those 239 candidates were dispatched to the `CourseRepair` work pool. Its launch omitted an explicit model/effort selector; the earlier claim that medium effort was pinned was not established by that call. File-level ownership reduces conflicting writes but does not prevent interrupted or incomplete manuscripts.

### Actual remaining scope, measured

Repairing existing files is only part of the gap. An earlier heuristic found **744 catalog rows** marked `planned` without a file link and generated **729 candidate filenames**. These are **not** 729 verified distinct missing teaching topics: repeated module/part catalog commitments, existing differently named files and non-topic rows require reconciliation. That scan also does not establish complete coverage of the source crosswalk or topics recorded outside collection catalogs.

The previous dispatch queues contained:

| Queue | Items | Purpose |
|---|---:|---|
| Repair | 239 | Existing-manuscript candidates from the structural screen |
| Create | 729 | Heuristic catalog candidates, not a deduplicated completeness manifest |

Both pools have now returned, with failures and incomplete saved manuscripts. An observed DeepSeek provider error was HTTP 429 with `session usage limit`; neither pool remains an active continuation queue. Returned files need individual verification before acceptance. The earlier snapshot also flagged **196 collection indexes** for structural work. Indexes can truthfully list actual draft artifacts alongside plain-text planned commitments before all teaching content is finished; no draft should be represented as reviewed or published.

### Renewed Codex request — case studies first

The user renewed the complete Parts 01–11 scope and explicitly prioritised case studies. Two concrete assignments were launched with the exact selector `openai-codex/gpt-5.6-terra:medium`: finish the interrupted Module 01 portfolio case, and author the original Module 11 credit-risk case in its canonical module collection. The part-level credit-risk row is a navigation commitment to that same case, not a reason to create a duplicate manuscript.

A fresh file-level snapshot found **118 case files**, with **16** passing the limited length/H2/repetition screen, and **171 cheat sheets**, with **51** passing it. Cases: 47 below 1,000 lines, 14 with an incorrect H2 sequence, 85 with repeated long prose; sheets: 27 below 1,000 lines, 35 with an incorrect H2 sequence, 97 with repeated long prose. Failure categories overlap. The 1,176-line portfolio case stopped after Section 11, demonstrating why physical length is not case completion. No part is certified complete by this snapshot or by the new job launches.

Integration smoke after the interrupted authoring: **783 Markdown files / 3,564 local links / zero missing paths / zero missing anchors**. Thirty-five broken anchor occurrences were resolved against actual current headings, or replaced with explicit pending-card status where the target section was not authored. Module 44's commerce collection now has all eleven index sections and a real fifteen-symbol glossary matching its manuscript; its recommendation, pricing and final troubleshooting commitments remain explicitly unwritten. Module 01's case acceptance checkbox was corrected because the inspected portfolio manuscript lacked Sections 12–15.