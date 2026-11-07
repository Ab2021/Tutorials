# Exact source-topic crosswalk

**Scope:** every visible node in [directory.txt](../Templates/directory.txt), which matches the first fenced tree in the [archived directory audit](../Templates/archive/TRAIN_DIRECTORY_STRUCTURE_AUDIT_56_IMAGES.md). The audit is a screenshot reconstruction, not a filesystem export. **Coverage state:** primarily planned; see [authored drafts](INDEX.md#authored-drafts) for work started.

## 1. Accounting and meaning

All **1,247 source entries** are assigned exactly once to an existing teaching home and an explicit anchored brief. This does not claim 1,247 distinct topics or lessons: source modules, cheat sheets, cases and repeated references overlap. Every visible technical title remains an explicit topic commitment in the linked registers; original course topics are retained. The counts below classify the original inventory, not the current editorial state: individual register entries may now be marked `partial-draft`.

| Disposition | Source entries | Meaning |
|---|---:|---|
| planned-topic | 941 | Original inventory classification for explicit visible technical topic/case/test-artifact scope; linked registers track subsequent partial drafts, and most topics are not yet authored or reviewed |
| supporting-artifact | 106 | Source navigation, generic exercises, references, PDFs, collators, archives and planning aids; not additional completed lessons |
| source-container | 126 | Visible organizing directory; its known children are separately accounted |
| unresolved-source | 74 | Collapsed contents or clipped names; preserve uncertainty and visible prefixes, never invent missing content |

**Zero unassigned entries; zero duplicated source assignments.** The 74 unresolved entries are not omitted from the inventory, but their unavailable contents cannot be guaranteed covered. Known child titles under a clipped directory can be planned while the full original path remains uncertain. “All visible topics planned” must never be shortened to “all source contents complete.”

## 2. Detailed plans

| Plan | Focused teaching groups | Assigned source entries |
|---|---:|---:|
| [Agents, Claude, LangChain/LangGraph and inference](00_SOURCE_COVERAGE_AGENTS.md) | 42 | 315 |
| [CUDA, fine-tuning, distributed training, VLM and speech](00_SOURCE_COVERAGE_CUDA_TRAINING.md) | 50 | 322 |
| [Biomedical edge, streaming, insurance and industry](00_SOURCE_COVERAGE_EDGE_INDUSTRY.md) | 26 | 245 |
| [Data science methods, production cases and graphs](00_SOURCE_COVERAGE_DATA_SCIENCE.md) | 26 | 365 |

Each of the **144 groups** supplies an existing home, a testable outcome, derivations/mechanisms, an illustrative numerical/trace exercise, alternative/failure pressure tests, and a bounded case decision/validation protocol. Its exact-source table names every assigned entry; each technical filename is a focused-topic minimum, not permission to replace detailed reasoning with a survey. The four plans and their module backlinks are syllabus extensions.

## 3. Source-family accounting

| Supplied top-level family | Entries | Planned topic entries | Supporting artifacts | Containers | Unresolved | Plan |
|---|---:|---:|---:|---:|---:|---|
| `.claude` | 1 | 0 | 0 | 0 | 1 | [Agents, Claude, LangChain/LangGraph and inference](00_SOURCE_COVERAGE_AGENTS.md) |
| `agents` | 1 | 0 | 0 | 0 | 1 | [Agents, Claude, LangChain/LangGraph and inference](00_SOURCE_COVERAGE_AGENTS.md) |
| `Agents_Stack_Infra` | 174 | 99 | 42 | 16 | 17 | [Agents, Claude, LangChain/LangGraph and inference](00_SOURCE_COVERAGE_AGENTS.md) |
| `Case_Studies_Streaming_Kafka_Flink_Redpanda` | 24 | 18 | 1 | 5 | 0 | [Biomedical edge, streaming, insurance and industry](00_SOURCE_COVERAGE_EDGE_INDUSTRY.md) |
| `Claude_Architect` | 92 | 65 | 2 | 22 | 3 | [Agents, Claude, LangChain/LangGraph and inference](00_SOURCE_COVERAGE_AGENTS.md) |
| `CUDA` | 141 | 87 | 18 | 14 | 22 | [CUDA, fine-tuning, distributed training, VLM and speech](00_SOURCE_COVERAGE_CUDA_TRAINING.md) |
| `deep_dive` | 43 | 37 | 2 | 3 | 1 | [Biomedical edge, streaming, insurance and industry](00_SOURCE_COVERAGE_EDGE_INDUSTRY.md) |
| `DS_Domain` | 360 | 313 | 17 | 26 | 4 | [Data science methods, production cases and graphs](00_SOURCE_COVERAGE_DATA_SCIENCE.md) |
| `Finetune_LLM` | 1 | 0 | 0 | 0 | 1 | [CUDA, fine-tuning, distributed training, VLM and speech](00_SOURCE_COVERAGE_CUDA_TRAINING.md) |
| `finetuning_course` | 180 | 143 | 8 | 23 | 6 | [CUDA, fine-tuning, distributed training, VLM and speech](00_SOURCE_COVERAGE_CUDA_TRAINING.md) |
| `GLCC` | 1 | 0 | 0 | 0 | 1 | [Biomedical edge, streaming, insurance and industry](00_SOURCE_COVERAGE_EDGE_INDUSTRY.md) |
| `Graph_ML_2025` | 5 | 4 | 0 | 1 | 0 | [Data science methods, production cases and graphs](00_SOURCE_COVERAGE_DATA_SCIENCE.md) |
| `HighEnd` | 152 | 125 | 8 | 9 | 10 | [Biomedical edge, streaming, insurance and industry](00_SOURCE_COVERAGE_EDGE_INDUSTRY.md) |
| `inference` | 15 | 9 | 2 | 3 | 1 | [Agents, Claude, LangChain/LangGraph and inference](00_SOURCE_COVERAGE_AGENTS.md) |
| `Insurance_Analytical_Modelling` | 22 | 15 | 3 | 2 | 2 | [Biomedical edge, streaming, insurance and industry](00_SOURCE_COVERAGE_EDGE_INDUSTRY.md) |
| `LangGraph_LangChain_2025` | 32 | 26 | 3 | 2 | 1 | [Agents, Claude, LangChain/LangGraph and inference](00_SOURCE_COVERAGE_AGENTS.md) |
| `learn` | 1 | 0 | 0 | 0 | 1 | [Biomedical edge, streaming, insurance and industry](00_SOURCE_COVERAGE_EDGE_INDUSTRY.md) |
| `raw` | 1 | 0 | 0 | 0 | 1 | [Biomedical edge, streaming, insurance and industry](00_SOURCE_COVERAGE_EDGE_INDUSTRY.md) |
| `speechtotxt` | 1 | 0 | 0 | 0 | 1 | [Biomedical edge, streaming, insurance and industry](00_SOURCE_COVERAGE_EDGE_INDUSTRY.md) |

## 4. Machine-readable authority and lookup

[00_SOURCE_TOPIC_CROSSWALK.json](00_SOURCE_TOPIC_CROSSWALK.json) contains one ordered row per reconstructed source node:

- `source_path`: exact label-derived path from the supplied tree; uncertainty markers are provenance annotations, not guaranteed literal filesystem names.
- `destination`: existing course-root-relative module/domain `INDEX.md`.
- `plan_file` and `plan_anchor`: the focused teaching brief and its explicit HTML anchor.
- `disposition`: one of the four roles above.

Search a source filename in the JSON or the four plans, follow its brief, then follow its teaching home. From a destination INDEX, follow “Source-tree coverage additions” back to the brief. The local Case_Studies/Cheat_Sheets indexes apply the shared contract; no links to unwritten lessons are fabricated.

## 5. Mathematical and template acceptance

The [authoring and numerical standard](00_AUTHORING_AND_NUMERICAL_STANDARDS.md) applies to every collection. It requires the case master’s 15 sections and seven-section INDEX, the sheet master’s 12 sections and 11-section INDEX, formula validity/derivation/five-part anatomy, independently checked topic-specific calculations, diagnostic failures, answered interview packs, honest evidence and final cards last. Non-mathematical topics use precise invariants and checked traces rather than decorative equations. Source-specific Common Issues counts remain unfilled targets until complete distinct entries exist.

The original five-row module topic lists and compact-card instructions cannot establish this coverage alone. The source plans explicitly add named agent architectures, framework/protocol semantics, CUDA instruction/layout/pipeline details, modern alignment/parallel-training objectives, speech/VLM adaptation, detailed actuarial/DS methods, biomedical signals/ARM/compilers and all visible source cases.

## 6. Verification and limits

Verification method: compare the exact source-entry set to the JSON entries; reject duplicates, missing or extra paths; resolve each destination and plan anchor; check every source entry occurs in its assigned brief; resolve course Markdown navigation and required authoring links; check root collection section counts; execute independent arithmetic for the shared worked example. The recorded results are maintained in the [coverage matrix](00_AUDIT_COVERAGE_MATRIX.md).

This verifies inventory/navigation and the exercised example arithmetic, not factual correctness of inaccessible source lessons, future equations, benchmark results, interview answers or hidden contents. Before authoring named-company/system cases, obtain authoritative primary evidence or label the design illustrative.
