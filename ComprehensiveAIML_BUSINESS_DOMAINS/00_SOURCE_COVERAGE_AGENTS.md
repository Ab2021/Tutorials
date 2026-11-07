# Agent stacks, Claude architecture, LangChain/LangGraph and inference — source coverage

**State:** planned topic architecture and future-content briefs. No full lesson, implemented system, measured outcome, completed issue entry or reviewed interview answer is claimed. Source inventory reviewed for planning only.

## Scope and authoring authority

This plan assigns **315 exact visible source entries** from the first fenced tree in the [source audit](../Templates/archive/TRAIN_DIRECTORY_STRUCTURE_AUDIT_56_IMAGES.md), matching the supplied tree. It extends the local course indexes rather than replacing their existing topics. Source contents were not supplied; names establish syllabus scope, not technical correctness or source access.

Apply the [course authoring and numerical standards](00_AUTHORING_AND_NUMERICAL_STANDARDS.md), [case master](../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md), and [cheat-sheet master](../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md). Future cases require 15 sections and a seven-section collection INDEX; future sheets require 12 sections and an 11-section collection INDEX. Each substantive equation needs validity, derivation and FORMULA → WHERE → CONSTRAINT → NUMERICAL EXAMPLE → INTERPRETATION, with units and a counterexample. Use a precise invariant/trace where a closed-form equation is not meaningful.

Each visible technical title in the registers below is a **separate focused-topic minimum**. Split independent methods into focused sheets; reuse a genuine equivalent sheet rather than duplicating it. A group-level exercise does not replace the topic-specific derivations, numerical examples, four diagnostic failures, checked interview answers, source ledger and final card required when each sheet is authored. Each distinct source case remains a separately framed case, not an oversized anthology.

All exercise inputs are **ILLUSTRATIVE** and all validation protocols are **proposed, not executed**. No named company, harness, framework, device, regulator or frontier model behavior is certified by a filename. Pin primary sources, versions, geography and dates before attributing architecture, rules or results. Existing 50/100 Common Issues titles remain targets; none are counted as completed.

Hidden folders and clipped filenames remain `unresolved-source`; visible prefixes and child titles may support limited topic planning but their full path/content is not invented. Ancillary READMEs, source PDFs, collators, archives and generic exercises are tracked without counting them as authored lessons. The permanent [source crosswalk](00_SOURCE_TOPIC_CROSSWALK.md) records every node and disposition.

## Focused teaching groups

- [Python async, concurrency, REST, WebSockets, LLM basics and Docker](#agent-prerequisites)
- [Overloaded agent term, ten-layer stack, request path and boundary distinctions](#agent-stack-model)
- [Infrastructure substrate, model asset, serving systems, interaction contracts and long-context stress](#agent-models-serving)
- [Control planes, session ownership, state management, isolation and run lifecycle](#agent-control-sessions)
- [Runtime responsibilities, durable execution, LangGraph, workflow patterns and runtime choice](#agent-runtimes)
- [Context, memory, transcript, assembly, RAG, Self-RAG, memory lifecycle/types and Sourcegraph code context](#agent-context-memory)
- [Tool schemas, MCP, capability versus execution, tool risk, MCP server and Sourcegraph MCP](#agent-tools-mcp)
- [Reflection, tool use, planning, multi-agent, orchestration, human review, composition and emerging patterns](#agent-patterns)
- [Execution surfaces, identity, trust, approval gates, sandboxing and blast radius](#agent-trust-approvals)
- [Observability versus evaluation, run tracing, evaluation frameworks, feedback loops and monitoring](#agent-observability)
- [OpenClaw gateway, sessions, lane-aware queues, WebSockets, runtime loop, file memory, concurrency and security](#agent-openclaw)
- [Source-specific Claude Code harness philosophy, Go-native engine, five-verb skills, PLANS.md, parallelism and integration](#agent-claude-harness)
- [Hermes self-improvement, autonomous skills, persistent memory, user modeling, platform deployment, tools and OpenClaw comparison](#agent-hermes)
- [NVIDIA hardware, CUDA operations, nvidia-smi, MIG, container toolkit, Kubernetes, DCGM, DGX/Base Command and TensorRT](#agent-nca-infrastructure)
- [Implementation practices, production patterns, domain uses, multi-agent operations, reliability, governance, cost, scaling, pilot rollout and checklists](#agent-production)
- [Agentic architecture: loop lifecycle, orchestration, subagent context, enforcement/handoff, SDK hooks, decomposition, sessions and practice](#claude-architecture)
- [Tool descriptions/boundaries, structured errors, distribution/tool choice, MCP integration, built-ins and practice](#claude-mcp)
- [CLAUDE.md hierarchy, commands/skills, path rules, plan versus direct mode, iterative refinement, CI/CD and practice](#claude-configuration)
- [Explicit criteria, few-shot prompting, structured output/tool use, validation/retry feedback, batch work, multi-pass review and practice](#claude-prompting)
- [Context preservation, escalation, error propagation, large-codebase exploration, human-review confidence, provenance/uncertainty and practice](#claude-reliability)
- [clinical decision support](#scenario-07)
- [legal contract analysis](#scenario-08)
- [financial fraud agents](#scenario-09)
- [predictive maintenance agents](#scenario-10)
- [SOC automation](#scenario-11)
- [claims automation](#scenario-12)
- [retail/e-commerce agents](#scenario-13)
- [research acceleration](#scenario-14)
- [production anti-patterns and MCP integration](#scenario-15)
- [Support, code generation, multi-agent research, developer productivity, CI/CD and structured extraction scenarios](#claude-scenarios)
- [Hands-on multi-tool agent, code configuration, extraction and multi-agent pipeline exercises](#claude-practice)
- [API, Agent SDK, MCP, Claude Code, anti-patterns and decision-flow references](#claude-reference)
- [Course overview, exam framing, practice questions and 60-question practice exam](#claude-exam)
- [LangChain foundations, ecosystem, LCEL, prompts/models/parsers](#lang-foundations)
- [LangGraph state, nodes/edges, agents, tools and multi-agent systems](#lang-graphs)
- [LangGraph memory/persistence and session isolation](#lang-memory)
- [LangChain RAG retrieval pipelines](#lang-rag)
- [Streaming, real-time output and interruption](#lang-streaming)
- [LangSmith observability, production deployment and dated emerging patterns](#lang-ops)
- [100 Common LangGraph/LangChain Issues](#lang-issues)
- [Six inference clusters, tool contracts, mock responses, manifest, archive and explainer](#inference-clusters)
- [Agent source references, hidden folders, PDFs and incomplete exam/harness evidence](#agent-source-artifacts)

<a id="agent-prerequisites"></a>

## Python async, concurrency, REST, WebSockets, LLM basics and Docker

**Teaching home:** [01_foundations_computing_and_problem_solving/02_python_sql_linux_and_engineering_basics/INDEX.md](01_foundations_computing_and_problem_solving/02_python_sql_linux_and_engineering_basics/INDEX.md). **Status:** planned.

**Testable outcome.** Trace asynchronous tasks, cancellation, API exchanges and isolated reproducible environments.

**Focused-sheet derivations and mechanisms.** Derive concurrency from request rate and mean service time under stable boundaries; distinguish CPU parallelism from async I/O.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 20 requests/s, mean 0.25 s per request: estimate mean in-flight work; trace cancellation with five tasks and one timeout. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose async I/O versus worker processes for a tool gateway with CPU-bound parsing and network calls.

**Proposed validation.** Exercise event-loop blocking, timeout cleanup, request/response schemas and container reproducibility.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Blocking event loop; orphan tasks; transport disconnect; mismatched environment. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Agents_Stack_Infra/Module_00_Prerequisites` | Source scope/container; not a completed lesson | `source-container` |
| `Agents_Stack_Infra/Module_00_Prerequisites/code` | OPEN — code/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra/Module_00_Prerequisites/01_python_async_and_concurrency.md` | python async and concurrency | `planned-topic` |
| `Agents_Stack_Infra/Module_00_Prerequisites/02_rest_apis_and_websockets.md` | rest apis and websockets | `planned-topic` |
| `Agents_Stack_Infra/Module_00_Prerequisites/03_llm_fundamentals.md` | llm fundamentals | `planned-topic` |
| `Agents_Stack_Infra/Module_00_Prerequisites/04_docker_and_containers.md` | docker and containers | `planned-topic` |
| `Agents_Stack_Infra/Module_00_Prerequisites/exercises.md` | exercises | `supporting-artifact` |
| `Agents_Stack_Infra/Module_00_Prerequisites/README.md` | README | `supporting-artifact` |

<a id="agent-stack-model"></a>

## Overloaded agent term, ten-layer stack, request path and boundary distinctions

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md). **Status:** planned.

**Testable outcome.** Assign ownership for each state/control/execution boundary and predict failures when layers collapse.

**Focused-sheet derivations and mechanisms.** Derive a request/state ownership trace; distinguish model asset, serving, control plane, runtime and execution surface.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Trace a 3-tool run with two sessions and a retried request; count which effects may occur once versus twice. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose explicit session/runtime/tool boundaries rather than one undifferentiated agent service.

**Proposed validation.** Replay duplicate requests and concurrent sessions; verify state owner and effect journal.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Collapsed trust boundaries; double-owned state; ambiguous retry owner; cross-session leakage. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Agents_Stack_Infra/Module_01_Agent_Stack_Overview` | Source scope/container; not a completed lesson | `source-container` |
| `Agents_Stack_Infra/Module_01_Agent_Stack_Overview/code` | OPEN — code/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra/Module_01_Agent_Stack_Overview/01_why_agent_is_overloaded.md` | why agent is overloaded | `planned-topic` |
| `Agents_Stack_Infra/Module_01_Agent_Stack_Overview/02_ten_layer_stack_model.md` | ten layer stack model | `planned-topic` |
| `Agents_Stack_Infra/Module_01_Agent_Stack_Overview/03_the_request_path.md` | the request path | `planned-topic` |
| `Agents_Stack_Infra/Module_01_Agent_Stack_Overview/04_critical_boundary_distinctions.md` | critical boundary distinctions | `planned-topic` |
| `Agents_Stack_Infra/Module_01_Agent_Stack_Overview/05_what_breaks_when_layers_collapse.md` | what breaks when layers collapse | `planned-topic` |
| `Agents_Stack_Infra/Module_01_Agent_Stack_Overview/06_builder_checklist.md` | builder checklist | `planned-topic` |
| `Agents_Stack_Infra/Module_01_Agent_Stack_Overview/exercises.md` | exercises | `supporting-artifact` |
| `Agents_Stack_Infra/Module_01_Agent_Stack_Overview/README.md` | README | `supporting-artifact` |

<a id="agent-models-serving"></a>

## Infrastructure substrate, model asset, serving systems, interaction contracts and long-context stress

**Teaching home:** [05_ai_platforms_data_systems_and_production_engineering/25_model_serving_inference_optimisation_and_cost/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/25_model_serving_inference_optimisation_and_cost/INDEX.md). **Status:** planned.

**Testable outcome.** Separate model capabilities from serving/runtime contracts and diagnose lower-layer failures.

**Focused-sheet derivations and mechanisms.** Derive context-token and KV-memory budgets with stated model dimensions and dtype; distinguish model quality from serving behavior.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Assume 8192-token context, 1024 reserved output and 512 instructions: compute remaining retrieval/tool capacity; vary reserve to 2048. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose serving and truncation policy for a long-context assistant without claiming context length guarantees retrieval quality.

**Proposed validation.** Sweep context lengths and distractors; record truncation, task correctness, memory and timeout signals.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Context overflow; unsupported tool schema; serving overload; model/endpoint version drift. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Agents_Stack_Infra/Module_02_Infrastructure_and_Models` | Source scope/container; not a completed lesson | `source-container` |
| `Agents_Stack_Infra/Module_02_Infrastructure_and_Models/code` | OPEN — code/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra/Module_02_Infrastructure_and_Models/01_infrastructure_substrate.md` | infrastructure substrate | `planned-topic` |
| `Agents_Stack_Infra/Module_02_Infrastructure_and_Models/02_model_asset.md` | model asset | `planned-topic` |
| `Agents_Stack_Infra/Module_02_Infrastructure_and_Models/03_serving_systems.md` | serving systems | `planned-topic` |
| `Agents_Stack_Infra/Module_02_Infrastructure_and_Models/04_interaction_contracts.md` | interaction contracts | `planned-topic` |
| `Agents_Stack_Infra/Module_02_Infrastructure_and_Models/05_long_context_stress_test.md` | long context stress test | `planned-topic` |
| `Agents_Stack_Infra/Module_02_Infrastructure_and_Models/06_lower_layer_failure_modes.md` | lower layer failure modes | `planned-topic` |
| `Agents_Stack_Infra/Module_02_Infrastructure_and_Models/exercises.md` | exercises | `supporting-artifact` |
| `Agents_Stack_Infra/Module_02_Infrastructure_and_Models/README.md` | README | `supporting-artifact` |

<a id="agent-control-sessions"></a>

## Control planes, session ownership, state management, isolation and run lifecycle

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md). **Status:** planned.

**Testable outcome.** Specify tenant/session/run identity and single-writer or conflict-resolution invariants.

**Focused-sheet derivations and mechanisms.** Derive permitted run state transitions and conflict handling from ownership; trace authenticated resume.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Two writers read version 7 and propose updates: trace compare-and-swap acceptance, conflict and safe retry. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose session persistence and ownership to prevent cross-tenant or concurrent-run corruption.

**Proposed validation.** Race two writers, expire sessions, restart owners and test unauthorized resume.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Lost update; stale resume; cross-session access; orphaned run. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Agents_Stack_Infra/Module_03_Control_Planes_and_Sessions` | Source scope/container; not a completed lesson | `source-container` |
| `Agents_Stack_Infra/Module_03_Control_Planes_and_Sessions/code` | OPEN — code/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra/Module_03_Control_Planes_and_Sessions/01_what_is_a_control_plane.md` | what is a control plane | `planned-topic` |
| `Agents_Stack_Infra/Module_03_Control_Planes_and_Sessions/02_session_ownership.md` | session ownership | `planned-topic` |
| `Agents_Stack_Infra/Module_03_Control_Planes_and_Sessions/03_state_management.md` | state management | `planned-topic` |
| `Agents_Stack_Infra/Module_03_Control_Planes_and_Sessions/04_session_isolation.md` | session isolation | `planned-topic` |
| `Agents_Stack_Infra/Module_03_Control_Planes_and_Sessions/05_run_lifecycle.md` | run lifecycle | `planned-topic` |
| `Agents_Stack_Infra/Module_03_Control_Planes_and_Sessions/exercises.md` | exercises | `supporting-artifact` |
| `Agents_Stack_Infra/Module_03_Control_Planes_and_Sessions/README.md` | README | `supporting-artifact` |

<a id="agent-runtimes"></a>

## Runtime responsibilities, durable execution, LangGraph, workflow patterns and runtime choice

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md). **Status:** planned.

**Testable outcome.** Recover an interrupted workflow without duplicating an external action.

**Focused-sheet derivations and mechanisms.** Derive checkpoint/replay boundaries and idempotency-key state transitions; separate durable state from durable effects.

**Numerical/trace exercise to author.** ILLUSTRATIVE: A 4-step workflow crashes after step 3 external commit but before checkpoint: trace replay and reconciliation with a stable operation ID. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Select graph runtime versus a simple deterministic workflow for a resumable approval flow.

**Proposed validation.** Inject crashes before/after external effects, change schemas and verify approved-payload binding.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Duplicate side effect; stale checkpoint; non-deterministic replay; incompatible state migration. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Agents_Stack_Infra/Module_04_Runtimes_and_Workflows` | Source scope/container; not a completed lesson | `source-container` |
| `Agents_Stack_Infra/Module_04_Runtimes_and_Workflows/code` | OPEN — code/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra/Module_04_Runtimes_and_Workflows/01_what_is_a_runtime.md` | what is a runtime | `planned-topic` |
| `Agents_Stack_Infra/Module_04_Runtimes_and_Workflows/02_durable_execution.md` | durable execution | `planned-topic` |
| `Agents_Stack_Infra/Module_04_Runtimes_and_Workflows/03_langgraph_deep_dive.md` | langgraph deep dive | `planned-topic` |
| `Agents_Stack_Infra/Module_04_Runtimes_and_Workflows/04_workflow_patterns.md` | workflow patterns | `planned-topic` |
| `Agents_Stack_Infra/Module_04_Runtimes_and_Workflows/05_choosing_runtime.md` | choosing runtime | `planned-topic` |
| `Agents_Stack_Infra/Module_04_Runtimes_and_Workflows/exercises.md` | exercises | `supporting-artifact` |
| `Agents_Stack_Infra/Module_04_Runtimes_and_Workflows/README.md` | README | `supporting-artifact` |

<a id="agent-context-memory"></a>

## Context, memory, transcript, assembly, RAG, Self-RAG, memory lifecycle/types and Sourcegraph code context

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/32_agent_memory_context_management_and_personalisation/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/32_agent_memory_context_management_and_personalisation/INDEX.md). **Status:** planned.

**Testable outcome.** Construct permission-correct context and distinguish retained facts from transient transcripts.

**Focused-sheet derivations and mechanisms.** Derive a token-allocation constraint, retrieval recall and memory expiry/update invariants; Self-RAG decisions need explicit evidence.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 6000-token budget: 700 instructions, 1000 history, 500 tools and 4 chunks of 800 tokens; calculate remaining output allowance and test one extra chunk. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose retrieval and memory retention for a repository assistant with revocable access.

**Proposed validation.** Test stale/deleted memories, conflicting facts, retrieval misses, code revision changes and authorization filters.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Stale memory; permission leakage; context truncation; embedding/index drift. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Agents_Stack_Infra/Module_05_Context_Memory_Retrieval` | Source scope/container; not a completed lesson | `source-container` |
| `Agents_Stack_Infra/Module_05_Context_Memory_Retrieval/code` | OPEN — code/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra/Module_05_Context_Memory_Retrieval/01_context_vs_memory_vs_transcript.md` | context vs memory vs transcript | `planned-topic` |
| `Agents_Stack_Infra/Module_05_Context_Memory_Retrieval/02_context_assembly.md` | context assembly | `planned-topic` |
| `Agents_Stack_Infra/Module_05_Context_Memory_Retrieval/03_retrieval_augmented_generation.md` | retrieval augmented generation | `planned-topic` |
| `Agents_Stack_Infra/Module_05_Context_Memory_Retrieval/04_self_rag.md` | self rag | `planned-topic` |
| `Agents_Stack_Infra/Module_05_Context_Memory_Retrieval/05_memory_lifecycle.md` | memory lifecycle | `planned-topic` |
| `Agents_Stack_Infra/Module_05_Context_Memory_Retrieval/06_memory_types_and_patterns.md` | memory types and patterns | `planned-topic` |
| `Agents_Stack_Infra/Module_05_Context_Memory_Retrieval/07_sourcegraph_code_context.md` | sourcegraph code context | `planned-topic` |
| `Agents_Stack_Infra/Module_05_Context_Memory_Retrieval/exercises.md` | exercises | `supporting-artifact` |
| `Agents_Stack_Infra/Module_05_Context_Memory_Retrieval/README.md` | README | `supporting-artifact` |

<a id="agent-tools-mcp"></a>

## Tool schemas, MCP, capability versus execution, tool risk, MCP server and Sourcegraph MCP

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/31_tools_mcp_protocols_and_enterprise_integrations/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/31_tools_mcp_protocols_and_enterprise_integrations/INDEX.md). **Status:** planned.

**Testable outcome.** Design typed tool contracts and permission-gated execution with observable failures.

**Focused-sheet derivations and mechanisms.** Derive validation/authorization/execution state and idempotent operation contracts; protocol version must be pinned.

**Numerical/trace exercise to author.** ILLUSTRATIVE: A 100-record batch permits 20 records/call and 3 calls/run: compute maximum permitted processed records and trace partial failures. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose a narrow MCP tool surface versus broad shell access for code-search and controlled edits.

**Proposed validation.** Validate malformed inputs, expired credentials, untrusted tool output, unauthorized paths and replayed writes.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Schema-valid but unauthorized input; broad capabilities; repeated writes; server/protocol mismatch. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Agents_Stack_Infra/Module_06_Tools_MCP_and_Capabilities` | Source scope/container; not a completed lesson | `source-container` |
| `Agents_Stack_Infra/Module_06_Tools_MCP_and_Capabilities/code` | OPEN — code/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra/Module_06_Tools_MCP_and_Capabilities/01_tool_schemas_and_definitions.md` | tool schemas and definitions | `planned-topic` |
| `Agents_Stack_Infra/Module_06_Tools_MCP_and_Capabilities/02_model_context_protocol.md` | model context protocol | `planned-topic` |
| `Agents_Stack_Infra/Module_06_Tools_MCP_and_Capabilities/03_capability_vs_execution.md` | capability vs execution | `planned-topic` |
| `Agents_Stack_Infra/Module_06_Tools_MCP_and_Capabilities/04_tool_categories_and_risk.md` | tool categories and risk | `planned-topic` |
| `Agents_Stack_Infra/Module_06_Tools_MCP_and_Capabilities/05_mcp_server_implementation.md` | mcp server implementation | `planned-topic` |
| `Agents_Stack_Infra/Module_06_Tools_MCP_and_Capabilities/06_sourcegraph_mcp_integration.md` | sourcegraph mcp integration | `planned-topic` |
| `Agents_Stack_Infra/Module_06_Tools_MCP_and_Capabilities/exercises.md` | exercises | `supporting-artifact` |
| `Agents_Stack_Infra/Module_06_Tools_MCP_and_Capabilities/README.md` | README | `supporting-artifact` |

<a id="agent-patterns"></a>

## Reflection, tool use, planning, multi-agent, orchestration, human review, composition and emerging patterns

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md). **Status:** planned.

**Testable outcome.** Select a bounded pattern composition rather than adding autonomous agents by default.

**Focused-sheet derivations and mechanisms.** Derive cost/termination bounds from step and branch budgets; compare sequential, parallel and supervisor flows.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Planner allows 4 branches, 3 calls each, 500 tokens/call: calculate worst permitted call and token budgets; halve branch limit. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose single workflow versus multi-agent decomposition for a research answer with independent evidence checks.

**Proposed validation.** Compare task success, evidence quality, budget exhaustion and disagreement under fixed inputs.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Unbounded reflection; delegation loops; correlated reviewers; unchecked human handoff. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Agents_Stack_Infra/Module_07_Agentic_Design_Patterns` | Source scope/container; not a completed lesson | `source-container` |
| `Agents_Stack_Infra/Module_07_Agentic_Design_Patterns/code` | OPEN — code/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra/Module_07_Agentic_Design_Patterns/01_patterns_overview.md` | patterns overview | `planned-topic` |
| `Agents_Stack_Infra/Module_07_Agentic_Design_Patterns/02_reflection_pattern.md` | reflection pattern | `planned-topic` |
| `Agents_Stack_Infra/Module_07_Agentic_Design_Patterns/03_tool_use_pattern.md` | tool use pattern | `planned-topic` |
| `Agents_Stack_Infra/Module_07_Agentic_Design_Patterns/04_planning_patterns.md` | planning patterns | `planned-topic` |
| `Agents_Stack_Infra/Module_07_Agentic_Design_Patterns/05_multi_agent_patterns.md` | multi agent patterns | `planned-topic` |
| `Agents_Stack_Infra/Module_07_Agentic_Design_Patterns/06_orchestration_patterns.md` | orchestration patterns | `planned-topic` |
| `Agents_Stack_Infra/Module_07_Agentic_Design_Patterns/07_human_in_the_loop_pattern.md` | human in the loop pattern | `planned-topic` |
| `Agents_Stack_Infra/Module_07_Agentic_Design_Patterns/08_pattern_composition.md` | pattern composition | `planned-topic` |
| `Agents_Stack_Infra/Module_07_Agentic_Design_Patterns/09_emerging_patterns_2025.md` | emerging patterns 2025 | `planned-topic` |
| `Agents_Stack_Infra/Module_07_Agentic_Design_Patterns/exercises.md` | exercises | `supporting-artifact` |
| `Agents_Stack_Infra/Module_07_Agentic_Design_Patterns/README.md` | README | `supporting-artifact` |

<a id="agent-trust-approvals"></a>

## Execution surfaces, identity, trust, approval gates, sandboxing and blast radius

**Teaching home:** [09_trust_security_governance_and_product_leadership/49_ai_security_privacy_safety_and_red_teaming/INDEX.md](09_trust_security_governance_and_product_leadership/49_ai_security_privacy_safety_and_red_teaming/INDEX.md). **Status:** planned.

**Testable outcome.** Bind approval to exact action content and isolate execution from model suggestions.

**Focused-sheet derivations and mechanisms.** Derive least-privilege authorization and approve/expire/revoke transitions; enumerate affected resources.

**Numerical/trace exercise to author.** ILLUSTRATIVE: A proposed batch targets 12 accounts but policy permits 5: trace eligibility filtering and approval expiry after payload changes. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Select sandbox and approval boundaries for a write-capable enterprise agent.

**Proposed validation.** Attempt injection, path traversal, credential reuse, payload mutation and cross-tenant execution.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Stale approval; ambient authority; sandbox escape assumptions; overly broad blast radius. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Agents_Stack_Infra/Module_08_Execution_Trust_Approvals` | Source scope/container; not a completed lesson | `source-container` |
| `Agents_Stack_Infra/Module_08_Execution_Trust_Approvals/code` | OPEN — code/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra/Module_08_Execution_Trust_Approvals/01_execution_surfaces.md` | execution surfaces | `planned-topic` |
| `Agents_Stack_Infra/Module_08_Execution_Trust_Approvals/02_identity_and_trust.md` | identity and trust | `planned-topic` |
| `Agents_Stack_Infra/Module_08_Execution_Trust_Approvals/03_approval_gates.md` | approval gates | `planned-topic` |
| `Agents_Stack_Infra/Module_08_Execution_Trust_Approvals/04_sandboxing_and_isolation.md` | sandboxing and isolation | `planned-topic` |
| `Agents_Stack_Infra/Module_08_Execution_Trust_Approvals/05_blast_radius_analysis.md` | blast radius analysis | `planned-topic` |
| `Agents_Stack_Infra/Module_08_Execution_Trust_Approvals/exercises.md` | exercises | `supporting-artifact` |
| `Agents_Stack_Infra/Module_08_Execution_Trust_Approvals/README.md` | README | `supporting-artifact` |

<a id="agent-observability"></a>

## Observability versus evaluation, run tracing, evaluation frameworks, feedback loops and monitoring

**Teaching home:** [05_ai_platforms_data_systems_and_production_engineering/28_ai_evaluation_monitoring_and_continuous_improvement/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/28_ai_evaluation_monitoring_and_continuous_improvement/INDEX.md). **Status:** mostly planned; [model evaluation and production monitoring](05_ai_platforms_data_systems_and_production_engineering/28_ai_evaluation_monitoring_and_continuous_improvement/Cheat_Sheets/04_Model_Evaluation_and_Production_Monitoring.md) is an unreviewed partial draft, not evidence that unavailable source files were inspected.

**Testable outcome.** Separate diagnosis of what happened from evidence of whether the task succeeded.

**Focused-sheet derivations and mechanisms.** Derive task-success numerator/denominator, trace linkage and cost per successful task; calibration is required for judge scores.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Of 200 runs, 160 finish, 140 meet rubric and total cost is 28 currency units: calculate completion, success and cost/success separately. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose release gates for an assistant whose completion rate improved but factual task success did not.

**Proposed validation.** Use held-out tasks, blinded adjudication, judge-human disagreement and failed/deferred-run accounting.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Proxy metric mismatch; missing spans; selected feedback; judge drift. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Agents_Stack_Infra/Module_09_Observability_and_Evaluation` | Source scope/container; not a completed lesson | `source-container` |
| `Agents_Stack_Infra/Module_09_Observability_and_Evaluation/code` | OPEN — code/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra/Module_09_Observability_and_Evaluation/01_observability_vs_evaluation.md` | observability vs evaluation | `planned-topic` |
| `Agents_Stack_Infra/Module_09_Observability_and_Evaluation/02_tracing_agent_runs.md` | tracing agent runs | `planned-topic` |
| `Agents_Stack_Infra/Module_09_Observability_and_Evaluation/03_evaluation_frameworks.md` | evaluation frameworks | `planned-topic` |
| `Agents_Stack_Infra/Module_09_Observability_and_Evaluation/04_feedback_loops.md` | feedback loops | `planned-topic` |
| `Agents_Stack_Infra/Module_09_Observability_and_Evaluation/05_production_monitoring.md` | production monitoring | `planned-topic` |
| `Agents_Stack_Infra/Module_09_Observability_and_Evaluation/exercises.md` | exercises | `supporting-artifact` |
| `Agents_Stack_Infra/Module_09_Observability_and_Evaluation/README.md` | README | `supporting-artifact` |

<a id="agent-openclaw"></a>

## OpenClaw gateway, sessions, lane-aware queues, WebSockets, runtime loop, file memory, concurrency and security

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md). **Status:** planned.

**Testable outcome.** Reconstruct proposed OpenClaw request and lane scheduling behavior from pinned source, separating it from generic agent patterns.

**Focused-sheet derivations and mechanisms.** Derive per-session serialization and queue conservation under explicitly verified lane semantics.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Two lanes receive 30 and 10 jobs/s with service 20 jobs/s each for 60 s: estimate backlog and explain why aggregate capacity hides a hot lane. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose queue admission and session isolation for a gateway with mixed long and short runs.

**Proposed validation.** Verify actual source revision; replay concurrency, disconnected clients, restart and unauthorized file-memory access.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Hot lane; interleaved session mutation; lost WebSocket client state; unsafe memory paths. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Agents_Stack_Infra/Module_10_OpenClaw_Architecture` | Source scope/container; not a completed lesson | `source-container` |
| `Agents_Stack_Infra/Module_10_OpenClaw_Architecture/code` | OPEN — code/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra/Module_10_OpenClaw_Architecture/01_openclaw_overview.md` | openclaw overview | `planned-topic` |
| `Agents_Stack_Infra/Module_10_OpenClaw_Architecture/02_gateway_and_sessions.md` | gateway and sessions | `planned-topic` |
| `Agents_Stack_Infra/Module_10_OpenClaw_Architecture/03_lane_aware_queue.md` | lane aware queue | `planned-topic` |
| `Agents_Stack_Infra/Module_10_OpenClaw_Architecture/04_websocket_protocol.md` | websocket protocol | `planned-topic` |
| `Agents_Stack_Infra/Module_10_OpenClaw_Architecture/05_agent_runtime_loop.md` | agent runtime loop | `planned-topic` |
| `Agents_Stack_Infra/Module_10_OpenClaw_Architecture/06_memory_as_files.md` | memory as files | `planned-topic` |
| `Agents_Stack_Infra/Module_10_OpenClaw_Architecture/07_concurrency_and_invariants.md` | concurrency and invariants | `planned-topic` |
| `Agents_Stack_Infra/Module_10_OpenClaw_Architecture/08_security_hardening.md` | security hardening | `planned-topic` |
| `Agents_Stack_Infra/Module_10_OpenClaw_Architecture/exercises.md` | exercises | `supporting-artifact` |
| `Agents_Stack_Infra/Module_10_OpenClaw_Architecture/README.md` | README | `supporting-artifact` |

<a id="agent-claude-harness"></a>

## Source-specific Claude Code harness philosophy, Go-native engine, five-verb skills, PLANS.md, parallelism and integration

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/33_coding_agents_ai_sdcl_and_developer_productivity/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/33_coding_agents_ai_sdcl_and_developer_productivity/INDEX.md). **Status:** planned.

**Testable outcome.** Distinguish source-specific harness claims from documented Claude product behavior and specify a reproducible integration.

**Focused-sheet derivations and mechanisms.** Derive skill preconditions, plan/execution state and bounded parallel work; no implementation language is asserted without source.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 8 independent tasks on 3 workers with illustrative 2 s equal durations: compute an ideal scheduling lower bound and a realizable wave schedule. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose native harness integration versus a thin script for bounded parallel repository changes.

**Proposed validation.** Obtain exact harness source; verify commands, state ownership, conflict handling and clean failure propagation.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Unavailable engine evidence; conflicting writes; stale plans; script/native semantic mismatch. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Agents_Stack_Infra/Module_11_Claude_Code_Harness` | Source scope/container; not a completed lesson | `source-container` |
| `Agents_Stack_Infra/Module_11_Claude_Code_Harness/code` | OPEN — code/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra/Module_11_Claude_Code_Harness/01_harness_philosophy.md` | harness philosophy | `planned-topic` |
| `Agents_Stack_Infra/Module_11_Claude_Code_Harness/02_go_native_engine.md` | go native engine | `planned-topic` |
| `Agents_Stack_Infra/Module_11_Claude_Code_Harness/03_five_verb_skills.md` | five verb skills | `planned-topic` |
| `Agents_Stack_Infra/Module_11_Claude_Code_Harness/04_plans_md_anatomy.md` | plans md anatomy | `planned-topic` |
| `Agents_Stack_Infra/Module_11_Claude_Code_Harness/05_parallel_execution.md` | parallel execution | `planned-topic` |
| `Agents_Stack_Infra/Module_11_Claude_Code_Harness/06_advanced_features.md` | advanced features | `planned-topic` |
| `Agents_Stack_Infra/Module_11_Claude_Code_Harness/07_integrating_harness.md` | integrating harness | `planned-topic` |
| `Agents_Stack_Infra/Module_11_Claude_Code_Harness/exercises.md` | exercises | `supporting-artifact` |
| `Agents_Stack_Infra/Module_11_Claude_Code_Harness/README.md` | README | `supporting-artifact` |

<a id="agent-hermes"></a>

## Hermes self-improvement, autonomous skills, persistent memory, user modeling, platform deployment, tools and OpenClaw comparison

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/32_agent_memory_context_management_and_personalisation/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/32_agent_memory_context_management_and_personalisation/INDEX.md). **Status:** planned.

**Testable outcome.** Evaluate skill promotion and persistent user memory without treating self-improvement as established benefit.

**Focused-sheet derivations and mechanisms.** Derive promotion/rollback invariants and retention permissions; compare evolving and fixed policies on held-out tasks.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Candidate skill solves 18/25 validation tasks versus baseline 17/25 but fails 2 safety gates: compute scores and explain why promotion is blocked. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose whether to promote an auto-generated skill for a multi-platform assistant.

**Proposed validation.** Pin source and evaluate held-out transfer, poisoning, user-memory deletion, regression and cross-platform permission differences.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Self-reinforcing error; contaminated evaluation; unwanted user profiling; platform permission drift. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Agents_Stack_Infra/Module_12_Hermes_Self_Evolving_Agent` | Source scope/container; not a completed lesson | `source-container` |
| `Agents_Stack_Infra/Module_12_Hermes_Self_Evolving_Agent/code` | OPEN — code/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra/Module_12_Hermes_Self_Evolving_Agent/01_self_improvement_architecture.md` | self improvement architecture | `planned-topic` |
| `Agents_Stack_Infra/Module_12_Hermes_Self_Evolving_Agent/02_autonomous_skill_generation.md` | autonomous skill generation | `planned-topic` |
| `Agents_Stack_Infra/Module_12_Hermes_Self_Evolving_Agent/03_persistent_memory_system.md` | persistent memory system | `planned-topic` |
| `Agents_Stack_Infra/Module_12_Hermes_Self_Evolving_Agent/04_user_modeling.md` | user modeling | `planned-topic` |
| `Agents_Stack_Infra/Module_12_Hermes_Self_Evolving_Agent/05_multi_platform_deployment.md` | multi platform deployment | `planned-topic` |
| `Agents_Stack_Infra/Module_12_Hermes_Self_Evolving_Agent/06_tool_ecosystem.md` | tool ecosystem | `planned-topic` |
| `Agents_Stack_Infra/Module_12_Hermes_Self_Evolving_Agent/07_comparing_openclaw_and_hermes.md` | comparing openclaw and hermes | `planned-topic` |
| `Agents_Stack_Infra/Module_12_Hermes_Self_Evolving_Agent/exercises.md` | exercises | `supporting-artifact` |
| `Agents_Stack_Infra/Module_12_Hermes_Self_Evolving_Agent/README.md` | README | `supporting-artifact` |

<a id="agent-nca-infrastructure"></a>

## NVIDIA hardware, CUDA operations, nvidia-smi, MIG, container toolkit, Kubernetes, DCGM, DGX/Base Command and TensorRT

**Teaching home:** [05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware/INDEX.md). **Status:** planned.

**Testable outcome.** Plan GPU resource partitioning and diagnose utilization, memory and inference bottlenecks from versioned tooling.

**Focused-sheet derivations and mechanisms.** Derive resource allocation and utilization denominators; distinguish device partitions, scheduler limits and application throughput.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Assumed usable memory 80 GiB, per-tenant reservation 18 GiB and 8 GiB shared reserve: calculate memory-only tenant bound, then discuss valid MIG profiles separately. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose partitioned versus shared GPU scheduling for isolated inference tenants.

**Proposed validation.** Verify supported device/profile combinations and measure memory pressure, contention, telemetry freshness and restore behavior.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Unsupported partition; memory-only sizing error; invisible container GPU; misleading utilization average. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Agents_Stack_Infra/Module_13_NCA_AIIO_Infrastructure` | Source scope/container; not a completed lesson | `source-container` |
| `Agents_Stack_Infra/Module_13_NCA_AIIO_Infrastructure/code` | OPEN — code/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra/Module_13_NCA_AIIO_Infrastructure/01_nvidia_hardware_overview.md` | nvidia hardware overview | `planned-topic` |
| `Agents_Stack_Infra/Module_13_NCA_AIIO_Infrastructure/02_cuda_for_infrastructure.md` | cuda for infrastructure | `planned-topic` |
| `Agents_Stack_Infra/Module_13_NCA_AIIO_Infrastructure/03_nvidia_smi_deep_dive.md` | nvidia smi deep dive | `planned-topic` |
| `Agents_Stack_Infra/Module_13_NCA_AIIO_Infrastructure/04_mig_multi_instance_gpu.md` | mig multi instance gpu | `planned-topic` |
| `Agents_Stack_Infra/Module_13_NCA_AIIO_Infrastructure/05_nvidia_container_toolkit.md` | nvidia container toolkit | `planned-topic` |
| `Agents_Stack_Infra/Module_13_NCA_AIIO_Infrastructure/06_kubernetes_for_ai.md` | kubernetes for ai | `planned-topic` |
| `Agents_Stack_Infra/Module_13_NCA_AIIO_Infrastructure/07_dcgm_and_monitoring.md` | dcgm and monitoring | `planned-topic` |
| `Agents_Stack_Infra/Module_13_NCA_AIIO_Infrastructure/08_dgx_and_base_command.md` | dgx and base command | `planned-topic` |
| `Agents_Stack_Infra/Module_13_NCA_AIIO_Infrastructure/09_tensorrt_and_inference.md` | tensorrt and inference | `planned-topic` |
| `Agents_Stack_Infra/Module_13_NCA_AIIO_Infrastructure/exercises.md` | exercises | `supporting-artifact` |
| `Agents_Stack_Infra/Module_13_NCA_AIIO_Infrastructure/README.md` | README | `supporting-artifact` |

<a id="agent-production"></a>

## Implementation practices, production patterns, domain uses, multi-agent operations, reliability, governance, cost, scaling, pilot rollout and checklists

**Teaching home:** [05_ai_platforms_data_systems_and_production_engineering/26_mlops_feature_platforms_and_model_lifecycle/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/26_mlops_feature_platforms_and_model_lifecycle/INDEX.md). **Status:** planned.

**Testable outcome.** Convert a pilot into an owned, versioned, observable service with bounded rollback and economics.

**Focused-sheet derivations and mechanisms.** Derive useful-outcome cost and capacity envelopes; treat checklist completion as evidence-gathering not certification.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 1000 tasks cost 120 currency units in model calls and 80 in review; 800 tasks succeed: calculate included cost per success and sensitivity at 600 successes. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose pilot continuation versus staged production release under explicit quality, cost and operational guardrails.

**Proposed validation.** Shadow/canary under representative load; replay dependency failures and verify owner-controlled recovery.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Unowned failure response; hidden review labor; incompatible rollout; scaling a wrong task metric. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Agents_Stack_Infra/Module_14_Implementation_and_Production` | Source scope/container; not a completed lesson | `source-container` |
| `Agents_Stack_Infra/Module_14_Implementation_and_Production/code` | OPEN — code/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra/Module_14_Implementation_and_Production/01_implementation_best_practices.md` | implementation best practices | `planned-topic` |
| `Agents_Stack_Infra/Module_14_Implementation_and_Production/02_production_architecture_patterns.md` | production architecture patterns | `planned-topic` |
| `Agents_Stack_Infra/Module_14_Implementation_and_Production/03_domain_use_cases.md` | domain use cases | `planned-topic` |
| `Agents_Stack_Infra/Module_14_Implementation_and_Production/04_multi_agent_orchestration.md` | multi agent orchestration | `planned-topic` |
| `Agents_Stack_Infra/Module_14_Implementation_and_Production/05_reliability_and_failure_modes.md` | reliability and failure modes | `planned-topic` |
| `Agents_Stack_Infra/Module_14_Implementation_and_Production/06_security_and_governance.md` | security and governance | `planned-topic` |
| `Agents_Stack_Infra/Module_14_Implementation_and_Production/07_cost_optimization.md` | cost optimization | `planned-topic` |
| `Agents_Stack_Infra/Module_14_Implementation_and_Production/08_scaling_strategies.md` | scaling strategies | `planned-topic` |
| `Agents_Stack_Infra/Module_14_Implementation_and_Production/09_from_pilot_to_production.md` | from pilot to production | `planned-topic` |
| `Agents_Stack_Infra/Module_14_Implementation_and_Production/10_framework_checklists.md` | framework checklists | `planned-topic` |
| `Agents_Stack_Infra/Module_14_Implementation_and_Production/exercises.md` | exercises | `supporting-artifact` |
| `Agents_Stack_Infra/Module_14_Implementation_and_Production/README.md` | README | `supporting-artifact` |

<a id="claude-architecture"></a>

## Agentic architecture: loop lifecycle, orchestration, subagent context, enforcement/handoff, SDK hooks, decomposition, sessions and practice

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md). **Status:** planned.

**Testable outcome.** Specify who owns state and enforces each handoff.

**Focused-sheet derivations and mechanisms.** Derive loop termination, delegation budgets and enforceable hook boundaries.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 3 agents, each limited to 4 tools and 600 tokens/tool: calculate maximum calls/tokens; trace one rejected handoff. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose centralized workflow versus multi-agent delegation for research with restricted tool access.

**Proposed validation.** Test tool-boundary enforcement, context isolation, resume, malformed handoffs and cancellation.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Prompt-only enforcement; leaked subagent context; unbounded loop; double-owned session. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect/course/01_Domain1_Agentic_Architecture` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/01_Domain1_Agentic_Architecture/01_01_agentic_loop_lifecycle.md` | agentic loop lifecycle | `planned-topic` |
| `Claude_Architect/course/01_Domain1_Agentic_Architecture/01_02_multi_agent_orchestration.md` | multi agent orchestration | `planned-topic` |
| `Claude_Architect/course/01_Domain1_Agentic_Architecture/01_03_subagent_spawning_context.md` | subagent spawning context | `planned-topic` |
| `Claude_Architect/course/01_Domain1_Agentic_Architecture/01_04_enforcement_and_handoff.md` | enforcement and handoff | `planned-topic` |
| `Claude_Architect/course/01_Domain1_Agentic_Architecture/01_05_agent_sdk_hooks.md` | agent sdk hooks | `planned-topic` |
| `Claude_Architect/course/01_Domain1_Agentic_Architecture/01_06_task_decomposition_strategies.md` | task decomposition strategies | `planned-topic` |
| `Claude_Architect/course/01_Domain1_Agentic_Architecture/01_07_session_management.md` | session management | `planned-topic` |
| `Claude_Architect/course/01_Domain1_Agentic_Architecture/01_08_domain1_practice_questions.md` | domain1 practice questions | `planned-topic` |

<a id="claude-mcp"></a>

## Tool descriptions/boundaries, structured errors, distribution/tool choice, MCP integration, built-ins and practice

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/31_tools_mcp_protocols_and_enterprise_integrations/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/31_tools_mcp_protocols_and_enterprise_integrations/INDEX.md). **Status:** planned.

**Testable outcome.** Specify schema, permissions and error recovery per tool.

**Focused-sheet derivations and mechanisms.** Derive retryability from side-effect and error semantics; distinguish model selection from execution authority.

**Numerical/trace exercise to author.** ILLUSTRATIVE: A write times out after commit and receives two retries: trace attempts versus effects with and without a stable operation key. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose tool granularity and structured error contracts for a multi-tool service agent.

**Proposed validation.** Replay malformed arguments, missing capabilities, ambiguous timeouts and unauthorized tool choices.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Ambiguous tool description; nonretryable retry; hidden partial success; excessive built-in authority. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect/course/02_Domain2_Tool_Design_MCP` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/02_Domain2_Tool_Design_MCP/02_01_tool_descriptions_boundaries.md` | tool descriptions boundaries | `planned-topic` |
| `Claude_Architect/course/02_Domain2_Tool_Design_MCP/02_02_structured_error_responses.md` | structured error responses | `planned-topic` |
| `Claude_Architect/course/02_Domain2_Tool_Design_MCP/02_03_tool_distribution_tool_choice.md` | tool distribution tool choice | `planned-topic` |
| `Claude_Architect/course/02_Domain2_Tool_Design_MCP/02_04_mcp_server_integration.md` | mcp server integration | `planned-topic` |
| `Claude_Architect/course/02_Domain2_Tool_Design_MCP/02_05_builtin_tools.md` | builtin tools | `planned-topic` |
| `Claude_Architect/course/02_Domain2_Tool_Design_MCP/02_06_domain2_practice_questions.md` | domain2 practice questions | `planned-topic` |

<a id="claude-configuration"></a>

## CLAUDE.md hierarchy, commands/skills, path rules, plan versus direct mode, iterative refinement, CI/CD and practice

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/33_coding_agents_ai_sdcl_and_developer_productivity/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/33_coding_agents_ai_sdcl_and_developer_productivity/INDEX.md). **Status:** planned.

**Testable outcome.** Resolve configuration scope and choose a bounded coding workflow.

**Focused-sheet derivations and mechanisms.** Derive path/config precedence from versioned documentation and a concrete configuration trace.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Root rule allows 20 commands, path policy allows 4 and task budget allows 3: trace the allowed intersection, not a policy override by prompting. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose plan-first versus direct edit for a repository change with CI authority boundaries.

**Proposed validation.** Pin product/config behavior; test conflicting instructions, path matching, CI secret exposure and review gates.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Wrong precedence; overbroad path match; unreviewed CI writes; assuming source-specific commands are universal. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect/course/03_Domain3_Claude_Code_Config` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/03_Domain3_Claude_Code_Config/03_01_claude_md_hierarchy.md` | claude md hierarchy | `planned-topic` |
| `Claude_Architect/course/03_Domain3_Claude_Code_Config/03_02_slash_commands_and_skills.md` | slash commands and skills | `planned-topic` |
| `Claude_Architect/course/03_Domain3_Claude_Code_Config/03_03_path_specific_rules.md` | path specific rules | `planned-topic` |
| `Claude_Architect/course/03_Domain3_Claude_Code_Config/03_04_plan_mode_vs_direct.md` | plan mode vs direct | `planned-topic` |
| `Claude_Architect/course/03_Domain3_Claude_Code_Config/03_05_iterative_refinement.md` | iterative refinement | `planned-topic` |
| `Claude_Architect/course/03_Domain3_Claude_Code_Config/03_06_ci_cd_integration.md` | ci cd integration | `planned-topic` |
| `Claude_Architect/course/03_Domain3_Claude_Code_Config/03_07_domain3_practice_questions.md` | domain3 practice questions | `planned-topic` |

<a id="claude-prompting"></a>

## Explicit criteria, few-shot prompting, structured output/tool use, validation/retry feedback, batch work, multi-pass review and practice

**Teaching home:** [04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/19_transformers_llms_and_foundation_model_internals/INDEX.md). **Status:** planned.

**Testable outcome.** Design falsifiable prompts and separate syntactic validation from task correctness.

**Focused-sheet derivations and mechanisms.** Derive schema validation, rubric denominator and retry budget; compare few-shot and simple baseline prompts.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 80 of 100 outputs parse; 60 are correct; retries cost 0.02 currency units each for 20 attempts: calculate parse/correct rates and extra spend. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose extraction prompt and feedback policy under a fixed review/cost budget.

**Proposed validation.** Use held-out examples, adversarial valid-but-wrong outputs, blind review and retry-exhaustion checks.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Schema mistaken for truth; few-shot leakage; endless retries; reviewers repeating one error. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect/course/04_Domain4_Prompt_Engineering` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/04_Domain4_Prompt_Engineering/04_01_explicit_criteria.md` | explicit criteria | `planned-topic` |
| `Claude_Architect/course/04_Domain4_Prompt_Engineering/04_02_few_shot_prompting.md` | few shot prompting | `planned-topic` |
| `Claude_Architect/course/04_Domain4_Prompt_Engineering/04_03_structured_output_tool_use.md` | structured output tool use | `planned-topic` |
| `Claude_Architect/course/04_Domain4_Prompt_Engineering/04_04_validation_retry_feedback.md` | validation retry feedback | `planned-topic` |
| `Claude_Architect/course/04_Domain4_Prompt_Engineering/04_05_batch_processing.md` | batch processing | `planned-topic` |
| `Claude_Architect/course/04_Domain4_Prompt_Engineering/04_06_multi_pass_review.md` | multi pass review | `planned-topic` |
| `Claude_Architect/course/04_Domain4_Prompt_Engineering/04_07_domain4_practice_questions.md` | domain4 practice questions | `planned-topic` |

<a id="claude-reliability"></a>

## Context preservation, escalation, error propagation, large-codebase exploration, human-review confidence, provenance/uncertainty and practice

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/32_agent_memory_context_management_and_personalisation/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/32_agent_memory_context_management_and_personalisation/INDEX.md). **Status:** planned.

**Testable outcome.** Preserve decision evidence through bounded context and escalate unsupported conclusions.

**Focused-sheet derivations and mechanisms.** Derive source-to-claim trace and confidence/abstention rules under explicit calibration.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Context holds 4000 tokens; 1000 instructions, 800 tools, 1200 evidence: calculate budget for history/output and identify evidence lost by naive truncation. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose summarization, retrieval or escalation for a multi-session repository investigation.

**Proposed validation.** Audit citations, missing error paths, unsupported confidence, stale file revisions and handoff loss.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Evidence dropped in summary; swallowed tool error; uncalibrated confidence; speculative provenance. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect/course/05_Domain5_Context_Reliability` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/05_Domain5_Context_Reliability/05_01_context_preservation.md` | context preservation | `planned-topic` |
| `Claude_Architect/course/05_Domain5_Context_Reliability/05_02_escalation_patterns.md` | escalation patterns | `planned-topic` |
| `Claude_Architect/course/05_Domain5_Context_Reliability/05_03_error_propagation.md` | error propagation | `planned-topic` |
| `Claude_Architect/course/05_Domain5_Context_Reliability/05_04_large_codebase_exploration.md` | large codebase exploration | `planned-topic` |
| `Claude_Architect/course/05_Domain5_Context_Reliability/05_05_human_review_confidence.md` | human review confidence | `planned-topic` |
| `Claude_Architect/course/05_Domain5_Context_Reliability/05_06_provenance_uncertainty.md` | provenance uncertainty | `planned-topic` |
| `Claude_Architect/course/05_Domain5_Context_Reliability/05_07_domain5_practice_questions.md` | domain5 practice questions | `planned-topic` |

<a id="scenario-07"></a>

## clinical decision support

**Teaching home:** [08_business_domains_and_decision_intelligence/42_healthcare_life_sciences_and_biomedical_ai/INDEX.md](08_business_domains_and_decision_intelligence/42_healthcare_life_sciences_and_biomedical_ai/INDEX.md). **Status:** planned.

**Testable outcome.** Frame one independently bounded domain decision with its own data, owner, risk and evidence limit.

**Focused-sheet derivations and mechanisms.** Derive the domain metric, decision-time data boundary and authorization/review invariant; split independent scenario decisions into cases.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Compute sensitivity from 90 true positives and 10 false negatives; do not infer clinical benefit. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** What evidence-backed assistance can be shown versus escalated to a clinician?

**Proposed validation.** Compare the manual/rules baseline with proposed assistance on an appropriate entity/time split; validate expert review, authorization, failure paths and domain utility.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Proxy improvement without utility; future information; missing expert handoff; unauthorized action. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/07_Healthcare` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/07_Healthcare/07_01_clinical_decision_support.md` | clinical decision support | `planned-topic` |

<a id="scenario-08"></a>

## legal contract analysis

**Teaching home:** [04_deep_learning_generative_ai_and_multimodal_systems/20_rag_context_engineering_and_enterprise_knowledge/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/20_rag_context_engineering_and_enterprise_knowledge/INDEX.md). **Status:** planned.

**Testable outcome.** Frame one independently bounded domain decision with its own data, owner, risk and evidence limit.

**Focused-sheet derivations and mechanisms.** Derive the domain metric, decision-time data boundary and authorization/review invariant; split independent scenario decisions into cases.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 20 clauses, 5 required risk clauses, 4 retrieved: compute clause-level recall and identify incomplete evidence. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Which clause claims are supported by the applicable contract/version and require legal review?

**Proposed validation.** Compare the manual/rules baseline with proposed assistance on an appropriate entity/time split; validate expert review, authorization, failure paths and domain utility.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Proxy improvement without utility; future information; missing expert handoff; unauthorized action. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/08_Legal` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/08_Legal/08_01_contract_analysis_system.md` | contract analysis system | `planned-topic` |

<a id="scenario-09"></a>

## financial fraud agents

**Teaching home:** [08_business_domains_and_decision_intelligence/43_financial_services_fraud_risk_and_fintech/INDEX.md](08_business_domains_and_decision_intelligence/43_financial_services_fraud_risk_and_fintech/INDEX.md). **Status:** planned.

**Testable outcome.** Frame one independently bounded domain decision with its own data, owner, risk and evidence limit.

**Focused-sheet derivations and mechanisms.** Derive the domain metric, decision-time data boundary and authorization/review invariant; split independent scenario decisions into cases.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 40 correct alerts among 100 reviewed; 25 additional alerts at 20% precision: calculate extra review yield. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Which alerts should reach investigators under a fixed review budget?

**Proposed validation.** Compare the manual/rules baseline with proposed assistance on an appropriate entity/time split; validate expert review, authorization, failure paths and domain utility.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Proxy improvement without utility; future information; missing expert handoff; unauthorized action. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/09_Financial_Services` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/09_Financial_Services/09_01_fraud_detection_agent.md` | fraud detection agent | `planned-topic` |

<a id="scenario-10"></a>

## predictive maintenance agents

**Teaching home:** [08_business_domains_and_decision_intelligence/45_manufacturing_industrial_iot_and_quality_ai/INDEX.md](08_business_domains_and_decision_intelligence/45_manufacturing_industrial_iot_and_quality_ai/INDEX.md). **Status:** planned.

**Testable outcome.** Frame one independently bounded domain decision with its own data, owner, risk and evidence limit.

**Focused-sheet derivations and mechanisms.** Derive the domain metric, decision-time data boundary and authorization/review invariant; split independent scenario decisions into cases.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Two false stops cost 500 each versus 100 currency units per inspection for 8 inspections: compare stated cost only. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Which detected faults justify inspection or planned maintenance?

**Proposed validation.** Compare the manual/rules baseline with proposed assistance on an appropriate entity/time split; validate expert review, authorization, failure paths and domain utility.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Proxy improvement without utility; future information; missing expert handoff; unauthorized action. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/10_Manufacturing` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/10_Manufacturing/10_01_predictive_maintenance_agent.md` | predictive maintenance agent | `planned-topic` |

<a id="scenario-11"></a>

## SOC automation

**Teaching home:** [08_business_domains_and_decision_intelligence/47_cybersecurity_security_operations_and_ai_defense/INDEX.md](08_business_domains_and_decision_intelligence/47_cybersecurity_security_operations_and_ai_defense/INDEX.md). **Status:** planned.

**Testable outcome.** Frame one independently bounded domain decision with its own data, owner, risk and evidence limit.

**Focused-sheet derivations and mechanisms.** Derive the domain metric, decision-time data boundary and authorization/review invariant; split independent scenario decisions into cases.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 120 alerts/hour and 6 minutes/review: calculate reviewer-hour demand before introducing automation. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Which response actions can be automated without exceeding incident authority?

**Proposed validation.** Compare the manual/rules baseline with proposed assistance on an appropriate entity/time split; validate expert review, authorization, failure paths and domain utility.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Proxy improvement without utility; future information; missing expert handoff; unauthorized action. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/11_Cybersecurity` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/11_Cybersecurity/11_01_soc_automation_agent.md` | soc automation agent | `planned-topic` |

<a id="scenario-12"></a>

## claims automation

**Teaching home:** [08_business_domains_and_decision_intelligence/43_financial_services_fraud_risk_and_fintech/INDEX.md](08_business_domains_and_decision_intelligence/43_financial_services_fraud_risk_and_fintech/INDEX.md). **Status:** planned.

**Testable outcome.** Frame one independently bounded domain decision with its own data, owner, risk and evidence limit.

**Focused-sheet derivations and mechanisms.** Derive the domain metric, decision-time data boundary and authorization/review invariant; split independent scenario decisions into cases.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 300 claims/day, 10% review rate and 8 minutes/review: calculate daily review hours. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Which extracted claims may be routed automatically versus adjuster review?

**Proposed validation.** Compare the manual/rules baseline with proposed assistance on an appropriate entity/time split; validate expert review, authorization, failure paths and domain utility.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Proxy improvement without utility; future information; missing expert handoff; unauthorized action. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/12_Insurance` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/12_Insurance/12_01_claims_automation_agent.md` | claims automation agent | `planned-topic` |

<a id="scenario-13"></a>

## retail/e-commerce agents

**Teaching home:** [08_business_domains_and_decision_intelligence/44_retail_ecommerce_media_and_personalisation/INDEX.md](08_business_domains_and_decision_intelligence/44_retail_ecommerce_media_and_personalisation/INDEX.md). **Status:** planned.

**Testable outcome.** Frame one independently bounded domain decision with its own data, owner, risk and evidence limit.

**Focused-sheet derivations and mechanisms.** Derive the domain metric, decision-time data boundary and authorization/review invariant; split independent scenario decisions into cases.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 10 recommended SKUs, 3 unavailable: calculate eligible candidate count before ranking. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Which catalog or order actions can execute under inventory and customer-policy constraints?

**Proposed validation.** Compare the manual/rules baseline with proposed assistance on an appropriate entity/time split; validate expert review, authorization, failure paths and domain utility.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Proxy improvement without utility; future information; missing expert handoff; unauthorized action. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/13_Retail_Ecommerce` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/13_Retail_Ecommerce/13_01_ecommerce_ai_agents.md` | ecommerce ai agents | `planned-topic` |

<a id="scenario-14"></a>

## research acceleration

**Teaching home:** [08_business_domains_and_decision_intelligence/48_energy_climate_public_sector_and_scientific_ai/INDEX.md](08_business_domains_and_decision_intelligence/48_energy_climate_public_sector_and_scientific_ai/INDEX.md). **Status:** planned.

**Testable outcome.** Frame one independently bounded domain decision with its own data, owner, risk and evidence limit.

**Focused-sheet derivations and mechanisms.** Derive the domain metric, decision-time data boundary and authorization/review invariant; split independent scenario decisions into cases.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 12 claims each require 2 independent sources, 18 admissible claim-source links available: calculate support shortfall without equating sources with truth. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Which research conclusions have retrievable primary evidence rather than unsupported synthesis?

**Proposed validation.** Compare the manual/rules baseline with proposed assistance on an appropriate entity/time split; validate expert review, authorization, failure paths and domain utility.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Proxy improvement without utility; future information; missing expert handoff; unauthorized action. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/14_Scientific_Research` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/14_Scientific_Research/14_01_research_acceleration_agent.md` | research acceleration agent | `planned-topic` |

<a id="scenario-15"></a>

## production anti-patterns and MCP integration

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/31_tools_mcp_protocols_and_enterprise_integrations/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/31_tools_mcp_protocols_and_enterprise_integrations/INDEX.md). **Status:** planned.

**Testable outcome.** Frame one independently bounded domain decision with its own data, owner, risk and evidence limit.

**Focused-sheet derivations and mechanisms.** Derive the domain metric, decision-time data boundary and authorization/review invariant; split independent scenario decisions into cases.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 4 tools each permit 3 attempts: bound attempts and trace an effect committed on a timed-out attempt. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Which cross-domain tool and deployment invariants prevent repeated effects or permission leaks?

**Proposed validation.** Compare the manual/rules baseline with proposed assistance on an appropriate entity/time split; validate expert review, authorization, failure paths and domain utility.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Proxy improvement without utility; future information; missing expert handoff; unauthorized action. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/15_Cross_Industry_Patterns` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/15_Cross_Industry_Patterns/15_01_production_anti_patterns.md` | production anti patterns | `planned-topic` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains/15_Cross_Industry_Patterns/15_02_mcp_integration_production.md` | mcp integration production | `planned-topic` |

<a id="claude-scenarios"></a>

## Support, code generation, multi-agent research, developer productivity, CI/CD and structured extraction scenarios

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/33_coding_agents_ai_sdcl_and_developer_productivity/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/33_coding_agents_ai_sdcl_and_developer_productivity/INDEX.md). **Status:** planned.

**Testable outcome.** Select a workflow and evaluation for each of the six distinct scenarios.

**Focused-sheet derivations and mechanisms.** For support derive escalation precision; for extraction separate field correctness and valid syntax; for coding distinguish passing checks from functional correctness.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 100 support tasks: 70 correct automatic resolutions, 20 justified escalations, 10 failures; calculate separate resolution and safe-handling rates. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose automation boundaries separately for customer support, coding, research, development workflow, CI/CD and extraction.

**Proposed validation.** Use task-specific correctness oracles, rollback and escalation; isolate repository/tenant data; do not treat one metric as sufficient for all six scenarios.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Unverified code; cross-agent evidence loss; CI authority creep; schema-only extraction score. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect/course/06_Scenarios_Deep_Dive` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/domains` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/06_01_customer_support_agent.md` | customer support agent | `planned-topic` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/06_02_code_generation_claude_code.md` | code generation claude code | `planned-topic` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/06_03_multi_agent_research.md` | multi agent research | `planned-topic` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/06_04_developer_productivity.md` | developer productivity | `planned-topic` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/06_05_ci_cd_pipeline.md` | ci cd pipeline | `planned-topic` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/06_06_structured_data_extraction.md` | structured data extraction | `planned-topic` |
| `Claude_Architect/course/06_Scenarios_Deep_Dive/DOMAIN_INDEX.md` | DOMAIN INDEX | `planned-topic` |

<a id="claude-practice"></a>

## Hands-on multi-tool agent, code configuration, extraction and multi-agent pipeline exercises

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/33_coding_agents_ai_sdcl_and_developer_productivity/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/33_coding_agents_ai_sdcl_and_developer_productivity/INDEX.md). **Status:** planned.

**Testable outcome.** Implement and demonstrate each exercise with observable outputs and explicit failure checks.

**Focused-sheet derivations and mechanisms.** Derive each exercise input/output contract and pass/fail oracle before implementation.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 5 input documents with 4 required fields each; 17 fields correct and one unauthorized field exposed: compute accuracy but reject on authorization guardrail. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose an exercise release that satisfies correctness and authority, not only tool-call completion.

**Proposed validation.** Run actual bounded tasks with missing fields, bad tool results, wrong configurations and repeated pipeline messages.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Mock echo mistaken for proof; overbroad config; extraction hallucination; repeated writes. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect/course/07_Hands_On_Exercises` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/07_Hands_On_Exercises/07_01_exercise_multi_tool_agent.md` | exercise multi tool agent | `planned-topic` |
| `Claude_Architect/course/07_Hands_On_Exercises/07_02_exercise_claude_code_config.md` | exercise claude code config | `planned-topic` |
| `Claude_Architect/course/07_Hands_On_Exercises/07_03_exercise_data_extraction.md` | exercise data extraction | `planned-topic` |
| `Claude_Architect/course/07_Hands_On_Exercises/07_04_exercise_multi_agent_pipeline.md` | exercise multi agent pipeline | `planned-topic` |

<a id="claude-reference"></a>

## API, Agent SDK, MCP, Claude Code, anti-patterns and decision-flow references

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/31_tools_mcp_protocols_and_enterprise_integrations/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/31_tools_mcp_protocols_and_enterprise_integrations/INDEX.md). **Status:** planned.

**Testable outcome.** Look up versioned contracts and reject an invalid integration choice.

**Focused-sheet derivations and mechanisms.** Specify API/tool state and error contracts, SDK ownership, and decision-tree disqualifiers instead of unqualified defaults.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Context 16000 tokens, input 12000, tool reserve 2500: calculate remaining output allowance; trace rejection if proposed output is 2000. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose an integration contract from API/SDK/MCP/native coding tools given deployment and authority constraints.

**Proposed validation.** Check pinned official references and representative request/error/resume traces for each distinct integration.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Outdated API; mixed SDK/protocol semantics; invalid context sum; unsafe default. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/08_Quick_Reference` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/08_Quick_Reference/08_01_api_reference_card.md` | api reference card | `planned-topic` |
| `Claude_Architect/course/08_Quick_Reference/08_02_agent_sdk_reference.md` | agent sdk reference | `planned-topic` |
| `Claude_Architect/course/08_Quick_Reference/08_03_mcp_reference.md` | mcp reference | `planned-topic` |
| `Claude_Architect/course/08_Quick_Reference/08_04_claude_code_reference.md` | claude code reference | `planned-topic` |
| `Claude_Architect/course/08_Quick_Reference/08_05_anti_patterns_cheatsheet.md` | anti patterns cheatsheet | `planned-topic` |
| `Claude_Architect/course/08_Quick_Reference/08_06_decision_flowcharts.md` | decision flowcharts | `planned-topic` |
| `Claude_Architect/course/part1` | OPEN — part1/  [contents not visible] | `unresolved-source` |
| `Claude_Architect/course/part2` | OPEN — part2/  [contents not visible] | `unresolved-source` |
| `Claude_Architect/course/[TRUNCATED/UNVERIFIED] instructor_8lsy243ftffjjy1cx9lm3o2bw_public_1773274827_Claud… (.pdf)` | OPEN — [TRUNCATED/UNVERIFIED] instructor_8lsy243ftffjjy1cx9lm3o2bw_public_1773274827_Claud… (.pdf) | `unresolved-source` |

<a id="claude-exam"></a>

## Course overview, exam framing, practice questions and 60-question practice exam

**Teaching home:** [10_architecture_practice_and_portfolio_capstones/54_ml_and_ai_system_design_interviews/INDEX.md](10_architecture_practice_and_portfolio_capstones/54_ml_and_ai_system_design_interviews/INDEX.md). **Status:** planned.

**Testable outcome.** Defend answers and diagnose wrong approaches across the five study domains without asserting an official certification blueprint.

**Focused-sheet derivations and mechanisms.** Derive scores and skill-domain coverage; retain checked answers and source-supported rationales, not trivia alone.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 60 questions: 45 correct, 10 wrong, 5 unanswered; compute overall accuracy and answered-only accuracy and explain denominator choice. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose which knowledge gaps to remediate from a practice exam instead of treating a pass score as production competence.

**Proposed validation.** Verify question answerability, distinctness, numerical solutions, changed-constraint follow-ups and current blueprint only from authoritative sources.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Unsupported exam claims; denominator gaming; memorized trivia; unverified answer key. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Claude_Architect/course/00_Course_Overview` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/00_Course_Overview/00_exam_overview.md` | exam overview | `planned-topic` |
| `Claude_Architect/course/09_Full_Practice_Exam` | Source scope/container; not a completed lesson | `source-container` |
| `Claude_Architect/course/09_Full_Practice_Exam/09_01_practice_exam_60q.md` | practice exam 60q | `planned-topic` |
| `Claude_Architect/course/claude_mcq.txt` | claude mcq | `supporting-artifact` |
| `Claude_Architect/course/guide_en.MD` | guide en | `planned-topic` |
| `Claude_Architect/course/guide_v0.pdf` | guide v0 | `supporting-artifact` |

<a id="lang-foundations"></a>

## LangChain foundations, ecosystem, LCEL, prompts/models/parsers

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/31_tools_mcp_protocols_and_enterprise_integrations/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/31_tools_mcp_protocols_and_enterprise_integrations/INDEX.md). **Status:** planned.

**Testable outcome.** Select, trace and diagnose the named framework capabilities using versioned behavior.

**Focused-sheet derivations and mechanisms.** Derive runnable composition, typed parsing and error propagation; distinguish LCEL composition from autonomous control.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 3 sequential calls take illustrative 100, 200 and 50 ms; calculate sequential time and contrast independent concurrent branches. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose LCEL chain versus a direct API function under explicit input/output contracts.

**Proposed validation.** Pin LangChain/LangGraph/LangSmith revisions; run bounded traces, failure injection and end-to-end task evaluation; historical patterns are not current universal recommendations.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Version mismatch; incorrect state/retry assumptions; unbounded work; metric without task validity. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `LangGraph_LangChain_2025/LangGraph_LangChain_Cheat_Sheets/01_Foundations_Ecosystem.md` | Foundations Ecosystem | `planned-topic` |
| `LangGraph_LangChain_2025/LangGraph_LangChain_Cheat_Sheets/02_LCEL_Core_Concepts.md` | LCEL Core Concepts | `planned-topic` |
| `LangGraph_LangChain_2025/LangGraph_LangChain_Cheat_Sheets/03_Prompts_Models_Parsers.md` | Prompts Models Parsers | `planned-topic` |
| `LangGraph_LangChain_2025/01_Foundations_and_Ecosystem.md` | Foundations and Ecosystem | `planned-topic` |
| `LangGraph_LangChain_2025/02_LangChain_Core_Concepts.md` | LangChain Core Concepts | `planned-topic` |

<a id="lang-graphs"></a>

## LangGraph state, nodes/edges, agents, tools and multi-agent systems

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md). **Status:** planned.

**Testable outcome.** Select, trace and diagnose the named framework capabilities using versioned behavior.

**Focused-sheet derivations and mechanisms.** Derive state reducers, checkpoint ownership, branching, interrupts and bounded tool loops from pinned runtime semantics.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Two branches append 2 and 3 facts to a state with 1 fact: trace merge versus overwrite and determine correct count. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose graph state/reducer and approved resume design for a delegated workflow.

**Proposed validation.** Pin LangChain/LangGraph/LangSmith revisions; run bounded traces, failure injection and end-to-end task evaluation; historical patterns are not current universal recommendations.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Version mismatch; incorrect state/retry assumptions; unbounded work; metric without task validity. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `LangGraph_LangChain_2025/LangGraph_LangChain_Cheat_Sheets/04_LangGraph_Architecture.md` | LangGraph Architecture | `planned-topic` |
| `LangGraph_LangChain_2025/LangGraph_LangChain_Cheat_Sheets/05_Nodes_Edges_Control_Flow.md` | Nodes Edges Control Flow | `planned-topic` |
| `LangGraph_LangChain_2025/LangGraph_LangChain_Cheat_Sheets/06_Agents_Tool_Use.md` | Agents Tool Use | `planned-topic` |
| `LangGraph_LangChain_2025/LangGraph_LangChain_Cheat_Sheets/07_Multi_Agent_Systems.md` | Multi Agent Systems | `planned-topic` |
| `LangGraph_LangChain_2025/03_LangGraph_Architecture.md` | LangGraph Architecture | `planned-topic` |
| `LangGraph_LangChain_2025/04_Agents_and_Tool_Use.md` | Agents and Tool Use | `planned-topic` |
| `LangGraph_LangChain_2025/05_Multi_Agent_Systems.md` | Multi Agent Systems | `planned-topic` |

<a id="lang-memory"></a>

## LangGraph memory/persistence and session isolation

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/32_agent_memory_context_management_and_personalisation/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/32_agent_memory_context_management_and_personalisation/INDEX.md). **Status:** planned.

**Testable outcome.** Select, trace and diagnose the named framework capabilities using versioned behavior.

**Focused-sheet derivations and mechanisms.** Derive session/store/checkpoint ownership, retention and authenticated resume.

**Numerical/trace exercise to author.** ILLUSTRATIVE: A 5000-token budget reserves 1000 instructions and 1500 retrieval: calculate space remaining for history plus output. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose short-term checkpoint state versus long-term memory for a tenant assistant.

**Proposed validation.** Pin LangChain/LangGraph/LangSmith revisions; run bounded traces, failure injection and end-to-end task evaluation; historical patterns are not current universal recommendations.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Version mismatch; incorrect state/retry assumptions; unbounded work; metric without task validity. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `LangGraph_LangChain_2025/LangGraph_LangChain_Cheat_Sheets/08_Memory_Persistence.md` | Memory Persistence | `planned-topic` |
| `LangGraph_LangChain_2025/06_Memory_and_Persistence.md` | Memory and Persistence | `planned-topic` |

<a id="lang-rag"></a>

## LangChain RAG retrieval pipelines

**Teaching home:** [04_deep_learning_generative_ai_and_multimodal_systems/20_rag_context_engineering_and_enterprise_knowledge/INDEX.md](04_deep_learning_generative_ai_and_multimodal_systems/20_rag_context_engineering_and_enterprise_knowledge/INDEX.md). **Status:** planned.

**Testable outcome.** Select, trace and diagnose the named framework capabilities using versioned behavior.

**Focused-sheet derivations and mechanisms.** Derive chunk eligibility, retrieval metrics, reranking, citation and index update semantics.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 4 relevant chunks, 3 retrieved among 5 results: compute precision and recall with stated relevance labels. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose filtered lexical/dense/hybrid retrieval before generation.

**Proposed validation.** Pin LangChain/LangGraph/LangSmith revisions; run bounded traces, failure injection and end-to-end task evaluation; historical patterns are not current universal recommendations.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Version mismatch; incorrect state/retry assumptions; unbounded work; metric without task validity. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `LangGraph_LangChain_2025/LangGraph_LangChain_Cheat_Sheets/09_RAG_Pipelines.md` | RAG Pipelines | `planned-topic` |
| `LangGraph_LangChain_2025/08_RAG_Pipelines.md` | RAG Pipelines | `planned-topic` |

<a id="lang-streaming"></a>

## Streaming, real-time output and interruption

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/34_conversational_ai_voice_agents_and_customer_operations/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/34_conversational_ai_voice_agents_and_customer_operations/INDEX.md). **Status:** planned.

**Testable outcome.** Select, trace and diagnose the named framework capabilities using versioned behavior.

**Focused-sheet derivations and mechanisms.** Derive chunk/event ordering, cancellation, client reconnect and tool-progress boundaries.

**Numerical/trace exercise to author.** ILLUSTRATIVE: First token at 400 ms and 100 subsequent tokens over 2 s: calculate first-token latency and generation rate separately. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose streamed events versus final-response delivery for a cancelable assistant.

**Proposed validation.** Pin LangChain/LangGraph/LangSmith revisions; run bounded traces, failure injection and end-to-end task evaluation; historical patterns are not current universal recommendations.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Version mismatch; incorrect state/retry assumptions; unbounded work; metric without task validity. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `LangGraph_LangChain_2025/LangGraph_LangChain_Cheat_Sheets/10_Streaming_Real_Time.md` | Streaming Real Time | `planned-topic` |
| `LangGraph_LangChain_2025/07_Streaming_and_Real_Time.md` | Streaming and Real Time | `planned-topic` |

<a id="lang-ops"></a>

## LangSmith observability, production deployment and dated emerging patterns

**Teaching home:** [05_ai_platforms_data_systems_and_production_engineering/28_ai_evaluation_monitoring_and_continuous_improvement/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/28_ai_evaluation_monitoring_and_continuous_improvement/INDEX.md). **Status:** planned.

**Testable outcome.** Select, trace and diagnose the named framework capabilities using versioned behavior.

**Focused-sheet derivations and mechanisms.** Derive trace/task metrics, evaluation splits, deployment release units and bounded failure handling.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 150 runs, 120 rubric successes, 30 currency units total: calculate success rate and cost/success including failures. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose evaluation and rollout gates before promoting a new graph/prompt release.

**Proposed validation.** Pin LangChain/LangGraph/LangSmith revisions; run bounded traces, failure injection and end-to-end task evaluation; historical patterns are not current universal recommendations.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Version mismatch; incorrect state/retry assumptions; unbounded work; metric without task validity. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `LangGraph_LangChain_2025` | Source scope/container; not a completed lesson | `source-container` |
| `LangGraph_LangChain_2025/LangGraph_LangChain_Cheat_Sheets` | Source scope/container; not a completed lesson | `source-container` |
| `LangGraph_LangChain_2025/LangGraph_LangChain_Cheat_Sheets/00_INDEX.md` | INDEX | `supporting-artifact` |
| `LangGraph_LangChain_2025/LangGraph_LangChain_Cheat_Sheets/11_LangSmith_Observability.md` | LangSmith Observability | `planned-topic` |
| `LangGraph_LangChain_2025/LangGraph_LangChain_Cheat_Sheets/12_Production_Deployment.md` | Production Deployment | `planned-topic` |
| `LangGraph_LangChain_2025/LangGraph_LangChain_Cheat_Sheets/13_2025_Patterns_Emerging.md` | Patterns Emerging | `planned-topic` |
| `LangGraph_LangChain_2025/09_LangSmith_Observability.md` | LangSmith Observability | `planned-topic` |
| `LangGraph_LangChain_2025/10_Production_Deployment.md` | Production Deployment | `planned-topic` |
| `LangGraph_LangChain_2025/11_2025_2026_Patterns.md` | 2026 Patterns | `planned-topic` |
| `LangGraph_LangChain_2025/all_content.txt` | all content | `supporting-artifact` |
| `LangGraph_LangChain_2025/INDEX.md` | INDEX | `supporting-artifact` |
| `LangGraph_LangChain_2025/learn` | OPEN — learn/  [contents not visible] | `unresolved-source` |

<a id="lang-issues"></a>

## 100 Common LangGraph/LangChain Issues

**Teaching home:** [10_architecture_practice_and_portfolio_capstones/55_incident_response_failure_modes_and_technical_leadership/INDEX.md](10_architecture_practice_and_portfolio_capstones/55_incident_response_failure_modes_and_technical_leadership/INDEX.md). **Status:** planned.

**Testable outcome.** Select, trace and diagnose the named framework capabilities using versioned behavior.

**Focused-sheet derivations and mechanisms.** Distinguish recurring assumption, numeric, scale and concept failures; retain a target of 100 genuinely distinct complete issues rather than copied rows.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Backlog grows 50 events/s for 120 s: compute 6000 pending events and identify whether throughput or buffer sizing is the cause. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose a discriminating diagnosis for checkpoint, tool, retrieval or deployment symptoms.

**Proposed validation.** Pin LangChain/LangGraph/LangSmith revisions; run bounded traces, failure injection and end-to-end task evaluation; historical patterns are not current universal recommendations.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Version mismatch; incorrect state/retry assumptions; unbounded work; metric without task validity. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `LangGraph_LangChain_2025/LangGraph_LangChain_Cheat_Sheets/14_100_Common_Issues.md` | Common Issues | `planned-topic` |
| `LangGraph_LangChain_2025/12_100_Common_Issues.md` | Common Issues | `planned-topic` |

<a id="inference-clusters"></a>

## Six inference clusters, tool contracts, mock responses, manifest, archive and explainer

**Teaching home:** [05_ai_platforms_data_systems_and_production_engineering/28_ai_evaluation_monitoring_and_continuous_improvement/INDEX.md](05_ai_platforms_data_systems_and_production_engineering/28_ai_evaluation_monitoring_and_continuous_improvement/INDEX.md). **Status:** planned.

**Testable outcome.** Specify independent acceptance scenarios for tool mechanics, path completion, reasoning, data integrity, safety/escalation and multi-agent behavior.

**Focused-sheet derivations and mechanisms.** Derive schema/path scope checks, immutable data invariants, safe-refusal/approval transitions and delegation bounds; mock responses test controlled conditions, not real integration.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Each of 6 clusters has 10 illustrative scenarios; 8 failures concentrated in safety cannot be hidden by 52/60 aggregate success: calculate cluster and overall rates. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose release-blocking cluster gates for an insurance assistant instead of average-only scoring.

**Proposed validation.** Exercise actual paths for each cluster with unauthorized paths, corrupted data, ambiguous tool results, unsafe actions and cross-agent state; separate mocked and real execution evidence.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Tool echo accepted as integration; path traversal; aggregate masks safety; lost or duplicated multi-agent effects. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `inference` | Source scope/container; not a completed lesson | `source-container` |
| `inference/cluster_based_test_cases` | Source scope/container; not a completed lesson | `source-container` |
| `inference/cluster_based_test_cases/cluster_based_test_cases` | Source scope/container; not a completed lesson | `source-container` |
| `inference/cluster_based_test_cases/cluster_based_test_cases/generic-insurance-test-cases` | OPEN — generic-insurance-test-cases/  [contents not visible] | `unresolved-source` |
| `inference/cluster_based_test_cases/cluster_based_test_cases/cluster_1_tool_mechanics.json` | cluster 1 tool mechanics | `planned-topic` |
| `inference/cluster_based_test_cases/cluster_based_test_cases/cluster_2_path_completion.json` | cluster 2 path completion | `planned-topic` |
| `inference/cluster_based_test_cases/cluster_based_test_cases/cluster_3_reasoning.json` | cluster 3 reasoning | `planned-topic` |
| `inference/cluster_based_test_cases/cluster_based_test_cases/cluster_4_data_integrity.json` | cluster 4 data integrity | `planned-topic` |
| `inference/cluster_based_test_cases/cluster_based_test_cases/cluster_5_safety_escalation.json` | cluster 5 safety escalation | `planned-topic` |
| `inference/cluster_based_test_cases/cluster_based_test_cases/cluster_6_multiagent.json` | cluster 6 multiagent | `planned-topic` |
| `inference/cluster_based_test_cases/cluster_based_test_cases/manifest.mf` | manifest | `supporting-artifact` |
| `inference/cluster_based_test_cases/cluster_based_test_cases/mock_responses.json` | mock responses | `planned-topic` |
| `inference/cluster_based_test_cases/cluster_based_test_cases/tools.json` | tools | `planned-topic` |
| `inference/cluster_based_test_cases/cluster_based_test_cases.zip` | cluster based test cases | `supporting-artifact` |
| `inference/cluster_based_test_cases/EXPLAINER.md` | EXPLAINER | `planned-topic` |

<a id="agent-source-artifacts"></a>

## Agent source references, hidden folders, PDFs and incomplete exam/harness evidence

**Teaching home:** [06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md](06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/INDEX.md). **Status:** planned.

**Testable outcome.** Separate visible lesson titles from inaccessible implementation and reference evidence.

**Focused-sheet derivations and mechanisms.** Define provenance identifiers and source-access/review state; derive no proprietary behavior from document filenames.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Classify 10 proposed claims into supported, illustrative and open; unsupported claims remain unreported as facts. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose whether sufficient source evidence exists to publish a named-system case or use a hypothetical design.

**Proposed validation.** Obtain original references, code revisions and accessible primary documentation; preserve source-access gaps.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Title mistaken for proof; invented implementation; inaccessible source treated as reviewed; stale historical claim. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `.claude` | OPEN — .claude/  [contents not visible] | `unresolved-source` |
| `agents` | OPEN — agents/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra` | Source scope/container; not a completed lesson | `source-container` |
| `Agents_Stack_Infra/Agent_Stack_Cheat_Sheets` | OPEN — Agent_Stack_Cheat_Sheets/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra/RESEARCH` | OPEN — RESEARCH/  [contents not visible] | `unresolved-source` |
| `Agents_Stack_Infra/agentic ai.pdf` | agentic ai | `supporting-artifact` |
| `Agents_Stack_Infra/agents whitepaper.pdf` | agents whitepaper | `supporting-artifact` |
| `Agents_Stack_Infra/ai agents.pdf` | ai agents | `supporting-artifact` |
| `Agents_Stack_Infra/ai_engg.pdf` | ai engg | `supporting-artifact` |
| `Agents_Stack_Infra/all_content.txt` | all content | `supporting-artifact` |
| `Agents_Stack_Infra/Contxt_engg_paul.pdf` | Contxt engg paul | `supporting-artifact` |
| `Agents_Stack_Infra/Embedding retrieval limitations.pdf` | Embedding retrieval limitations | `supporting-artifact` |
| `Agents_Stack_Infra/gen ai use cases.pdf` | gen ai use cases | `supporting-artifact` |
| `Agents_Stack_Infra/LLM sys design.pdf` | LLM sys design | `supporting-artifact` |
| `Agents_Stack_Infra/LLM system design.pdf` | LLM system design | `supporting-artifact` |
| `Agents_Stack_Infra/prompt report.pdf` | prompt report | `supporting-artifact` |
| `Agents_Stack_Infra/README.md` | README | `supporting-artifact` |

## Publication boundary

This is complete **visible-source topic planning**, not completed teaching content or hidden-source coverage. Preserve all existing course topics. Before publishing individual artifacts, obtain sources, author the required structures and topic-specific mathematics, execute numerical/trace checks and relevant validation, answer interview questions, finish actual issue counts, verify links/prerequisites, and update local catalog status and review records.
