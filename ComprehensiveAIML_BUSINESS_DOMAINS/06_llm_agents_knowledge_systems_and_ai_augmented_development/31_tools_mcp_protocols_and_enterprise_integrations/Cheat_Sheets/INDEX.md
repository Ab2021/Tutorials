# Module 31 Cheat Sheets — Tools, MCP, Protocols and Enterprise Integrations

## 1. Overview
For integration engineers familiar with APIs and authorization, this collection supports protocol lifecycle, tool boundaries and safe recovery decisions. The MCP and tool-schema sheets are unreviewed drafts; publication requires source, technical and security review.

## 2. Topic ladder
| Level | Topic and outcome | Prerequisite / skill | Status |
|---|---|---|---|
| Foundation | [Tool schema and side-effect boundaries](02_Tool_Schema_and_Side_Effect_Boundaries.md): specify inputs, semantic validation, side effects and recovery. | JSON and API contracts. | Draft; unreviewed |
| Core | MCP server checklist: trace initialization and discovery and diagnose a missing tool. | API lifecycle and authentication. | Planned |
| Core | Side-effect risk matrix: classify operations and choose a safe retry boundary. | Transactions and authorization. | Planned |
| Production | [MCP-01 Model Context Protocol](01_model_context_protocol.md): trace protocol and execution boundaries, sizing and recovery. | JSON-RPC, identity and distributed failure. | Draft; unreviewed; publication checks pending |

## 3. Quick topic lookup
| Question | Exact section | Use and warning |
|---|---|---|
| What is the protocol versus executor boundary? | [Definitions](01_model_context_protocol.md#section-3--foundational-definitions) | Protocol success alone does not guarantee business success. |
| What if a write times out? | [Failure modes](01_model_context_protocol.md#section-8--failure-modes--edge-cases) | Inspect authoritative status before retrying; outcome can be ambiguous. |
| What do retries cost? | [Formulas](01_model_context_protocol.md#section-4--core-formulas-with-full-derivation) | Independence is a model assumption, not a service guarantee. |

## 4. Reading paths
- **Beginner:** Learn JSON, HTTP and authentication; read [definitions](01_model_context_protocol.md#section-3--foundational-definitions) and the [mental model](01_model_context_protocol.md#section-2--concept-map--mental-model); the schema guide is planned.
- **Practitioner:** Read MCP-01 through [selection](01_model_context_protocol.md#section-7--comparison--selection-table) and failure cases, checking permissions and write semantics.
- **Advanced:** Read the entire draft, including sizing and its card; use the server checklist and risk matrix only after they are authored.

## 5. Prerequisite skills
Read JSON-RPC messages, distinguish authentication from authorization, understand retries and transactions, and compute simple probabilities. The draft [header](01_model_context_protocol.md#section-1--file-header) states its scope and prerequisites.

## 6. What this collection does NOT cover
The server checklist and full side-effect risk matrix are planned. Protocol compliance does not establish endpoint authorization or exactly-once business effects. The tool-schema draft is not a security certification or production integration recipe.

## 7. Master symbol glossary
| Symbol / term | Meaning; unit / type / domain | First use |
|---|---|---|
| MCP | Model Context Protocol; protocol for host/server interaction, not an executor guarantee. | [Definitions](01_model_context_protocol.md#section-3--foundational-definitions) |
| p | Attempt failure probability; dimensionless scalar in [0,1], under the independent retry model. | [Formulas](01_model_context_protocol.md#section-4--core-formulas-with-full-derivation) |

## 8. Quick Reference Card directory
| Card | Rule / use | Condition and avoid |
|---|---|---|
| [MCP-01 card](01_model_context_protocol.md#section-12--quick-reference-card) | Separate protocol, permissions and executor effects. | Check versioned behavior and provider contract; never infer committed effects from timeout. |

## 9. Common Issues locator
The final Common Issues artifact is not authored; its count is unknown. Use [MCP-01 failures](01_model_context_protocol.md#section-8--failure-modes--edge-cases): missing tools require checking discovery and principal filtering; write timeouts require authoritative status and reconciliation. These are hypothetical cases, not observed incident counts.

## 10. Empirical sources, versions and collection maintenance
The original source directories were unavailable and not inspected. MCP-01 is a draft; version-sensitive protocol claims, security guidance and illustrative arithmetic need independent review. Consult [standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [template](../../../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md); recheck protocol versions, transport and authorization behavior when dependencies change. No review date or production measurement is asserted.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

### Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR006 | LangGraph checkpoints and interrupts; authenticated resume; approval binding and expiry; idempotent submission | Crash-window replay, duplicate webhooks, changed purchase payloads, checkpoint migration and reconciliation tests. Boundary: External APIs may not support exactly-once effects; approval waiting times and run counts are exercise assumptions. | [REF-IR006](../../../../INTERVIEW_CASES/Agentic_AI_System/Langraph/durable_human_approval_graph.md) | planned; not written |
| PLAN-2026-IR014 | Schema versioning; semantic checks; dead-letter replay; idempotency; quality-constrained tiered RAG routing | Gold field accuracy and reconciliation for extraction; full-path cost and grounded quality by tier for routing. Boundary: Contains two reference problems: author separate narratives; valid JSON and cheap token prices do not establish successful outcomes. | [REF-IR014](../../../../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Structured_Extraction_And_Cost_Routing.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->

## 11. Collection acceptance checklist
- [ ] Author all three planned guides and a final Common Issues artifact containing at least 25 substantive issues.
- [ ] Independently review MCP-01's 12 sections, arithmetic, source versions and security boundaries.
- [ ] Validate links and current protocol/provider behavior; record actual review dates and evidence.
- [ ] Keep hypothetical diagnostics separate from observed incidents and measurements.
