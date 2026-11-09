# Module 49 Case Studies — AI Security, Privacy, Safety and Red Teaming

> **Collection status:** draft. Follow the [course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [case master](../../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md). An index row is not a completed case; evidence level, implementation stage, and editorial state are independent.

## 1. Collection metadata

**Domain:** AI security, privacy, safety, and red teaming. **Purpose:** bounded decisions about tool-enabled AI risk. **Case types:** `llm-agentic`, `incident`, and `analytical` as appropriate. **Evidence policy:** new examples are illustrative or not-measured unless a primary source is cited at claim level. **Source scope:** [directory inventory, lines 86–94](../../../../Templates/directory.txt), [trust brief](../../../00_SOURCE_COVERAGE_AGENTS.md#agent-trust-approvals), and [safety brief](../../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-safety); source bodies are unavailable.

## 2. Case catalog

| Case ID | Linked title | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result/limitation | Editorial status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CASE-49-001 | [Approval-bound export](01_Agent_Data_Exfiltration_Simulation.md) | data export | llm-agentic | permit bounded export or retain manual baseline | authorization-scope | approval-payload-binding; evaluation-mismatch | illustrative | proposed | null | no execution/outcome measured | draft |
| CASE-49-002 | Secure RAG design review | retrieval | llm-agentic | select retrieval/authorization boundary | authorization-scope | data-leakage | not-measured | proposed | null | planned | planned |
| CASE-49-003 | Tenant adapter isolation | model customization | llm-agentic | train/select/rollback tenant adapter | authorization-scope | atomic-release-bundle; rollout-regression | not-measured | proposed | null | PLAN-2026-IR004 reference only | planned |
| CASE-49-004 | Release evaluation gate | safety evaluation | llm-agentic | promote candidate after required suites | evaluation-mismatch | data-leakage; rollout-regression | not-measured | proposed | null | PLAN-2026-IR008 reference only | planned |

## 3. Canonical reusable patterns

| Pattern | Mechanism | Use when | Do not infer |
|---|---|---|---|
| authorization-scope | executor checks authenticated principal, tenant, purpose, quota, expiry | an AI proposal can cause effect | prompt text grants authority |
| approval-payload-binding | reviewer sees and approves exact canonical payload | human decision is required | digest establishes semantic correctness |
| evaluation-mismatch | oracle measures committed effect and benign task behavior | release score may hide harm | passing suite proves safety |
| data-leakage | enforce time/tenant/purpose boundary through lineage | retrieval/adapters/evaluation may mix scope | filter name proves isolation |
| atomic-release-bundle | promote compatible policy/model/tool/index state together | components interact | an alias rollback is sufficient |

## 4. Similarity / contrast map

CASE-49-001 and CASE-49-002 share tool/retrieval trust boundaries; the first focuses on an export effect, the second on secure information access. CASE-49-003 focuses on adapter/session isolation rather than tool execution. CASE-49-004 focuses on evidence for promotion rather than runtime authorization. All require an explicit baseline and a denial/recovery trace; none may claim a company outcome from the planning references.

## 5. Questions / future-case backlog

- What secure-RAG access policy handles document revocation during a live session?
- Which tenant adapter artifact tuple must be pinned at inference and rollback?
- What evaluation manifest constitutes a critical safety gate, and which failures are inconclusive?
- How should an incident case distinguish model text harm from committed external harm?
- Which visible safety source topic requires a separate case rather than a supporting sheet?

## 6. Review / retirement log

| Date | Item | Review event | Outcome | Trigger for recheck |
|---|---|---|---|---|
| 2026-11-09 | CASE-49-001 | authored illustrative draft | not reviewed; no result measured | any executor/identity/policy change |
| 2026-11-09 | CASE-49-002–004 | catalog reconciliation | planned only | authoring begins |

## 7. Pre-publish checks

- [x] Original two case rows retained; reference-informed cases crosswalked.
- [x] One bounded illustrative case links its supporting sheets and declares evidence boundary.
- [ ] Every catalog case has 15 full sections, 12–18 fully developed interview answers, checked arithmetic/traces, and at least 1,000 substantive lines.
- [ ] All case claims have primary evidence or remain expressly illustrative/not-measured.
- [ ] Local navigation and artifacts have been independently reviewed; no case is published.
