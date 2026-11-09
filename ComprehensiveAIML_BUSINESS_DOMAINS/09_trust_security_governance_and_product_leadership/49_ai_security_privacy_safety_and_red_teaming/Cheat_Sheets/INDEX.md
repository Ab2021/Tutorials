# Module 49 Cheat Sheets — AI Security, Privacy, Safety and Red Teaming

> **Collection status:** draft collection. Individual files below are authoring drafts, not reviewed security guidance or evidence of a deployment.
> **Authoring contract:** [course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [cheat-sheet master](../../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md). Each authored sheet must eventually meet the full 12-section and 1,000-substantive-line gate; no listed planning row is completed merely by appearing here.

## 1. Overview

This collection teaches an AI workflow owner to separate untrusted content, model proposals, authenticated identity, authorization, approval, execution, and recovery. It covers threat modelling, prompt injection, tool abuse, supply-chain boundaries, red-team evaluation, safety evidence, and incident response. Readiness means a reader can trace a proposed harmful action to an enforceable boundary, calculate a stated exposure/capacity rule, run a declared oracle, and state what evidence is still missing.

## 2. Topic ladder

| Level | File | Outcome | Prerequisites | Key method | Editorial state |
|---|---|---|---|---|---|
| 1 | [01 Threat Model and Execution Trust](01_AI_Threat_Model_and_Execution_Trust.md) | Bound effects by authenticated scope and approval binding | identity/API basics | `B_max=q*c*b` | draft |
| 2 | [02 Prompt Injection and Red-Team Evaluation](02_Prompt_Injection_and_Red_Team_Evaluation.md) | Gate a release on effectful adversarial evidence | 01 | UCR/FRR and effect oracle | draft |
| 2 | Supply-chain integrity and dependency provenance | Verify model/data/tool artifact lineage | 01 | signed manifest and revocation trace | planned |
| 3 | Safety, alignment, interpretability, and grounding | Separate evidence of behavior from safety claims | 02 | false-refusal/unsafe-compliance slices | planned |
| 3 | Incident response and recovery | Contain, investigate, and recover from an AI incident | 01–02 | effect-journal reconciliation | planned |
| Final | Common Issues | Diagnose recurring trust failures | all completed sheets | issue-specific oracle | planned |

Visible source titles `execution_surfaces`, `identity_and_trust`, `approval_gates`, `sandboxing_and_isolation`, and `blast_radius_analysis` are covered as explicit scope in File 01 but source bodies were unavailable; expansion into independently complete focused lessons remains open. Fine-tuning Module 13 titles remain separate planned scope: alignment principles, safety training, mechanistic interpretability, and hallucination causes/mitigation.

## 3. Quick topic lookup

| Question | File / section | Use condition | Failure warning |
|---|---|---|---|
| Can model text execute an effect? | 01 §4.3, §7 | any tool or external action | prompt policy is not authorization |
| How much can one approval export? | 01 §4.2–5 | quota-based capability | count commit, not requests |
| What is an unsafe-compliance rate? | 02 §4.1 | adjudicated prohibited suite | not an attack prevalence estimate |
| Why add benign tests? | 02 §4.1, §5 | refusal-capable system | low UCR can hide unusable blocking |
| How do I test indirect injection? | 02 §4.3, §8 | retrieval/tool content enters context | text-only oracle misses effects |

## 4. Reading paths — three options

**Beginner:** 01 §§2–5 → 02 §§2–5. Objective: identify an execution boundary and interpret UCR/FRR.

**Practitioner:** 01 → 02 → supply-chain integrity (planned) → incident response (planned). Objective: design a release-ready control/evidence path. Do not skip the planned files when a real deployment depends on them.

**Advanced:** 01 → 02 → all planned files → final Common Issues (planned). Objective: challenge threat assumptions, evaluator fidelity, and recovery evidence. Optional extension: [approval-gate case](../../../../06_llm_agents_knowledge_systems_and_ai_augmented_development/30_agentic_systems_workflows_and_orchestration/Case_Studies/01_enterprise_agent_approval_gate.md), which is hypothetical and unreviewed.

## 5. Prerequisite skills

| Skill | Why needed | Available reference |
|---|---|---|
| Authenticated identity and authorization | distinguish principal from model text | [01 §3](01_AI_Threat_Model_and_Execution_Trust.md#3-foundational-definitions) |
| Typed API/tool contracts | identify schema versus policy checks | [01 §7](01_AI_Threat_Model_and_Execution_Trust.md#7-comparison--selection-table) |
| Conditional rates | calculate UCR and FRR | [02 §4](02_Prompt_Injection_and_Red_Team_Evaluation.md#4-core-formulas-with-full-derivation) |
| Trace and receipt interpretation | establish whether an effect occurred | [02 §8](02_Prompt_Injection_and_Red_Team_Evaluation.md#8-failure-modes--edge-cases) |

## 6. What this collection does NOT cover

It does not certify compliance, establish applicable law, replace a system-specific penetration-test authorization, or claim a product/model/vendor is secure. It does not teach cryptographic implementation, secure coding, model training, or forensic preservation in full; those require specialist handoffs. Unavailable original source bodies remain unavailable.

## 7. Master symbol glossary

| Symbol/term | Meaning | Units/type | Domain | First use |
|---|---|---|---|---|
| `p` | scenario path-success probability | fraction | [0,1] | [01 §4.1](01_AI_Threat_Model_and_Execution_Trust.md#41-base-expected-loss-priority) |
| `I` | conditional incident loss | currency/event | stated scenario | 01 §4.1 |
| `q,c,b` | items/call, calls/approval, bytes/item | records/call, calls, bytes/record | nonnegative limits | 01 §4.2 |
| `B_max` | maximum payload under authorization | bytes | payload-only cap | 01 §4.2 |
| `UCR` | unsafe-compliance rate | fraction | adjudicated prohibited tests | [02 §4.1](02_Prompt_Injection_and_Red_Team_Evaluation.md#41-unsafe-compliance-and-false-refusal-rates) |
| `FRR` | false-refusal rate | fraction | adjudicated benign tasks | 02 §4.1 |
| `C_w` | weighted coverage | fraction | declared category weights | 02 §4.2 |

## 8. Quick Reference Card directory

| File | Card | Core rule | Critical assumption | Avoid when |
|---|---|---|---|---|
| [01](01_AI_Threat_Model_and_Execution_Trust.md#12-quick-reference-card) | §12 | executor authorizes exact effect | authenticated attributes are trustworthy | model output alone carries authority |
| [02](02_Prompt_Injection_and_Red_Team_Evaluation.md#12-quick-reference-card) | §12 | measure effectful UCR and benign FRR | oracle observes all commits | suite is text-only |

## 9. Common Issues locator

`99_25_Common_Issues.md` is planned and does not yet exist. It must contain at least 25 distinct complete issues under the 12-section troubleshooting structure, including issue ID, category, provenance, competing causes, diagnostic signal, correction, prevention, verification, and related sheet. Existing part-level `99_30_Common_Issues.md` is cross-module support, not a substitute for this module-local final artifact.

## 10. Sources, versions and collection maintenance

| Source ID | Primary source | Claim/scope | Version/date | Accessed | Limitation / recheck trigger |
|---|---|---|---|---|---|
| SRC-49-01 | NIST, *AI RMF 1.0*, https://doi.org/10.6028/NIST.AI.100-1 | risk-management vocabulary | January 2023 | 2026-11-09 | framework, not system evidence; recheck revised edition |
| SRC-49-02 | OWASP Prompt Injection Prevention Cheat Sheet, https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html | layered prompt-injection controls | live page | 2026-11-09 | not a deployment result; recheck on policy/tool change |
| SRC-49-03 | [directory inventory](../../../Templates/directory.txt#L86) | visible source-name provenance | inventory snapshot | 2026-11-09 | source bodies/code unavailable |
| SRC-49-04 | [agent trust brief](../../../00_SOURCE_COVERAGE_AGENTS.md#agent-trust-approvals) | syllabus additions and boundaries | planned brief | 2026-11-09 | not a source lesson |
| SRC-49-05 | [safety brief](../../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-safety) | four safety-topic commitments | planned brief | 2026-11-09 | not empirical safety evidence |

## 11. Collection acceptance checklist

- [x] Indexed original sheet commitments preserved and two real draft files linked.
- [x] Source inventory and planning-brief boundaries are explicit.
- [x] Topic ladder, lookup, reading paths, prerequisites, exclusions, glossary, card directory, issue locator, and source register exist.
- [ ] Every original/source-added topic has a separate 1,000-substantive-line authored sheet.
- [ ] Final module-local Common Issues has 25 distinct complete entries.
- [ ] Each draft has independent technical/editorial review and all local links are publication-checked.
