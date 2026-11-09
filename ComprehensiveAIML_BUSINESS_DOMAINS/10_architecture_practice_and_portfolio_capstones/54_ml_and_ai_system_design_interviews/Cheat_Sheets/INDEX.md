# Module 54 Cheat Sheets — ML and AI System Design Interviews

## 1. Overview
This collection teaches requirement discovery, ML/LLM architecture, capacity/cost, and trade-off communication. Its first foundation sheet is an illustrative, trace-checked decision framework; it does not certify interview performance, production architecture, vendor behavior, or benchmark results.

## 2. Topic ladder
| Level | File/topic | Outcome | Prerequisite | Status |
|---|---|---|---|---|
| Foundation | [01 Interview decision framework](01_Interview_Decision_Framework.md) | turn ambiguity into a decision contract; compute scoped action/capacity/exam-score checks; diagnose failure boundaries | prior system/ML basics | draft; illustrative arithmetic and DERIVED traces |
| Core | architecture whiteboard prompts | draw owned data/control/failure paths | framework | planned |
| Core | metric selection | define objective, guardrails, denominator | statistics | planned |
| Advanced | IR020 routing/counters/search | defend distributed correctness contract | DSA/distributed systems | planned |
| Advanced | IR026 operation-mix structures | select structure under resource/failure bounds | DSA | planned |
| Advanced | IR034 cache/typeahead/rank/dedupe | state freshness and consistency contract | DSA/serving | planned |
| Final | Common Issues | diagnose at least 25 complete issues | all sheets | planned; no file |

## 3. Quick topic lookup
| Question | File | Section | Warning |
|---|---|---|---|
| What do I ask first? | [01 Interview framework](01_Interview_Decision_Framework.md#2-concept-map--mental-model) | Sections 2–3 | prompt is ambiguous | tools are not requirements |
| Which denominator should I use? | [01 Interview framework](01_Interview_Decision_Framework.md#44-practice-exam-scoring-is-a-decision-not-a-vanity-percentage) | Section 4.4 | score/remediation decision | answered-only can hide omissions |
| Is a cache safe to publish? | planned IR034 sheet | consistency contract | cache/typeahead work | stale rank is not exact rank |

## 4. Reading paths — three options
**Beginner:** [framework](01_Interview_Decision_Framework.md) → architecture prompts → metrics. **Practitioner:** framework plus capacity/cost and one bounded design case. **Advanced:** all planned DSA reference additions, then failure/recovery drills. Only the framework link is an authored artifact; remaining names are planned.

## 5. Prerequisite skills
Problem decomposition; queues and rates; ML evaluation; basic retrieval; distributed consistency; and complexity analysis. The module index lists course-module prerequisites. No hidden source body is assumed available.

## 6. Exclusions and handoffs
This collection does not certify interview performance, production architecture, vendor behavior, or benchmark results. Algorithm-specific proofs and implementation belong in their relevant DSA homes; visible interview examples remain reference-only.

## 7. Master symbol glossary
| Symbol | Meaning | Units/type | First planned use |
|---|---|---|---|
| `lambda` | arrival rate | requests/s | capacity sheet |
| `mu` | service rate | requests/s | capacity sheet |
| `B` | backlog | requests | capacity sheet |
| `p` | probability, never feature count | dimensionless | metrics sheet |

## 8. Quick Reference Card directory
| File/card | Core rule | Use condition | Critical assumption | Avoid condition |
|---|---|---|---|---|
| [01 card](01_Interview_Decision_Framework.md#12-quick-reference-card) | decision → metric → capacity → recovery → evidence | bounded design answer | stated units and ownership | unsupported performance claim |

## 9. Common Issues locator
No issue file exists. The final file must retain 12 headings and contain >=25 distinct complete issues with required categories, provenance, competing causes, diagnostics, correction, prevention, and verification. The diagnostics in File 01 are teaching dossiers, not the final collection artifact.

## 10. Sources, versions and maintenance
| Source | Scope | Limitation | Trigger |
|---|---|---|---|
| [Module index](../INDEX.md) | five original topics and three plans | planned syllabus only | index/source change |
| [01 framework](01_Interview_Decision_Framework.md) | DERIVED decision, queue, scoring, latency, and idempotency traces; ILLUSTRATIVE inputs | not a deployment or exam result | metric, policy, or source-plan change |
| [REF-IR020](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_03.md) | reference example | not evidence of implementation | source review |
| [REF-IR026](../../../../INTERVIEW_CASES/DSA/index_basics/INTERVIEW_CASE_STUDIES.md) | reference example | methods must be separately authored | source review |
| [REF-IR034](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/INTERVIEW_SYSTEM_DESIGN_CASES.md) | reference example | no proprietary results | source review |

## 11. Collection acceptance checklist
- [ ] Every original and reference-informed commitment has a substantive artifact.
- [ ] Every sheet has 12 sections, checked task, diagnostics, interview answers, and final card.
- [ ] A complete final Common Issues artifact exists.
- [ ] Primary factual sources and unstable claims are version-pinned and reviewed.
