# Module 54 Cheat Sheets — ML and AI System Design Interviews

## 1. Overview
This collection will teach requirement discovery, ML/LLM architecture, capacity/cost, and trade-off communication. No individual sheet exists yet; planning rows are not links or completed lessons.

## 2. Topic ladder
| Level | File/topic | Outcome | Prerequisite | Status |
|---|---|---|---|---|
| Foundation | interview framework | turn ambiguity into a decision contract | prior system/ML basics | planned |
| Core | architecture whiteboard prompts | draw owned data/control/failure paths | framework | planned |
| Core | metric selection | define objective, guardrails, denominator | statistics | planned |
| Advanced | IR020 routing/counters/search | defend distributed correctness contract | DSA/distributed systems | planned |
| Advanced | IR026 operation-mix structures | select structure under resource/failure bounds | DSA | planned |
| Advanced | IR034 cache/typeahead/rank/dedupe | state freshness and consistency contract | DSA/serving | planned |
| Final | Common Issues | diagnose at least 25 complete issues | all sheets | planned; no file |

## 3. Quick topic lookup
| Question | File | Section | Warning |
|---|---|---|---|
| What do I ask first? | planned interview framework | decision contract | tools are not requirements |
| Which metric matters? | planned metric selection | objective/guardrails | proxy can reverse value |
| Is a cache safe to publish? | planned IR034 sheet | consistency contract | stale rank is not exact rank |

## 4. Reading paths — three options
**Beginner:** framework → architecture prompts → metrics. **Practitioner:** beginner path plus capacity/cost and one bounded design case. **Advanced:** all planned DSA reference additions, then failure/recovery drills. Paths become links only when files exist.

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
No cards exist. Each authored sheet must end in its own card; planned names above are not card links.

## 9. Common Issues locator
No issue file exists. The final file must retain 12 headings and contain >=25 distinct complete issues with required categories, provenance, competing causes, diagnostics, correction, prevention, and verification.

## 10. Sources, versions and maintenance
| Source | Scope | Limitation | Trigger |
|---|---|---|---|
| [Module index](../INDEX.md) | five original topics and three plans | planned syllabus only | index/source change |
| [REF-IR020](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_03.md) | reference example | not evidence of implementation | source review |
| [REF-IR026](../../../../INTERVIEW_CASES/DSA/index_basics/INTERVIEW_CASE_STUDIES.md) | reference example | methods must be separately authored | source review |
| [REF-IR034](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/INTERVIEW_SYSTEM_DESIGN_CASES.md) | reference example | no proprietary results | source review |

## 11. Collection acceptance checklist
- [ ] Every original and reference-informed commitment has a substantive artifact.
- [ ] Every sheet has 12 sections, checked task, diagnostics, interview answers, and final card.
- [ ] A complete final Common Issues artifact exists.
- [ ] Primary factual sources and unstable claims are version-pinned and reviewed.
