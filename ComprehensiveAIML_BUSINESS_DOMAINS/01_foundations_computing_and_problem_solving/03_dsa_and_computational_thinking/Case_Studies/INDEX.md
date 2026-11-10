# 03 Dsa And Computational Thinking Case Studies

## 1. Collection metadata
Illustrative teaching cases for bounded decisions. Evidence, implementation stage, and editorial state are separate; every listed artifact remains draft and no case claims a deployment.

## 2. Case catalog
| Case ID | Title | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-001 | [01 Design Rate Limiter ](01_Design_Rate_Limiter.md) | foundations | analytical | bounded illustrative decision | correctness-contract | — | illustrative | proposed | 2026-11-10 | Draft; source verification and publication review OPEN |
| CASE-002 | [02 Implement Top K Stream Processing ](02_Implement_Top_K_Stream_Processing.md) | foundations | analytical | bounded illustrative decision | correctness-contract | — | illustrative | proposed | 2026-11-10 | Draft; source verification and publication review OPEN |

## 3. Canonical reusable patterns
- **Correctness contract:** define the input cutoff, output, owner, and oracle before optimizing.
- **Bounded work:** cap admission and retain a safe deferral path when capacity is uncertain.
- **Idempotent decision ledger:** one final action per stable decision key.

## 4. Similarity / contrast map
All cases use illustrative queue arithmetic and a human approval boundary. Their decision units differ; no workload or outcome transfers between them without new evidence.

## 5. Questions and future-case backlog
Source-derived reference commitments and unavailable source bodies remain OPEN. Future cases need primary evidence or explicitly illustrative scope, not reconstructed results.

## 6. Review / retirement log
| Date | Change | Editorial state | Limitation |
|---|---|---|---|
| 2026-11-10 | Canonical cases cataloged | draft | publication review, source verification, and operational validation OPEN |

## 7. Pre-publish checks
- [x] Each listed link resolves to a canonical manuscript.
- [x] Evidence and stage are not reported as measured deployment.
- [ ] Independently review source claims, detailed arithmetic, links, and editorial completeness before publication.
