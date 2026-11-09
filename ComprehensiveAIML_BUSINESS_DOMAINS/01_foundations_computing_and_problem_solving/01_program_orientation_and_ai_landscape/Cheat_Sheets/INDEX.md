# Module 01 Cheat Sheets — Program Orientation and AI Landscape

## 1. Overview

These draft sheets teach decision framing, capability/deployment classification, and evidence-led technology review. They do not reconstruct unavailable source containers `GLCC`, `learn`, `raw`, or `speechtotxt` listed in [`directory.txt`](../../../../Templates/directory.txt). Every empirical candidate claim needs current primary evidence.

## 2. Topic ladder

| Level | File | Outcome | Prerequisite | Status |
|---|---|---|---|---|
| Foundation | [01 AI Use-Case Charter](01_AI_Use_Case_Charter.md) | define decision, baseline, risk and gate | none | draft |
| Foundation | [02 AI Market Map](02_AI_Market_Map.md) | choose capability/control category | 01 | draft |
| Core | [03 Technology Due Diligence](03_Technology_Due_Diligence.md) | turn claims into acceptance tests | 01–02 | draft |
| Final | Common Issues | diagnose recurring framing/claim failures | all sheets | not authored |

## 3. Quick topic lookup

| Question | File / section | Use condition | Failure warning |
|---|---|---|---|
| What problem exists before model choice? | 01 §2–4 | new proposal | solution-first framing |
| Which deployment is credible? | 02 §2, §7 | comparing modes | treating model label as control design |
| What proves a vendor claim? | 03 §3–5 | procurement/pilot | averaging a hard constraint away |

## 4. Reading paths — three options

- **Beginner:** 01 → 02; frame a decision, then classify the smallest credible option.
- **Practitioner:** 01 → 02 → 03; write a charter and acceptance protocol.
- **Advanced:** 01 → 02 → 03 → module case; challenge assumptions, capacity and evidence expiry.

## 5. Prerequisite skills

Readers need basic percentage arithmetic, ability to name a workflow owner, and a distinction between constructed examples and measurements. No prior ML implementation is required.

## 6. Exclusions and handoffs

This collection does not certify security, legal compliance, vendor performance, or source-container contents. Hand off model implementation to Module 02 and capacity/reliability design to Module 04.

## 7. Master symbol glossary

| Symbol | Meaning | Units/type | Valid range | First use |
|---|---|---|---|---|
| `N` | eligible units/window | count | nonnegative | 02 §4 |
| `q` | routed-to-review fraction | fraction | [0,1] | 02 §4 |
| `r` | review capacity/window | count | nonnegative | 02 §4 |
| `V` | estimated incremental value | currency/window | assumption-dependent | 02 §4 |
| `TCO` | total cost of ownership | currency/window | declared inclusions | 03 §4 |

## 8. Quick Reference Card directory

| File | Card | Core rule | Critical assumption |
|---|---|---|---|
| 01 | §8 | charter before model choice | baseline is observable |
| 02 | §12 | `q*N <= r` | stable-rate sizing approximation |
| 03 | §12 | hard gates precede scores | candidate scope is known |

## 9. Common Issues locator

No Common Issues artifact exists yet. The course contract requires a final twelve-section troubleshooting artifact with at least 25 complete distinct issues; it is not represented by a placeholder link.

## 10. Sources, versions and maintenance

| Source | Claim supported | Reviewed | Limitation / trigger |
|---|---|---|---|
| [`directory.txt`](../../../../Templates/directory.txt) lines 1–13 | visible inventory and unavailable bodies | 2026-11-09 | names do not establish content |
| NIST AI RMF / SP 800-53 / SP 800-161r1 | cited method vocabulary | 2026-11-09 | recheck version and candidate applicability |

All files are draft, unreviewed teaching content. Arithmetic is hand-derived; no system or vendor test has been executed.

## 11. Acceptance checklist

- [x] Three linked draft sheets with twelve ordered sections and final cards
- [x] Prerequisite order, glossary, reading paths, and source boundary
- [ ] Each sheet reaches the course’s 1,000-substantive-line requirement
- [ ] Final 25-issue Common Issues artifact
- [ ] Independent link, arithmetic, source, and editorial review