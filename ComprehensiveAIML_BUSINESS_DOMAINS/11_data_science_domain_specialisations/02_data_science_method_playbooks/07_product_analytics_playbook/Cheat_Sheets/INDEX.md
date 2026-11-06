# Product Analytics Playbook — Cheat Sheets

## 1. Overview

This collection supports product analysts designing, measuring, and diagnosing decisions. Its first authored reference is an unreviewed draft, not a published or production-validated result. Readers need probability, rates, confidence intervals, SQL joins, and event-time concepts. The [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) govern all sheets. The [source coverage brief](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-methods-07) identifies the teaching home and remaining planned topics.

## 2. Topic ladder

| Level | Reference | Outcome and prerequisite | Status |
|---|---|---|---|
| Foundations | Instrumentation plan | Define event and identity contracts; basic SQL | planned |
| Foundations | North-star metric guide | Specify numerator, denominator, population and business objective | planned |
| Core / advanced | [08 A/B Testing Product Experiments](08_AB_Testing_Product_Experiments.md) | Design an assignment-level experiment, compute an effect, and diagnose failures; probability, confidence intervals and SQL | draft; unreviewed |
| Advanced | Experimentation guardrails | Define safety limits and monitoring | planned; partially addressed by sheet 08 |
| Final | Common Issues | At least 25 complete diagnostic entries | planned; not authored |

The source-path number 08 is a provenance identifier, not a claim that prerequisite sheets 01–07 have been authored here.

## 3. Quick topic lookup

| Question | Location | Use condition / warning |
|---|---|---|
| What is the assignment effect? | [Definitions](08_AB_Testing_Product_Experiments.md#3-foundational-definitions) | Define eligibility before treatment. |
| How do I compute a confidence interval? | [Derivations](08_AB_Testing_Product_Experiments.md#4-core-formulas-with-full-derivation) | Match independent unit, fixed-horizon design, and approximation assumptions. |
| What does this example imply for launch? | [Worked example](08_AB_Testing_Product_Experiments.md#5-worked-numerical-example) | Constructed numbers, not a measured deployment. |
| How do I diagnose mismatch or contamination? | [Failure modes](08_AB_Testing_Product_Experiments.md#8-failure-modes--edge-cases) | Distinguish assignment, delivery, and outcome logging. |
| How do I prepare for interviews? | [Interview questions](08_AB_Testing_Product_Experiments.md#11-interview--exam-questions) | Answers are draft and not independently reviewed. |

## 4. Reading paths

- **Beginner:** Review external probability and confidence-interval basics; read Sections 1–5 of sheet 08, then its card.
- **Practitioner:** Read sheet 08 Sections 1–9 in order, then rehearse the interview questions.
- **Advanced:** Read the full sheet, focusing on interference, design-dependent uncertainty, diagnostics, and theorem boundaries; independently consult the source ledger.
- These paths are intentionally short because other files are planned, not written.

## 5. Prerequisite skills

Calculate proportions and differences, interpret confidence intervals, and understand independent random assignment. Join event tables without fanout, distinguish event time from ingestion time, and define a product metric at the assigned-unit grain. A reader lacking these skills should consult an introductory statistics text and SQL documentation before beginning; the missing foundation sheets are not implied to exist.

## 6. What this collection does NOT cover

A complete causal inference curriculum, bandit optimization, statistical software implementation, and regulated clinical trials are outside sheet 08. Product instrumentation and standalone guardrail references remain planned. The sheet does not establish that any hypothetical production test occurred.

## 7. Master symbol glossary

| Symbol / term | Meaning | Units / type | Domain / first use |
|---|---|---|---|
| `n_A`, `n_B` | Eligible assigned-unit counts | nonnegative integer counts | Positive for rate calculation; [Section 4](08_AB_Testing_Product_Experiments.md#4-core-formulas-with-full-derivation). |
| `x_A`, `x_B` | Units with a qualifying binary outcome | integer counts | Between zero and corresponding `n`; Section 4. |
| `p_A`, `p_B` | Observed arm conversion rates | scalar proportions | Zero to one; Section 4. |
| `d` | Treatment minus control rate | proportion / percentage points | -1 to 1 for binary outcome; Section 4. |
| `r` | Difference divided by control rate | dimensionless relative lift | Requires positive control rate; Section 4. |
| `SE` | Estimated standard error of the contrast | proportion | Nonnegative; Section 4. |
| `N` | Total assigned units | integer count | Allocation diagnostic; Section 4; differs from event count `E` in Section 9. |
| `X`, `Y` | Pre-treatment covariate and outcome | unit-level scalar variables | Adjustment requires pre-treatment `X`; Section 4. |
| `alpha`, `beta` | Type-I and type-II error probabilities | scalar probabilities | Planning at a specified alternative; Section 4. |

## 8. Quick Reference Card directory

| Card | Rule | Condition / avoid |
|---|---|---|
| [Product experiments](08_AB_Testing_Product_Experiments.md#12-quick-reference-card) | Compare assigned arms; evaluate uncertainty and guardrails | Use the actual assignment mechanism; do not filter ITT on treatment-affected exposure. |

## 9. Common Issues locator

The final collection-level Common Issues artifact is **planned**, with zero completed collection-level issues; there is no link to an unwritten file. Sheet 08 contains its own hypothetical diagnostics in [Section 8](08_AB_Testing_Product_Experiments.md#8-failure-modes--edge-cases); those are not a completed collection-wide issues artifact.

## 10. Empirical sources, versions and maintenance

| Item | Claim and evidence | Status / maintenance |
|---|---|---|
| Sheet 08, v0.1 | Constructed examples and derived formulas; sources listed in Section 10 | Draft; not independently reviewed. Recheck calculation, links, and assumptions before publication. |
| Source inventory | `DS_Domain/Product_Analytics_Cheat_Sheets/08_AB_Testing_Product_Experiments.md` maps to `ds-methods-07` | Source filename is an inventory identifier; underlying content was not inspected. |
| Production behavior | Allocation, logging, interference and operational performance | Not measured; validate against the actual platform and workload. |

Review when assignment configuration, metric semantics, instrumentation, dependencies, or statistical policy changes. An editorial draft is not evidence that a production intervention was executed.

## 11. Collection acceptance checklist

- [x] Teaching home and authored draft identified without claiming source implementation was reviewed.
- [x] Authored sheet has a final card and a visible review status.
- [ ] Independent technical, numerical, source, and link review completed.
- [ ] Remaining foundational and planned topics authored as needed.
- [ ] Collection-level Common Issues artifact authored with at least 25 complete issues.
- [ ] All collection navigation, glossary, evidence records, and publication gates independently checked.
