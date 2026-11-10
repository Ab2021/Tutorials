# Data Science Method Playbooks

## 1. Overview
This family is a navigation layer for ten data-science method playbooks. It preserves the audit-derived topic names while distinguishing authored draft sheets from planned topics and un-authored cases. The available sheets teach bounded reusable calculations and diagnostics; they are not evidence of any named organization's implementation or outcome. Readiness for a real decision requires the named domain's evaluation population, time boundary, operational constraint, and independent review.

## 2. Topic ladder
| Order | Playbook | Available authored draft sheets | Original topic commitments | Status |
|---:|---|---|---:|---|
| 01 | [Computer vision](01_computer_vision_playbook/INDEX.md) | Object detection | 19 | partial draft coverage |
| 02 | [Fraud modelling](02_fraud_modelling_playbook/INDEX.md) | Evaluation/capacity; calibration; imbalance/thresholds | 16 | partial draft coverage |
| 03 | [KDA pharma](03_kda_pharma_playbook/INDEX.md) | None | 18 | planned |
| 04 | [Marketing analytics](04_marketing_analytics_playbook/INDEX.md) | Causal inference | 19 | partial draft coverage |
| 05 | [NLP and LLM](05_nlp_llm_playbook/INDEX.md) | RAG | 18 | partial draft coverage |
| 06 | [Pricing](06_pricing_playbook/INDEX.md) | None | 20 | planned |
| 07 | [Product analytics](07_product_analytics_playbook/INDEX.md) | A/B testing | 21 | partial draft coverage |
| 08 | [Recommendation systems](08_recsys_playbook/INDEX.md) | Candidate generation, eligibility and ranking | 22 | partial draft coverage |
| 09 | [Search and AdTech](09_search_adtech_playbook/INDEX.md) | BM25 and classical retrieval | 21 | partial draft coverage |
| 10 | [Time series](10_time_series_playbook/INDEX.md) | Rolling-origin validation and demand accuracy | 21 | partial draft coverage |

## 3. Quick topic lookup
| Question | Teaching home | Start with | Warning |
|---|---|---|---|
| How do I match boxes and measure detection? | [Computer vision](01_computer_vision_playbook/Cheat_Sheets/04_Object_Detection.md) | Section 4 | Box overlap is not mask or deployment evaluation. |
| How should scarce fraud reviews be allocated? | [Fraud modelling](02_fraud_modelling_playbook/Cheat_Sheets/01_Fraud_Evaluation_Calibration_and_Capacity.md) | Section 4 | Action value and labels must be credible. |
| Is a marketing effect causal? | [Marketing analytics](04_marketing_analytics_playbook/Cheat_Sheets/21_Causal_Inference_Marketing.md) | Section 4 | Association is not an intervention effect. |
| Is retrieved evidence adequate for an answer? | [NLP and LLM](05_nlp_llm_playbook/Cheat_Sheets/13_RAG_Retrieval_Augmented_Generation.md) | Section 4 | Retrieval does not authorize disclosure. |
| Is a product experiment interpretable? | [Product analytics](07_product_analytics_playbook/Cheat_Sheets/08_AB_Testing_Product_Experiments.md) | Section 4 | Preserve assignment and outcome definitions. |
| Is retrieval eligible and comparable? | [Recommendation systems](08_recsys_playbook/Cheat_Sheets/01_Candidate_Generation_Eligibility_and_Ranking.md) | Section 4 | Use the as-of authorized catalog. |
| How is lexical retrieval scored? | [Search and AdTech](09_search_adtech_playbook/Cheat_Sheets/02_BM25_and_Classical_Retrieval.md) | Section 4 | Analyzer and corpus statistics define the score. |
| How should demand forecasts be replayed? | [Time series](10_time_series_playbook/Cheat_Sheets/04_Rolling_Origin_Validation_And_Demand_Accuracy.md) | Section 4 | Score only information available at the origin. |

## 4. Reading paths
- **Beginner:** Select one domain, read its Cheat_Sheets INDEX prerequisites, then its available sheet through the worked example. No nonexistent prerequisite is implied.
- **Practitioner:** Read the available sheet end-to-end, reproduce only its illustrative arithmetic, and map its stated assumptions to a real decision contract.
- **Advanced:** Use the domain index to identify un-authored adjacent topics, then independently evaluate data timing, evidence, operational costs, and governance before adaptation.

## 5. Prerequisite skills
Readers need proportions and conditional probability; timestamped data and point-in-time joins; experimental or holdout reasoning; and domain-specific workflow knowledge. Vector, ranking, optimization, or time-series prerequisites are declared in each local collection rather than silently assumed.

## 6. What this family does NOT cover
This index does not complete the 195 original topical commitments, author any of the planned cases, establish company-specific results, or certify production readiness. The source catalog names were planning scope only; source bodies were unavailable. Follow each child collection's explicit boundary and handoff.

## 7. Master symbol glossary
| Symbol/term | Meaning | Units/type | Scope |
|---|---|---|---|
| `t` | Decision or observation time | timestamp | Must be defined locally. |
| `y` | Outcome or target | domain-specific quantity | Availability and maturity matter. |
| `p` | Probability or score only when locally defined | dimensionless scalar | Never assume calibration. |
| `K` | Capacity or rank cutoff only when locally defined | count | Overloaded across domains; consult local sheet. |
| `N` | Population/count only when locally defined | count | Denominator must be stated. |

## 8. Quick Reference Card directory
Available Section 12 cards: [CV](01_computer_vision_playbook/Cheat_Sheets/04_Object_Detection.md#section-12--quick-reference-card), [fraud](02_fraud_modelling_playbook/Cheat_Sheets/01_Fraud_Evaluation_Calibration_and_Capacity.md#12-quick-reference-card), [marketing](04_marketing_analytics_playbook/Cheat_Sheets/21_Causal_Inference_Marketing.md#section-12--quick-reference-card), [RAG](05_nlp_llm_playbook/Cheat_Sheets/13_RAG_Retrieval_Augmented_Generation.md#12-quick-reference-card), [experiments](07_product_analytics_playbook/Cheat_Sheets/08_AB_Testing_Product_Experiments.md#12-quick-reference-card), [recsys](08_recsys_playbook/Cheat_Sheets/01_Candidate_Generation_Eligibility_and_Ranking.md#12-quick-reference-card), [search](09_search_adtech_playbook/Cheat_Sheets/02_BM25_and_Classical_Retrieval.md#12-quick-reference-card), and [forecasting](10_time_series_playbook/Cheat_Sheets/04_Rolling_Origin_Validation_And_Demand_Accuracy.md#12-quick-reference-card). Card presence is not substantive acceptance of the surrounding draft.

## 9. Common Issues locator
No child collection has a completed final Common Issues artifact. Local Section 8 failure exercises are not collection-wide issue records. Do not report an issue count beyond the actual files indexed in a child collection.

## 10. Empirical sources, versions and maintenance
The source assignments are in [Data science source coverage](../../00_SOURCE_COVERAGE_DATA_SCIENCE.md) and the [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md). Both state that visible names establish scope, not access to original bodies. Existing sheet examples are labelled illustrative/derived where applicable and remain draft/unreviewed unless their own frontmatter says otherwise. Revisit navigation when a child artifact, source, metric definition, or decision-time assumption changes.

## 11. Collection acceptance checklist
- [x] Ten child playbooks and existing draft-sheet links are enumerated without claiming planned material is authored.
- [x] Three reading paths, lookup, prerequisites, glossary, cards, issues state, and source limits are explicit.
- [ ] Every original topic and source-assigned commitment is authored as a complete sheet or case.
- [ ] Each applicable collection has its final Common Issues artifact with the required complete issue count.
- [ ] Independent technical, source, navigation, and editorial review is complete.
