# Data science methods, production cases and graph ML — source coverage

**State:** planned topic architecture and future-content briefs. No full lesson, implemented system, measured outcome, completed issue entry or reviewed interview answer is claimed. Source inventory reviewed for planning only.

## Scope and authoring authority

This plan assigns **365 exact visible source entries** from the first fenced tree in the [source audit](../Templates/archive/TRAIN_DIRECTORY_STRUCTURE_AUDIT_56_IMAGES.md), matching the supplied tree. It extends the local course indexes rather than replacing their existing topics. Source contents were not supplied; names establish syllabus scope, not technical correctness or source access.

Apply the [course authoring and numerical standards](00_AUTHORING_AND_NUMERICAL_STANDARDS.md), [case master](../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md), and [cheat-sheet master](../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md). Future cases require 15 sections and a seven-section collection INDEX; future sheets require 12 sections and an 11-section collection INDEX. Each substantive equation needs validity, derivation and FORMULA → WHERE → CONSTRAINT → NUMERICAL EXAMPLE → INTERPRETATION, with units and a counterexample. Use a precise invariant/trace where a closed-form equation is not meaningful.

Each visible technical title in the registers below is a **separate focused-topic minimum**. Split independent methods into focused sheets; reuse a genuine equivalent sheet rather than duplicating it. A group-level exercise does not replace the topic-specific derivations, numerical examples, four diagnostic failures, checked interview answers, source ledger and final card required when each sheet is authored. Each distinct source case remains a separately framed case, not an oversized anthology.

All exercise inputs are **ILLUSTRATIVE** and all validation protocols are **proposed, not executed**. No named company, harness, framework, device, regulator or frontier model behavior is certified by a filename. Pin primary sources, versions, geography and dates before attributing architecture, rules or results. Existing 50/100 Common Issues titles remain targets; none are counted as completed.

Hidden folders and clipped filenames remain `unresolved-source`; visible prefixes and child titles may support limited topic planning but their full path/content is not invented. Ancillary READMEs, source PDFs, collators, archives and generic exercises are tracked without counting them as authored lessons. The permanent [source crosswalk](00_SOURCE_TOPIC_CROSSWALK.md) records every node and disposition.

## Focused teaching groups

- [Fraud detection](#ds-cases-01)
- [Graph ML production decisions](#ds-cases-02)
- [Pharma key-driver analysis](#ds-cases-03)
- [Marketing measurement](#ds-cases-04)
- [NLP/LLM production systems](#ds-cases-05)
- [Pricing and revenue decisions](#ds-cases-06)
- [Product experimentation and growth](#ds-cases-07)
- [Recommendation pipelines](#ds-cases-08)
- [Search and advertising systems](#ds-cases-09)
- [Forecasting production decisions](#ds-cases-10)
- [US pharma commercialization decisions](#ds-cases-11)
- [Computer vision](#ds-methods-01)
- [Fraud modeling](#ds-methods-02)
- [Pharma KDA](#ds-methods-03)
- [Marketing analytics](#ds-methods-04)
- [NLP and LLM methods](#ds-methods-05)
- [Pricing methods](#ds-methods-06)
- [Product analytics](#ds-methods-07)
- [Recommender methods](#ds-methods-08)
- [Search and AdTech methods](#ds-methods-09)
- [Time series and demand planning](#ds-methods-10)
- [Production ML disciplines, architectures, features, monitoring, governance, training/serving, PoC roadmap and production traps](#ds-ml-design)
- [ML product-sense framing, metrics, recommendations, search, fraud, NLP, forecasting, agents, CV, causal experiments and PM interviews](#ds-product-sense)
- [ML design cases, hidden architecture/production guides, 2022+ cases, PDFs and source utilities](#ds-practical-sources)
- [Graph theory, GNNs, knowledge graphs/ontologies and fraud detection](#graph-source)
- [DS source roots, production-case catalog and hidden US commercialization cheat sheets](#ds-source)

<a id="ds-cases-01"></a>

## Fraud detection

**Teaching home:** [11_data_science_domain_specialisations/01_production_ml_case_library/01_fraud_modelling/INDEX.md](11_data_science_domain_specialisations/01_production_ml_case_library/01_fraud_modelling/INDEX.md). **Status:** planned.

**Testable outcome.** Frame, design and challenge each of the five separately listed production cases; do not combine independent decisions into one narrative.

**Focused-sheet derivations and mechanisms.** Derive expected error cost, precision/recall at investigator capacity, calibration and label availability; graph signals do not establish fraud.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 10000 events include 100 frauds; 200 alerts include 60 frauds: compute precision, recall, false positives and reviews per true positive. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Five independent case questions are registered in the exact-topic table below.

**Proposed validation.** Temporal/entity split, mature labels, fixed review capacity, adversarial shift and alert/decision replay.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Future-information leakage; proxy-only success; selected or immature outcomes; unmeasured operational failure. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Case_Studies_Production_ML_Systems/Fraud_Modelling` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Fraud_Modelling/01_Real_Time_Payment_Card_Fraud_Detection.md` | Real Time Payment Card Fraud Detection | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Fraud_Modelling/02_Insurance_Claims_Fraud_Detection.md` | Insurance Claims Fraud Detection | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Fraud_Modelling/03_Account_Takeover_Application_Fraud.md` | Account Takeover Application Fraud | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Fraud_Modelling/04_AML_Transaction_Monitoring_System.md` | AML Transaction Monitoring System | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Fraud_Modelling/05_GNN_Marketplace_Fraud_Ring_Detection.md` | GNN Marketplace Fraud Ring Detection | `planned-topic` |

**Independent case briefs (not measured; proposed):**

| Source case | Concrete decision question |
|---|---|
| Real Time Payment Card Fraud Detection | Which payment-card transactions should be blocked or reviewed under a customer-friction budget? |
| Insurance Claims Fraud Detection | Which claims warrant investigation without treating delayed adjudication as immediate truth? |
| Account Takeover Application Fraud | Which account/application actions should be challenged when identity signals are incomplete? |
| AML Transaction Monitoring System | Which AML alerts improve investigator value at fixed capacity without equating alerts to crime? |
| GNN Marketplace Fraud Ring Detection | Do marketplace graph features add confirmed fraud-ring discovery beyond account-level scoring? |

<a id="ds-cases-02"></a>

## Graph ML production decisions

**Teaching home:** [11_data_science_domain_specialisations/01_production_ml_case_library/02_graph_ml/INDEX.md](11_data_science_domain_specialisations/01_production_ml_case_library/02_graph_ml/INDEX.md). **Status:** planned.

**Testable outcome.** Frame, design and challenge each of the five separately listed production cases; do not combine independent decisions into one narrative.

**Focused-sheet derivations and mechanisms.** Derive graph construction time, neighbor aggregation, inductive/transductive splits, graph retrieval and propagation/ranking objectives.

**Numerical/trace exercise to author.** ILLUSTRATIVE: A node has neighbor features 2,4,9: compute an explicitly unweighted mean aggregate; remove the future edge and recompute. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Five independent case questions are registered in the exact-topic table below.

**Proposed validation.** Temporal edge snapshots, node/entity holdouts, degree-skew memory tests, non-graph baseline and investigation/relevance labels.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Future-information leakage; proxy-only success; selected or immature outcomes; unmeasured operational failure. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Case_Studies_Production_ML_Systems/Graph_ML` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Graph_ML/01_GNN_Fraud_AML_Ring_Detection.md` | GNN Fraud AML Ring Detection | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Graph_ML/02_Graph_Based_Recommendation_GraphSAGE_PinSage.md` | Graph Based Recommendation GraphSAGE PinSage | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Graph_ML/03_Knowledge_Graph_Enterprise_Search_QA.md` | Knowledge Graph Enterprise Search QA | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Graph_ML/04_Social_Network_Link_Prediction.md` | Social Network Link Prediction | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Graph_ML/05_Graph_Based_Marketing_Attribution_Customer_Journey.md` | Graph Based Marketing Attribution Customer Journey | `planned-topic` |

**Independent case briefs (not measured; proposed):**

| Source case | Concrete decision question |
|---|---|
| GNN Fraud AML Ring Detection | Which graph-enriched fraud/AML alerts add value beyond independent scoring? |
| Graph Based Recommendation GraphSAGE PinSage | Which GraphSAGE/PinSage candidates improve recommendation under exposure and cold-start constraints? |
| Knowledge Graph Enterprise Search QA | Which knowledge-graph facts support authorized enterprise QA? |
| Social Network Link Prediction | Which proposed social links are useful under temporal and identity boundaries? |
| Graph Based Marketing Attribution Customer Journey | Which graph-based marketing journey associations support attribution, and which remain noncausal? |

<a id="ds-cases-03"></a>

## Pharma key-driver analysis

**Teaching home:** [11_data_science_domain_specialisations/01_production_ml_case_library/03_kda_solutions_pharma/INDEX.md](11_data_science_domain_specialisations/01_production_ml_case_library/03_kda_solutions_pharma/INDEX.md). **Status:** mostly planned; [HCP promotional-response drivers](11_data_science_domain_specialisations/01_production_ml_case_library/03_kda_solutions_pharma/Case_Studies/01_HCP_Promotional_Response_Driver_Analysis_Launch_Brand.md) is an illustrative, unreviewed draft. The original case contents remain unavailable; other cases are planned.

**Testable outcome.** Frame, design and challenge each of the five separately listed production cases; do not combine independent decisions into one narrative.

**Focused-sheet derivations and mechanisms.** Derive response models, contribution decomposition, confounding and causal versus associative driver importance.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Illustrative sales index changes from 100 to 115 with two additive modeled contributions 8 and 7: reconcile the decomposition but do not infer causal effects. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Five independent case questions are registered in the exact-topic table below.

**Proposed validation.** Hold out time/brand/HCP as relevant, audit promotion timing, patient persistence censoring, payer policy changes and causal identification.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Future-information leakage; proxy-only success; selected or immature outcomes; unmeasured operational failure. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Case_Studies_Production_ML_Systems/KDA_Solutions_Pharma` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Case_Studies_Production_ML_Systems/KDA_Solutions_Pharma/01_HCP_Promotional_Response_Driver_Analysis_Launch_Brand.md` | HCP Promotional Response Driver Analysis Launch Brand | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/KDA_Solutions_Pharma/02_Patient_Adherence_Persistency_Driver_Analytics.md` | Patient Adherence Persistency Driver Analytics | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/KDA_Solutions_Pharma/03_Brand_Performance_Decomposition_LOE_Mature_Brand.md` | Brand Performance Decomposition LOE Mature Brand | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/KDA_Solutions_Pharma/04_Payer_Formulary_Access_Driver_Analysis.md` | Payer Formulary Access Driver Analysis | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/KDA_Solutions_Pharma/05_Omnichannel_Next_Best_Action_Driver_Attribution.md` | Omnichannel Next Best Action Driver Attribution | `planned-topic` |

**Independent case briefs (not measured; proposed):**

| Source case | Concrete decision question |
|---|---|
| HCP Promotional Response Driver Analysis Launch Brand | Which launch-brand HCP promotion drivers justify a testable action? |
| Patient Adherence Persistency Driver Analytics | Which patient adherence/persistence drivers support a feasible intervention rather than selected correlation? |
| Brand Performance Decomposition LOE Mature Brand | Which factors explain mature-brand performance around loss of exclusivity under a stated decomposition? |
| Payer Formulary Access Driver Analysis | Which payer/formulary access barriers warrant action given policy timing? |
| Omnichannel Next Best Action Driver Attribution | Which omnichannel next-best-action attribution signals support an intervention test? |

<a id="ds-cases-04"></a>

## Marketing measurement

**Teaching home:** [11_data_science_domain_specialisations/01_production_ml_case_library/04_marketing_analytics/INDEX.md](11_data_science_domain_specialisations/01_production_ml_case_library/04_marketing_analytics/INDEX.md). **Status:** planned.

**Testable outcome.** Frame, design and challenge each of the five separately listed production cases; do not combine independent decisions into one narrative.

**Focused-sheet derivations and mechanisms.** Derive adstock/saturation, attribution, discounted CLV, churn intervention value and randomized geo incrementality under explicit identification.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Hypothetical geo treatment grows 100 to 130 and control 100 to 110: calculate difference-in-differences under parallel-trend assumptions. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Five independent case questions are registered in the exact-topic table below.

**Proposed validation.** Seasonal holdouts, geo contamination/interference, baseline comparison, causal falsification and action-level profit guardrails.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Future-information leakage; proxy-only success; selected or immature outcomes; unmeasured operational failure. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Case_Studies_Production_ML_Systems/Marketing_Analytics` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Marketing_Analytics/01_Bayesian_MMM_CPG_Brand.md` | Bayesian MMM CPG Brand | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Marketing_Analytics/02_Multi_Touch_Attribution_Ecommerce.md` | Multi Touch Attribution Ecommerce | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Marketing_Analytics/03_CLV_Modeling_Value_Based_Bidding_Subscription.md` | CLV Modeling Value Based Bidding Subscription | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Marketing_Analytics/04_Churn_Prediction_Retention_Campaign_Telecom.md` | Churn Prediction Retention Campaign Telecom | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Marketing_Analytics/05_Geo_Lift_Incrementality_Testing_Paid_Media.md` | Geo Lift Incrementality Testing Paid Media | `planned-topic` |

**Independent case briefs (not measured; proposed):**

| Source case | Concrete decision question |
|---|---|
| Bayesian MMM CPG Brand | Which CPG channel spend allocation is supported by Bayesian MMM uncertainty? |
| Multi Touch Attribution Ecommerce | Which ecommerce touchpoints can receive credit without pretending attribution equals incrementality? |
| CLV Modeling Value Based Bidding Subscription | Which subscription users justify value-based bids using calibrated lifetime-value estimates? |
| Churn Prediction Retention Campaign Telecom | Which telecom retention offers generate incremental value rather than merely target likely churn? |
| Geo Lift Incrementality Testing Paid Media | Does paid media produce measurable geo-level incremental outcomes under the planned experiment? |

<a id="ds-cases-05"></a>

## NLP/LLM production systems

**Teaching home:** [11_data_science_domain_specialisations/01_production_ml_case_library/05_nlp_llm/INDEX.md](11_data_science_domain_specialisations/01_production_ml_case_library/05_nlp_llm/INDEX.md). **Status:** planned.

**Testable outcome.** Frame, design and challenge each of the five separately listed production cases; do not combine independent decisions into one narrative.

**Focused-sheet derivations and mechanisms.** Derive retrieval coverage, moderation threshold economics, embedding similarity, extraction field metrics and summarization evidence constraints.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 4 relevant documents, 3 retrieved among 5 candidates: compute precision/recall; then assess citation support separately. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Five independent case questions are registered in the exact-topic table below.

**Proposed validation.** Permission/time filters, human rubric, language/domain slices, injection cases, held-out labels and abstention behavior.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Future-information leakage; proxy-only success; selected or immature outcomes; unmeasured operational failure. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Case_Studies_Production_ML_Systems/NLP_LLM` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Case_Studies_Production_ML_Systems/NLP_LLM/01_Enterprise_RAG_Knowledge_Assistant.md` | Enterprise RAG Knowledge Assistant | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/NLP_LLM/02_LLM_Content_Moderation_At_Scale.md` | LLM Content Moderation At Scale | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/NLP_LLM/03_Semantic_Search_Ecommerce_Embeddings.md` | Semantic Search Ecommerce Embeddings | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/NLP_LLM/04_Document_Entity_Extraction_Pipeline.md` | Document Entity Extraction Pipeline | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/NLP_LLM/05_LLM_Support_Ticket_Summarization_Pipeline.md` | LLM Support Ticket Summarization Pipeline | `planned-topic` |

**Independent case briefs (not measured; proposed):**

| Source case | Concrete decision question |
|---|---|
| Enterprise RAG Knowledge Assistant | When may an enterprise knowledge assistant answer, abstain or escalate? |
| LLM Content Moderation At Scale | Which content should be blocked or reviewed under moderation capacity and harm constraints? |
| Semantic Search Ecommerce Embeddings | Which semantic search candidates add relevance beyond lexical retrieval? |
| Document Entity Extraction Pipeline | Which extracted document fields can pass validation without manual review? |
| LLM Support Ticket Summarization Pipeline | Which support-ticket summaries preserve required evidence and improve the defined handoff task? |

<a id="ds-cases-06"></a>

## Pricing and revenue decisions

**Teaching home:** [11_data_science_domain_specialisations/01_production_ml_case_library/06_pricing/INDEX.md](11_data_science_domain_specialisations/01_production_ml_case_library/06_pricing/INDEX.md). **Status:** planned.

**Testable outcome.** Frame, design and challenge each of the five separately listed production cases; do not combine independent decisions into one narrative.

**Focused-sheet derivations and mechanisms.** Derive elasticity/demand, constrained revenue versus contribution margin, capacity allocation, insurance exposure and fairness guards.

**Numerical/trace exercise to author.** ILLUSTRATIVE: At price 10 currency units and 100 units demand with variable cost 6, compare price 11 and demand 90 using revenue and contribution margin. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Five independent case questions are registered in the exact-topic table below.

**Proposed validation.** Elasticity identification, stockout censoring, temporal/geo interference, constrained optimization, policy fairness and controlled rollout.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Future-information leakage; proxy-only success; selected or immature outcomes; unmeasured operational failure. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Case_Studies_Production_ML_Systems/Pricing` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Pricing/01_Dynamic_Pricing_Ecommerce_Retail.md` | Dynamic Pricing Ecommerce Retail | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Pricing/02_Airline_Hotel_Revenue_Management.md` | Airline Hotel Revenue Management | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Pricing/03_Insurance_Pricing_Rating_Engine.md` | Insurance Pricing Rating Engine | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Pricing/04_Rideshare_Surge_Pricing.md` | Rideshare Surge Pricing | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Pricing/05_SaaS_Usage_Based_Pricing_Optimization.md` | SaaS Usage Based Pricing Optimization | `planned-topic` |

**Independent case briefs (not measured; proposed):**

| Source case | Concrete decision question |
|---|---|
| Dynamic Pricing Ecommerce Retail | Which ecommerce prices improve contribution margin under demand and fairness constraints? |
| Airline Hotel Revenue Management | Which airline/hotel capacity prices balance perishable inventory and uncertain demand? |
| Insurance Pricing Rating Engine | Which insurance rating factors support exposure-correct, governed pricing? |
| Rideshare Surge Pricing | Which rideshare surge choices balance availability, customer friction and driver response? |
| SaaS Usage Based Pricing Optimization | Which SaaS usage tiers support value and retention without hiding customer cost shocks? |

<a id="ds-cases-07"></a>

## Product experimentation and growth

**Teaching home:** [11_data_science_domain_specialisations/01_production_ml_case_library/07_product_analytics/INDEX.md](11_data_science_domain_specialisations/01_production_ml_case_library/07_product_analytics/INDEX.md). **Status:** planned.

**Testable outcome.** Frame, design and challenge each of the five separately listed production cases; do not combine independent decisions into one narrative.

**Focused-sheet derivations and mechanisms.** Derive metric trees, funnel/cohort denominators, experiment assignment, churn timing and ranking/growth feedback.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 1000 signups, 600 activations and 150 paid conversions: calculate step conversion and end-to-end conversion; define cohort windows. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Five independent case questions are registered in the exact-topic table below.

**Proposed validation.** Exposure/eligibility, user/account splits, sample-ratio/interference checks, metric guardrails, label maturity and action outcomes.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Future-information leakage; proxy-only success; selected or immature outcomes; unmeasured operational failure. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Case_Studies_Production_ML_Systems/Product_Analytics` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Product_Analytics/01_Onboarding_Funnel_Optimization_SaaS.md` | Onboarding Funnel Optimization SaaS | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Product_Analytics/02_AB_Testing_Experimentation_Platform.md` | AB Testing Experimentation Platform | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Product_Analytics/03_Churn_Prediction_Retention_Subscription_Product.md` | Churn Prediction Retention Subscription Product | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Product_Analytics/04_Marketplace_Search_Discovery_Ranking.md` | Marketplace Search Discovery Ranking | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Product_Analytics/05_Growth_Virality_Referral_Program_Analytics.md` | Growth Virality Referral Program Analytics | `planned-topic` |

**Independent case briefs (not measured; proposed):**

| Source case | Concrete decision question |
|---|---|
| Onboarding Funnel Optimization SaaS | Which onboarding step should be changed to improve a defined activation outcome? |
| AB Testing Experimentation Platform | Which experimentation platform contract supports valid assignment, exposure and stopping? |
| Churn Prediction Retention Subscription Product | Which subscription retention actions improve incremental value at fixed review/contact capacity? |
| Marketplace Search Discovery Ranking | Which marketplace discovery changes improve completed useful transactions beyond clicks? |
| Growth Virality Referral Program Analytics | Which referral/growth interventions add incremental retained users rather than duplicate attribution? |

<a id="ds-cases-08"></a>

## Recommendation pipelines

**Teaching home:** [11_data_science_domain_specialisations/01_production_ml_case_library/08_recommendation_systems/INDEX.md](11_data_science_domain_specialisations/01_production_ml_case_library/08_recommendation_systems/INDEX.md). **Status:** planned.

**Testable outcome.** Frame, design and challenge each of the five separately listed production cases; do not combine independent decisions into one narrative.

**Focused-sheet derivations and mechanisms.** Derive candidate/rank/re-rank objectives, negative sampling, sequential state, exposure bias and diversity/freshness constraints.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Relevant set size 4; top-5 list contains 3 relevant items: compute precision/recall and contrast a position-weighted ranking metric. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Five independent case questions are registered in the exact-topic table below.

**Proposed validation.** Temporal/user/content splits, cold start, replay biases, online guardrails, availability/rights and long-term retention.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Future-information leakage; proxy-only success; selected or immature outcomes; unmeasured operational failure. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Case_Studies_Production_ML_Systems/RecSys` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Case_Studies_Production_ML_Systems/RecSys/01_Video_RecSys_Candidate_Generation_Ranking.md` | Video RecSys Candidate Generation Ranking | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/RecSys/02_Ecommerce_Product_Recommendation.md` | Ecommerce Product Recommendation | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/RecSys/03_Music_Session_Based_Recommendation.md` | Music Session Based Recommendation | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/RecSys/04_News_Feed_Ranking_Freshness_Diversity.md` | News Feed Ranking Freshness Diversity | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/RecSys/05_Ads_Recommendation_CTR_Ranking.md` | Ads Recommendation CTR Ranking | `planned-topic` |

**Independent case briefs (not measured; proposed):**

| Source case | Concrete decision question |
|---|---|
| Video RecSys Candidate Generation Ranking | Which video candidate/ranking stages improve eligible viewing beyond a popularity baseline? |
| Ecommerce Product Recommendation | Which ecommerce recommendations improve useful purchase outcomes under stock constraints? |
| Music Session Based Recommendation | Which session-aware music recommendations support short-term intent without ignoring diversity? |
| News Feed Ranking Freshness Diversity | Which news ranking policy balances relevance, freshness and viewpoint/content constraints? |
| Ads Recommendation CTR Ranking | Which ads recommendation/ranking decisions optimize the stated outcome without proxy-only CTR gains? |

<a id="ds-cases-09"></a>

## Search and advertising systems

**Teaching home:** [11_data_science_domain_specialisations/01_production_ml_case_library/09_search_adtech/INDEX.md](11_data_science_domain_specialisations/01_production_ml_case_library/09_search_adtech/INDEX.md). **Status:** planned.

**Testable outcome.** Frame, design and challenge each of the five separately listed production cases; do not combine independent decisions into one narrative.

**Focused-sheet derivations and mechanisms.** Derive BM25/dense retrieval, LTR, query intent, auction allocation/payment, CTR calibration and budget pacing.

**Numerical/trace exercise to author.** ILLUSTRATIVE: At predicted click probabilities 0.02 and 0.01 with bids 3 and 5 currency/click, compare expected gross value per impression before quality/eligibility rules. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Five independent case questions are registered in the exact-topic table below.

**Proposed validation.** Relevance/auction oracles, unbiased exposure/propensity where support exists, invalid traffic, pacing under bursts and privacy boundaries.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Future-information leakage; proxy-only success; selected or immature outcomes; unmeasured operational failure. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Case_Studies_Production_ML_Systems/Search_AdTech` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Search_AdTech/01_Ecommerce_Search_Ranking_LTR_Neural_Retrieval.md` | Ecommerce Search Ranking LTR Neural Retrieval | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Search_AdTech/02_Real_Time_Bidding_Ad_Auction_System.md` | Real Time Bidding Ad Auction System | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Search_AdTech/03_CTR_Prediction_Display_Search_Ads.md` | CTR Prediction Display Search Ads | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Search_AdTech/04_Query_Autocomplete_Query_Understanding.md` | Query Autocomplete Query Understanding | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/Search_AdTech/05_Ad_Fraud_Brand_Safety_Detection.md` | Ad Fraud Brand Safety Detection | `planned-topic` |

**Independent case briefs (not measured; proposed):**

| Source case | Concrete decision question |
|---|---|
| Ecommerce Search Ranking LTR Neural Retrieval | Which ecommerce retrieval/ranking pipeline improves relevant, available product discovery? |
| Real Time Bidding Ad Auction System | Which real-time bid/auction design respects latency, eligibility and budget constraints? |
| CTR Prediction Display Search Ads | Which CTR estimates support calibrated ranking under position and selection bias? |
| Query Autocomplete Query Understanding | Which autocomplete/query-understanding design balances freshness, relevance and bounded memory? |
| Ad Fraud Brand Safety Detection | Which ad-fraud and brand-safety signals justify block/review decisions at required error costs? |

<a id="ds-cases-10"></a>

## Forecasting production decisions

**Teaching home:** [11_data_science_domain_specialisations/01_production_ml_case_library/10_time_series_forecasting/INDEX.md](11_data_science_domain_specialisations/01_production_ml_case_library/10_time_series_forecasting/INDEX.md). **Status:** mostly planned; unreviewed drafts cover [retail demand](11_data_science_domain_specialisations/01_production_ml_case_library/10_time_series_forecasting/Case_Studies/01_Retail_CPG_Demand_Forecasting_At_Scale.md), [ride-hailing demand](11_data_science_domain_specialisations/01_production_ml_case_library/10_time_series_forecasting/Case_Studies/02_Ride_Hailing_Demand_Forecasting.md), and [energy load](11_data_science_domain_specialisations/01_production_ml_case_library/10_time_series_forecasting/Case_Studies/03_Energy_Load_Forecasting.md). ETA and the remaining cases are planned; the unavailable original cases were not inspected.

**Testable outcome.** Frame, design and challenge each of the five separately listed production cases; do not combine independent decisions into one narrative.

**Focused-sheet derivations and mechanisms.** Derive temporal features, rolling-origin loss, intervals, hierarchy coherence and error-to-operational-cost mapping.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Actual demand 100,120,80 and forecast 90,130,80: compute MAE and compare a stated seasonal/naive baseline on the same horizon. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Five independent case questions are registered in the exact-topic table below.

**Proposed validation.** Rolling-origin backtest, exogenous availability, censoring, temporal/geo slices, interval calibration and decision-level consequences.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Future-information leakage; proxy-only success; selected or immature outcomes; unmeasured operational failure. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Case_Studies_Production_ML_Systems/TimeSeries_Forecasting` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Case_Studies_Production_ML_Systems/TimeSeries_Forecasting/01_Retail_CPG_Demand_Forecasting_At_Scale.md` | Retail CPG Demand Forecasting At Scale | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/TimeSeries_Forecasting/02_RideHailing_ETA_Demand_Forecasting.md` | RideHailing ETA Demand Forecasting | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/TimeSeries_Forecasting/03_Financial_Time_Series_Volatility_Forecasting.md` | Financial Time Series Volatility Forecasting | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/TimeSeries_Forecasting/04_Energy_Load_Forecasting.md` | Energy Load Forecasting | `partial-draft`: [hypothetical energy load case](11_data_science_domain_specialisations/01_production_ml_case_library/10_time_series_forecasting/Case_Studies/03_Energy_Load_Forecasting.md); source inaccessible |
| `DS_Domain/Case_Studies_Production_ML_Systems/TimeSeries_Forecasting/05_Server_Metrics_Anomaly_Detection.md` | Server Metrics Anomaly Detection | `planned-topic` |

**Independent case briefs (not measured; proposed):**

| Source case | Concrete decision question |
|---|---|
| Retail CPG Demand Forecasting At Scale | Which retail/CPG forecasts support replenishment under stockout censoring? |
| RideHailing ETA Demand Forecasting | Which ride-hailing ETA/demand predictions support dispatch under spatial/temporal shifts? |
| Financial Time Series Volatility Forecasting | Which financial volatility forecasts remain valid under regime and dependence assumptions? |
| Energy Load Forecasting | Which energy-load forecasts support capacity decisions with calibrated uncertainty? |
| Server Metrics Anomaly Detection | Which server-metric anomaly policy distinguishes failures from normal changing load? |

<a id="ds-cases-11"></a>

## US pharma commercialization decisions

**Teaching home:** [11_data_science_domain_specialisations/01_production_ml_case_library/11_us_commercialization_decision_intelligence/INDEX.md](11_data_science_domain_specialisations/01_production_ml_case_library/11_us_commercialization_decision_intelligence/INDEX.md). **Status:** planned.

**Testable outcome.** Frame, design and challenge each of the five separately listed production cases; do not combine independent decisions into one narrative.

**Focused-sheet derivations and mechanisms.** Derive action eligibility, constrained resource allocation, launch uncertainty, MMM identification and HCP segmentation stability.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 3 territories need 5,8,10 rep-days with 18 available: formulate allocation constraints and compare feasible priorities rather than inventing an optimal solution. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Five independent case questions are registered in the exact-topic table below.

**Proposed validation.** HCP/time/territory splits, policy eligibility, intervention design, market access and launch assumptions; compare manual/rule baselines.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Future-information leakage; proxy-only success; selected or immature outcomes; unmeasured operational failure. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Case_Studies_Production_ML_Systems/US_Commercialization_Decision_Intelligence` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Case_Studies_Production_ML_Systems/US_Commercialization_Decision_Intelligence/01_Next_Best_Action_Engine_Pharma_Sales_Reps.md` | Next Best Action Engine Pharma Sales Reps | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/US_Commercialization_Decision_Intelligence/02_Territory_Alignment_Sales_Resource_Allocation.md` | Territory Alignment Sales Resource Allocation | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/US_Commercialization_Decision_Intelligence/03_New_Product_Launch_Forecasting.md` | New Product Launch Forecasting | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/US_Commercialization_Decision_Intelligence/04_Omnichannel_Marketing_Measurement_Pharma_MMM.md` | Omnichannel Marketing Measurement Pharma MMM | `planned-topic` |
| `DS_Domain/Case_Studies_Production_ML_Systems/US_Commercialization_Decision_Intelligence/05_HCP_Segmentation_Targeting_Model.md` | HCP Segmentation Targeting Model | `planned-topic` |

**Independent case briefs (not measured; proposed):**

| Source case | Concrete decision question |
|---|---|
| Next Best Action Engine Pharma Sales Reps | Which next-best action may a sales representative take for an eligible HCP? |
| Territory Alignment Sales Resource Allocation | Which territory/resource allocation is feasible and valuable under travel/capacity constraints? |
| New Product Launch Forecasting | Which launch forecast is decision-useful given adoption, access and uncertainty? |
| Omnichannel Marketing Measurement Pharma MMM | Which pharma omnichannel spend changes have identifiable incremental effects? |
| HCP Segmentation Targeting Model | Which HCP segments support stable and fair targeting decisions under changing behavior? |

<a id="ds-methods-01"></a>

## Computer vision

**Teaching home:** [11_data_science_domain_specialisations/02_data_science_method_playbooks/01_computer_vision_playbook/INDEX.md](11_data_science_domain_specialisations/02_data_science_method_playbooks/01_computer_vision_playbook/INDEX.md). **Status:** planned.

**Testable outcome.** Derive, compute, select and diagnose every explicitly registered focused topic below, in prerequisite order.

**Focused-sheet derivations and mechanisms.** Derive image transforms/classical features, CNN/detection/segmentation losses, GAN/VAE/diffusion, ViT/contrastive learning, 3D/video/multimodal representations and IoU/Dice metrics; separate training and deployment objectives.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Masks of 100 and 120 pixels overlap in 80: compute intersection/union and Dice; repeat with disjoint masks. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose manual/classical/deep vision by data, modality and cost constraints; maintain separate retail, clinical, vehicle, manufacturing, security and agricultural cases.

**Proposed validation.** Dataset/entity/site splits, label/coordinate alignment, augmentation only in training, domain robustness, deployment and explainability.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Coordinate mismatch; near-duplicate leakage; invalid augmentation; unsupported real-time throughput. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Computer_Vision_Cheat_Sheets` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/00_PLANNING_REPORT.md` | PLANNING REPORT | `supporting-artifact` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/01_Image_Fundamentals_Preprocessing.md` | Image Fundamentals Preprocessing | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/02_Classical_CV_Feature_Engineering.md` | Classical CV Feature Engineering | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/03_CNN_Architecture_Fundamentals.md` | CNN Architecture Fundamentals | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/04_Object_Detection.md` | Object Detection | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/05_Image_Segmentation.md` | Image Segmentation | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/06_Classification_Transfer_Learning.md` | Classification Transfer Learning | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/07_Generative_Models_GAN_VAE.md` | Generative Models GAN VAE | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/08_Diffusion_Models.md` | Diffusion Models | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/09_Vision_Transformers_ViT.md` | Vision Transformers ViT | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/10_Multimodal_Vision_Language.md` | Multimodal Vision Language | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/11_3D_Vision_Point_Clouds.md` | D Vision Point Clouds | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/12_Video_Understanding.md` | Video Understanding | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/13_Self_Supervised_Contrastive_Learning.md` | Self Supervised Contrastive Learning | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/14_CV_Evaluation_Metrics.md` | CV Evaluation Metrics | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/15_Data_Augmentation_Training_CV.md` | Data Augmentation Training CV | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/16_CV_Retail_Fashion.md` | CV Retail Fashion | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/17_CV_Healthcare_Medical_Imaging.md` | CV Healthcare Medical Imaging | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/18_CV_Autonomous_Vehicles.md` | CV Autonomous Vehicles | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/19_CV_Manufacturing_Quality_Control.md` | CV Manufacturing Quality Control | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/20_CV_Surveillance_Security.md` | CV Surveillance Security | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/21_CV_Agriculture_Remote_Sensing.md` | CV Agriculture Remote Sensing | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/22_CV_Efficiency_Deployment.md` | CV Efficiency Deployment | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/23_Explainability_CV.md` | Explainability CV | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/24_MLOps_Production_CV.md` | MLOps Production CV | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/25_Interview_Questions_CV.md` | Interview Questions CV | `planned-topic` |
| `DS_Domain/Computer_Vision_Cheat_Sheets/26_Practical_Use_Cases_CV.md` | Practical Use Cases CV | `planned-topic` |

<a id="ds-methods-02"></a>

## Fraud modeling

**Teaching home:** [fraud modelling playbook](11_data_science_domain_specialisations/02_data_science_method_playbooks/02_fraud_modelling_playbook/INDEX.md). **Status:** partial, unreviewed drafts cover [imbalance and thresholds](11_data_science_domain_specialisations/02_data_science_method_playbooks/02_fraud_modelling_playbook/Cheat_Sheets/03_Imbalanced_Fraud_Learning_and_Thresholds.md), [evaluation](11_data_science_domain_specialisations/02_data_science_method_playbooks/02_fraud_modelling_playbook/Cheat_Sheets/01_Fraud_Evaluation_Calibration_and_Capacity.md) and [calibration](11_data_science_domain_specialisations/02_data_science_method_playbooks/02_fraud_modelling_playbook/Cheat_Sheets/02_Calibration_and_Probability_Scoring.md); remaining topics are planned.

**Testable outcome.** Derive, compute, select and diagnose every explicitly registered focused topic below, in prerequisite order.

**Focused-sheet derivations and mechanisms.** Derive class imbalance, sampling/calibration correction, thresholds/rules, anomaly/sequence/NLP/identity/graph signals and expected review utility across payment, insurance, health FWA, takeover, ecommerce and AML.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Base rate 1%, sensitivity 80%, false-positive rate 2% across 10000 events: compute true/false positives and precision. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose rules, supervised score, anomaly or graph augmentation under label maturity and investigation capacity.

**Proposed validation.** Mature temporal/entity labels, sample correction, capacity curves, subgroup reliability, adversarial shift and prevention-action feedback.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Resampling distorts posterior; review selection bias; synthetic identity leakage; immature fraud labels. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Fraud_Modelling_Cheat_Sheets` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/00_PLANNING_REPORT.md` | PLANNING REPORT | `supporting-artifact` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/01_Fraud_Taxonomy_Industry_Landscape.md` | Fraud Taxonomy Industry Landscape | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/02_Data_Foundations_Fraud.md` | Data Foundations Fraud | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/03_Feature_Engineering_Fraud.md` | Feature Engineering Fraud | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/04_Imbalanced_Learning.md` | Imbalanced Learning | `partial-draft` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/05_Supervised_Fraud_Models.md` | Supervised Fraud Models | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/06_Evaluation_Metrics_Fraud.md` | Evaluation Metrics Fraud | `partial-draft` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/07_Rule_Based_Systems_Hybrid.md` | Rule Based Systems Hybrid | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/08_Anomaly_Detection_Unsupervised.md` | Anomaly Detection Unsupervised | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/09_Network_Graph_Analytics.md` | Network Graph Analytics | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/10_Sequence_Temporal_Models.md` | Sequence Temporal Models | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/11_NLP_Text_Fraud_Signals.md` | NLP Text Fraud Signals | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/12_Identity_Verification_Synthetic_ID.md` | Identity Verification Synthetic ID | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/13_Insurance_Claims_Fraud.md` | Insurance Claims Fraud | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/14_Healthcare_Fraud_Waste_Abuse.md` | Healthcare Fraud Waste Abuse | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/15_Payment_Card_Transaction_Fraud.md` | Payment Card Transaction Fraud | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/16_Account_Takeover_Application_Fraud.md` | Account Takeover Application Fraud | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/17_Ecommerce_Marketplace_Fraud.md` | Ecommerce Marketplace Fraud | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/18_Money_Laundering_AML.md` | Money Laundering AML | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/19_Real_Time_Streaming_Fraud.md` | Real Time Streaming Fraud | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/20_Explainability_Fraud_Models.md` | Explainability Fraud Models | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/21_Model_Monitoring_Drift_Fraud.md` | Model Monitoring Drift Fraud | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/22_Adversarial_Fraud_Arms_Race.md` | Adversarial Fraud Arms Race | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/23_Graph_Neural_Networks_Fraud.md` | Graph Neural Networks Fraud | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/24_LLMs_GenAI_Fraud.md` | LLMs GenAI Fraud | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/25_50_Fraud_Interview_Questions.md` | Fraud Interview Questions | `planned-topic` |
| `DS_Domain/Fraud_Modelling_Cheat_Sheets/26_Practical_Use_Cases_Industry_Problems.md` | Practical Use Cases Industry Problems | `planned-topic` |

<a id="ds-methods-03"></a>

## Pharma KDA

**Teaching home:** [11_data_science_domain_specialisations/02_data_science_method_playbooks/03_kda_pharma_playbook/INDEX.md](11_data_science_domain_specialisations/02_data_science_method_playbooks/03_kda_pharma_playbook/INDEX.md). **Status:** planned.

**Testable outcome.** Derive, compute, select and diagnose every explicitly registered focused topic below, in prerequisite order.

**Focused-sheet derivations and mechanisms.** Derive statistical driver analysis, promotional response, MMM, patient/HCP/payer factors, brand decomposition, causal identification, launch versus mature brand, validation and governance; distinguish KDA from neighboring frameworks.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Observed response is 20/100 for contacted and 10/100 for uncontacted HCPs: compute association and state why no causal uplift is identified. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose interpretable driver analysis versus causal experiment for a specific brand or access decision.

**Proposed validation.** Time/brand/provider splits, collinearity/stability checks, confound falsification, stakeholder storytelling tied to actual evidence.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Association presented as causation; correlated importance misread; loss-of-exclusivity shift; inaccessible patient data. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/00_PLANNING_REPORT.md` | PLANNING REPORT | `supporting-artifact` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/01_KDA_Fundamentals_Pharma.md` | KDA Fundamentals Pharma | `planned-topic` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/02_Driver_Analysis_Statistical_Methods.md` | Driver Analysis Statistical Methods | `planned-topic` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/03_Promotional_Response_Modeling.md` | Promotional Response Modeling | `planned-topic` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/04_Marketing_Mix_Modeling_Pharma.md` | Marketing Mix Modeling Pharma | `planned-topic` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/05_HCP_Engagement_Driver_Analytics.md` | HCP Engagement Driver Analytics | `planned-topic` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/06_Patient_Level_Driver_Analytics.md` | Patient Level Driver Analytics | `planned-topic` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/07_Brand_Performance_Decomposition.md` | Brand Performance Decomposition | `planned-topic` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/08_Payer_Access_Driver_Analytics.md` | Payer Access Driver Analytics | `planned-topic` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/09_Therapeutic_Area_Nuances_KDA.md` | Therapeutic Area Nuances KDA | `planned-topic` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/10_KDA_for_Launch_vs_Mature_Brands.md` | KDA for Launch vs Mature Brands | `planned-topic` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/11_Causal_Inference_for_KDA.md` | Causal Inference for KDA | `planned-topic` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/12_Model_Validation_Governance_KDA.md` | Model Validation Governance KDA | `planned-topic` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/13_KDA_Technology_Stack.md` | KDA Technology Stack | `planned-topic` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/14_KDA_Outputs_Storytelling.md` | KDA Outputs Storytelling | `planned-topic` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/15_Case_Studies_KDA_Pharma.md` | Case Studies KDA Pharma | `planned-topic` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/16_KDA_vs_Other_Analytics_Frameworks.md` | KDA vs Other Analytics Frameworks | `planned-topic` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/17_Interview_Questions_KDA.md` | Interview Questions KDA | `planned-topic` |
| `DS_Domain/KDA_Solutions_Pharma_Cheat_Sheets/18_Practical_Use_Cases_KDA.md` | Practical Use Cases KDA | `planned-topic` |

<a id="ds-methods-04"></a>

## Marketing analytics

**Teaching home:** [11_data_science_domain_specialisations/02_data_science_method_playbooks/04_marketing_analytics_playbook/INDEX.md](11_data_science_domain_specialisations/02_data_science_method_playbooks/04_marketing_analytics_playbook/INDEX.md). **Status:** planned.

**Testable outcome.** Derive, compute, select and diagnose every explicitly registered focused topic below, in prerequisite order.

**Focused-sheet derivations and mechanisms.** Derive CAC/CLV, attribution, MMM, RFM, campaign/bid optimization, channel funnel and causal incrementality; include email/social/SEO/CRM/programmatic and ecommerce/SaaS/finance/health workflows.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Spend 10000 currency units acquires 200 eligible customers: compute CAC; distinguish paid clicks from new customers and include retention sensitivity. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose channel allocation/retention action from incremental value rather than attributed conversions alone.

**Proposed validation.** Cohort/geo/time definitions, controlled lift where feasible, adstock/saturation identification, delayed outcomes, privacy and cost coverage.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Double-counted attribution; selected retention sample; channel interference; revenue mistaken for margin. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Marketing_Analytics_Cheat_Sheets` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/00_PLANNING_REPORT.md` | PLANNING REPORT | `supporting-artifact` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/01_Marketing_Analytics_Fundamentals.md` | Marketing Analytics Fundamentals | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/02_Customer_Acquisition_CAC_Analytics.md` | Customer Acquisition CAC Analytics | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/03_Attribution_Models.md` | Attribution Models | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/04_Customer_Lifetime_Value.md` | Customer Lifetime Value | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/05_Marketing_Mix_Modelling.md` | Marketing Mix Modelling | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/06_Digital_Channel_Analytics.md` | Digital Channel Analytics | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/07_Email_Marketing_Analytics.md` | Email Marketing Analytics | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/08_Social_Media_Analytics.md` | Social Media Analytics | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/09_Customer_Segmentation_RFM.md` | Customer Segmentation RFM | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/10_Campaign_Optimisation.md` | Campaign Optimisation | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/11_Programmatic_Advertising_Analytics.md` | Programmatic Advertising Analytics | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/12_SEO_Content_Analytics.md` | SEO Content Analytics | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/13_CRM_Loyalty_Analytics.md` | CRM Loyalty Analytics | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/14_Marketing_Ecommerce.md` | Marketing Ecommerce | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/15_Marketing_SaaS_B2B.md` | Marketing SaaS B2B | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/16_Marketing_Financial_Services.md` | Marketing Financial Services | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/17_Marketing_Healthcare.md` | Marketing Healthcare | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/18_Experimentation_Marketing.md` | Experimentation Marketing | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/19_Marketing_Automation_Analytics.md` | Marketing Automation Analytics | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/20_Marketing_Data_Stack.md` | Marketing Data Stack | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/21_Causal_Inference_Marketing.md` | Causal Inference Marketing | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/22_Marketing_Interview_Questions.md` | Marketing Interview Questions | `planned-topic` |
| `DS_Domain/Marketing_Analytics_Cheat_Sheets/23_Practical_Use_Cases_Marketing.md` | Practical Use Cases Marketing | `planned-topic` |

<a id="ds-methods-05"></a>

## NLP and LLM methods

**Teaching home:** [11_data_science_domain_specialisations/02_data_science_method_playbooks/05_nlp_llm_playbook/INDEX.md](11_data_science_domain_specialisations/02_data_science_method_playbooks/05_nlp_llm_playbook/INDEX.md). **Status:** planned.

**Testable outcome.** Derive, compute, select and diagnose every explicitly registered focused topic below, in prerequisite order.

**Focused-sheet derivations and mechanisms.** Derive classical text/embeddings, RNN/LSTM/GRU, attention/transformers, BERT/GPT, transfer/scaling/alignment, RAG/agents, multimodal efficiency and evaluation; dated frontier/era comparisons require evidence.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Query vector (1,0), document vectors (1,1) and (0,1): compute cosine similarities then explain lexical/permission limitations. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose classical classifier, encoder retrieval, decoder generation or RAG under label, evidence and latency constraints.

**Proposed validation.** Train-only preprocessing, corpus/test contamination, retrieval permission, task/human evaluation, language slices and source/version review.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Tokenization mismatch; unsupported frontier claims; hallucinated citations; efficiency metric hides quality loss. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/NLP_LLM_Cheat_Sheets` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/00_PLANNING_REPORT.md` | PLANNING REPORT | `supporting-artifact` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/01_NLP_Fundamentals_Text_Processing.md` | NLP Fundamentals Text Processing | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/02_Classical_NLP_Pre2013.md` | Classical NLP Pre2013 | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/03_Word_Embeddings_Era.md` | Word Embeddings Era | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/04_Sequence_Models_RNN_LSTM_GRU.md` | Sequence Models RNN LSTM GRU | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/05_Attention_Mechanism.md` | Attention Mechanism | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/06_Transformer_Architecture.md` | Transformer Architecture | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/07_BERT_Encoder_Models.md` | BERT Encoder Models | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/08_GPT_Autoregressive_LLMs.md` | GPT Autoregressive LLMs | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/09_Transfer_Learning_Fine_Tuning.md` | Transfer Learning Fine Tuning | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/10_Scaling_Laws_Large_LLMs.md` | Scaling Laws Large LLMs | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/11_RLHF_Alignment.md` | RLHF Alignment | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/12_Open_Source_LLMs.md` | Open Source LLMs | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/13_RAG_Retrieval_Augmented_Generation.md` | RAG Retrieval Augmented Generation | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/14_LLM_Agents_Tool_Use.md` | LLM Agents Tool Use | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/15_Multimodal_LLMs.md` | Multimodal LLMs | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/16_LLM_Efficiency_Compression.md` | LLM Efficiency Compression | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/17_NLP_Evaluation_Benchmarks.md` | NLP Evaluation Benchmarks | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/18_NLP_Applications_Industry.md` | NLP Applications Industry | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/19_2025_2026_Frontier_LLMs.md` | 2026 Frontier LLMs | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/20_NLP_Era_Comparison_Pros_Cons.md` | NLP Era Comparison Pros Cons | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/21_NLP_LLM_Interview_Questions.md` | NLP LLM Interview Questions | `planned-topic` |
| `DS_Domain/NLP_LLM_Cheat_Sheets/22_Practical_Use_Cases_NLP_LLM.md` | Practical Use Cases NLP LLM | `planned-topic` |

<a id="ds-methods-06"></a>

## Pricing methods

**Teaching home:** [11_data_science_domain_specialisations/02_data_science_method_playbooks/06_pricing_playbook/INDEX.md](11_data_science_domain_specialisations/02_data_science_method_playbooks/06_pricing_playbook/INDEX.md). **Status:** planned.

**Testable outcome.** Derive, compute, select and diagnose every explicitly registered focused topic below, in prerequisite order.

**Focused-sheet derivations and mechanisms.** Derive economic/value/cost/competitive price models, elasticity, optimization, dynamic pricing, yield, behavioral factors, game theory, experiments, domain pricing and fairness.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Price rises 10% while quantity falls 20% under a stated local approximation: estimate elasticity and explain identification and nonlinear limits. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose static policy, constrained demand model or experiment when feedback and competitor behavior can reverse naive optimization.

**Proposed validation.** Price endogeneity, stockout censoring, margin/capacity, customer fairness and temporal interference; include negative/zero baselines appropriately.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Confounded elasticity; revenue-only objective; unbounded competitor loop; discriminatory proxy. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Pricing_Cheat_Sheets` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Pricing_Cheat_Sheets/00_PLANNING_REPORT.md` | PLANNING REPORT | `supporting-artifact` |
| `DS_Domain/Pricing_Cheat_Sheets/01_Pricing_Fundamentals_Economics.md` | Pricing Fundamentals Economics | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/02_Price_Elasticity_Demand_Models.md` | Price Elasticity Demand Models | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/03_Cost_Value_Competitive_Pricing.md` | Cost Value Competitive Pricing | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/04_Price_Optimization_ML.md` | Price Optimization ML | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/05_Dynamic_Pricing_Algorithms.md` | Dynamic Pricing Algorithms | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/06_Revenue_Management_Yield_Optimization.md` | Revenue Management Yield Optimization | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/07_Psychological_Behavioral_Pricing.md` | Psychological Behavioral Pricing | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/08_Pricing_Experiments_AB_Testing.md` | Pricing Experiments AB Testing | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/09_Game_Theory_Competitive_Pricing.md` | Game Theory Competitive Pricing | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/10_Pricing_Insurance_Actuarial.md` | Pricing Insurance Actuarial | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/11_Pricing_Ecommerce_Retail.md` | Pricing Ecommerce Retail | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/12_Pricing_SaaS_Subscription.md` | Pricing SaaS Subscription | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/13_Pricing_Marketplaces_Platforms.md` | Pricing Marketplaces Platforms | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/14_Pricing_Financial_Services.md` | Pricing Financial Services | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/15_Pricing_Healthcare_Pharma.md` | Pricing Healthcare Pharma | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/16_Pricing_Auctions_Real_Time_Bidding.md` | Pricing Auctions Real Time Bidding | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/17_Pricing_Fairness_Regulation.md` | Pricing Fairness Regulation | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/18_Real_Time_Pricing_Systems.md` | Real Time Pricing Systems | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/19_Price_Monitoring_Competitive_Intel.md` | Price Monitoring Competitive Intel | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/20_Pricing_Interview_Questions.md` | Pricing Interview Questions | `planned-topic` |
| `DS_Domain/Pricing_Cheat_Sheets/21_Practical_Use_Cases_Pricing.md` | Practical Use Cases Pricing | `planned-topic` |

<a id="ds-methods-07"></a>

## Product analytics

**Teaching home:** [11_data_science_domain_specialisations/02_data_science_method_playbooks/07_product_analytics_playbook/INDEX.md](11_data_science_domain_specialisations/02_data_science_method_playbooks/07_product_analytics_playbook/INDEX.md). **Status:** planned.

**Testable outcome.** Derive, compute, select and diagnose every explicitly registered focused topic below, in prerequisite order.

**Focused-sheet derivations and mechanisms.** Derive metric framework, catalog quality, behavior/funnels/cohorts/retention, experiments, feature adoption, search/personalization, pricing/inventory, mobile and growth across ecommerce/SaaS/marketplaces.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Cohort A has 80 of 100 users retained at day 7 and cohort B 180 of 300: calculate cohort and pooled retention with consistent eligibility. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose product intervention from a valid metric tree and user workflow rather than a favorable dashboard.

**Proposed validation.** Instrumentation reconciliation, denominator stability, account/device identity, eligibility/exposure, seasonality and launch guardrails.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Denominator change; survivorship bias; event duplication; click metric conflicts with value. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Product_Analytics_Cheat_Sheets` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/00_PLANNING_REPORT.md` | PLANNING REPORT | `supporting-artifact` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/01_Product_Analytics_Fundamentals.md` | Product Analytics Fundamentals | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/02_Product_Metrics_Framework.md` | Product Metrics Framework | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/03_Catalogue_Management_Data.md` | Catalogue Management Data | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/04_User_Behaviour_Analytics.md` | User Behaviour Analytics | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/05_Funnel_Conversion_Analysis.md` | Funnel Conversion Analysis | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/06_Retention_Churn_Analysis.md` | Retention Churn Analysis | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/07_Cohort_Segmentation_Analysis.md` | Cohort Segmentation Analysis | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/08_AB_Testing_Product_Experiments.md` | AB Testing Product Experiments | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/09_Feature_Adoption_Usage_Analytics.md` | Feature Adoption Usage Analytics | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/10_Search_Discovery_Product.md` | Search Discovery Product | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/11_Personalisation_Analytics.md` | Personalisation Analytics | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/12_Pricing_Catalogue_Analytics.md` | Pricing Catalogue Analytics | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/13_Inventory_Demand_Forecasting.md` | Inventory Demand Forecasting | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/14_Mobile_App_Analytics.md` | Mobile App Analytics | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/15_Product_Analytics_Ecommerce.md` | Product Analytics Ecommerce | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/16_Product_Analytics_SaaS.md` | Product Analytics SaaS | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/17_Product_Analytics_Marketplace.md` | Product Analytics Marketplace | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/18_Growth_Virality_Analytics.md` | Growth Virality Analytics | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/19_Causal_Inference_Product.md` | Causal Inference Product | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/20_Product_Analytics_Tools_Stack.md` | Product Analytics Tools Stack | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/21_Product_Analytics_Interview_Questions.md` | Product Analytics Interview Questions | `planned-topic` |
| `DS_Domain/Product_Analytics_Cheat_Sheets/22_Practical_Use_Cases_Product_Analytics.md` | Practical Use Cases Product Analytics | `planned-topic` |

<a id="ds-methods-08"></a>

## Recommender methods

**Teaching home:** [11_data_science_domain_specialisations/02_data_science_method_playbooks/08_recsys_playbook/INDEX.md](11_data_science_domain_specialisations/02_data_science_method_playbooks/08_recsys_playbook/INDEX.md). **Status:** mostly planned; [candidate generation and eligibility](11_data_science_domain_specialisations/02_data_science_method_playbooks/08_recsys_playbook/Cheat_Sheets/01_Candidate_Generation_Eligibility_and_Ranking.md) is a partial unreviewed draft. The source files themselves were unavailable.

**Testable outcome.** Derive, compute, select and diagnose every explicitly registered focused topic below, in prerequisite order.

**Focused-sheet derivations and mechanisms.** Derive collaborative/content/hybrid models, matrix factorization, deep/sequential/graph recommendation, bandit exploration, feature engineering and evaluation; include retail/media/news/finance/health applications and fairness/scaling.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Observed reward 30/100 for arm A versus 20/50 for B: calculate rates but distinguish uncertainty and nonrandom selection before allocating traffic. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose popularity/content/collaborative/sequential/bandit under cold-start, exposure, diversity and freshness constraints.

**Proposed validation.** Temporal/user splits, negative sampling bias, off-policy support, exploration safety and online usefulness/retention.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Exposure mistaken for preference; feedback loops; cold-start omission; unsupported off-policy extrapolation. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/RecSys_Cheat_Sheets` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/RecSys_Cheat_Sheets/00_PLANNING_REPORT.md` | PLANNING REPORT | `supporting-artifact` |
| `DS_Domain/RecSys_Cheat_Sheets/01_RecSys_Fundamentals_Taxonomy.md` | RecSys Fundamentals Taxonomy | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/02_Collaborative_Filtering_Matrix_Factorization.md` | Collaborative Filtering Matrix Factorization | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/03_Content_Based_Filtering.md` | Content Based Filtering | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/04_Hybrid_RecSys_Ensemble.md` | Hybrid RecSys Ensemble | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/05_Deep_Learning_RecSys.md` | Deep Learning RecSys | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/06_Sequential_Session_Based_RecSys.md` | Sequential Session Based RecSys | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/07_Knowledge_Graph_RecSys.md` | Knowledge Graph RecSys | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/08_Contextual_Bandits_Exploration.md` | Contextual Bandits Exploration | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/09_RecSys_Evaluation_Metrics.md` | RecSys Evaluation Metrics | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/10_Feature_Engineering_RecSys.md` | Feature Engineering RecSys | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/11_RecSys_Ecommerce_Retail.md` | RecSys Ecommerce Retail | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/12_RecSys_Streaming_Media.md` | RecSys Streaming Media | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/13_RecSys_News_Content.md` | RecSys News Content | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/14_RecSys_Financial_Services.md` | RecSys Financial Services | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/15_RecSys_Healthcare.md` | RecSys Healthcare | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/16_RecSys_Travel_Hospitality.md` | RecSys Travel Hospitality | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/17_Scalability_Real_Time_RecSys.md` | Scalability Real Time RecSys | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/18_Fairness_Diversity_Bias_RecSys.md` | Fairness Diversity Bias RecSys | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/19_LLMs_GenAI_RecSys.md` | LLMs GenAI RecSys | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/20_AB_Testing_Experimentation_RecSys.md` | AB Testing Experimentation RecSys | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/21_RecSys_Interview_Questions.md` | RecSys Interview Questions | `planned-topic` |
| `DS_Domain/RecSys_Cheat_Sheets/22_Practical_Use_Cases_RecSys.md` | Practical Use Cases RecSys | `planned-topic` |

<a id="ds-methods-09"></a>

## Search and AdTech methods

**Teaching home:** [11_data_science_domain_specialisations/02_data_science_method_playbooks/09_search_adtech_playbook/INDEX.md](11_data_science_domain_specialisations/02_data_science_method_playbooks/09_search_adtech_playbook/INDEX.md). **Status:** mostly planned; [BM25 and classical retrieval](11_data_science_domain_specialisations/02_data_science_method_playbooks/09_search_adtech_playbook/Cheat_Sheets/02_BM25_and_Classical_Retrieval.md) is an unreviewed draft; original source contents are unavailable.

**Testable outcome.** Derive, compute, select and diagnose every explicitly registered focused topic below, in prerequisite order.

**Focused-sheet derivations and mechanisms.** Derive IR/BM25, LTR/dense retrieval, query intent/indexing, auctions/RTB, targeting, quality, pacing, attribution/privacy and generative search; split search relevance from advertising economics.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Two terms have document frequencies 10 and 100 in 1000 documents: calculate a specified IDF variant, then state smoothing and corpus assumptions. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose lexical, neural or hybrid search and an eligible auction/ranking policy by workload and business constraints.

**Proposed validation.** Comparable relevance sets, position/selection bias, auction correctness, privacy eligibility, cache/index freshness and budget conservation.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Zero-frequency formula misuse; bid/payment confusion; overspent pacing; unauthorized retrieval. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Search_AdTech_Cheat_Sheets` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/00_PLANNING_REPORT.md` | PLANNING REPORT | `supporting-artifact` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/01_Search_Fundamentals_IR.md` | Search Fundamentals IR | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/02_BM25_TF_IDF_Classical_Retrieval.md` | BM25 TF IDF Classical Retrieval | `partial-draft`: [BM25 and classical retrieval](11_data_science_domain_specialisations/02_data_science_method_playbooks/09_search_adtech_playbook/Cheat_Sheets/02_BM25_and_Classical_Retrieval.md); source inaccessible |
| `DS_Domain/Search_AdTech_Cheat_Sheets/03_Learning_to_Rank.md` | Learning to Rank | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/04_Neural_Search_Dense_Retrieval.md` | Neural Search Dense Retrieval | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/05_Query_Understanding_NLP.md` | Query Understanding NLP | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/06_Search_Indexing_Infrastructure.md` | Search Indexing Infrastructure | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/07_AdTech_Fundamentals.md` | AdTech Fundamentals | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/08_Auction_Theory_Mechanisms.md` | Auction Theory Mechanisms | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/09_Real_Time_Bidding_Programmatic.md` | Real Time Bidding Programmatic | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/10_Ad_Targeting_Audience.md` | Ad Targeting Audience | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/11_Ad_Quality_Relevance_Ranking.md` | Ad Quality Relevance Ranking | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/12_Bid_Optimisation_Budget_Pacing.md` | Bid Optimisation Budget Pacing | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/13_Ad_Measurement_Attribution.md` | Ad Measurement Attribution | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/14_Ad_Fraud_Brand_Safety.md` | Ad Fraud Brand Safety | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/15_Search_Ecommerce.md` | Search Ecommerce | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/16_Search_Enterprise_Knowledge.md` | Search Enterprise Knowledge | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/17_AdTech_Social_Media.md` | AdTech Social Media | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/18_AdTech_CTV_Video.md` | AdTech CTV Video | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/19_Privacy_Cookieless_AdTech.md` | Privacy Cookieless AdTech | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/20_Search_Personalisation.md` | Search Personalisation | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/21_Search_AdTech_ML_Systems.md` | Search AdTech ML Systems | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/22_Generative_AI_Search_AdTech.md` | Generative AI Search AdTech | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/23_Search_AdTech_Interview_Questions.md` | Search AdTech Interview Questions | `planned-topic` |
| `DS_Domain/Search_AdTech_Cheat_Sheets/24_Practical_Use_Cases_Search_AdTech.md` | Practical Use Cases Search AdTech | `planned-topic` |

<a id="ds-methods-10"></a>

## Time series and demand planning

**Teaching home:** [11_data_science_domain_specialisations/02_data_science_method_playbooks/10_time_series_playbook/INDEX.md](11_data_science_domain_specialisations/02_data_science_method_playbooks/10_time_series_playbook/INDEX.md). **Status:** planned.

**Testable outcome.** Derive, compute, select and diagnose every explicitly registered focused topic below, in prerequisite order.

**Focused-sheet derivations and mechanisms.** Derive ARIMA/ETS/state space/decomposition, temporal features, ML/deep/foundation models, probabilistic/hierarchical forecasts, inventory and domain consequences; include energy/finance and anomaly detection.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Two child forecasts 40 and 70 sum to 110 while parent predicts 100: calculate incoherence and propose a stated reconciliation objective rather than arbitrary rescaling. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose naive/statistical/ML/hierarchical probabilistic forecast for the actual horizon, data cadence and operational decision.

**Proposed validation.** Rolling-origin splits, exogenous availability, intermittent/censored demand, interval coverage, hierarchy coherence and decision sensitivity.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Future leakage; unavailable covariate; incoherent hierarchy; inventory censoring misread as low demand. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/00_PLANNING_REPORT.md` | PLANNING REPORT | `supporting-artifact` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/01_Time_Series_Fundamentals.md` | Time Series Fundamentals | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/02_Classical_Statistical_Methods_ARIMA.md` | Classical Statistical Methods ARIMA | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/03_Exponential_Smoothing_ETS_State_Space.md` | Exponential Smoothing ETS State Space | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/04_Time_Series_Decomposition_Seasonality.md` | Time Series Decomposition Seasonality | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/05_Feature_Engineering_Time_Series.md` | Feature Engineering Time Series | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/06_ML_For_Time_Series.md` | ML For Time Series | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/07_Deep_Learning_Time_Series.md` | Deep Learning Time Series | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/08_Transformer_Foundation_Models_TS.md` | Transformer Foundation Models TS | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/09_Probabilistic_Forecasting.md` | Probabilistic Forecasting | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/10_Hierarchical_Reconciliation_Forecasting.md` | Hierarchical Reconciliation Forecasting | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/11_Demand_Planning_Fundamentals.md` | Demand Planning Fundamentals | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/12_Supply_Chain_Inventory_Optimization.md` | Supply Chain Inventory Optimization | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/13_Retail_CPG_Demand_Forecasting.md` | Retail CPG Demand Forecasting | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/14_Energy_Utilities_Forecasting.md` | Energy Utilities Forecasting | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/15_Financial_Time_Series.md` | Financial Time Series | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/16_Healthcare_Epidemiological_Forecasting.md` | Healthcare Epidemiological Forecasting | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/17_Multivariate_Time_Series.md` | Multivariate Time Series | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/18_Anomaly_Detection_Time_Series.md` | Anomaly Detection Time Series | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/19_Causal_Inference_Time_Series.md` | Causal Inference Time Series | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/20_Model_Selection_Ensembles_TS.md` | Model Selection Ensembles TS | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/21_Evaluation_Metrics_Backtesting.md` | Evaluation Metrics Backtesting | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/22_Explainability_Time_Series_Models.md` | Explainability Time Series Models | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/23_Production_MLOps_Time_Series.md` | Production MLOps Time Series | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/24_Interview_Questions_Time_Series.md` | Interview Questions Time Series | `planned-topic` |
| `DS_Domain/TimeSeries_Forecasting_Cheat_Sheets/25_Practical_Use_Cases_TS_Demand.md` | Practical Use Cases TS Demand | `planned-topic` |

<a id="ds-ml-design"></a>

## Production ML disciplines, architectures, features, monitoring, governance, training/serving, PoC roadmap and production traps

**Teaching home:** [11_data_science_domain_specialisations/03_practical_ml_design_and_interviews/01_ml_system_design_production/INDEX.md](11_data_science_domain_specialisations/03_practical_ml_design_and_interviews/01_ml_system_design_production/INDEX.md). **Status:** planned.

**Testable outcome.** Design and defend a full production ML path across the source 12 disciplines and every listed loss/optimization/architecture trap.

**Focused-sheet derivations and mechanisms.** Derive loss/objective and feature-time consistency, drift/feedback metrics, training/serving memory and cost; source 50 Common ML Issues plus additional 15 issues are distinct coverage targets, not completed incidents.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 100 predictions/s at 20 ms mean service time suggests mean concurrency 2 under stable Little-law boundaries; contrast that mean with unmeasured burst/tail requirements. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose feature/serving/deployment design for one bounded recommendation/search/LLM or domain ML decision.

**Proposed validation.** Point-in-time feature joins, naive baseline, calibration/thresholds, restored checkpoints, model/data schema compatibility, real useful-outcome cost and failure injection.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Wrong optimization objective; training-serving skew; stale features; unowned rollback. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/00_INDEX.md` | INDEX | `supporting-artifact` |
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/01_Production_ML_12_Disciplines.md` | Production ML 12 Disciplines | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/02_Architecture_Patterns.md` | Architecture Patterns | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/03_Feature_Store_Design.md` | Feature Store Design | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/04_Model_Monitoring_and_Drift.md` | Model Monitoring and Drift | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/05_MLOps_Governance_Compliance.md` | MLOps Governance Compliance | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/06_Training_Infrastructure.md` | Training Infrastructure | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/07_Serving_Scaling_Optimization.md` | Serving Scaling Optimization | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/08_PoC_to_Production_Roadmap.md` | PoC to Production Roadmap | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/09_Framework_Tool_Selection.md` | Framework Tool Selection | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/10_Domain_Specific_ML_Patterns.md` | Domain Specific ML Patterns | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/11_LLM_GenAI_Systems.md` | LLM GenAI Systems | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/12_Recommendations_Search_Systems.md` | Recommendations Search Systems | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/13_50_Common_ML_Issues.md` | Common ML Issues | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/14_Loss_Functions_Optimization_Traps.md` | Loss Functions Optimization Traps | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/15_System_Architecture_Production_Traps.md` | System Architecture Production Traps | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Design_Cheat_Sheets/16_Additional_15_Issues_from_PDF.md` | Additional 15 Issues from PDF | `planned-topic` |

<a id="ds-product-sense"></a>

## ML product-sense framing, metrics, recommendations, search, fraud, NLP, forecasting, agents, CV, causal experiments and PM interviews

**Teaching home:** [11_data_science_domain_specialisations/03_practical_ml_design_and_interviews/02_ml_product_sense/INDEX.md](11_data_science_domain_specialisations/03_practical_ml_design_and_interviews/02_ml_product_sense/INDEX.md). **Status:** planned.

**Testable outcome.** Turn a stakeholder problem into a scoped action, proxy/guardrail metric tree and a falsifiable evaluation.

**Focused-sheet derivations and mechanisms.** Derive metric denominators and decision value; distinguish prediction accuracy, causal intervention and product adoption; contextualize each of the 12 visible guide chapters independently.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Product has 1000 visitors, 200 search users and 40 purchases: compute overall and conditional conversion and state which denominator matches the decision. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose whether ML is justified over workflow/rules for each product-sense family.

**Proposed validation.** Challenge scope, decision owner, unavailable data, feedback loops, success proxy, experiment feasibility and a simpler baseline; numerical interview answers require checks.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Solution before framing; proxy gaming; unsupported causality; no-action baseline omitted. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Practical_ML_Design/ML_Product_Sense_Guide` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Practical_ML_Design/ML_Product_Sense_Guide/00_product_sense_mental_model.md` | product sense mental model | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Product_Sense_Guide/01_problem_framing_and_scoping.md` | problem framing and scoping | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Product_Sense_Guide/02_metrics_design_and_measurement.md` | metrics design and measurement | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Product_Sense_Guide/03_recommendations_personalization_product_sense.md` | recommendations personalization product sense | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Product_Sense_Guide/04_search_and_retrieval_product_sense.md` | search and retrieval product sense | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Product_Sense_Guide/05_fraud_trust_safety_product_sense.md` | fraud trust safety product sense | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Product_Sense_Guide/06_nlp_text_content_product_sense.md` | nlp text content product sense | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Product_Sense_Guide/07_prediction_forecasting_product_sense.md` | prediction forecasting product sense | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Product_Sense_Guide/08_llm_agents_product_sense.md` | llm agents product sense | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Product_Sense_Guide/09_computer_vision_product_sense.md` | computer vision product sense | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Product_Sense_Guide/10_causal_inference_experiments_product_sense.md` | causal inference experiments product sense | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Product_Sense_Guide/11_ml_pm_interview_playbook.md` | ml pm interview playbook | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ML_Product_Sense_Guide/all_content.txt` | all content | `supporting-artifact` |
| `DS_Domain/Practical_ML_Design/ML_Product_Sense_Guide/README.md` | README | `supporting-artifact` |
| `DS_Domain/Practical_ML_Design/ML_Product_Sense_Guide/replicate_structure.py` | replicate structure | `supporting-artifact` |

<a id="ds-practical-sources"></a>

## ML design cases, hidden architecture/production guides, 2022+ cases, PDFs and source utilities

**Teaching home:** [11_data_science_domain_specialisations/03_practical_ml_design_and_interviews/01_ml_system_design_production/INDEX.md](11_data_science_domain_specialisations/03_practical_ml_design_and_interviews/01_ml_system_design_production/INDEX.md). **Status:** planned.

**Testable outcome.** Distinguish a visible named design case/source from inaccessible case bodies and architecture collections.

**Focused-sheet derivations and mechanisms.** Register source-to-decision provenance and reproduction boundaries; no PDF filename establishes the quality of an implementation.

**Numerical/trace exercise to author.** ILLUSTRATIVE: A hypothetical 3-stage system has 2 independently usable fallback modes: trace valid/invalid releases and count only end-to-end completed tasks. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose which accessible production design source can support a bounded case, and which remains open.

**Proposed validation.** Obtain originals, source revision, data availability and artifact-backed outcomes; preserve hidden architecture/use-case libraries as unresolved.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Unavailable source promoted to evidence; collation mistaken for independent lesson; empty guide link; duplicated case counted twice. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain/Practical_ML_Design` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Practical_ML_Design/ML_LLM_Use_Cases_2022+` | OPEN — ML_LLM_Use_Cases_2022+/  [contents not visible] | `unresolved-source` |
| `DS_Domain/Practical_ML_Design/ML_System_Architecture_Designs` | OPEN — ML_System_Architecture_Designs/  [contents not visible] | `unresolved-source` |
| `DS_Domain/Practical_ML_Design/ML_System_Design_Production_Guide` | OPEN — ML_System_Design_Production_Guide/  [contents not visible] | `unresolved-source` |
| `DS_Domain/Practical_ML_Design/ML System Design Case Study.md` | ML System Design Case Study | `planned-topic` |
| `DS_Domain/Practical_ML_Design/ml_sys_des.pdf` | ml sys des | `supporting-artifact` |
| `DS_Domain/Practical_ML_Design/mlengg_note.pdf` | mlengg note | `supporting-artifact` |

<a id="graph-source"></a>

## Graph theory, GNNs, knowledge graphs/ontologies and fraud detection

**Teaching home:** [03_machine_learning_optimisation_and_intelligent_retrieval/15_graph_ml_knowledge_graphs_and_network_analytics/INDEX.md](03_machine_learning_optimisation_and_intelligent_retrieval/15_graph_ml_knowledge_graphs_and_network_analytics/INDEX.md). **Status:** planned.

**Testable outcome.** Construct a graph with explicit semantics, apply a justified network/GNN method and diagnose leakage.

**Focused-sheet derivations and mechanisms.** Derive degree/connectivity/traversal bounds, neighborhood aggregation and knowledge-graph relation semantics; separate graph structure from proven fraud.

**Numerical/trace exercise to author.** ILLUSTRATIVE: Adjacency lists A:[B,C], B:[D], C:[D], D:[]: trace BFS, unique node/edge visits and a one-hop mean feature aggregate with stated features. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose graph traversal, rules, learned GNN or knowledge-graph retrieval for a bounded fraud/search decision.

**Proposed validation.** Time-correct edges, entity resolution, holdout nodes, relation consistency, degree-skew memory and a non-graph baseline.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Future edges; identity collision; degree explosion; connectedness treated as guilt. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `Graph_ML_2025` | Source scope/container; not a completed lesson | `source-container` |
| `Graph_ML_2025/01_Graph_Theory_Foundations.md` | Graph Theory Foundations | `planned-topic` |
| `Graph_ML_2025/02_Graph_Neural_Networks.md` | Graph Neural Networks | `planned-topic` |
| `Graph_ML_2025/03_Knowledge_Graphs_Ontologies.md` | Knowledge Graphs Ontologies | `planned-topic` |
| `Graph_ML_2025/04_Fraud_Detection.md` | Fraud Detection | `planned-topic` |

<a id="ds-source"></a>

## DS source roots, production-case catalog and hidden US commercialization cheat sheets

**Teaching home:** [11_data_science_domain_specialisations/01_production_ml_case_library/11_us_commercialization_decision_intelligence/INDEX.md](11_data_science_domain_specialisations/01_production_ml_case_library/11_us_commercialization_decision_intelligence/INDEX.md). **Status:** planned.

**Testable outcome.** Preserve 55 explicit production case titles and all method families without inferring collapsed commercialization sheet contents.

**Focused-sheet derivations and mechanisms.** Crosswalk workflow/decision/method/evidence separately; maintain source IDs and distinguish catalog entries from authored content.

**Numerical/trace exercise to author.** ILLUSTRATIVE: 11 visible case families with 5 cases each yield 55 planned source cases; this count excludes backlinks and unknown cases. Show inputs → rule/derivation → substitution/trace → result with units → independent check → decision. Vary one material assumption and explain the failure boundary.

**Bounded case decision.** Choose a source-backed or explicitly illustrative commercial decision case, not a generic company story.

**Proposed validation.** Obtain hidden US commercialization sheet contents before declaring them covered at leaf level; use existing explicit 5 commercialization cases for visible scope only.

**Selection and diagnostic pressure tests.** Compare the simplest credible baseline with the proposed method under the same workload and objective. Distinct failure themes: Hidden family counted as authored; duplicate catalog count; generic hybrid case type; invented case outcome. Each final failure entry needs competing causes, a discriminating expected signal, bounded mitigation, prevention and verification.

**Exact source entries and visible topic commitments:**

| Exact source entry | Focused topic / source role | Disposition |
|---|---|---|
| `DS_Domain` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Case_Studies_Production_ML_Systems` | Source scope/container; not a completed lesson | `source-container` |
| `DS_Domain/Case_Studies_Production_ML_Systems/00_INDEX.md` | INDEX | `supporting-artifact` |
| `DS_Domain/US_Commercialization_Decision_Intelligence_Cheat_Sheets` | OPEN — US_Commercialization_Decision_Intelligence_Cheat_Sheets/  [contents not visible] | `unresolved-source` |

## Publication boundary

This is complete **visible-source topic planning**, not completed teaching content or hidden-source coverage. Preserve all existing course topics. Before publishing individual artifacts, obtain sources, author the required structures and topic-specific mathematics, execute numerical/trace checks and relevant validation, answer interview questions, finish actual issue counts, verify links/prerequisites, and update local catalog status and review records.
