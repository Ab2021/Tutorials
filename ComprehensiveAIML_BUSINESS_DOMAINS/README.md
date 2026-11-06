# Comprehensive AI, ML and Business Domains

> A 2026 learning and reference architecture spanning computing foundations, data science, machine learning, AI engineering, LLM agents, edge robotics, business domains, governance, and capstone practice.

## How this library is organised

This is not a vendor-centred catalog. It is ordered by the decisions and system layers that make modern AI useful:

~~~text
Foundations -> Data and decision science -> ML and optimisation -> Foundation models
    -> AI platforms -> Agents and knowledge systems -> Edge AI and robotics
    -> Domain systems -> Trust and leadership -> Architecture practice and capstones
~~~

Every part and module contains an **INDEX.md**, plus indexed collections for teaching content:

- **Case_Studies/** — bounded decisions, technical walkthroughs, evaluation, failure analysis, and evidence.
- **Cheat_Sheets/** — focused theory, derivations, checked numerical examples, selection rules, diagnostics, and final reference cards.

At the root, [Case_Studies](./Case_Studies/INDEX.md) and [Cheat_Sheets](./Cheat_Sheets/INDEX.md) hold cross-domain plans. Most of the course remains planned; [authored drafts are listed in the course index](INDEX.md#authored-drafts) and are not independently reviewed. All collections follow the [master-template authoring and numerical standards](00_AUTHORING_AND_NUMERICAL_STANDARDS.md); a short card or topic title alone is not complete content.

The [source-topic crosswalk](00_SOURCE_TOPIC_CROSSWALK.md) assigns every visible entry from the supplied tree to an existing teaching home and a focused brief. Its 144 teaching groups extend the original local topic lists. Hidden contents and clipped filenames remain explicitly unresolved.

## Programme map

| Part | Focus | Why it sits here | Modules |
|---:|---|---|---:|
| 01 | [Foundations, Computing and Problem Solving](./01_foundations_computing_and_problem_solving/INDEX.md) | A prerequisite spine: programming, algorithms, systems thinking, mathematics, and software practice. | 5 |
| 02 | [Data, Analytics and Decision Science](./02_data_analytics_and_decision_science/INDEX.md) | Turn raw operational data into trustworthy metrics, experiments, forecasts, and decisions. | 5 |
| 03 | [Machine Learning, Optimisation and Intelligent Retrieval](./03_machine_learning_optimisation_and_intelligent_retrieval/INDEX.md) | Core predictive, prescriptive, graph, search, and recommendation methods for production decisions. | 6 |
| 04 | [Deep Learning, Generative AI and Multimodal Systems](./04_deep_learning_generative_ai_and_multimodal_systems/INDEX.md) | Representation learning, foundation models, post-training, and multimodal product systems. | 6 |
| 05 | [AI Platforms, Data Systems and Production Engineering](./05_ai_platforms_data_systems_and_production_engineering/INDEX.md) | The infrastructure and operating model needed to train, deploy, observe, and improve AI safely. | 7 |
| 06 | [LLM Agents, Knowledge Systems and AI-Augmented Development](./06_llm_agents_knowledge_systems_and_ai_augmented_development/INDEX.md) | Reliable agent workflows, tool integrations, context engineering, and software-delivery automation. | 5 |
| 07 | [Edge AI, Robotics and Cyber-Physical Systems](./07_edge_ai_robotics_and_cyber_physical_systems/INDEX.md) | Low-latency, sensor-rich intelligence that perceives, reasons, acts, and operates under real-world constraints. | 7 |
| 08 | [Business Domains and Decision Intelligence](./08_business_domains_and_decision_intelligence/INDEX.md) | Translate methods into high-value, regulated, and operationally realistic domain systems. | 7 |
| 09 | [Trust, Security, Governance and Product Leadership](./09_trust_security_governance_and_product_leadership/INDEX.md) | Design AI that is secure, evaluated, compliant, adopted, and economically sound. | 5 |
| 10 | [Architecture Practice and Portfolio Capstones](./10_architecture_practice_and_portfolio_capstones/INDEX.md) | Synthesize the curriculum through design interviews, incidents, and end-to-end portfolio work. | 3 |
| 11 | [Data Science Domain Specialisations](./11_data_science_domain_specialisations/INDEX.md) | Detailed production cases, method playbooks, and industry workflows complement the prerequisite spine. | 15 families |

## Recommended paths

| Goal | Minimum path |
|---|---|
| Data scientist | 01-02 -> 05-12 -> 17-21 -> 26-28 -> a domain module -> 56 |
| ML / AI engineer | 02-04 -> 06, 09, 11 -> 19-25 -> 26-29 -> 54-56 |
| LLM / agent engineer | 02-04 -> 18-21 -> 25, 28-34 -> 49-50 -> 54-56 |
| Edge robotics engineer | 02-05 -> 17, 19, 22-25 -> 35-41 -> 45-46 -> 55-56 |
| AI product / leader | 01, 05, 07-08 -> 19-20, 28 -> 42-48 -> 49-53 -> 56 |

## Why these areas are prioritised now

The 2026 design foregrounds AI and big data, networks/cybersecurity, and technological literacy; the World Economic Forum identifies them as the fastest-growing skills and places AI, robotics, advanced energy, and sensor networks at the centre of workforce change. Foundation-model adoption, evaluation, security, data infrastructure, and production operations therefore sit alongside theory. [WEF Future of Jobs 2025](https://www.weforum.org/publications/the-future-of-jobs-report-2025)  
The curriculum also treats physical AI as a first-class path rather than an elective: current robotics platforms combine multimodal models, synthetic data, simulation, and edge inference for manufacturing, logistics, healthcare, agriculture, and retail. [NVIDIA Jetson Thor / physical AI overview](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/)  
Finally, it emphasises value and evaluation rather than model novelty: generative-AI private investment reached USD 33.9B in 2024, while the 2026 AI Index reports broad organisational adoption and growing stress on trustworthy evaluation. [Stanford AI Index 2025](https://hai.stanford.edu/ai-index/2025-ai-index-report/economy) Â· [Stanford AI Index 2026](https://hai.stanford.edu/assets/files/ai_index_report_2026.pdf)

## Naming and authoring convention

Folders retain stable part/module numbers. **INDEX.md** is authoritative local navigation; the [source plans](00_AUDIT_COVERAGE_MATRIX.md) add the detailed scope omitted by older umbrella rows. When authoring, use descriptive underscore-separated filenames, preserve the master's 15 case sections or 12 sheet sections, and update dependencies, evidence, review records and editorial state (**planned / draft / reviewed / published / needs-review**, plus deprecated where applicable). Do not count planning briefs as completed artifacts.

## Data-science domain library

The curriculum-level modules are complemented by a detailed [Data Science Domain Specialisations library](./11_data_science_domain_specialisations/INDEX.md). It explicitly covers the audit's production ML case families and DS cheat-sheet families, then adds deep industry maps for omnichannel retail, e-commerce/marketplaces, quick commerce, supply chain, streaming media, pharma, insurance, financial services, marketing/AdTech, mobility, SaaS/telecom, and energy/public-sector work.

See [Audit Coverage Matrix](./00_AUDIT_COVERAGE_MATRIX.md) for source-family navigation and [Source Topic Crosswalk](00_SOURCE_TOPIC_CROSSWALK.md) for exact-entry accounting, evidence limits, and verification. All visible topics are planned; inaccessible source contents are not claimed covered.
