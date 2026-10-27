# 2025–2026 Supply Chain, AI, and Analytics Source Pack

> **Purpose:** Teaching anchors and dated evidence for the 2026 edition. These are snapshots, not forecasts. Verify before quoting in a live engagement.
> **Compiled:** 27 October 2026.
> **Rule:** Use the primary source where possible. Vendor surveys are labelled as vendor surveys. Analyst predictions are labelled as predictions, never as outcomes.

---

## 1. AI and analytics adoption

| Claim | Source | Date | Status |
|---|---|---|---|
| Gartner predicts 70% of large organizations will adopt AI-based supply chain forecasting or touchless demand planning by 2030 | Gartner press release | 16 Sep 2025 | Analyst prediction |
| Gartner's 2026 supply chain technology trends group into autonomy/agency (polyfunctional robots, physical AI, agentic AI, multiagent systems), specialization/intelligence (intelligent simulation, domain-specific language models), and trust/governance (product provenance, decision governance) | Gartner press release | 30 Jun 2026 | Analyst view |
| Only 8% of surveyed supply chain leaders currently use agentic AI; 46% have no plans | Blue Yonder, Supply Chain Compass 2026 (678 respondents, autumn 2025) | 2026 | Vendor survey |
| About 44% AI adoption in supply chains, but only about 30% report measurable value and under 25% have maturity at scale | BCG, AI's New Mandate in Supply Chains | May 2026 | Consultancy survey |
| 71% plan to invest in generative/agentic AI over 3–5 years; only 10% would trust AI with fully independent decisions | RELEX, State of Supply Chain 2026 (514 leaders) | Jan 2026 | Vendor survey |
| 10.6% report AI embedded; 82.4% still review AI recommendations before acting | Logistics Reply, 2026 AI Survey | 2026 | Vendor survey |
| Gartner Top 25 for 2026 is led by Schneider Electric, NVIDIA, Walmart, Cisco, Lenovo; themes are autonomous workforce, network-centric strategy, and end-to-end orchestration | Gartner press release | 17 Jun 2026 | Analyst ranking |

**Teaching use:** The gap between intent (70%+) and embedded deployment (about 10%) is the central commercial fact of the 2026 course. It justifies modules on adoption, data engineering, MLOps, model risk, and planner workflow.

**URLs**
- https://www.gartner.com/en/newsroom/press-releases/2025-09-16-gartner-predicts-70-percent-of-large-orgs-will-adopt-ai-based-supply-chain-forecasting-to-predict-future-demand-by-2030
- https://www.gartner.com/en/newsroom/press-releases/2026-06-30-gartner-identifies-top-supply-chain-technology-trends-for-2026
- https://www.gartner.com/en/newsroom/press-releases/2026-06-17-gartner-announces-2026-rankings-of-the-global-supply-chain-top-25
- https://www.bcg.com/assets/2026/executive-perspectives-ais-new-mandate-in-supply-chains.pdf
- https://www.relexsolutions.com/news/relex-report-ai-moves-into-core-supply-chain-decisions-as-volatility-persists/
- https://edge.sitecorecloud.io/blueyonderie8e6-corporateefb0-prod6ccb-be06/media/project/blueyonder/corporate/files/resources/supply-chain-compass-spotlight-on-ai-and-technology.pdf

---

## 2. Forecasting methods and benchmarks

| Finding | Source | Date | Caveat |
|---|---|---|---|
| M5 Accuracy competition used Walmart daily data: 42,840 hierarchical series, 1,913 training days, 28-day horizon, WRMSSE metric, 5,507 teams | Makridakis, Spiliotis, Assimakopoulos, M5 results paper | 2021 | Historical benchmark |
| M5 winner used an equal-weighted combination of 220 LightGBM models with a Tweedie objective, per store, store-category, and store-department; WRMSSE 0.520, 22.4% better than the ES bu benchmark | M5 results paper; In & Jung, IJF | 2021–2022 | Competition result, not a production guarantee |
| All top-50 M5 methods were ML and beat the statistical benchmark by more than 14%; cross-learning, exogenous variables, combination, and time-based validation were the drivers | M5 results paper | 2021 | Retail-specific |
| Time-series foundation models (Chronos, TimesFM, Moirai, TTM, ROSE, Timer, MOMENT) show no single dominant model; zero-shot models beat task-specific models on 10 of 21 datasets; scaling laws do not hold strictly; LLM-based models generally underperform | TSFM-Bench, KDD 2025, arXiv:2410.11802 | 2025 | Benchmark-dependent |
| Test-set contamination is a serious risk: one model's training set is another's test set; only about 7% of datasets were never used for pretraining or fine-tuning in one audit | arXiv:2510.13654 | 2025 | Preprint audit |
| Classical bi-seasonal MSTL was never statistically beaten by foundation models in a five-market electricity-price study | arXiv:2506.08113 | 2025 | Domain-specific |
| Zero-shot foundation models were beaten by linear and seasonal-naive baselines on real cloud workloads | arXiv:2502.12944 | 2025 | Domain-specific |

**Teaching use:** Module 07 insists on naive and seasonal-naive baselines; Module 08 treats foundation models as candidates, not defaults; Module 18 teaches leakage audits for pretrained models; Module 24 requires benchmark provenance.

**URLs**
- https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf
- https://github.com/Mcompetitions/M5-methods
- https://arxiv.org/html/2410.11802v5
- https://arxiv.org/html/2510.13654v1

---

## 3. Trade, freight, and network strategy

| Fact | Source | Date |
|---|---|---|
| 94% of surveyed executives expect 2026 trade growth to match or exceed 2025; tariffs and economic shocks are top risks; 90% expect trade barriers to rise or hold | DP World Global Trade Observatory 2026 (3,500+ executives) | 2026 |
| Resilience over cost-cutting: 51% cite supplier diversification, 44% higher inventories, 36% friendshoring, 32% nearshoring; 46% plan new routes | DP World | 2026 |
| Red Sea transits recovering to roughly 30–35% of normal; Gemini expects near-full Suez return in Q1 2027 | FreightWaves, citing Lars Jensen | 20 Oct 2026 |
| Asia–USWC about $8,126/FEU; Asia–USEC about $9,700/FEU; Asia–North Europe about $3,442/FEU; Asia–Med about $3,927/FEU | FreightWaves / Freightos | Oct 2026 |
| Port congestion, typhoon backlogs, and bunker costs set a higher rate floor; carriers push GRIs into 2027 contract tendering | Freightos Global Freight Outlook | Oct 2026 |

**Teaching use:** Module 11 (network design) and Module 12 (transportation) use these as scenario inputs, not constants. Students must date-stamp freight assumptions. Module 14A uses Red Sea and Hormuz as disruption cases.

**URLs**
- https://www.freightos.com/freight-resources/freightos-global-outlook-october-2026/
- https://www.freightwaves.com/news/how-new-capacity-is-raising-rates-in-chaotic-ocean-market
- https://www.worldports.org/shippers-put-resilience-ahead-of-cost-cutting-survey/

---

## 4. Regulation and standards

| Instrument | Status as of Oct 2026 | Supply chain relevance |
|---|---|---|
| EU AI Act (Regulation 2024/1689) | In force 1 Aug 2024. Prohibitions and AI literacy from 2 Feb 2025; GPAI obligations from 2 Aug 2025; enforcement from 2 Aug 2026. AI Omnibus political agreement 7 May 2026, in force 27 Jul 2026. Annex III high-risk from 2 Dec 2027; Annex I embedded systems from 2 Aug 2028 | Workforce analytics, critical infrastructure, safety components, decision governance; Art. 25 value-chain provider duties |
| CSDDD (Directive (EU) 2022/2464 as amended) | Omnibus I in force 18 Mar 2026; transposition extended to 26 Jul 2028; scope narrowed to >5,000 employees and €1.5bn turnover; climate transition plan and civil liability provisions removed | Supplier due diligence, chain-of-activities risk, grievance mechanisms |
| CSRD / ESRS | Omnibus I (Directive (EU) 2026/470) raises thresholds to >€450m turnover and >1,000 employees from FY2027; value-chain cap for protected undertakings ≤1,000 employees; Scope 3 phased relief | Scope 3, supplier data requests, assurance |
| CBAM (Regulation (EU) 2023/956) | Definitive regime from 1 Jan 2026; 50-tonne mass threshold; authorised declarant; 50% certificate holding from 1 Jan 2027; first declaration and surrender 30 Sep 2027 | Embedded emissions, supplier carbon data, customs |
| DSCSA (U.S.) | Enhanced package-level electronic tracing since 27 Nov 2023; phased exemptions: manufacturers 27 May 2025, wholesalers 27 Aug 2025, large dispensers 27 Nov 2025, small dispensers 27 Nov 2026; GS1 EPCIS recommended | Pharma serialization, EPCIS, 3PL data exchange |
| EU FMD | Falsified Medicines Directive; product verification and tamper evidence | Pharma distribution |
| UFLPA / forced-labor import rules | U.S. import detention risk; tracing and supplier mapping | Supplier traceability, customs |
| Incoterms 2020 | Contractual delivery terms | Cost and risk transfer, landed cost |
| GS1 EPCIS 2.0 | Event standard for visibility and traceability | Serialization, chain of custody, control towers |

**Caution:** Regulatory dates move. The above reflects published sources at the cutoff. Re-verify before relying on any deadline.

**URLs**
- https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- https://eur-lex.europa.eu/eli/dir/2022/2464/2026-03-18/eng
- https://eur-lex.europa.eu/eli/dir/2026/470/oj/eng
- https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism/cbam-definitive-regime_en
- https://www.fda.gov/drugs/drug-supply-chain-security-act-dscsa/fdas-implementation-drug-supply-chain-security-act-dscsa-requirements
- https://www.business-humanrights.org/en/latest-news/csddd-omnibus/

---

## 5. Pricing, competition, and antitrust

| Development | Source | Date |
|---|---|---|
| DOJ and RealPage reached a proposed settlement restricting use of non-public, competitively sensitive data | U.S. DOJ press release | Nov 2025 |
| DOJ/FTC statements of interest argue algorithmic price fixing is per se unlawful and no competitor-to-competitor communication is required | DOJ/FTC | 2025–2026 |
| Gibson v. Cendyn (9th Cir.) dismissed claims against shared hotel revenue-management software; en banc rehearing denied Nov 2025 | Ninth Circuit | 2025 |
| OECD G7 report catalogues algorithm-facilitated cartels, hub-and-spoke, vertical agreements, and autonomous tacit collusion | OECD | Oct 2025 |
| California and New York algorithmic-pricing laws took effect; RealPage constitutional challenges pending | State legislation and litigation | 2026 |

**Teaching use:** Module 16 must teach pricing optimization with a legal and ethical boundary. A model that recommends a price using a competitor's non-public signal is a legal risk, not a clever feature. Module 22 covers governance.

**URLs**
- https://www.justice.gov/opa/pr/justice-department-reaches-proposed-settlement-greystar-largest-us-landlord-end-its
- https://www.reuters.com/legal/legalindustry/collusion-by-code-understanding-algorithmic-pricing-antitrust-enforcement--pracin-2026-03-20/
- https://www.oecd.org/content/dam/oecd/en/publications/reports/2025/10/algorithmic-pricing-and-competition-in-g7-jurisdictions_f936689b/f36dacf8-en.pdf

---

## 6. Automation, robotics, and fulfillment

| Fact | Source | Date |
|---|---|---|
| Amazon surpassed 1 million warehouse robots and introduced DeepFleet, a generative-AI fleet coordination model | Amazon / Forbes | Jul 2025 |
| Amazon Vulcan is its first robot with force-feedback touch; handles about 75% of item types at human-comparable speed | About Amazon | 2025 |
| Amazon plans >€10bn to modernize EU fulfillment centers; Proteus deploys in Europe H1 2027; STARK scales to 15 EU sites by 2027 | About Amazon | 2025–2026 |
| Ocado's On-Grid Robotic Pick picked 30M+ items in 2024 with under 100 arms; about 500 arms expected by end-2025; about 40% of groceries packed, targeting about 80% | Ocado / The Verge | 2024–2025 |
| Warehouse automation economics depend on throughput, SKU mix, labor availability, and uptime — not robot count | Inference from published cases | 2026 |

**Teaching use:** Module 13 uses these as evidence for automation economics. Students must model payback under labor rate, throughput, and uptime assumptions rather than repeat vendor claims.

**URLs**
- https://www.aboutamazon.com/news/operations/amazon-vulcan-robot-pick-stow-touch
- https://www.aboutamazon.com/news/operations/amazon-europe-robotics-delivery-investment
- https://www.ocadogroup.com/newsroom/stories/ocado-robotic-arms
- https://www.theverge.com/robot/719880/ocado-online-grocery-automation-krogers-luton-ogrp-robot-grid

---

## 7. Digital twins and control towers

| Finding | Source | Date | Caveat |
|---|---|---|---|
| Digital twin market estimated at $21bn (2024) and projected at $39bn in 2026; at least 50% of projects fail by some estimates | Schmidt et al., Production Planning & Control, Delphi study | 2026 | Market sizing is uncertain; treat as directional |
| Only 16% of reviewed supply chain "digital twins" achieve closed-loop capability; many are simulation models or digital shadows | Benhamou, Giard, Lamouri, IJPE 291 | 2026 | Literature review of 51 cases |
| Adoption is driven more by competitive pressure and market uncertainty than by perceived relative advantage; complexity is a barrier | Huang, Ghadge, Yates, IJLRA (260 practitioners) | 2026 | Food supply chains |
| Control tower leaders in Nucleus 2025 matrix: Blue Yonder, E2open, Infor, Kinaxis, o9; theme is AI agents that monitor, simulate, and recommend | Nucleus Research | Sep 2025 | Analyst matrix |
| Planning leaders in Nucleus 2025: Blue Yonder, Infor, John Galt, Kinaxis, o9, RELEX, Wolters Kluwer CCH Tagetik | Nucleus Research | Jul 2025 | Analyst matrix |

**Teaching use:** Module 19 separates visibility, control tower, and decision orchestration. Module 20 requires a validated simulation, not a dashboard called a twin.

**URLs**
- https://www.tandfonline.com/doi/pdf/10.1080/09537287.2026.2711341
- https://ideas.repec.org/a/eee/proeco/v291y2026ics0925527325003275.html
- https://nucleusresearch.com/news/nucleus-research-releases-2025-control-tower-technology-value-matrix/
- https://nucleusresearch.com/news/nucleus-research-releases-2025-supply-chain-planning-technology-value-matrix/

---

## 8. Sustainability and supply chain finance

| Fact | Source | Date |
|---|---|---|
| ESRS E1-6 requires gross Scope 1, 2, and 3 GHG emissions in tCO2e by significant category, using GHG Protocol | EUR-Lex delegated regulation 2023/2772 as amended | 2025–2026 |
| Scope 3 may be omitted in the first year for undertakings ≤750 employees; value-chain data may use estimates and proxies for the first three years | ESRS 1 and E1 phase-in | 2026 |
| CSRD thresholds rise to >€450m turnover and >1,000 employees from FY2027 | Directive (EU) 2026/470 | 26 Feb 2026 |
| CBAM definitive period starts 1 Jan 2026 with a 50-tonne threshold and first surrender 30 Sep 2027 | European Commission | 2026 |

**Teaching use:** Module 22 and Module 29 teach emissions accounting as a data problem: activity data, emission factors, boundaries, allocation, and assurance. Warn against spend-based estimates presented as supplier-specific measurements.

**URLs**
- https://finance.ec.europa.eu/financial-markets/company-reporting-and-auditing/company-reporting/corporate-sustainability-reporting_en
- https://eur-lex.europa.eu/eli/reg_del/2023/2772/2025-01-01/eng

---

## 9. Datasets and benchmarks to verify before use

| Dataset | Domain | Access | Note |
|---|---|---|---|
| M5 (Walmart) | Retail demand | Public, Kaggle | 42,840 series; 28-day horizon; WRMSSE |
| M4 | General forecasting | Public | 100,000 series; multiple frequencies |
| Corporación Favorita | Grocery demand | Public, Kaggle | Promotions and store metadata |
| Rossmann | Retail sales | Public, Kaggle | Store-level daily sales |
| UCI Online Retail II | E-commerce transactions | Public, UCI | Returns and cancellations |
| DataCo Smart Supply Chain | Orders and shipping | Public, Mendeley | Synthetic-ish, useful for teaching |
| Instacart Market Basket | Basket affinity | Public | Anonymized orders |
| Dunnhumby Complete Journey | Retail transactions | Public, registration | Household panel |
| NASA C-MAPSS | Predictive maintenance | Public, NASA | Turbofan degradation |
| Bosch Production Line | Manufacturing quality | Public, Kaggle | Highly imbalanced |
| SECOM | Manufacturing quality | Public, UCI | High-dimensional, missing data |
| Scania APS Failure | Truck component failure | Public, UCI | Cost-sensitive |
| MetroPT | Air compressor telemetry | Public, UCI | Anomaly detection |
| SKAB | Water pump anomalies | Public, GitHub | Anomaly detection |
| UN Comtrade | Trade flows | Public API | Country/product level |
| World Bank LPI | Logistics performance | Public | Country-level index |
| GDELT | Event data | Public API | Protest, conflict, disaster signals |
| ACLED | Conflict events | Registration | Sub-national conflict |
| NOAA | Weather and climate | Public | Demand and disruption drivers |

**Rule:** every dataset used in a lab must state its license, collection period, known defects, and whether it is real, synthetic, or anonymized. Do not present a benchmark score as a production estimate.

---

## 10. What changed since the 2020 edition

1. **Forecasting moved from statistical engines to ML and hybrid pipelines**, with LightGBM-style cross-learning dominant in retail competitions and foundation models entering as candidates, not replacements.
2. **Agentic AI arrived in marketing before it arrived in planning.** Actual embedded use is low; human-in-the-loop remains the norm.
3. **Resilience became a board-level objective.** Network redesign, supplier diversification, and inventory buffers are now strategy, not just cost optimization.
4. **Regulation expanded from financial and privacy to supply chain due diligence, carbon, and AI governance.**
5. **Visibility data exploded, but decision quality improved less.** Control towers without decision rights and data governance become expensive dashboards.
6. **Digital twins are oversold.** Most deployed artifacts are simulations or digital shadows; closed-loop twins are rare.
7. **Talent shifted toward hybrid skills**: SQL and data engineering, forecasting, optimization, planning process, and commercial communication.

---

## 11. Claims boundary

This pack is a teaching aid. It is not investment, legal, trade-compliance, or procurement advice. Market sizes, freight rates, adoption percentages, and regulatory dates are dated snapshots and may be wrong or superseded. Re-verify against the primary source before using any figure in a commercial decision.
