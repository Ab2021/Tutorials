# Module 28 — Sources and Further Reading

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** Self-paced (12–16 hours for the annotated bibliography)
> **Track fit:** Common core
> **Prerequisites:** Modules 00–05; access to a library or the public web
> **Tools:** A reference manager, a link checker, and a source log
> **Industry question:** "Which sources can I rely on when a planning decision, a legal obligation, or a million-dollar inventory position depends on the answer?"

---

## 1. Why this module exists

Supply chain analytics is unusually vulnerable to stale and self-interested sources. A freight rate from last quarter, a vendor benchmark, a regulation that was amended, or a market-size projection repeated without its method can all lead a team to a confident wrong answer.

The problem is not a lack of information. It is **source quality and provenance**. A planner citing a blog post, a modeler citing a leaderboard, and an executive citing a press release are all making an evidence claim. This module teaches you to classify the claim, find the primary source, date it, and state its limits.

Three habits matter:

1. **Go upstream.** A news article about a regulation is not the regulation. A vendor blog about a benchmark is not the benchmark.
2. **Date everything.** Supply chain conditions and rules change. A number without a date is not evidence.
3. **Separate fact, inference, and forecast.** "Freight rates were $X on date Y" is a fact. "Rates will fall" is a forecast. "This will save 10%" is an inference that needs a model and assumptions.

## 2. Learning outcomes

By the end of this module, you can:

- Classify a source by type, authority, and proximity to the primary evidence.
- Locate the primary source for a standard, regulation, benchmark, or dataset.
- Distinguish enacted law, final rule, proposed rule, guidance, and enforcement discretion.
- Detect vendor and analyst claims that lack a comparator or a validation design.
- Build an annotated bibliography with a consistent evidence grade.
- Maintain a source log that records access dates and changes.
- Recognize when a source is too old for the decision at hand.
- Write a source-evaluation memo that a legal or finance reviewer can accept.

## 3. Mental model: the source ladder

```
Primary record (law, standard, dataset, filing, peer-reviewed paper)
        ↑
Official summary or guidance (regulator, standards body)
        ↑
Independent analysis (peer-reviewed review, official statistics)
        ↑
Analyst research (Gartner, Forrester, Nucleus, IDC)
        ↑
Vendor material (case study, blog, press release)
        ↑
News, social, conference slide
```

Move up the ladder for any claim that carries money, safety, or legal risk. Move down only for orientation and hypothesis generation.

### 3.1 Evidence grades

| Grade | Meaning | Example | Use |
|---|---|---|---|
| A | Primary, authoritative, dated | EUR-Lex text, GS1 specification, FDA guidance, peer-reviewed paper | Binding or high-stakes claims |
| B | Official summary or independent synthesis | European Commission explainer, NIST framework, systematic review | Interpretation and context |
| C | Reputable analyst or industry survey with method disclosed | Gartner press release, CSCMP report, DP World survey | Market context, labelled as such |
| D | Vendor claim or single case study | Product page, customer story, conference talk | Hypothesis, not proof |
| E | Unverified, anonymous, or undated | Social post, undated blog, unattributed slide | Do not use in a decision |

### 3.2 The five questions

For any source, ask:

1. **Who produced it, and what do they gain?**
2. **Is it primary or derived?**
3. **What is the date and the data period?**
4. **What is the method, sample, and comparator?**
5. **What would make this claim false?**

If you cannot answer all five, downgrade the grade.

## 4. Standards, regulation, and official sources

### 4.1 Standards bodies

| Body | What it governs | Why it matters | Where to start |
|---|---|---|---|
| GS1 | GTIN, GLN, SSCC, EPCIS, barcodes | Product and location identity, traceability, serialization | https://www.gs1.org/standards |
| ASCM | SCOR DS, CPIM/CSCP bodies of knowledge | Process reference model and planning vocabulary | https://www.ascm.org/ |
| ISO | 9001, 22301, 28000, 31000, 42001, 27001 | Quality, continuity, security, risk, AI management | https://www.iso.org/ |
| IEC | 62264 (ISA-95), 62541 (OPC UA) | Enterprise-control integration and industrial communication | https://www.iec.ch/ |
| ISA | ISA-95, ISA-88 | Manufacturing systems architecture and batch control | https://www.isa.org/ |
| UN/CEFACT | EDIFACT and trade data standards | Cross-border document exchange | https://unece.org/trade/uncefact |
| WCO | Harmonized System codes | Customs classification and trade analytics | https://www.wcoomd.org/ |
| ICC | Incoterms 2020 | Delivery terms, cost and risk transfer | https://iccwbo.org/ |

### 4.2 Regulation and government

| Instrument | Jurisdiction | Supply chain relevance | Primary source |
|---|---|---|---|
| EU AI Act (Regulation 2024/1689) | EU | High-risk AI, transparency, value-chain duties | https://eur-lex.europa.eu/eli/reg/2024/1689/oj |
| GDPR | EU | Personal data in customer, employee, and supplier data | https://eur-lex.europa.eu/eli/reg/2016/679/oj |
| CSDDD | EU | Supplier due diligence and chain of activities | https://eur-lex.europa.eu/eli/dir/2022/2464/oj |
| CSRD / ESRS | EU | Scope 3 and value-chain reporting | https://finance.ec.europa.eu/ |
| CBAM (Regulation 2023/956) | EU | Embedded emissions on imports | https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism_en |
| DSCSA | U.S. | Pharmaceutical serialization and tracing | https://www.fda.gov/drugs/drug-supply-chain-security-act-dscsa |
| EU FMD | EU | Falsified medicines and verification | https://health.ec.europa.eu/ |
| UFLPA | U.S. | Forced-labor import risk and tracing | https://www.cbp.gov/ |
| FTC Act / Sherman Act | U.S. | Pricing, information sharing, competition | https://www.ftc.gov/ and https://www.justice.gov/atr |
| NIST AI RMF | U.S. | AI risk management | https://www.nist.gov/itl/ai-risk-management-framework |

> **Note:** Dates and scope change. The research pack records the position as of 27 October 2026. Re-verify before relying on a deadline.

### 4.3 Official statistics and trade data

| Source | What it provides | Caveat |
|---|---|---|
| UN Comtrade | Bilateral trade flows by product | Reported values, lags, re-exports |
| World Bank Logistics Performance Index | Country logistics quality | Survey-based, periodic |
| OECD | Trade, industrial production, policy analysis | Aggregate, not firm-level |
| U.S. Census / BEA | Trade and economic data | Classification changes |
| Eurostat | EU trade and production | Reporting lags |
| National statistics offices | Local production, prices, employment | Definitions vary |

## 5. Foundational operations literature

### 5.1 Supply chain strategy and structure

- **Fisher, M. L. (1997). "What is the Right Supply Chain for Your Product?"** *Harvard Business Review*, March–April. The functional versus innovative product distinction; still the clearest framing of why one supply chain design cannot serve every product.
- **Lee, H. L., Padmanabhan, V., Whang, S. (1997). "Information Distortion in a Supply Chain: The Bullwhip Effect."** *Management Science*, 43(4), 546–558. The canonical explanation of demand amplification through batch ordering, price fluctuation, rationing, and information asymmetry.
- **Mentzer, J. T. et al. (2001). "Defining Supply Chain Management."** *Journal of Business Logistics*, 22(2), 1–25. A definitional anchor; useful when a team uses "supply chain" to mean logistics only.
- **Chopra, S., Meindl, P.** *Supply Chain Management: Strategy, Planning, and Operation.* The standard graduate text for network, inventory, and coordination.
- **Simchi-Levi, D., Kaminsky, P., Simchi-Levi, E.** *Designing and Managing the Supply Chain.* Strong on network design, risk, and procurement.
- **Christopher, M.** *Logistics and Supply Chain Management.* The service-driven logistics view.
- **Sheffi, Y.** *The Resilient Enterprise.* Disruption, flexibility, and redundancy.
- **Bowersox, D. J. et al.** *Supply Chain Logistics Management.* Logistics operations and integration.

### 5.2 Inventory and production

- **Silver, E. A., Pyke, D. F., Peterson, R.** *Inventory Management and Production Planning and Scheduling.* The reference for policies, lot sizing, and service measures.
- **Zipkin, P. H.** *Foundations of Inventory Management.* Rigorous treatment of stochastic inventory.
- **Clark, A. J., Scarf, H. (1960). "Optimal Policies for a Multi-Echelon Inventory Problem."** *Management Science*, 6(4), 475–490. The foundation of multi-echelon theory.
- **Graves, S. C., Willems, S. P. (2000). "Optimizing Strategic Safety Stock Placement in Supply Chains."** *Manufacturing & Service Operations Management*, 2(1), 68–83. Guaranteed-service modeling and safety-stock placement.
- **Sherbrooke, C. C.** METRIC-based inventory models. Multi-echelon spare-parts foundations.
- **Wagner, H. M., Whitin, T. M. (1958). "Dynamic Version of the Economic Lot Size Model."** *Management Science*, 5(1), 89–96. Dynamic lot sizing.
- **Hopp, W. J., Spearman, M. L.** *Factory Physics.* Throughput, queueing, variability, and production control.
- **Nahmias, S., Olsen, T. L.** *Production and Operations Analysis.* A broad production text.
- **Cachon, G., Terwiesch, C.** *Matching Supply with Demand.* Accessible operations fundamentals.

### 5.3 Optimization and simulation

- **Bertsimas, D., Sim, M. (2004). "The Price of Robustness."** *Operations Research*, 52(1), 35–53. Robust optimization.
- **Birge, J. R., Louveaux, F.** *Introduction to Stochastic Programming.* Scenario-based optimization.
- **Powell, W. B.** *Approximate Dynamic Programming.* Large-scale sequential decisions.
- **Sutton, R. S., Barto, A. G.** *Reinforcement Learning: An Introduction.* The standard RL text.
- **Law, A. M.** *Simulation Modeling and Analysis.* Discrete-event simulation and validation.
- **Banks, J. et al.** *Discrete-Event System Simulation.* Input modeling and experimental design.

## 6. Forecasting literature

- **Hyndman, R. J., Athanasopoulos, G.** *Forecasting: Principles and Practice*, 3rd ed. https://otexts.com/fpp3/ Free, current, and practical. Start here.
- **Box, G. E. P., Jenkins, G. M., Reinsel, G. C.** *Time Series Analysis: Forecasting and Control.* ARIMA foundations.
- **Makridakis, S., Wheelwright, S. C., Hyndman, R. J.** *Forecasting: Methods and Applications.* Classical methods and evaluation.
- **Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2020). "The M4 Competition."** *International Journal of Forecasting*, 36(1). Large-scale benchmark.
- **Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2022). "The M5 Accuracy Competition."** *International Journal of Forecasting*, 38(4). Retail hierarchy, promotions, and ML dominance. https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf
- **Croston, J. D. (1972). "Forecasting and Stock Control for Intermittent Demands."** *Operational Research Quarterly*, 23(3), 289–303.
- **Syntetos, A. A., Boylan, J. E. (2005). "The Accuracy of Intermittent Demand Estimates."** *International Journal of Forecasting*, 21(2), 303–314. ADI/CV² classification and Croston bias.
- **Timmermann, A. (2006). "Forecast Combinations."** In *Handbook of Economic Forecasting*. Why combination works and when it does not.
- **Gneiting, T., Raftery, A. E. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation."** *JASA*, 102(477), 359–378. CRPS and proper scoring.
- **Koenker, R., Hallock, K. F. (2001). "Quantile Regression."** *Journal of Economic Perspectives*, 15(4), 143–156.
- **Bass, F. M. (1969). "A New Product Growth for Model Consumer Durables."** *Management Science*, 15(5), 215–227.
- **TSFM-Bench (2025).** arXiv:2410.11802. Foundation-model benchmarking across domains.
- **"Time Series Foundation Models: Benchmarking Challenges and Requirements" (2025).** arXiv:2510.13654. Contamination and leakage risks.

## 7. Causal inference and experimentation

- **Hernán, M. A., Robins, J. M.** *Causal Inference: What If.* Free online. Target trial framing and DAGs.
- **Angrist, J. D., Pischke, J.-S.** *Mostly Harmless Econometrics.* IV, DiD, RDD.
- **Imbens, G. W., Rubin, D. B.** *Causal Inference for Statistics, Social, and Biomedical Sciences.* Potential outcomes.
- **Abadie, A., Diamond, A., Hainmueller, J. (2010). "Synthetic Control Methods."** *JASA*, 105(490), 493–505.
- **Athey, S., Imbens, G. W. (2017). "The State of Applied Econometrics."** *Journal of Economic Perspectives*, 31(2), 3–32. Practical causal toolkit.
- **Lewis, R. A., Rao, J. M. (2015). "The Unfavorable Economics of Measuring the Returns to Advertising."** *Quarterly Journal of Economics*, 130(4), 1941–1973. Power and noise in experiments.
- **Blake, T., Nosko, C., Tadelis, S. (2015). "Consumer Heterogeneity and Paid Search Effectiveness."** *Econometrica*, 83(1), 155–174. A famous null result and its implications.
- **Kohavi, R., Tang, D., Xu, Y.** *Trustworthy Online Controlled Experiments.* Practical A/B testing.

## 8. Industry research and market evidence

Use these for context and hypothesis generation, never as validation of your model.

| Source | What it is good for | Caveat |
|---|---|---|
| Gartner Supply Chain Top 25 and Predicts | Peer and analyst view of capability and trends | Composite methodology; analyst judgment |
| CSCMP State of Logistics Report | U.S. logistics cost and capacity context | Annual snapshot; sponsor influence |
| MIT Center for Transportation and Logistics | Research and industry practice | Varies by project |
| McKinsey, BCG, Deloitte supply chain practice | Adoption surveys and operating-model views | Consultancy marketing; method often partial |
| DP World Global Trade Observatory | Trade sentiment and resilience survey | Sponsor survey |
| Freightos, FreightWaves, Drewry | Freight rates and capacity | Dated; lane-specific |
| RELEX, Blue Yonder, Nucleus Research | Vendor and adoption landscape | Vendor or analyst interest |
| OECD, WTO, UNCTAD | Trade and maritime data | Aggregate and lagged |

### 8.1 How to read an industry survey

1. Who funded it?
2. Who was sampled, and how many?
3. What was the question wording?
4. Is the result a percentage of respondents or a projection?
5. Is the comparison year-over-year or a vendor's first survey?
6. Does the report disclose non-response and weighting?
7. Is the headline supported by the table?

A survey that says "71% plan to invest" is not evidence that 71% have invested. Module 00's adoption table depends on this distinction.

## 9. Datasets and benchmarks

### 9.1 Demand and retail

| Dataset | Grain and scope | Access | Known issues |
|---|---|---|---|
| M5 (Walmart) | Daily SKU-store, 42,840 series, 28-day horizon | Kaggle, public | Censored by stockouts; 2011–2016 |
| M4 | 100,000 series, multiple frequencies | Public | Mixed domains; not supply chain specific |
| Corporación Favorita | Daily store-item sales, promotions | Kaggle, public | Grocery-specific; missing data |
| Rossmann | Daily store sales | Kaggle, public | Store-level, not SKU-level |
| UCI Online Retail II | Transaction lines, 2009–2011 | UCI, public | Returns, cancellations, one retailer |
| DataCo Smart Supply Chain | Orders, shipping, late delivery | Mendeley, public | Synthetic-like; teaching only |
| Instacart Market Basket | Anonymized orders | Public | No prices or inventory |
| Dunnhumby Complete Journey | Household transactions | Registration | Panel, not full demand |

### 9.2 Manufacturing and maintenance

| Dataset | Use | Access | Caveat |
|---|---|---|---|
| NASA C-MAPSS | Turbofan degradation | Public | Simulated degradation |
| Bosch Production Line | Quality prediction | Kaggle | Extreme imbalance |
| SECOM | Semiconductor process quality | UCI | High-dimensional, missing |
| Scania APS Failure | Component failure | UCI | Cost-sensitive |
| MetroPT | Air compressor telemetry | UCI | Anomaly labels imperfect |
| SKAB | Water pump anomalies | GitHub | Small |

### 9.3 External signals

| Source | Signal | Access |
|---|---|---|
| UN Comtrade | Trade flows | Public API |
| World Bank LPI | Logistics performance | Public |
| GDELT | Events, conflict, protest | Public API |
| ACLED | Conflict events | Registration |
| NOAA | Weather and climate | Public |
| Open Supply Hub | Facility locations | Public |
| CDP | Corporate environmental disclosures | Registration |

> **Rule:** every dataset in a lab must state its license, collection period, known defects, and whether it is real, synthetic, or anonymized. A benchmark score is not a production estimate.

## 10. Tools and documentation

| Category | Tools | Documentation |
|---|---|---|
| Forecasting | statsforecast, sktime, darts, GluonTS | Read the evaluation and backtesting docs, not only the model API |
| Gradient boosting | LightGBM, XGBoost, CatBoost | Check objective functions for count or Tweedie demand |
| Optimization | OR-Tools, Pyomo, HiGHS, Gurobi, CPLEX | Read solver status and infeasibility guidance |
| Simulation | SimPy, AnyLogic, Arena, Salabim | Validate input distributions and warm-up |
| Data | DuckDB, Polars, dbt, Airflow, Dagster | Check incremental and late-data behavior |
| MLOps | MLflow, Evidently, Great Expectations, DVC | Check drift and data-quality semantics |
| Visualization | matplotlib, plotly, Power BI, Tableau | Label units, dates, and uncertainty |

## 11. Professional communities

| Community | Focus | Why join |
|---|---|---|
| INFORMS | Operations research and analytics | Methods, job market, case competitions |
| POMS | Production and operations management | Academic-practice bridge |
| MSOM Society | Operations management research | Rigorous inventory and supply chain work |
| DSI | Decision sciences | Applied analytics and teaching |
| CSCMP | Logistics and supply chain practice | Practitioner network and State of Logistics |
| ASCM | Supply chain profession and certification | SCOR, CPIM, CSCP |
| GS1 communities | Standards and interoperability | Implementation guidance |
| Open Supply Hub | Facility transparency | Supplier mapping and due diligence |

## 12. Source-evaluation checklist

Use this before citing any source in a memo, dossier, or capstone.

- [ ] Author or institution is named.
- [ ] Date of publication and data period are stated.
- [ ] The source is primary, or the path to the primary source is given.
- [ ] The sample, method, and comparator are described.
- [ ] The claim is distinguishable from the headline.
- [ ] The unit, currency, and time bucket are stated.
- [ ] The jurisdiction and legal status are stated (for regulation).
- [ ] The source's commercial interest is disclosed.
- [ ] The claim is still current for the decision.
- [ ] A second independent source exists for high-stakes claims.
- [ ] The access date is recorded.
- [ ] The exact URL or citation is preserved.

## 13. Keeping current

### 13.1 A personal source log

Maintain a table with:

| Field | Example |
|---|---|
| Claim | "CBAM definitive period began 1 Jan 2026" |
| Source | European Commission, CBAM definitive regime page |
| URL | https://taxation-customs.ec.europa.eu/... |
| Date | Accessed 27 Oct 2026 |
| Grade | A |
| Status | Current |
| Next review | Apr 2027 |
| Note | 50-tonne threshold; first surrender 30 Sep 2027 |

### 13.2 Review cadence

| Domain | Review frequency | Trigger |
|---|---|---|
| Regulation | Quarterly | New guidance, court decision, election |
| Freight and trade | Monthly | Rate shock, route change, tariff |
| Standards | Annually | Version release |
| Methods | Annually | Major benchmark or library change |
| Datasets | On use | License or version change |

### 13.3 Avoiding citation drift

- Link to the stable landing page, not a search result.
- Save a PDF or snapshot for anything that may change.
- Quote the exact sentence you rely on.
- Do not cite a source you have not opened.
- Do not cite a DOI you cannot resolve.
- Do not present an abstract as if you read the paper.

## 14. Assignment, rubric, and sources

### 14.1 Assignment

**Annotated bibliography and source log (individual, 12 hours).**

Select ten sources across at least four categories: regulation or standard, peer-reviewed method, industry evidence, and dataset or tool documentation. For each:

1. Full citation with URL and access date.
2. Evidence grade (A–E).
3. The specific claim you would use it for.
4. Method, sample, and limitations.
5. One sentence on what would make the claim false.
6. A note on how you would re-verify it in six months.

Then write a 500-word memo on one contested claim in supply chain analytics, comparing two sources that disagree.

### 14.2 Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Source quality | 25% | Blogs and undated pages | Mix of primary and secondary | Primary sources for every high-stakes claim |
| Citation accuracy | 20% | Missing dates or URLs | Complete citations | Resolvable, archived, and quoted precisely |
| Method critique | 20% | Repeats the claim | Identifies sample and comparator | Explains what would falsify it |
| Legal and regulatory care | 15% | Treats guidance as law | Distinguishes status and jurisdiction | Tracks amendments and dates |
| Synthesis | 10% | List of links | Thematic annotations | Shows disagreement and resolves it |
| Maintenance | 10% | No review plan | Notes a review date | Source log with triggers and owners |

### 14.3 Sources for this module

- Hyndman, R. J., Athanasopoulos, G., *Forecasting: Principles and Practice*. https://otexts.com/fpp3/
- Hernán, M. A., Robins, J. M., *Causal Inference: What If*. https://miguelhernan.org/whatifbook
- EUR-Lex. https://eur-lex.europa.eu/
- GS1. https://www.gs1.org/standards
- ASCM. https://www.ascm.org/
- ISO. https://www.iso.org/
- NIST AI RMF. https://www.nist.gov/itl/ai-risk-management-framework
- FDA DSCSA. https://www.fda.gov/drugs/drug-supply-chain-security-act-dscsa
- European Commission, CBAM. https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism_en
- UN Comtrade. https://comtradeplus.un.org/
- World Bank LPI. https://lpi.worldbank.org/
- M5 methods. https://github.com/Mcompetitions/M5-methods

> **Next:** [Module 29](29_industry_electives.md) applies the program to six industry contexts.
