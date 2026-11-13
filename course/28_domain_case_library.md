# Module 28 — Domain Case Library and Sector Playbooks

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** Self-paced, 18–24 hours
> **Track fit:** Common core; use with Modules 21–24 and 29
> **Prerequisites:** Modules 00, 01, 07, 10
> **Tools:** A case notebook, the source ladder in section 14, and the sector primers in Module 29
> **Industry question:** "What actually happened when a real supply chain tried this, and what transfers to my decision?"

---

## 1. Why industry cares

Methods are portable; context is not. A case is the cheapest way to learn the context:

- What decision was actually made?
- What data did the organization have, and what was missing?
- What intervention was tried?
- What evidence supports the claimed outcome?
- What would not transfer to another company?

Analysts who know cases ask better questions in interviews and avoid repeating known failures. This module is a **case library**, not a history of famous companies. Each case is summarized in a consistent anatomy so it can be compared, challenged, and reused.

## 2. Learning outcomes

By the end of this module, you can:

- Read a supply chain case and separate claim from evidence.
- Apply a consistent case anatomy to a new situation.
- Compare cases across sectors and identify what transfers.
- Build a sector playbook from cases and standards.
- Detect survivorship, selection, and vendor-claim bias in case writing.
- Teach a case to a non-technical audience.
- Use cases to frame a domain project or interview answer.
- Maintain a personal, sourced case library.

## 3. Mental model

```text
  CONTEXT ──> DECISION ──> DATA ──> INTERVENTION ──> EVIDENCE ──> OUTCOME
     │            │          │            │              │            │
  sector,      who, what,  available,   model, process, quality,     measured
  constraints  when       missing      org change      source       result
     │            │          │            │              │            │
     └────────────┴──────────┴────────────┴──────────────┴────────────┘
                              TRANSFER TEST
                    what moves, what does not, what must be checked
```

### 3.1 Case anatomy

| Element | Question |
|---|---|
| Context | Sector, product, network, constraints, date |
| Decision | Who decided what, at what cadence |
| Data | Systems, grain, coverage, quality |
| Intervention | Model, process, organization, technology |
| Evidence | Independent, company, vendor, press, or anecdote |
| Outcome | Metric, magnitude, period, attribution |
| Transfer | What generalizes; what is specific |
| Limits | What the case does not prove |

## 4. Core concepts

### 4.1 Evidence grades

| Grade | Source | Use |
|---|---|---|
| A | Peer-reviewed or audited | Strong evidence |
| B | Regulatory filing or official statistic | Reliable facts |
| C | Company case study or investor claim | Directional |
| D | Vendor marketing | Hypothesis only |
| E | Press summary or anecdote | Lead, not proof |

A case can be useful at grade C if the claim is labelled. It cannot be used as proof of causality.

### 4.2 The transfer test

Ask five questions before reusing a case:

1. Is the decision the same?
2. Is the product economics similar?
3. Is the data environment similar?
4. Is the operating constraint similar?
5. Is the evidence strong enough to justify the risk?

If three answers are no, the case is an analogy, not a plan.

### 4.3 Case selection bias

Published cases skew toward:

- Success.
- Large firms.
- Vendors with marketing budgets.
- Recent technology.
- English-language markets.

Counteract with failures, small firms, public-sector cases, and non-English sources where possible.

### 4.4 Building a sector playbook

A playbook contains:

- The three decisions that matter most.
- The data that is usually available and usually missing.
- The methods that fit the constraint.
- The regulatory boundary.
- Two or three cases with evidence grades.
- A default measurement plan.
- The common failure modes.

### 4.5 A worked evidence audit

A vendor page states: "AI forecasting cut inventory by 30% at a European grocery chain."

| Question | What to demand | What a weak answer sounds like |
|---|---|---|
| Compared with what? | The baseline method and its parameters | "Their old spreadsheet" |
| Over what period? | At least 12 months, including a seasonal peak | "A six-week pilot" |
| Which scope? | Stores, categories, and SKUs included | "The whole business" |
| Measured how? | Inventory value or units, and at what service level | "30% less stock" |
| Service held constant? | Fill rate before and after | Silence on availability |
| Attribution? | Control stores, staggered rollout, or a counterfactual | "We deployed it and inventory fell" |
| Selection? | Were pilot stores chosen for good data? | "They volunteered" |
| Denominator? | Total inventory or only the modelled categories | Unclear |

Read carefully, the claim is grade D: a vendor marketing statement with no baseline, no control, and no service-level disclosure. It is a hypothesis worth testing, not a result worth repeating.

A grade B version would read: "Across 214 stores and 12 months, forecast-driven replenishment reduced average inventory value by 12% while holding fill rate at 97.5% or above; 60 matched control stores showed a 2% decline over the same period." That statement names the baseline, the period, the scope, the metric, the service constraint, and the counterfactual. It is also far less exciting, which is usually a sign that it is closer to true.

### 4.6 Counter-case discipline

Every case in a portfolio should be paired with a counter-case or a documented failure boundary.

| Claim | Counter-case | What it teaches |
|---|---|---|
| Centralizing inventory reduces cost | Long lead times and service targets can reverse it | Centralization trades transport and lead time for holding cost |
| Faster delivery wins customers | Speed without accuracy raises contacts and churn | Promise reliability matters more than promise length |
| More data improves forecasts | Promotional and new-item series may not benefit | Data quality and relevance beat volume |
| Automation reduces labour cost | Fixed cost and utilization may dominate | Volume and variability decide the case |
| AI beats classical models | M5 and TSFM-Bench show conditional gains | Strong baselines and local evaluation are mandatory |

A portfolio with no counter-cases is a sales deck.

## 5. Data, standards, and workflow

### 5.1 Case note template

```text
Case:
Sector / geography / date:
Decision and owner:
Product and network:
Data available / missing:
Intervention:
Evidence grade and source:
Outcome and period:
What transferred:
What did not:
Open questions:
```

### 5.2 Source workflow

1. Find a primary source (paper, filing, regulator, company report).
2. Record the date and the claim.
3. Grade the evidence.
4. Find a counter-case or a limitation.
5. Write the case note with the grade visible.
6. Re-check before teaching or citing.

### 5.3 Standards to anchor cases

- ASCM SCOR DS for process framing.
- GS1 for identifiers and event data.
- ISO 22301 for continuity.
- ISO 31000 for risk.
- EU AI Act and NIST AI RMF for AI cases.
- Country competition and consumer law for pricing cases.

## 6. Case library by domain

### 6.1 Retail and e-commerce

| Case | Decision | Method | Evidence | Lesson |
|---|---|---|---|---|
| Walmart M5 | Forecast 42,840 series | LightGBM ensemble, WRMSSE | A (competition) | Cross-learning and hierarchy matter |
| Amazon regionalization | Place inventory close to demand | Network design, promise | C (company) | The promise drives the network |
| Zalando returns | Reduce avoidable returns | Fit analytics | C (company) | Returns are a product problem |
| Ocado automation | Automate grocery fulfillment | AS/RS, routing | C (company) | Density and utilization decide economics |
| Shein small-batch | Reduce forecast risk | Real-time demand, small orders | E (press) | Responsive supply can substitute for forecast accuracy |
| Instacart substitution | Replace unavailable items | Recommendation, rules | C (company) | Availability is a ranking constraint |

### 6.2 CPG and consumer goods

| Case | Decision | Method | Evidence | Lesson |
|---|---|---|---|---|
| P&G–Walmart continuous replenishment | Share demand and replenish | EDI, VMI | C/D (company) | Collaboration needs incentives and trust |
| Trade promotion ROI | Allocate trade spend | Incrementality, MMM | B/C | Average lift hides unprofitable events |
| New product launch | Forecast without history | Analogy, attributes | C | Distribution and awareness dominate early |
| Private label growth | Assortment and margin | Category analytics | C | Margin and share trade-offs |

### 6.3 Manufacturing and industrial

| Case | Decision | Method | Evidence | Lesson |
|---|---|---|---|---|
| Toyota JIT | Reduce waste and inventory | Kanban, andon | A/B (long literature) | JIT depends on supplier proximity and stability |
| Tōhoku 2011 | Recover from tier-2 disruption | Multi-tier mapping | A/B | Tier-3 visibility is a strategic risk |
| Boeing 787 | Outsource to partners | Supplier management | A/B (GAO, press) | Coordination cost can exceed labor savings |
| VW–Prevent 2016 | Supplier compliance | Audit and controls | B/C | Supplier risk is legal and reputational |
| Semiconductor shortage 2021 | Allocate scarce chips | Allocation policy | A/B | Forecast accuracy cannot fix capacity |

### 6.4 Logistics and distribution

| Case | Decision | Method | Evidence | Lesson |
|---|---|---|---|---|
| UPS ORION | Optimize routes | Heuristics, dispatch | C (company) | Adoption and exception handling decide value |
| Maersk NotPetya | Recover from cyberattack | Resilience, rebuild | B/C | Recovery is part of resilience |
| Red Sea diversion 2023–2026 | Reroute around Suez | Network and cost | B (shipping data) | Lead-time buffers and freight volatility |
| Convoy shutdown | Build a digital freight marketplace | Marketplace economics | B/C | Network effects do not guarantee margin |
| Flexport | Digital forwarding | Visibility and service | C | Software alone does not fix operations |

### 6.5 Pharma, cold chain, and humanitarian

| Case | Decision | Method | Evidence | Lesson |
|---|---|---|---|---|
| COVID-19 vaccine allocation | Allocate scarce doses | Allocation ethics, cold chain | A/B | Fairness is a policy choice, not a model output |
| DSCSA serialization | Trace packages | EPCIS, serialization | A (regulation) | Data quality is a compliance issue |
| GLP-1 supply constraint | Allocate demand surge | Capacity and access | B/C | Demand forecasting cannot create supply |
| Haiti 2010 | Coordinate relief | Logistics cluster | A/B | Data coordination is a bottleneck |
| Ebola 2014–2015 | Last-mile response | Needs assessment | A/B | Access and trust determine delivery |

### 6.6 Cross-cutting AI and analytics

| Case | Decision | Method | Evidence | Lesson |
|---|---|---|---|---|
| Zillow Offers | Price and buy homes | Automated valuation | B/C | Model risk becomes balance-sheet risk |
| Knight Capital | Deploy trading logic | Change control | A (SEC) | Deployment is a risk event |
| Target 2013 | Vendor access | Security architecture | A/B | Third-party access is a supply chain risk |
| M5 winner | Forecast retail | LightGBM ensemble | A | Baselines and features beat complexity |
| Foundation-model benchmarks | Choose a model | Zero-shot evaluation | A/B | Contamination and local baselines matter |

### 6.7 Quick commerce and on-demand delivery

| Case | Decision | Method | Evidence | Lesson |
|---|---|---|---|---|
| Getir and Gopuff expansion | Where to open dark stores | Density and unit economics | C/E (company, press) | Speed does not create a moat without basket density |
| Instacart batching | Batch or single-order delivery | Dispatch and routing | C (company) | Batching trades cost against lateness |
| Domino's tracking | Promise and delivery visibility | Order tracking, routing | C (company) | The promise is a product feature |
| Meituan dispatch | Match orders to couriers | Marketplace algorithms | B/C | Dispatch is a two-sided market design |
| Deliveroo / Just Eat | Courier supply and pricing | Incentives, surge | C/E | Labour and regulatory context decide viability |

### 6.8 Omnichannel and returns

| Case | Decision | Method | Evidence | Lesson |
|---|---|---|---|---|
| Walmart ship-from-store | Fulfil digital orders from stores | Inventory visibility, routing | C (company) | Proximity can beat a dedicated DC |
| Target drive-up | Same-day pickup | Store operations, staging | C (company) | Pickup is a store process, not a website feature |
| Nordstrom integrated inventory | One view of stock | Item master, ATP | C (company) | Visibility precedes routing |
| Zalando returns | Reduce avoidable returns | Fit analytics, reverse logistics | C (company) | Returns are a product and data problem |
| REI used gear | Recover value from returns | Disposition, resale | C (company) | Returns can be a product line |

## 7. Deep dives

### 7.1 M5 — what a benchmark can and cannot tell you

**Context:** Walmart daily sales, 42,840 hierarchical series, 28-day horizon.
**Decision:** Which forecasting approach to use for retail scale.
**Evidence:** Competition results and winning solutions (grade A for the benchmark, C for commercial claims).
**Outcome:** The winning approach combined many LightGBM models with a Tweedie objective and strong features; WRMSSE improved substantially over the seasonal-naive benchmark.
**Transfer:** hierarchy, cross-learning, and promotion features matter.
**Limit:** the dataset is U.S. retail and does not include all costs or constraints of a real planning system.

### 7.2 UPS ORION — value is in the workflow

**Context:** Route optimization across a large delivery fleet.
**Decision:** Whether to adopt optimized routes and how to let drivers deviate.
**Evidence:** Company claims of mileage and cost savings (grade C).
**Outcome:** A multi-year deployment with measurable savings reported by the company.
**Transfer:** optimization value depends on dispatcher and driver adoption.
**Limit:** the exact savings are not independently audited and depend on network density.

### 7.3 Zillow Offers — when a model carries inventory risk

**Context:** Algorithmic home buying at scale.
**Decision:** Which homes to buy and at what price.
**Evidence:** Company disclosures and public reporting (grade B/C).
**Outcome:** A large write-down and the wind-down of the business.
**Transfer:** model error can become capital loss; stress testing and human review matter.
**Limit:** housing is not a typical supply chain, but the risk pattern is general.

### 7.4 Red Sea diversion — a live stress test

**Context:** Attacks on shipping in the Red Sea diverted vessels around the Cape of Good Hope from late 2023.
**Decision:** Reroute, absorb delay, or pay for alternatives.
**Evidence:** Freight-rate and transit data, carrier advisories (grade B).
**Outcome:** Longer transit times, higher rates, and inventory buffer pressure.
**Transfer:** lead-time variability is a first-class planning input.
**Limit:** the duration and magnitude are specific to the event.

### 7.5 Quick commerce — why density decides the answer

**Context:** Several dark-store operators expanded rapidly in 2021–2023, then contracted or exited markets.
**Decision:** Where to place dark stores, what promise to offer, and how large a basket to target.
**Evidence:** Company statements and press reporting (grade C/E); audited unit economics are rarely disclosed.
**Outcome:** Operators with dense demand and larger baskets persisted; others closed stores or markets.
**Transfer:** the break-even is driven by orders per store per day, basket size, courier cost, and waste, not by the promise alone.
**Limit:** the market and regulatory context differ by country, and private-company economics are not verifiable.

### 7.6 Omnichannel — the store becomes a node

**Context:** Large retailers with dense store networks added ship-from-store and pickup.
**Decision:** Which stores fulfil which orders, and how to protect the in-store experience.
**Evidence:** Company disclosures and investor materials (grade C).
**Outcome:** Store-based fulfilment shortened delivery times and reduced some delivery cost, while adding store labour and accuracy risk.
**Transfer:** proximity is an asset only when inventory accuracy, labour capacity, and process discipline exist.
**Limit:** the reported speed and cost gains are company claims and depend on store format and market density.

### 7.7 Pharmaceutical cold chain — validation is the product

**Context:** Vaccines, biologics, and many injectables must stay within a qualified temperature range from manufacture to administration.
**Decision:** How to design lanes, packaging, monitoring, and excursion response so that product remains usable and auditable.
**Evidence:** Regulatory guidance and pharmacopoeial standards (grade B); company performance data is rarely public.
**What the domain adds:**
- The product's quality is a function of its **time–temperature history**, not only its location. A shipment that arrives on time but has an unexcused excursion is a failure.
- **Qualification** (transport validation, thermal mapping, lane risk assessment) precedes routine analytics. A model that recommends a cheaper lane must fit inside a validated envelope.
- **Excursion management** is a decision problem: quarantine, assess mean kinetic temperature, consult the manufacturer, release or destroy. The data needed is the logger record, the stability budget, and the disposition authority.
- **Data integrity** rules (ALCOA+ principles) mean the analytics pipeline itself may be audited. An untraceable transformation is a compliance defect, not only a reproducibility one.

**Transfer:** in any regulated domain, the measurement system and the decision record are part of the product. This is the strongest counter-example to the "move fast" analytics playbook.
**Limit:** the specific stability budgets and release rules are product- and regulator-specific and cannot be generalized from a case note.

## 8. Labs

### Lab 28.1 — Case note

Write a one-page case note for a case in your sector, using the template and an explicit evidence grade.

**Deliverable:** Case note and source list.

### Lab 28.2 — Transfer test

Take a case from another sector and apply the five-question transfer test. Recommend whether to reuse the approach.

**Deliverable:** Transfer memo.

### Lab 28.3 — Sector playbook

Build a two-page playbook for one sector: decisions, data, methods, regulation, cases, measurement.

**Deliverable:** Playbook.

### Lab 28.4 — Counter-case

Find a failure or contradicting case for a popular success story. Explain what it changes.

**Deliverable:** Counter-case note.

### Lab 28.5 — Teaching case

Turn a case into a 20-minute teaching session with discussion questions and a decision exercise.

**Deliverable:** Session plan and slides.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Case accuracy | 20% | Summary from memory | Primary source and dates | Conflicting sources reconciled |
| Evidence grading | 20% | Treats claims as facts | Labels source type | Distinguishes claim from proof |
| Decision framing | 20% | Describes company | Names decision and owner | Explains constraints and alternatives |
| Transfer analysis | 20% | Assumes it generalizes | Lists differences | Tests and rejects where needed |
| Communication | 10% | Long summary | Clear note | Decision-ready and teachable |
| Source hygiene | 10% | No sources | Links and dates | Traceable and re-checkable |

## 10. Regulatory, ethical, and commercial lens

- **Defamation and accuracy:** case notes about real firms must be factual and sourced.
- **Confidentiality:** do not use non-public information from an employer.
- **Survivorship:** success stories omit failures; say so.
- **Labor and safety:** cases about productivity must include worker impact.
- **Sustainability claims:** treat green claims as claims until measured.
- **Competition:** pricing cases must respect antitrust boundaries.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Hagiography | Only successes | Selection bias | Wrong lessons | Add failures |
| Vendor case study | No independent source | Marketing | Overstated benefit | Grade and caveat |
| Causal overreach | "They did X and grew" | Confounding | Bad decision | Ask for counterfactual |
| Context blindness | "Amazon does it" | Scale and capital differ | Failed copy | Transfer test |
| Stale case | Pre-2020 numbers | Regime change | Wrong benchmark | Date every claim |
| Single-source | One article | No verification | Error | Two sources |
| Metric mismatch | Revenue vs margin | Wrong objective | Misleading | Define metric |
| Ethics omission | No labor or environment | Narrow framing | Harm | Add stakeholder lens |
| Confidential leak | Internal data | No permission | Legal risk | Use public sources |
| Teaching without decision | Story time | No exercise | Low learning | Add decision prompt |

## 12. Interview and discussion questions

1. What makes a case useful? *(Hint: decision, evidence, transfer.)*
2. How do you grade evidence? *(Hint: A–E ladder.)*
3. Why do successful cases mislead? *(Hint: survivorship and selection.)*
4. How would you test whether a case transfers? *(Hint: five questions.)*
5. What is the counterfactual in a company case? *(Hint: what would have happened otherwise.)*
6. How do you use a case in an interview? *(Hint: decision, method, evidence, lesson.)*
7. Which case changed your view of forecasting? *(Hint: M5, Zillow, COVID.)*
8. What case would you use to teach inventory? *(Hint: newsvendor, HP DeskJet.)*
9. How do you handle conflicting sources? *(Hint: primary source and date.)*
10. What is a sector playbook? *(Hint: decisions, data, methods, regulation, cases.)*
11. Which case is most overused? *(Hint: Amazon, Toyota; explain limits.)*
12. What would you add to the library? *(Hint: failure, small firm, non-English.)*

## 13. Key terms

| Term | Definition |
|---|---|
| Case anatomy | Consistent structure for a case note |
| Evidence grade | A–E rating of source strength |
| Transfer test | Five questions about applicability |
| Survivorship bias | Only successes are visible |
| Selection bias | Cases are not a random sample |
| Counterfactual | What would have happened otherwise |
| Primary source | Original paper, filing, or regulator |
| Secondary source | Summary or press report |
| Vendor claim | Marketing assertion, not evidence |
| Sector playbook | Decision-oriented summary for a sector |
| Benchmark | Standard dataset or result for comparison |
| Confounding | Another factor explains the outcome |
| Attribution | Share of outcome caused by intervention |
| Counter-case | A failure or contradiction |
| Decision owner | Person accountable for the choice |
| Case library | Organized collection of sourced cases |

## 14. Sources and further reading

### Case sources and standards

- ASCM, SCOR Digital Standard. https://www.ascm.org/
- GS1, EPCIS 2.0. https://www.gs1.org/standards/epcis
- ISO 22301. https://www.iso.org/standard/75106.html
- ISO 31000. https://www.iso.org/standard/65694.html
- U.S. SEC, EDGAR filings. https://www.sec.gov/edgar/
- U.S. GAO reports. https://www.gao.gov/

### Peer-reviewed and professional

- Lee, H. L., Padmanabhan, V., Whang, S., "Information Distortion in a Supply Chain," *Management Science*, 1997.
- Lee, H. L., Billington, C., "The Evolution of Supply-Chain-Management Models and Practice at Hewlett-Packard," *Interfaces*, 1995.
- Cachon, G., Terwiesch, C., *Matching Supply with Demand*.
- Sheffi, Y., *The Resilient Enterprise*.
- Chopra, S., Meindl, P., *Supply Chain Management*.
- Simchi-Levi, D. et al., *Designing and Managing the Supply Chain*.

### Industry and public data

- Kaggle, M5 Forecasting — Accuracy and Uncertainty. https://www.kaggle.com/competitions/m5-forecasting-accuracy
- UPS, ORION materials. https://www.ups.com/
- Maersk, public statements and annual reports. https://www.maersk.com/
- Zillow Group, shareholder letters. https://investors.zillowgroup.com/
- FDA, DSCSA. https://www.fda.gov/drugs/drug-supply-chain-security-act-dscsa
- UN Comtrade. https://comtradeplus.un.org/
- World Bank LPI. https://lpi.worldbank.org/

> **Next:** [Module 29](29_industry_electives.md) goes deeper into six sector primers and their decision contexts.
