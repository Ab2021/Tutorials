# Module 14 — Procurement, Supplier Risk, and Spend Analytics

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 30 hours (12 lecture, 10 lab, 8 project)
> **Track fit:** Manufacturing & Industrial Operations core; recommended for Retail & E-commerce and Logistics & Distribution
> **Prerequisites:** Module 02 (master data); Module 03 (entity resolution and joins); Module 04 (causal reasoning); basic cost accounting
> **Tools:** Python 3.12, pandas, DuckDB, rapidfuzz, NetworkX, matplotlib, PuLP or OR-Tools for award optimization; optional LLM extraction for contract text
> **Industry question:** "Who are we really buying from, what are we really paying, and which of these suppliers can take the business down with them?"

---

## 1. Why industry cares

Procurement is where the supply chain commits its money. A typical manufacturer spends 60–70% of revenue with external suppliers, so a 2% improvement in purchased cost is worth more than a 5% improvement in internal operating cost. Yet procurement data is the messiest in the enterprise: supplier names are free text, item descriptions are inconsistent, contracts sit in PDFs, and the same part is bought under four different part numbers by three business units.

Four facts define the 2026 context.

1. **Regulation moved from disclosure to due diligence.** The EU Corporate Sustainability Due Diligence Directive (CSDDD) was amended by Omnibus I, in force 18 March 2026, with transposition extended to 26 July 2028 and scope narrowed to companies above 5,000 employees and €1.5 billion turnover. The EU Carbon Border Adjustment Mechanism entered its definitive regime on 1 January 2026 with a 50-tonne mass threshold. Supplier data is now a compliance input, not a nice-to-have.
2. **Concentration risk is measurable.** Supplier fragmentation looks efficient until a single tier-2 site, a single port, or a single smelter stops. Multi-tier mapping moved from a consulting exercise to a data product.
3. **Tail spend is where the leakage lives.** Usually 80% of suppliers carry under 20% of spend, and that tail is where maverick buying, duplicate vendors, and invoice fraud concentrate.
4. **LLMs made contract data accessible but not trustworthy by default.** Extraction is now cheap; verification is not. A contract clause extracted with 95% accuracy still produces 5% wrong obligations at scale.

The cost of getting this wrong is concrete: a should-cost gap left on the table, a single-source supplier that fails, a Scope 3 number that fails assurance, or a duplicate-payment stream that runs for years because nobody reconciled the vendor master.

## 2. Learning outcomes

By the end of this module, you can:

- Map the source-to-pay process and identify the data created at each step.
- Build a spend cube from accounts-payable and purchase-order data, including supplier normalization and category classification.
- Apply UNSPSC and GS1 GPC hierarchies and explain where an internal taxonomy is better.
- Segment a supplier portfolio on the Kraljic matrix and choose a sourcing strategy per quadrant.
- Build a should-cost model and a total cost of ownership model, and reconcile them to the quoted price.
- Formulate a sourcing award as a mixed-integer program with capacity, quality, and volume constraints.
- Design and critique a reverse auction, including qualification, reserve price, and post-award risk.
- Build a weighted supplier scorecard and test its ranking for weight sensitivity.
- Assess supplier financial distress and operational risk with transparent, documented indicators.
- Map multi-tier supply relationships and compute concentration and single-source exposure.
- Design contract and compliance data extraction with a verification sample.
- Calculate Scope 3 category 1 emissions from spend and activity data, and state the uncertainty.
- Detect duplicate payments and vendor-master anomalies with reproducible rules.

## 3. Mental model

### 3.1 Source-to-pay

```
need ──► requisition ──► sourcing ──► contract ──► PO ──► receipt ──► invoice ──► payment
  │          │             │            │          │        │           │          │
spec      approval      RFQ/auction   terms,     price,   GRN,      3-way      terms,
          budget        award         SLA        qty      quality    match      discount
  │          │             │            │          │        │           │          │
  └──────────┴─────────────┴────────────┴──────────┴────────┴───────────┴──────────┘
                     every step writes data that procurement analytics consumes
```

The analytical assets sit at the joins: requisition to contract (was this bought under a contract?), PO to invoice (did we pay the contracted price?), and invoice to vendor master (is this a real, unique supplier?).

### 3.2 The three procurement questions

| Question | Artifact | Method |
|---|---|---|
| What are we buying, and from whom? | Spend cube | Normalize, classify, aggregate |
| Is the price right? | Should-cost and TCO | Cost build-up, landed cost, lifecycle cost |
| What could go wrong? | Risk scorecard | Financial, operational, geographic, compliance |

### 3.3 The Kraljic matrix

| | Low supply risk | High supply risk |
|---|---|---|
| **High profit impact** | Leverage: compete, aggregate volume, auction | Strategic: partner, joint planning, dual source |
| **Low profit impact** | Non-critical: simplify, catalogue, automate | Bottleneck: secure supply, qualify alternates, buffer |

The quadrant determines the tactic, not the category name. A "strategic" label on a low-spend item is a common self-importance error; the axes are profit impact and supply risk, both measured.

## 4. Core concepts

### 4.1 Spend cube construction

A spend cube has three dimensions: **category** (what), **supplier** (who), and **business unit or time** (where/when). Building it is 70% data work and 30% analysis.

| Step | Action | Failure if skipped |
|---|---|---|
| 1. Extract | Pull AP invoices, PO lines, and payment terms | Invoice-only view misses unissued POs and accruals |
| 2. Normalize suppliers | Fuzzy match names, tax IDs, addresses, bank accounts | Duplicate vendors inflate supplier count and hide concentration |
| 3. Classify | Map descriptions to UNSPSC or internal taxonomy | "Miscellaneous" becomes the largest category |
| 4. Adjust | Exclude taxes, intercompany, rebates; convert currency at a stated rate | Double-counted spend and distorted shares |
| 5. Attribute | Assign business unit, cost center, contract status | No accountability for maverick buying |
| 6. Validate | Reconcile to GL and to the CPO report | The cube disagrees with finance and loses credibility |

**Supplier normalization** is the highest-value step. Match on normalized name (strip legal suffixes, punctuation, stop words), tax identifier, address, and bank account. Use blocking and fuzzy scoring with a human review queue above a threshold. A vendor master with 12,000 records often collapses to 7,000 real entities.

**Contract coverage** is the killer metric: the share of spend with a linked contract. Anything below roughly 70% means the category strategy is not reaching the transaction.

### 4.2 Worked spend cube

Ten invoices from an AP extract, normalized and classified. Total spend $4,804,500.

| Raw supplier | Description | Amount | UNSPSC segment | Category |
|---|---|---:|---|---|
| ACME INDUSTRIAL SUPPLY INC | M8 hex bolts, zinc | $412,000 | Fasteners | Direct |
| Acme Industrial Supply | Nitrile safety gloves | $96,500 | Safety apparel | Indirect |
| PACIFIC LOGISTICS LLC | Ocean freight Shanghai–LA | $1,840,000 | Freight transport | Logistics |
| Pacific Logistics LLC | Detention charges | $74,200 | Freight transport | Logistics |
| NORDIC PACKAGING AB | Corrugated cartons 60×40 | $638,000 | Cartons | Packaging |
| NORDIC PACKAGING AB | Pallet wrap | $88,400 | Packaging film | Packaging |
| GLOBAL IT SOLUTIONS GMBH | Managed service desk, 12 months | $1,260,000 | IT services | IT |
| GLOBAL IT SOLUTIONS GMBH | Docking stations | $143,600 | Computer accessories | IT |
| SUNRISE FACILITIES | Janitorial services Q3 | $212,000 | Cleaning services | Facilities |
| SUNRISE FACILITIES | HVAC filter replacement | $39,800 | Filters | MRO |

Roll-up:

| Category | Spend | Share | Cumulative |
|---|---:|---:|---:|
| Logistics | $1,914,200 | 39.8% | 39.8% |
| IT | $1,403,600 | 29.2% | 69.0% |
| Packaging | $726,400 | 15.1% | 84.1% |
| Direct (fasteners) | $412,000 | 8.6% | 92.7% |
| Facilities | $212,000 | 4.4% | 97.1% |
| PPE | $96,500 | 2.0% | 99.1% |
| MRO | $39,800 | 0.8% | 100.0% |

Arithmetic check: $1,914,200 + $1,403,600 + $726,400 + $412,000 + $212,000 + $96,500 + $39,800 = $4,804,500. Logistics share = 1,914,200 ÷ 4,804,500 = 39.8%.

**Tail spend:** categories under $250,000 (Facilities, PPE, MRO) total $348,300, or 7.2% of spend. In a real cube, the tail is usually 10–20% of spend spread across 70–80% of suppliers, and it is where duplicate vendors and maverick buying hide. Two suppliers in this extract appear under two spellings, which is exactly how a vendor master doubles in size.

### 4.3 Classification: UNSPSC, GS1 GPC, and internal taxonomies

| System | Structure | Strengths | Weaknesses |
|---|---|---|---|
| UNSPSC | Four levels: segment, family, class, commodity (8 digits) | Cross-industry, free, widely mapped | Generic; poor granularity for direct materials |
| GS1 GPC | Segment, family, class, brick | Retail and consumer goods alignment | Not built for industrial indirect spend |
| Internal taxonomy | Designed for the category strategy | Actionable at the level of a sourcing lever | Not comparable across companies |
| eCl@ss | Class and property structure | Strong for technical attributes | Mapping effort |

UNSPSC is the default for a first cube because it is free and cross-industry. Its weakness is that it classifies what a thing *is*, not how it is *bought*: two items in the same commodity code may sit in different Kraljic quadrants. Most mature organizations maintain UNSPSC for reporting plus an internal category tree for strategy, with a crosswalk maintained as a governed asset.

Classification is a text problem with a long tail. Rules handle the top 60–70% of spend; a classifier handles the rest; a human reviews a sample. Measure accuracy by spend-weighted agreement, not by row count: getting 95% of rows right while missing the three largest categories is a failure.

### 4.4 Kraljic segmentation and sourcing strategy

| Quadrant | Spend profile | Supply risk | Strategy | Analytics |
|---|---|---|---|---|
| Strategic | High | High | Partnership, joint planning, long contracts, dual source | Total cost, supplier financials, multi-tier mapping |
| Leverage | High | Low | Compete, aggregate volume, reverse auction | Price benchmarking, bid analysis, should-cost |
| Bottleneck | Low | High | Secure supply, qualify alternatives, buffer | Lead-time distribution, single-source exposure |
| Non-critical | Low | Low | Simplify, catalogue, automate, consolidate | Tail-spend analytics, maverick buying |

Score the axes explicitly. Profit impact: spend, revenue exposure, margin sensitivity. Supply risk: number of qualified sources, lead time, substitutability, geographic concentration, supplier financial health. A 1–5 score per factor with published weights beats a subjective placement.

### 4.5 Should-cost and total cost of ownership

**Should-cost** builds the price from the supplier's cost structure:

```
should-cost = material + conversion + overhead + SG&A + profit
```

Worked example for a machined component:

| Element | Basis | Amount |
|---|---|---:|
| Material | 2.1 kg at $2.90/kg | $6.09 |
| Conversion | 0.8 machine-hours at $30/h | $24.00 |
| Overhead | 12% of conversion | $2.88 |
| SG&A | 8% of subtotal | $2.64 |
| Profit | 10% of subtotal | $3.56 |
| **Should-cost** | | **$39.17** |

A quote of $46.00 is 17.5% above should-cost. The gap is a negotiation agenda, not a conclusion: the supplier may have higher scrap, smaller volume, or a capability premium.

**TCO** adds the costs of owning, not just buying:

| Cost element | Offshore | Domestic |
|---|---:|---:|
| Unit price (FOB / ex-works) | $12.40 | $14.05 |
| Freight per unit | $0.62 | $0.10 |
| Insurance (0.4% of FOB) | $0.05 | — |
| Duty (6.5% of CIF) | $0.85 | — |
| Port handling and inland | $0.40 | — |
| Landed cost | $14.32 | $14.15 |
| Inventory carrying (22%/yr × transit days) | $0.39 (45 days) | $0.04 (5 days) |
| **TCO per unit** | **$14.71** | **$14.19** |

Arithmetic check: CIF = 12.40 + 0.62 + 0.05 = $13.07; duty = 6.5% × 13.07 = $0.85; landed = 13.07 + 0.85 + 0.40 = $14.32. Carrying = 22% × (45/365) × 14.32 = $0.39. The domestic option wins by $0.51 per unit, or $5,149 on 10,000 units — before quality, obsolescence, and payment-term effects.

TCO must also include quality cost (inspection, scrap, warranty), switching cost, and the cost of managing the relationship. A model that stops at landed cost systematically favors distant suppliers.

### 4.6 Sourcing optimization

Award allocation is a mixed-integer program:

```
minimise   Σ_i Σ_j (p_ij · x_ij) + fixed-cost terms
subject to Σ_j x_ij = d_i                    demand satisfied
           Σ_i x_ij ≤ capacity_j             supplier capacity
           x_ij ≥ m_ij · y_j                 minimum volume if awarded
           x_ij ≤ M_ij · y_j                 maximum volume if awarded
           quality and lead-time constraints
           x_ij ≥ 0, y_j ∈ {0,1}
```

| Constraint type | Why it matters | Common modelling error |
|---|---|---|
| Capacity | Prevents an award the supplier cannot deliver | Ignoring ramp-up; capacity is not instant |
| Minimum volume | Suppliers price on volume | Allowing 1% awards that no supplier will accept |
| Volume discount | Tiered pricing changes the optimum | Linearizing incorrectly and overstating savings |
| Incumbent switching cost | Qualification takes time | Treating a new supplier as available on day one |
| Quality and lead time | Price is not the only award criterion | Scoring quality separately then ignoring it in the model |
| Risk concentration | Avoid over-award to one supplier | Optimizing cost into a single-source outcome |

Run the model with and without a concentration constraint and show the cost of resilience explicitly. That difference is the number the CPO needs.

### 4.7 Auctions and negotiations

| Format | Mechanism | Best for | Risk |
|---|---|---|---|
| English reverse auction | Bidders see rank, price falls | Commoditized, qualified supply base | Late price shaving, quality erosion |
| Japanese reverse auction | Price falls until one bidder remains | Simple, homogeneous items | Fewer signals; abrupt |
| Sealed-bid | One shot, no feedback | Limited competition, complex specs | Winner's curse |
| Dutch | Price falls, first acceptor wins | Time-sensitive purchases | Strategic waiting |

A reverse auction needs: a qualified bidder pool (at least three credible), a clear specification, a should-cost-derived reserve price, and a total-cost evaluation that includes logistics and quality. Without those, the auction selects for optimism, and the winner discovers the job is unprofitable — then recovers through change orders.

Negotiation analytics: prepare the walk-away, the should-cost, the supplier's likely cost position, and the value of non-price terms (payment days, volume flexibility, lead time, warranty). A 1% price reduction and 30 extra payment days are not equivalent; convert both to cash.

### 4.8 Supplier scorecards

| Criterion | Weight | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|---:|
| Quality (PPM, returns) | 25% | 4 | 5 | 3 | 4 | 5 |
| Delivery (OTIF) | 25% | 5 | 3 | 4 | 4 | 4 |
| Cost competitiveness | 20% | 3 | 4 | 5 | 2 | 1 |
| Responsiveness | 10% | 4 | 3 | 4 | 5 | 5 |
| ESG and compliance | 10% | 4 | 4 | 3 | 4 | 3 |
| Innovation | 10% | 3 | 4 | 3 | 5 | 3 |
| **Weighted score (out of 100)** | **100%** | **79.0** | **78.0** | **75.0** | **76.0** | **71.0** |

Arithmetic check for supplier A: `(4×25 + 5×25 + 3×20 + 4×10 + 4×10 + 3×10) ÷ 5 = 395 ÷ 5 = 79.0`.

The ranking is fragile. Raise the cost weight to 30% and cut quality to 15%: supplier C rises to 79.0 while A falls to 77.0 and the winner changes. **Always report the weights and a sensitivity run**; a scorecard without a sensitivity analysis is an opinion with arithmetic.

```python
"""Scorecard, weight sensitivity, and concentration — pure Python, no dependencies."""
WEIGHTS = {"quality": .25, "delivery": .25, "cost": .20,
           "responsiveness": .10, "esg": .10, "innovation": .10}
SCORES = {
    "A": {"quality": 4, "delivery": 5, "cost": 3, "responsiveness": 4, "esg": 4, "innovation": 3},
    "B": {"quality": 5, "delivery": 3, "cost": 4, "responsiveness": 3, "esg": 4, "innovation": 4},
    "C": {"quality": 3, "delivery": 4, "cost": 5, "responsiveness": 4, "esg": 3, "innovation": 3},
    "D": {"quality": 4, "delivery": 4, "cost": 2, "responsiveness": 5, "esg": 4, "innovation": 5},
    "E": {"quality": 5, "delivery": 4, "cost": 1, "responsiveness": 5, "esg": 3, "innovation": 3},
}

def weighted(scores, weights):
    """Scores are 1-5; return a 0-100 weighted score."""
    assert abs(sum(weights.values()) - 1.0) < 1e-9, "weights must sum to 1"
    return {k: round(sum(v[c] * weights[c] for c in weights) / 5 * 100, 1)
            for k, v in scores.items()}

base = weighted(SCORES, WEIGHTS)
cost_heavy = weighted(SCORES, {**WEIGHTS, "quality": .15, "cost": .30})

print("base        ", dict(sorted(base.items(), key=lambda x: -x[1])))
print("cost-heavy  ", dict(sorted(cost_heavy.items(), key=lambda x: -x[1])))
print("rank changed:", max(base, key=base.get) != max(cost_heavy, key=cost_heavy.get))

shares = [45, 25, 15, 10, 5]                 # percent of category spend
hhi = sum(s * s for s in shares)             # 0-10,000 scale
print(f"HHI {hhi}, largest supplier {max(shares)}%")
```

Output: base ranking A 79.0, B 78.0, D 76.0, C 75.0, E 71.0; cost-heavy ranking C 79.0, A 77.0, B 76.0, D 72.0, E 63.0; the rank changed; HHI 3,000 with a 45% largest supplier.

Scorecards must also distinguish **capability** (can they do it) from **performance** (did they do it). A supplier with a perfect quality score on a low-volume, simple part is not proven on a complex one.

### 4.9 Supplier risk: financial and operational

| Risk type | Indicators | Data source | Caveat |
|---|---|---|---|
| Financial distress | Altman Z-score, interest coverage, days payable, credit rating, payment behaviour | Financial statements, credit bureau, D&B | Private suppliers disclose late and partially |
| Liquidity | Cash ratio, revolver availability | Filings, credit reports | Seasonal |
| Operational | OTIF trend, capacity utilization, backlog, lead-time variance | ERP, supplier portal | Self-reported |
| Quality | PPM, deviations, audit findings | QMS | Under-reporting |
| Concentration | Share of our spend, share of their revenue | Spend cube, supplier data | Their revenue is often estimated |
| Geographic | Site location vs hazard exposure | Supplier master, GIS | Tier-2 unknown |
| Compliance | Sanctions, forced labour, conflict minerals | Screening providers, Open Supply Hub | Coverage gaps |

The Altman Z-score for a public manufacturer: `Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 1.0X5`, where X1 is working capital/assets, X2 retained earnings/assets, X3 EBIT/assets, X4 market value of equity/liabilities, X5 sales/assets. The classic cut-offs (above 2.99 safe, below 1.81 distress) were derived on a 1960s US manufacturing sample; use them as a screening signal, not a verdict, and never apply them to a private company without adjusting X4.

Financial risk scoring must be **explainable**. A black-box score that a category manager cannot interrogate will be ignored or gamed. Publish the factors, their weights, and the evidence date.

### 4.10 Multi-tier mapping and concentration

A tier-1 supplier list is not a supply chain. Multi-tier mapping asks: which tier-2 and tier-3 sites feed this tier-1, and what else depends on them?

Sources: supplier declarations, industry databases (Open Supply Hub), customs and bill-of-lading data, and purchased risk feeds. Each has coverage and accuracy limits; a map is a hypothesis with evidence, not a fact.

Concentration metrics:

```
share_j = spend with supplier j ÷ total category spend
HHI = Σ share_j²  (on a 0–10,000 scale)
```

Worked example, five suppliers with 45%, 25%, 15%, 10%, 5%: `HHI = 2,025 + 625 + 225 + 100 + 25 = 3,000`. The US merger guidelines treat an HHI above 1,800 as highly concentrated for market definition; the analogy to a supplier base is useful but not a legal test. Report HHI alongside the largest-supplier share and the number of qualified alternatives, because HHI alone hides a 45% single point of failure behind four small suppliers.

**Single-source exposure** = spend on parts with exactly one qualified source ÷ total spend. Track it by part criticality, not only by dollars.

### 4.11 Contracts and compliance data

| Field | Source | Extraction difficulty |
|---|---|---|
| Effective and expiry dates | Contract | Low |
| Payment terms | Contract | Low |
| Price and escalators | Contract, amendments | Medium |
| Volume commitments | Contract | Medium |
| Service levels and penalties | Contract | High |
| Termination and force majeure | Contract | High |
| Data protection and audit rights | Contract | Medium |
| ESG and code-of-conduct obligations | Contract, supplier code | Medium |

LLM extraction pipeline: parse to text, chunk by clause, extract with a schema, then **verify a stratified random sample of at least 100 clauses per field against the source**. Report precision and recall per field, not an overall accuracy. A missed termination clause is not equivalent to a missed notice address.

Contract data quality issues are predictable: amendments stored separately, scanned PDFs with no text layer, multiple languages, and related agreements that override each other. Build a clause-level provenance link so every extracted value points to a page and paragraph.

### 4.12 ESG and Scope 3

Scope 3 category 1 (purchased goods and services) is usually the largest part of a manufacturer's footprint and the hardest to measure.

| Method | Input | Accuracy | Use |
|---|---|---|---|
| Spend-based | Spend × industry emission factor | Low; ±50% or worse | Screening, prioritization |
| Average-data | Mass or units × commodity factor | Medium | Category baseline |
| Supplier-specific | Supplier activity data × supplier factor | High, if verified | Reporting, target setting |
| Hybrid | Supplier-specific for top categories, estimates elsewhere | Medium–high | Practical default |

Under ESRS E1-6, gross Scope 1, 2, and 3 emissions must be reported by significant category using the GHG Protocol. Phase-in relief allows estimates and proxies for value-chain data in the first three years for some undertakings, but the direction of travel is primary data. The analytic obligation is to label the method per category and quantify the uncertainty; a single blended number presented as measured is misleading.

Supplier ESG scoring should separate **disclosure** (did they report), **performance** (what are the numbers), and **assurance** (was it verified). EcoVadis, CDP, and SBTi are useful inputs; none is a substitute for a site audit on a high-risk category.

### 4.13 Fraud and duplicate payments

| Test | Logic | Typical signal |
|---|---|---|
| Exact duplicate | Same vendor, invoice number, amount, date | Re-submitted invoice |
| Fuzzy duplicate | Same amount within 2%, date within 10 days, similar number | Re-keyed invoice |
| Vendor master duplicate | Same tax ID or bank account, different name | Shell or split vendor |
| Split purchase | Multiple POs just below approval threshold, same requester, short window | Approval circumvention |
| Benford's law | First-digit distribution vs expected | Anomalous manual entries |
| Round-number invoices | High share of exact round amounts | Estimated or fabricated invoices |
| Bank detail change | Vendor bank account changed then payment | Business email compromise |
| PO/ Invoice mismatch | Invoice without receipt, or quantity above received | Payment for undelivered goods |

The Association of Certified Fraud Examiners estimates that organizations lose about 5% of revenue to fraud annually (2024 *Report to the Nations*); treat it as an order-of-magnitude estimate from survey data, not a measurement of any specific firm. [VERIFY: exact figure and sample for the 2024 edition.]

Fraud analytics produces alerts, not proof. Every rule needs a triage owner, a false-positive rate, and a documented investigation path. A rule that fires 10,000 times a month is not a control.

## 5. Data, standards, and workflow

| Data object | Grain | Source | Requirement |
|---|---|---|---|
| Invoice | Invoice line | AP / ERP | Vendor ID, PO, amount, tax, currency, date |
| Purchase order | PO line | ERP | Material, quantity, price, delivery date |
| Goods receipt | Receipt line | WMS / ERP | Quantity received, date, quality status |
| Vendor master | Vendor | ERP / MDM | Legal name, tax ID, bank, address, parent |
| Contract | Contract × clause | CLM / shared drive | Dates, terms, obligations, provenance |
| Supplier profile | Supplier × site | SRM / risk provider | Capacity, certifications, financials |
| Category | Category node | Taxonomy | UNSPSC crosswalk, owner, strategy |
| ESG data | Supplier × year × metric | CDP, EcoVadis, supplier portal | Method, boundary, assurance status |

**Standards.** UNSPSC for category classification; GS1 GPC for retail products; ISO 20400:2017 for sustainable procurement guidance; ISO 37001 for anti-bribery management; ISO 31000 for risk management; the GHG Protocol Corporate Standard and Scope 3 Standard for emissions; ESRS E1-6 for EU sustainability reporting; the EU CSDDD and CBAM regulations for due diligence and carbon border data.

**Data-quality checks.** Currency converted at a documented rate and date; tax excluded consistently; intercompany spend excluded; credit notes netted; vendor duplicates resolved before any concentration metric; contract linkage present for the majority of spend; category assignment reviewed by the category owner.

**Workflow.** Build the cube, reconcile to the GL, publish category profiles, then run sourcing and risk analytics on the reconciled base. A cube that does not tie to finance is a debate, not a dataset.

## 6. Industry case studies

### 6.1 Boeing 787 Dreamliner outsourcing (2003–2011)

**Context.** Boeing designed the 787 with a tier-1 partner model in which major sections were built by suppliers and integrated by Boeing, with roughly 70% of the aircraft outsourced — a much higher share than prior programmes.

**Intervention.** Global partners designed and built large integrated sections, shifting engineering and integration work to the supply base, with the expectation of faster development and lower cost.

**Evidence.** The programme ran more than three years late, with repeated schedule slips; Boeing publicly acknowledged supplier coordination and integration problems, and the US Government Accountability Office reported on the programme's challenges. Boeing later acquired the South Carolina facility of a key partner and brought work back in-house.

**Outcome.** Significant cost overruns and delays, followed by vertical integration of critical work and a redesign of the partner model for later programmes.

**What transferred.** The should-cost and capability question: outsourcing a module does not outsource the integration risk, and a tier-1 partner's capacity and maturity must be assessed as rigorously as a direct supplier's.

**What did not.** The conclusion that outsourcing is bad. The failure was in governance, requirements maturity, and partner capability assessment, not in the make-or-buy principle.

### 6.2 Volkswagen and the Prevent supplier dispute (2016)

**Context.** Volkswagen relied on a small number of specialist suppliers for certain seat-cover and casting components, with limited qualified alternatives.

**Intervention.** A commercial dispute with the supplier group Prevent led to a halt in component deliveries, and Volkswagen suspended production at several German plants for a period in August 2016.

**Evidence.** Reported extensively in the German and international business press; both parties confirmed the dispute and the production interruption. [VERIFY: exact dates, plants, and settlement terms before reuse.]

**Outcome.** Production resumed after a settlement; the episode became a reference case for single-source exposure in a just-in-sequence environment.

**What transferred.** Concentration risk is not only a financial-health question. A commercially healthy supplier in a dispute can stop a line as effectively as an insolvent one. Contract terms, escalation clauses, and qualified alternates are the mitigations.

**What did not.** The implication that dual sourcing always pays. For engineered, tooling-intensive components, qualification cost can exceed the expected disruption cost; the right answer is a tested contingency, not necessarily a second source in serial production.

### 6.3 Apple's supplier responsibility programme (2005–2025)

**Context.** Apple's manufacturing is almost entirely outsourced, exposing it to labour, environmental, and conflict-minerals risk in its supply base.

**Intervention.** Apple publishes an annual Supplier List and Supplier Responsibility Report, runs supplier audits, requires smelter and refiner participation in independent third-party audit programmes for conflict minerals, and operates a supplier clean-energy programme.

**Evidence.** Company-published reports with audit counts, findings, and remediation statistics. The underlying audit data are not independently reproducible, and independent investigators have challenged specific findings. [VERIFY: current report year and specific percentages.]

**Outcome.** A large, sustained supplier-compliance apparatus and measurable movement on smelter participation and supplier renewable energy.

**What transferred.** The operating model: a public supplier list, a standardized audit, a remediation loop, and a small set of metrics tracked year over year.

**What did not.** The assurance level. Self-published audit statistics are a management claim; for a regulated disclosure, the data need external verification and a stated boundary.

### 6.4 Unilever's sustainable sourcing programme (2010–2025)

**Context.** Unilever's Sustainable Living Plan committed to sourcing agricultural raw materials sustainably, covering commodities such as palm oil, tea, cocoa, and soy.

**Intervention.** Certification through third-party standards (RSPO for palm oil, Rainforest Alliance, Fairtrade), supplier engagement, and smallholder programmes, with annual reporting against a target.

**Evidence.** Company-reported sourcing percentages and progress against the commitment; the definition of "sustainably sourced" includes both certified and independently verified volumes, which critics argue is broader than certification alone.

**Outcome.** A large share of agricultural volume sourced to the company's standard, with continued debate about the difference between certification and impact.

**What transferred.** The measurement architecture — commodity-level targets, certification as a data field, and a public progress report — is a workable model for Scope 3 and responsible sourcing programmes.

**What did not.** The inference that certification equals outcome. Certification audits process, not deforestation or livelihood outcomes; impact requires separate measurement.

### 6.5 Kraljic's portfolio matrix (1983)

**Context.** Purchasing was managed transactionally, with little segmentation by strategic importance.

**Intervention.** Kraljic's *Harvard Business Review* article "Purchasing Must Become Supply Management" proposed segmenting purchased items by profit impact and supply risk, with a distinct strategy per quadrant.

**Evidence.** A conceptual framework grounded in observation of industrial purchasing practice, not a controlled study. It has been widely adopted and critiqued for the difficulty of scoring the axes objectively.

**Outcome.** The dominant segmentation framework in procurement for four decades.

**What transferred.** The core insight: one sourcing strategy cannot serve a portfolio, and the strategy should follow the risk-value position of the item.

**What did not.** The implication that the matrix is a one-time exercise. Quadrant position changes with market structure and supplier health, so the segmentation must be refreshed.

## 7. Labs

### Lab 14.1 — Build a spend cube

**Objective.** Turn a raw AP extract into a reconciled, classified spend cube.

**Data.** 24 months of invoice lines (50,000–200,000 rows) with vendor name, description, amount, currency, cost center, and PO reference.

**Steps.** (1) Normalize vendor names and deduplicate using name, tax ID, and bank account. (2) Convert currency at a documented rate. (3) Classify to a two-level taxonomy using rules plus a text classifier. (4) Measure spend-weighted classification accuracy on a labelled sample. (5) Roll up by category, supplier, and business unit. (6) Reconcile the total to the GL extract. (7) Compute tail spend, contract coverage, and supplier count.

**Deliverable.** A spend cube workbook or DuckDB database plus a validation note.

**Acceptance criteria.** Reconciliation within 1% of GL; duplicate vendors resolved with a documented rule; classification accuracy reported spend-weighted; tail spend and contract coverage computed.

**Stretch.** Add a maverick-buying analysis: spend with suppliers that have no contract and no PO.

### Lab 14.2 — Kraljic segmentation and scorecard

**Objective.** Segment a supplier portfolio and build a weighted scorecard with sensitivity.

**Data.** Lab 14.1 cube plus supplier performance data (PPM, OTIF, lead time) and financial indicators.

**Steps.** (1) Score profit impact and supply risk with published factors. (2) Place suppliers in quadrants and assign a strategy per quadrant. (3) Build a six-criterion scorecard with weights. (4) Rank suppliers and run three alternative weight sets. (5) Report rank reversals and explain which decision they affect. (6) Recommend a sourcing action per quadrant.

**Deliverable.** A segmentation map, a scorecard, and a two-page sourcing recommendation.

**Acceptance criteria.** Axis factors and weights stated; quadrant assignment reproducible; scorecard arithmetic correct; sensitivity run shows at least one rank reversal; recommendations tied to quadrant.

**Stretch.** Add a risk overlay from Lab 14.4 and identify suppliers that are both strategic and financially distressed.

### Lab 14.3 — Should-cost and TCO

**Objective.** Build a should-cost model and a TCO comparison, and use both in a negotiation brief.

**Data.** Course-provided cost breakdowns for three components, quoted prices, freight and duty rates, payment terms, and quality history.

**Steps.** (1) Build should-cost from material, conversion, overhead, SG&A, and profit. (2) Compare with quote and quantify the gap. (3) Build landed cost under two Incoterms. (4) Add inventory carrying, quality, and switching cost. (5) Compute TCO per unit and per year. (6) Write the negotiation brief with a target, a walk-away, and the non-price levers.

**Deliverable.** A should-cost model, a TCO comparison, and a one-page negotiation brief.

**Acceptance criteria.** Cost build-up transparent; Incoterm effects correct; carrying cost and quality cost included; target and walk-away justified by the model.

**Stretch.** Add a supplier financial constraint and show how payment terms change the supplier's effective price.

### Lab 14.4 — Supplier risk score and multi-tier map

**Objective.** Build an explainable supplier risk score and map concentration.

**Data.** Supplier financials (or proxies), spend by supplier and part, site locations, and a tier-2 relationship list.

**Steps.** (1) Compute financial indicators and a transparent score. (2) Compute operational indicators from delivery and quality data. (3) Compute HHI and single-source exposure by category. (4) Build a tier-2 graph in NetworkX and compute node criticality. (5) Identify the top ten exposures and the mitigation for each. (6) State the evidence quality for every input.

**Deliverable.** A risk scorecard, a network map, and a top-ten exposure register.

**Acceptance criteria.** Every indicator sourced and dated; score explainable; HHI arithmetic shown; single-source exposure computed by part criticality; evidence quality labelled.

**Stretch.** Add a scenario in which the largest tier-2 site is unavailable for eight weeks and estimate the revenue at risk.

### Lab 14.5 — Duplicate payment and vendor anomaly detection

**Objective.** Build reproducible rules that find recoverable duplicate payments and vendor-master anomalies.

**Data.** 12 months of invoice and payment data with vendor master, bank details, and approval thresholds.

**Steps.** (1) Implement exact and fuzzy duplicate tests. (2) Test vendor-master duplicates on tax ID and bank account. (3) Test split purchases below approval thresholds. (4) Run a Benford first-digit test on manual invoices. (5) Estimate recoverable value and false-positive rate per rule. (6) Design a triage workflow with owners and SLAs.

**Deliverable.** A rules notebook, an exception report, and a control design note.

**Acceptance criteria.** Rules deterministic and documented; false-positive rate measured on a reviewed sample; recoverable value estimated with assumptions; no rule presented as proof of fraud.

**Stretch.** Add a graph analysis of shared addresses, bank accounts, and phone numbers across vendors.

## 8. Assignment — Supplier risk scorecard

Produce a **procurement decision pack** for one category: a reconciled spend cube, a segmentation, a should-cost and TCO analysis, a risk scorecard, and an award recommendation.

| Section | Content |
|---|---|
| 1. Scope | Category, spend, business units, contract calendar |
| 2. Spend cube | Reconciliation, classification accuracy, tail spend, contract coverage |
| 3. Segmentation | Kraljic placement with factors and weights |
| 4. Price analysis | Should-cost, quote gap, TCO comparison |
| 5. Supplier performance | Scorecard with weights and sensitivity |
| 6. Risk | Financial, operational, concentration, multi-tier exposure |
| 7. Sourcing options | Award scenarios with cost and concentration trade-off |
| 8. ESG | Scope 3 method, supplier data quality, compliance gaps |
| 9. Recommendation | Award, contract terms, risk mitigations, savings estimate |
| 10. Controls | Duplicate-payment and maverick-buying controls |

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Spend cube and data quality | 20% | Category totals from a spreadsheet | Normalized, classified, reconciled | Spend-weighted accuracy, duplicate resolution, contract coverage |
| Segmentation and strategy | 15% | Labels suppliers by name | Kraljic with stated factors | Refreshable scoring, strategy per quadrant, sensitivity |
| Should-cost and TCO | 20% | Compares unit prices | Landed cost with freight and duty | Full cost build-up, carrying, quality, switching cost, negotiation use |
| Risk analysis | 20% | Single financial ratio | Multi-factor score | Explainable, dated, multi-tier, concentration by criticality |
| Sourcing and award | 15% | Picks the cheapest | Weighted evaluation | Optimization with capacity, minimum volume, concentration constraint |
| ESG, controls, communication | 10% | Mentions sustainability | Scope 3 method stated | Uncertainty quantified, fraud controls designed, decision-ready memo |

## 10. Regulatory, ethical, and commercial lens

| Area | Requirement or risk | Practical control |
|---|---|---|
| Due diligence | EU CSDDD as amended by Omnibus I (in force 18 March 2026; transposition by 26 July 2028) | Chain-of-activities mapping, grievance mechanism, documented remediation |
| Carbon border | CBAM definitive regime from 1 January 2026, 50-tonne threshold, first surrender 30 September 2027 | Supplier embedded-emissions data, declarant processes |
| Forced labour | UFLPA rebuttable presumption; import detention risk | Tier-2+ traceability, audit evidence, supplier declarations |
| Conflict minerals | Dodd-Frank Section 1502 reporting for covered issuers | Smelter and refiner identification, independent audit |
| Bribery | ISO 37001; FCPA and UK Bribery Act exposure through agents | Third-party due diligence, gift and hospitality controls |
| Sanctions | Restricted-party screening | Screen suppliers, owners, and banks; retain evidence |
| Competition | Bid-rigging and information exchange among bidders | Separate communications, audit bid patterns, legal review |
| Sustainability reporting | ESRS E1-6 Scope 1, 2, 3 by category; phase-in reliefs | Method per category, uncertainty range, assurance readiness |
| Supplier diversity | Programme targets and reporting | Classify consistently; verify certifications; avoid front companies |
| Payment conduct | Late payment regulation and supplier liquidity impact | Monitor payment terms and actual days paid |

Two obligations: **never present a spend-based emissions estimate as a measured supplier footprint**, and **never present a risk score as a finding of wrongdoing**. Both are defensible only with their method and evidence attached.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Dirty vendor master | Supplier count grows faster than spend | No dedupe rule | False concentration, duplicate payments | Normalize on tax ID and bank account |
| "Miscellaneous" category | Largest category is unclassified | Rules stop at the head | Strategy cannot be targeted | Classifier plus human review; spend-weighted accuracy |
| Cube does not tie to GL | Finance disputes the numbers | Taxes, intercompany, rebates mishandled | Analytics loses authority | Reconcile before publishing |
| Kraljic by opinion | Same quadrant for everything | Axes not measured | Wrong strategy per category | Publish factors and weights |
| Should-cost as accusation | Supplier walks out | Model ignores their constraints | Relationship damage | Use as agenda, not verdict |
| TCO stops at landed cost | Distant supplier looks cheapest | Carrying and quality excluded | Wrong award | Full lifecycle cost |
| Auction without qualification | Unqualified bidder wins | Price-only selection | Delivery failure, change orders | Qualify, set reserve, evaluate total cost |
| Scorecard weights hidden | Ranking changes when challenged | No sensitivity | Decision reversed late | Publish weights and sensitivity |
| Single financial ratio | Supplier fails without warning | Altman applied blindly | Uninsured loss | Multi-factor, dated, monitored |
| Tier-1 only mapping | "We are diversified" | Tier-2 invisible | Hidden single point of failure | Multi-tier mapping with evidence quality |
| Spend-based Scope 3 as measured | One number, no method | Screening data used for disclosure | Assurance failure | Label method and uncertainty |
| Fraud rules unowned | 10,000 alerts a month | No triage capacity | Control ignored | Owner, SLA, false-positive target |

## 12. Interview and discussion questions

| # | Question | Answer hints |
|---:|---|---|
| 1 | How do you build a spend cube from AP data? | Extract, normalize vendors, classify, adjust for tax/intercompany/currency, reconcile to GL. |
| 2 | Why is supplier normalization the highest-value step? | It changes concentration, duplicate payment, and tail-spend conclusions; names are free text. |
| 3 | When is UNSPSC the wrong taxonomy? | When you need sourcing-lever granularity; UNSPSC classifies what a thing is, not how it is bought. |
| 4 | How do you place a supplier on the Kraljic matrix? | Score profit impact and supply risk with published factors; refresh as markets change. |
| 5 | What is the difference between should-cost and TCO? | Should-cost builds the supplier's price; TCO adds owning costs over the lifecycle. |
| 6 | How would you structure a reverse auction? | Qualify bidders, clear spec, reserve from should-cost, total-cost evaluation, post-award risk. |
| 7 | A scorecard ranks A first. What do you check? | Weights and sensitivity; rank reversals; capability vs performance; data recency. |
| 8 | How do you assess a private supplier's financial health? | Payment behaviour, credit reports, bank references, filings where available; adjust Z-score for private firms. |
| 9 | What is single-source exposure and how do you measure it? | Spend on parts with one qualified source ÷ total, weighted by criticality. |
| 10 | How do you measure Scope 3 category 1 honestly? | Method per category, supplier-specific where possible, uncertainty range, assurance status. |
| 11 | How do you find duplicate payments? | Exact and fuzzy invoice matching, vendor-master duplicates, split POs, Benford; measure false positives. |
| 12 | Why does LLM contract extraction need a verification sample? | Extraction errors are silent; report precision and recall per field with provenance. |

## 13. Key terms

| Term | Definition |
|---|---|
| Source-to-pay | End-to-end process from need identification to payment |
| Spend cube | Three-dimensional view of spend by category, supplier, and unit or time |
| Tail spend | Low-value transactions spread across many suppliers |
| Maverick buying | Purchases outside contracted channels or approved suppliers |
| UNSPSC | United Nations Standard Products and Services Code, four-level hierarchy |
| GS1 GPC | Global Product Classification for retail products |
| Kraljic matrix | Segmentation by profit impact and supply risk |
| Should-cost | Price built from the supplier's cost structure |
| TCO | Total cost of ownership, including lifecycle and transaction costs |
| Landed cost | Product cost plus freight, insurance, duty, and handling |
| Reverse auction | Auction in which sellers compete and price falls |
| Reserve price | Maximum acceptable price, often derived from should-cost |
| Winner's curse | Winning bid that is unprofitable because of over-optimism |
| OTIF | On time in full |
| PPM | Parts per million defective |
| Altman Z-score | Composite financial distress score |
| HHI | Herfindahl-Hirschman Index, sum of squared market shares |
| Single-source exposure | Spend on parts with only one qualified supplier |
| Multi-tier mapping | Identifying suppliers beyond tier 1 |
| Contract coverage | Share of spend linked to a contract |
| Clause provenance | Link from extracted contract data to its source location |
| Scope 3 category 1 | Purchased goods and services emissions |
| Spend-based method | Emissions estimated from spend and industry factors |
| Supplier diversity | Programme to include underrepresented-owned suppliers |
| Three-way match | PO, receipt, and invoice reconciliation |
| Split purchase | Dividing a purchase to avoid an approval threshold |
| Benford's law | Expected first-digit distribution used as a fraud screen |
| Due diligence | Investigation of supplier legal, ethical, and environmental risk |

## 14. Sources and further reading

**Standards and regulation**

- Regulation (EU) 2022/2464 (CSRD) and Directive (EU) 2026/470 (Omnibus I amendments). EUR-Lex. Accessed 27 October 2026.
- Directive (EU) 2024/1760 (CSDDD), as amended by Omnibus I — transposition timeline and scope. EUR-Lex. [VERIFY: consolidated text and transposition status at time of use.]
- Regulation (EU) 2023/956 (CBAM), definitive regime from 1 January 2026. European Commission. https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism — accessed 27 October 2026.
- ISO 20400:2017, *Sustainable procurement — Guidance*.
- ISO 37001:2016, *Anti-bribery management systems*.
- ISO 31000:2018, *Risk management — Guidelines*.
- GHG Protocol, *Corporate Accounting and Reporting Standard* and *Corporate Value Chain (Scope 3) Standard*. https://ghgprotocol.org — accessed 27 October 2026.
- UNSPSC. https://www.unspsc.org — accessed 27 October 2026.
- GS1, *Global Product Classification*. https://www.gs1.org/standards/gpc — accessed 27 October 2026.
- U.S. Dodd-Frank Wall Street Reform and Consumer Protection Act, Section 1502 (conflict minerals).
- Uyghur Forced Labor Prevention Act (UFLPA), U.S. CBP guidance. https://www.cbp.gov — accessed 27 October 2026.

**Peer-reviewed and methodological**

- Kraljic, P. "Purchasing Must Become Supply Management." *Harvard Business Review* 61(5), 1983.
- Ellram, L. M. "Total Cost of Ownership: An Analysis Approach for Purchasing." *International Journal of Logistics Management* 6(2), 1995.
- Carr, A. S., and Pearson, J. N. "Strategically Managed Buyer–Supplier Relationships and Performance Outcomes." *Journal of Operations Management* 17(5), 1999.
- Jap, S. D. "An Exploratory Study of the Introduction of Online Reverse Auctions." *Journal of Marketing* 67(3), 2003. [VERIFY: volume and issue.]
- Williamson, O. E. "Credible Commitments: Using Hostages to Support Exchange." *American Economic Review* 73(4), 1983.
- Altman, E. I. "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy." *Journal of Finance* 23(4), 1968.
- Carter, C. R., and Rogers, D. S. "A Framework of Sustainable Supply Chain Management: Moving Toward New Theory." *International Journal of Physical Distribution & Logistics Management* 38(5), 2008.
- Monczka, R. M., et al. *Purchasing and Supply Chain Management*, 6th ed. Cengage, 2016.

**Industry and company evidence**

- Association of Certified Fraud Examiners, *Occupational Fraud 2024: A Report to the Nations*. https://www.acfe.com — accessed 27 October 2026. Survey estimate; verify the 5% figure.
- Boeing 787 programme reporting, including GAO reports on acquisition challenges and Boeing earnings commentary, 2007–2011. [VERIFY: specific GAO report number before citing.]
- Reporting on the 2016 Volkswagen–Prevent supplier dispute (Reuters, *Handelsblatt*, *Financial Times*). [VERIFY: dates, plants, and settlement.]
- Apple, Supplier List and Supplier Responsibility reports. https://www.apple.com/supplier-responsibility/ — accessed 27 October 2026. Company claims.
- Unilever, Sustainable Living Plan and annual progress reports. Company claims; independent critique exists.
- Open Supply Hub, facility-level supply chain data. https://opensupplyhub.org — accessed 27 October 2026.
- CDP disclosure platform. https://www.cdp.net — accessed 27 October 2026.

**Tools and datasets**

- `rapidfuzz` for supplier name matching. https://github.com/rapidfuzz/RapidFuzz — accessed 27 October 2026.
- NetworkX for multi-tier graph analysis. https://networkx.org — accessed 27 October 2026.
- PuLP and OR-Tools for award optimization. https://coin-or.github.io/pulp/ and https://developers.google.com/optimization — accessed 27 October 2026.
- DuckDB for local spend-cube SQL. https://duckdb.org — accessed 27 October 2026.

> **Note:** Spend, cost, and risk figures in this module are worked examples with stated assumptions. They are not benchmarks; replace them with your own data before making an award decision.

---

> **Next:** [Module 14A](14A_resilience_disruption_analytics.md) — the supplier you just scored sits inside a network, and the next question is what happens to the whole system when one node stops.
