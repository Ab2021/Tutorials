# Module 14A — Supply Chain Risk, Resilience, and Disruption Analytics

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 30 hours (12 lecture, 10 lab, 8 project)
> **Track fit:** Common core for Manufacturing & Industrial Operations; recommended for Retail & E-commerce and Logistics & Distribution
> **Prerequisites:** Module 11 (network design); Module 14 (supplier risk); Module 10 recommended (inventory buffers); basic probability and simulation
> **Tools:** Python 3.12, pandas, NumPy, NetworkX, matplotlib, SimPy optional; GDELT, ACLED, NOAA, and port data APIs
> **Industry question:** "If this node stops on Monday, how long until the customer feels it, how long until we recover, and what would it cost to close the gap?"

---

## 1. Why industry cares

Resilience became a board-level objective because the disruptions stopped being rare. In the fifteen years after 2010 the industry absorbed a Japanese earthquake, Thai floods, a Suez blockage, a pandemic, a semiconductor shortage, a European land war, and a Red Sea diversion — each of which removed capacity that no single company controlled.

The commercial case is not "avoid all disruption". It is to know, before the event, which nodes matter, how long the network can survive without them, and which interventions buy the most survival time per dollar.

Three facts define the 2026 context.

1. **Trade policy is a live risk, not a background condition.** DP World's Global Trade Observatory 2026 (3,500+ executives) reported that 94% expect 2026 trade growth to match or exceed 2025, with tariffs and economic shocks as top risks and 90% expecting trade barriers to rise or hold. Resilience spending is being justified against that expectation.
2. **Red Sea capacity is still abnormal.** As of October 2026, FreightWaves reported Suez transits recovering to roughly 30–35% of normal, with Gemini expecting a near-full return in Q1 2027. A "normal" assumption in a network model is currently wrong.
3. **Detection is cheap; discrimination is not.** GDELT, ACLED, weather models, and AIS port data are free or inexpensive. The hard problem is separating a signal that predicts a real disruption from noise that generates alerts nobody acts on.

The cost of getting this wrong has two forms: the disruption you did not see, and the resilience you paid for that never paid back. Both are failures of measurement, and both are addressable.

## 2. Learning outcomes

By the end of this module, you can:

- Construct a risk taxonomy that maps hazards to the nodes and flows they affect.
- Build a network exposure map from bill of materials, supplier sites, and logistics lanes.
- Compute time-to-recover (TTR) and time-to-survive (TTS) for a product line and interpret the gap.
- Distinguish robustness, resilience, and antifragility, and choose the property the decision requires.
- Design a stress test with correlated disruption scenarios and report a loss distribution, not a single number.
- Compare buffering, dual sourcing, nearshoring, and inventory placement on cost per unit of risk reduced.
- Evaluate an early-warning signal on precision, recall, lead time, and false-positive cost.
- Apply ISO 22301 business impact analysis concepts: MTPD, RTO, and RPO.
- Read an insurance programme for business interruption and contingent business interruption coverage gaps.
- Build a resilience scorecard without collapsing it into a single composite score.
- Write a board memo that states the risk appetite, the proposed spend, and the residual exposure.

## 3. Mental model

### 3.1 The disruption timeline

```
 event ──► detect ──► respond ──► recover ──► normal
   │         │          │           │
   │      time to     time to     time to
   │      detect      respond     recover (TTR)
   │
   └──────── time to survive (TTS): inventory and alternate capacity that
            keep the customer supplied while recovery happens
```

Resilience is the ratio of two clocks. If TTR exceeds TTS, the customer experiences a stockout regardless of how good the recovery plan is.

### 3.2 Robustness, resilience, antifragility

| Property | Definition | Design response | Cost profile |
|---|---|---|---|
| Robustness | Withstands a shock without changing state | Redundancy, overcapacity, stronger contracts | High fixed cost |
| Resilience | Absorbs, adapts, and recovers | Flexibility, visibility, alternate routes | Moderate, option-like |
| Antifragility | Improves under stress | Learning loops, modularity, optionality | Cultural and organizational |

Most companies buy robustness and call it resilience. A second warehouse is robustness; the ability to re-route volume to a third-party site in 72 hours is resilience.

### 3.3 The four levers

| Lever | Mechanism | Speed to implement | Recurring cost | Best when |
|---|---|---|---|---|
| Buffer | Inventory, capacity, time | Weeks | Carrying and idle capacity | TTR long, demand stable |
| Flexibility | Alternate sites, multi-skilled labor, convertible capacity | Months | Lower, option-like | Demand and supply both uncertain |
| Visibility | Detection, tracking, supplier transparency | Months | Data and integration | Blind spots dominate |
| Collaboration | Supplier and customer coordination | Quarters | Relationship investment | Shared exposure, joint planning |

## 4. Core concepts

### 4.1 Risk taxonomy

| Category | Examples | Primary nodes affected | Detection source |
|---|---|---|---|
| Natural | Earthquake, flood, hurricane, wildfire, pandemic | Sites, ports, roads | NOAA, Copernicus, USGS |
| Geopolitical | Tariffs, sanctions, conflict, export controls | Borders, suppliers, lanes | GDELT, ACLED, government notices |
| Supplier | Insolvency, fire, quality failure, capacity loss | Tier-1 and tier-2 sites | Financial data, audits, news |
| Transport | Port closure, carrier failure, canal blockage, strike | Lanes, ports, hubs | AIS, port data, carrier notices |
| Cyber | Ransomware, outage, data breach, third-party software | IT/OT systems, EDI, WMS | Incident feeds, vendor advisories |
| Demand | Sudden surge, collapse, mix shift | Inventory, capacity | POS, orders, market data |
| Regulatory | Product bans, forced-labor rules, carbon border | Products, markets, suppliers | Regulators, customs |
| Financial | Credit contraction, currency, commodity price | Suppliers, working capital | Markets, credit data |

A taxonomy is useful only if each entry maps to an owned mitigation. A risk register with 300 rows and no owners is a filing cabinet.

### 4.2 Mapping and exposure

Exposure mapping joins three graphs: the **product** graph (BOM and substitutions), the **supply** graph (supplier sites and their suppliers), and the **logistics** graph (lanes, ports, DCs).

| Exposure metric | Formula | Reading |
|---|---|---|
| Revenue at risk | revenue through node ÷ total revenue | Financial materiality |
| Single-source exposure | spend on single-qualified parts ÷ total spend | Substitution difficulty |
| Node criticality | betweenness or flow-through | Structural importance |
| Geographic concentration | share of volume in one country or seismic zone | Correlated hazard |
| Days of cover | inventory ÷ daily demand | Survival time without the node |
| Recovery dependency | number of products requiring requalification | TTR driver |

NetworkX is sufficient for the graph work: build a directed graph of site-to-site flows, compute betweenness centrality, and identify articulation points — nodes whose removal disconnects the network.

**The tier-2 problem.** Most exposure maps stop at tier 1 because that is the data a company has. Tier-2 mapping requires supplier declarations, industry databases such as Open Supply Hub, customs records, or purchased risk feeds. Coverage is partial and must be labelled as such.

### 4.3 Time-to-recover and time-to-survive

**Time to survive (TTS):** how long the network can meet demand from inventory and alternate supply after a node fails.

```
TTS (weeks) = available inventory ÷ (demand rate − alternate supply rate)
```

**Time to recover (TTR):** how long until the failed node, or a qualified substitute, restores supply. TTR is driven by physical repair, tooling, requalification, and regulatory approval — not by the supplier's optimistic estimate.

Worked example:

| Input | Value |
|---|---:|
| Weekly demand | 1,200 units |
| Finished-goods and WIP inventory | 3,600 units |
| Alternate supply available | 400 units/week |
| TTR estimate | 7 weeks |

```
TTS = 3,600 ÷ (1,200 − 400) = 4.5 weeks
gap = TTR − TTS = 7.0 − 4.5 = 2.5 weeks
lost units = 2.5 × (1,200 − 400) = 2,000 units
```

At a $240 price and $85 contribution, that is $480,000 of revenue and $170,000 of contribution at risk — **assuming alternate supply continues throughout the gap**.

If the alternate supply cannot be used until qualification completes, the whole gap is served at the full demand rate: `2.5 × 1,200 = 3,000 units`, or $720,000 of revenue and $255,000 of contribution. **State which assumption applies**; the difference is 50% of the answer.

A TTR/TTS ratio above 1 means the network cannot survive the disruption. The mitigation target is either to raise TTS (buffer, alternate supply) or lower TTR (pre-qualified alternates, modular tooling, design for substitution).

### 4.4 Stress testing

A stress test is a controlled experiment on the network model: remove a node, hold demand and policy fixed, and measure service and financial loss over time.

| Test design choice | Options | Consequence |
|---|---|---|
| Disruption duration | Fixed (e.g. 8 weeks) vs distribution | Fixed understates tail risk |
| Correlation | Independent nodes vs common-cause | Independent understates correlated shocks |
| Demand response | Constant vs surge vs collapse | Constant misses hoarding effects |
| Mitigation response | None vs modelled | No-response overstates loss |
| Metric | Units short, revenue, contribution, service days | Revenue overstates cash impact |

The Monte Carlo below replaces the point-estimate TTR with a lognormal distribution and demand and alternate supply with normal distributions, then reports the loss distribution.

```python
"""TTR/TTS stress test. Pure Python; run with a fixed seed for reproducibility."""
import math
import random
import statistics

SEED, RUNS = 42, 20_000
TTS_WEEKS = 4.5
DEMAND_MEAN, DEMAND_SD = 1_200, 120
ALT_MEAN, ALT_SD = 400, 60
TTR_MEDIAN, TTR_SIGMA = 7.0, 0.45
CONTRIBUTION = 85.0

def one_run(rng):
    ttr = rng.lognormvariate(math.log(TTR_MEDIAN), TTR_SIGMA)
    demand = max(0.0, rng.gauss(DEMAND_MEAN, DEMAND_SD))
    alt = max(0.0, rng.gauss(ALT_MEAN, ALT_SD))
    lost_units = max(0.0, ttr - TTS_WEEKS) * max(0.0, demand - alt)
    return lost_units

rng = random.Random(SEED)
losses = sorted(one_run(rng) for _ in range(RUNS))
mean_loss = statistics.mean(losses)
p95 = losses[int(0.95 * RUNS) - 1]
prob_gap = sum(1 for x in losses if x > 0) / RUNS

print(f"P(TTR > TTS)        {prob_gap:.2%}")
print(f"mean lost units     {mean_loss:,.0f}")
print(f"p95 lost units      {p95:,.0f}")
print(f"mean contribution   ${mean_loss * CONTRIBUTION:,.0f}")
```

Expected output: `P(TTR > TTS) 83.7%`, mean 2,719 units, p95 8,391 units, mean contribution at risk about $231,100. Compare that with the deterministic 2,000-unit answer: the distribution shows a tail more than three times the mean, which is the number that matters for risk appetite.

### 4.5 Scenario analysis

| Scenario | Annual probability | Mechanism | Revenue at risk | Margin impact | Response |
|---|---:|---|---:|---:|---|
| Port closure, 3 weeks | 15% | Lane diversion, dwell | $4.2M | $924k | Pre-booked alternate port, buffer |
| Tier-2 fire, 8 weeks | 5% | Single site, requalification | $9.6M | $2.11M | Dual source, pre-qualified alternate |
| Cyber outage, 5 days | 20% | WMS/EDI unavailable | $1.8M | $396k | Manual fallback, cyber insurance |
| Tariff step change | 30% | Landed cost increase | — | $1.4M | Origin diversification, pricing pass-through |
| **Expected annual loss** | | | | **$743k** | |

Arithmetic check: `0.15 × 924,000 = 138,600`; `0.05 × 2,112,000 = 105,600`; `0.20 × 396,000 = 79,200`; `0.30 × 1,400,000 = 420,000`; total `$743,400`.

The expected loss is the number to compare against resilience spend — not the worst case, and not zero. A board that approves spend only against worst cases over-invests; one that approves against the mean under-invests in tail protection.

### 4.6 Buffering versus flexibility

| Mitigation | Annual cost | Mean loss reduction | Net benefit | Payback logic |
|---|---:|---:|---:|---|
| Base case (no action) | $0 | — | — | Mean loss $231.1k |
| +800 units buffer (TTS 5.5) | $42.2k | $52.4k | +$10.2k | Marginal |
| +2,000 units buffer (TTS 7.0) | $105.6k | $113.4k | +$7.8k | Marginal |
| Pre-qualified alternate (TTR 3.0) | ~$180k one-time + $30k/yr | $213.4k | +$183.4k/yr | Strong |
| Dual source 30% (TTS 7.5) | Qualification + price premium | $129.3k | Depends on premium | Case-specific |

Arithmetic: buffer carrying cost = units × unit cost × 22% per year. `2,000 × $240 × 0.22 = $105,600`. Loss reduction = `231.1k − 117.7k = $113.4k`, net `+$7.8k`.

The teaching point is uncomfortable and important: **at these assumptions, inventory buffers are roughly break-even, while compressing TTR is highly valuable.** Buffers protect against short disruptions and are easy to approve; they are a poor substitute for a qualified alternate source when TTR is long. Run the numbers for your own case rather than copying this conclusion.

### 4.7 Dual sourcing and nearshoring

| Option | Resilience benefit | Cost penalty | When it fails |
|---|---|---|---|
| Dual source, 70/30 split | Removes single point of failure; preserves leverage | 3–8% unit cost, qualification cost, quality variance | Both sources share a tier-2 site or region |
| Dual source, 50/50 | Maximum redundancy | Loses volume leverage; coordination cost | Too little volume for either supplier to invest |
| Nearshoring | Cuts lead time and TTR; reduces tariff exposure | 10–25% unit cost in many categories | New region has its own hazard; ecosystem missing |
| Friendshoring | Political risk reduction | Cost and capacity constraints | "Friendly" is a policy judgement, not a guarantee |
| Regionalization | Shorter lanes, faster recovery | Inventory duplication | Demand density too low |

The dual-sourcing decision rule: **dual source when the expected loss reduction exceeds the qualification cost plus the unit-price premium over the planning horizon.** Compute both sides; "resilience" is not a free good and "cost" is not the only number.

Nearshoring is not automatically resilient. Moving from one earthquake zone to another, or to a region with a single port and no second-tier supplier base, trades one exposure for another. Map the new region's hazards before celebrating the lead-time reduction.

### 4.8 Inventory placement

Where the buffer sits determines what it protects.

| Placement | Protects against | Cost | Drawback |
|---|---|---|---|
| Raw material at plant | Supplier failure | Carrying, obsolescence | Late product-mix changes strand it |
| WIP at decoupling point | Process failure | Carrying, cycle time | Limited flexibility |
| Finished goods at DC | Transport and demand shocks | Highest carrying, widest coverage | Expensive per unit of protection |
| Strategic reserve, supplier-held | Supplier capacity | Contract and storage fee | Access may be contested in a crisis |
| Customer-side consignment | Delivery failure | Working capital at customer | Limited control |

Risk pooling argues for centralizing safety stock; resilience argues for decentralizing it near demand. The resolution is segment-specific: pool stable, low-value items; decentralize critical, high-margin items.

### 4.9 Disruption detection

| Source | What it detects | Latency | Coverage | Main weakness |
|---|---|---|---|---|
| GDELT | News-coded events, protests, conflict, disaster | Near real-time | Global | Noise and duplicate coding |
| ACLED | Organized conflict and protest events | Days | Global, registration | Political violence focus only |
| NOAA / Copernicus | Weather and climate hazards | Hours to days | Global | Exposure mapping still required |
| Port authority and AIS | Congestion, closures, vessel diversion | Hours | Major ports | AIS gaps and spoofing |
| Supplier financial feeds | Distress signals | Weeks to months | Public companies best | Private suppliers opaque |
| Carrier and 3PL notices | Lane and capacity changes | Hours | Contracted lanes | Reactive and incomplete |
| Internal signals | Order pattern, lead-time drift, OTIF decline | Days | Own network | Late by construction |

Signal evaluation:

| Metric | Definition | Target |
|---|---|---|
| Precision | True disruptions ÷ alerts raised | Depends on cost per false alarm; often 20–40% is acceptable |
| Recall | Detected disruptions ÷ actual disruptions | > 70% for material events |
| Median lead time | Alert time to impact time | Days, not hours, for most supply events |
| Alert volume | Alerts per analyst per day | Matched to triage capacity |
| Coverage | Share of spend or nodes monitored | Stated, not assumed |

A signal that fires 500 times a day with 5% precision is not an early-warning system; it is a reason analysts stop reading. Design alerts with corroboration (two independent sources), persistence (two consecutive periods), and asset matching (the event touches a known node).

### 4.10 Resilience metrics

| Metric | Definition | Use |
|---|---|---|
| TTR | Weeks to restore supply after node failure | Design and sourcing decisions |
| TTS | Weeks the network survives without the node | Buffer sizing |
| TTR/TTS ratio | Recovery time ÷ survival time | Single most useful summary |
| Time to detect | Event to internal awareness | Detection investment |
| Time to respond | Awareness to action | Playbook quality |
| Revenue at risk | Revenue through exposed nodes | Board reporting |
| Single-source exposure | Share of spend or revenue with one qualified source | Sourcing strategy |
| Days of cover | Inventory ÷ daily demand, by critical item | Buffer monitoring |
| Recovery test pass rate | Exercises meeting RTO ÷ exercises | BCP readiness |
| Buffer cost | Carrying + idle capacity cost | Cost of resilience |

Do not collapse these into one composite score. A composite hides which lever is weak and invites gaming. Report a small dashboard with thresholds.

### 4.11 Business continuity: ISO 22301

ISO 22301:2019 defines the requirements for a business continuity management system. The analytical core is the **business impact analysis (BIA)**:

| Concept | Definition | Analytics input |
|---|---|---|
| MTPD | Maximum tolerable period of disruption | Customer contracts, revenue impact curve |
| RTO | Recovery time objective for a process or site | Process dependency map |
| RPO | Recovery point objective for data | System backup and replication design |
| BIA | Ranking of processes by impact over time | Revenue and regulatory impact models |

The BIA is a time-series question: impact at day 1, day 3, week 2, week 8. A process with a 48-hour MTPD needs a different plan from one with a 30-day MTPD, and the plan's cost should scale accordingly. ISO 22301 also requires exercising and continual improvement; an untested plan is a document, not a capability.

ISO 31000:2018 supplies the risk management process, and ISO 28000:2022 the security and resilience management system for the supply chain. They are complementary: 31000 for the process, 22301 for continuity, 28000 for security.

### 4.12 Insurance and contracts

| Coverage | What it pays | Common gap |
|---|---|---|
| Property | Physical damage | Business interruption not included |
| Business interruption (BI) | Lost profit and continuing costs after own-site damage | Waiting period; gross profit definition |
| Contingent BI (CBI) | Loss from damage at a supplier or customer site | Named-supplier requirements; sublimits |
| Cargo | Goods in transit | Marine perils, packing warranties |
| Cyber | Response and business interruption | Exclusions for systemic events; waiting period |
| Trade credit | Customer insolvency | Supplier insolvency often uninsured |

Insurance transfers financial loss; it does not restore customers, capacity, or reputation. The analytic question is the **coverage gap**: modelled loss minus (limit minus deductible), adjusted for the waiting period and exclusions. A programme with a 72-hour waiting period does not respond to a 48-hour outage.

Contracts are the other transfer mechanism. Force majeure clauses, liquidated damages, supply assurance obligations, and termination rights determine who bears the loss. Review them against the scenarios, not in the abstract.

### 4.13 The cost of resilience

```
total resilience cost = buffer carrying + idle capacity + qualification + insurance + detection + coordination
residual exposure = expected loss after mitigation
```

The efficient frontier:

```
 expected
 annual
 loss
   │
   │ ●
   │   ●
   │      ●
   │          ●  ← diminishing returns
   │              ●
   └────────────────────────► annual resilience spend
```

Spend until the marginal reduction in expected loss equals the marginal cost. That point is different for every product line, which is why a single corporate resilience budget is a blunt instrument.

## 5. Data, standards, and workflow

| Data object | Grain | Source | Requirement |
|---|---|---|---|
| Supplier site | Site | SRM, supplier declaration | Address, geocode, capability, capacity |
| BOM | Parent × component | PLM / ERP | Effective dates; alternates and substitutions |
| Part-source link | Part × site | ERP, supplier data | Qualified source flag, qualification date |
| Lane | Origin–destination | TMS | Mode, transit time, capacity, alternate routes |
| Inventory | Item × location | ERP / WMS | On-hand, in-transit, allocated |
| Event feed | Event | GDELT, ACLED, NOAA, AIS | Timestamp, location, type, source |
| Financial | Supplier × period | Filings, credit bureau | Statement date and currency |
| Contract | Contract × clause | CLM | Force majeure, assurance, penalties |

**Standards.** ISO 22301:2019 (business continuity), ISO 31000:2018 (risk management), ISO 28000:2022 (security and resilience), ISO 22316:2017 (organizational resilience), the Sendai Framework for Disaster Risk Reduction 2015–2030, and the GHG Protocol where emissions are part of the exposure.

**Data-quality checks.** Site geocodes resolve to the correct facility; BOM effective dates are respected; alternate sources are marked with their qualification status, not assumed available; transit times are observed, not scheduled; event feeds are de-duplicated across sources; financial data carries its statement date.

**Workflow.** Map exposure, compute TTR/TTS, stress test, evaluate mitigations, decide, then monitor. The monitoring loop must feed back into the map: a new supplier or lane changes the exposure.

## 6. Industry case studies

### 6.1 Tōhoku earthquake and the Renesas Naka fab (2011)

**Context.** The 11 March 2011 earthquake and tsunami damaged industrial sites across northern Japan, including Renesas Electronics' Naka fabrication facility in Hitachinaka, a major source of automotive microcontrollers.

**Intervention.** Renesas restored clean-room space and production in phases, with support from automotive customers and government; automakers dispatched engineers and coordinated on allocation.

**Evidence.** Renesas and customer statements, and contemporaneous industry reporting: partial production resumed in June 2011, with full recovery later in 2011. Automotive assembly plants worldwide cut output for weeks.

**Outcome.** A multi-month recovery that exposed the automotive industry's dependence on a small number of microcontroller fabs.

**What transferred.** The TTR problem: physical repair is only part of recovery; requalification and allocation take longer. The case drove supplier concentration mapping and long-term supply agreements in automotive semiconductors.

**What did not.** The lesson was not fully learned — the 2020–2023 chip shortage repeated the pattern with a different trigger, because the mapping stopped at tier 1 and the contracts did not assure capacity.

### 6.2 Suez Canal blockage (2021)

**Context.** The container ship Ever Given grounded in the Suez Canal on 23 March 2021, blocking a route carrying roughly 12% of global trade.

**Intervention.** Salvage and dredging refloated the ship on 29 March; carriers diverted vessels around the Cape of Good Hope or waited, and shippers scrambled for alternatives.

**Evidence.** Canal authority and carrier statements; Lloyd's List estimated the blockage held up about $9.6 billion of trade per day. The estimate is an industry calculation, not an audited figure.

**Outcome.** Six days of blockage produced weeks of congestion and schedule disruption because vessels arrive in waves at already-constrained ports.

**What transferred.** The difference between the event duration and the recovery duration. TTR for a six-day blockage was measured in weeks of network recovery, which is the number that matters for TTS.

**What did not.** The assumption that a single-chokepoint event is rare. The Red Sea diversion two years later removed the same route for months.

### 6.3 Automotive semiconductor shortage (2020–2023)

**Context.** Automakers cut orders in early 2020 as demand collapsed, then demand rebounded faster than expected while semiconductor capacity had shifted to consumer electronics.

**Intervention.** OEMs increased direct engagement with chip suppliers, signed long-term agreements, and in some cases built buffer inventories; governments convened industry coordination.

**Evidence.** AlixPartners estimated the shortage would cost the auto industry about $210 billion in revenue in 2021; the estimate is a consultancy projection, not an audited outcome. OEMs reported production cuts in quarterly results.

**Outcome.** Multi-year allocation, a structural shift toward direct semiconductor sourcing and buffer stocks, and a lasting increase in lead-time assumptions.

**What transferred.** The bullwhip mechanism at the tier boundary: order cuts transmitted a false demand signal upstream, and capacity did not return quickly. Visibility into tier-2 and tier-3 capacity is the mitigation.

**What did not.** The forecast. Nobody predicted the duration; the value came from building options, not from predicting the event.

### 6.4 Red Sea diversions (2023–2026)

**Context.** Attacks on commercial shipping in the Red Sea from late 2023 led carriers to divert Asia–Europe services around the Cape of Good Hope.

**Intervention.** Carriers added transit time and vessels, imposed surcharges, and reworked schedules; shippers shifted some volume to air and rail, and renegotiated contracts.

**Evidence.** FreightWaves reported in October 2026 that Suez transits had recovered to roughly 30–35% of normal, with Gemini expecting a near-full return in Q1 2027; Freightos reported Asia–North Europe at about $3,442 per FEU.

**Outcome.** A multi-year change in effective lead time, inventory in transit, and freight cost, with partial recovery underway.

**What transferred.** Lead-time variability, not just lead time, drives safety stock. A route that takes 14 extra days with high variance requires far more buffer than the mean suggests.

**What did not.** The assumption that a shipping disruption is a freight-cost problem. The larger cost was working capital and stockouts caused by longer, less reliable transit.

### 6.5 CrowdStrike outage (19 July 2024)

**Context.** A faulty configuration update to CrowdStrike's Falcon sensor caused Windows systems to crash, affecting about 8.5 million devices according to Microsoft.

**Intervention.** CrowdStrike remediated the update; affected organizations recovered systems manually, with airlines, ports, hospitals, and logistics operators among the most visible.

**Evidence.** Microsoft's estimate of affected devices, CrowdStrike's incident report, and regulatory and industry reporting. The operational impact on specific supply chains is documented unevenly.

**Outcome.** Days of disruption for some operators, and a reassessment of third-party software concentration and manual fallback capability.

**What transferred.** Cyber risk is an operational-continuity risk, not only a data-protection risk. The mitigation is tested manual fallback and recovery sequencing, not just patching.

**What did not.** The assumption that a vendor's size or reputation implies resilience. Concentration in a critical vendor is a single point of failure regardless of that vendor's quality.

## 7. Labs

### Lab 14A.1 — Exposure map and risk register

**Objective.** Build a node-level exposure map and a risk register with owners and mitigations.

**Data.** BOM, supplier site list with geocodes, lane data, spend, revenue by product, and inventory.

**Steps.** (1) Build the product–site–lane graph in NetworkX. (2) Compute revenue at risk, single-source exposure, and betweenness centrality. (3) Overlay hazard layers (seismic, flood, cyclone) on site locations. (4) Identify the top ten exposed nodes. (5) Assign an owner and a mitigation to each. (6) State the coverage limitation of the tier-2 data.

**Deliverable.** An exposure map, a top-ten register, and a coverage statement.

**Acceptance criteria.** Graph construction documented; metrics reproducible; hazard overlay sourced; every register row has an owner and a mitigation; tier-2 coverage labelled.

**Stretch.** Add a substitution graph and compute how much revenue can be re-routed within one week.

### Lab 14A.2 — TTR/TTS stress test

**Objective.** Quantify survival time, recovery time, and the loss distribution for three disruption scenarios.

**Data.** Product-line demand, inventory, alternate capacity, supplier recovery estimates, and contribution margin.

**Steps.** (1) Compute deterministic TTS and TTR for each scenario. (2) Identify the gap and the loss under both shortfall assumptions. (3) Run the Monte Carlo with 20,000 draws and a fixed seed. (4) Report mean, p95, and probability of a gap. (5) Test three mitigations and compute net benefit. (6) Recommend the mitigation with the best net benefit per dollar.

**Deliverable.** A stress-test workbook and a two-page decision memo.

**Acceptance criteria.** Both shortfall assumptions shown; distributions reported with percentiles; mitigation costs included; recommendation justified by net benefit.

**Stretch.** Add correlated TTR across two suppliers sharing a region and show the change in the p95.

### Lab 14A.3 — Early-warning signal evaluation

**Objective.** Evaluate whether an external signal predicts disruptions better than chance, and at what cost.

**Data.** 24 months of GDELT or ACLED events, weather alerts, and port congestion, joined to a labelled list of supply disruptions.

**Steps.** (1) Define the event of interest and the matching rule to a node. (2) Build alerts with thresholds and persistence. (3) Compute precision, recall, and median lead time. (4) Compute the cost per true alert and per false alert. (5) Tune the threshold and plot the precision-recall trade-off. (6) Recommend an operating point and a triage workflow.

**Deliverable.** A signal evaluation report with a precision-recall curve and an operating recommendation.

**Acceptance criteria.** Labelled ground truth documented; matching rule explicit; lead time measured from alert to impact; false-positive cost estimated; operating point justified.

**Stretch.** Combine two signals and show whether corroboration improves precision without destroying recall.

### Lab 14A.4 — Scenario analysis and board memo

**Objective.** Translate exposure into a decision that a board can approve or reject.

**Data.** Lab 14A.1 and 14A.2 outputs plus insurance programme details and contract terms.

**Steps.** (1) Build five scenarios with probabilities and impacts. (2) Compute expected annual loss. (3) Map insurance coverage and identify the gap after waiting periods and sublimits. (4) Cost three mitigation options. (5) Compute residual exposure per option. (6) Write a one-page memo with a recommendation, a risk appetite statement, and what would change the answer.

**Deliverable.** A scenario table, a coverage-gap analysis, and a board memo.

**Acceptance criteria.** Probabilities justified (not invented); expected loss arithmetic correct; insurance terms read correctly; residual exposure shown; memo states the decision and the trigger for review.

**Stretch.** Add a stress case at the 99th percentile and show the capital or liquidity consequence.

## 8. Assignment — Resilience stress test

Produce a **resilience stress test** for one product line or network: the exposure map, the TTR/TTS gap, the loss distribution under three scenarios, the mitigation comparison, and a board-level recommendation.

| Section | Content |
|---|---|
| 1. Scope | Product line, network boundary, decision owner, risk appetite |
| 2. Exposure map | Nodes, flows, revenue at risk, single-source exposure |
| 3. TTR/TTS | Deterministic and probabilistic, with assumptions |
| 4. Scenarios | Five scenarios with probability and impact |
| 5. Stress test | Loss distribution, p95, probability of customer impact |
| 6. Mitigations | Buffer, alternate source, nearshoring, insurance, with costs |
| 7. Cost of resilience | Net benefit and residual exposure per option |
| 8. Insurance and contracts | Coverage gap and contractual transfer |
| 9. Recommendation | Chosen portfolio of mitigations and the trigger for review |
| 10. Monitoring | Signals, thresholds, owners, exercise cadence |

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Exposure mapping | 20% | Lists suppliers | Maps products to sites | Tier-2 coverage labelled, criticality computed, hazards overlaid |
| TTR/TTS analysis | 20% | Single recovery estimate | Deterministic gap computed | Distribution, assumptions explicit, both shortfall cases |
| Scenario and stress test | 20% | Three unrelated what-ifs | Probabilities and impacts | Correlated scenarios, p95, sensitivity to assumptions |
| Mitigation economics | 20% | Recommends buffers | Cost and benefit compared | Net benefit per dollar, residual exposure, diminishing returns |
| Insurance and contracts | 10% | Mentions insurance | Coverage summarized | Gap after waiting periods and sublimits; contractual transfer |
| Communication | 10% | Technical report | Clear memo | Board-ready decision with risk appetite and review trigger |

## 10. Regulatory, ethical, and commercial lens

| Area | Requirement or risk | Practical control |
|---|---|---|
| Business continuity | ISO 22301:2019 management system; sector regulators may require continuity plans | BIA with MTPD, RTO, RPO; annual exercise |
| Critical infrastructure | EU CER Directive (2022/2557) and NIS2 (2022/2555) for relevant entities | Identify obligations; incident reporting |
| Due diligence | EU CSDDD as amended; forced-labor import rules | Supplier mapping, grievance mechanism, remediation |
| Sanctions | Diverting through restricted parties or regions | Screen routes and counterparties |
| Insurance disclosure | Material risk and coverage must be represented accurately | Reconcile modelled loss with programme limits |
| Workforce | Crisis response can concentrate risk on frontline workers | Safety protocols, consultation, fair overtime |
| Customer equity | Allocation during shortage favours powerful customers by default | Publish allocation rules; protect essential customers |
| Data | Event feeds and supplier data may be commercially sensitive | Access controls; aggregation |

Two obligations: **never present a scenario probability as a forecast**, and **never present insurance limits as resilience**. A probability is a judgement with a basis; a limit is a contract that may not respond.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Tier-1-only map | "We are diversified" | Tier-2 invisible | Hidden single point of failure | Multi-tier mapping with labelled coverage |
| Optimistic TTR | Supplier says 4 weeks | No requalification time included | Buffer sized too small | Add qualification and ramp-up to TTR |
| TTS without alternate supply | Inventory ÷ full demand | Alternate capacity ignored | Understated survival | Model net shortfall explicitly |
| Single-point scenario | One disruption tested | No correlation | Understated tail | Correlated scenarios, p95 |
| Composite resilience score | One number for the board | Weights hide weakness | Wrong investment | Dashboard with thresholds |
| Alert flood | Analysts ignore the feed | No precision target | Detection failure | Corroboration, persistence, triage capacity |
| Buffer as default answer | Every risk gets inventory | Carrying cost ignored | Expensive, slow, sometimes useless | Compare with TTR compression |
| Insurance assumed complete | "We are covered" | Waiting period, sublimit, exclusion | Uninsured loss | Coverage-gap analysis |
| Nearshoring as resilience | Lead time falls, cost rises | New region's hazards unmapped | Traded risk, not reduced | Hazard map the new region |
| Plan never exercised | Document dated two years ago | Untested assumptions | Failure in the event | Annual exercise with findings |
| Risk register without owners | 300 rows, no actions | Ownership absent | Nobody acts | Owner, date, mitigation per row |
| Over-investment in robustness | Idle capacity everywhere | Worst-case-only justification | Cost without proportional benefit | Expected-loss and marginal analysis |

## 12. Interview and discussion questions

| # | Question | Answer hints |
|---:|---|---|
| 1 | Define TTR and TTS. | Recovery time after a node fails; survival time from inventory and alternate supply. |
| 2 | Why is TTR usually underestimated? | Qualification, tooling, and ramp-up are omitted; suppliers quote repair, not requalification. |
| 3 | What is the difference between robustness and resilience? | Withstanding versus absorbing and recovering; redundancy versus flexibility. |
| 4 | How would you map tier-2 exposure? | Supplier declarations, industry databases, customs data, risk feeds; label coverage limits. |
| 5 | When is a buffer the wrong answer? | When TTR is long and compressible, or the item is obsolete-prone; flexibility often wins. |
| 6 | How do you evaluate an early-warning signal? | Precision, recall, lead time, false-positive cost, alert volume versus triage capacity. |
| 7 | What does ISO 22301 require analytically? | BIA with MTPD, RTO, RPO; exercising; continual improvement. |
| 8 | What is contingent business interruption? | Coverage for loss caused by damage at a supplier or customer site; named-supplier and sublimit traps. |
| 9 | Why not one composite resilience score? | It hides which lever is weak and is easy to game; use a dashboard with thresholds. |
| 10 | How do you set scenario probabilities? | Base rates, expert elicitation with documented reasoning, and sensitivity; never present as forecast. |
| 11 | What did the chip shortage teach about tier-2 visibility? | Order cuts transmitted false demand; capacity at tier-2/3 determined recovery. |
| 12 | How much should a company spend on resilience? | Until marginal expected-loss reduction equals marginal cost, subject to risk appetite. |

## 13. Key terms

| Term | Definition |
|---|---|
| Resilience | Ability to absorb, adapt, and recover from disruption |
| Robustness | Ability to withstand a shock without changing state |
| Antifragility | Ability to improve under stress |
| TTR | Time to recover supply after a node fails |
| TTS | Time to survive on inventory and alternate supply |
| MTPD | Maximum tolerable period of disruption |
| RTO | Recovery time objective |
| RPO | Recovery point objective for data |
| BIA | Business impact analysis |
| Single point of failure | Node whose failure stops the flow |
| Node criticality | Structural importance of a node in the network |
| Betweenness centrality | Share of shortest paths passing through a node |
| Revenue at risk | Revenue dependent on an exposed node |
| Days of cover | Inventory divided by daily demand |
| Dual sourcing | Qualifying two suppliers for the same item |
| Nearshoring | Moving supply closer to the market |
| Friendshoring | Moving supply to politically aligned countries |
| Risk pooling | Aggregating inventory to reduce total safety stock |
| Decoupling point | Inventory position that separates processes |
| GDELT | Global event database coded from news |
| ACLED | Armed Conflict Location and Event Data |
| AIS | Automatic Identification System for vessel tracking |
| Precision | True alerts divided by all alerts |
| Recall | Detected events divided by all events |
| CBI | Contingent business interruption insurance |
| Waiting period | Time deductible before BI cover responds |
| Force majeure | Contract clause excusing performance after certain events |
| Expected annual loss | Probability-weighted sum of scenario losses |
| Residual exposure | Loss remaining after mitigation |
| Cost of resilience | Total cost of buffers, options, insurance, and detection |

## 14. Sources and further reading

**Standards and regulation**

- ISO 22301:2019, *Security and resilience — Business continuity management systems — Requirements*.
- ISO 31000:2018, *Risk management — Guidelines*.
- ISO 28000:2022, *Security and resilience — Security management systems for the supply chain*.
- ISO 22316:2017, *Security and resilience — Organizational resilience — Principles and attributes*.
- Directive (EU) 2022/2557 (CER) and Directive (EU) 2022/2555 (NIS2). EUR-Lex. Accessed 27 October 2026.
- UN Office for Disaster Risk Reduction, *Sendai Framework for Disaster Risk Reduction 2015–2030*. https://www.undrr.org — accessed 27 October 2026.
- Directive (EU) 2024/1760 (CSDDD), as amended by Omnibus I. EUR-Lex. [VERIFY: consolidated text and transposition status.]

**Peer-reviewed and methodological**

- Simchi-Levi, D., Schmidt, W., Wei, Y., and Zhang, P. Y. "From Superstorms to Factory Fires: Managing Unpredictable Supply-Chain Disruptions." *Harvard Business Review*, January–February 2014. Source of the TTR/TTS framing.
- Chopra, S., and Sodhi, M. S. "Managing Risk to Avoid Supply-Chain Breakdown." *MIT Sloan Management Review* 46(1), 2004.
- Christopher, M., and Peck, H. "Building the Resilient Supply Chain." *International Journal of Logistics Management* 15(2), 2004.
- Sheffi, Y., and Rice, J. B. "A Supply Chain View of the Resilient Enterprise." *MIT Sloan Management Review* 47(1), 2005.
- Pettit, T. J., Croxton, K. L., and Fiksel, J. "Ensuring Supply Chain Resilience: Development and Implementation of an Assessment Tool." *Journal of Business Logistics* 34(1), 2013.
- Ivanov, D., and Dolgui, A. "Viability of Intertwined Supply Networks: Extending the Supply Chain Resilience Angles Towards Survivability." *International Journal of Production Research* 58(10), 2020.
- Simchi-Levi, D., and Simchi-Levi, J. "We Need a Stress Test for Critical Supply Chains." *Harvard Business Review*, April 2020.

**Industry and company evidence**

- DP World, *Global Trade Observatory 2026*. Survey of 3,500+ executives. https://www.worldports.org/shippers-put-resilience-ahead-of-cost-cutting-survey/ — accessed 27 October 2026.
- FreightWaves, Red Sea transit recovery reporting, 20 October 2026. https://www.freightwaves.com/news/how-new-capacity-is-raising-rates-in-chaotic-ocean-market — accessed 27 October 2026.
- Freightos, *Freightos Global Outlook*, October 2026. https://www.freightos.com/freight-resources/freightos-global-outlook-october-2026/ — accessed 27 October 2026.
- Renesas Electronics statements and 2011 earthquake recovery reporting; automotive production impact coverage. [VERIFY: exact recovery dates by line.]
- Lloyd's List estimate of $9.6 billion per day of trade held up by the Suez blockage, 2021. Industry estimate.
- AlixPartners estimate of $210 billion in 2021 automotive revenue loss from the chip shortage. Consultancy projection.
- Microsoft, "Helping our customers through the CrowdStrike outage," 20 July 2024. https://blogs.microsoft.com — accessed 27 October 2026. Company statement: about 8.5 million devices.
- CrowdStrike, external root cause analysis of the 19 July 2024 incident. Company report.

**Tools and datasets**

- GDELT Project. https://www.gdeltproject.org — accessed 27 October 2026.
- ACLED. https://acleddata.com — accessed 27 October 2026.
- NOAA National Centers for Environmental Information. https://www.ncei.noaa.gov — accessed 27 October 2026.
- Copernicus Emergency Management Service. https://emergency.copernicus.eu — accessed 27 October 2026.
- Open Supply Hub. https://opensupplyhub.org — accessed 27 October 2026.
- NetworkX. https://networkx.org — accessed 27 October 2026.

> **Note:** Scenario probabilities and loss estimates in this module are teaching assumptions. A probability without a documented basis is an opinion; label it as one before it reaches a board.

---

> **Next:** [Module 15](15_manufacturing_production_planning.md) — recovery plans become production plans, and the next constraint is the line, the changeover, and the schedule.
