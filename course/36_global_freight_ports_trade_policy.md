# Module 36 — Global Freight, Ports, and Trade Policy

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (12 lecture, 10 lab, 4 project)
> **Track fit:** Advanced elective; Logistics and Global Operations tracks
> **Prerequisites:** Modules 11, 12, 14, 19
> **Tools:** Python, AIS and port-call data, customs tariff extracts, landed-cost models
> **Industry question:** "What does it actually cost, end to end, to move a container from factory to shelf?"

---

## 1. Why industry cares

Global freight is the physical layer of international trade and the place where supply chain plans meet reality. A container that misses a vessel, a customs classification that is wrong, or a port that congests can erase the margin on an entire order.

Four forces shape the domain:

- **Rates are volatile.** Ocean spot rates have moved by multiples within months, while contract rates reset annually.
- **Transit time is variable.** Port congestion, blank sailings, weather, and chokepoint disruption make nominal lead times unreliable.
- **Trade policy is an operating constraint.** Tariffs, rules of origin, sanctions, and carbon rules change what can move and at what cost.
- **The port is a bottleneck with its own physics.** Berth, crane, yard, and gate capacity determine dwell time, not the shipping line alone.

The analytics problem is **total landed cost and reliability under policy and network uncertainty**.

## 2. Learning outcomes

By the end of this module, you can:

- Build a total landed cost model including freight, insurance, duty, fees, and financing.
- Apply Incoterms correctly and identify who bears which cost and risk.
- Classify goods and estimate duty under a tariff schedule.
- Determine origin and apply a trade agreement's rules of origin.
- Explain port operations and diagnose congestion from vessel and yard data.
- Measure transit-time variability and translate it into inventory.
- Compare ocean, air, rail, and road on cost, time, and emissions.
- Explain demurrage, detention, and free-time economics.
- Assess exposure to chokepoints, sanctions, and trade-policy change.
- Build a freight procurement and hedging strategy.

## 3. Mental model

```text
   ORIGIN                          TRANSIT                         DESTINATION
   factory ──> inland ──> port ──> ocean ──> port ──> inland ──> DC / customer
     │          │          │         │         │          │
   export     drayage    terminal   vessel   terminal    drayage
   customs    booking    handling   schedule  customs     delivery
   docs       container  berth/crane          release     appointment
```

At every arrow there is a **document, a cost, a risk transfer, and a clock**. The landed cost and the lead time are the sum of all of them.

### 3.1 The four clocks of international freight

| Clock | Measured from | Typical failure |
|---|---|---|
| Production | Order release | Late finish |
| Origin dwell | Pickup to vessel departure | Missed cutoff |
| Transit | Departure to arrival | Congestion, blank sailing |
| Destination dwell | Arrival to delivery | Customs, exam, yard |

Most "shipping delays" are actually origin or destination dwell, not ocean transit.

## 4. Core concepts

### 4.1 Incoterms

Incoterms 2020 define cost, risk, and obligation transfer between seller and buyer. The common ones:

| Term | Seller delivers | Risk passes | Freight paid by |
|---|---|---|---|
| EXW | At seller's premises | At pickup | Buyer |
| FCA | To carrier nominated by buyer | On handover | Buyer |
| FOB | On board vessel | On loading | Buyer |
| CIF | On board, insurance to destination | On loading | Seller |
| CFR | On board | On loading | Seller |
| DAP | At named place, not unloaded | At destination | Seller |
| DDP | At destination, duty paid | At destination | Seller |

The analytical trap: **CIF and FOB look similar but allocate risk differently from cost**. A landed-cost model must state which term the price is quoted on.

### 4.2 Total landed cost

\[
\text{Landed cost} = \text{goods} + \text{freight} + \text{insurance} + \text{duty} + \text{fees} + \text{financing} + \text{risk}
\]

Duty depends on **classification, valuation, and origin**:

\[
\text{Duty} = \text{customs value} \times \text{duty rate}
\]

Customs value is usually transaction value adjusted to the Incoterm, plus assists and royalties where applicable.

### 4.3 Classification, valuation, and origin

| Element | Question | Consequence of error |
|---|---|---|
| HS classification | What is the product? | Wrong duty, penalties |
| Customs valuation | What is it worth? | Under/overpayment, penalty |
| Origin | Where was it made? | Lost preference or illegal claim |
| Rules of origin | Does it qualify? | Duty not avoided |
| Documentation | Can it be proved? | Delay and inspection |

**Rules of origin** can require tariff shift, regional value content, or specific processing. A product assembled in a country is not automatically "originating" there.

### 4.4 Port operations

A container port is a set of coupled resources:

- **Berth:** vessel slot along the quay.
- **Quay crane:** moves boxes between vessel and dock.
- **Yard:** temporary storage by block and stack.
- **Gate and rail:** landside evacuation.

Metrics: berth productivity (moves per hour), vessel turnaround, yard dwell, gate turn time, and crane intensity. Congestion appears when arrival rate exceeds evacuation rate; the yard fills, dwell rises, and vessels wait.

The analytics question is often **forecasting dwell and demurrage risk**, not optimising the port.

### 4.5 Demurrage and detention

| Charge | Applies to | Clock starts |
|---|---|---|
| Demurrage | Container inside the terminal | After free time from discharge |
| Detention | Container outside the terminal | After free time from pickup |

Free time is negotiated. A 5-day free time with a 9-day dwell means 4 chargeable days. For five containers at $75 per day each:

\[
5 \times 4 \times 75 = \$1{,}500
\]

Demurrage is a symptom: the cause is customs, documentation, yard capacity, or inland transport.

### 4.6 Transit-time variability

Nominal transit time is a mean, not a promise. If ocean transit is 30 days with a standard deviation of 4, and air is 5 days with a standard deviation of 1, the safety stock difference is large:

\[
SS = z \times \sigma_{LT} \times \text{daily demand}
\]

For daily demand of 100 units, z = 1.65:

- Ocean: 1.65 × 4 × 100 = 660 units.
- Air: 1.65 × 1 × 100 = 165 units.

The inventory saving from air may or may not offset the freight premium. That is the modal-choice calculation.

### 4.7 Freight rates and procurement

| Market | Pricing | Contract |
|---|---|---|
| Ocean spot | Weekly index-linked | Per shipment |
| Ocean contract | Annual or quarterly | Minimum quantity |
| Air | Spot and capacity agreements | Seasonal |
| Road | Tariff and fuel surcharge | Annual |
| Rail | Block train and slot | Annual |

Freight procurement is a portfolio problem: contract for base volume, spot for peaks, and index-linked clauses for volatility. Bunker adjustment, peak season surcharge, and congestion surcharge must be modelled explicitly.

### 4.8 Chokepoints and network risk

| Chokepoint | Exposure | Mitigation |
|---|---|---|
| Suez Canal | Asia–Europe | Cape routing, air, nearshore |
| Panama Canal | Asia–US East | Suez or US West, rail |
| Strait of Hormuz | Energy | Alternative supply |
| Red Sea | Asia–Europe | Cape, insurance, rerouting |
| Malacca Strait | Asia trade | Diversified ports |

Disruption shifts transit time, insurance, and cost simultaneously. A scenario model needs all three.

### 4.9 Trade policy as an operating constraint

- **Tariffs:** ad valorem, specific, or compound; changes can be announced with little notice.
- **Anti-dumping and countervailing duties:** product- and country-specific.
- **Sanctions and export controls:** restrict parties, destinations, and items.
- **Forced-labour rules:** import bans with supply-chain tracing (for example UFLPA).
- **Carbon border rules:** embedded emissions reporting and cost (for example CBAM).
- **Free trade agreements:** preferential rates if origin rules are met.

Analytics must maintain a **tariff and policy register** with effective dates and a product-to-code mapping.

### 4.10 Modal choice and emissions

| Mode | Cost | Speed | Variability | Emissions per tonne-km |
|---|---|---|---|---|
| Ocean | Lowest | Slowest | High | Lowest |
| Rail | Low–medium | Medium | Medium | Low |
| Road | High | Fast | Medium | High |
| Air | Highest | Fastest | Low | Highest |

The choice is multi-criteria: cost, lead time, variability, emissions, and customer promise. A single cost-per-kg comparison ignores the inventory and service consequences.

### 4.11 Worked example — landed cost

A supplier quotes FOB $10,000 for 1,000 units. Ocean freight is $1,200, insurance is 0.5% of FOB plus freight, duty is 6%, and port and brokerage fees are $150. Payment terms require 30 days of financing at 6% annual.

\[
\text{Insurance} = 0.005 \times (10{,}000 + 1{,}200) = 56
\]

\[
\text{Duty} = 0.06 \times (10{,}000 + 1{,}200 + 56) = 675.36
\]

\[
\text{Financing} \approx 10{,}000 \times 0.06 \times 30/365 = 49.32
\]

\[
\text{Landed cost} = 10{,}000 + 1{,}200 + 56 + 675.36 + 150 + 49.32 = 12{,}130.68
\]

Landed cost per unit is $12.13, versus the $10.00 FOB price. A 21% difference from the quoted price is normal. Any sourcing decision that compares FOB prices across countries without landed cost is comparing the wrong numbers.

### 4.12 Worked example — demurrage exposure

A shipment of 40 containers arrives with 5 free days. The importer expects a 12-day dwell because of a customs exam. At $90 per container per day for demurrage:

\[
40 \times (12 - 5) \times 90 = \$25{,}200
\]

If the exam is caused by a classification error, the cost of the error is not the duty difference; it is the demurrage plus the delay. That is why classification quality is a supply chain control, not a finance detail.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | Source | Use |
|---|---|---|
| Vessel positions | AIS | ETA and congestion |
| Port calls | Port community systems | Berth and dwell |
| Container status | Carrier API / EDI | Milestones |
| Booking and B/L | TMS / forwarder | Shipment identity |
| Customs declaration | Broker / customs system | Duty and release |
| Tariff schedule | Customs authority | Duty rate |
| Freight invoices | Carrier / forwarder | Cost audit |
| Demurrage invoices | Terminal / carrier | Charge validation |
| Trade policy | Regulator / trade press | Policy register |

### 5.2 Standards and documents

- **Incoterms 2020**, ICC. https://iccwbo.org/
- **HS nomenclature**, World Customs Organization. https://www.wcoomd.org/
- **WCO SAFE Framework** and **AEO** authorised economic operator programmes.
- **IMO** conventions, including SOLAS and MARPOL; **IMO 2020** sulphur cap.
- **ISPS Code** for port and ship security.
- **Bill of Lading**, **air waybill**, **certificate of origin**, **packing list**, **commercial invoice**.
- **UCP 600** for letters of credit.
- **CBAM** and EU trade rules; national customs law.

### 5.3 Workflow

1. Classify the product and confirm origin.
2. Agree Incoterms and contract terms.
3. Book freight and plan the route.
4. Prepare and check documentation.
5. Export customs clearance.
6. Ocean or air transit.
7. Import clearance and duty payment.
8. Terminal handling and inland delivery.
9. Cost audit and dispute resolution.

## 6. Industry case studies

### 6.1 Suez Canal blockage (2021)

**Context:** A container vessel grounded in the Suez Canal and blocked traffic for days.
**Intervention:** Salvage, rerouting, and network recovery.
**Evidence:** Official reports and trade data (grade B/C).
**Outcome:** Transit times extended, rates rose, and operators rerouted around the Cape.
**What transferred:** a single chokepoint can reset the transit-time distribution for a large share of trade.
**What did not:** the specific duration and cost.

### 6.2 COVID-19 container rate spike

**Context:** Demand shifted to goods while port and vessel capacity were constrained.
**Intervention:** Contract and spot purchasing, chartering, and equipment repositioning.
**Evidence:** Freight index data and regulatory reviews (grade B/C).
**Outcome:** Spot rates rose several-fold, and contract coverage became a competitive advantage.
**What transferred:** freight is a strategic input, not a tactical cost line.
**What did not:** specific rate levels.

### 6.3 US–China tariff changes

**Context:** Tariffs were imposed and changed on many product categories.
**Intervention:** Reclassification, sourcing shifts, and origin planning.
**Evidence:** Customs and trade reporting (grade B/C).
**Outcome:** Some supply moved to other countries; some costs were absorbed; some were passed through.
**What transferred:** tariff exposure must be modelled at the product–origin level.
**What did not:** specific rates and product lists, which change.

### 6.4 A composite failure — the reclassification that cost more than the duty

**Context:** An importer reclassified a product to a lower-duty code without supporting documentation.
**Intervention:** None; the broker filed and the goods shipped.
**Evidence:** Composite teaching case.
**Outcome:** Customs examined the shipment, assessed the correct duty plus penalties, and held the goods.
**What transferred:** classification is a documented technical judgement, not a guess at the lowest rate.
**What did not:** the specific penalty.

## 7. Labs

### 7.1 Lab 36.1 — Landed cost model

Build a landed-cost model for 20 SKUs from three origins with Incoterms, freight, insurance, duty, and financing. Compare FOB and landed cost rankings.

**Deliverable:** Model and sourcing recommendation.

### 7.2 Lab 36.2 — Classification and duty

Classify 15 products into HS codes, estimate duty, and document the reasoning. Identify the highest-risk classifications.

**Deliverable:** Classification file and risk register.

### 7.3 Lab 36.3 — Transit variability and safety stock

Using shipment milestone data, estimate transit-time distributions by lane and mode. Translate into safety stock and compare with air freight.

**Deliverable:** Variability analysis and modal recommendation.

### 7.4 Lab 36.4 — Port congestion diagnosis

Using vessel and port-call data, measure berth waiting, crane productivity, and yard dwell. Identify whether the bottleneck is berth, crane, yard, or gate.

**Deliverable:** Diagnostic report.

### 7.5 Lab 36.5 — Policy shock scenario

Model a tariff increase and a chokepoint disruption together. Estimate landed-cost change, transit-time change, and mitigation options.

**Deliverable:** Scenario model and response plan.

## 8. Assignment / project

Design the international flow for one product family.

**Required components:**

1. Product classification and origin.
2. Incoterms and contract structure.
3. Landed-cost model.
4. Modal and routing choice.
5. Transit-time variability and inventory impact.
6. Port and customs process.
7. Trade-policy exposure and mitigation.
8. Emissions estimate.
9. Risks, assumptions, and limits.

**Length:** 10–14 pages plus model.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Landed cost | 25% | Adds freight | Full cost model | Duty, financing, and risk |
| Trade compliance | 20% | Names HS code | Classification with rationale | Origin and agreement analysis |
| Freight and routing | 20% | Chooses mode | Cost-time comparison | Variability and chokepoint risk |
| Port and customs | 15% | Describes process | Dwell and demurrage analysis | Bottleneck diagnosis |
| Policy and risk | 10% | Lists tariffs | Policy register | Scenario and mitigation |
| Communication | 10% | Descriptive | Clear recommendation | Decision-ready and honest |

## 10. Regulatory, ethical, and commercial lens

- **Customs law:** false classification, valuation, or origin is a legal offence.
- **Sanctions and export control:** screening is mandatory; violations carry severe penalties.
- **Forced labour:** import bans require supply-chain tracing, not only supplier declarations.
- **Environment:** sulphur limits, carbon reporting, and border carbon rules are expanding.
- **Security:** ISPS, AEO, and cargo screening obligations apply to ports and carriers.
- **Corruption:** freight and customs intermediation is a bribery risk.
- **Labour:** seafarer welfare and port labour conditions are part of the supply chain's social footprint.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| FOB-only comparison | Supplier looks cheapest | Landed cost ignored | Wrong sourcing | Landed-cost model |
| Wrong HS code | Duty lower than expected | Misclassification | Penalty and delay | Classification review |
| Origin overclaim | Preference claimed | Rules not met | Duty recovered | Origin file |
| Free time ignored | Demurrage invoices | Dwell exceeds free time | Cost | Dwell forecasting |
| Nominal lead time | Stockouts | Variability ignored | Service failure | Distribution-based planning |
| Spot-only freight | Rate shock | No contract cover | Cost spike | Contract portfolio |
| Single chokepoint | No alternative route | Geographic concentration | Disruption | Route diversity |
| Policy lag | Sudden duty change | No monitoring | Margin loss | Policy register |
| Documentation error | Customs exam | Missing certificate | Delay and cost | Document checklist |
| Emissions omission | Customer or regulator challenge | Not measured | Compliance risk | Emissions accounting |

## 12. Interview and discussion questions

1. What is the difference between FOB and CIF? *(Hint: cost and risk allocation.)*
2. Why is landed cost higher than the FOB price? *(Hint: freight, insurance, duty, fees, financing.)*
3. What determines duty? *(Hint: classification, valuation, origin.)*
4. Why is origin not the same as where the product was assembled? *(Hint: rules of origin.)*
5. What causes port congestion? *(Hint: arrival versus evacuation capacity.)*
6. What is the difference between demurrage and detention? *(Hint: inside versus outside the terminal.)*
7. How does transit-time variability affect inventory? *(Hint: safety stock proportional to standard deviation.)*
8. When is air freight justified? *(Hint: value density, service, inventory saving.)*
9. How do you manage chokepoint risk? *(Hint: route diversity, inventory, nearshore.)*
10. How would you build a trade-policy register? *(Hint: product, origin, rate, effective date, source.)*
11. What is the emissions trade-off between air and ocean? *(Hint: air is far more carbon-intensive per tonne-km.)*
12. What is the biggest compliance risk in global freight? *(Hint: classification, origin, sanctions.)*

## 13. Key terms

| Term | Definition |
|---|---|
| Incoterms | Standard trade terms for cost and risk transfer |
| FOB | Free on board |
| CIF | Cost, insurance, and freight |
| DDP | Delivered duty paid |
| HS code | Harmonised System tariff classification |
| Landed cost | Total cost of goods delivered |
| Rules of origin | Criteria for preferential origin |
| Demurrage | Charge for container overstay in terminal |
| Detention | Charge for container overstay outside terminal |
| Free time | Days before demurrage or detention applies |
| Dwell time | Time a container or vessel spends in port |
| Berth productivity | Container moves per vessel-hour |
| Blank sailing | Cancelled vessel call |
| AIS | Automatic Identification System for vessels |
| B/L | Bill of lading |
| AEO | Authorised economic operator |
| CBAM | EU Carbon Border Adjustment Mechanism |
| UFLPA | US Uyghur Forced Labor Prevention Act |
| Chokepoint | Narrow passage critical to trade |

## 14. Sources and further reading

### Standards and regulation

- ICC, Incoterms 2020. https://iccwbo.org/
- World Customs Organization, HS nomenclature. https://www.wcoomd.org/
- WTO trade facilitation and tariff data. https://www.wto.org/
- IMO, MARPOL and sulphur limits. https://www.imo.org/
- EU CBAM. https://taxation-customs.ec.europa.eu/
- US CBP, UFLPA and import requirements. https://www.cbp.gov/
- ICC, UCP 600.

### Data and reports

- UNCTAD, Review of Maritime Transport. https://unctad.org/
- World Bank, Container Port Performance Index. https://www.worldbank.org/
- US Federal Maritime Commission, ocean freight reports. https://www.fmc.gov/
- Drewry and Xeneta freight indices (commercial data; verify licence).
- IEA, transport emissions data. https://www.iea.org/

### Peer-reviewed and professional

- Notteboom, T., Rodrigue, J.-P., port economics and container terminal operations.
- Cariou, P., research on maritime economics and emissions.
- Chopra, S., Meindl, P., *Supply Chain Management*, global logistics chapters.
- Bowersox, D., Closs, D., Cooper, M., *Supply Chain Logistics Management*.

### Industry

- World Shipping Council. https://www.worldshipping.org/
- International Air Transport Association, cargo guidance. https://www.iata.org/
- Port authority annual reports and performance dashboards.

> **Next:** [Module 37](37_circular_economy_reverse_networks.md) turns the network around: returns, repair, remanufacture, and the reverse supply chain.
