# Module 35 — Public Sector, Defence, and Humanitarian Logistics

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (12 lecture, 10 lab, 4 project)
> **Track fit:** Advanced elective; Logistics and Public Policy tracks
> **Prerequisites:** Modules 11, 12, 14, 14A, 29
> **Tools:** Python, OR-Tools, QGIS, synthetic mission and pipeline data
> **Industry question:** "How do you supply a mission you cannot forecast, under rules you cannot break?"

---

## 1. Why industry cares

Public-sector supply chains — government, defence, humanitarian, and emergency response — are judged on **mission assurance**, not margin. The customer is a population, a deployed force, or a public service. The constraints are political, legal, and ethical as much as economic.

Three features distinguish them:

- **Demand is event-driven and politically shaped.** A flood, a conflict, or a mandate creates demand that no historical series predicts well.
- **Procurement is rule-bound.** Open tendering, framework agreements, audit, and anti-corruption rules limit the solution space.
- **Failure is public.** A stockout in a hospital or a shelter is a news event and a political event, not a lost sale.

The analytics problem is **planning under deep uncertainty with hard procedural constraints**, where the value of a model is often transparency and speed rather than precision.

## 2. Learning outcomes

By the end of this module, you can:

- Map a public-sector supply chain from budget to beneficiary or end user.
- Design a procurement approach for a requirement with uncertain volume and strict rules.
- Build a humanitarian pipeline model with lead time, pre-positioning, and funding constraints.
- Estimate needs from a needs assessment and convert them into commodity requirements.
- Plan a defence spares and availability strategy, including performance-based logistics.
- Evaluate strategic stockpiles for emergencies and national resilience.
- Design a last-mile distribution network for a fragile or low-capacity setting.
- Use cash-based and in-kind assistance together and model the trade-off.
- Build reporting and audit analytics that satisfy donors and oversight bodies.
- Recognise the ethical and security risks of humanitarian and defence data.

## 3. Mental model

```text
   BUDGET ──> PROCUREMENT ──> SUPPLIER / DONOR ──> PIPELINE ──> FIELD / UNIT ──> BENEFICIARY / MISSION
     │            │                 │                  │              │
   mandate      rules            contracts         transport       distribution
   and law      and audit        and funding       and customs     and security
```

Two supply chains sit underneath:

- **Push:** pre-positioned stock and framework contracts moved on a plan.
- **Pull:** requisitions from field units or partners pulled through the pipeline.

Most public-sector supply chains are hybrid, and the hybrid is where the analytics gets hard.

### 3.1 The mission clock

| Clock | Defence | Humanitarian |
|---|---|---|
| Operational | Mission start, deployment | Emergency onset, funding window |
| Supply | Lead time, sustainment | Procurement and pipeline |
| Political | Mandate, budget cycle | Donor visibility and media |

A supply chain that is optimal on cost but misses the operational clock has failed on the only measure that matters.

## 4. Core concepts

### 4.1 Public procurement

Public buying must be **transparent, competitive, and defensible**. Instruments:

| Instrument | Use | Analytics question |
|---|---|---|
| Open tender | Large, defined requirement | Bid evaluation and whole-life cost |
| Framework agreement | Repeat requirement | Call-off volume and price tiers |
| Restricted tender | Complex or sensitive requirement | Supplier qualification |
| Direct award | Urgency or sole source | Justification and audit |
| Framework with mini-competition | Multiple suppliers, local choice | Award rule and fairness |

Rules come from the **WTO Government Procurement Agreement**, EU procurement directives, national law, and donor rules. The analytics must be auditable: why this supplier, why this price, why this quantity.

### 4.2 Whole-life cost, not lowest price

Public procurement often mandates evaluation on **whole-life cost** or **most economically advantageous tender (MEAT)**. The model:

\[
\text{Whole-life cost} = \text{purchase} + \text{operating} + \text{maintenance} + \text{disposal} - \text{residual value}
\]

For a fleet or equipment purchase, this can reverse the ranking versus lowest price. The model must state discount rate, horizon, and assumptions.

### 4.3 Defence supply and availability

Defence supply chains support platforms — aircraft, ships, vehicles, radios — over decades. Key concepts:

- **Availability:** the share of time a platform is mission-capable.
- **Performance-based logistics (PBL):** the supplier is paid for availability, not parts.
- **Contractor logistics support:** industry operates the supply chain.
- **Security of supply:** single-source and obsolescence risk.
- **ITAR and export control:** parts and data are controlled.
- **Obsolescence:** components go out of production before the platform retires.

The analytics problem is sparing for a fleet with uncertain failure and a long horizon, under availability targets and budget cycles.

### 4.4 Strategic stockpiles

Governments hold reserves for emergencies: medical countermeasures, fuel, food, minerals. The questions:

- What events are we sizing for?
- What is the coverage target — days, population, critical functions?
- How do we rotate stock before expiry?
- Who pays for readiness?
- How do we release stock without creating dependence?

A stockpile is an insurance policy with a carrying cost. The analysis is a cost-of-preparedness versus cost-of-shortfall comparison, not an inventory optimisation.

### 4.5 Humanitarian needs and pipeline

Humanitarian supply starts with a **needs assessment**:

\[
\text{Requirement} = \text{affected population} \times \text{standard per person} \times \text{duration} - \text{local capacity}
\]

The pipeline must cover lead time plus distribution:

\[
\text{Pipeline stock} = \text{daily requirement} \times \text{pipeline length}
\]

Pre-positioning at regional hubs shortens the first response but ties up funding and risks expiry. The trade-off is explicit.

### 4.6 Last mile in fragile settings

Last-mile distribution differs from commercial delivery:

- Roads may be impassable or insecure.
- Partners (local NGOs, community groups) do the distribution.
- Beneficiary lists may be incomplete or contested.
- Cash transfers may be preferable to in-kind.
- Protection and safeguarding are operational constraints.

The network design problem includes **access, security, and partner capacity**, not just distance.

### 4.7 Cash versus in-kind

| Mode | Strength | Weakness |
|---|---|---|
| In-kind | Controlled quality, visible | Logistics cost, market distortion |
| Cash | Local markets, dignity, speed | Inflation, market capacity, targeting |
| Voucher | Choice with control | Supplier network required |
| Hybrid | Balances | Complexity and reconciliation |

The analytics question is not ideological. It is: does the local market have the supply, and can beneficiaries access it?

### 4.8 Donor reporting and audit

Humanitarian funding comes with reporting obligations: quantities delivered, beneficiaries reached, costs, and outcomes. Standards:

- **IATI** for aid transparency data.
- **HXL** for humanitarian data exchange.
- **Sphere** standards for humanitarian response quality.
- **Cluster system** for coordination by sector.

Analytics must reconcile **procurement, pipeline, distribution, and beneficiary** data. The common failure is a report that cannot be traced back to delivery.

### 4.9 Worked example — humanitarian food pipeline

A response covers 100,000 people for 90 days. The ration is 0.5 kg per person per day.

\[
\text{Daily requirement} = 100{,}000 \times 0.5 = 50{,}000 \text{ kg} = 50 \text{ t/day}
\]

\[
\text{Total requirement} = 50 \times 90 = 4{,}500 \text{ t}
\]

If the pipeline from procurement to distribution is 60 days, the pipeline must hold:

\[
50 \times 60 = 3{,}000 \text{ t}
\]

The first 30 days must come from pre-positioned stock or local purchase, because procurement cannot deliver earlier. If the regional hub holds 1,500 t, the gap is 1,500 t that must be sourced locally or through a donor in kind.

This is why pre-positioning matters: it is the only supply that exists in the first weeks.

### 4.10 Worked example — defence fleet availability

A fleet of 40 aircraft flies 2,000 hours per year each. A critical component fails once every 1,500 flight hours, so expected demand is:

\[
40 \times 2{,}000 / 1{,}500 = 53.3 \text{ failures per year}
\]

If repair turnaround is 90 days and the component is not repairable on base, the pipeline requirement is approximately:

\[
53.3 \times (90/365) \approx 13.1 \text{ units in repair}
\]

To meet a 90% availability target, the fleet needs enough spares to cover demand during turnaround plus a buffer for variability. If the target requires 15 units of pipeline stock and the current pool is 10, the gap is 5 units — or a shorter repair turnaround. The cheapest fix is often repair capacity, not more stock.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | Source | Use |
|---|---|---|
| Requirement | Needs assessment / capability plan | Demand |
| Budget | Finance system | Constraint |
| Contract | Procurement system | Price and terms |
| Purchase order | ERP | Pipeline |
| Shipment | TMS / 3PL | Lead time and status |
| Warehouse | WMS | Stock and expiry |
| Distribution | Partner reports | Beneficiary reach |
| Beneficiary | Registration / targeting | Allocation |
| Security | Incident reports | Access constraints |
| Donor | Grant reporting | Compliance |

### 5.2 Standards

- **WTO Government Procurement Agreement.** https://www.wto.org/english/tratop_e/gproc_e/gp_gpa_e.htm
- **EU public procurement directives** (2014/24/EU, 2014/25/EU).
- **UN Procurement Manual** and **UNGM** supplier portal. https://www.ungm.org/
- **Sphere Handbook** for humanitarian standards. https://spherestandards.org/
- **IATI** aid transparency standard. https://iatistandard.org/
- **HXL** humanitarian exchange language. https://hxlstandard.org/
- **ITAR** and national export control regimes.
- **NATO** codification and supply standards where applicable.

### 5.3 Workflow

1. Define requirement and budget.
2. Choose procurement instrument.
3. Contract and order.
4. Move through the pipeline.
5. Clear customs and receive.
6. Store and rotate.
7. Distribute or issue.
8. Report and reconcile.
9. Audit and learn.

## 6. Industry case studies

### 6.1 Humanitarian response to a sudden-onset disaster

**Context:** An earthquake displaces hundreds of thousands of people.
**Intervention:** Cluster coordination, pre-positioned stock, and local procurement.
**Evidence:** Evaluations and humanitarian reports (grade B/C); outcomes vary by response.
**Outcome:** Pre-positioned stock and local markets supplied the first weeks; later supply depended on pipeline and funding.
**What transferred:** the first 72 hours are decided by pre-positioning, not by procurement speed.
**What did not:** specific tonnages and timelines.

### 6.2 Strategic medical stockpile rotation

**Context:** A national stockpile of medical countermeasures.
**Intervention:** Rotating stock into routine supply before expiry, with replenishment.
**Evidence:** Government audit and programme reporting (grade B).
**Outcome:** Rotation reduced expiry write-off but required coordination with routine procurement.
**What transferred:** a stockpile without a rotation rule becomes a write-off.
**What did not:** specific costs and quantities.

### 6.3 Performance-based logistics in defence

**Context:** Availability-based contracts for platform support.
**Intervention:** Supplier paid for mission-capable rates rather than parts.
**Evidence:** Government audit and academic review (grade B/C).
**Outcome:** Availability improved in some contracts and worsened in others; measurement and risk allocation were decisive.
**What transferred:** PBL works when availability is measurable and the supplier controls the drivers.
**What did not:** specific rates and savings.

### 6.4 A composite failure — the report that could not be traced

**Context:** A donor required beneficiary counts by district.
**Intervention:** Reports compiled from partner spreadsheets with no common identifier.
**Evidence:** Composite teaching case.
**Outcome:** Numbers could not be reconciled to deliveries, and the audit qualified the report.
**What transferred:** reporting must be designed with the supply chain data model, not after it.
**What did not:** the specific audit outcome.

## 7. Labs

### 7.1 Lab 35.1 — Procurement evaluation

Evaluate three bids on a whole-life-cost basis with discounting, risk, and qualitative criteria. Produce an auditable scoring model.

**Deliverable:** Evaluation model and award memo.

### 7.2 Lab 35.2 — Humanitarian pipeline

Build a pipeline model for a 90-day response with pre-positioning, local purchase, and donor in-kind. Identify the first-week gap.

**Deliverable:** Pipeline model and sourcing plan.

### 7.3 Lab 35.3 — Defence spares and availability

Model fleet availability for a critical component with failure, repair turnaround, and stock. Compare more stock versus faster repair.

**Deliverable:** Availability model and recommendation.

### 7.4 Lab 35.4 — Last-mile network

Design a distribution network in a low-capacity setting with road access, security, and partner capacity constraints.

**Deliverable:** Network design and risk assessment.

### 7.5 Lab 35.5 — Reporting reconciliation

Reconcile procurement, delivery, and beneficiary data for a synthetic programme. Identify the discrepancies and design a control.

**Deliverable:** Reconciliation report and control design.

## 8. Assignment / project

Design a public-sector supply chain for one of: a disaster response, a defence sustainment programme, or a national stockpile.

**Required components:**

1. Requirement and demand assessment.
2. Procurement strategy and rule compliance.
3. Pipeline and pre-positioning plan.
4. Distribution and last-mile design.
5. Funding and donor reporting.
6. Risk, security, and ethics assessment.
7. Metrics and audit trail.
8. Model, assumptions, and limits.

**Length:** 10–14 pages plus model.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Requirement | 20% | States need | Needs-based estimate | Uncertainty and scenario range |
| Procurement | 20% | Describes tender | Rule-compliant process | Whole-life and risk evaluation |
| Pipeline | 20% | Lists stock | Lead-time pipeline | Pre-positioning trade-off |
| Last mile | 15% | Describes delivery | Network design | Access, security, partner capacity |
| Reporting | 15% | Produces numbers | Reconciled report | Auditable data model |
| Ethics and risk | 10% | Mentions risk | Mitigation plan | Protection and do-no-harm analysis |

## 10. Regulatory, ethical, and commercial lens

- **Legality:** procurement must follow the applicable regime; emergency exceptions must be documented.
- **Anti-corruption:** public buying is a corruption risk; segregation of duties and audit trails are controls.
- **Do no harm:** humanitarian data and distribution can expose beneficiaries to risk.
- **Neutrality and impartiality:** humanitarian actors must be seen as independent.
- **Export control:** defence items and technical data are controlled.
- **Data protection:** beneficiary data is personal and sensitive; minimise and secure it.
- **Local economy:** procurement can strengthen or distort local markets.
- **Transparency:** donors and citizens have a right to see how public money was spent.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No pre-positioning | First-week gap | Pipeline too long | Delayed response | Regional hubs |
| Lowest-price award | Poor outcomes | Whole-life ignored | Higher cost later | MEAT evaluation |
| Volume uncertainty | Contract disputes | No call-off mechanism | Price or supply failure | Framework with tiers |
| Stockpile expiry | Write-off rising | No rotation rule | Waste and shortage | Rotation policy |
| Availability target unmeasured | Disputed PBL | No data | Payment disputes | Metric definition |
| Last-mile assumption | Access denied | Security or roads | Undelivered aid | Contingency routing |
| Cash without market | Inflation | Supply not available | Reduced purchasing power | Market assessment |
| Reporting gap | Audit qualification | No common identifier | Funding risk | Data model |
| Single source | Obsolescence | Supplier exit | Sustainment gap | Dual source and lifetime buy |
| Data exposure | Beneficiary list leak | Poor access control | Harm | Data protection |

## 12. Interview and discussion questions

1. How is public procurement different from commercial procurement? *(Hint: rules, transparency, audit.)*
2. What is whole-life cost, and why does it matter? *(Hint: purchase, operate, maintain, dispose.)*
3. Why is pre-positioning decisive in humanitarian response? *(Hint: lead time exceeds the first-response window.)*
4. How do you estimate humanitarian requirements? *(Hint: population × standard × duration − capacity.)*
5. What is performance-based logistics? *(Hint: pay for availability, not parts.)*
6. How do you size a strategic stockpile? *(Hint: scenario, coverage, rotation, cost of readiness.)*
7. When is cash better than in-kind? *(Hint: market capacity, access, dignity.)*
8. What makes last-mile distribution hard in fragile settings? *(Hint: access, security, partners.)*
9. How do you make donor reporting auditable? *(Hint: common identifiers and reconciliation.)*
10. What is the risk of a single-source defence component? *(Hint: obsolescence and security of supply.)*
11. How would you measure mission assurance? *(Hint: availability, response time, coverage.)*
12. What are the ethical limits of analytics in humanitarian settings? *(Hint: do no harm, consent, targeting.)*

## 13. Key terms

| Term | Definition |
|---|---|
| MEAT | Most economically advantageous tender |
| Framework agreement | Pre-agreed terms for repeat call-offs |
| Whole-life cost | Total cost over the asset or contract life |
| Performance-based logistics | Contract paying for availability outcomes |
| Mission-capable rate | Share of platforms available for tasking |
| Obsolescence | Component no longer produced or supported |
| Strategic stockpile | Government reserve for emergencies |
| Needs assessment | Structured estimate of humanitarian need |
| Pipeline | Stock in transit and in process |
| Pre-positioning | Placing stock ahead of an event |
| Last mile | Final distribution to beneficiaries or users |
| Cash transfer | Direct money instead of goods |
| Sphere standards | Humanitarian quality and accountability standards |
| IATI | International Aid Transparency Initiative |
| HXL | Humanitarian Exchange Language |
| Cluster system | Sector coordination mechanism in humanitarian response |
| ITAR | US International Traffic in Arms Regulations |
| Do no harm | Principle of avoiding negative impact on affected people |

## 14. Sources and further reading

### Standards and regulation

- WTO Government Procurement Agreement. https://www.wto.org/english/tratop_e/gproc_e/gp_gpa_e.htm
- EU Directive 2014/24/EU on public procurement. https://eur-lex.europa.eu/
- UN Global Marketplace. https://www.ungm.org/
- Sphere Handbook. https://spherestandards.org/
- IATI Standard. https://iatistandard.org/
- HXL. https://hxlstandard.org/
- US Directorate of Defense Trade Controls, ITAR. https://www.pmddtc.state.gov/

### Reports and official sources

- UN OCHA, humanitarian response plans and situation reports. https://www.unocha.org/
- WFP, logistics and supply chain reports. https://www.wfp.org/
- National audit offices, defence and procurement reports (for example US GAO, UK NAO).
- World Bank, public procurement guidance. https://www.worldbank.org/

### Peer-reviewed and professional

- Van Wassenhove, L. N., "Humanitarian aid logistics: supply chain management in high gear."
- Tomasini, R., Van Wassenhove, L. N., *Humanitarian Logistics*.
- Kovács, G., Spens, K., research on humanitarian supply chains.
- Pohl, E., et al., defence spares and availability modelling.

### Industry

- Fritz Institute, humanitarian logistics research.
- Chartered Institute of Procurement & Supply, public procurement guidance. https://www.cips.org/

> **Next:** [Module 36](36_global_freight_ports_trade_policy.md) moves to the physical and regulatory network that carries all of this trade across borders.
