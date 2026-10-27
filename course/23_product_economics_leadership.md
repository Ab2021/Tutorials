# Module 23 — Product, Economics, and Leadership

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (14 lecture, 8 lab, 4 case clinic)
> **Track fit:** Common core; required for all tracks
> **Prerequisites:** Modules 00, 07, 10, 21
> **Tools:** Spreadsheet or Python, a value-case template, an ERP or P&L extract
> **Industry question:** "Who pays for this work, what decision changes, what is it worth, and who has to adopt it?"

---

## 1. Why industry cares

Most analytics projects do not fail because the model is wrong. They fail because no one owns the decision, the benefit is never baselined, the planner does not trust the output, or the vendor contract pays for seats while the value comes from inventory. A forecast with better WAPE and no adoption is a cost center.

Industry cares about four questions:

1. **Value:** How much cash, cost, service, or risk does the change create, and over what period?
2. **Cost:** What does it cost to build, integrate, run, and change the organization?
3. **Adoption:** Who uses the output, in which workflow, with what authority to override?
4. **Risk:** What happens if the model is wrong, the vendor fails, or the benefit is double-counted?

This module teaches the commercial and organizational work that turns a model into a funded, adopted, and measured capability. It is the bridge between Module 21 (deployment) and Module 24 (model risk).

## 2. Learning outcomes

By the end of this module, you can:

- Map stakeholders into user, buyer, beneficiary, and blocker, and design for each.
- Build a value case with a baseline, a counterfactual, and a benefit range.
- Calculate ROI, NPV, payback, and total cost of ownership for a planning initiative.
- Quantify working-capital impact from inventory, receivables, and payables.
- Build a cost-to-serve model and use it to prioritize customers and products.
- Evaluate build versus buy and the main vendor business models.
- Design an adoption plan with a named owner, workflow, and override policy.
- Prioritize a portfolio with explicit value, risk, and dependency criteria.
- Write a one-page executive decision memo that survives finance review.

## 3. Mental model

```text
        DECISION
           |
   who decides? ----> USER (runs the workflow)
           |           BUYER (owns the budget)
           |           BENEFICIARY (gets the outcome)
           |           BLOCKER (can stop it)
           v
   what changes? ---> physical flow / information flow / financial flow
           |
   what is it worth? -> revenue, cost, working capital, risk, service, carbon
           |
   what does it cost? -> build + integrate + run + change + retire
           |
   how do we know? ---> baseline, counterfactual, benefits realization
           |
   will they use it? --> workflow, trust, authority, incentives
```

The mental model is a chain, not a checklist. A weak link anywhere breaks the case. A strong ROI with no owner is a proposal. A strong owner with no baseline is a hope.

### 3.1 The four stakeholder roles

| Role | Cares about | Evidence they need | Common failure |
|---|---|---|---|
| User | Workload, trust, exceptions | Accuracy where they act, override path | Tool adds work |
| Buyer | Budget, payback, risk | NPV, TCO, references | Benefit not attributable |
| Beneficiary | Service, cost, cash, risk | Outcome metric and baseline | Benefit claimed, not measured |
| Blocker | Control, compliance, power | Audit trail, legal review, governance | Late veto |

A single person can hold more than one role. The CFO may be buyer and beneficiary; the planning manager may be user and blocker. Name the roles explicitly, because the design changes when the buyer is not the beneficiary.

### 3.2 The value taxonomy

| Value type | Mechanism | Measurement | Time to realize |
|---|---|---|---|
| Cost reduction | Less overtime, expedite, waste | P&L line, activity cost | 3–12 months |
| Working capital | Lower inventory, better terms | Balance sheet, cash flow | 3–18 months |
| Revenue | Better availability, fewer lost sales | Sales, margin, share | 6–18 months |
| Service | OTIF, fill rate, lead time | Operational KPI | 1–6 months |
| Risk avoidance | Fewer disruptions, less exposure | Expected loss, insurance | Ongoing |
| Carbon | Fewer tonnes-km, less waste | Scope 1/3 emissions | 6–24 months |
| Productivity | Planner hours, analyst hours | Time study, headcount | 1–6 months |

Most business cases overstate revenue and understate adoption cost. Cost and working capital are easier to attribute and defend. Start there.

## 4. Core concepts

### 4.1 Baseline and counterfactual

A benefit is a **difference between two futures**, not an absolute outcome.

- **Baseline:** what would happen without the initiative, measured over a defined period.
- **Counterfactual:** the specific alternative you are comparing against (current process, naive forecast, vendor A).
- **Attribution:** the share of the difference that the initiative caused.

| Baseline type | Use | Risk |
|---|---|---|
| Historical average | Stable process | Ignores trend and shocks |
| Trailing 12 months | Seasonal business | Contaminated by one-off events |
| Current-process simulation | Process change | Model risk in the simulation |
| Control group | Pilot or geo test | Spillover and selection |
| Bottom-up standard | New process | Standards may be stale |

**Rule:** the baseline must be frozen before the pilot and signed by finance. A baseline chosen after the result is not a baseline.

### 4.2 ROI, NPV, and payback

Let \(B_t\) be the benefit in period \(t\), \(C_t\) the cost, \(r\) the discount rate, and \(n\) the horizon.

\[
\text{NPV} = \sum_{t=1}^{n} \frac{B_t - C_t}{(1+r)^t} - C_0
\]

\[
\text{ROI} = \frac{\sum_t (B_t - C_t)}{C_0 + \sum_t C_t}
\]

\[
\text{Payback} = \text{first } t \text{ where cumulative net benefit} \ge 0
\]

Industry interpretation:

- Use a discount rate the CFO recognizes. Do not invent one.
- Show NPV and payback together. A high NPV with a five-year payback may still fail a capital committee.
- Separate **hard** benefits (cash, cost) from **soft** benefits (service, risk). Finance discounts soft benefits; say so.
- Include a **ramp**: adoption is rarely 100% in month one.
- Include **run cost** forever, not only in year one.

### 4.3 Worked example — inventory reduction

A retailer holds €40M of inventory at cost. COGS is €240M, so days inventory outstanding:

\[
\text{DIO} = \frac{40}{240} \times 365 = 60.8 \text{ days}
\]

A multi-echelon policy is expected to cut DIO by 6 days. Annual COGS stays flat, so the cash release is:

\[
\Delta \text{Inventory} = \frac{6}{365} \times 240 = €3.95\text{M}
\]

The carrying cost is not the full €3.95M. Use the company's holding rate (capital, storage, obsolescence, insurance). At 18%:

\[
\text{Annual carrying saving} = 3.95 \times 0.18 = €0.71\text{M}
\]

The cash release is €3.95M once; the P&L saving is €0.71M per year. A business case that adds both as annual benefit double-counts.

### 4.4 Total cost of ownership

TCO includes everything the initiative consumes over its life:

| Cost category | Examples |
|---|---|
| Build | Licenses, implementation, integration, model development |
| Data | Acquisition, cleansing, storage, quality monitoring |
| Infrastructure | Compute, cloud, warehouse, monitoring |
| Change | Training, process redesign, backfill, communications |
| Run | Support, upgrades, on-call, vendor management |
| Risk | Audit, validation, insurance, contingency |
| Exit | Data extraction, contract termination, replacement |

A vendor quote is usually 30–50% of TCO. Ask for the rest.

### 4.5 Working capital and cash conversion

\[
\text{Cash-to-cash cycle} = \text{DIO} + \text{DSO} - \text{DPO}
\]

| Lever | Analytics contribution | Constraint |
|---|---|---|
| DIO down | Better forecast, multi-echelon, assortment | Service floor, lead time |
| DSO down | Credit risk, dispute analytics | Customer power |
| DPO up | Payment-term analytics | Supplier viability, law |

Extending DPO is not free. It transfers cost to suppliers, can breach payment regulations, and raises supply risk. Model the second-order effect.

### 4.6 Cost-to-serve

Average cost per order hides cross-subsidy. Cost-to-serve allocates activity costs to customers, products, or channels:

| Activity | Cost driver | Allocation |
|---|---|---|
| Order processing | Orders | Per order |
| Pick and pack | Lines, units | Per line or unit |
| Delivery | Stops, distance, drops | Per stop or km |
| Returns | Returns | Per return |
| Service | Contacts, claims | Per contact |
| Inventory | Average inventory | Holding rate |

A customer with small, frequent, high-return orders can be unprofitable at a healthy gross margin. Cost-to-serve turns that insight into a pricing, MOQ, or channel decision.

### 4.7 Build versus buy

| Factor | Favors build | Favors buy |
|---|---|---|
| Differentiation | Decision is core IP | Standard process |
| Data | Proprietary, sensitive | Vendor can aggregate |
| Speed | No deadline pressure | Need in weeks |
| Talent | Strong internal team | Scarce skills |
| Cost | High volume amortizes | Low volume |
| Control | Regulatory or audit need | Vendor certified |
| Roadmap | Unique workflow | Vendor roadmap fits |

The worst outcome is **buy and rebuild**: pay for a platform, then recreate it internally. Decide the boundary before signing.

### 4.8 Vendor landscape and business models

| Segment | Examples | Typical pricing | Value metric |
|---|---|---|---|
| ERP / planning suite | SAP IBP, Oracle, Microsoft Dynamics 365 | Subscription + implementation | Seats, modules |
| Demand planning | Blue Yonder, Kinaxis, o9, RELEX | Subscription + implementation | Revenue, modules |
| Visibility / control tower | project44, FourKites, Shippeo | Per shipment or per lane | Shipments |
| Optimization | Gurobi, FICO, ORTEC | License or compute | Seats, cores |
| Forecasting / ML platform | Databricks, Dataiku, custom | Consumption | Compute |
| Data / benchmarks | Circana, NielsenIQ, GfK | Subscription | Coverage |

Pricing that charges per seat can punish the adoption you need. Pricing that charges per shipment aligns with volume but not with value. Ask how the vendor's incentive maps to your outcome.

### 4.9 Adoption and change

Adoption is a design problem, not a communications problem.

| Adoption lever | Concrete action |
|---|---|
| Workflow fit | Put the output where the planner already works |
| Trust | Show accuracy on the planner's own items |
| Authority | Define override rules and accountability |
| Workload | Remove a task, do not add one |
| Incentive | Align KPIs with the new behavior |
| Training | Scenario-based, not feature-based |
| Visible sponsor | An executive who uses the output |
| Feedback | A channel for errors and a fix loop |

**Trust calibration:** a planner who ignores the model and a planner who follows it blindly are both miscalibrated. Design for appropriate reliance: show uncertainty, explain drivers, and make overrides visible and auditable.

### 4.10 Benefits realization

A benefit is realized only when it appears in a measured result.

| Stage | Evidence |
|---|---|
| Identified | A mechanism and an estimate |
| Baselined | A signed baseline and a counterfactual |
| Committed | In the budget or target |
| Realized | In the P&L, cash flow, or KPI |
| Sustained | Held for at least two cycles |

Assign a benefit owner. Review at 30, 90, and 180 days. If the benefit does not appear, investigate whether the model, the adoption, or the mechanism failed.

### 4.11 Portfolio prioritization

Score initiatives on:

| Criterion | Weight | Scale |
|---|---:|---|
| Decision value | 25% | 1–5 |
| Evidence strength | 20% | 1–5 |
| Feasibility | 15% | 1–5 |
| Strategic fit | 15% | 1–5 |
| Risk | 15% | 1–5 (inverse) |
| Dependency | 10% | 1–5 (inverse) |

A weighted score is a conversation aid, not a decision rule. Use it to expose disagreement about weights.

### 4.12 Executive communication

A one-page memo has five parts:

1. **Situation:** the decision and why now.
2. **Complication:** what is broken or at stake.
3. **Question:** the choice the executive must make.
4. **Answer:** the recommendation, with the value range.
5. **Evidence and risk:** the baseline, the counterfactual, the key assumption, and what would change the answer.

Keep the model in the appendix. The memo is the product.

## 5. Data, standards, and workflow

### 5.1 Data needed for a value case

| Data | Source | Grain | Check |
|---|---|---|---|
| COGS and inventory | ERP / finance | Month, SKU, location | Reconcile to GL |
| Service KPIs | OMS / WMS | Day, order | Definition consistency |
| Cost rates | Finance | Activity | Approved rate card |
| Volumes | TMS / WMS | Shipment, line | Completeness |
| Headcount and hours | HR / WMS | Week, role | Overtime included |
| Contract terms | Procurement | Vendor | Renewal dates |
| Baseline period | Finance | Month | Frozen and signed |

### 5.2 Workflow

1. Frame the decision and name the owner.
2. Build the baseline with finance.
3. Estimate the benefit range with a low, base, and high case.
4. Build the TCO with procurement and IT.
5. Calculate NPV, payback, and working capital.
6. Map stakeholders and design adoption.
7. Write the memo and pre-read.
8. Present, decide, and record the decision.
9. Instrument benefits realization.
10. Review and retire or scale.

### 5.3 Benefit-quality checks

- Is the baseline signed and dated?
- Is the counterfactual explicit?
- Is the benefit incremental, not diverted from another initiative?
- Is the cash release separated from the P&L saving?
- Is the ramp realistic?
- Is the run cost included?
- Is the benefit owner named?
- What would make the benefit zero?

## 6. Industry case studies

### 6.1 UPS ORION — route optimization

**Context:** UPS needed to reduce miles and fuel across a large delivery fleet.
**Intervention:** ORION (On-Road Integrated Optimization and Navigation) used heuristics to optimize routes dynamically.
**Evidence:** UPS publicly claimed savings of roughly 100 million miles and $300–400 million annually at scale. Treat these as company claims, not independent audit figures.
**Outcome:** The program became a reference case for optimization value.
**What transferred:** the discipline of measuring miles, stops, and fuel; the long deployment horizon; the role of driver adoption.
**What did not:** the exact savings depend on network density and route structure; another company cannot copy the number.

### 6.2 Maersk and IBM TradeLens — platform economics

**Context:** A blockchain-based trade platform intended to digitize container shipping documentation.
**Intervention:** A consortium platform with carriers, ports, and customs authorities.
**Evidence:** TradeLens was discontinued in 2022–2023 after failing to reach commercial viability and network adoption.
**Outcome:** A well-publicized failure of a technically credible platform.
**What transferred:** the lesson that a platform needs incentive alignment and a viable business model, not only technology.
**What did not:** the conclusion that blockchain is useless; the failure was commercial and organizational.

### 6.3 Retailer inventory reduction programs

**Context:** Grocers and general retailers have repeatedly targeted days of inventory through better forecasting and replenishment.
**Intervention:** Centralized replenishment, multi-echelon policies, and vendor collaboration.
**Evidence:** Company reports describe inventory reductions and service improvements, but the attribution is usually confounded with assortment, store, and pricing changes.
**Outcome:** Mixed; some programs sustained gains, others gave back the inventory when service fell.
**What transferred:** the need for a service floor and a benefit owner.
**What did not:** the assumption that inventory reduction is always a benefit.

### 6.4 A failed analytics pilot — composite

**Context:** A distributor built a demand forecast with strong offline accuracy.
**Intervention:** A vendor model replaced planner judgment for a subset of SKUs.
**Evidence:** Offline WAPE improved by 12%; after launch, planners overrode most recommendations and service did not improve.
**Outcome:** The pilot was declared a technical success and a business failure.
**What transferred:** adoption, workflow, and trust are part of the product.
**What did not:** the belief that accuracy alone drives value.

Use this composite only as a teaching case; it is not a citation.

## 7. Labs

### Lab 23.1 — Stakeholder and decision map

Choose a real or realistic initiative. Map user, buyer, beneficiary, and blocker. For each, write what they need to say yes.

**Deliverable:** One-page map plus a risk note on the blocker.

**Acceptance criteria:** Roles are distinct; evidence needs are specific; the blocker has a mitigation.

### Lab 23.2 — ROI and TCO model

Build a three-case (low, base, high) NPV model with build, integration, change, run, and exit costs. Include a ramp and a discount rate.

**Deliverable:** Spreadsheet or Python model plus a sensitivity table.

**Acceptance criteria:** Cash flow timing is explicit; hard and soft benefits are separated; payback and NPV reconcile; assumptions are listed.

### Lab 23.3 — Cost-to-serve

Using order, line, stop, and return data, allocate activity costs to 50 customers. Identify the five most and least profitable after cost-to-serve.

**Deliverable:** Cost-to-serve table and a pricing or service recommendation.

**Acceptance criteria:** Drivers are justified; allocation is documented; recommendations name the trade-off.

### Lab 23.4 — Executive memo

Write a one-page memo recommending or rejecting the initiative. Include the decision, value range, key risk, and what would change your mind.

**Deliverable:** Memo plus a five-minute presentation.

**Acceptance criteria:** SCQA structure; numbers reconcile to the model; the recommendation is explicit.

### Lab 23.5 — Vendor evaluation

Build a weighted scorecard for three real vendors in one segment. Include TCO, roadmap, data ownership, exit, and reference checks.

**Deliverable:** Scorecard and a recommendation memo.

**Acceptance criteria:** Criteria are weighted before scoring; TCO is comparable; exit is addressed.

## 8. Assignment / project

Build a full value case and adoption plan for a forecasting, inventory, or network initiative.

**Required components:**

1. Decision, owner, and cadence.
2. Baseline and counterfactual, signed by a finance reviewer.
3. Benefit model with low, base, and high cases.
4. TCO with build, integration, change, run, and exit.
5. NPV, payback, and working-capital impact.
6. Stakeholder map and adoption plan.
7. Benefits-realization plan with owners and review dates.
8. One-page executive memo.
9. A list of what would make the case fail.

**Length:** 8–12 pages plus model appendix.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Decision and ownership | 15% | Vague problem | Named decision and owner | Decision, counterfactual, and authority |
| Baseline quality | 15% | Uses last year's average | Frozen baseline and counterfactual | Baseline signed and tested for shocks |
| Value model | 20% | Single number | Three cases and ramp | Separates cash, P&L, risk, and service |
| Cost model | 15% | License cost only | Build and run costs | Full TCO including change and exit |
| Adoption plan | 15% | Training plan | Workflow and override policy | Incentives, trust, and feedback loop |
| Communication | 10% | Descriptive | Clear memo | Decision-ready and honest about risk |
| Benefits realization | 10% | No plan | Owner and review dates | Instrumented and tied to P&L |

## 10. Regulatory, ethical, and commercial lens

- **Payment terms:** extending DPO can breach late-payment rules and harm small suppliers.
- **Pricing and promotions:** value cases must not assume anticompetitive coordination.
- **Privacy:** customer-level cost-to-serve uses personal data; minimize and aggregate.
- **Labor:** productivity benefits may mean headcount reduction; plan the transition honestly.
- **Incentives:** a planner measured on forecast accuracy will behave differently from one measured on service.
- **Vendor lock-in:** data egress, model portability, and exit costs are commercial and ethical issues.
- **Greenwashing:** carbon benefits need a measured baseline and a recognized method.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Benefit double count | Cash and P&L added | Same inventory counted twice | Inflated case | Separate one-time and recurring |
| Baseline drift | Baseline changed after pilot | Moving goalposts | No accountability | Freeze and sign |
| Pilot purgatory | Success but no rollout | No owner or budget | Wasted spend | Fund scale-up upfront |
| Seat-based trap | License grows with users | Adoption penalized | Cost overrun | Negotiate usage pricing |
| Tool sprawl | Three overlapping platforms | Procurement by team | Integration cost | Portfolio review |
| Adoption theater | Training attendance as metric | No behavior change | No benefit | Measure usage and overrides |
| Soft-benefit inflation | Risk avoidance dominates | Hard to falsify | Finance distrust | Cap soft benefits |
| Vendor lock-in | Proprietary data model | Exit cost | Weak negotiating position | Contract for portability |

## 12. Interview and discussion questions

1. How do you build a baseline when the process is changing? *(Hint: simulate, pilot, or use a control group; freeze before launch.)*
2. What is the difference between cash release and P&L saving? *(Hint: inventory reduction is one-time cash; carrying cost is recurring.)*
3. How would you value a service improvement? *(Hint: expected lost margin, churn, penalty, and willingness to pay.)*
4. When should you build rather than buy? *(Hint: differentiation, data, speed, talent, control.)*
5. How do you handle a blocker who is also the user? *(Hint: redesign the workflow, give authority, escalate with evidence.)*
6. What makes an analytics benefit defensible to finance? *(Hint: incremental, measured, owned, reconciled.)*
7. How do you prioritize a portfolio with a fixed budget? *(Hint: value, evidence, feasibility, risk, dependency.)*
8. Why do pilots fail to scale? *(Hint: no owner, no budget, no workflow change, no incentive.)*
9. How do you measure adoption? *(Hint: usage, override rate, time saved, decision change.)*
10. What would make you recommend against your own proposal? *(Hint: name the falsifier.)*
11. How do you avoid double-counting benefits across initiatives? *(Hint: one benefit ledger, one owner per line.)*
12. What is the right discount rate? *(Hint: the company's, from finance.)*

## 13. Key terms

| Term | Definition |
|---|---|
| Baseline | The measured state without the initiative |
| Counterfactual | The explicit alternative used for comparison |
| Attribution | The share of an outcome caused by the initiative |
| NPV | Present value of net benefits minus initial cost |
| ROI | Net benefit divided by total cost |
| Payback | Time to recover the investment |
| TCO | Total cost of ownership over the life of the initiative |
| DIO | Days inventory outstanding |
| DSO | Days sales outstanding |
| DPO | Days payables outstanding |
| Cash-to-cash | DIO + DSO − DPO |
| Cost-to-serve | Activity-based cost of serving a customer or channel |
| Hard benefit | A cash or cost effect visible in financial statements |
| Soft benefit | A service, risk, or experience effect |
| Ramp | The adoption curve used in the value model |
| Benefits realization | The process of proving and sustaining a benefit |
| Build vs buy | The decision to develop or procure a capability |
| Vendor lock-in | Switching costs created by a vendor relationship |
| Trust calibration | Matching reliance on a model to its reliability |
| Automation bias | Over-reliance on a model's output |
| Portfolio prioritization | Ranking initiatives under a fixed budget |
| Executive memo | A one-page decision document |
| Value metric | The unit a vendor charges for |
| Exit cost | The cost of leaving a vendor or retiring a system |

## 14. Sources and further reading

### Standards and frameworks

- ASCM, SCOR Digital Standard. https://www.ascm.org/
- ISO 31000, Risk management. https://www.iso.org/standard/65694.html
- AICPA, SOC 2. https://www.aicpa-cima.com/

### Peer-reviewed and professional

- Kaplan, R. S., Norton, D. P., *The Balanced Scorecard*.
- Kaplan, R. S., Anderson, S. R., *Time-Driven Activity-Based Costing*, *Harvard Business Review*, 2004.
- Christensen, C. M., *The Innovator's Dilemma* (for build/buy and disruption).
- Kotter, J. P., *Leading Change*.
- Davenport, T. H., Harris, J. G., *Competing on Analytics*.
- Kahneman, D., *Thinking, Fast and Slow* (for trust and bias).

### Industry

- UPS, ORION program materials. https://www.ups.com/
- Maersk and IBM, TradeLens discontinuation announcement, 2022. https://www.maersk.com/
- Gartner, Supply Chain Top 25. https://www.gartner.com/en/supply-chain
- McKinsey, supply chain analytics and value-capture reports. https://www.mckinsey.com/
- Deloitte, supply chain planning surveys. https://www.deloitte.com/

### Tools

- A spreadsheet or Python (pandas, numpy-financial).
- ERP and finance extracts from your organization.
- Vendor RFP templates from procurement.

> **Next:** [Module 24](24_quality_evaluation_model_risk.md) turns the value case into a governed, validated model.
