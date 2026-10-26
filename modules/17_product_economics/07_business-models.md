# Business Models

> Module 17 — Product, Economics, and Leadership
> Topic 07 of 09
> Estimated study time: 4 hours
> Prerequisites: 02 (Value Quantification), 04 (Reimbursement)
> Version 2026.3

## 1. Why this topic exists

Two companies can sell the same product to the same buyer at the same price and have completely
different businesses. The difference is the business model: what triggers payment, who bears the
risk of the product not working, and when the cash arrives. Those three choices determine gross
margin, working capital, sales motion, and how the company survives a bad quarter.

Most teams inherit a business model rather than choosing one. A per-seat subscription is the
default because it is familiar, not because it fits. A per-test model appears when the product
maps to a procedure. A shared-savings model appears when the buyer says "we cannot afford it, but
we would pay from the savings." Each is a decision with consequences that are visible only in the
model.

### 1.1 The same product, four models

Consider a chronic kidney disease prediction and care-coordination product sold to a payer:

- Per seat, to the care-management team.
- Per member per month, across the covered population.
- Per risk assessment, when the model scores a patient.
- Shared savings, as a percentage of avoided dialysis cost.

The nominal revenue ranges from about 205,000 to 1.8 million United States dollars (USD) over
three years in the worked example. The risk-adjusted present value ranges from 414,000 to 792,000.
The ranking of the models by headline revenue is not the ranking by value.

### 1.2 Why the choice is strategic

The business model determines what the company must be good at.

| Model | What the company must be good at | What kills it |
|---|---|---|
| Per seat | Adoption and renewal | Seat contraction, low use |
| Per member per month | Population health sales and data access | Payer churn, population definition |
| Per test or per case | Volume and throughput | Volume shortfall, price pressure |
| Shared savings | Measurement, attribution, and patience | Validation disputes, cash lag |
| Device plus service | Logistics and field service | Capital cycles, service cost |
| Risk-sharing | Actuarial modelling and capital | Adverse selection, tail risk |

A company that is good at per-seat sales and switches to shared savings has changed its core
competence requirement, not just its price list.

### 1.3 What this topic covers

It covers the design space of business models for biomedical data products, the arithmetic that
distinguishes them, and the constraints that limit them. It does not cover fundraising,
valuation, or corporate finance. It does not cover pricing psychology in detail; that is a
subtopic of commercial strategy.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Identify the unit of value and the risk bearer in a given business model | Exercise 1; self-check 1 |
| 2 | Compare models on gross margin, cash timing, and revenue volatility | Exercise 2; self-check 3 |
| 3 | Compute risk-adjusted present value for a contingent revenue stream | Exercise 3; self-check 4 |
| 4 | Design a contract mechanism with a defined measurement and true-up | Exercise 4; interview question 5 |
| 5 | Assess attribution risk and propose a mitigation | Exercise 5; self-check 6 |
| 6 | Choose a model that fits the buyer's budget and approval path | Exercise 6; interview question 7 |
| 7 | Test a model against fraud and abuse and revenue recognition constraints | Exercise 7; self-check 8 |
| 8 | Build a hybrid model with a floor and an upside | Exercise 8; interview question 10 |

## 3. The core mental model

A business model is a triple: the unit that triggers payment, the party that bears the risk of
non-performance, and the timing of cash. Every model in the market is a point in that space.

### 3.1 The three axes

```text
  UNIT OF VALUE
    per seat  ---- per site ---- per member ---- per event ---- per outcome
       |                                                        |
       |     more predictable            more aligned           |
       |     less aligned                less predictable       |
       v                                                        v
  RISK BEARER
    customer bears  ---------------------------->  vendor bears
    (subscription)                                  (shared savings,
                                                     warranty, at-risk)

  CASH TIMING
    prepaid ---- monthly ---- quarterly ---- milestone ---- post-validation
```

Read the diagram as three independent choices. A per-seat model with monthly cash and customer-
borne risk is the safest for the vendor. A per-outcome model with post-validation cash and
vendor-borne risk is the most aligned and the most dangerous.

Alignment and predictability trade off. A model that perfectly aligns incentives pays the vendor
only when the outcome occurs, which is exactly when the vendor has the least control.

### 3.2 The revenue quality ladder

Not all revenue is equal. Four properties determine quality:

1. **Recurrence.** Does it repeat without a new sale?
2. **Contractedness.** Is it committed for a term, or at will?
3. **Predictability.** Can it be forecast within a narrow band?
4. **Margin.** How much of it survives cost to serve?

A per-seat subscription scores high on all four. A shared-savings contract scores low on
predictability and timing and high on margin. Investors and finance teams price these differently,
and so should the operating plan.

### 3.3 Pricing anchors

Every price is anchored to one of three things.

| Anchor | Basis | When it works | Risk |
|---|---|---|---|
| Cost-plus | Cost to serve plus a margin | Commodity, procurement-driven | Leaves value on the table |
| Competitive | The alternative's price | Crowded market, substitutable | Race to the bottom |
| Value-based | A share of the value created | Differentiated, measurable outcome | Attribution disputes |

Value-based pricing is not the same as an outcomes-based contract. A product can be priced at a
value-based level and billed monthly on a per-seat basis. The contract mechanism and the price
level are separate decisions, and conflating them is a common error.

### 3.4 The hybrid pattern

Most durable models are hybrids: a floor that covers cost to serve, plus a variable component
that shares upside. The floor protects the vendor's solvency; the variable component aligns the
buyer's incentive.

```text
  revenue = base_fee (covers cost to serve and some margin)
          + variable_fee (share of measured value, capped)
```

A hybrid is usually easier to sell than a pure outcome model, because the buyer can budget the
base and the vendor can fund operations. It is also easier to account for, because the base is
fixed consideration and only the variable component is constrained.

### 3.5 Scoring revenue quality

A practical way to compare models is to score each on the four properties and weight them by what
the business needs. The score is not a valuation; it is a structured argument.

| Property | Question | Weight for a venture-funded vendor | Weight for a bootstrapped vendor |
|---|---|---:|---:|
| Recurrence | Does the revenue repeat without a new sale? | 0.30 | 0.35 |
| Contractedness | Is it committed for a term? | 0.25 | 0.20 |
| Predictability | Can it be forecast within a narrow band? | 0.30 | 0.30 |
| Margin | How much survives cost to serve? | 0.15 | 0.15 |

A shared-savings contract scores well on margin and poorly on predictability and recurrence. That
profile suits a company with capital and a long horizon, and it is a poor fit for a company that
needs to forecast next quarter. The weights encode the company's constraint, and two companies
scoring the same contract differently can both be right.

### 3.6 Redraw test
Draw the three axes and place your current model and your proposed model on them. If the two
points differ on more than one axis, the change is a business model change, not a pricing change,
and it requires a different operating capability.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit |
|---|---|---|---|
| Business model | The unit of payment, risk allocation, and cash timing | Used to mean pricing | Design |
| Pricing | The level and structure of charges | Confused with the model | Currency |
| Unit of value | The thing that triggers payment | Confused with the unit of use | Definition |
| Per seat | Price per named user | Confused with per active user | Currency per user |
| Per site | Price per facility | Confused with per bed | Currency per site |
| PMPM | Per member per month | Confused with per patient | Currency per member-month |
| Per test | Price per test or assessment | Confused with per report | Currency per test |
| Per case | Price per episode or admission | Confused with per diem | Currency per case |
| Shared savings | Vendor receives a share of measured savings | Confused with a discount | Percentage |
| Risk-sharing | Vendor bears part of a financial loss | Confused with warranty | Contract term |
| Outcomes-based | Payment contingent on a measured outcome | Confused with pay-for-reporting | Contract term |
| Warranty | Vendor refunds if a product fails to perform | Confused with an SLA | Contract term |
| At-risk | Vendor's payment depends on performance | Used loosely | Contract term |
| Capitation | Fixed payment per member for a defined scope | Confused with PMPM software pricing | Currency per member |
| Gross margin | Revenue minus cost of goods sold | Confused with operating margin | Percentage |
| Cost to serve | Direct cost of delivering the product to a customer | Underestimated | Currency per customer |
| COGS | Cost of goods sold | Includes support inconsistently | Accounting |
| Contribution margin | Revenue minus variable cost | Confused with gross margin | Currency |
| CAC | Customer acquisition cost | Confused with cost of sale | Currency |
| LTV | Lifetime value | Computed without churn | Currency |
| Payback | Time to recover acquisition cost | Computed without gross margin | Months |
| ARR | Annual recurring revenue | Includes non-recurring revenue | Currency per year |
| NRR | Net revenue retention | Confused with gross retention | Percentage |
| Churn | Loss of customers or revenue | Confused with non-renewal | Percentage |
| Expansion | Additional revenue from existing customers | Confused with price increase | Currency |
| Working capital | Cash tied up between cost and collection | Ignored in model design | Currency |
| Revenue recognition | Accounting rules for when revenue is recorded | Ignored in contract design | Standard |
| Variable consideration | Consideration contingent on future events | Recognised too early | ASC 606 |
| Constraint | Limit on recognising variable consideration | Confused with a cap | ASC 606 |
| True-up | Reconciliation of actual versus estimated payment | Confused with a rebate | Mechanism |
| Rebate | Return of part of the price on a condition | Confused with a discount | Mechanism |
| Attribution | Assigning an outcome to an intervention | Assumed to be 100 percent | Percentage |
| Guardrails | Criteria that define when payment applies | Omitted from contracts | Contract term |
| Benchmark | Comparison used to define savings | Chosen to flatter | Definition |
| Risk corridor | Shared band of upside and downside | Confused with a cap | Contract term |
| Anti-kickback | Prohibition on remuneration for referrals | Ignored in outcomes contracts | Statute |
| Stark law | Physician self-referral prohibition | Confused with anti-kickback | Statute |
| Value-based safe harbour | Protection for qualifying value-based arrangements | Assumed to cover any outcome deal | Regulation |

## 5. The mechanics, step by step

The procedure chooses and designs a business model for a product with a defined value proposition.
It assumes Topic 02 has produced a value estimate and Topic 04 has established the payment route.

### Step 1. Identify the unit of value, not the unit of use

**Input:** the value model from Topic 02.
**Output:** the smallest unit at which value is created.
**Rationale:** the billing unit should track the value unit. A product that creates value per
patient-year should not be priced per API call, because the price will not scale with the benefit
and the buyer will notice.

Test: if the unit doubles, does the value roughly double? If not, the unit is wrong.

| Value created per | Candidate unit | Mismatch to avoid |
|---|---|---|
| Member-month of managed risk | PMPM | Per-seat pricing that ignores population size |
| Encounter with a decision | Per encounter | Per-seat pricing that ignores volume |
| Test interpreted | Per test | Subscription that caps tests |
| Episode of care | Per case | Per-diem pricing that rewards length |
| Outcome achieved | Per outcome | Per-seat pricing that ignores results |

### Step 2. Decide who bears the risk of non-performance

**Input:** the value model and the buyer's risk tolerance.
**Output:** an explicit allocation of risk between buyer and vendor.
**Rationale:** someone bears the risk that the product does not deliver. In a subscription, the
buyer bears it, because they pay whether or not value materialises. In a shared-savings contract,
the vendor bears part of it. The allocation should be a decision, not a by-product of the pricing
format.

Risks to allocate explicitly:

| Risk | Description | Natural bearer |
|---|---|---|
| Adoption risk | Users do not use the product | Shared; vendor can influence |
| Effect risk | The product does not work as predicted | Vendor, if it made the claim |
| Attribution risk | Savings cannot be attributed | Shared; depends on measurement |
| Population risk | The eligible population is smaller than expected | Buyer |
| Regulatory risk | The payment route changes | Shared |
| Technology risk | The platform fails or is breached | Vendor |

A contract that places all six on one party will be priced for that, or will not be signed.

### Step 3. Map cash timing and working capital

**Input:** the risk allocation.
**Output:** a month-by-month cash schedule.
**Rationale:** a model with 18-month payment lag requires the vendor to fund delivery for 18
months. That is a financing requirement, and it should be visible in the plan.

```text
working_capital_need = months_of_lag x monthly_cost_to_serve
                     + implementation_cost_before_first_payment
```

A vendor with a 6 million USD annual cost base and an 18-month lag needs roughly 9 million USD of
working capital before the first payment, which is more than most early-stage companies can raise
for a single contract. This is why shared savings is usually a model for established companies or
for contracts with a base fee.

### Step 4. Set the price level with an explicit anchor

**Input:** the value estimate, the competitive alternatives, and the cost to serve.
**Output:** a price with a stated anchor and a rationale.
**Rationale:** a price without an anchor is arbitrary and will be negotiated to the buyer's
anchor, which is usually the cheapest alternative.

Three anchors, computed:

```text
cost_floor     = cost_to_serve / (1 - target_gross_margin)
competitive    = the price of the alternative the buyer would otherwise use
value_ceiling  = a share of the value created, typically 10 to 30 percent
```

The price should sit above the cost floor and below the value ceiling, and the negotiation will
determine where. A price below the cost floor is a subsidy; a price above the value ceiling will
not survive a business case review.

### Step 5. Model gross margin and cost to serve

**Input:** the delivery model.
**Output:** gross margin by model, with cost-to-serve drivers identified.
**Rationale:** the same product can have very different margins depending on the model, because
the cost to serve differs. A per-test model with cloud inference per test has a variable cost that
scales; a PMPM model has a mostly fixed cost that does not.

| Cost driver | Per seat | PMPM | Per test | Shared savings |
|---|---|---|---|---|
| Hosting | Fixed | Fixed | Variable | Fixed |
| Support | Per user | Per member | Per customer | Per customer |
| Implementation | Per site | Per payer | Per site | Per payer |
| Clinical review | Per user | Per population | Per positive | Per population |
| Measurement and reporting | Low | Medium | Low | High |

The classification of a cost as fixed or variable determines which model survives a volume change.
A per-test model with cloud inference has a cost that falls when volume falls, so a demand shock
compresses revenue and cost together. A PMPM model has a fixed support cost, so the same shock
compresses revenue while cost stays, and margin collapses. That asymmetry is the reason a vendor
with high fixed costs prefers volume-linked pricing and a vendor with low fixed costs can afford a
subscription.

The cost-to-serve calculation should be built bottom-up for one account and then tested at three
volumes. A model that shows a 79 percent margin at the planned volume and a 40 percent margin at
half the volume is a model with an operating leverage problem, and the plan should say so.

### Step 6. Stress-test revenue volatility

**Input:** the revenue model.
**Output:** revenue under low, base, and high scenarios.
**Rationale:** a model whose revenue can fall 60 percent in a bad year cannot support the same cost
base as a model with 10 percent variance.

For each model, vary the driver by plus or minus one standard deviation and record the revenue
range. For a per-seat model, the driver is seats; for PMPM, members; for per-test, volume; for
shared savings, both the measured savings and the validation probability.

### Step 7. Design the contract mechanism

**Input:** the risk allocation and the price level.
**Output:** the payment mechanics, in operational terms.
**Rationale:** a business model is only real when the contract specifies what is measured, how,
by whom, over what period, and what happens when the measurement is disputed.

The mechanism specification:

| Element | Question | Example |
|---|---|---|
| Trigger | What event creates a payment obligation? | A validated risk assessment |
| Measure | What quantity is counted? | Members screened within 30 days |
| Source | Which system produces the number? | Payer claims plus vendor log |
| Reconciliation | How are discrepancies resolved? | Joint review within 45 days |
| True-up | When is the estimate corrected? | Annually, against claims runout |
| Cap | Is there a maximum payment? | 1.5 times the base fee |
| Floor | Is there a minimum payment? | Base fee covers cost to serve |
| Term | Over what period is performance measured? | 12 months, paid at 18 |
| Exit | What happens on termination? | Pro-rata settlement |

### Step 8. Address attribution before signing

**Input:** the outcome being measured.
**Output:** an attribution method and a dispute process.
**Rationale:** attribution is the most common source of post-signing conflict. A payer that
cannot separate the product's effect from concurrent initiatives will discount the payment, and
the vendor will litigate or accept.

Attribution methods, in order of preference:

1. **Randomised or stepped comparison** in the payer's population.
2. **Matched comparison group** with pre-registered matching criteria.
3. **Difference-in-differences** against a trend.
4. **Pre-post with a negotiated discount** for secular trend.
5. **Expert adjudication** by a joint committee.

Methods 1 and 2 are credible and expensive. Method 4 is cheap and disputed. The contract should
specify the method, the discount for trend, and the dispute process before any patient is enrolled.

### Step 9. Build the expansion path

**Input:** the initial contract.
**Output:** a defined path from the first unit to a larger one.
**Rationale:** a model that cannot expand has a capped lifetime value. Expansion can come from
population, sites, modules, or price.

| Expansion lever | Mechanism | Risk |
|---|---|---|
| Population | Add covered lives or conditions | Payer budget cycle |
| Sites | Add facilities | Sales cost per site |
| Modules | Add products to the platform | Product complexity |
| Price | Contractual escalator | Buyer resistance |
| Geography | Add regions or countries | Regulatory variation |

A per-seat model expands with headcount, which is slow. A PMPM model expands with population,
which is faster if the payer grows. A per-test model expands with volume, which is the fastest and
the most exposed to utilisation management.

Expansion should be quantified, not assumed. Take the initial contract and compute the revenue at
each expansion step:

| Step | Lever | Revenue effect | Time to realise |
|---|---|---|---:|
| Initial | One condition, one region | Base | 0 |
| Add condition | Module expansion | Plus 35 to 60 percent of base | 6 to 12 months |
| Add population | New value-based group | Proportional to members added | 3 to 9 months |
| Add sites | New facilities | Proportional to sites added | 6 to 18 months |
| Add geography | New region or country | Proportional, with new regulatory cost | 12 to 24 months |

A model whose only expansion lever is price has a capped lifetime value, because price increases
are negotiated once and then defended at every renewal. A model with a population lever can grow
without a new sale, which is the property that makes PMPM attractive to a vendor despite its thin
margin.

### Step 10. Check the fit with the sales motion

**Input:** the target segment and the deal size.
**Output:** a cost-of-sale estimate and a payback.
**Rationale:** a 30,000 USD annual contract cannot support a 200,000 USD enterprise sales cycle.
The model must fit the motion, or the motion must change.

```text
viable_ACV >= CAC / target_payback_years / gross_margin
```

At a 20-month payback target, a 78 percent gross margin, and a 386,000 USD cost per win, the
minimum viable annual contract value is about 297,000 USD. A product priced at 30,000 USD per year
must be sold through a channel, self-service, or a much cheaper motion.

### Step 11. Test the regulatory and accounting constraints

**Input:** the proposed mechanism.
**Output:** a list of constraints and required approvals.
**Rationale:** outcomes-based arrangements between vendors and providers can implicate the federal
Anti-Kickback Statute and the Physician Self-Referral Law, and contingent revenue has accounting
consequences.

In the United States:

- The Anti-Kickback Statute at 42 U.S.C. 1320a-7b(b) prohibits remuneration intended to induce
  referrals of items payable by federal health care programmes. Discounts and outcomes-based
  payments can be structured to fit a safe harbour, but the structure matters.
- The Physician Self-Referral Law (Stark) at 42 U.S.C. 1395nn prohibits certain referrals where a
  physician has a financial relationship with the entity, with exceptions including value-based
  arrangements.
- ASC 606 requires variable consideration to be estimated and constrained to the amount probable
  of not reversing. A shared-savings stream that might be clawed back cannot be recognised as
  revenue on the same schedule as a subscription.

A vendor should obtain counsel before signing an outcomes-based arrangement with a provider that
receives federal programme payments. The regulatory analysis is not optional and is not covered by
this text.

Outside the United States the constraints differ but do not disappear. In England, a product
adopted through the NHS MedTech Funding Mandate or a NICE recommendation is paid from a
nationally defined pathway, and the vendor's model is usually a price to the NHS rather than a
shared-savings arrangement. In the European Union, the Medical Device Regulation imposes
post-market surveillance obligations that continue for the life of the device, and a subscription
model does not transfer that obligation to the buyer. A vendor that prices a European contract on
a United States cost base will underprice its regulatory overhead.

### Step 12. Choose, document, and set the review trigger

**Input:** all of the above.
**Output:** a one-page business model decision record.
**Rationale:** the model will change, and the record of why it was chosen is what allows a
rational change later rather than a drift.

The decision record states: the unit of value, the risk allocation, the cash timing, the price
anchor, the gross margin, the volatility range, the regulatory constraints, and the conditions
under which the model would be revisited.

### 5.13 Branch points and edge cases

**No reimbursement pathway.** Without a code or coverage, PMPM and shared savings are the only
viable models for a payer-facing product, because there is no per-service payment to attach to.
For a provider-facing product, the model must fit an operating budget, which usually means a
subscription priced to cost avoidance.

**Single-payer versus multi-payer.** In a single-payer system, PMPM and capitation are natural
because the payer bears the full cost of the population. In a multi-payer system, attribution
across payers is difficult, and a provider may bear costs while another payer captures savings,
which makes shared savings hard to contract.

**Capital versus operating budget.** A device-plus-service model requires the buyer to fund
capital, which lengthens the cycle and requires a return calculation. A subscription converts the
same purchase into operating spend, which is easier to approve but more easily cancelled.

**Pilot-to-contract transition.** A pilot is often free or heavily discounted, which sets a price
anchor at zero. The contract should state the production price at the start of the pilot, or the
negotiation will begin from the pilot price.

**Procurement security review.** A per-test model with cloud inference creates a data flow that
must be reviewed. A model that requires sending patient data to the vendor's cloud may fail a
security review that an on-premises deployment would pass, which changes the cost structure and
therefore the model.

**Champion turnover.** A shared-savings contract depends on an internal sponsor who defends the
measurement methodology. If the sponsor leaves, the contract is renegotiated, usually downward.

**Price transparency and most-favoured-nation clauses.** A payer may require a most-favoured-
customer term, which constrains pricing across accounts. A per-seat price that varies by segment
can trigger it.

### 5.14 Decision table for model selection

| Buyer type | Budget type | Value measurability | Attribution feasibility | Recommended model |
|---|---|---|---|---|
| Provider, department | Operating | Low | Not applicable | Per seat or per site subscription |
| Provider, enterprise | Operating | Medium | Not applicable | Enterprise subscription with volume tiers |
| Provider, value-based contract | Operating | High | Feasible within the system | Shared savings with a base fee |
| Payer, commercial | Medical loss ratio | Medium | Feasible with claims | PMPM |
| Payer, Medicare Advantage | Medical loss ratio | High | Feasible with claims | PMPM plus shared savings |
| Employer or plan sponsor | Operating | Medium | Feasible with claims | PMPM or per eligible employee |
| Diagnostic lab or imaging | Operating | High | Not applicable | Per test |
| Device manufacturer | Capital plus service | Medium | Not applicable | Device sale plus service contract |
| Any buyer, unmeasurable value | Any | Low | Not feasible | Do not use an outcome model |

The last row is the most important. An outcomes-based contract requires a measurable outcome and a
feasible attribution method. Without both, the contract will produce a dispute rather than an
incentive.

## 6. Worked example

> **Note:** the payer, the product, and all figures are a composite constructed for teaching. They
> are illustrative 2026 USD and not reported results. The clinical and cost parameters are
> plausible but not drawn from a specific contract.

### 6.1 The product and the four models

A vendor has a chronic kidney disease (CKD) progression model with a care-coordination workflow.
It identifies patients likely to progress to dialysis within 24 months and prompts a care manager
to intervene.

A regional payer with 480,000 covered lives, of whom 24,000 have diagnosed CKD stage 3 or 4, is
the prospect. The vendor is considering four models:

| Model | Unit | Price | Annual volume |
|---|---|---:|---:|
| A | Per seat, care management team | 95 USD per seat per month | 180 seats |
| B | Per member per month | 0.85 USD PMPM | 24,000 members |
| C | Per risk assessment | 38 USD per assessment | 6,500 assessments |
| D | Shared savings | 15 percent of avoided dialysis cost | 9,600 covered members |

Model D covers only 40 percent of the CKD population, because the payer will apply it to the
members in its value-based arrangements, where savings are retained.

### 6.2 Revenue and margin

| Model | Annual revenue (USD) | Cost to serve (USD) | Gross profit (USD) | Margin |
|---|---:|---:|---:|---:|
| A per seat | 205,200 | 38,880 | 166,320 | 81.1% |
| B PMPM | 244,800 | 49,200 | 195,600 | 79.9% |
| C per assessment | 247,000 | 27,300 | 219,700 | 88.9% |
| D shared savings, expected | 325,987 | 52,000 | 273,987 | 84.0% |

The per-assessment model has the highest margin because its cost is almost entirely variable and
the price is well above the marginal inference and support cost. The PMPM model has the lowest
margin because the cost to serve is fixed and the price is thin.

### 6.3 The shared-savings build-up

| Step | Value |
|---|---:|
| Covered members | 9,600 |
| Annual dialysis starts at 3.5 percent | 336 |
| Avoided starts at a 12 percent relative reduction | 40.3 |
| Annual cost per dialysis patient | 98,000 USD |
| Gross savings | 3,951,360 USD |
| Vendor share at 15 percent | 592,704 USD |
| Expected value at 55 percent validation probability | 325,987 USD |
| Net after cost to serve | 273,987 USD |

Two assumptions carry the result. The 55 percent validation probability reflects the risk that
the payer's measurement does not confirm the effect, because of attribution, claims runout, or a
methodology dispute. The 12 percent relative reduction is the modelled effect; if the payer's own
evaluation finds 7 percent, the savings fall proportionally.

### 6.4 Risk-adjusted present value

The shared-savings payment lags performance by 18 months, so performance years one to three are
paid at the end of years two, three, and four.

| Model | Annual gross profit (USD) | Timing | Five-year PV at 12% (USD) |
|---|---:|---|---:|
| A per seat | 166,320 | Monthly, years 1 to 5 | 599,546 |
| B PMPM | 195,600 | Monthly, years 1 to 5 | 705,094 |
| C per assessment | 219,700 | Monthly, years 1 to 5 | 791,969 |
| D shared savings | 273,987 | Years 2, 3, 4 | 587,563 |

The shared-savings model has the highest annual gross profit and the lowest present value of the
four. Its nominal three-year gross share is 1,778,112 USD, which is three times the present value
of 587,563 USD. That gap is the cost of contingency and delay, and it is the number a finance
team will apply when deciding whether to fund the contract.

### 6.5 Break-even volumes

Against model A's annual revenue of 205,200 USD:

| Model | Volume needed to match A |
|---|---:|
| Per seat | 180 seats |
| PMPM at 0.85 USD | 20,118 members |
| Per assessment at 38 USD | 5,400 assessments |
| Shared savings, expected | 6,043 covered members |

The shared-savings volume is the most interesting. It says the model only matches the per-seat
revenue if the payer applies it to at least 6,043 members, which is 63 percent of the 9,600 the
vendor expects. If the payer restricts the arrangement to a smaller value-based population, the
model underperforms the simple subscription.

### 6.6 Two diagnostics that reveal a weak business model

**Diagnostic 1: the attribution test.** Recompute the shared-savings value at different
attributable shares of the savings.

| Attributable share | Gross share (USD) | Net (USD) | PV (USD) |
|---:|---:|---:|---:|
| 30% | 177,811 | 45,796 | 98,210 |
| 50% | 296,352 | 110,994 | 238,025 |
| 70% | 414,893 | 176,191 | 377,840 |
| 85% | 503,798 | 225,089 | 482,702 |
| 100% | 592,704 | 273,987 | 587,563 |

At 50 percent attribution, the model is worth 238,025 USD over five years, less than half the
per-seat subscription. A payer that attributes half the savings to concurrent programmes, trend,
or regression to the mean is not being unreasonable; that is a normal measurement outcome. A
vendor that has not modelled this scenario has modelled the best case and called it the base case.

**Diagnostic 2: the contract-length test.** The 18-month lag means a three-year contract yields
only two payment cycles.

| Contract length | Payment cycles | PV at 12% (USD) |
|---:|---:|---:|
| 3 years | 2 | 413,440 |
| 4 years | 3 | 587,563 |
| 5 years | 4 | 743,031 |

A three-year contract, which is the most a payer will typically sign with a new vendor, delivers
413,440 USD of present value, 31 percent less than the per-seat subscription. The model requires a
four-year term to match, and a five-year term to exceed.

### 6.7 Varying one assumption

Hold everything else constant and vary the probability that the payer validates the savings.

| Validation probability | Expected share (USD) | Net (USD) | PV (USD) |
|---:|---:|---:|---:|
| 0.30 | 177,811 | 125,811 | 269,801 |
| 0.45 | 266,717 | 214,717 | 460,459 |
| 0.55 | 325,987 | 273,987 | 587,563 |
| 0.70 | 414,893 | 362,893 | 778,221 |
| 0.85 | 503,798 | 451,798 | 968,878 |
| 1.00 | 592,704 | 540,704 | 1,159,536 |

The conclusion changes at about 0.70. Above that, shared savings beats per seat on present value;
below it, the simple subscription wins. A 70 percent validation probability is plausible only when
the contract specifies a randomised or matched comparison and the payer has agreed the
methodology in advance. That is a contracting decision, not a modelling one.

### 6.8 The recommendation and the hybrid

The analysis supports a hybrid rather than a pure model:

- A base fee of 9,000 USD per month, which covers cost to serve of about 4,333 USD per month and a
  modest margin, and is budgetable by the payer.
- A shared-savings component at 12 percent of validated avoided dialysis cost, capped at 1.5 times
  the base fee.
- A four-year term, which provides three payment cycles.
- A pre-agreed attribution method: a matched comparison group with pre-registered matching
  criteria and a 25 percent discount for secular trend.

At the base case, the hybrid produces a base of 108,000 USD per year plus an expected variable
component. The variable component, discounted for the 25 percent trend adjustment, is 444,528 USD
of gross share at the base effect, times 0.55 validation, or 244,490 USD. Total expected gross
profit is roughly 108,000 minus 52,000 plus 244,490, or about 300,490 USD per year, with a floor
that keeps the contract solvent if validation fails.

That structure is less exciting than the pure shared-savings deal and more fundable. It is also
the structure most likely to survive a finance review, because the downside is bounded and the
upside is real.

### 6.9 What the comparison does not capture

Three things are missing from the table, and each would change the ranking.

The first is the cost of measurement. A shared-savings contract requires claims analysis,
actuarial support, and a dispute process, which the model treats as part of the 52,000 USD cost to
serve. In practice these costs can reach 5 to 10 percent of contract value, which would reduce the
shared-savings net by a further 30,000 to 60,000 USD per year.

The second is the value of the reference. A shared-savings contract with a regional payer is often
the reference that unlocks other payers, and that option value is real but not in the present
value. A vendor may rationally accept a lower NPV on the first contract for that reason.

The third is the risk of a dispute. A contested validation consumes management attention and legal
cost, neither of which appears in the model. The probability of a dispute is not the same as the
probability of non-validation; a contract can validate at 60 percent and still spend a year in
argument about the 40 percent.

## 7. Code, query, or calculation

The script compares the four models on revenue, margin, risk-adjusted present value, break-even
volume, attribution, and contract length.

```python
"""Comparing business models for one CKD product."""

RATE = 0.12
YEARS = 5

SEATS, SEAT_PRICE, SEAT_COGS = 180, 95.0, 18.0
MEMBERS, PMPM, PMPM_COGS_MONTH = 24_000, 0.85, 4_100.0
ASSESSMENTS, ASSESS_PRICE, ASSESS_COGS = 6_500, 38.0, 4.20
COVERED = 9_600
PROGRESSION = 0.035
REL_REDUCTION = 0.12
DIALYSIS_COST = 98_000.0
SHARE = 0.15
VALIDATION_P = 0.55
D_COGS = 52_000.0


def annuity_factor(rate, years):
    return sum(1 / (1 + rate) ** t for t in range(1, years + 1))


def model_d(p=VALIDATION_P, share=SHARE, reduction=REL_REDUCTION, attribution=1.0):
    avoided = COVERED * PROGRESSION * reduction * attribution
    savings = avoided * DIALYSIS_COST
    gross_share = savings * share
    expected = gross_share * p
    return dict(avoided=avoided, savings=savings, gross_share=gross_share,
                expected=expected, net=expected - D_COGS)


a_rev, a_cogs = SEATS * SEAT_PRICE * 12, SEATS * SEAT_COGS * 12
b_rev, b_cogs = MEMBERS * PMPM * 12, PMPM_COGS_MONTH * 12
c_rev, c_cogs = ASSESSMENTS * ASSESS_PRICE, ASSESSMENTS * ASSESS_COGS
d = model_d()

print("ANNUAL REVENUE AND MARGIN")
print("  model                        revenue      COGS   gross profit  margin")
for name, rev, cogs in (("A per seat", a_rev, a_cogs),
                        ("B PMPM", b_rev, b_cogs),
                        ("C per assessment", c_rev, c_cogs),
                        ("D shared savings expected", d["expected"], D_COGS)):
    print(f"  {name:28s} ${rev:10,.0f}  ${cogs:9,.0f}  ${rev - cogs:11,.0f}"
          f"  {(rev - cogs) / rev:6.1%}")

print("\nSHARED SAVINGS BUILD-UP")
print(f"  covered members          {COVERED:,}")
print(f"  dialysis starts          {COVERED * PROGRESSION:,.0f}")
print(f"  avoided starts           {d['avoided']:.1f}")
print(f"  gross savings            ${d['savings']:,.0f}")
print(f"  share at {SHARE:.0%}            ${d['gross_share']:,.0f}")
print(f"  expected at {VALIDATION_P:.0%}       ${d['expected']:,.0f}")
print(f"  net after cost to serve  ${d['net']:,.0f}")

af = annuity_factor(RATE, YEARS)
pv_a = (a_rev - a_cogs) * af
pv_b = (b_rev - b_cogs) * af
pv_c = (c_rev - c_cogs) * af
pv_d = d["net"] * sum(1 / (1 + RATE) ** t for t in (2, 3, 4))

print("\nFIVE-YEAR PV AT 12%")
print(f"  annuity factor        {af:.4f}")
for name, pv in (("A per seat", pv_a), ("B PMPM", pv_b),
                 ("C per assessment", pv_c), ("D shared savings", pv_d)):
    print(f"  {name:20s} ${pv:12,.0f}")
print(f"  D nominal 3-year share ${d['gross_share'] * 3:,.0f}")

print("\nVOLUME TO MATCH MODEL A")
print(f"  members at ${PMPM:.2f}       {a_rev / 12 / PMPM:,.0f}")
print(f"  assessments at ${ASSESS_PRICE:.0f}   {a_rev / ASSESS_PRICE:,.0f}")
print(f"  covered members         "
      f"{a_rev / (PROGRESSION * REL_REDUCTION * DIALYSIS_COST * SHARE * VALIDATION_P):,.0f}")

print("\nSENSITIVITY: validation probability")
print("  p     expected share   net        PV")
for p in (0.30, 0.45, 0.55, 0.70, 0.85, 1.00):
    dd = model_d(p=p)
    pv = dd["net"] * sum(1 / (1 + RATE) ** t for t in (2, 3, 4))
    print(f"  {p:4.2f}  ${dd['expected']:12,.0f}  ${dd['net']:9,.0f}  ${pv:11,.0f}")

print("\nSENSITIVITY: attributable share")
print("  attribution  gross share   net        PV")
for att in (0.30, 0.50, 0.70, 0.85, 1.00):
    dd = model_d(attribution=att)
    pv = dd["net"] * sum(1 / (1 + RATE) ** t for t in (2, 3, 4))
    print(f"  {att:9.0%}  ${dd['gross_share']:12,.0f}  ${dd['net']:9,.0f}  ${pv:11,.0f}")

print("\nCONTRACT LENGTH WITH 18-MONTH LAG")
for years in (3, 4, 5):
    payments = max(0, years - 1)
    pv = d["net"] * sum(1 / (1 + RATE) ** t for t in range(2, 2 + payments))
    print(f"  {years}-year contract: {payments} payment cycles, PV ${pv:,.0f}")

print("\nHYBRID: BASE FEE PLUS SHARED SAVINGS")
base_fee_month = 9_000.0
base_annual = base_fee_month * 12
trend_discount = 0.75
variable = d["gross_share"] * trend_discount * VALIDATION_P
print(f"  base fee per year        ${base_annual:,.0f}")
print(f"  variable, trend-adjusted ${variable:,.0f}")
print(f"  total gross profit       ${base_annual - D_COGS + variable:,.0f}")
print(f"  floor if nothing validates ${base_annual - D_COGS:,.0f}")
```

Expected output:

```text
ANNUAL REVENUE AND MARGIN
  model                        revenue      COGS   gross profit  margin
  A per seat                  $   205,200  $   38,880  $    166,320  81.1%
  B PMPM                      $   244,800  $   49,200  $    195,600  79.9%
  C per assessment            $   247,000  $   27,300  $    219,700  88.9%
  D shared savings expected   $   325,987  $   52,000  $    273,987  84.0%

SHARED SAVINGS BUILD-UP
  covered members          9,600
  dialysis starts          336
  avoided starts           40.3
  gross savings            $3,951,360
  share at 15%            $592,704
  expected at 55%       $325,987
  net after cost to serve  $273,987

FIVE-YEAR PV AT 12%
  annuity factor        3.6048
  A per seat            $     599,546
  B PMPM                $     705,094
  C per assessment      $     791,969
  D shared savings      $     587,563
  D nominal 3-year share $1,778,112

VOLUME TO MATCH MODEL A
  members at $0.85       20,118
  assessments at $38   5,400
  covered members         6,043

SENSITIVITY: validation probability
  p     expected share   net        PV
  0.30  $     177,811  $  125,811  $    269,801
  0.45  $     266,717  $  214,717  $    460,459
  0.55  $     325,987  $  273,987  $    587,563
  0.70  $     414,893  $  362,893  $    778,221
  0.85  $     503,798  $  451,798  $    968,878
  1.00  $     592,704  $  540,704  $  1,159,536

SENSITIVITY: attributable share
  attribution  gross share   net        PV
        30%  $     177,811  $   45,796  $     98,210
        50%  $     296,352  $  110,994  $    238,025
        70%  $     414,893  $  176,191  $    377,840
        85%  $     503,798  $  225,089  $    482,702
       100%  $     592,704  $  273,987  $    587,563

CONTRACT LENGTH WITH 18-MONTH LAG
  3-year contract: 2 payment cycles, PV $413,440
  4-year contract: 3 payment cycles, PV $587,563
  5-year contract: 4 payment cycles, PV $743,031

HYBRID: BASE FEE PLUS SHARED SAVINGS
  base fee per year        $108,000
  variable, trend-adjusted $244,490
  total gross profit       $300,490
  floor if nothing validates $56,000
```

The most useful line in the output is the last one. A hybrid with a 56,000 USD floor is a business
a finance team can fund; a pure shared-savings contract with a zero floor is a bet on a
measurement that has not yet been agreed.

Limitations: the model treats validation probability as independent of contract design, which it
is not; a pre-agreed methodology raises the probability. It also ignores the cost of the
measurement itself, which in a shared-savings arrangement includes claims analysis, actuarial
support, and dispute resolution, and can run to 5 to 10 percent of the contract value.

## 8. Data, standards, and regulatory context

### 8.1 Fraud and abuse constraints on outcomes-based arrangements

Outcomes-based and shared-savings arrangements between a vendor and a provider can create
remuneration that flows in connection with referrals. Two federal statutes are relevant in the
United States.

The **Anti-Kickback Statute** at 42 U.S.C. 1320a-7b(b) prohibits knowingly offering or paying
remuneration to induce referrals of items or services payable by federal health care programmes.
The statute has safe harbours, including a discount safe harbour and a value-based safe harbour
for certain arrangements at 42 CFR 1001.952. A shared-savings payment that is structured as a
discount on the price of the vendor's own product is generally distinguishable from a payment for
referrals, but the structure and the documentation matter.

The **Physician Self-Referral Law (Stark)** at 42 U.S.C. 1395nn prohibits a physician from
referring designated health services to an entity with which the physician has a financial
relationship, subject to exceptions. The value-based exceptions at 42 CFR 411.357 include
requirements on the arrangement's commercial reasonableness, the absence of volume-based
compensation tied to referrals, and the writing and signature of the agreement.

The practical implication: an outcomes-based contract should be reviewed by counsel, should be in
writing, should tie payment to a measured clinical or cost outcome rather than to the volume of
referrals, and should be priced at fair market value. A vendor that offers a share of savings
without that review is taking a regulatory risk that no margin justifies.

### 8.2 Revenue recognition

Under ASC 606, revenue from a contract with a customer is recognised when control of the promised
good or service transfers, in an amount reflecting the consideration to which the entity expects
to be entitled. Variable consideration, including a shared-savings payment, must be estimated and
included only to the extent that it is probable that a significant reversal will not occur when
the uncertainty is resolved.

That constraint has a direct commercial consequence. A shared-savings stream that depends on an
18-month claims validation may not be recognisable as revenue until the validation occurs, even
though the service was delivered. A public company reporting under United States generally
accepted accounting principles must therefore treat the arrangement as a contingent asset, which
affects reported revenue, sales compensation, and investor communication.

A subscription is fixed consideration and is recognised ratably. That is one reason subscriptions
dominate: not because they are better for the customer, but because they are simpler to account
for and to forecast.

### 8.3 Value-based payment programmes as a template

The Medicare Shared Savings Program provides a public template for how shared-savings contracts
are structured: a benchmark, a minimum savings rate, a sharing rate that varies by risk, quality
gates, and a settlement process after claims runout. A vendor contracting with an accountable care
organisation should understand the ACO's own arrangement, because the vendor's share is usually a
fraction of the ACO's share, and the ACO's quality gates apply upstream.

The general structure is:

```text
vendor_payment = (benchmark - actual_cost) x aco_share x vendor_share x quality_factor
```

Each term is a negotiation. Each term is also a place where the vendor's expected value can fall
by half.

### 8.4 Pricing transparency and contract terms

Payers increasingly require most-favoured-customer terms, price floors, and audit rights over the
vendor's pricing to other customers. A per-seat price that varies by segment can trigger these
clauses. A PMPM price is easier to defend because it is tied to the population rather than to the
buyer's negotiating leverage.

### 8.5 What the rules do not settle

None of these rules determines which model is commercially best. They constrain the design space:
an outcomes contract that cannot be measured will fail commercially, and one that is structured as
inducement will fail legally. Within those constraints, the choice is a matter of strategy,
capability, and the buyer's budget.

## 9. Industry practice and case evidence

### 9.1 Documented: device manufacturers have run risk-sharing arrangements for years

A review of new payment models in medical technology documents several concrete structures:
Medtronic's arrangement for an antibacterial sleeve, reported as covering hospital costs if a
surgical-site infection occurs across roughly 1,000 contracts; a St. Jude Medical and Abbott
arrangement for a cardiac resynchronisation lead offering a 45 percent rebate if a lead revision
is needed within a year; a Johnson & Johnson discount on repeat ablation within a year for a
catheter; Medtronic and Aetna gain-sharing on insulin pumps; a Bard arrangement sharing savings
from an antimicrobial catheter; and a Philips monitoring-as-a-service agreement with Jackson
Health (Deloitte, *New payment models in medtech*, 2024).

These are warranties and performance guarantees attached to a product sale, not pure outcomes
contracts. They transfer a defined, bounded risk rather than a share of an uncertain outcome. That
is why they are commercially workable: the vendor can price the expected cost of the guarantee.

### 9.2 Documented: risk sharing helps only under specific conditions

Towse and Fenwick analysed performance-based risk-sharing arrangements using a value-of-
information framework and found that risk sharing improved outcomes in 9 of 16 payer and innovator
expectation combinations, with the result depending heavily on who funds the research and the
administration of the scheme (Value in Health, 2024).

An international review of outcomes-based agreements synthesised 49 studies across 31 countries
and identified the practical barriers: administrative burden, weak data infrastructure, contract
ambiguity, and difficulty enforcing rebates or delisting when results are negative. The enablers
were small scheme scale, automated electronic health record or registry data, independent
third-party analysis, transparency, and clinical champions (Institute of Health Economics, 2024).

The conclusion is not that outcomes-based models do not work. It is that they work when the
measurement infrastructure exists and the scheme is small enough to administer. A vendor selling
to a payer with claims data and a value-based contract is in a different position from one selling
to a fragmented provider group.

### 9.3 Documented: binary outcomes create payment volatility

A real-world test of an outcomes-based agreement for a metastatic breast cancer medicine in Wales
found that binary outcomes such as one-year survival produce high payment volatility, while
continuous outcomes and linear payment models stabilise payments (PharmacoEconomics, 2024). The
same analysis noted 99 outcomes-based agreements executed or announced in the United States
between 2009 and 2021.

For a digital health product, the analogue is a binary endpoint such as "avoided dialysis start"
versus a continuous one such as "change in estimated glomerular filtration rate." The binary
endpoint is easier to explain and much more volatile, because it depends on a small number of
events.

### 9.4 Documented: gross margins vary by model

Published benchmarks for health technology businesses place business-to-business software as a
service gross margins at roughly 75 to 87 percent, electronic health record subscriptions at 65 to
80 percent, enterprise hospital licences at 55 to 70 percent because of implementation intensity,
and digital therapeutics and consumer subscriptions at 75 to 85 percent. Tech-enabled services,
which include clinical labour, range from about 25 percent at the bottom quartile to 60 percent at
the top quartile at scale (industry benchmark compilations, 2024 to 2026).

The pattern is that margin falls as human labour enters the cost of goods. A model that requires
care managers to deliver the service is a services business with software attached, whatever the
pricing label says.

### 9.5 Documented: price levels differ by payment route

A benchmark of 1,137 digital health solutions found average prices of about 340 USD per user per
quarter for reimbursement-enabled solutions against about 125 USD for non-reimbursed solutions,
with substantial variation by country and clinical area (research2guidance, 2024).

The direction of causation is ambiguous, but the correlation is consistent with the argument in
Topic 04: a payment route raises the price the market will bear, because the buyer is spending
someone else's money against a defined benefit.

### 9.6 Inference: what this implies for model choice

The following is inference from the sources above, not a documented finding.

First, bounded risk transfers (warranties, rebates on a defined event) are easier to sell and to
price than open-ended savings shares. A vendor should prefer a bounded guarantee unless it has the
actuarial capability and the capital to carry an open-ended one.

Second, the measurement infrastructure determines the model. A buyer without claims data cannot
support a shared-savings contract, and a contract that assumes data the buyer does not have will
fail at validation.

Third, the model should be chosen to fit the buyer's budget cycle, not the vendor's preference.
A payer budgets the medical loss ratio annually; a provider budgets operations annually; a
capital committee budgets five years. The model that matches the cycle closes faster.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Model chosen by default | "We always do per seat" | No analysis of alternatives | Structural mismatch with value | Evaluate at least three models |
| Unit of value mismatched | Price does not scale with population | Per-seat pricing for a population product | Buyer arbitrages the mismatch | Align unit with value driver |
| Unbounded risk transfer | Vendor accepts any savings share | No cap or floor | Insolvency on a bad measurement | Cap and floor the arrangement |
| No attribution method | "We will agree the methodology later" | Dispute deferred, not resolved | Payment withheld | Specify method pre-signature |
| Binary outcome | Payment all-or-nothing on a rare event | Small numbers, high variance | Revenue swings of 50 percent | Use continuous or linear measures |
| Cash lag ignored | No working capital plan | 18-month lag funded from operations | Cash crisis mid-contract | Model lag and secure financing |
| Price anchored to cost | Discount approved to win | Margin below cost to serve | Negative unit economics at scale | Anchor to value or competition |
| Margin assumes no labour | Model treats care management as software | Human cost in COGS | Margin halves | Classify the business honestly |
| Regulatory review skipped | Shared savings signed without counsel | Anti-kickback or Stark exposure | Contract void, penalties | Legal review before signature |
| Revenue recognised early | Contingent revenue booked at signing | ASC 606 constraint ignored | Restatement risk | Involve finance early |
| Expansion undefined | No path beyond the first contract | LTV capped | CAC never recovered | Define expansion levers |
| Sales motion mismatch | 30,000 USD ACV with enterprise sales | Cost of sale exceeds contract | Negative CAC payback | Change motion or price |

Two red flags are diagnostic. The first is a business model with no floor in a contingent
arrangement; that is not alignment, it is an unhedged bet. The second is a gross margin above 90
percent in a product that requires clinical labour to deliver; the labour is being omitted from
the cost of goods, and the margin will collapse when it is counted.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The value unit is measurable | Model design depends on it | No data source | Pilot the measurement |
| 2 | Validation probability is estimable | Risk-adjusted PV needs it | No precedent | Use comparable contracts |
| 3 | The effect size holds in the payer's population | Savings calculation depends on it | Different case mix | Local pilot |
| 4 | Cost to serve is stable | Margin depends on it | Volume or support intensity changes | Track cost per account |
| 5 | The buyer will accept the risk allocation | Contract must be signed | Buyer transfers all risk | Negotiate early |
| 6 | Attribution is feasible | Shared savings requires it | No comparison group | Design the measurement |
| 7 | The contract term is long enough | Cash timing depends on it | Buyer insists on three years | Model the shorter term |
| 8 | Regulatory review clears the structure | Contract validity | Inducement finding | Counsel review |
| 9 | Revenue recognition is as modelled | Financial reporting | ASC 606 constraint | Finance review |
| 10 | Expansion is available | LTV requires it | Contract caps population | Negotiate expansion terms |

The deepest limit is that a business model is a bet on a future measurement, and the measurement
is usually controlled by the party that pays. A vendor can negotiate the methodology, but it
cannot control the payer's data, the payer's concurrent initiatives, or the payer's budget. The
model should therefore be designed so that the worst plausible measurement is survivable, not so
that the best plausible measurement is impressive.

## 12. Exercises

### Exercise 1. Unit of value

**Deliverable:** for three products, identify the unit of value, the unit of use, and whether they
match.
**Criteria:** each unit is defined operationally; at least one mismatch is identified with its
consequence. **Time:** 30 minutes.

### Exercise 2. Model comparison

**Deliverable:** a table comparing per-seat, PMPM, per-test, and shared-savings models for one
product on revenue, margin, volatility, and cash timing.
**Criteria:** at least four rows and six columns; each cell has a value or an explicit "not
applicable"; the recommendation is stated. **Time:** 90 minutes.

### Exercise 3. Risk-adjusted PV

**Deliverable:** compute the present value of a shared-savings stream with a 600,000 USD annual
expected share, a 60 percent validation probability, a two-year lag, and a four-year contract at a
10 percent discount rate.
**Criteria:** correct discounting; the probability applied to the cash flow, not the discount
rate; the number of payment cycles stated. **Time:** 30 minutes.

### Exercise 4. Contract mechanism

**Deliverable:** a one-page mechanism specification with trigger, measure, source, reconciliation,
true-up, cap, floor, and exit.
**Criteria:** all nine elements present; the measure is auditable; the dispute process has a named
decision maker and a deadline. **Time:** 60 minutes.

### Exercise 5. Attribution design

**Deliverable:** an attribution plan for a shared-savings contract, with the method, the matching
criteria, the trend discount, and the dispute process.
**Criteria:** the method is one of the five in Section 5.8; the discount is justified; the plan is
pre-registered. **Time:** 60 minutes.

### Exercise 6. Budget fit

**Deliverable:** map four buyer types to a recommended model and explain the budget constraint
that drives the choice.
**Criteria:** each recommendation cites the buyer's budget cycle or approval path; at least one
recommendation rejects an outcome model. **Time:** 45 minutes.

### Exercise 7. Regulatory and accounting screen

**Deliverable:** a 400-word memo identifying the fraud and abuse and revenue recognition issues in
a proposed shared-savings contract, with the questions to put to counsel and finance.
**Criteria:** both statutes named; the ASC 606 constraint explained; at least three specific
questions. **Time:** 60 minutes.

### Exercise 8. Hybrid design

**Deliverable:** design a hybrid with a base fee, a variable component, a cap, and a floor, and
compute the gross profit at three validation probabilities.
**Criteria:** the floor covers cost to serve; the cap is justified; the three scenarios are
computed and interpreted. **Time:** 90 minutes.

## 13. Self-check answers

**1. What are the three axes of a business model?**
Unit of value, risk bearer, and cash timing. Two models that differ on more than one axis are
different businesses, not different prices.

**2. Why is a per-seat model poorly matched to a population health product?**
Because the value scales with the population and the price scales with the care team's headcount.
A payer that expands coverage gets more value without paying more, which is a mismatch the buyer
will exploit and the vendor cannot defend at renewal.

**3. Which model has the highest gross margin and why?**
In the worked example the per-assessment model at 88.9 percent, because its cost is almost
entirely variable and small relative to price. A high margin is not necessarily the best model;
the per-assessment model also has the most volatile volume.

**4. Compute the Exercise 3 answer.** Expected annual cash = 600,000 x 0.60 = 360,000 USD. A
four-year contract with a two-year lag yields payments at the end of years 3 and 4, so two cycles.
PV at 10 percent = 360,000 / 1.1^3 + 360,000 / 1.1^4 = 270,472 + 245,884 = 516,356 USD.

**5. Why does a shared-savings contract need a floor?**
Because the vendor incurs cost to serve regardless of whether the savings validate. Without a
floor, a failed measurement produces negative gross profit and, at scale, insolvency.

**6. What is attribution risk and how is it mitigated?**
The risk that the measured savings cannot be attributed to the product, because of concurrent
initiatives, secular trend, or regression to the mean. It is mitigated by a pre-agreed method,
ideally a randomised or matched comparison, with a stated trend discount and a dispute process.

**7. Why might a payer prefer PMPM to per seat?**
Because PMPM aligns the price with the population whose cost the payer bears, is budgetable
against the medical loss ratio, and does not require the payer to police the vendor's seat count.

**8. What does ASC 606 require for contingent revenue?**
That variable consideration be estimated and included only to the extent it is probable that a
significant reversal will not occur. A shared-savings stream that may be clawed back is therefore
constrained, often until validation.

**9. Why is a binary outcome harder to contract than a continuous one?**
Because it is driven by a small number of events, so the measured result is volatile and the
payment swings widely for reasons unrelated to product performance. Continuous outcomes and linear
payment schedules stabilise both.

**10. What is the difference between value-based pricing and an outcomes-based contract?**
Value-based pricing is a price level anchored to the value created. An outcomes-based contract is
a payment mechanism where part of the price is contingent on a measured outcome. A product can
have one without the other.

**11. What is a most-favoured-customer clause and why does it constrain pricing?**
A term requiring that the buyer receive pricing at least as favourable as any other customer. It
constrains segment-specific discounts, because a lower price to one buyer triggers the clause for
another.

**12. Rubric for Exercise 8.** Full marks require a floor above cost to serve, a cap justified by
the buyer's risk tolerance, three computed scenarios, and a statement of the gross profit at zero
validation. Half marks if the hybrid is described without the floor computation.

## 14. Interview questions

**1. How would you choose a business model?**
Strong: works through unit of value, risk allocation, cash timing, and buyer budget, then compares
risk-adjusted PV. Weak: says it depends on the customer, without a method.

**2. Why is per-seat pricing so common?**
Strong: because it is predictable, easy to budget, easy to account for, and maps to a procurement
line, and notes that it misaligns when value scales with population or volume. Weak: says it is
standard practice.

**3. When would you use shared savings?**
Strong: when the outcome is measurable, attribution is feasible, the contract term exceeds the
lag, and the vendor can fund the working capital or has a base fee. Weak: when the buyer says they
cannot afford it.

**4. How do you price a product with no comparable?**
Strong: anchors to value created and to the buyer's cost of the alternative, with a cost floor,
and presents a range. Weak: picks a number from a competitor.

**5. How do you design an outcomes contract that will not be disputed?**
Strong: pre-agrees the measure, source, reconciliation, true-up, and dispute process, and prefers
a randomised or matched comparison. Weak: says they would write a clear contract.

**6. What is the working capital impact of an 18-month payment lag?**
Strong: quantifies it as months of lag times monthly cost to serve plus implementation cost, and
connects it to runway. Weak: says it is a cash flow issue.

**7. How does the buyer's budget cycle affect model choice?**
Strong: payer budgets the medical loss ratio annually, provider budgets operations annually,
capital committee budgets multi-year, and the model should match the cycle. Weak: says buyers
prefer monthly.

**8. What are the fraud and abuse risks in an outcomes contract?**
Strong: names the Anti-Kickback Statute and Stark, and says counsel must review the structure,
fair market value, and the absence of volume-based referral compensation. Weak: says compliance
will review it.

**9. How do you compute the present value of a contingent stream?**
Strong: probability-weights the cash flow, applies the lag, discounts at the cost of capital, and
counts the payment cycles the term allows. Weak: discounts the nominal amount.

**10. When would you build a hybrid?**
Strong: when the buyer needs a budgetable base and the vendor needs a floor, with a capped
variable component. Weak: says hybrids are always better.

**11. How does gross margin differ between software and tech-enabled services?**
Strong: gives ranges and explains that clinical labour in the cost of goods is the driver. Weak:
says software margins are higher.

**12. What would make you change a business model after launch?**
Strong: names a trigger such as churn concentrated in low-volume accounts, a validation dispute
rate, or a change in the payment route, and says what evidence would trigger the change. Weak:
says they would change if it was not working.

## 15. Key terms

- **Anti-kickback statute** — prohibition on remuneration for referrals.
- **ARR** — annual recurring revenue.
- **At-risk** — payment contingent on performance.
- **Attribution** — assigning an outcome to an intervention.
- **Capitation** — fixed payment per member for a scope of services.
- **COGS** — cost of goods sold.
- **Constraint** — ASC 606 limit on recognising variable consideration.
- **Contribution margin** — revenue minus variable cost.
- **Cost to serve** — direct cost of delivering to a customer.
- **Gross margin** — revenue minus cost of goods sold.
- **Guardrail** — criterion defining when payment applies.
- **Hybrid model** — base fee plus variable component.
- **LTV** — lifetime value.
- **Most-favoured-customer** — clause requiring best available pricing.
- **NRR** — net revenue retention.
- **Outcomes-based contract** — payment contingent on a measured outcome.
- **PMPM** — per member per month.
- **Per case** — price per episode.
- **Per seat** — price per named user.
- **Per test** — price per test or assessment.
- **Revenue recognition** — accounting rules for when revenue is recorded.
- **Risk corridor** — shared band of upside and downside.
- **Risk-sharing** — vendor bears part of a loss.
- **Shared savings** — vendor receives a share of measured savings.
- **Stark law** — physician self-referral prohibition.
- **True-up** — reconciliation of actual versus estimated payment.
- **Unit of value** — the thing that triggers payment.
- **Value-based pricing** — price anchored to value created.
- **Variable consideration** — consideration contingent on future events.
- **Warranty** — refund or remedy if performance fails.
- **Working capital** — cash tied up between cost and collection.

## 16. Sources

All URLs accessed 26 October 2026.

1. Towse A, Fenwick E. It takes 2 to tango: setting out the conditions in which performance-based
   risk-sharing arrangements work for both parties. *Value in Health*. 2024.
   https://doi.org/10.1016/j.jval.2024.03.2196
2. Institute of Health Economics. *Barriers and Enablers for Outcomes-based Agreements: An
   International Review*. September 2024.
   https://innovation.ihe.ca/wp-content/uploads/2024/09/OBA-Barriers-and-Enablers-Sep2024.pdf
3. Deloitte. *New Payment Models in Medtech*. 2024.
   https://www.deloitte.com/content/dam/insights/articles/2024/6461_medtech-payment-models/di-medtech-payment-models.pdf
4. Value-Based Healthcare in Practice: IDEATE, a collaboration to design and test an
   outcomes-based agreement for a medicine in Wales. *PharmacoEconomics*. 2024.
   https://doi.org/10.1007/s40273-024-01445-0
5. Illustrating the financial consequences of outcome-based payment models from a payer's
   perspective: the case of atidarsagene autotemcel (Libmeldy). *Value in Health*. 2024.
   https://doi.org/10.1016/j.jval.2024.05.010
6. Bessemer Venture Partners. *Benchmarks for Growing Health Tech Businesses*.
   https://www.bvp.com/atlas/benchmarks-for-growing-health-tech-businesses
7. research2guidance. *Global Price and Business Model Benchmarking*, Health.Tech Conference 2024.
   https://research2guidance.com/r2gs-global-price-and-business-model-benchmarking-first-results-discussed-at-the-health-tech-conference-2024/
8. Stripe. *Pricing Models for AI in Healthcare*.
   https://stripe.com/resources/more/pricing-models-for-ai-healthcare-tools
9. Centers for Medicare and Medicaid Services. *Medicare Shared Savings Program*.
   https://www.cms.gov/medicare/payment/prospective-payment-systems/medicare-shared-savings-program
10. US Department of Health and Human Services, Office of Inspector General. *Anti-Kickback Statute
    and safe harbours*, 42 CFR 1001.952.
    https://oig.hhs.gov/compliance/safe-harbors/
11. Centers for Medicare and Medicaid Services. *Physician Self-Referral (Stark) Law*, 42 CFR 411.
    https://www.cms.gov/medicare/physician-self-referral
12. Financial Accounting Standards Board. *Accounting Standards Codification Topic 606, Revenue
    from Contracts with Customers*.
    https://www.fasb.org/standards/topics/asc-606

Figures in Section 6 are illustrative and constructed for teaching. The clinical parameters are
plausible but not drawn from a specific contract. No company outcome is asserted.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`06_adoption.md`](06_adoption.md)
- Next topic: [`08_leadership.md`](08_leadership.md)
- Related topic: [`03_health-economics.md`](03_health-economics.md)
- Module survey: [`../../course/17_product_economics_leadership.md`](../../course/17_product_economics_leadership.md)
- Labs: [`../../labs/`](../../labs/)
