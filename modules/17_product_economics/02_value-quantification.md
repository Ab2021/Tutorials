# Value Quantification

> Module 17 — Product, Economics, and Leadership
> Topic 02 of 09
> Estimated study time: 4 hours
> Prerequisites: 01 (Product Discovery)
> Version 2026.3

## 1. Why this topic exists

Every health technology pitch contains a number. Most of those numbers are produced by a process
that would fail a peer review: a time saving multiplied by a salary, an avoided event multiplied
by a list price, a percentage lifted from a vendor brochure. The number is usually too large, it
usually double counts, and it usually answers a question the buyer did not ask.

Value quantification is the discipline of turning a clinical or operational effect into a claim
that survives a finance committee, a payer's evidence review, and a renewal conversation two
years later. It is not optimistic arithmetic. It is the opposite: it is the practice of finding
every place where value leaks between the model's output and the buyer's bank account.

The stakes are asymmetric. An overstated value case wins the deal and loses the renewal, because
the buyer measures the promised benefit and finds it absent. An understated case loses the deal
and preserves the relationship. The second error is cheaper, which is why experienced teams
present a range and a break-even rather than a headline.

### 1.1 The three currencies

Clinical benefit, economic value, and willingness to pay are different quantities, measured in
different units, by different people. Confusing them is the most common analytical error in the
field.

- **Clinical benefit** is a change in health outcome: fewer infections, lower blood pressure,
  earlier diagnosis. It is measured in events, rates, or quality-adjusted life years (QALYs).
- **Economic value** is the monetised consequence of that change to a defined party: avoided
  cost, freed capacity, additional revenue, reduced risk. It is measured in currency per period,
  with a stated perspective.
- **Willingness to pay** is what a specific buyer will actually commit from a specific budget. It
  is revealed by a signature, not by a survey.

A product can score well on the first, poorly on the second, and zero on the third. Screening
programmes that prevent expensive complications frequently have strong clinical benefit and
negative economic value to the provider who must fund them, because the savings accrue to a payer
years later. That is a real and common situation, not a modelling failure.

### 1.2 Why the number is usually wrong

Four mechanisms account for most overstatement.

1. **The leakage assumption.** A time saving per encounter is multiplied by every encounter,
   ignoring that only some clinicians adopt the tool, only some encounters use it, and only some
   saved minutes are redeployed rather than absorbed.
2. **The double count.** The same minutes are valued twice: once as time saved at a loaded rate,
   and again as additional visits generated from the freed time.
3. **The price basis error.** A list price, an allowed amount, a negotiated rate, and a paid
   amount are four different numbers. Using the wrong one changes the answer by 30 to 60 percent.
4. **The attribution error.** The benefit is credited entirely to the product when part of it
   would have occurred anyway, through secular trends, concurrent initiatives, or regression to
   the mean.

Each of these is detectable before the model is presented. Section 5 gives the checks.

### 1.3 Who this topic is for

Data scientists produce most of the inputs to a value model: effect sizes, base rates,
stratification, and uncertainty. If you do not understand how those inputs become money, you will
supply the wrong statistic, usually a relative risk reduction when the buyer needs an absolute
number in the buyer's population.

Product managers own the model and must be able to defend every assumption in it. Finance and
commercial teams will translate it into a contract; they need the range and the break-even.

### 1.4 What this topic does not cover

It does not cover formal cost-effectiveness analysis, which is Topic 03. It does not cover
pricing strategy, which is Topic 07. It does not cover reimbursement mechanics, which is Topic
04. It gives the bridge between an effect and a defensible financial claim.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Distinguish clinical benefit, economic value, and willingness to pay for a given product | Exercise 1; self-check 1 |
| 2 | Choose and state a perspective and time horizon that match the buyer's decision | Exercise 2; self-check 2 |
| 3 | Convert an effect size into physical units in the buyer's population | Exercise 3; self-check 4 |
| 4 | Apply the four conversion rules and state which one a claim relies on | Exercise 4; interview question 4 |
| 5 | Quantify leakage from adoption, use, and realization | Exercise 5; self-check 5 |
| 6 | Build a full cost stack including review, false positives, and maintenance | Exercise 6; self-check 6 |
| 7 | Detect double counting and attribution error in a value model | Exercise 7; self-check 7 |
| 8 | Write a value claim with a range, a break-even, and an explicit basis | Exercise 8; interview question 10 |

## 3. The core mental model

Value is a chain with four multipliers and one subtraction. Each link can reduce the number by a
large factor, and the reduction is multiplicative, which is why plausible per-unit savings
produce implausible total savings.

### 3.1 The value chain

```text
  EFFECT                    physical change per eligible event
    |                       e.g. 1.6 minutes saved per documented encounter
    v
  x ADOPTION                share of users who use the tool at all
    |                       0.62 in the worked example
    v
  x USE RATE                share of eligible events where the tool is used
    |                       0.58
    v
  x REALIZATION             share of the nominal effect that actually occurs
    |                       0.70, because some saved time is absorbed
    v
  x CONVERSION              share of realized effect that becomes cash or capacity
    |                       0.35, because freed minutes need demand and rooms to become visits
    v
  GROSS VALUE               in the buyer's units
    |
  - FULL COST               licence, implementation, integration, training, review,
    |                       false positives, support, maintenance, exit
    v
  NET VALUE                 the only number a finance committee cares about
```

The chain is the single most useful artifact in this topic. Every disagreement about a value
case can be located on it. "The vendor says 3 minutes, we measured 1.6" is an EFFECT argument.
"They assume every physician uses it" is an ADOPTION argument. "You cannot bill for saved time"
is a CONVERSION argument. Naming the link ends the argument faster than arguing about the total.

### 3.2 The four conversion rules

An effect becomes money by exactly one of four routes. A claim that does not name its route is
not a financial claim.

| Rule | When it applies | What it produces | What must be true |
|---|---|---|---|
| Cash | The effect changes a payment or a purchase | Currency saved or earned | The budget line moves within the contract term |
| Capacity | The effect frees a constrained resource that can be refilled | Additional units at contribution margin | Demand exists and the constraint is real |
| Risk | The effect reduces the probability or size of a loss | Expected value of avoided loss | The loss is borne by the buyer and is insurable or budgeted |
| Option | The effect creates a capability usable later | Strategic value, not cash | Someone will pay for the option, eventually |

The capacity rule is the one most often abused. Freeing physician minutes creates value only if
there is unmet demand and the physical capacity to serve it. A clinic that is fully booked with a
six-week wait converts saved minutes into additional visits. A clinic with empty slots converts
them into nothing, and the honest claim is staff experience, not revenue.

### 3.3 Value leaks in both directions

Leakage is not only a reason to reduce a claim. Some effects are larger than the model says
because they are omitted:

- Reduced clinician turnover, which has a replacement cost typically modelled at 0.5 to 2.0 times
  annual salary for physicians, is usually excluded because attribution is hard.
- Reduced after-hours work, which is real to the clinician and invisible to the ledger.
- Reduced downstream utilisation, which accrues to a payer rather than the provider.
- Reduced malpractice or audit exposure, which is an option on a low-probability loss.

The discipline is not to include everything. It is to state what is excluded and why, so that the
exclusion is a decision rather than an oversight.

### 3.4 The break-even framing

The strongest form of a value claim is not "the benefit is X." It is "the benefit exceeds the
cost if the effect is at least Y." The second statement requires the buyer to disagree with a
threshold rather than with a forecast, and thresholds are easier to accept.

In the worked example, the annual cost is 551,520 United States dollars (USD) at steady state and
the benefit is 999,077 USD at 1.6 minutes saved per encounter. The break-even is 0.88 minutes.
That single number is more persuasive than the ROI, because no one believes 1.6 and everyone can
form a view about 0.88.

### 3.5 Value is a distribution, not a point

Every input has a range. The output therefore has a range. Presenting a point estimate invites a
binary argument about whether it is right; presenting a range invites a conversation about which
input to test first. The input to test first is the one with the largest effect on the output per
unit of measurement cost, which is usually adoption or realization, not the effect size.

### 3.6 Redraw test

Draw the chain, place your current value model on it, and write the number you used at each
multiplier. If any multiplier is absent, you have assumed it equals one. That assumption is
almost always wrong and is almost never written down.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or basis |
|---|---|---|---|
| Clinical benefit | Change in health outcome attributable to the intervention | Used interchangeably with value | Events, rates, QALYs |
| Economic value | Monetised consequence of the change to a defined party | Presented without a perspective | Currency per period |
| Willingness to pay | Maximum a buyer will commit from a budget | Measured by survey | Currency |
| Perspective | The party whose costs and benefits are counted | Left unstated | Provider, payer, patient, society |
| Time horizon | Period over which costs and benefits are counted | Chosen to flatter the result | Years |
| Base rate | Frequency of the event in the buyer's population before intervention | Taken from a trial population | Events per 1,000 per year |
| Relative risk reduction (RRR) | Proportional reduction in event rate | Reported without the base rate | Proportion |
| Absolute risk reduction (ARR) | Difference in event rates | Omitted from commercial claims | Events per 1,000 |
| Number needed to treat (NNT) | Patients treated to prevent one event | Used as a value metric without cost | Patients |
| Effect size | Magnitude of change per eligible unit | Confused with statistical significance | Units per event |
| Leakage | Loss between nominal effect and realized benefit | Ignored in vendor models | Multiplier |
| Adoption rate | Share of eligible users who use the tool | Confused with licence count | Proportion |
| Use rate | Share of eligible events where the tool is used by an adopter | Confused with adoption | Proportion |
| Realization | Share of nominal effect that actually occurs | Assumed to be 1.0 | Proportion |
| Conversion | Share of realized effect that becomes cash or capacity | Unstated in most models | Proportion |
| Cash conversion | Effect that moves a budget line | Claimed without evidence of a budget line | Currency |
| Capacity conversion | Effect that frees a constrained resource | Claimed where demand is absent | Units at margin |
| Contribution margin | Revenue minus variable cost per unit | Confused with price or with profit | Currency per unit |
| Avoided cost | Cost that would have been incurred and now is not | Confused with revenue | Currency |
| Cost avoidance | Category of benefit where spend does not occur | Treated as cash by providers | Currency |
| Hard saving | Reduction in actual spend, visible in the ledger | Claimed for avoided admissions that were not budgeted | Currency |
| Soft saving | Avoided future or notional cost | Presented as hard saving | Currency |
| Fully loaded cost | Cost including benefits, overhead, and support | Confused with salary | Currency per hour |
| List price | Published price before negotiation | Used as the buyer's cost | Currency |
| Negotiated price | Price agreed in a contract | Confused with allowed amount | Currency |
| Allowed amount | Maximum a payer will recognise for a service | Confused with paid amount | Currency |
| Paid amount | Amount actually transferred | Confused with cost | Currency |
| Total cost of ownership (TCO) | All costs over the life of the asset | Reduced to licence fee | Currency over term |
| Ramp | Period before steady-state benefit is reached | Assumed away | Months |
| Run rate | Annualised value at current performance | Confused with realised annual value | Currency per year |
| Discount rate | Rate used to convert future value to present value | Chosen arbitrarily | Percent per year |
| Net present value (NPV) | Sum of discounted future net cash flows | Computed without stating the rate | Currency |
| Internal rate of return (IRR) | Discount rate at which NPV equals zero | Applied to non-conventional cash flows | Percent |
| Payback period | Time until cumulative net benefit covers investment | Computed from go-live rather than from spend | Months |
| Return on investment (ROI) | Net benefit divided by cost | Computed as benefit divided by cost | Ratio |
| Break-even effect | Effect size at which net value is zero | Omitted from most business cases | Units per event |
| Double counting | Valuing the same effect through two rules | Common with time savings | Error |
| Attribution | Assigning a share of the observed change to the intervention | Assumed to be 100 percent | Proportion |
| Regression to the mean | Extreme baseline values move toward average without intervention | Mistaken for effect | Statistical artefact |
| Counterfactual | What would have happened without the intervention | Replaced by the pre-period | Comparison |

## 5. The mechanics, step by step

The procedure produces a value model that a finance reviewer can audit. It assumes you have
completed discovery and have a defined wedge. Each step states its input, its output, and the
reason it exists.

### Step 1. Choose the perspective and write it at the top of the model

**Input:** the buying centre map from discovery.
**Output:** one sentence naming the party whose costs and benefits are counted.
**Rationale:** the same intervention has different value to a provider, a payer, a patient, and
society. A model without a perspective is not conservative or comprehensive; it is incoherent,
because it will count a cost to one party against a benefit to another.

The four standard perspectives:

| Perspective | Counts | Typical buyer question |
|---|---|---|
| Provider | Costs and revenues of the institution | Does this improve margin or capacity? |
| Payer | Covered population costs and quality | Does this improve outcomes per dollar of premium? |
| Patient | Out-of-pocket cost, time, health | Is this worth my money and time? |
| Societal | All costs and benefits, including productivity | Is this a good use of collective resources? |

Write the perspective in the header, for example: "Perspective: 240-physician ambulatory group,
operating budget, three-year horizon, 2026 USD." Every subsequent disagreement can be tested
against it.

### Step 2. Fix the unit of analysis and the time horizon

**Input:** perspective.
**Output:** a unit (per encounter, per patient, per admission), a population, and a horizon.
**Rationale:** unit errors are the most common arithmetic defect. A per-encounter saving applied
to a per-patient population overstates by the ratio of encounters to patients.

Rules:

- The unit must match the level at which the effect occurs. If the model saves time per note,
  the unit is the note, not the patient.
- The population must be the eligible population, not the total population. A sepsis model
  applies to suspected sepsis presentations, not to all admissions.
- The horizon must be long enough to capture the cost of the intervention and short enough that
  the benefit is measurable. Three years is typical for software; ten years for capital
  equipment; lifetime for prevention.

Edge case: a benefit that occurs after the horizon is not zero; it is excluded, and the exclusion
must be stated. Understating a benefit is acceptable if it is labelled.

### Step 3. Establish the baseline from measured data

**Input:** operational data, claims, registry.
**Output:** a baseline with a source and a date for each figure.
**Rationale:** the baseline is the denominator of every claim. A baseline from a vendor case study
is not evidence about the buyer's population.

For each baseline figure record: value, source system, date range, inclusion criteria, and known
defects. Example: "30-day readmission rate 21.6 percent, discharge registry, calendar 2025,
excludes observation stays, excludes transfers to other acute hospitals." That level of
specificity prevents the most common review objection, which is that the number measures
something slightly different from what the buyer measures.

### Step 4. Estimate the effect size with a stated evidence hierarchy

**Input:** published evidence, local pilot, or expert estimate.
**Output:** a point estimate and a range, with the source class named.
**Rationale:** the effect size is usually the least certain input and the most argued about. Naming
its evidence class converts an argument about belief into an argument about study quality.

| Class | Source | Typical use | Adjustment required |
|---|---|---|---|
| A | Randomised trial in a comparable population | Strongest | Check population and care model |
| B | Prospective study with a concurrent comparison | Good | Check selection and confounders |
| C | Pre-post study in a comparable setting | Weak | Assume part of the effect is secular |
| D | Vendor case study or internal pilot without control | Indicative | Apply a discount, typically 30 to 50 percent |
| E | Expert estimate | Last resort | Present as a range, not a point |

For class C and D evidence, the counterfactual is the weak point. A pre-post reduction of 18
percent in readmissions may include a secular decline of 5 percent, a concurrent initiative of 6
percent, and regression to the mean of 3 percent, leaving 4 percent attributable.

### Step 5. Convert the effect into physical units in the buyer's population

**Input:** baseline, effect size, eligible population.
**Output:** units per year: events avoided, minutes freed, visits added, admissions prevented.
**Rationale:** physical units are auditable. Money is not. A reviewer can check whether 7,993
additional visits fit in the building; they cannot check whether 999,077 USD is real until the
units are visible.

Worked conversion:

```text
eligible_events_per_year       = 907,200 encounters
effect_per_event               = 1.6 minutes
nominal_minutes                = 1,451,520 minutes
nominal_hours                  = 24,192 hours
```

At this point the number is already too large, because it assumes every encounter uses the tool.
That assumption is removed in Step 6.

### Step 6. Apply the leakage multipliers

**Input:** nominal units.
**Output:** realized units.
**Rationale:** this is where most value cases fail. Each multiplier is independently defensible
and jointly devastating.

| Multiplier | Question | Typical value | Evidence source |
|---|---|---|---|
| Adoption | What share of eligible users use it at all? | 0.3 to 0.7 in year one | Product analytics, comparable deployments |
| Use rate | Of an adopter's eligible events, what share uses it? | 0.4 to 0.7 | Product analytics |
| Realization | What share of the nominal effect actually occurs? | 0.5 to 0.8 | Time-motion study, not survey |
| Conversion | What share becomes cash or capacity? | 0.2 to 0.5 | Scheduling data, demand analysis |

Apply them multiplicatively and show each one on its own line. A model that shows a single
"efficiency factor" of 0.25 invites the reviewer to challenge the factor; a model that shows four
factors invites the reviewer to challenge the one they know is wrong, which is more productive.

Edge case: adoption and use rate can rise over time, so year one and steady state differ. Model
both. The year-one number determines payback; the steady-state number determines whether the
contract survives renewal.

### Step 7. Apply the conversion rule

**Input:** realized units.
**Output:** currency per year, with the rule named.
**Rationale:** the conversion rule determines whether the claim is cash, capacity, risk, or
option. Each has a different evidentiary standard.

**Cash rule.** The effect must change a payment or a purchase. Test: name the budget line and the
person who controls it. If the spend is not in the budget, avoided spend is a soft saving.

**Capacity rule.** The effect frees a resource that can be refilled. Test: is there unmet demand
and physical capacity? If the clinic has a six-week wait and empty procedure rooms, freed minutes
become visits. If not, they do not.

```text
capacity_value = freed_hours x conversion_share / hours_per_unit x contribution_margin
```

**Risk rule.** The effect reduces the probability or size of a loss. Test: is the loss borne by
the buyer, and is it budgeted or insured? A prevented never-event has a large social value and a
small expected financial value to a specific hospital, unless the hospital's insurance premium or
penalty exposure changes.

**Option rule.** The effect creates a capability. Test: who will pay for the option and when? If
the answer is "a future payer," the option has value but not a price today.

Most commercial models use the capacity rule and describe it as the cash rule. That substitution
is the single most common misrepresentation in the category.

### Step 8. Cost the full intervention

**Input:** vendor quote, internal estimates, workflow analysis.
**Output:** a cost stack by year, including one-time and recurring items.
**Rationale:** the denominator is as important as the numerator and is usually understated by
omitting the buyer's own labour.

The full stack:

| Cost category | One-time | Recurring | Commonly omitted |
|---|---|---|---|
| Licence or subscription | — | Yes | Overage and volume tiers |
| Implementation | Yes | — | Buyer project management time |
| Integration | Yes | Yes | Interface maintenance after go-live |
| Training | Yes | Yes | Backfill for clinicians in training |
| Clinical review of outputs | — | Yes | The largest hidden cost in decision support |
| False positive handling | — | Yes | Nursing or technician time per alert |
| Validation and monitoring | Yes | Yes | Annual revalidation, drift monitoring |
| Security and compliance | Yes | Yes | Audit, penetration testing, business associate agreement maintenance |
| Support and escalation | — | Yes | Internal service desk tier 1 |
| Exit and data migration | — | Yes | Contractual exit assistance |

The clinical review line is the one to model explicitly. A model that produces 40 alerts per day,
each requiring 90 seconds of nursing review, consumes 1.0 hour per day, or about 250 hours per
year, which at a loaded rate of 65 USD per hour is 16,250 USD per year. That cost frequently
exceeds the licence.

### Step 9. Choose the summary metric that matches the decision

**Input:** net cash flows.
**Output:** one primary metric and at most two secondary metrics.
**Rationale:** different metrics answer different questions, and a model that reports all of them
lets the presenter choose the most favourable.

| Metric | Answers | Use when |
|---|---|---|
| ROI | How much net value per unit of cost? | Comparing initiatives of similar size |
| NPV | Is the investment worth more than its cost of capital? | Capital allocation |
| IRR | What return does the cash flow imply? | Comparing to a hurdle rate |
| Payback | When does the buyer recover the spend? | Operating budget, short horizons |
| Break-even effect | How small can the benefit be and still justify the cost? | Communicating risk |
| Cost per unit of benefit | What does one unit cost? | Comparing modalities within a category |

For an operating-budget software purchase, payback and break-even effect are usually the most
persuasive. For a capital purchase, NPV against the organisation's hurdle rate is the language
finance uses.

### Step 10. Discount and ramp

**Input:** annual cash flows.
**Output:** discounted flows and a payback date.
**Rationale:** a benefit in year three is worth less than the same benefit today, and the first
year rarely delivers steady-state benefit.

Conventions to state explicitly:

- Discount rate: use the organisation's stated rate. Ten percent is a common planning
  assumption; three percent is common in health technology assessment; the difference matters
  over long horizons.
- Timing: end-of-year discounting is standard. State whether year one is discounted.
- Ramp: model adoption as a curve, not a step. A six-month linear ramp to steady state is a
  reasonable default for a workflow tool.

```text
NPV = sum over t of (net_cash_t / (1 + r)^t)
```

### Step 11. Run the attribution and double-count checks

**Input:** the completed model.
**Output:** a corrected model and a list of the corrections.
**Rationale:** these two checks catch most of the error that survives Steps 1 to 10.

**Attribution check.** Subtract the share of the effect that would have occurred without the
product. Sources of non-attributable change: secular trend, concurrent initiatives, regression to
the mean, and changes in coding or documentation. A defensible default for a pre-post estimate is
to discount by 30 to 50 percent; for a controlled estimate, by the confidence interval's lower
bound.

**Double-count check.** For each pair of benefit lines, ask whether they derive from the same
underlying effect. Time saved and additional visits from the same minutes are the classic pair.
So are avoided readmissions and avoided bed-days, when the bed-day is the readmission.

A practical rule: a benefit line may be counted once, under one rule, and every other appearance
of it must be labelled as non-additive context.

### Step 12. Write the claim with its basis

**Input:** the model.
**Output:** a one-paragraph claim with a range, a break-even, and a basis statement.
**Rationale:** the claim is what survives the meeting. It must be short enough to repeat and
precise enough to be audited.

Template:

> At [perspective], over [horizon], the intervention is estimated to produce [range] in net
> value, on a [price basis, year] basis, assuming [effect] (source class [X]), [adoption], and
> [conversion rule]. The break-even effect is [Y]. The estimate excludes [list]. The three
> assumptions most likely to be wrong are [list].

That paragraph is harder to write than a slide and much harder to misquote.

### 5.13 Branch points and edge cases

**No reimbursement pathway.** When no code or coverage policy pays for the service, the
conversion rule cannot be cash-from-payment. It must be cost avoidance against an existing budget
or capacity against existing demand. The value case then needs the budget line and the manager
who owns it, and the model should present the benefit in that manager's units.

**Single-payer versus multi-payer.** In a single-payer system, the payer and the provider may
share a global budget, so avoided admissions reduce cost for the same party that funds the
intervention. In a multi-payer system, the provider bears the cost and may lose revenue when
admissions fall, while the payer gains. The model must state which of these applies; the sign of
the provider's ROI can reverse.

**Capital versus operating budget.** A capital purchase is evaluated on NPV against a hurdle rate
over the asset life, with depreciation and no cancellation. An operating subscription is
evaluated on annual budget impact and is cancellable. A model built for one and presented to the
other will be rejected on process grounds regardless of its content.

**Pilot-to-contract transition.** Pilot benefits are often measured in a favourable setting with
extra support. The contract-year model must use the adoption and realization observed in the
pilot discounted toward the general population, not the pilot's best-performing unit.

**Procurement security review.** Security and compliance costs are recurring and frequently
omitted. Add a line for audit, penetration testing, and any architectural change required by the
buyer's questionnaire, and expect the review to add eight to sixteen weeks before benefit starts.

**Champion turnover.** If the value case depends on a champion's initiative, the benefit can stop
when the champion leaves. Model the benefit as accruing to the workflow, not to the person, and
state what happens to adoption if the champion departs.

**Price basis ambiguity.** When a buyer cites a competitor's price, establish whether it is list,
negotiated, or an introductory rate with a step-up. A three-year model using a year-one
introductory price understates cost by 20 to 40 percent.

### 5.14 Decision table for the conversion rule

The choice of conversion rule depends on the combination of budget movement, demand, and loss
exposure. More than two outcomes are possible.

| Budget line moves | Constrained resource | Loss borne by buyer | Conversion rule | Evidentiary standard |
|---|---|---|---|---|
| Yes, within term | Any | Any | Cash | Budget document and controller sign-off |
| No | Yes, with unmet demand | Any | Capacity | Scheduling data and contribution margin |
| No | Yes, without unmet demand | Any | Capacity at partial value or none | Demand analysis |
| No | No | Yes, budgeted or insured | Risk | Actuarial or penalty exposure model |
| No | No | No | Option only | Named future payer and trigger |
| No | No | No, and no future payer | No economic value; clinical or strategic only | Do not monetise |
| Unknown | Any | Any | Stop and resolve before modelling | Discovery follow-up |

The last row is common and is not a failure. A model built on an unknown budget line will be
rebuilt after the first finance meeting.

## 6. Worked example

> **Note:** the medical group below is a composite constructed for teaching. Every financial
> figure is an illustrative estimate in 2026 USD, not a reported result. Effect-size inputs are
> drawn from the published studies cited in Section 9 and adjusted; the adjustments are stated.

### 6.1 The situation

Northgate Medical Group is a 240-physician multispecialty ambulatory group with 210 clinic days
per year and an average of 18 visits per physician per day, giving 907,200 annual encounters. It
is evaluating an ambient documentation product that listens to the encounter and drafts the note.

The chief medical officer supports it on burnout grounds. The chief financial officer has asked
for a three-year value case on the operating budget. The group has a six-week wait for new
appointments in four of nine specialties, which matters for the conversion rule.

### 6.2 Inputs and their basis

| Input | Value | Basis and source class |
|---|---:|---|
| Annual encounters | 907,200 | Scheduling system, 2025, measured |
| Physicians | 240 | HR roster, measured |
| Licence price | 179 USD per physician per month | Illustrative 2026 list price; negotiated price unknown |
| Implementation | 165,000 USD | Internal estimate, class E |
| Support | 36,000 USD per year | Internal estimate, class E |
| Minutes saved per documented encounter | 1.6 | Class B to C, adjusted downward from trial estimates; see Section 9 |
| Adoption by month 12 | 62% | Class D, comparable deployments |
| Use rate among adopters | 58% | Class D, product analytics analogues |
| Realization | 70% | Class E, time-motion inference |
| Capacity conversion | 35% | Class E, based on four specialties with waiting lists |
| Physician minutes per visit | 16 | Measured, scheduling standard |
| Contribution margin per visit | 125 USD | Finance estimate, 2026, specialty-weighted |
| Loaded physician cost | 185 USD per hour | HR compensation model, 2026 |

Every class D and E input is an assumption and is labelled as one. The two class A inputs, the
encounter count and the physician count, are the least interesting numbers in the model and the
only ones that are measured.

### 6.3 The value chain, computed

```text
nominal minutes      = 907,200 x 1.6            = 1,451,520 minutes
effective encounters = 907,200 x 0.62 x 0.58    = 326,229
realized minutes     = 326,229 x 1.6 x 0.70     = 365,376 minutes
realized hours       = 365,376 / 60             = 6,090 hours
billable hours       = 6,090 x 0.35             = 2,131 hours
incremental visits   = 2,131 / (16 / 60)        = 7,993 visits
annual benefit       = 7,993 x 125              = 999,077 USD
```

The difference between the nominal 1,451,520 minutes and the realized 365,376 minutes is the
leakage: a factor of 3.97. A vendor model quoting the nominal figure and the loaded rate would
claim 4.47 million USD of annual value. The defensible figure is 999,077 USD, and even that
depends on a conversion assumption the buyer has not tested.

### 6.4 The cost stack

| Item | Year 1 (USD) | Years 2-3 (USD per year) |
|---|---:|---:|
| Licence, 240 x 179 x 12 | 515,520 | 515,520 |
| Implementation and integration | 165,000 | 0 |
| Support and maintenance | 36,000 | 36,000 |
| **Total** | **716,520** | **551,520** |

Deliberately excluded from this stack, and therefore from the case: internal project management
time, clinician training backfill, and the information security review. Each is real. Including
them would add an estimated 60,000 to 90,000 USD in year one and 10,000 to 20,000 USD per year
thereafter. The exclusion makes the case look better than it is; it is stated here so that the
reviewer can add it back.

### 6.5 Returns

Year one carries a ramp: adoption builds through the year, so year-one benefit is 55 percent of
steady state.

| Metric | Value |
|---|---:|
| Year 1 benefit | 549,492 USD |
| Year 1 net | -167,028 USD |
| Year 2 and 3 net | 447,557 USD per year |
| Three-year net | 728,086 USD |
| Three-year total cost | 1,819,560 USD |
| Three-year ROI, net divided by cost | 0.40 |
| NPV at 10 percent, year-end discounting | 554,294 USD |
| Payback from go-live | 16.5 months |
| Cost per physician hour freed | 90.57 USD |
| Break-even minutes saved, steady state | 0.88 minutes |
| Break-even minutes saved, year one | 1.15 minutes |

Three observations.

First, the case is positive but not spectacular. A 40 percent three-year ROI on an operating
budget item is fundable if the strategic rationale is accepted; it is not fundable if the finance
committee requires a 2 to 1 return.

Second, the cost per physician hour freed, 90.57 USD, is roughly half the loaded physician cost
of 185 USD per hour. On the surface that looks like a bargain: pay 90.57 USD to free an hour
worth 185 USD. But the hour is only worth 185 USD if it is used for billable work, and the model
already assumes only 35 percent is. The remaining 65 percent is worth whatever the physician does
with it, which the ledger does not record.

Third, the break-even of 0.88 minutes is the most persuasive number in the analysis. Published
estimates for ambient documentation range from about 0.5 to 2.2 minutes per note, so the question
for the buyer is not whether the product works but whether it works by at least 53 seconds per
note in this group. That is a testable question with a local pilot.

### 6.6 Two diagnostics that reveal a weak business case

**Diagnostic 1: the realization test.** Set realization to zero, meaning the saved minutes are
absorbed into the same working day rather than redeployed. Cash value falls to zero and the ROI
becomes minus 100 percent, because the group pays 551,520 USD per year and receives nothing it
can bank. This is not a hypothetical: it is the observed outcome in at least one large
health-system evaluation of ambient documentation, cited in Section 9. If a vendor's model cannot
survive realization being set to zero, the model is a capacity claim in disguise and must be
presented as one.

**Diagnostic 2: the double-count test.** The model's time value at the loaded rate is 6,090 hours
x 185 USD = 1,126,578 USD. If a presenter adds that to the capacity value of 999,077 USD, the
total becomes 2,125,655 USD and the year-two ROI appears as 2.85 rather than 0.81. The error is
to value the same minutes twice, once as time and once as the visits the time produces. The
correct total counts each minute once, under one rule.

A third diagnostic is worth running even though it is not in the required pair: the price-basis
test. Recompute the licence line at 250 USD per physician per month, a plausible negotiated range
for a premium product, and the annual licence becomes 720,000 USD. Year-two net falls to 279,077
USD and the three-year ROI to 0.15. The deal's viability is sensitive to a price the buyer has
not yet been quoted.

### 6.7 Varying one assumption

The base case assumed 1.6 minutes saved per documented encounter. The table holds everything else
constant.

| Minutes saved | Hours freed | Incremental visits | Annual benefit (USD) | Year-2 net (USD) | ROI |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 1,903 | 2,498 | 312,211 | -239,309 | -0.43 |
| 0.8 | 3,045 | 3,996 | 499,538 | -51,982 | -0.09 |
| 1.1 | 4,187 | 5,495 | 686,865 | 135,345 | 0.25 |
| 1.4 | 5,328 | 6,994 | 874,192 | 322,672 | 0.59 |
| 1.6 | 6,090 | 7,993 | 999,077 | 447,557 | 0.81 |
| 1.9 | 7,231 | 9,491 | 1,186,404 | 634,884 | 1.15 |
| 2.2 | 8,373 | 10,990 | 1,373,730 | 822,210 | 1.49 |

The conclusion changes at about 0.9 minutes. Below that the product destroys value on this
buyer's cash basis; above 1.4 minutes it is comfortably fundable. Note that the published range
spans both sides of the threshold, which means the honest answer is "we do not know yet, and here
is the pilot that would tell us."

Now vary adoption instead, holding the effect at 1.6 minutes.

| Adoption | Effective encounters | Annual benefit (USD) | Year-2 net (USD) | ROI |
|---:|---:|---:|---:|---:|
| 25% | 131,544 | 402,853 | -148,667 | -0.27 |
| 35% | 184,162 | 563,995 | 12,475 | 0.02 |
| 45% | 236,779 | 725,136 | 173,616 | 0.31 |
| 62% | 326,229 | 999,077 | 447,557 | 0.81 |
| 75% | 394,632 | 1,208,560 | 657,040 | 1.19 |
| 85% | 447,250 | 1,369,702 | 818,182 | 1.48 |

Adoption breaks even at about 34 percent. That is a more actionable threshold than the effect
size, because adoption is under the vendor's and the group's control. A deployment plan that
cannot credibly reach one third of physicians within twelve months should not be funded, whatever
the per-note saving.

### 6.8 The recommendation

The analysis supports a decision to pilot, not to buy. The conditions: a six-month pilot in the
four specialties with waiting lists, where capacity conversion is credible; a pre-registered
measurement of minutes per note using time-motion sampling rather than self-report; an adoption
target of 35 percent by month six as a continuation gate; and a re-run of this model with local
data before any three-year commitment.

The value of the exercise is not the 554,294 USD NPV. It is the identification of three numbers
that determine the outcome: 0.88 minutes, 34 percent adoption, and the realization assumption,
which is the one that most needs local measurement.

## 7. Code, query, or calculation

The script below implements the value chain, the four conversions used in this case, ROI, NPV,
payback, break-even, and the two diagnostics. It uses only the standard library.

```python
"""Value quantification for an ambient documentation deployment."""

VISITS = 907_200
PHYSICIANS = 240
PRICE_PMPM = 179.0        # USD per physician per month, illustrative list price
ADOPTION = 0.62
USE_RATE = 0.58
REALIZATION = 0.70
CAPACITY_SHARE = 0.35
MIN_PER_VISIT = 16.0
CONTRIBUTION = 125.0      # USD contribution margin per incremental visit
LOADED_RATE = 185.0       # USD per physician hour
IMPL = 165_000.0
SUPPORT = 36_000.0
YEAR1_RAMP = 0.55


def license_cost():
    return PHYSICIANS * PRICE_PMPM * 12


def year1_cost():
    return IMPL + license_cost() + SUPPORT


def steady_cost():
    return license_cost() + SUPPORT


def value(minutes_saved, adoption=ADOPTION, realization=REALIZATION,
          capacity_share=CAPACITY_SHARE):
    """Return (annual benefit in USD, physician hours freed)."""
    effective = VISITS * adoption * USE_RATE * realization
    freed_h = effective * minutes_saved / 60.0
    added_visits = freed_h * capacity_share / (MIN_PER_VISIT / 60.0)
    return added_visits * CONTRIBUTION, freed_h


def break_even_minutes(cost):
    per_minute = (VISITS * ADOPTION * USE_RATE * REALIZATION * CAPACITY_SHARE
                  / (MIN_PER_VISIT / 60.0) * CONTRIBUTION / 60.0)
    return cost / per_minute


def npv(rate, flows, start=0):
    return sum(cf / (1 + rate) ** (t + start) for t, cf in enumerate(flows))


def irr(flows, lo=-0.9, hi=10.0, tol=1e-9):
    """Bisection IRR, valid for a conventional sign pattern."""
    if npv(lo, flows) * npv(hi, flows) > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo, flows) * npv(mid, flows) <= 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def payback_months(annual_net):
    """annual_net: list of annual net cash, index 0 = year 1."""
    cumulative = 0.0
    for year, cf in enumerate(annual_net, start=1):
        if cumulative + cf >= 0:
            return (year - 1) * 12 + (-cumulative) / cf * 12
        cumulative += cf
    return float("inf")


M = 1.6
benefit, freed = value(M)
y1c, sc = year1_cost(), steady_cost()
flows = [benefit * YEAR1_RAMP - y1c, benefit - sc, benefit - sc]

print("UNIT ECONOMICS")
print(f"  annual encounters              {VISITS:,}")
print(f"  licence per year               ${license_cost():,.0f}")
print(f"  year 1 all-in cost             ${y1c:,.0f}")
print(f"  steady-state annual cost       ${sc:,.0f}")
print(f"  effective encounters           {VISITS * ADOPTION * USE_RATE:,.0f}")
print(f"  physician hours freed          {freed:,.0f}")
print(f"  incremental visits             {freed * CAPACITY_SHARE / (MIN_PER_VISIT / 60):,.0f}")
print(f"  annual benefit                 ${benefit:,.0f}")
print(f"  cost per physician hour freed  ${sc / freed:,.2f}")

print("\nRETURNS")
print(f"  year 1 net                     ${flows[0]:,.0f}")
print(f"  year 2 and 3 net               ${flows[1]:,.0f}")
print(f"  three-year net                 ${sum(flows):,.0f}")
print(f"  three-year ROI                 {sum(flows) / (y1c + 2 * sc):.2f}")
print(f"  NPV at 10%                     ${npv(0.10, flows, start=1):,.0f}")
print(f"  payback                        {payback_months(flows):.1f} months")
print(f"  break-even minutes, steady     {break_even_minutes(sc):.2f}")
print(f"  break-even minutes, year 1     {break_even_minutes(y1c):.2f}")

print("\nSENSITIVITY: minutes saved")
print("  minutes  hours  visits    benefit     year-2 net    ROI")
for m in (0.5, 0.8, 1.1, 1.4, 1.6, 1.9, 2.2):
    v, h = value(m)
    net = v - sc
    print(f"  {m:5.1f}  {h:6,.0f}  {h * CAPACITY_SHARE / (MIN_PER_VISIT / 60):6,.0f}"
          f"  ${v:>10,.0f}  ${net:>10,.0f}  {net / sc:6.2f}")

print("\nSENSITIVITY: adoption")
print("  adoption  effective    benefit     year-2 net    ROI")
for a in (0.25, 0.35, 0.45, 0.62, 0.75, 0.85):
    v, _ = value(M, adoption=a)
    net = v - sc
    print(f"  {a * 100:6.0f}%  {VISITS * a * USE_RATE:9,.0f}  ${v:>10,.0f}"
          f"  ${net:>10,.0f}  {net / sc:6.2f}")

print("\nDIAGNOSTICS")
time_value = freed * LOADED_RATE
print(f"  time value at loaded rate      ${time_value:,.0f}")
print(f"  capacity value                 ${benefit:,.0f}")
print(f"  naive sum (double count)       ${time_value + benefit:,.0f}"
      f"  ROI {(time_value + benefit - sc) / sc:.2f}")
print(f"  correct, capacity only         ${benefit:,.0f}  ROI {(benefit - sc) / sc:.2f}")
zero, _ = value(M, realization=0.0)
print(f"  realization zero               ${zero:,.0f}  ROI {(zero - sc) / sc:.2f}")

print("\nIRR CHECK ON A CONVENTIONAL CAPITAL CASE")
cap = [-1_200_000, 380_000, 420_000, 450_000, 420_000, 390_000]
print(f"  NPV at 10%                     ${npv(0.10, cap, start=0):,.0f}")
print(f"  IRR                            {irr(cap) * 100:.1f}%")
```

Expected output:

```text
UNIT ECONOMICS
  annual encounters              907,200
  licence per year               $515,520
  year 1 all-in cost             $716,520
  steady-state annual cost       $551,520
  effective encounters           326,229
  physician hours freed          6,090
  incremental visits             7,993
  annual benefit                 $999,077
  cost per physician hour freed  $90.57

RETURNS
  year 1 net                     $-167,028
  year 2 and 3 net               $447,557
  three-year net                 $728,086
  three-year ROI                 0.40
  NPV at 10%                     $554,294
  payback                        16.5 months
  break-even minutes, steady     0.88
  break-even minutes, year 1     1.15

SENSITIVITY: minutes saved
  minutes  hours  visits    benefit     year-2 net    ROI
    0.5   1,903   2,498  $   312,211  $  -239,309   -0.43
    0.8   3,045   3,996  $   499,538  $   -51,982   -0.09
    1.1   4,187   5,495  $   686,865  $   135,345    0.25
    1.4   5,328   6,994  $   874,192  $   322,672    0.59
    1.6   6,090   7,993  $   999,077  $   447,557    0.81
    1.9   7,231   9,491  $ 1,186,404  $   634,884    1.15
    2.2   8,373  10,990  $ 1,373,730  $   822,210    1.49

SENSITIVITY: adoption
  adoption  effective    benefit     year-2 net    ROI
      25%    131,544  $   402,853  $  -148,667   -0.27
      35%    184,162  $   563,995  $    12,475    0.02
      45%    236,779  $   725,136  $   173,616    0.31
      62%    326,229  $   999,077  $   447,557    0.81
      75%    394,632  $ 1,208,560  $   657,040    1.19
      85%    447,250  $ 1,369,702  $   818,182    1.48

DIAGNOSTICS
  time value at loaded rate      $1,126,578
  capacity value                 $999,077
  naive sum (double count)       $2,125,655  ROI 2.85
  correct, capacity only         $999,077  ROI 0.81
  realization zero               $0  ROI -1.00

IRR CHECK ON A CONVENTIONAL CAPITAL CASE
  NPV at 10%                     $359,679
  IRR                            21.0%
```

Three properties of the script are worth keeping. The `value` function takes the leakage
multipliers as parameters, so a sensitivity sweep is one call. The `break_even_minutes` function
inverts the model analytically rather than searching, so it is exact. And the diagnostics are
computed in the same run as the base case, so they cannot be quietly dropped from the
presentation.

One limitation: the model treats adoption as a scalar rather than a curve, and it assumes the
adopter population has a homogeneous effect size. In practice the early adopters are the
clinicians with the worst documentation burden and the largest effect, so the year-one effect per
adopter is higher than steady state. That biases the ramp upward, not the steady state, which is
why the steady-state break-even is the number to trust.

## 8. Data, standards, and regulatory context

Value quantification has no single regulator, but it has a set of conventions and disclosure
requirements that determine whether a number is defensible. This section covers the ones that
recur.

### 8.1 Price, cost, and payment are not synonyms

The most frequent factual error in a value model is the price basis. The four distinct quantities:

| Quantity | Definition | Where it appears |
|---|---|---|
| List price | Published undiscounted price | Vendor price book, group purchasing organisation catalogue |
| Negotiated price | Contracted price after discount | Contract, sometimes confidential |
| Allowed amount | Maximum a payer recognises for a service | Fee schedule, remittance advice |
| Paid amount | Amount actually transferred after cost sharing | Remittance advice, bank |

For products, the analogue is list price versus effective price after volume tiers, rebates, and
multi-year step-ups. A model built on a list price and presented as the buyer's cost overstates
cost and therefore understates ROI, which is the conservative direction but still an error. A
model built on a negotiated price that the buyer has not been offered is a forecast, not a cost.

For services, the analogue matters more. The 2024 national non-facility rate for Current
Procedural Terminology (CPT) code 92229 was reported at 45.75 USD per screening, but the amount a
practice actually collects depends on payer mix, coinsurance, and denials. A business case for an
autonomous screening camera that assumes the Medicare allowed amount for every screening will
overstate revenue by the share of uninsured, Medicaid, and denied claims.

### 8.2 Cost data sources in the United States

Value models built on provider costs need a source that is both credible and current.

- **Healthcare Cost and Utilization Project (HCUP)** statistical briefs report mean and median
  costs per stay by condition and payer, derived from the Nationwide Readmissions Database and
  the National Inpatient Sample. HCUP Statistical Brief #304 reports an overall 30-day
  readmission rate of 13.9 per 100 index admissions and a mean readmission cost of 16,300 USD
  against 14,500 USD for index admissions, using 2020 data. The cost is a hospital cost, not a
  charge and not a payment.
- **Medicare cost reports** (Form CMS-2552-10) give facility-level cost-to-charge ratios and
  overhead allocation. They are public but lag by two years.
- **Hospital Price Transparency** rule at 45 CFR Part 180 requires hospitals to publish
  machine-readable standard charge files. These are charges, not costs, and the gap between them
  is large and variable. They are useful for relative comparisons and misleading for absolute
  value.
- **Transparency in Coverage** rules require insurers to publish negotiated rates and allowed
  amounts for covered items. These are the best public source for allowed amounts, subject to
  completeness problems.

Every monetary figure in a model should carry a tag: list, negotiated, allowed, paid, cost, or
charge. A table without the tag is not auditable.

### 8.3 Health technology assessment frameworks

Two frameworks are directly relevant to digital health value claims.

**The Institute for Clinical and Economic Review (ICER) and Peterson Health Technology Institute
(PHTI) framework for digital health technologies**, published in 2023, uses budget impact
analysis rather than long-term cost-effectiveness as its core economic method, and sets threshold
pricing benchmarks of 15 percent overall cost savings versus usual care or a 3 to 1 return on
investment, with cost neutrality as a fallback. It reports results as incremental cost per user,
per 1,000 users, and per user per month.

That 3 to 1 benchmark is a useful external calibration. A digital health product with a modelled
ROI below 3 to 1 in the framework's terms is not necessarily bad, but it will not clear the bar
that this framework sets for a positive value assessment.

**The National Institute for Health and Care Excellence (NICE) Evidence Standards Framework for
Digital Health Technologies** (updated 2022) tiers technologies by risk and function and
specifies the evidence expected at each tier. For technologies with limited clinical risk, the
framework accepts evidence on outcomes and cost-consequence rather than full cost-utility. It
also expects an economic analysis proportionate to the decision, which for many operational tools
means cost-consequence rather than cost per quality-adjusted life year (QALY).

### 8.4 Willingness-to-pay thresholds

The United States has no formal national cost-effectiveness threshold. ICER uses a range of
100,000 to 150,000 USD per QALY as the basis for its value-based price benchmarks, and describes
it as a benchmark rather than a policy threshold.

Published practice has shifted. An analysis of 3,276 United States cost-effectiveness analyses
from 1990 to 2021 found that references to 50,000 USD per QALY fell from 33.6 percent of studies
in 2000-2009 to 12.6 percent in 2020-2021, while references to 100,000 USD per QALY rose from 9.1
percent in the 1990s to 47.0 percent in 2020-2021, and 150,000 USD per QALY rose from zero to 23.6
percent (Neumann and Kim, discussed in the Tufts cost-effectiveness registry literature; see
Section 16).

The practical lesson: quote a threshold only with its source and year. A model that says "cost
effective at the standard 50,000 USD per QALY threshold" is using a benchmark that most current
analyses no longer use.

### 8.5 Regulatory constraints on claims

Value claims are commercial speech, but several regimes constrain them.

- **Food and Drug Administration (FDA) promotional rules.** For a device, claims about clinical
  benefit must be consistent with the cleared or approved intended use. An economic claim that
  implies a clinical outcome outside the indication is a promotional problem, not merely a
  marketing one.
- **Federal Trade Commission (FTC) substantiation.** Advertising claims in the United States must
  be substantiated by competent and reliable evidence. A modelled ROI presented as a result, rather
  than as an estimate, is a substantiation risk.
- **Securities and Exchange Commission (SEC) disclosure.** For public companies, value claims in
  investor materials carry disclosure obligations. This is why public filings describe pilot
  results precisely and marketing decks do not.

### 8.6 What the standards do not settle

None of these frameworks tells you whether a particular buyer will pay. Budget impact analysis
answers whether a payer's spend changes; it does not answer whether the payer's budget committee
will approve. The gap between a positive cost-effectiveness result and a signed contract is the
subject of Topics 04 and 05.

## 9. Industry practice and case evidence

### 9.1 Documented: ambient documentation time savings are real but modest

Ma and colleagues evaluated an ambient artificial intelligence (AI) scribe at a Stanford pilot
with 45 physicians across eight specialties between October 2023 and January 2024. The tool was
used in 55 percent of 17,428 encounters. Median time per note fell by 0.57 minutes, from 4.86 to
3.64 minutes; daily documentation time fell by 6.89 minutes; after-hours electronic health record
(EHR) time fell by 5.17 minutes per day; and total EHR time fell by 19.95 minutes per day. The
authors described the savings as modest with high heterogeneity across users (Journal of the
American Medical Informatics Association, 2024, doi:10.1093/jamia/ocae304).

That is the best public anchor for the effect size in the worked example. The 1.6 minutes used
there is above the 0.57-minute median note-time reduction and closer to the total EHR time effect
when expressed per encounter; the choice is stated as an assumption and the sensitivity shows
what happens at 0.5 minutes.

### 9.2 Documented: a large deployment with no productivity gain

Haberle and colleagues evaluated a different ambient documentation product at Intermountain
Health with 99 providers and reported positive provider-engagement trends but no productivity
gains, with a statistically significant worsening of after-hours EHR time (Journal of the
American Medical Informatics Association, 2024).

This result is the empirical basis for Diagnostic 1 in Section 6.6. It is not an outlier to be
discounted; it is evidence that realization is not automatic. A value model that cannot survive
realization being set to zero has not been tested against the published record.

### 9.3 Documented: adoption intensity determines the benefit

Apathy, Holmgren, and colleagues studied team-based documentation support, including scribes,
across 18,265 physicians. High-intensity adopters, defined as those with more than 40 percent of
note text produced by others, saved about 54 minutes per week of note time, a 21 percent
reduction, rising to 72 minutes per week, 28 percent, after a learning curve of about 20 weeks.
They also saw 2.5 to 4.5 additional visits per week. Low-intensity adopters saw no time savings
(JAMA Internal Medicine, 2024).

The finding is directly relevant to the leakage model: adoption is not binary and the benefit is
concentrated in a minority of intensive users. A model that applies an average effect to an
average adopter will overstate the total, because the distribution is skewed.

### 9.4 Documented: virtual scribes show smaller effects

Rotenstein and colleagues studied virtual scribes across 144 physicians and found a reduction in
total EHR time of 5.6 minutes per appointment, with note time down 1.3 minutes and after-hours
time down 1.1 minutes, with the largest benefit among medical specialists and physicians with high
baseline EHR time (JAMA Network Open, 2024).

The pattern across these four studies is consistent: effects exist, they are small relative to
the claims in vendor materials, and they are concentrated in high-baseline users and intensive
adopters.

### 9.5 Documented: the benchmark a digital health value assessment uses

The ICER-PHTI framework described in Section 8.3 sets 15 percent cost savings or a 3 to 1 ROI as
the threshold for a favourable value assessment of a digital health technology, with cost
neutrality acceptable in some categories. A vendor whose model shows a 0.4 three-year ROI, as in
the worked example, should expect a payer or health technology assessment body to classify the
product as cost-neutral at best, and should plan the evidence strategy accordingly.

### 9.6 Inference: what this implies for commercial practice

The following is inference, not a documented finding. Three practices follow from the evidence.

First, model the adopter distribution rather than the average. If 20 percent of physicians
produce 60 percent of the time savings, a licence priced per physician and valued per average
physician will look better than it is for the buyer, and the buyer will discover this at
renewal.

Second, price against the conservative end of the published effect range. A product priced so
that it pays back at 1.6 minutes per note but not at 0.8 will generate a renewal conversation
that the vendor loses.

Third, measure realization locally rather than asserting it. The one input that most changes the
answer is the least often measured, because measuring it requires time-motion work after
deployment, which most deployments do not budget.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Nominal-effect arithmetic | Model multiplies per-event saving by all events | Leakage multipliers omitted | Value overstated by 2 to 4 times | Apply adoption, use, realization, conversion explicitly |
| Double counting | Time saved and visits added in the same total | Same minutes valued twice | ROI inflated by 2 to 3 times | One rule per effect; label non-additive context |
| Soft saving presented as cash | "Avoided admissions" with no budget line | Avoided spend is not a reduction in spend | Renewal fails when the ledger does not move | Name the budget line and controller |
| Capacity claim without demand | Freed minutes valued at margin in a clinic with empty slots | Conversion assumes unmet demand | Benefit never materialises | Check scheduling data before modelling |
| Price-basis error | A rate quoted without year or basis | List, allowed, paid, and cost conflated | Error of 30 to 60 percent | Tag every monetary figure |
| Vendor effect size | Effect from a vendor case study used unadjusted | Selection and publication bias | Optimistic by 30 to 50 percent | Discount class D evidence explicitly |
| No ramp | Full benefit assumed in month one | Adoption takes time | Payback understated, cash flow wrong | Model a ramp curve |
| Omitted review cost | Cost stack has only licence and implementation | Clinical review is invisible in quotes | True cost 1.5 to 3 times licence | Add review and false-positive lines |
| Omitted security cost | No line for audit or architecture change | Procurement requirements discovered late | 8 to 16 week delay plus rework | Request the questionnaire early |
| Attribution ignored | Pre-post change credited entirely to product | Secular trend and concurrent initiatives | Overstatement of 30 to 50 percent | Use a comparison group or discount |
| One number, no range | Single point estimate in the deck | Uncertainty hidden | Binary argument, lost credibility | Present range and break-even |
| Denominator drift | Population quietly widened between drafts | Model tuned to hit a target | Reviewer finds it and discounts everything | Freeze the population at Step 2 |

Two red flags are diagnostic of a model that will not survive review. The first is an ROI above
5 to 1 for an operational tool; most such tools do not have that much value in them, and the
number usually indicates leakage multipliers were omitted. The second is a cost stack with fewer
than six lines; every real deployment has integration, training, support, and review costs, and a
stack without them is a quote, not a cost.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The effect size transfers to this population | Published evidence is the only pre-deployment source | Case mix or workflow differs | Run a local time-motion sample |
| 2 | Adoption reaches the modelled level | Benefit scales with use | Clinicians reject the tool or forget it | Pilot with product analytics |
| 3 | Saved time is partly redeployed | Conversion requires it | Work expands to fill the time | Observe scheduling after deployment |
| 4 | Demand exists for the freed capacity | Capacity value needs it | Waiting lists are short or rooms are full | Check scheduling and room utilisation |
| 5 | Contribution margin is stable | Value per visit depends on payer mix | Payer mix shifts or rates fall | Use finance's forward-looking margin |
| 6 | Costs are complete | The denominator must be honest | Integration or security costs appear later | Compare with a comparable deployment |
| 7 | The benefit persists past year one | Renewal depends on it | Novelty effect fades | Measure at months 3, 6, and 12 |
| 8 | The counterfactual is the pre-period | No control group is available | Secular trend or concurrent initiative | Use a matched comparison |
| 9 | The buyer's budget actually moves | Cash conversion requires it | Savings are absorbed into a global budget | Ask the controller which line changes |
| 10 | Price is stable for the term | Multi-year models need it | Step-ups or volume tiers apply | Read the contract, not the quote |

The deepest limit is that value quantification measures what is countable, and the things that
make clinicians adopt a tool, such as reduced cognitive load or a better end to the day, are
mostly uncountable. A model that ignores them will understate adoption and therefore understate
benefit. A model that monetises them will overstate benefit. The defensible position is to
quantify the countable, name the uncountable, and let the buyer decide how much weight to give
each.

## 12. Exercises

### Exercise 1. Three currencies

**Deliverable:** a one-page table separating clinical benefit, economic value, and willingness to
pay for a product you know, with the unit and the party for each.
**Criteria:** each row names a unit and a party; at least one row shows a case where the three
point in different directions; no row uses "value" without a qualifier. **Time:** 45 minutes.

### Exercise 2. Perspective and horizon

**Deliverable:** four short value statements for the same product, one for each perspective in the
table in Step 1, each with a stated horizon.
**Criteria:** the four statements differ in sign or magnitude, not just wording; costs and
benefits are assigned to the same party within each statement. **Time:** 60 minutes.

### Exercise 3. Absolute conversion

**Deliverable:** for a relative risk reduction of 22 percent and a base rate of 180 events per
1,000 patient-years in a population of 12,000, compute the events avoided per year and the number
needed to treat.
**Criteria:** correct arithmetic; the base rate and the population are used correctly; the answer
is expressed in events per year and in patients. **Time:** 30 minutes.

### Exercise 4. Conversion rule selection

**Deliverable:** for five described scenarios, select the conversion rule from Section 3.2 and
justify it in two sentences each.
**Criteria:** each justification names the budget line, the constrained resource, or the loss
bearer as appropriate; at least one scenario is correctly assigned no economic value. **Time:** 45
minutes.

### Exercise 5. Leakage model

**Deliverable:** a spreadsheet or script that applies four leakage multipliers to a nominal effect
and reports the realized value, with each multiplier's source labelled.
**Criteria:** multipliers are multiplicative and shown separately; the nominal and realized
values are both visible; one sensitivity table varies the two most uncertain multipliers.
**Time:** 2 hours.

### Exercise 6. Cost stack

**Deliverable:** a ten-line cost stack for a decision support deployment, including clinical
review and false-positive handling, with a one-time and recurring column.
**Criteria:** at least ten lines; review cost computed from alert volume and review time; every
line has a source or is marked as an estimate. **Time:** 90 minutes.

### Exercise 7. Audit an existing model

**Deliverable:** a written audit of a value model from a public vendor case study or a course
example, identifying leakage, double counting, and price-basis errors.
**Criteria:** at least three distinct errors found; each linked to a line in the chain; a corrected
estimate given with its assumptions. **Time:** 2 hours.

### Exercise 8. Write the claim

**Deliverable:** a 120-word value claim using the template in Step 12 for the case in Section 6,
including a range, a break-even, and an exclusions list.
**Criteria:** perspective and horizon stated; every number carries a basis; the range is justified
by a named sensitivity; no number appears without a unit. **Time:** 45 minutes.

## 13. Self-check answers

**1. A product reduces length of stay by 0.4 days. Is that clinical benefit, economic value, or
willingness to pay?**
It is none of them yet. It is a physical effect. It becomes clinical benefit if it is linked to an
outcome such as readmission or functional status, economic value if the freed bed-day is converted
to a capacity or cost figure for a named party, and willingness to pay only when a buyer commits a
budget. The distinction matters because each requires different evidence.

**2. Why must the perspective be stated before the numbers are computed?**
Because the same effect can be a benefit to one party and a cost to another. A prevented admission
reduces cost for a payer and reduces revenue for a fee-for-service provider. Computing first and
choosing a perspective afterwards guarantees an incoherent model.

**3. How does the unit of analysis create errors?**
If the effect is per note and the population is per patient, multiplying by patients instead of
notes understates or overstates by the ratio of notes to patients, which can be three or more.
The rule is that the unit must match the level at which the effect occurs.

**4. Compute events avoided and NNT.** Base rate 180 per 1,000 patient-years, relative reduction
22 percent, population 12,000. Absolute reduction = 180 x 0.22 = 39.6 events per 1,000
patient-years. Events avoided = 39.6 x 12 = 475.2 per year. NNT = 1 / 0.0396 = 25.3 patients per
year to prevent one event. Note that NNT is expressed per year here; without the time unit it is
meaningless.

**5. What are the four leakage multipliers and their typical ranges?**
Adoption, typically 0.3 to 0.7 in year one; use rate among adopters, 0.4 to 0.7; realization, 0.5
to 0.8; conversion to cash or capacity, 0.2 to 0.5. Multiplied, they reduce a nominal effect by a
factor of roughly 5 to 80. The product of the midpoints in the worked example is
0.5 x 0.55 x 0.65 x 0.35 = 0.063.

**6. Why is clinical review a cost line rather than a benefit?**
Because reviewing alerts consumes clinician time that is paid for and could be used elsewhere. It
is a real resource cost of the intervention, even though no invoice is issued. Models that omit it
understate cost by more than the licence in high-alert-volume products.

**7. How do you detect double counting?**
For each pair of benefit lines, ask whether both derive from the same underlying effect. If
removing one changes the other, they are the same effect and only one may be counted. A second
test: sum the benefit lines and compare with the physical units; if the physical units cannot
support the total, one line is duplicated.

**8. What does a 3 to 1 ROI benchmark mean, and why does it matter?**
It means three dollars of benefit per dollar of cost, which is the threshold the ICER-PHTI digital
health framework uses for a favourable value assessment, alongside a 15 percent cost-saving
alternative. It matters because a model showing 0.4 will be classified as cost-neutral at best by
an assessor using that framework, regardless of how the vendor describes it.

**9. Why is the break-even effect more persuasive than the ROI?**
Because it changes the question from "is your forecast right" to "is the effect above a
threshold," and the buyer can form a view about the threshold from their own experience. It also
makes the model's risk explicit, which builds credibility rather than eroding it.

**10. What is the difference between ROI and NPV?**
ROI is a ratio of net benefit to cost over a period and ignores timing. NPV discounts future cash
flows at a stated rate and gives a currency amount, which is what capital allocation requires. A
project can have a high ROI and a negative NPV if the benefit arrives late.

**11. Rubric for Exercise 6.** Full marks require ten or more lines, a computed review cost with
alert volume and minutes per alert shown, separate one-time and recurring columns, at least one
line for security and compliance, and every line sourced or marked as an estimate. Half marks if
the stack is complete but the review cost is asserted rather than computed.

**12. Rubric for Exercise 8.** Full marks require perspective, horizon, a currency range, a
break-even, at least three exclusions, a price basis with year, and an explicit statement of the
three most uncertain assumptions. Zero if any number lacks a unit or a basis.

## 14. Interview questions

**1. How do you build a value case?**
Strong: walks the chain from effect to adoption to use to realization to conversion, then subtracts
a full cost stack, and finishes with a break-even and a range. Weak: describes ROI as benefits
minus costs without saying how either is measured.

**2. What is the difference between cost avoidance and a hard saving?**
Strong: a hard saving reduces an actual expenditure visible in the ledger; cost avoidance prevents
a spend that was not budgeted, so the money is not freed for other uses. Weak: treats them as
synonyms, which is the most common way provider value cases overstate returns.

**3. How do you value clinician time?**
Strong: distinguishes loaded cost, opportunity cost, and capacity value, and says that time is
worth the loaded rate only if it is redeployed. Weak: multiplies minutes by salary and stops.

**4. Which conversion rule would you use for a sepsis alert?**
Strong: depends on whether the buyer bears the cost of the avoided event and whether the ICU bed
is constrained; if beds are full, capacity; if the payer bears the readmission cost, risk or no
value to the provider. Weak: picks cash without naming a budget line.

**5. Your model shows a 4 to 1 ROI. What do you check first?**
Strong: leakage multipliers and double counting, in that order, then the price basis. Weak: says
they would validate the effect size, which is rarely the largest error.

**6. How do you handle uncertainty?**
Strong: a range from named input ranges, a break-even, and a statement of which input to measure
next and how. Weak: a single sensitivity table with no interpretation.

**7. What is the right time horizon?**
Strong: long enough to capture the cost of the intervention and the benefit, short enough that the
benefit is measurable, and consistent with the buyer's budget cycle and contract term. Weak: "five
years" as a default.

**8. How would you respond to a buyer who says your ROI is too high?**
Strong: agrees, identifies the omitted costs, and presents a revised, lower figure with the
concession visible, which builds trust. Weak: defends the number.

**9. What is the difference between list price and allowed amount?**
Strong: defines both and gives an example where the gap changes a business case, such as a
screening service reimbursed below list. Weak: uses them interchangeably.

**10. How do you write a value claim for an executive?**
Strong: one paragraph, perspective, horizon, range, break-even, basis, exclusions. Weak: a slide
with a big number and no basis.

**11. What evidence would make you abandon a value case?**
Strong: names a threshold on the break-even effect or on adoption, and says they would abandon if
the local pilot fell below it. Weak: says they would abandon if the numbers did not work, without
a threshold.

**12. Why do digital health value assessments often use budget impact rather than cost per
QALY?**
Strong: because many digital tools affect cost and process rather than health utility directly,
QALYs capture little of the effect, and payers decide on affordability within a budget period.
Weak: says budget impact is easier, which is true but not the reason.

## 15. Key terms

- **Absolute risk reduction** — difference in event rates between groups.
- **Adoption rate** — share of eligible users who use a tool at all.
- **Allowed amount** — maximum a payer recognises for a service.
- **Attribution** — share of an observed change assigned to the intervention.
- **Break-even effect** — effect size at which net value is zero.
- **Capacity conversion** — turning freed resource into additional units at margin.
- **Cash conversion** — an effect that moves a budget line.
- **Clinical benefit** — change in health outcome.
- **Contribution margin** — revenue minus variable cost per unit.
- **Conversion rule** — the mechanism by which an effect becomes money.
- **Cost avoidance** — prevented spend that was not budgeted.
- **Counterfactual** — what would have happened without the intervention.
- **Discount rate** — rate used to convert future value to present value.
- **Double counting** — valuing one effect through two rules.
- **Economic value** — monetised consequence to a defined party.
- **Effect size** — magnitude of change per eligible unit.
- **Fully loaded cost** — cost including benefits, overhead, and support.
- **Hard saving** — reduction in actual spend visible in the ledger.
- **Internal rate of return** — discount rate at which NPV is zero.
- **Leakage** — loss between nominal and realized effect.
- **List price** — published price before negotiation.
- **Negotiated price** — contracted price after discount.
- **Net present value** — sum of discounted future net cash flows.
- **Number needed to treat** — patients treated to prevent one event.
- **Option value** — value of a capability available in future.
- **Paid amount** — amount actually transferred.
- **Payback period** — time until cumulative net benefit covers investment.
- **Perspective** — party whose costs and benefits are counted.
- **Realization** — share of nominal effect that actually occurs.
- **Regression to the mean** — extreme values moving toward average without intervention.
- **Relative risk reduction** — proportional reduction in event rate.
- **Return on investment** — net benefit divided by cost.
- **Risk conversion** — expected value of a reduced loss.
- **Ramp** — period before steady-state benefit.
- **Soft saving** — avoided future or notional cost.
- **Time horizon** — period over which costs and benefits are counted.
- **Total cost of ownership** — all costs over the life of the asset.
- **Use rate** — share of eligible events where an adopter uses the tool.
- **Willingness to pay** — maximum a buyer will commit from a budget.

## 16. Sources

All URLs accessed 26 October 2026.

1. Ma S, et al. Ambient artificial intelligence scribes: utilization and impact on documentation
   time. *Journal of the American Medical Informatics Association*. 2024.
   https://doi.org/10.1093/jamia/ocae304
2. Haberle T, et al. Use of an ambient artificial intelligence tool to improve quality of clinical
   documentation. *Journal of the American Medical Informatics Association*. 2024.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11452835/
3. Apathy NC, Holmgren AJ, et al. Physician EHR time and visit volume following adoption of
   team-based documentation support. *JAMA Internal Medicine*. 2024.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11348094/
4. Rotenstein LS, et al. Virtual scribes and physician time spent on electronic health records.
   *JAMA Network Open*. 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11127114/
5. Pearson SD, et al. ICER-Peterson Health Technology Institute value assessment framework for
   digital health technologies. *Journal of Comparative Effectiveness Research*. 2023.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC10734316/
6. Institute for Clinical and Economic Review and Peterson Health Technology Institute.
   *ICER-PHTI Assessment Framework for Digital Health Technologies*, version 1.0, 2023.
   https://phti.org/wp-content/uploads/sites/3/2023/09/ICER-PHTI-Assessment-Framework-for-Digital-Health-Technologies-1.0.pdf
7. Gomes M, Murray E, Raftery J. Economic evaluation of digital health interventions:
   methodological issues and recommendations for practice. *PharmacoEconomics*. 2022.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC8821841/
8. Benedetto V, et al. Analytical frameworks and outcome measures in economic evaluations of
   digital health interventions: a methodological systematic review. *Medical Decision Making*.
   2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC9742632/
9. National Institute for Health and Care Excellence. *Evidence Standards Framework for Digital
   Health Technologies*. Updated 2022.
   https://www.nice.org.uk/about/nice-our-programmes/evidence-standards-framework-for-digital-health-technologies
10. Agency for Healthcare Research and Quality. *Characteristics of 30-Day All-Cause Hospital
    Readmissions, 2016-2020*. HCUP Statistical Brief #304.
    https://hcup-us.ahrq.gov/reports/statbriefs/sb304-readmissions-2016-2020.jsp
11. Centers for Medicare and Medicaid Services. *Hospital Price Transparency*, 45 CFR Part 180.
    https://www.cms.gov/priorities/key-initiatives/hospital-price-transparency
12. Centers for Medicare and Medicaid Services. *Transparency in Coverage*, 45 CFR Part 147.
    https://www.cms.gov/priorities/key-initiatives/health-plan-price-transparency
13. Neumann PJ, Kim DD. Cost-effectiveness thresholds used by study authors, 1990-2021.
    *JAMA*. 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10114019/
14. Tang PC, Wilson E, Mehrotra A. Paying for AI in healthcare: setting the right precedent
    amidst growing use. *Health Affairs Forefront*. April 2024.
    https://www.healthaffairs.org/content/forefront/paying-artificial-intelligence-health-care-setting-right-precedent-amidst-growing-use
15. Federal Trade Commission. *Advertising and Marketing Basics: Substantiation*.
    https://www.ftc.gov/business-guidance/resources/advertising-marketing-basics

Figures in Section 6 are illustrative estimates constructed for teaching. No company outcome or
reimbursement rate is invented. Where a range or benchmark is cited, the year and basis are given
in place.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`01_product-discovery.md`](01_product-discovery.md)
- Next topic: [`03_health-economics.md`](03_health-economics.md)
- Related topic: [`07_business-models.md`](07_business-models.md)
- Module survey: [`../../course/17_product_economics_leadership.md`](../../course/17_product_economics_leadership.md)
- Labs: [`../../labs/`](../../labs/)
