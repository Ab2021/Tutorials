# Demand Sensing versus Forecasting

> Module 09 — Demand Sensing, Promotions, and New Product Forecasting
> Topic 01 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 05; Module 07 topics 01 and 08; Module 08 topics 01 and 05
> Version 2026.5

---

## 1. Why this topic exists

Every planning organisation of any size now has a slide that says "demand sensing". The slide
usually shows a funnel: historical data at the bottom, real-time signals at the top, and a number
in the middle described as "the short-term forecast". Somebody in the room has been asked to build
it. That person is usually a data scientist, and the question they are actually being asked is not
"can you ingest point-of-sale data" but "can you make the plan better between now and Thursday".

Those are different questions, and the gap between them is where most sensing programmes lose
their budget. A company can double the frequency of its forecast and improve nothing, because the
order was frozen before the new forecast existed. It can add six external signals and improve
nothing, because the signals arrive after the decision they were meant to inform. It can improve
measured forecast accuracy by eight percent and reduce inventory by less than one percent, because
the inventory parameter is refreshed once a quarter and the improvement never reaches the policy.

The confusion is not technical. It is definitional. "Demand sensing" is used in vendor material,
internal programme charters, and academic papers to mean at least four different things:

1. A **daily re-forecast** — the same model, run more often. This changes cadence, not information.
2. A **point-of-sale driven update** — a new dependent variable, so the forecast is of consumer
   offtake rather than of the order the retailer placed.
3. A **downstream-inventory-adjusted demand signal** — a reconstruction of true consumer demand
   from orders net of the change in the retailer's stock. This is a state-estimation problem, not
   a forecasting one.
4. A **machine-learning nowcast** — a different function class, using many covariates at a short
   horizon, often including weather, search interest, and calendar structure.

Each of these has a different evidence base, a different failure mode, and a different value
proposition. A business case that mixes them is not comparable to any published study, and a
vendor claim that mixes them cannot be tested. This topic separates them and keeps them apart.

Three failures recur in practice, and they motivate the rest of the file.

**Failure one: a signal is ingested because it exists.** Retailer inventory feeds, weather APIs,
search indices, and logistics events are cheap to acquire and easy to display. None is free to
steward, and none is worth anything unless it arrives early enough to change a decision. The test
is arithmetic — arrival day against decision gate — and it is rarely performed. Section 3 draws
it; section 7 computes it.

**Failure two: accuracy is reported at the wrong horizon.** Sensing improves short-horizon accuracy
far more than long-horizon accuracy. If the decision consumes a four-week-ahead forecast, the
one-week-ahead improvement is a number about a different decision. In section 6 the one-week gain
is 8.2% and the four-week gain is 2.2%; only the second is real for that business.

**Failure three: a sensing improvement is not a policy improvement.** A better forecast changes
nothing unless something downstream consumes it: an order quantity, a safety-stock parameter, an
allocation rule, a service target. That distinction separates a dashboard from an intervention.
Section 5 step 14 formalises it; section 6 prices it.

### 1.1 What this topic is, and what it is not

This topic is the **signal-and-lead-time layer**. It sits above the forecasting methods and below
the business decision. It does not teach you how to build a forecasting model — Module 07 does
that, and its topic 01 (forecastability) and topic 08 (rolling-origin backtesting) are
prerequisites here. It
does not teach you how to reconcile forecasts across a product hierarchy — Module 08 does that,
and its topics 01 and 05 are prerequisites here, because the sensing output has to live inside a
hierarchy whether you designed one or not. It does not teach you how to run a pilot — Module 05
does that, and this topic assumes you can name a randomised rollout, a stepped wedge, a
difference-in-differences design, and a matched comparison without having them re-derived.

What it does is narrower and harder: given a list of candidate signals and a set of decisions,
decide which signals to ingest, how to construct the short-horizon update, how to measure whether
it helped, and how to notice when it stopped helping.

> **Note:** The commercial history of the term "demand sensing" is poorly documented in the public
> record. Vendor origin stories should be treated as marketing. The underlying idea — separating a
> long-cycle plan from a short-cycle correction, and feeding the correction with downstream data —
> is older than the label and appears in the forecasting literature under "short-horizon
> forecasting", "nowcasting", and "demand signal processing". Nothing in this topic depends on who
> coined it.

Each of those is a paragraph in a textbook and a quarter of work in a company.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** the four constructions that travel under the name demand sensing — daily
   re-forecast, point-of-sale (POS) driven update, downstream-inventory-adjusted demand, and
   machine-learning nowcast — and **state** which one a given programme is actually building.
2. **Apply** the lead-time test to a candidate signal: compute its usable arrival day, compare it
   with the decision gate and the order freeze, and **decide** whether it can change the order or
   only the execution.
3. **Classify** a signal inventory into leading, lagging, contemporaneous, and not-a-signal, with
   a stated reason for each classification, and **defend** the "not a signal" cases.
4. **Reconstruct** consumer offtake from order and channel-inventory data using the accounting
   identity, and **state** the three conditions under which the reconstruction fails.
5. **Design** a short-horizon update that consumes a small number of tested signals, and
   **specify** what it is and is not allowed to change in the plan.
6. **Evaluate** a sensing intervention at the decision horizon and on the decision metric, using
   the designs taught in Module 05, and **separate** a sensing effect from a policy effect.
7. **Diagnose** signal death — a feed that still arrives but no longer carries information — and
   **specify** the monitors that would have caught it.
8. **Critique** a vendor or internal sensing business case for horizon mismatch, definitional
   drift, and unrealised value.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold two pictures. The first is a timeline: signals arrive at different times, and the only thing
that makes one of them matter is a decision gate it can still reach. The second is a chain of
identities: the order you receive upstream is consumer demand plus or minus a change in somebody's
inventory, and every distortion the bullwhip literature describes is a violation of that identity
in one direction or another.

### 3.1 The signal pipeline in time

```text
 Working-day axis. Day 0 = the order gate, the moment the order is committed upstream.

      -28      -21      -14       -7        0        +7      +21
       |        |        |        |        |         |        |
 PLAN  |........|........|........|........|         |        |
 LOCK  |<============ order freeze (F days) ========>|        |
       |        |        |        |        |         |        |
 gate  |        |        |        |        |#########|        |
       |        |        |        |        | order   |        |
       |        |        |        |        | fixed   |        |
       |        |        |        |        |         |        |
 SIGNAL AVAILABILITY (day the signal becomes usable, relative to the gate)
       |        |        |        |        |         |        |
 weather forecast  ....>                                      -14   BEFORE
 calendar / holiday  ..>                                    -365   BEFORE
 search interest  ..............>                              -6   BEFORE
 POS scan  .........................>                          -2   BEFORE (inside freeze)
 retailer stock     ....................>                      +1   AFTER
 open orders (real time)  ..................>                  +0   AFTER
 own shipment actuals  ........................>               +5   AFTER
       |        |        |        |        |         |        |
       +--------+--------+--------+--------+---------+--------+
           ^                                    ^
    still inside the                    cannot change this order;
    flexible window: can                can change only execution
    change the order                    (allocate, expedite, rebalance)
```

Read the diagram from the gate backwards. The gate is a fixed moment: a planner submits a number,
a system transmits it, a supplier commits capacity against it. Before that moment there is a
freeze — a period, usually measured in days, during which changes to the committed plan are
blocked because material has been bought or capacity has been booked. A signal that arrives inside
the freeze but before the gate can still be *seen*; it cannot be *acted on* for this order. A
signal that arrives after the gate is not late by a little; it is late by a whole cycle.

The diagram is deliberately drawn with a wide freeze relative to the gate. In many organisations
the freeze is longer than the interval between gates, which means the effective decision lead
time is not the gate cadence at all. A company that reviews weekly but freezes four weeks is
making a monthly decision every week.

### 3.2 The four constructions, side by side

| Construction | What changes | Where the new information is | Typical evidence needed |
|---|---|---|---|
| Daily re-forecast | Cadence | None, unless the input data is fresher | That the extra runs do not add variance or churn |
| POS-driven update | Dependent variable | Consumer offtake replaces the retailer's order | That offtake predicts the decision better than orders do |
| Downstream-inventory-adjusted demand | State estimate | The retailer's stock position | That the inventory feed is complete and timely |
| Machine-learning nowcast | Function class and covariates | Many signals, short horizon | That the covariates add information out of sample |

These are not stages of maturity. They are four different products, and a programme that claims
all four is usually claiming that it has a model, a data feed, an inventory integration, and a
neural network, which describes a lot of software and no particular decision.

### 3.3 The demand identity, and why upstream data is distorted

```text
  CONSUMER          RETAILER           WHOLESALER         MANUFACTURER
  offtake   --->    orders     --->    orders     --->    orders
  CV = 0.12          CV = 0.31          CV = 0.44          CV = 0.52
                      ^                  ^                  ^
                      |                  |                  |
               + batching           + rationing         + frozen horizon
               + promotion          + lead-time          + forecast error
                 pull-forward         hedging              compounding

  The identity that connects them, at every link and every period t:

      orders_t  =  offtake_t  +  [ channel_stock_t - channel_stock_(t-1) ]  +  leakage_t

  De-layering recovers offtake from the left-hand side:

      offtake_t  =  orders_t  -  delta(channel_stock)_t  -  leakage_t
```

The identity is an accounting fact, not a model. It holds for any node in any chain, provided
"offtake" means what left the node to the next one down, "orders" means what the node was asked
for, and "stock" means everything the node holds that was counted in the order decision. That
last clause is where the argument lives: stock in transit, stock on consignment, stock at a
regional depot that the feed omits, and stock that exists physically but not in the system
("phantom inventory") all break the identity silently.

Read the coefficient of variation row in the diagram as illustrative, not as a universal law. The
direction — variability rising upstream — is one of the most replicated findings in operations
management. The magnitudes are specific to a product, a channel, and a measurement bucket, and
quoting them as benchmarks is a mistake this course will keep warning you about.

### 3.4 The three tests every signal must pass

A signal is worth ingesting only if it passes three tests, in this order. Failing any one of them
ends the evaluation, which is why the order matters: the first test is arithmetic and cheap, and
the third is expensive.

1. **Arrival test.** Does the signal become usable before the decision gate, with enough margin to
   survive the planning cycle? Computed from timestamps, not from vendor promises.
2. **Information test.** Does the signal explain forecast error that the incumbent forecast does
   not already explain? Computed out of sample, against a placebo.
3. **Decision test.** Does acting on the signal change the decision in a way that improves the
   decision's own loss function? Computed against the policy, not against the forecast.

The three tests correspond to the three failures in section 1. Most sensing programmes run test
two, skip test one, and never run test three.

### 3.5 Where the value actually sits

| Situation | Forecast horizon that matters | Where sensing helps | Where it cannot help |
|---|---|---|---|
| Make-to-stock consumer goods, 4-week frozen horizon | 3–5 weeks | Retailer stock and offtake, if the freeze can be shortened | Anything inside the freeze |
| Make-to-order (MTO) industrial, quoted lead time 6 weeks | 6–8 weeks | Order book conversion, pipeline of quotes | Consumer-level signals, which do not exist |
| E-commerce, replenished daily | 1–7 days | Search, weather, calendar, basket data | Long-horizon signals, which arrive too early |
| Fashion retail, 12-week buy cycle | 8–12 weeks | Early sell-through at the first stores | The buy decision itself, already committed |
| Pharma, tender-driven | 4–12 weeks | Tender calendars, wholesaler stock | Demand sensing in the consumer sense |

The pattern is that sensing value is concentrated where three things hold at once: the decision
cadence is short, the freeze is short relative to the cadence, and a downstream signal exists that
the firm does not already own. Remove any one and the value collapses. This is why the same
technology produces a compelling case study in one company and a failed pilot in another, and why
neither result generalises.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Demand signal | Any observable series used as evidence about future demand | Treated as a synonym for demand | Units per period, or an index |
| Demand sensing | A short-horizon update to a plan that consumes current signals | Used to mean any daily forecast run | Named construction plus horizon |
| Nowcast | An estimate of a quantity for the current or immediately next period | Used loosely for any short forecast | Units per period |
| Forecast horizon | The distance from the forecast origin to the period being forecast | Reported without the origin | Weeks or days ahead |
| Information lead time | Elapsed time from the event to its availability to the decision-maker | Confused with transport lead time | Days |
| Decision gate | The fixed moment at which a commitment is made and transmitted | Undocumented in most organisations | Date and time, with time zone |
| Order freeze | The period before the gate during which changes to the commitment are blocked | Confused with the gate itself | Days before the gate |
| Planning time fence | The system parameter that enforces the freeze | Assumed to equal the contractual freeze | Days |
| Signal latency | Event-to-usable time, including extraction, transport, load, and refresh | Quoted as the vendor's feed frequency | Hours or days |
| Freshness margin | Gate minus freeze minus signal latency; the spare days available | Rarely computed | Days; negative means unusable |
| Actionability envelope | The set of decisions a signal can still change given the freeze | Assumed to be "the forecast" | List of decisions |
| Point of sale (POS) | The record created when a consumer buys at a checkout | Used to mean any retail data feed | Transaction count or units |
| Scanner data | POS data aggregated by a market-research provider | Treated as complete market coverage | Units per store per week |
| Sell-in | Units sold by a manufacturer or distributor to a retailer | Used as demand | Units per period |
| Sell-out | Units sold by a retailer to consumers | Confused with sell-in during promotions | Units per period |
| Offtake | Synonym for sell-out, used for consumption from a channel | Used without stating the node | Units per period |
| Ship-to demand | Demand expressed as shipments to a named destination | Treated as consumer demand | Units per destination |
| Downstream inventory | Stock held by a customer or channel partner | Omitted from the demand identity | Units at a date |
| Channel inventory | Total stock in the distribution channel between the firm and the consumer | Confused with own finished goods | Units at a date |
| De-layering | Recovering offtake from order data by netting out channel-stock change | Confused with removing a distribution tier | Units per period |
| Bullwhip effect | Upstream amplification of downstream demand variability | Used to mean any variability | Variance ratio, dimensionless |
| Variance ratio | Variance of orders divided by variance of the demand they serve | Computed across different buckets | Dimensionless |
| Open order | A commitment to deliver that has not yet been fulfilled | Treated as demand intent | Units plus requested date |
| Order book | The stock of open orders at a point in time | Used as a demand forecast | Units at a date |
| Backorder | Demand that could not be served and remains owed | Confused with an open order | Units |
| Allocation | The rule by which constrained supply is distributed among claimants | Treated as neutral | Rule identifier |
| Rationing | Allocation under shortage, which invites strategic ordering | Confused with allocation in general | Rule plus shortage state |
| Shortage gaming | Ordering more than needed to protect allocation | Invisible in the order series | Order-to-need ratio |
| Stockout-censored demand | Demand that existed but produced no sale because stock was unavailable | Entered as a zero | Boolean availability flag |
| Artificial zero | A zero caused by unavailability rather than absence of demand | Counted as demand evidence | Boolean flag |
| Order batching | Consolidating needs into periodic or lot-sized orders | Read as a demand pattern | Order day-of-week concentration |
| Pull-forward | Demand moved earlier in time by a promotion or price change | Booked as incremental demand | Units displaced across periods |
| Baseline | The counterfactual level absent the intervention | Confused with the plan | Units per period |
| Weather sensitivity | The share of demand variance attributable to weather variables | Assumed to be uniform across a category | Elasticity or variance share |
| Search interest | An index of query volume for terms related to a product | Treated as purchase intent | Index, 0–100 |
| Placebo signal test | Re-running the model with the signal shuffled or shifted in time | Skipped in almost every business case | Accuracy difference |
| Signal decay | Loss of incremental information in a signal over time | Discovered only when the forecast degrades | Rolling information gain |
| Signal registry | The catalogue of signals with owner, grain, latency, and contract | Maintained as a spreadsheet nobody owns | Record per signal |
| Forecast value added (FVA) | Accuracy improvement of a forecast step over a naive benchmark | Reported without the benchmark | Error difference |
| Weighted absolute percentage error (WAPE) | Sum of absolute errors divided by sum of actuals | Compared across different aggregations | Proportion, 0–1 |
| Decision delta | The change in a decision caused by a forecast change | Assumed equal to the forecast change | Units, currency, or service |
| Policy improvement | A change to a decision rule, not to a forecast | Conflated with a sensing gain | Rule version |
| Collaborative Planning, Forecasting and Replenishment (CPFR) | A voluntary inter-firm process for joint planning | Treated as a data standard | Process guideline |
| Electronic data interchange (EDI) | Structured business-document exchange between firms | Used to mean any integration | Transaction set identifier |
| New product introduction (NPI) | The launch phase of an item | Forecast with mature-item signals | Launch date |

Six of these carry most of the argument, and each deserves a paragraph.

**Demand sensing versus daily re-forecast.** Running the same model every morning instead of
every Monday produces a series of forecasts that differ because the model's inputs changed by one
day of data. That is a cadence change. It can be useful — it shortens the lag between a signal and
a decision — but it introduces no new information and it is not what the published evidence about
POS data is measuring. When a vendor says "daily demand sensing", ask which model, with which
inputs, and whether any input is new.

**Sell-in versus sell-out.** Sell-in is what the retailer bought; sell-out is what the consumer
bought. In a steady state they differ by the change in the retailer's stock, and over a year they
are nearly equal. In any shorter window, and especially around a promotion, they diverge sharply.
A firm that forecasts sell-in while calling it demand will over-produce into a promotion and
under-produce after it, and the error will look like a forecast failure when it is a definition
failure. Everything in section 5 about de-layering exists to close this gap.

**Downstream inventory.** This is the term that makes the demand identity usable. It is not the
firm's own stock, not stock in transit under the firm's ownership, and not stock the retailer has
already sold. It is the physical units sitting at the customer's locations at a date, which the
customer may report weekly, may report only for some locations, and may define differently from
the firm. The measurement differences are the reason de-layering fails more often than the
arithmetic does.

**Freeze versus gate.** The gate is when you commit. The freeze is how far ahead of the gate you
must have stopped changing things. These are usually owned by different functions — the gate by
planning, the freeze by procurement or manufacturing — and neither is typically written down in
the analytics documentation. A sensing programme that does not know both numbers cannot compute
its own freshness margin, and therefore cannot know whether it is useful.

**Decision delta.** This is the concept that separates a sensing project from a reporting project.
It is the change in the decision that results from the forecast change, in the decision's units.
If the order quantity is rounded to full pallets of 600 cases and the forecast revision is 46
cases, the decision delta is usually zero. A programme that reports the forecast revision and not
the decision delta is reporting an input, not an outcome.

**Policy improvement.** A policy improvement changes the rule — the service target, the safety
factor, the review period, the freeze length, the rounding increment. It usually requires no new
data and often produces a larger effect than any sensing programme. The two are complements, and
the reason to keep them separate is that a sensing business case that quietly includes a policy
change will attribute the policy's value to the data, and will then fail to reproduce it anywhere
else.

## 5. The mechanics, step by step

This section runs the three tests of section 3.4 as a procedure, then builds the update, then
measures it, then governs it. The numbering matters because the cheap tests come first and a
programme that starts at step 9 has usually already spent its budget.

### Phase A — Establish the decision

**Step 1. Name the decision, its owner, its cadence, and its gate.**

Write four fields:

- **Decision.** "How many cases of this item-location to order from the plant" is a decision.
  "Improve forecast accuracy" is not.
- **Owner.** A role, not a department. The demand planner for the category, the replenishment
  manager for the distribution centre (DC).
- **Cadence.** Weekly, daily, twice weekly. This sets the maximum number of opportunities per
  period.
- **Gate.** The exact moment of commitment, with time zone. Thursday 17:00 Central European Time,
  not "Thursday".

If the gate cannot be established, the programme cannot pass step 5 and should stop. This is not
a bureaucratic point: an undocumented gate is the most common reason a technically sound sensing
model produces no measured benefit.

**Step 2. Establish the freeze and the flexible fraction.**

The freeze is how many days before the gate changes stop being accepted. The flexible fraction is
how much of the volume can still move after the freeze. Both are usually discoverable from the
planning system's time-fence parameter and from the procurement contracts, and both are usually
different from what the planning team believes.

A useful check: ask three people — a planner, a buyer, and a plant scheduler — for the freeze
length. If the answers differ, the freeze is not a single number and the sensing horizon has to be
computed per lane.

**Step 3. Write the actionability envelope.**

List what a signal arriving at each point in time could still change:

| Arrival relative to gate | What it can change |
|---|---|
| More than one cycle early | The forecast, the plan, the order, the safety-stock parameter |
| Inside the freeze, before the gate | The order at the margin, expedite decisions, allocation, mix |
| At the gate | Nothing for this order; possibly the next |
| After the gate | Execution only: allocate, expedite, rebalance, substitute |
| After delivery | Nothing; this is measurement, not sensing |

The envelope is the honest scope of the programme. A sensing layer that can only change execution
should be measured on execution outcomes — expedite cost, allocation fairness, service recovery —
not on forecast accuracy.

### Phase B — Test the signals

**Step 4. Measure latency from timestamps, not from documentation.**

For each candidate signal, measure the distribution of event-to-usable lag over at least eight
weeks. Report the median, the 90th percentile, and the share of days with no data. The 90th
percentile matters more than the median, because the weeks when the feed is late are
disproportionately the weeks when something unusual is happening.

Four components produce the lag, and they have different owners:

1. **Event to record** — the scan at the till, the shipment at the dock, the order in the
   retailer's system.
2. **Record to transmission** — the retailer's batch job, which may be nightly, weekly, or
   triggered.
3. **Transmission to availability** — the EDI or API transfer, which can fail silently.
4. **Availability to usability** — extraction, validation, transformation, and the refresh of the
   table the planner actually reads.

Most programmes measure component 3 and call it latency. Component 4 is frequently the largest and
is entirely inside the firm's control.

**Step 5. Apply the arrival test.**

Compute `freshness margin = (gate day − freeze days) − usable day`, all relative to a common
origin. A negative margin means the signal is unusable for this decision. Section 7 works the
arithmetic for a representative set of signals.

Two subtleties:

- The gate is periodic but the signal is not. A daily POS feed will beat a weekly gate on some days
  and miss it on others. Compute the margin for every gate in a quarter and report the share of
  gates the signal makes, not a single number.
- The freeze may differ by item class. A long-lead component may be frozen eight weeks out while a
  packaging variant is frozen two weeks out. The sensing horizon is then a property of the item,
  not of the business.

**Step 6. Apply the information test against the incumbent, not against nothing.**

The comparison that matters is not "signal versus no signal" but "signal versus the forecast the
company already has". That forecast already contains history, seasonality, promotion calendars,
and the planner's judgement. A signal that correlates with demand but adds nothing beyond the
incumbent is decoration.

The test has three parts:

- **Incremental error reduction.** Fit the incumbent model with and without the signal, on the
  same rolling origin, at the decision horizon. Report the difference in WAPE or in the loss
  function the decision actually uses. Do not report R².
- **Placebo.** Repeat with the signal shuffled within a window, or shifted by a season, and
  confirm that the accuracy gain disappears. If it does not disappear, the "gain" is coming from
  the model class or the extra parameters, not the signal.
- **Stability.** Repeat across at least three non-overlapping evaluation windows and report
  whether the sign of the effect is consistent. A signal that helps in one quarter and hurts in
  the next is not yet a signal.

**Step 7. Establish direction and confound structure.**

A correlation between an external series and demand is not evidence that the series causes
demand, nor that it will keep predicting it. Three confounds recur:

- **Common cause.** Weather and ice-cream sales both follow the season. The weather series adds
  information only where it departs from the seasonal expectation — which is to say, in the
  anomaly, not the level.
- **Reverse causation.** Search interest and sales can both respond to a television campaign. The
  search index is then a measure of the campaign, not of demand, and it will mislead when the
  campaign changes.
- **Measurement coupling.** A retailer's stock feed and its orders are generated by the same
  replenishment system. Correlating them with each other measures the system, not the demand.

**Step 8. Cost the signal.**

Every signal carries a cost that is not on the vendor invoice: contract negotiation, data
engineering, validation, stewardship, and the analyst time to investigate its anomalies. A signal
that improves WAPE by 1% and costs a tenth of a full-time analyst is a bad trade at most volumes.
Put the cost next to the decision delta in the same units.

### Phase C — Build the short-horizon update

**Step 9. Choose the construction explicitly.**

Decide which of the four constructions in section 3.2 you are building, and write it into the
specification. Most confusion in this field is a programme that has built a daily re-forecast and
described it as inventory-adjusted demand sensing.

A defensible default for a consumer-goods firm with retailer data is: a POS-and-channel-stock
update of the offtake forecast, at item-location-week, used to revise the order at the gate and
the safety-stock parameter at its refresh. That is construction 2 plus 3, and it is testable.

**Step 10. De-layer before you forecast.**

Apply the identity from section 3.3. Three conditions must hold for the reconstruction to be
trustworthy:

1. **Completeness.** The channel-stock feed covers every location whose orders are in the order
   series. If it covers 80% of stores, the residual is a mixture of demand and the uncovered 20%.
2. **Consistency of timing.** The stock snapshot and the order series must be cut at the same
   moment. A stock snapshot taken Sunday midnight against orders transmitted Friday contains two
   days of unrecorded movement.
3. **Consistency of definition.** "Stock" must include what the retailer includes when it decides
   to order. Excluding stock in transit to the store understates the position and produces
   spurious restocking signals.

When the conditions fail, the honest move is to forecast the order series directly and to say so,
rather than to construct a offtake series whose error structure nobody understands.

**Step 11. Keep the sensing forecast subordinate to the plan.**

The sensing output is a revision, not a replacement. Three rules make the relationship work:

- The sensing layer may change the short-horizon numbers within a stated band; outside the band it
  raises an exception rather than overwriting.
- The sensing layer may not change the structure of the plan — the product hierarchy, the service
  targets, the frozen commitments.
- Every revision is stamped with the signal set, the model version, and the time it was produced,
  so that a later review can reconstruct why the number moved.

**Step 12. Fix the grain and the horizon bands.**

State the output grain (item-location-day, item-location-week, category-week) and the horizon
bands (0–7 days, 8–28 days, 29–90 days) and which decisions consume which band. A sensing output
with no stated consumer is a dataset, not a forecast.

### Phase D — Measure

**Step 13. Evaluate at the decision horizon on the decision metric.**

The decision horizon is the lead time the gate implies, not the shortest horizon the model
produces. If the gate orders for delivery in four weeks, the relevant accuracy is four-week-ahead
accuracy. Report the full horizon profile, and headline the decision horizon.

The decision metric is the one the decision owner is measured on: cases of inventory, cost of
expedite, service level, waste. Forecast accuracy is an intermediate variable and is worth
reporting only alongside the metric it is supposed to move.

**Step 14. Separate sensing from policy.**

Run the decision-delta audit:

1. Take the sensing forecast and the baseline forecast for the same item-weeks.
2. Apply both to the *same* decision rule and compute the two decision sequences.
3. Count the item-weeks where the decision differs, and by how much.
4. Apply the *improved* rule with the baseline forecast, and compare with the improved rule and the
   sensing forecast.

The four cells tell you where the value is:

| | Baseline forecast | Sensing forecast |
|---|---|---|
| **Baseline policy** | Status quo | Pure sensing effect |
| **Improved policy** | Pure policy effect | Combined |

If the pure sensing effect is near zero and the pure policy effect is large, the programme is a
policy programme with a data component. Report it as such. It may still be worth doing; it is a
different investment.

**Step 15. Design the evaluation as a causal design, not a backtest.**

A rolling-origin backtest answers "would this model have predicted better". It does not answer
"did the organisation make better decisions", because the organisation's behaviour is part of the
treatment. Use the designs taught in Module 05: a randomised rollout across comparable
item-locations, a stepped wedge across regions, a difference-in-differences against a matched
control, or a regression discontinuity at an eligibility threshold. The Module 05 material covers
the assumptions and the diagnostics; do not re-derive them here, and do not accept a before-and-
after comparison as a substitute.

> **Note:** The most common evaluation defect in sensing programmes is that the pilot period
> coincides with a promotion, a price change, or a competitor exit. A before-and-after readout then
> attributes the promotion's effect to the signal. Any design that cannot separate the two is not
> an evaluation.

### Phase E — Govern

**Step 16. Register the signals and assign ownership.**

A signal registry record contains: signal name, source system, contractual basis, grain, latency
distribution, refresh schedule, coverage share, validation rules, owner role, and the date of the
last information test. Without the last field the registry is an inventory, not a control.

**Step 17. Monitor for signal death.**

A signal dies when it still arrives but no longer carries information. Five monitors catch it:

- **Coverage.** Share of item-locations with a valid value in the last refresh. A fall of more than
  a few points is a data failure, not a demand event.
- **Distribution shift.** Mean, variance, and the share of zeros against a trailing baseline. A
  retailer that changes its store estate changes all three.
- **Residual information.** The rolling correlation between the signal's innovation and the
  forecast error. When it falls toward zero, the signal has been absorbed — or has stopped
  working.
- **Coefficient stability.** For a regression or a tree ensemble, the direction and magnitude of
  the signal's contribution over rolling windows. A sign flip is a structural break.
- **Placebo gap.** Re-run the shuffled-signal test at each quarterly refresh. When the placebo
  catches up with the real signal, the real signal is dead.

**Step 18. Keep a kill switch and a change log.**

Every sensing input should be switchable off without a release cycle, and every change to the
signal set, the model, or the horizon band should be logged with the evidence that justified it.
This is the same discipline Module 08 asks of a routing table, applied to a data pipeline.

### 5.1 The signal taxonomy, treated honestly

| Signal | What it measures | Typical latency | Lead or lag | Main failure |
|---|---|---|---|---|
| POS / scanner | Consumer purchase at a known store | 1–3 days | Contemporaneous to slightly leading | Censored by stockouts; distorted by promotion |
| Retailer inventory feed | Stock at the customer's locations | 2–7 days | Leading for the customer's next order | Incomplete coverage; definition drift |
| Order book / open orders | Commitments the customer has made | Hours to 1 day | Leading for near-term shipments | Contains batching, gaming, and allocation |
| Weather | Conditions affecting demand | Forecast available 10–14 days | Genuinely leading | Only material for weather-sensitive items |
| Calendar and holidays | Structural demand shifts | Known years ahead | Leading, but not "sensing" | Confused with a signal; it is a calendar |
| Search / web interest | Query volume for related terms | 1–3 days | Leading for considered purchases | Spurious correlation; campaign coupling |
| Logistics and shipment events | Movement of the firm's own goods | Hours to days | Supply-side, not demand-side | Mistaken for demand |
| Order batching | Order day-of-week and lot sizing | None | A not-signal | Reproduces the buyer's calendar |
| Allocation and rationing | Supply rule under constraint | None | A not-signal | Orders inflate; the rule is the signal |
| Price and promotion changes | Own commercial actions | Known in advance | A not-signal for demand | Pulls demand forward and books it as lift |
| Stockout-censored sales | Availability, not demand | None | A not-signal | Zeros measure emptiness |

Three of these rows are the ones that get argued about, so state the position plainly.

**POS data is the best general-purpose demand proxy and it is not demand.** It is what sold, at
the stores that had stock, at the prices that were on display. Its errors are informative — a
stockout-censored week understates demand by an amount you can estimate from the availability
record — and that is more than can be said for an order series, whose errors are produced by the
buyer's incentives. But a firm that treats POS as ground truth will systematically under-forecast
in promoted weeks and over-forecast immediately after them.

**Calendar effects are not sensing.** A material share of what gets presented as sensing gain in
retail is the correction of a calendar error: a week that contains a public holiday, a month with
five Fridays, a shift in the Easter date, a school-holiday week. These are deterministic and known
years in advance. Correcting them is valuable and should be done in the baseline model, not
attributed to a real-time signal. If a business case's gain disappears once the calendar is
corrected, the case was a calendar case.

**Order batching, allocation, and price changes are not signals.** They are the mechanisms that
make the order series differ from demand. Ingesting them as features teaches a model to reproduce
the buyer's behaviour. Ingesting them as *explanatory variables* — "this week's order is inflated
because it is the last order of the quarter and the buyer is protecting allocation" — is legitimate
and is what a good analyst does instead.

### 5.2 Why the upstream order series is a distorted echo

The bullwhip mechanism is not a modelling artifact; it is produced by four behaviours, each of
which is locally rational. Demand signal processing means each tier forecasts from its own orders,
which are already distorted. Order batching means the buyer's calendar, not the consumer's, sets
the timing. Price fluctuation means promotions pull demand forward and then create a trough.
Shortage gaming means the allocation rule, not the shortage, produces the over-order. These are
the four mechanisms named in Lee, Padmanabhan and Whang (1997), and sixty years of subsequent work
has added detail without changing the list.

The practical consequence for sensing is that **the further upstream you sit, the less your order
data is about demand and the more it is about your customers' inventory policy**. A manufacturer
supplying a retailer sees orders that are a function of the retailer's target stock, its review
period, and its forecast — three quantities the manufacturer does not observe. De-layering
recovers the demand only to the extent that the inventory feed makes those quantities observable.

There is also a measurement trap worth knowing. Aggregating data across products, locations, or
time reduces measured bullwhip even when the underlying behaviour is unchanged, because
independent distortions cancel in the sum. Jin, Williams, Waller and Hofer (2015) demonstrate this
directly: the apparent inconsistency in the bullwhip literature is partly an artifact of
measurement aggregation. So a company that "measures no bullwhip" at the category-month level may
have a great deal of it at the item-store-week level, and the level at which the sensing layer
operates is the level at which it matters.

### 5.3 A sensing gain is not a policy gain

Suppose the sensing layer reduces four-week-ahead WAPE from 22.6% to 22.1%. Something must consume
that. Four consumers are possible, and they produce very different value:

| Consumer | How the gain enters | Typical realisation |
|---|---|---|
| The forecast display | Nothing changes | Zero |
| The order quantity, recomputed each cycle | Order moves by the forecast revision | Small; absorbed by rounding |
| The safety-stock parameter | Parameter falls with forecast error | Medium; realised on the refresh cycle |
| The service target or allocation rule | Policy changes | Potentially large; not attributable to the signal |

The first row describes most deployed sensing systems. The fourth row describes most of the value,
and it is a policy change that the sensing data enabled rather than caused. Reporting them
together is the standard way a sensing business case overstates its return.

## 6. Worked example

Halden Beverage is an illustrative company with internally consistent numbers. It sells a
carbonated soft drink through one grocery retailer in one country. We follow one
stock-keeping unit (SKU), Citrus-24, at one distribution centre, Rotterdam, for one quarter.

### 6.1 The setting

| Quantity | Value | Unit |
|---|---:|---|
| Weekly demand at the DC, mean | 4,200 | cases per week |
| Weekly demand, standard deviation | 900 | cases per week |
| Coefficient of variation | 0.214 | dimensionless |
| Supply lead time from plant | 3 | weeks |
| Review period | 1 | week |
| Protection interval for safety stock | 4 | weeks |
| Target cycle service level | 95% | proportion, z = 1.65 |
| Order multiple accepted by the plant | 600 | cases (one pallet layer) |
| Order gate | Thursday, 17:00 | weekly |
| Freeze | 4 | working days before the gate |
| Carrying cost | 18 | currency units per case per year |
| Item-DC combinations in scope | 240 | count |

The order placed at the gate covers delivery in three weeks and demand through the following
week, which is why the protection interval is four weeks. The four-week-ahead forecast is the one
the gate consumes. This single sentence determines everything that follows, and in most companies
it is not written down anywhere.

### 6.2 The forecast at four horizons

Halden evaluates two forecasts on the same rolling origin over twelve weeks:

- **Baseline.** A weekly statistical forecast using history, seasonality, and the promotion
  calendar. It is the incumbent.
- **Sensing.** The same model family, with POS scan data at retailer-store level and the retailer's
  DC stock feed added as inputs. This is construction 2 plus 3 from section 3.2.

| Horizon | Baseline WAPE | Sensing WAPE | Absolute gain | Relative gain |
|---:|---:|---:|---:|---:|
| 1 week | 18.4% | 16.9% | 1.5 pts | 8.2% |
| 2 weeks | 19.9% | 18.6% | 1.3 pts | 6.5% |
| 3 weeks | 21.4% | 20.3% | 1.1 pts | 5.1% |
| 4 weeks | 22.6% | 22.1% | 0.5 pts | 2.2% |

WAPE is the sum of absolute errors divided by the sum of actuals, a proportion.

The headline that a programme office would publish is 8.2%. The number that matters to the gate is
2.2%. Both are correct computations. They are answers to different questions, and the gap between
them is the most common misrepresentation in this field — not usually deliberate, and almost
always unchallenged.

### 6.3 The lead-time test applied to Halden's signals

| Signal | Usable day | Margin against `gate − freeze` | Verdict |
|---|---:|---:|---|
| Holiday calendar | −365 | 361 days | Usable; belongs in the baseline |
| Weather forecast | −14 | 10 days | Usable |
| Search interest | −6 | 2 days | Usable, narrowly |
| POS scan | −2 | −2 days | Inside the freeze; execution only |
| Retailer DC stock | +1 | −5 days | After the gate; execution only |
| Open orders | +0.2 | −4.2 days | At the gate; next cycle |
| Own shipment actuals | +5 | −9 days | After the gate; measurement only |

The POS signal — the centrepiece of the sensing programme — misses the gate. It arrives two days
after the freeze closes. It can inform expediting, reallocation, and the next cycle's forecast, but
it cannot change the order that the four-week-ahead forecast was meant to inform.

This is not a defect of the POS data. It is a property of the decision. The correct responses are
to shorten the freeze (a policy change), to move the gate later (also a policy change), or to
accept that the value is in execution and measure it there. A team that builds a better model and
ignores this table will report an accuracy gain and deliver no inventory benefit, and will not
understand why.

### 6.4 From accuracy to inventory

Safety stock is `z × σ_e`, where `σ_e` is the standard deviation of forecast error over the
protection interval. For a normal error with WAPE `w` and mean demand `m`, a working approximation
is `σ_e ≈ 1.25 × w × m`, since WAPE approximates 0.8 of the mean absolute deviation and the mean
absolute deviation approximates 0.8 of the standard deviation.

```text
baseline:  sigma_e = 1.25 x 0.226 x 4,200 = 1,187 cases
           SS      = 1.65 x 1,187          = 1,959 cases

sensing:   sigma_e = 1.25 x 0.221 x 4,200 = 1,160 cases
           SS      = 1.65 x 1,160          = 1,914 cases

reduction  = 1,959 - 1,914                 =    45 cases per item-DC
```

At a carrying cost of 18 currency units per case per year, that is 810 per item-DC per year, or
194,400 across 240 item-DC combinations. That is the **arithmetic ceiling** of the programme: what
the inventory would fall by if every parameter were refreshed immediately and every order were
placed in exact cases.

Two frictions reduce it. The first is rounding: orders are placed in 600-case pallets, so a
revision of 46 cases — the median absolute revision at four weeks in this pilot — changes the
pallet count only when it crosses a boundary. The second is the parameter refresh: the safety
factor is re-derived quarterly from a trailing 13-week error estimate, so the improvement enters
the policy on average six weeks after the forecast.

### 6.5 The decision-delta audit

Run step 14 from section 5 on the pilot data.

| Cell | Forecast | Policy | Average inventory | Order changes vs. status quo |
|---|---|---|---:|---:|
| 1 | Baseline | Baseline | 4,059 cases | — |
| 2 | Sensing | Baseline | 4,059 cases | 0% of item-weeks |
| 3 | Baseline | Improved | 3,981 cases | 31% of item-weeks |
| 4 | Sensing | Improved | 3,943 cases | 38% of item-weeks |

Average inventory here is cycle stock (2,100 cases) plus safety stock. The "improved" policy
refreshes the safety factor from the trailing error estimate each cycle instead of quarterly, and
uses a finer rounding increment of 300 cases negotiated with the plant.

Read cell 2: the sensing forecast with the existing policy changes nothing at all. The order
quantity is `forecast + safety stock − inventory position`, and with a quarterly-refreshed safety
factor the forecast revision is absorbed within the rounding. The sensing programme in isolation
has a decision delta of zero.

Read cell 3: the policy change alone — a refresh cadence and a rounding increment — removes 78
cases of inventory per item-DC, or 44 more than the sensing forecast's ceiling. The policy change
requires no new data.

Read cell 4: the combination removes 116 cases per item-DC, of which 38 are attributable to the
sensing input given the improved policy. That 38 is the only defensible claim for the data
programme.

### 6.6 What the programme actually earned

| Quantity | Per item-DC | Across 240 | Unit |
|---|---:|---:|---|
| Sensing ceiling | 45 cases | 10,800 cases | cases |
| Realised, with rounding and refresh lag | 16 cases | 3,840 cases | cases |
| Annual value at 18 per case | 288 | 69,120 | currency per year |
| Policy-only value | 78 cases | 18,720 cases | cases |
| Annual value of policy change | 1,404 | 336,960 | currency per year |
| Programme cost: data, engineering, stewardship | — | 260,000 | currency per year |

The sensing programme, on this scope, costs more than it returns. The policy change, which
requires a conversation with the plant about rounding and a change to a refresh parameter, returns
more than both. Both statements are specific to this business, and the second is the one that
survives scrutiny.

### 6.7 The pilot readout and its confound

The twelve-week pilot showed average inventory 1.4% below the matched control DC. It also
overlapped a retailer promotion in week 7 that pulled forward roughly a tenth of the following
month's volume. A before-and-after comparison would have credited the programme with the
promotion's inventory effect. The correct design is the material of Module 05.

> **Note:** Every number here is illustrative and internally consistent; none is a benchmark. The
> structure of the argument — horizon mismatch, freeze exclusion, rounding, refresh lag, confounded
> readout — is what transfers.

## 7. Code, query, or calculation

The calculation below is the lead-time test of section 3.4, done by hand. It is the first thing to
run on any candidate signal, and it requires no software.

```text
Lead-time test, Halden Beverage, SKU Citrus-24, Rotterdam DC
Working days. Day 0 = order gate. Freeze F = 4 days before the gate.

  signal                 usable day   test: usable < -4    margin
  -----------------------------------------------------------------
  holiday calendar           -365     PASS                 361 d
  weather forecast            -14     PASS                  10 d
  search interest              -6     PASS                   2 d
  POS scan                     -2     FAIL                  -2 d
  retailer DC stock            +1     FAIL                  -5 d
  open orders                +0.2     FAIL                -4.2 d
  own shipment actuals         +5     FAIL                  -9 d

  margin = (0 - F) - usable day = -4 - usable day
  PASS: the signal can still change this order.
  FAIL: the signal can change execution, or the next cycle, only.
```

Three checks before trusting the table.

First, confirm the gate. If the gate is actually Wednesday rather than Thursday, every margin
shifts by one day and the search signal moves from 2 days of margin to 3.

Second, confirm the freeze is a single number. If long-lead components are frozen eight weeks out,
the freeze for those items is 40 working days, and no signal in the table passes except the
calendar.

Third, compute the margin per gate rather than once. A daily feed against a weekly gate makes the
gate on some weeks and misses it on others; the share of gates made is the number the business
case needs, not the nominal margin.

## 8. Data, standards, and regulatory context

Nothing in this topic is regulated. What governs the data is a mixture of voluntary standards,
contract terms, and privacy law, and the contract terms usually bind harder than the standards.

### 8.1 Message and identity standards

| Standard | Governs | What it constrains here | Grade |
|---|---|---|---|
| ANSI X12 852 | Product activity data | The sell-through and inventory feed from a retailer; weekly by location if configured | B |
| ANSI X12 867 | Product transfer and resale | Resale and inventory movement reporting | B |
| EDIFACT PRODAT / INVRPT | European equivalents | Product data and inventory report messages | B |
| GS1 GTIN | Trade item identity | The join key between POS and internal item master | B |
| GS1 GLN | Location identity | The join key between store scans and DCs | B |
| GS1 EPCIS 2.0 | Event data | Object-level evidence of receipt, stocking, and unavailability | B |
| ISO 8601 | Date and week representation | Bucket boundaries and week 53 | B |
| 4-4-5 retail calendar | Retail period boundaries | A bucket that is not a calendar month | C |

The 852 transaction is the workhorse of consumer-goods sensing, and its limitations matter more
than its specification. It is a *reporting* message, not an event stream: it carries aggregates by
item and location for a period, so it cannot tell you that a shelf was empty on Tuesday afternoon.
That is why stockout censoring is usually estimated rather than observed, and why EPCIS — the only
general standard that can evidence an object-level event — is the strongest structural remedy
where a firm can get it deployed.

### 8.2 Process frameworks, which are conventions

CPFR (Collaborative Planning, Forecasting and Replenishment) is a voluntary inter-firm process
guideline, not a data standard, and its adoption says nothing about data quality. SCOR is a
reference vocabulary. Grade C for both. Demand-driven MRP and similar frameworks are
vendor-adjacent methodologies; Grade D.

### 8.3 Contracts, privacy, and the constraints nobody documents

Three constraints bind more tightly than any standard.

**Retailer data agreements.** POS and inventory feeds are provided under contract. The contract
typically restricts use to the named retailer's products, prohibits sharing with other retailers,
and may prohibit use in pricing or assortment decisions. A model trained on one retailer's store
data and deployed across the estate may breach the agreement, and the analytics team is rarely
shown the clause.

**Personal data.** Loyalty-card and e-commerce basket data are personal data under the General Data
Protection Regulation (GDPR) in the European Union and comparable regimes elsewhere. Aggregate
search indices are not. A sensing layer that uses individual-level purchase data needs a lawful
basis, minimisation, and retention rules; one that uses store-level aggregates usually does not.
The boundary is at the grain, and getting it wrong is a legal problem rather than a modelling one.

**Statistical disclosure.** Retailer contracts often forbid publishing numbers from which a
competitor could infer the retailer's sales. This constrains what can appear in a shared forecast,
not just in a report, and it is one reason CPFR-style collaboration is less common in practice
than the literature implies.

> **Note:** Regulatory content in this course is educational. Data-sharing agreements and privacy
> obligations require review by the organisation's legal function before a sensing feed is
> onboarded.

## 9. Industry practice and case evidence

### 9.1 What the academic evidence actually shows

The literature is more heterogeneous than either vendor material or textbook summaries suggest.
Four findings carry most of the weight.

| Study | Design | Finding | Grade |
|---|---|---|---|
| Hartzel and Wood (2017), *European Journal of Operational Research* | 60,651 orders, 25 distribution centres, hierarchical linear model | 11.2% overall forecast accuracy improvement from shared POS data; effect curvilinear in order size and variability; largest for infrequently ordered items | A |
| Narayanan, Sahin and Robinson (2019), *Journal of Operations Management* | Daily data, one large consumer-products chain | POS data improve accuracy for demand planning but **not** for order-fulfilment planning | A |
| Saarinen et al. (2025), *International Journal of Operations and Production Management* | Conceptual replication of the POS-integration result | POS integration did **not** significantly improve accuracy; value reframed as demand-planning alignment | A |
| Prabhudesai et al. (2020), SAS conference paper | Food CPG, 2012–2019 weekly shipments | Product-level accuracy 81.2% to 86.7%; product-location 73.5% to 80.6%; POS plus inventory added 1–2 points at short lags, with higher positive bias | B |

Read together, the picture is coherent. POS data can improve forecast accuracy, and the
improvement is real in well-designed studies. It is also conditional: on the decision being demand
planning rather than order fulfilment, on the item being infrequently ordered, on a short horizon,
and on clean data. The replication finding is the one a business case never cites: the effect is
not robust enough to generalise, and a programme that assumes it will reproduce the best published
result is assuming the best case.

Abolghasemi, Rostami-Tabar and Syntetos (2023), as reported in the search results for this topic,
found order history outperforming POS data at distribution-centre level — consistent with
Narayanan et al., since the DC's decision is an order-fulfilment decision. The pattern is not "POS
good" or "POS bad"; a signal helps the decision whose information set it belongs to.

### 9.2 Why the estimates disagree

1. **Different dependent variables.** Some studies forecast consumer demand, others orders, others
   shipments. A gain on one is not a gain on another.
2. **Different horizons.** A 1-week gain can be several times a 4-week gain, as section 6 shows.
3. **Different portfolios.** The curvilinear result means the average hides the distribution.
4. **Different baselines.** Comparing with a naive benchmark is not comparing with an incumbent.

### 9.3 What vendors claim, and why the numbers drift

The most-repeated commercial claim in this area is a range of roughly 5–20% forecast accuracy
improvement with 5–10% safety-stock or inventory reduction. Its provenance is unstable. As returned
by the searches for this topic: Kinaxis attributes "5–20% improvements in forecast accuracy and
5–10% reductions in safety stock" to Kearney (2023) with no link to a primary study; the Kearney
and AWS ebook states 10–20% accuracy improvement and 5–10% *inventory* reduction; a secondary trade
article attributes the same 5–20% and 5–10% figures to McKinsey and different figures to Kearney.
No primary document with a stated sample, baseline, or method appeared.

Treat every such figure as Grade D. That does not make it false; it makes it untestable. A range
that has been attributed to two consultancies and has drifted between "safety stock" and
"inventory" is not a benchmark, and a business case that adopts it as a target has adopted a
marketing artefact.

### 9.4 Signals beyond POS: weather and search

**Weather.** The evidence is stronger and more specific than for POS. Steinker, Hoberg and
Thonemann (2017) studied a large e-commerce fashion retailer and found forecast error reductions
of 8.6–12.2% on average and up to 50.6% on summer weekends, translating into workforce-planning
value. Chan and Wahab (2024) report weather explaining up to 47% additional variance for
individual products and 56% for categories. Against that, Lv, Han and Hu (2023) found weather
information did not help basic clothing items and acted as noise, while helping seasonal items
substantially. The conditional structure is the finding: weather is a leading signal for
weather-sensitive items at short horizons, and a distraction elsewhere. Grade A for the first two,
B for the third.

**Search.** Also conditional. Fritzsch et al. (2019) found product-level search volume improved
conventional time-series forecasts for two consumer audio products; Wu, Xu and Bach (2023) found
Google Trends improved forecasts for conventional but not organic avocado sales; Wachter, Widmer
and Klein (2019) report out-of-sample error reductions of about 5% from keyword selection in
automotive sales. The recurring caveat is that search interest tracks campaigns as much as intent,
so a change in media plan can break the relationship without any change in demand. Grade A for the
individual studies; the heterogeneity means no general claim is supported.

### 9.5 What practitioners actually deploy

Most deployed sensing is construction 1 — a daily re-run of an existing forecast with a few
external covariates, presented through an exception dashboard. The MIT capstone on the topic
(Humphrey and Laiño, 2018) assessed three levers at a firm already below 20% mean absolute
percentage error and found results mixed across companies, with data quality of the additional
signals the main discriminator, and recommended piloting latency reduction before adding signals.
Grade C: a single-firm capstone, useful as a caution rather than as evidence of prevalence.

## 10. Failure modes and red flags

| Failure | Red flag in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Horizon mismatch | "Accuracy improved 8%" with no horizon stated | Sensing gains decay with horizon; the headline uses the shortest | Business case built on a decision nobody makes | Report the full horizon profile; headline the gate horizon |
| Freeze ignored | Nobody in the room can state the freeze length | The signal arrives inside the frozen window | Model improves, orders do not | Compute the freshness margin before building |
| Decision delta of zero | Forecast updates daily; order quantity unchanged for months | The policy does not consume the forecast | No value; dashboard maintenance cost | Run the decision-delta audit |
| Signal never tested against the incumbent | Correlation with demand is shown; incremental error is not | The incumbent already contains the information | Spurious feature, added complexity | Incremental WAPE plus placebo |
| Placebo not run | Gain survives when the signal is shuffled | The gain comes from model class or extra parameters | False attribution | Shuffle-in-time test at every refresh |
| Confounded pilot readout | Pilot period overlaps a promotion or a competitor exit | Treatment and event coincide | Effect overstated, sometimes by an order of magnitude | Module 05 designs: stepped wedge, DiD, matched control |
| Censored POS taken as demand | Zero-sale weeks retained without an availability flag | Stockouts suppress measured demand | Systematic under-forecast where availability is poor | Availability join; censoring rate reported |
| Coverage drift | Same item, fewer stores reporting each month | Retailer estate changes; feed silently narrows | Forecast represents a shrinking population | Monitor coverage share per item-week |
| Definition drift in the inventory feed | Channel stock jumps by a constant at a system change | Retailer redefines the snapshot | De-layering produces a false offtake spike | Change log on the counterparty side; break detection |
| Signal death | Feed arrives on time; incremental information is zero | Retailer changed vendor, or the relationship broke | Programme continues on decoration | Rolling residual-information and placebo-gap monitors |
| Rounding absorbs the revision | Median revision below the order multiple | Order quantity is quantised | Zero decision delta despite accuracy gain | Negotiate a finer increment, or measure at the policy level |
| Policy attributed to data | Combined programme reports one number | Sensing and policy change launched together | Value not reproducible elsewhere | Four-cell decomposition from step 14 |

Two deserve a sentence beyond the table. **Signal death** has the longest latency: nothing fails,
nothing alerts, and cost continues while benefit goes to zero. It is caught only by a monitor that
tests information rather than availability. **Rounding absorbs the revision** most often turns a
genuine accuracy gain into no operational change, invisibly, unless the order sequence is compared
with and without the sensing forecast.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test |
|---|---|---|---|
| The decision gate is known and stable | The lead-time test is computed against it | Margins are wrong; signals judged usable that are not | Ask three functions; compare with the system time fence |
| The freeze is a single number per item class | Sets the actionability envelope | Horizon computed for the wrong commitment | Stratify by long-lead component |
| Signal latency is measured, not documented | Vendor frequency is not usable latency | Overstated margins; late decisions | Timestamp distribution over eight weeks |
| The incumbent forecast is the right comparator | Incremental information is defined against it | Gain overstated; feature adds nothing | Rolling-origin comparison on the same inputs |
| The channel-stock feed is complete and consistently cut | De-layering is an identity, not a model | Reconstructed offtake mixes demand with coverage error | Reconcile orders, stock, and offtake across a closed period |
| The evaluation period is free of confounding events | Causal attribution requires it | Effect attributed to the wrong cause | Module 05 design; event log review |
| The decision rule consumes the forecast | Otherwise there is no decision delta | Zero value despite accuracy gain | Four-cell audit |
| The signal's relationship to demand is stable | Forecasts extrapolate it | Accuracy degrades silently | Rolling coefficient and placebo-gap monitors |
| The portfolio is homogeneous enough to pool results | Aggregate claims need it | Average hides items where the signal hurts | Segment by order frequency and volume |
| The programme cost is counted | Value is a net figure | Negative-return programme reported as positive | Cost the stewardship explicitly |

Three limits deserve plain statement. **The framework cannot tell you whether a signal is causal.**
It tests timeliness and incremental out-of-sample information; causation needs the identification
work of Module 05 and category knowledge. **The framework assumes a decision exists.** Where the
order is set by contract or fixed schedule, no sensing layer helps, and the honest recommendation is
to change the contract. **The value estimates are local.** Section 6's numbers are illustrative;
published estimates range from zero to double-digit improvements, and that distribution is the
finding.

## 12. Exercises

### Exercise 1 — Map the gate (recall and application, 40 minutes)

For a planning process you know, establish the gate, the freeze, the cadence, and the flexible
fraction, and interview at least two functions to check agreement.

**Deliverable:** a one-page timeline in the style of section 3.1, with the four fields stated.
**Acceptance criteria:** a named gate with time zone; the freeze sourced from a system parameter
or a contract, not from memory; disagreements between functions recorded rather than averaged.
**Estimated time:** 40 minutes.

### Exercise 2 — Lead-time arithmetic (calculation, 30 minutes)

Take five candidate signals for that process and compute usable day, freshness margin, and verdict
for each, per gate over eight weeks.

**Deliverable:** a table with the five signals and the share of gates each signal makes.
**Acceptance criteria:** latency measured or explicitly assumed with the assumption stated; at least
one signal fails the test and is reported as failing; the arithmetic is reproducible by hand.
**Estimated time:** 30 minutes.

### Exercise 3 — Classify a signal inventory (application, 45 minutes)

Classify twelve signals as leading, lagging, contemporaneous, or not-a-signal, with a reason.

**Deliverable:** a twelve-row table with classification, reason, and the decision each could change.
**Acceptance criteria:** at least three items are classified as not-a-signal with a mechanism named;
calendar effects are separated from sensing; no item is classified without a stated latency.
**Estimated time:** 45 minutes.

### Exercise 4 — De-layer a channel (application, 50 minutes)

Reconstruct offtake for one item over thirteen weeks and compare it with reported POS.

**Deliverable:** a three-series plot or table (orders, reconstructed offtake, POS) and a reconciliation.
**Acceptance criteria:** the three completeness conditions are tested rather than assumed; the
residual is quantified; at least one failure condition is demonstrated or explicitly ruled out.
**Estimated time:** 50 minutes.

### Exercise 5 — Price the decision delta (calculation, 45 minutes)

For a policy with a stated order multiple and refresh cadence, compute the share of item-weeks in
which a forecast revision changes the order.

**Deliverable:** the four-cell table from section 5 step 14, populated with your numbers.
**Acceptance criteria:** the policy is stated in full; rounding and refresh lag are modelled; the
sensing-only cell is reported even if it is zero.
**Estimated time:** 45 minutes.

### Exercise 6 — Audit a business case (critique, 40 minutes)

Obtain a vendor or internal sensing case and audit it against sections 3.4, 9.3, and 10.

**Deliverable:** a one-page audit with a verdict per test and three questions for the owner.
**Acceptance criteria:** horizon, comparator, and placebo are each addressed; every vendor figure
is graded; at least one criticism concerns the decision rather than the model.
**Estimated time:** 40 minutes.

### Exercise 7 — Design the monitor (design, 50 minutes)

Specify the five signal-death monitors for one feed, with thresholds, owners, and responses.

**Deliverable:** a one-page monitor specification and a mock dashboard layout.
**Acceptance criteria:** each monitor has a numeric threshold and a named owner role; the placebo
gap is included; the kill-switch condition is stated.
**Estimated time:** 50 minutes.

### Exercise 8 — Defend the recommendation (design and critique, 40 minutes)

Recommend whether a named firm should build a sensing layer, a policy change, both, or neither.

**Deliverable:** a one-page memo with the recommendation, the evidence, and the reversal condition.
**Acceptance criteria:** the recommendation distinguishes sensing from policy value; the evidence
is graded; the reversal condition is specific and testable.
**Estimated time:** 40 minutes.

## 13. Self-check answers

**Exercise 1.** A strong answer reports a gate such as "Thursday 17:00 CET", a freeze sourced from
the planning system's time fence (for example four working days), and a flexible fraction estimated
from the share of volume on short-lead components. The interview check usually produces three
different freeze values; the correct response is to record the disagreement and stratify by item
class, not to average.

**Exercise 2.** The arithmetic is `margin = (gate − freeze) − usable day`. A signal with a
two-day median lag and a four-day freeze against a weekly gate makes the gate in most weeks and
misses it when the 90th-percentile lag exceeds six days. The expected finding is that at least one
signal — usually POS — sits inside the freeze, and that the share of gates made is materially below
100%.

**Exercise 3.** The not-a-signal set should include order batching (the buyer's calendar), allocation
or rationing (the supply rule), and price or promotion changes (the firm's own action). Calendar and
holiday structure should be classified as deterministic and assigned to the baseline, not to
sensing. A submission that classifies all twelve as signals has not done the exercise.

**Exercise 4.** A correct reconstruction shows orders and offtake diverging by exactly the change in
channel stock. Typical findings: the feed covers 85% of stores, so the residual has a coverage
component; the stock snapshot is cut two days after the order transmission, so the delta includes
unrecorded movement; and the retailer's "stock" excludes in-transit-to-store units, so restocking
signals appear one period early.

**Exercise 5.** With an order multiple of 600 cases and a median absolute revision of 46, the
sensing-only cell usually changes zero orders. With a 300-case increment and a per-cycle refresh of
the safety factor, the combined cell changes roughly a third of item-weeks. The answer should
report both cells and state which one the business case is claiming.

**Exercise 6.** Look for: a stated horizon (usually absent); a comparator (usually a naive
baseline); a placebo (almost always absent); a decision (usually "the planner"); and figures graded
D. Three good questions: "At what horizon, and is that the horizon the order uses?"; "What happened
to accuracy when you shuffled the signal?"; "Which order changed, on what date, by how much?"

**Exercise 7.** A defensible specification monitors coverage share (alert below a stated floor),
distribution shift (mean and zero share against a trailing baseline), residual information
(rolling correlation of signal innovation with forecast error), coefficient stability, and the
placebo gap at each quarterly refresh. The kill switch triggers when the placebo gap falls below a
stated threshold for two consecutive refreshes.

**Exercise 8.** A strong memo separates the arithmetic ceiling from the realised value, compares it
with the policy-only value, and reports the programme cost. In most consumer-goods settings the
policy change dominates, and the recommendation is to do the policy change first, then re-test
whether sensing adds anything on top. The reversal condition is a short freeze, a daily gate, and a
category where the four-week signal has demonstrated incremental information.

## 14. Interview questions

**1. "What is demand sensing?"**

Strong: names the four constructions and asks which one the interviewer means, then defines it as a short-horizon update that consumes current signals to change a decision.
Weak: says "using real-time data to forecast better", which describes a dashboard.

**2. "Why might a demand sensing project improve accuracy and reduce no inventory?"**

Strong: gives the horizon mismatch, the freeze, rounding, and the refresh lag, and says the order did not change.
Weak: blames data quality or change management.

**3. "How do you decide whether a signal is worth ingesting?"**

Strong: the arrival, information, and decision tests, in that order, with the placebo.
Weak: says "test whether it correlates with sales".

**4. "What is the difference between sell-in and sell-out?"**

Strong: defines both and states the identity linking them through channel-stock change.
Weak: says sell-in is to the retailer and sell-out is to the consumer, and stops.

**5. "Why does the bullwhip effect make upstream order data misleading?"**

Strong: names signal processing, batching, price fluctuation, and shortage gaming, and notes each is a rational local behaviour.
Weak: says "forecasts get worse upstream".

**6. "How would you evaluate a sensing pilot?"**

Strong: names a Module 05 design, states the decision metric and the decision horizon, and separates sensing from policy.
Weak: proposes a before-and-after accuracy comparison.

**7. "A vendor claims 5–20% accuracy improvement and 5–10% safety-stock reduction. What do you ask?"**

Strong: asks for the primary study, the horizon, the baseline, the decision, and notes the attribution has drifted between consultancies.
Weak: accepts the range as a planning assumption.

**8. "What is signal death and how do you detect it?"**

Strong: a feed that arrives but no longer carries information, detected by residual-information and placebo-gap monitors.
Weak: says "when the data stops arriving".

**9. "Why can weather be a strong signal in one category and noise in another?"**

Strong: explains conditional sensitivity, the anomaly versus the level, and the short horizon at which it is usable.
Weak: says weather affects everything.

**10. "Your sensing model and your safety-stock policy disagree. Who wins?"**

Strong: the policy wins for the cycle; the disagreement is logged and reviewed against the decision metric, not the forecast.
Weak: the model wins because it is objective.

## 15. Key terms

- **Demand sensing** — a short-horizon update that consumes current signals to change a decision.
- **Decision gate** — the moment a commitment is made; the reference point for every lead-time calculation.
- **Order freeze** — the period before the gate during which changes are blocked.
- **Planning time fence** — the system parameter that enforces the freeze.
- **Signal latency** — event-to-usable time, including extraction and refresh.
- **Freshness margin** — gate minus freeze minus usable day; negative means unusable.
- **Actionability envelope** — the decisions a signal can still change.
- **POS** — point of sale; the consumer transaction record.
- **Sell-in / sell-out** — sales to the retailer and sales by the retailer.
- **De-layering** — recovering offtake from orders by netting out channel-stock change.
- **Channel inventory** — stock between the firm and the consumer.
- **Bullwhip effect** — upstream amplification of downstream variability.
- **Shortage gaming** — over-ordering to protect allocation.
- **Stockout-censored demand** — demand suppressed by unavailability.
- **Baseline** — the counterfactual level absent the intervention.
- **Placebo signal test** — re-running with the signal shuffled in time.
- **Signal decay** — loss of incremental information over time.
- **WAPE** — sum of absolute errors divided by sum of actuals.
- **Decision delta** — the change in the decision caused by a forecast change.
- **Policy improvement** — a change to a decision rule rather than a forecast.

## 16. Sources

**Peer-reviewed and academic (Grade A unless noted)**

- Lee, H. L., Padmanabhan, V., and Whang, S. (1997). "Information Distortion in a Supply Chain: The Bullwhip Effect." *Management Science*, 43(4), 546–558. DOI: 10.1287/mnsc.43.4.546.
- Hartzel, K. S., and Wood, C. A. (2017). "Factors that affect the improvement of demand forecast accuracy through point-of-sale reporting." *European Journal of Operational Research*, 260(1), 171–182. DOI: 10.1016/j.ejor.2016.11.047.
- Narayanan, A., Sahin, F., and Robinson, E. P. (2019). "Demand and order-fulfilment planning: The impact of point-of-sale data, retailer orders and distribution center orders on forecast accuracy." *Journal of Operations Management*. DOI: 10.1002/joom.1026.
- Jin, Y., Williams, B. D., Waller, M. A., and Hofer, A. R. (2015). "Masking the bullwhip effect in retail: the influence of data aggregation." *International Journal of Physical Distribution and Logistics Management*. DOI: 10.1108/IJPDLM-11-2014-0264.
- Steinker, S., Hoberg, K., and Thonemann, U. W. (2017). "The Value of Weather Information for E-Commerce Operations." *Production and Operations Management*. DOI: 10.1111/poms.12721.
- Chan, T., and Wahab, M. I. M. (2024). "A machine learning framework for predicting weather impact on retail sales." *Supply Chain Analytics*. DOI: 10.1016/j.sca.2024.100058.
- Fritzsch, S., Wenger, P., Sibbertsen, P., and Ullmann, G. (2019). "Can Google Trends improve sales forecasts on a product level?" *Applied Economics Letters*, 27(17), 1409–1414. DOI: 10.1080/13504851.2019.1686110.
- Wu, L., Xu, X., and Bach, M. (2023). "Using Google Trends to predict and forecast avocado sales." *Journal of Marketing Analytics*, 11(4), 629–641.
- Millner, A., and Heyen, D. (2021). "Prediction: The Long and the Short of It." *American Economic Journal: Microeconomics*. DOI: 10.1257/mic.20180240.
- Regnier, E. (2020). "What Is Six Hours Worth? The Impact of Lead Time on Tropical-Storm Preparation Decisions." *Decision Analysis*. DOI: 10.1287/deca.2019.0396.
- Kovtun, V., Giloni, A., and Hurvich, C. (2019). "The value of sharing disaggregated information in supply chains." *European Journal of Operational Research*.
- Saarinen, J., Huttunen, J., Rehman, A. et al. (2025). "Revisiting the value of data sharing in retail supply chain demand planning." *International Journal of Operations and Production Management*, 45(11), 1910. Conceptual replication; reports no significant accuracy improvement from POS integration.

**Conference and practitioner research (Grade B)**

- Prabhudesai, S. et al. (2020). "Using Machine Learning and Demand Sensing to Enhance Short-Term Forecasting for CPGs." SAS Global Forum proceedings, paper 4405-2020. Industry co-authorship; single firm.
- Lv, H., Han, S., and Hu, Y. (2023). Clothing sales forecast considering weather information. *Journal of Textile Science and Technology*. Single market.
- Wachter, J., Widmer, T., and Klein, A. (2019). "Predicting Automotive Sales using Pre-Purchase Online Search Data." FedCSIS proceedings.
- Yardley, S., and Petropoulos, F. (2021). "Beyond Error Measures to the Utility and Cost of the Forecasts." *Foresight*, issue 63. Practitioner journal.

**Standards and official (Grade B)**

- ANSI X12 transaction sets 852 and 867; EDIFACT PRODAT and INVRPT.
- GS1 General Specifications: GTIN, GLN, and EPCIS 2.0.
- ISO 8601, date and time representation.
- Regulation (EU) 2016/679, General Data Protection Regulation.

**Company, capstone, and commentary (Grade C or D)**

- Humphrey, S., and Laiño, P. (2018). "Improving Forecast Accuracy through Demand Sensing." MIT Center for Transportation and Logistics capstone with Johnson & Johnson Vision Care. Grade C.
- Supply Chain Management Review (2018), summary of the above. Grade C.
- Kinaxis, "What is demand sensing?" attributing 5–20% accuracy and 5–10% safety-stock figures to Kearney (2023). Grade D.
- Kearney and AWS, *Signals Amid the Noise*, stating 10–20% accuracy and 5–10% inventory figures. Grade D.
- Infor, Doss, Logio, Horizon Solutions, "what is demand sensing" pages. Grade D.

**Note on citations.** Bibliographic details are as returned by search and have not been checked against publisher records line by line. Where a DOI was returned it is given; where a detail was uncertain the entry says so. The vendor range in section 9.3 is reported as it circulates, not endorsed.

## 17. Links

- This module's index: [`_index.md`](_index.md)
- Part II index: [`../_index.md`](../_index.md)
- Module survey file: [`../../../course/09_demand_sensing_promotions_npi.md`](../../../course/09_demand_sensing_promotions_npi.md)
- Forecastability and the limits of any forecast: [`../07_time_series_forecasting/01_forecastability-benchmarks.md`](../07_time_series_forecasting/01_forecastability-benchmarks.md)
- Error metrics, which this topic uses at the decision horizon: [`../07_time_series_forecasting/09_error-metrics-cost.md`](../07_time_series_forecasting/09_error-metrics-cost.md)
- Hierarchy and intermittent classification: [`../08_hierarchical_intermittent_probabilistic_forecasting/05_intermittent-classification.md`](../08_hierarchical_intermittent_probabilistic_forecasting/05_intermittent-classification.md)
- Randomised and quasi-experimental designs for pilots: [`../../01_foundations/05_experimentation_pilots_evidence/_index.md`](../../01_foundations/05_experimentation_pilots_evidence/_index.md)
- Part I index: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)

Sibling topics in this module, not yet written:

- [`02_sales-decomposition.md`](02_sales-decomposition.md)
- [`04_baseline-incrementality.md`](04_baseline-incrementality.md)
- [`05_cannibalization-halo.md`](05_cannibalization-halo.md)
- [`10_post-launch-learning.md`](10_post-launch-learning.md)







