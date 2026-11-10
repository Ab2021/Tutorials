---
case_id: "CASE-01-06-001"
case_title: "Dynamic pricing for an ecommerce retail assortment"
case_version: "2.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "hybrid"
domain: "Ecommerce retail pricing"
primary_pattern: "causal-confounding"
secondary_patterns: ["evaluation-mismatch", "constraint-violation"]
evidence_level: "illustrative"
source_basis: "constructed teaching inputs; no company data, system, or measured outcome is described"
as_of_date: "2026-11-10"
owner_role: "Pricing director (decision owner); pricing science lead (analysis owner)"
source_index: "INDEX.md"
review_trigger: "Change in landed cost basis, payment or marketplace fees, eligibility rules, price-movement limits, or the price-experiment design"
---

# Dynamic pricing for an ecommerce retail assortment

**Evidence disclaimer.** This is a constructed teaching case. The retailer, catalogue, cost figures,
conversion rates, elasticities, capacity limits and financial totals below are **ILLUSTRATIVE**
inputs chosen so the arithmetic is checkable by hand. No system is deployed, no experiment was run,
no company result is reported, and no retailer's practice is described. Every claim about the world
is either arithmetic that this document performs or an assumption labelled as one. The case exists to
show how a price recommendation is converted into a bounded, refusable decision with a stated owner,
a stated cut-off, and a stated way to be wrong.

## 1. The One-Line Framing

Given one eligible product-market-day, publish one shelf price for the following trading day, chosen
by the pricing director from a candidate set that the pricing science team can defend, so that
expected contribution per available unit rises without breaching price-movement, inventory-reserve,
or customer-protection rules, compared with holding the previously approved price.

The decision is a *single* number per product-day. It is not a personalised offer, not a discount
campaign, and not a markdown ladder. The naive framing — "predict demand, pick the price that
maximises predicted revenue" — fails for three separate reasons, each of which this case costs out.
Revenue ignores the cost of goods, so a price cut looks better than it is. Predicted demand at a
price the catalogue has never charged is an extrapolation, not a measurement. And the elasticity
that the extrapolation relies on is estimated from observational variation that is entangled with
promotions, stockouts and seasonality.

The arithmetic below produces a defensible answer that is smaller than the proposal suggests. A
planned 10% across-the-board price cut is contribution-negative by 3.37%, or 8.78% once a $1.10
per-order variable picking cost is included. The elasticity implied by the cut's own observed
conversion lift is 2.19, while the cut requires 2.52 to break even on contribution. Under a binding
inventory cap the same catalogue supports a *price increase* to 19.39, worth 17.93% more contribution
than the baseline price.

**Takeaway:** the useful output of a pricing model is not "the optimal price"; it is a ranked,
constraint-filtered candidate set plus the one number that decides between the top two candidates.

## 2. Context & Background

A mid-sized online retailer sells replenishable household and personal-care goods from its own
warehouse. Prices are set centrally per market, published to the catalogue, and read by every
customer in that market for the interval the price is live. There is one currency, one legal
jurisdiction, and one fulfilment network in the pilot scope.

The current process is manual and slow. A category merchant reviews a weekly margin report, decides
which products to reprice, and applies the change through a spreadsheet upload that is validated by
a rules service. The rules service enforces a floor, a ceiling, and a maximum percentage move per
day. Nothing in the process estimates demand response. When a product goes out of stock, the
merchant often cuts its price on the assumption that demand has fallen, which is the opposite of the
correct inference.

Four accountability boundaries are routinely blurred, and separating them is most of the design.

*Merchandising* owns assortment, vendor terms and the commercial relationship. It decides which
products are in scope and what the contractual floor is.

*Pricing science* owns the demand model, the elasticity estimate, the candidate generator and the
evaluation report. It recommends; it does not publish.

*Commerce engineering* owns the price publication path, the cache invalidation behaviour, and whether
the price a customer sees on the listing page equals the price in the cart and the price on the
receipt. It does not decide prices.

*Finance* owns the contribution definition, the landed-cost basis, the fee schedule and the
maturation of return-related adjustments. It signs off on whether a claimed saving is real.

Two supporting functions gate any change. *Legal* confirms that no price is individual-specific and
that no protected attribute enters the decision. *Inventory operations* supplies sellable on-hand
and the safety reserve that any markdown must respect.

The scale used for planning is deliberately round: 10,000 eligible SKUs, one recommendation per SKU
per local day, one market. That is a workload assumption for capacity arithmetic, not a statement
about any retailer's catalogue. Peak trading hours affect read capacity on the publication path and
are a separate engineering concern from the batch recommendation volume, which is flat.

Two structural properties of this problem are worth stating before any method is chosen. First, the
decision repeats daily, so a small per-decision error compounds across 3.65 million decisions a year
at the assumed scale; a 0.3% average contribution improvement is worth having, and a 3% average
regression is existential. Second, price is a public, observable, remembered quantity. A customer who
buys at 18.00 and sees 16.20 the next day has a legitimate grievance that no contribution model
accounts for.

## 3. The Business Question

**The decision.** For each eligible product-market-day, may the pricing director authorise
publication of a model-recommended price, replacing the merchant's held price, for the following
trading day?

The owner is the pricing director. The team can approve, reject, approve for a bounded subset, or
defer. Deferral is a real option, not a courtesy: the baseline continues, the merchant keeps
discretion, and the cost of deferral is the contribution left on the table, which is unmeasured.

The trigger is a quarterly commercial review at which the merchant organisation has asked whether
the manual process should be replaced. The deadline is the start of the next planning quarter, after
which the change would collide with a seasonal peak and be deferred by default.

The primary objective is incremental realised contribution per eligible product-day over the
maturity window, compared with the held-price baseline on the same products and days. Contribution is
defined as revenue minus landed cost minus payment processing minus marketplace or channel fees,
minus prespecified variable fulfilment cost. It excludes fixed costs, because fixed costs do not
change when a price changes and including them would dilute exactly the comparison the decision
needs.

Four things are explicitly *not* acceptable ways to improve that objective. Converting a stockout
into "demand fell" is not. Raising price on a product whose supply is about to fail is not. Charging
different customers different prices for the same product in the same market is not. Publishing a
price that the cart or the checkout will not honour is not.

The scope excludes personalised and member-specific pricing, regulated or age-restricted products,
third-party marketplace seller offers where the retailer does not control the price, advertising
auctions, and any renegotiation of supplier terms. Those are separate decisions with different owners.

A price change also has a non-financial cost that the objective does not capture. It consumes
customer trust, and trust is consumed faster than it is rebuilt. The decision therefore carries a
guardrail on the frequency and magnitude of moves that is not tradeable against contribution.

The final decision may legitimately be "collect evidence". A negative answer with a measured reason
is a better outcome than a launch with an unmeasured hope, and the case is written so that the
negative answer is reachable.

## 4. Assumptions & Constraints

| ID | Condition and type | Status / verification route | Consequence if false | Owner / trigger |
|---|---|---|---|---|
| A1 | Demand responds to own price with a stable elasticity over the tested range | Unverified; needs randomised price contrasts, not observational regression | Candidate ranking inverts; hold baseline | Pricing science; assortment or season change |
| A2 | Sellable inventory is known at the cut-off and excludes allocations | Unverified; reconcile warehouse snapshot against fulfilment events | Recommends a price for units that do not exist; freeze SKU | Inventory operations; feed lag > 2 h |
| A3 | Landed cost is effective-dated and payable at the decision time | Unverified; reconcile against the payables ledger | Contribution sign error on thin-margin SKUs; exclude SKU | Finance; vendor or freight change |
| A4 | Payment and channel fees are known and constant per order | Unverified; fee schedule version must be pinned to the decision | Margin overstated by the fee delta; recompute candidate set | Finance; fee renegotiation |
| A5 | The same price is served on listing, product, cart and checkout | Unverified; synthetic purchase journey plus receipt comparison | Broken price promise; stop publication and roll back | Commerce engineering; cache or version drift |
| A6 | Products are independent enough that per-product pricing does not move neighbours | Unverified; measure cross-price response in the test | Optimises each SKU into a worse assortment position | Pricing science; substitute-pair refresh |
| C1 | Price must lie within the vendor-contracted floor and the category ceiling | Contractual gate; merchandising maintains the effective-dated bounds | Candidate is filtered out, not penalised; publish baseline | Merchandising; contract amendment |
| C2 | A published price may move at most 10% from the last published price | Internal policy for this case, not a law | Feasible set may become empty; hold and escalate | Pricing director; policy review |
| C3 | No price may depend on an individual's identity or a protected attribute | Legal gate; counsel review outstanding before any exposure | Stop the programme; audit the feature store | Legal; new data source |
| C4 | Accepted orders keep their contracted price | Contractual safeguard; checkout confirms | Goodwill cost and possible regulatory exposure | Commerce; checkout inconsistency |
| C5 | A markdown requires sellable stock above a 15-day safety reserve | Inventory safety rule for this case | Markdown disallowed; SKU excluded from the batch | Inventory operations; reserve revision |
| C6 | No price may be published for a SKU with an open cost dispute | Finance control | Excluded from the batch; count and report the exclusion | Finance; dispute opened |

Three assumptions can reverse the recommendation on their own: A1, A2 and A3. A1 removes the
mechanism, A2 misreports the binding resource, and A3 flips the sign of the objective on every SKU
whose gross margin is thin. A4 and A5 are cheaper to test and more likely to be true, but a failure
in either produces a hard operational stop rather than a soft financial error, so they are gated
before exposure rather than monitored after it.

A scenario constraint is not evidence that a real jurisdiction imposes it. C2, C5 and C6 are
written as this case's rules so that the feasibility arithmetic has something to bite on. A real
programme must replace them with the actual contractual and regulatory text, and must record who
approved each bound and when.

An infeasible candidate set is an expected outcome, not an exception. On a day when the landed cost
jumps and the price-move limit binds, the model's preferred price is more than 10% away from the
published price and every candidate is filtered out. The correct behaviour is to publish the held
price and raise an escalation, not to relax the constraint silently or to interpolate toward the
preferred price in the hope that partial movement is close enough.

## 5. Data, Tools & Method

### 5a. Data Specification

**Grain.** The recommendation grain is product × market × local trading day. Every other table is
either a dimension of that grain or an event that aggregates into it.

**Decision cut-off.** The cut-off is 02:00 local time on the day before the priced day, which is
after the overnight warehouse reconciliation and before the merchant's morning review. Every input
must carry an ingestion timestamp at or before that cut-off. A record that arrived at 02:07 is not
available to the 02:00 decision, no matter how useful it would have been.

**Population.** Eligible SKUs are those owned by the retailer, in the pilot market, with a valid
landed cost, no open cost dispute, no vendor price restriction, and at least one sellable unit. The
population is defined before randomisation and its membership is recorded per day, because an
eligibility rule that changes mid-test changes the estimand.

**Keys.** The decision key is (product_id, market_id, trading_date, policy_version). Exposure events
add (session_id, event_timestamp); order lines add (order_line_id); inventory snapshots add
(snapshot_timestamp). Every join is on stable identifiers, and one-to-many relationships are declared
rather than assumed.

**Units, explicitly.** Price is currency per unit, inclusive of tax, as displayed. Landed cost is
currency per unit, exclusive of recoverable tax, effective-dated. Quantity is fulfilled units, not
ordered units and not sessions. Conversion is orders per session, dimensionless. Contribution is
currency per product-day. Elasticity is a pure number, defined as the percentage change in quantity
per one percent change in price, and carries a sign convention that is stated once and never flipped.

**What the fields mean, and what they do not.**

Displayed price, cart price and paid price are recorded as three independent fields, not derived from
one another, because the entire A5 failure mode consists of them disagreeing.

Stock quantity is sellable on-hand minus allocations. Reservations, in-transit replenishment and
supplier purchase orders are separate fields. Treating in-transit as available is the most common way
to recommend a price for inventory that will not arrive.

Sales quantity is fulfilled units. It is *not* latent demand. A day on which the SKU sold out at
noon records a small number that means "we ran out", not "customers did not want it". Stockout time
must be recorded separately, and any product-day with a stockout inside the trading window is either
excluded or handled with a censored likelihood.

Zero sales on a day when the SKU was listed and available is a genuine observation and must be kept.
Zero sales on a day when the SKU was unlisted is not an observation of demand at all and must be
dropped. Confusing the two silently biases elasticity toward zero.

For causal work, the log must contain the assignment probability, the eligibility flag, the cluster
identifier, the time block, and the price actually displayed. Without the displayed price, an
analysis cannot distinguish a price that was assigned from a price that was overridden by the rules
service at the last moment.

**Exclusions, recorded not implied.** Exclude product-days with an invalid cost record, a product
identity conflict, a contractual price restriction, unresolved inventory, or an open finance dispute.
Count exclusions by reason per day and compare the excluded set against the included set on baseline
price and baseline conversion. If the excluded set is materially different, the estimand has changed
and the report must say so.

**Time split.** Train on a contiguous historical window, tune on the next window, and evaluate on a
final holdout that is never touched during model selection. Encoders, imputation statistics and
scalers are fit on the training window only. The holdout is time-ordered, not randomly interleaved,
because the deployment uses the model to predict forward.

**Delayed labels.** Return-related adjustments mature over a fixed window after the order. The
maturity window is declared in advance. Late adjustments that arrive after maturity are accumulated
in a reconciliation report and are never used to relabel a past evaluation. A silently relabelled
history makes every prior decision look better or worse than it was, and destroys the audit trail.

### 5b. Tools & Stack

The design assumes five components and asserts nothing about which products implement them.

An append-only warehouse holds immutable event snapshots and effective-dated dimensions. Immutability
matters because the evaluation must be replayable against the data that existed at the time.

A batch job, run after the cut-off, validates inputs, fits or loads the demand model, enumerates the
approved candidate prices, applies the hard constraints, and writes a ranked recommendation table
with the reason code for every filtered candidate.

A versioned artefact registry stores the model binary, the feature specification, the constraint set,
the cost basis and the fee schedule as one atomic bundle. A price recommendation is only meaningful
relative to the exact combination that produced it; storing the model without the cost basis makes
the recommendation unreproducible.

A publication service accepts an approved version and writes it to the catalogue with cache
invalidation. It refuses to publish a version that has not passed the constraint check, and it
records the version identifier against every published price.

A monitoring pipeline joins exposures, orders, inventory and finance outcomes after maturity and
emits the decision metrics.

Infrastructure owners select runtime, orchestration and storage engines. This document does not.

### 5c. Method Summary

Five candidate methods were considered. They are listed with the reason each was kept or dropped,
because the method-selection record is part of the evidence.

**Hold the approved price.** This is the baseline and the default. It is transparent, requires no
model, and is wrong only in the sense that it leaves measurable value uncaptured — which is exactly
what the pilot is trying to establish.

**A rule-based markdown ladder.** If stock cover exceeds a threshold, cut by a fixed step. This is
what many merchants do manually. It is auditable and cheap, but it responds to stock rather than to
demand, and it systematically cuts price precisely when the correct action may be to hold.

**Observational regression on historical price variation.** Fit quantity on price with controls. It
predicts well inside the historical distribution and is available immediately. It fails as a causal
device because historical price changes were made *because* someone expected demand to move;
promotions, competitor reactions and stockouts all enter the price variable.

**A randomised price-grid experiment within the approved bounds.** Assign each eligible product-day
to one of four approved price points, hold everything else constant, and estimate response from the
contrast. This is the only option that identifies the object the decision needs. It costs real
revenue during the test and it perturbs customers, which is why it is bounded.

**A contextual bandit.** Allocate traffic adaptively to learn and earn simultaneously. It was
rejected for the initial pilot. Adaptive assignment makes the estimator harder to defend to a
finance audience, it concentrates exposure on the currently-best arm and therefore starves the
contrast that resolves uncertainty, and it interacts badly with the 10% price-move constraint
because arm availability changes with the previously published price.

The chosen sequence is: run the randomised grid, estimate elasticity with an interval, then decide
whether the point estimate is far enough from the decision boundary to act. The rest of this case
executes that sequence on constructed inputs.

### 5d. Provenance, Quality, Exclusions, Leakage

Every input in the constructed panel is generated for this document. There is no source system, so
there is no provenance chain to audit; what can be audited is whether the arithmetic that consumes
the inputs is internally consistent, and that is what section 6 checks.

Four quality checks would need to pass before any real analysis, and each has a specific failure it
detects.

*Row-count reconciliation.* The count of product-days in the decision table must equal the count of
eligible SKUs multiplied by the number of trading days in the window, minus recorded exclusions.
A mismatch means a join duplicated or dropped rows, and duplicated rows inflate apparent precision.

*Price-surface consistency.* For a sample of product-days, the listing price, cart price and paid
price must be equal. Any disagreement is an A5 failure observed in production, not a modelling
problem, and it invalidates any analysis that uses "price" as a single field.

*Cost-basis continuity.* The landed cost effective on the decision date must equal the cost used in
the recommendation. A one-day lag between a vendor price increase and the cost record produces a
recommendation that is profitable on paper and loss-making in the ledger.

*Stockout attribution.* Fulfilled units plus recorded lost units must reconcile to modelled demand
within tolerance on days with a stockout. If they do not, the censoring model is wrong and every
elasticity estimated from those days is biased.

Leakage in this problem does not usually look like a target column. It looks like a feature that is
computed after the fact: return rate measured over the full maturity window, "days since last
stockout" computed from a table that was backfilled, competitor price scraped at end of day and
joined to the same day's decision, or a demand forecast that was itself revised after the outcome was
known. The defence is mechanical: every feature must resolve to a source table with an ingestion
timestamp, and the ingestion timestamp must be at or before the cut-off. A feature that cannot
produce that timestamp is excluded regardless of its measured value.

### 5e. Reproducibility & Method-Selection Ledger

| Decision | Options considered | Chosen | Reason | Reversal condition |
|---|---|---|---|---|
| Identification strategy | Observational regression; randomised grid; bandit | Randomised price grid | Only design that separates price response from the reasons prices moved | Grid infeasible at the SKU level; then use regression with explicit bias bound |
| Objective | Revenue; gross margin; contribution after variable cost | Contribution after variable cost and fees | Revenue rewards unprofitable volume; gross margin omits per-order cost | Finance revises the contribution definition |
| Price grid | 2 cells; 4 cells; continuous | 4 cells, ±5% and ±10% in equal steps around the held price | Four cells allow a curvature check; two cannot | Constraint C2 binds and removes outer cells |
| Demand functional form | Linear; log-log constant elasticity; spline | Log-log constant elasticity | Coefficient is the elasticity the decision needs; linear form's elasticity varies with price | Residual curvature persists after a curvature test |
| Uncertainty | Point estimate; bootstrap interval; analytic interval | Analytic interval plus bootstrap cross-check | Analytic interval is auditable; bootstrap catches non-normal residuals | Heavy-tailed residuals |
| Decision rule | Argmax predicted contribution; act only if interval clears threshold | Act only if the interval excludes the no-change contribution | Prevents acting on noise that the pilot cannot resolve | Interval narrows below one step of the price grid |
| Aggregation | Per-SKU decisions; category decisions | Per-SKU with category-level guardrail | Per-SKU matches the constraint structure; category guardrail prevents assortment distortion | Substitution measured as material |

The reproducibility requirement is concrete. Given the seed, the panel construction script, the grid
assignment and the cost basis version, a second analyst must reproduce every number in this case to
the printed precision. That is the standard the numbers below are held to: they are all closed-form
or four-function arithmetic on stated inputs.

## 6. Analysis Walkthrough

The walkthrough computes one decision eight times from different angles, then shows where the
arithmetic stops being valid. Each block states the formula, the symbol units, the condition under
which it holds, a numerical example with intermediate values, and the interpretation a decision
maker should take from it. All inputs are **ILLUSTRATIVE**.

The base inputs are fixed here and never varied silently: a held price of 18.00 currency units per
unit, a baseline quantity of 1,000 units per product-day, a landed cost of 9.40 per unit, a payment
processing fee of 6.0% of revenue, and a marketplace or channel fee of 2.5% of revenue. The two fees
combine into a single retention factor, which is the first derivation.

### Step 1 — Convert the fee schedule into a retention factor

```text
FORMULA [M-01]: k = 1 - r_pay - r_ch
WHERE: k = net revenue retention factor [dimensionless, 0..1];
       r_pay = payment processing rate on revenue [fraction];
       r_ch  = marketplace or channel fee rate on revenue [fraction].
CONSTRAINT: both fees must be percentage-of-revenue and must not have a fixed component; a fixed
            per-order fee is a cost per order, not a reduced retention factor, and belongs in the
            margin, not here.
NUMERICAL EXAMPLE: r_pay = 0.060; r_ch = 0.025;
                  k = 1 - 0.060 - 0.025 = 1 - 0.085 = 0.915.
INTERPRETATION: of every 100.00 of revenue, 91.50 reaches the contribution line; the remaining 8.50
                is paid to intermediaries and is not available to cover cost of goods.
```

The distinction in the constraint is not pedantry. A processor that charges 6% plus 0.30 per
transaction cannot be represented by a retention factor alone, because the fixed component does not
scale with price. Substituting a blended percentage in that case understates the cost at low prices
and overstates it at high prices, which is precisely the range where the pricing decision lives. The
correct form adds the fixed fee to the per-order cost and keeps the percentage in the factor.

### Step 2 — Contribution margin per unit

```text
FORMULA [M-02]: m(p) = k * p - c
WHERE: m(p) = contribution margin per unit at price p [currency/unit];
       p = published shelf price [currency/unit];
       c = landed cost per unit, effective at the decision date [currency/unit].
CONSTRAINT: c must be the cost payable for the units actually sold in that window; using a standard
            cost that differs from the payable cost by more than the margin makes the sign wrong.
NUMERICAL EXAMPLE: p = 18.00; k = 0.915; c = 9.40;
                  k * p = 0.915 * 18.00 = 16.47;
                  m(18.00) = 16.47 - 9.40 = 7.07 per unit.
INTERPRETATION: at the held price, each unit contributes 7.07 to fixed cost and profit after
                variable cost; the same SKU at 16.20 contributes 0.915*16.20 - 9.40 = 5.42.
```

Read the two numbers together. A 10% price cut reduces the per-unit margin from 7.07 to 5.42, a fall
of 1.65 or 23.3%. That ratio is the entire reason elasticity matters: the seller must find 30.4%
more units to stand still, because 7.07 / 5.42 = 1.3037. Any elasticity estimate below the value
that delivers 1.3037 units of demand ratio for a 0.90 price ratio is an argument against the cut,
and that comparison can be made before any model is fitted.

The break-even condition can be stated as a required elasticity, which is more useful than a margin
ratio because it is directly comparable to the estimate.

### Step 3 — The elasticity required to keep contribution flat

```text
FORMULA [M-03]: e_req = ln[m(p_0)/m(p_1)] / ln[p_0/p_1]
WHERE: e_req = elasticity magnitude required for equal contribution [dimensionless, positive];
       p_0 = held price [currency/unit]; p_1 = candidate price [currency/unit];
       m(.) = contribution margin per unit [currency/unit].
CONSTRAINT: valid for constant-elasticity demand q(p) = A p^(-e); under other forms the requirement
            is a local slope, not a constant, and must be re-derived at the candidate price.
NUMERICAL EXAMPLE: m(18.00) = 7.07; m(16.20) = 5.42;
                  ln(7.07/5.42) = ln(1.3037) = 0.26530;
                  ln(18.00/16.20) = ln(1.11111) = 0.10536;
                  e_req = 0.26530 / 0.10536 = 2.517.
INTERPRETATION: the 10% cut breaks even on contribution only if demand rises by at least the
                elasticity 2.52 allows, i.e. by about 1.11111^2.517 - 1 = 30.4% in units.
```

This is the most useful number in the case, because it converts a modelling question into a
falsifiable commercial claim. The merchant proposing the cut is asserting, implicitly, that this
catalogue's demand elasticity exceeds 2.52 at this price point. That assertion can be tested with
four price cells and a few thousand sessions per arm, and it can be rejected.

### Step 4 — Estimate the elasticity from the price test

The constructed test assigns eligible product-days to four approved price points and records the
conversion rate for each. Four cells allow a curvature check that two cells cannot provide.

| Cell | Price | Conversion | Units per 1,000 sessions |
|---|---:|---:|---:|
| C1 | 16.20 | 0.0519 | 51.9 |
| C2 | 17.10 | 0.0479 | 47.9 |
| C3 | 18.00 | 0.0412 | 41.2 |
| C4 | 18.90 | 0.0352 | 35.2 |

```text
FORMULA [M-04]: ln d_i = a + b * ln p_i + u_i, and e_hat = -b
WHERE: d_i = conversion rate in cell i [orders/session, dimensionless];
       p_i = price in cell i [currency/unit]; b = slope [dimensionless];
       e_hat = estimated elasticity magnitude [dimensionless]; u_i = residual.
CONSTRAINT: b is estimated by ordinary least squares on four points, so the interval uses t with
            4 - 2 = 2 degrees of freedom; with fewer than three distinct prices the interval is
            unbounded in practice and cannot support a decision.
NUMERICAL EXAMPLE: x_i = ln p_i = {2.78501, 2.83908, 2.89037, 2.93916};
                  y_i = ln d_i = {-2.95844, -3.03864, -3.18932, -3.34671};
                  mean x = 2.86341; mean y = -3.13328;
                  S_xx = 0.013204; S_xy = -0.033689;
                  b = -0.033689 / 0.013204 = -2.5515;  e_hat = 2.5515.
INTERPRETATION: a 1% higher price is associated with 2.55% lower conversion inside the tested band.
```

Two checks decide whether that slope can carry a decision. The first is fit. R-squared is 0.974, and
the residuals are +2.55%, -3.20%, -1.27%, +2.03% on the fitted values, so they alternate in sign
rather than tracing a curve. That alternation is the evidence that a constant-elasticity form is not
obviously wrong here; a monotone residual pattern would be evidence of curvature and would send the
analysis to a spline or a quadratic in logs.

The second is precision. The standard error of the slope is 0.2928, so the 95% interval with two
degrees of freedom is -2.5515 ± 4.303 × 0.2928, which is [-3.811, -1.292]. Translating to the
positive elasticity convention, the interval on e is [1.292, 3.811].

That interval is very wide, and the width is the honest headline. The decision boundary derived in
Step 3 is 2.517, which sits *inside* the interval. So the test as constructed cannot tell the
decision maker whether the cut is contribution-positive. This is not a defect of the analysis; it is
the finding, and it is the kind of finding that saves a programme.

The direct two-cell contrast gives a narrower but differently biased estimate. Comparing only C1 and
C3 gives e = ln(0.0519/0.0412) / ln(18.00/16.20) = 0.23088 / 0.10536 = 2.191. It is narrower because
it uses two well-separated cells, and it is less trustworthy because it cannot detect curvature. The
honest report contains both numbers and says which one the decision uses for which purpose: the
four-cell slope for the interval, the two-cell contrast for the point forecast.

### Step 5 — Contribution per session, the correct unit of comparison

Product-day contribution is the wrong unit when arms have different session counts. Sessions differ
because a price change can move click-through as well as conversion, so the comparison must be
normalised before it is aggregated.

```text
FORMULA [M-05]: pi(p) = d(p) * m(p)
WHERE: pi(p) = expected contribution per session at price p [currency/session];
       d(p) = conversion rate [orders/session]; m(p) = margin per unit [currency/order].
CONSTRAINT: one order equals one unit; if basket size varies with price, replace m(p) by the
            average basket contribution and re-check that the basket did not change.
NUMERICAL EXAMPLE: control p = 18.00, d = 0.0412;
                  pi_control = 0.0412 * 7.070 = 0.291284 per session;
                  treatment p = 16.20, d = 0.0519;
                  pi_treatment = 0.0519 * 5.423 = 0.281454 per session;
                  delta = 0.281454 - 0.291284 = -0.009830, i.e. -3.37%.
INTERPRETATION: the cut raises conversion by 26.0% and revenue per session by 13.4%, and lowers
                contribution per session by about 3.4%.
```

The three percentage changes are computed from the same two cells and they point in three different
directions:

```text
revenue per session: 18.00*0.0412 = 0.741600   ->  16.20*0.0519 = 0.840780,  +13.37%
units per session:   0.0412        = 0.041200   ->  0.0519        = 0.051900,  +25.97%
contribution/sess:   0.291284                   ->  0.281454,                -3.37%
```

This is the case's central mechanism. Any dashboard built on revenue would have reported a win.
The programme would have scaled a change that loses money, and the loss would have been invisible
until a finance reconciliation months later, by which time the price would have been treated as
validated.

### Step 6 — Add the per-order variable cost

The margin in Step 2 excludes picking and packaging, which is variable per order and does not scale
with price. Adding it changes both the level and the ranking.

```text
FORMULA [M-06]: pi_net(p) = d(p) * [m(p) - w]
WHERE: w = variable per-order fulfilment cost [currency/order], for example pick, pack and label;
       pi_net(p) = contribution per session after per-order cost [currency/session].
CONSTRAINT: w must be per order and must not vary with price; if packaging is charged as a percentage
            of order value it belongs in k, not here.
NUMERICAL EXAMPLE: w = 1.10;
                  net margin at 18.00 = 7.070 - 1.10 = 5.970;
                  net margin at 16.20 = 5.423 - 1.10 = 4.323;
                  pi_net_control = 0.0412 * 5.970 = 0.245964;
                  pi_net_treatment = 0.0519 * 4.323 = 0.224364;
                  delta = -0.021600, i.e. -8.78%.
INTERPRETATION: the per-order cost does not change the sign of the comparison, it widens the loss,
                because the treatment arm has 26% more orders each carrying the same fixed cost.
```

The required conversion rate is the operationally meaningful number, because it can be compared
against the observed lift. Holding the control's net contribution per session fixed at 0.245964 and
solving for the treatment conversion gives 0.245964 / 4.323 = 0.056897, or 5.690%. Against a control
of 4.12% that is a relative uplift of 38.10%, versus the 26.0% actually observed. The gap between
26.0% and 38.1% is the size of the argument, and it is small enough to be closed by a different
assumption about w — which is why w must be pinned by finance before the test, not after.

The required elasticity under this cost basis is a third log ratio, and it is worth computing because
it shows how much the per-order cost hardens the threshold: ln(5.970/4.323) = ln(1.38100) = 0.32281,
divided by ln(1.11111) = 0.10536, gives 3.064. The per-order cost raises the bar from 2.52 to 3.06
because it compresses the margin on both arms, and it compresses the treatment arm proportionally
more.

### Step 7 — The optimal price under constant elasticity

```text
FORMULA [M-07]: p* = e * c / [(e - 1) * k]
WHERE: p* = contribution-maximising price [currency/unit]; e = elasticity magnitude (> 1);
       c = landed cost [currency/unit]; k = retention factor [dimensionless].
CONSTRAINT: requires e > 1 strictly. At e = 1 the denominator vanishes; below e = 1 the expression is
            negative and the true optimum is unbounded, which is a statement about the model, not
            about the market.
NUMERICAL EXAMPLE: e = 2.191, c = 9.40, k = 0.915;
                  numerator = 2.191 * 9.40 = 20.595;
                  denominator = (2.191 - 1) * 0.915 = 1.191 * 0.915 = 1.08977;
                  p* = 20.595 / 1.08977 = 18.90.
INTERPRETATION: with the two-cell elasticity, the contribution-maximising price is 18.90, not the
                16.20 the merchant proposed; the direction of the recommendation is opposite.
```

The derivation is short enough to state, because knowing where p* comes from is what tells you when
it will betray you. Contribution per unit of demand is k*p - c, and demand is A p^(-e). Total
contribution is A p^(-e)(k p - c). Differentiating with respect to p and setting to zero gives
A[-e p^(-e-1)(k p - c) + k p^(-e)] = 0, so k p = e(k p - c), so k p (e - 1) = e c, and p* follows.
Every step assumes e is constant in p, which is the assumption the four-cell test is designed to
probe.

The flatness of the objective around p* is the most important practical property. Evaluating the
constructed demand curve at seven prices gives:

| Price | Units per day | Margin per unit | Contribution | Versus 18.00 |
|---:|---:|---:|---:|---:|
| 16.20 | 1,259.71 | 5.4230 | 6,831.40 | -3.37% |
| 17.00 | 1,133.44 | 6.1550 | 6,976.30 | -1.33% |
| 18.00 | 1,000.00 | 7.0700 | 7,070.00 | baseline |
| 18.90 | 898.60 | 7.8935 | 7,093.11 | +0.33% |
| 20.00 | 793.83 | 8.9000 | 7,065.13 | -0.07% |
| 21.00 | 713.34 | 9.8150 | 7,001.44 | -0.97% |
| 22.00 | 644.21 | 10.7300 | 6,912.33 | -2.23% |

Between 17.00 and 22.00 the contribution varies by 2.21% end to end (6,912.33 to 7,065.13) while the
price varies by 29.4%. The objective is nearly flat over that range, while the uncertainty in the
elasticity is wide enough to move the implied optimum from 13.93 to 45.49. The correct conclusion is
not "raise price to 18.90"; it is "the data cannot distinguish these prices, so choose on
non-contribution grounds — customer perception, competitive position, and the risk of being wrong in
the expensive direction."

### Step 8 — The binding inventory constraint

When supply is limited, the objective changes: contribution per available unit, not per unit of
demand. The unconstrained optimum sells more units than exist.

```text
FORMULA [M-08]: p_cap = p_0 * (Q / q_0)^(-1/e), valid when q(p_cap) = Q
WHERE: p_cap = price at which predicted demand equals the available quantity [currency/unit];
       Q = sellable units available for the day [units]; q_0 = demand at p_0 [units].
CONSTRAINT: requires Q < q(p*) so the cap binds; if Q >= q(p*) the constraint is inactive and the
            answer is p* from M-07. Requires e estimated in the same range, not extrapolated.
NUMERICAL EXAMPLE: Q = 850 units; p_0 = 18.00; q_0 = 1,000; e = 2.191;
                  Q/q_0 = 0.850; ln(0.850) = -0.162519;
                  (-1/2.191) * (-0.162519) = 0.074164;  exp(0.074164) = 1.076983;
                  p_cap = 18.00 * 1.076983 = 19.3857;
                  margin at 19.3857 = 0.915*19.3857 - 9.40 = 17.7379 - 9.40 = 8.3379;
                  contribution = 850 * 8.3379 = 7,087.23.
INTERPRETATION: with 850 units to sell, 19.3857 yields 7,087.23 against 850 * 7.070 = 6,009.50 at
                the held price, an improvement of 1,077.73 or 17.93%, and it is far larger than any
                price effect the demand model could resolve.
```

The cap result dominates the elasticity question. The value of moving from 18.00 to 19.39 comes from
selling the same 850 units at a higher margin, and it is worth almost 18% against a price-optimisation
gain of 0.33% with full supply. That asymmetry is the practical lesson: in a supply-constrained
catalogue, scarcity pricing is worth more than demand-response accuracy, and the team's effort should
follow the money.

Note what the constraint does to the 10% price-move policy. A move from 18.00 to 19.39 is +7.7%, inside
the limit. A month of such moves would accumulate, because the limit is measured against the last
published price, not against an approved reference. A programme that repeatedly applies the cap will
walk the price upward in 7.7% steps, and the guardrail as written will not stop it. The policy needs a
reference-price anchor, not a step limit, and that is an organisational fix that no model can supply.

### Boundary case — where the formula breaks

The constant-elasticity optimum is not robust near e = 1, and the failure is instructive.

```text
FORMULA [M-07] LIMIT CHECK: p* = e c / [(e-1) k] as e -> 1+
NUMERICAL EXAMPLE: e = 1.05; c = 9.40; k = 0.915;
                  p* = 1.05*9.40 / (0.05*0.915) = 9.87 / 0.045750 = 215.74.
INTERPRETATION: a plausible-looking elasticity estimate produces a price 12 times the held price.
                The formula is arithmetically correct and commercially absurd, which means the input
                is being used outside its range, not that the price is an opportunity.
```

Three constraints bound the answer and none of them is inside the formula. Price must exceed the
zero-margin price c/k = 9.40/0.915 = 10.27, or the seller loses money on every unit. Price must
exceed the zero-net-margin price (c + w)/k = 10.50/0.915 = 11.47 once the per-order cost is included.
And price must lie inside the approved floor and ceiling, with a move of no more than 10% from the
published price. Those constraints live outside the optimisation, and the model that ignores them
will produce recommendations that are correct and unactionable.

The deeper boundary is the identification boundary. The four-cell test spans 16.20 to 18.90, a range
of ±8%. The elasticity 2.55 estimated there is being used in Step 8 to justify 19.39, which is 15%
above the top of the tested range, and in the limit check to justify 215.74, which is twelve times
it. Extrapolating a constant elasticity outside the tested band is the single most common way this
arithmetic is abused in practice, and the abuse is invisible because the numbers still compute.

**What the walkthrough establishes.** The planned cut is contribution-negative under every cost
definition tried, by 3.37% or 8.78% depending on whether the per-order cost is included. The elasticity
interval is too wide to resolve the question the decision asks. The scarcity case is worth 17.9% and
is robust to the elasticity input. The objective is flat enough that price choice within ±10% is a
judgement call, not an optimisation result. Any of these findings would be reportable on its own; the
combination means the programme should not be launched as proposed.

## 7. The Finding

Do not authorise the proposed blanket 10% price reduction. Do not authorise model-driven pricing
either, on the evidence available. Authorise a bounded scarcity rule for the subset of SKUs whose
sellable stock is below the cover threshold, and commission a properly powered price test for the
remainder.

The constructed evidence supports three separate statements, and they have different strengths.

**Statement one, arithmetic and firm.** Under the stated cost basis, a 10% cut requires an elasticity
of at least 2.52 to hold contribution constant, and of at least 3.06 once the per-order cost is
included. The observed lift implies 2.19. The cut is contribution-negative by 3.37% before the
per-order cost and by 8.78% after it. No modelling assumption enters this statement; it is division.

**Statement two, estimate and uncertain.** The elasticity estimated from four price cells is 2.55 with
a 95% interval of [1.29, 3.81]. The interval contains the decision boundary. On this evidence the
pricing team cannot say whether repricing in either direction improves contribution.

**Statement three, structural and robust.** When sellable stock is below predicted demand, the price
that clears the available units at full margin beats the unconstrained optimum by a wide margin
(17.93% over the held price in the worked example), and that conclusion depends only on the direction
of demand, not on the magnitude of the elasticity.

The finding therefore separates the question into one part that can be acted on now and one part that
requires more evidence. Acting on the whole proposal would overreach; deferring everything would
leave 17.93% of contribution uncaptured on the constrained subset for a quarter.

The recommendation carries a fourth statement about what was not measured. No baseline replay was
executed, no model was trained, no capacity study exists, no customer-response study on the price
change exists, and no competitive reaction was observed. The arithmetic is a worksheet built on
constructed inputs, and every number above would change if the inputs did.

## 8. System/Solution Design

The design has two paths that must be reasoned about separately, because they fail differently and
are owned by different teams. The batch path produces a recommendation and can be wrong safely. The
publication path touches customers and can be wrong dangerously.

### 8a. Offline batch path

The job runs once per trading day, after the 02:00 cut-off, and is idempotent: re-running it for the
same day and policy version must produce the same recommendation table, and the table is keyed by
(product, market, day, policy_version) with a uniqueness assertion.

Stage one is input assembly. The job reads the eligible-SKU dimension effective at the decision date,
the inventory snapshot stamped at or before the cut-off, the effective-dated cost, the fee schedule
version, and the constraint set version. Every read is versioned and the version identifiers are
written into the output row. A missing input fails the run for the affected SKUs only; it never
substitutes a default, because a default cost silently prices inventory at zero.

Stage two is candidate generation. For each SKU the job enumerates the approved grid, which is the
held price plus four offsets within the 10% move limit, filtered by the contractual floor and ceiling
and by the safety-reserve rule for markdowns. The candidate set is written out, including the
filtered candidates and their reason codes, because an empty feasible set is a finding that must be
visible rather than inferred.

Stage three is scoring. For each feasible candidate the job computes predicted demand, predicted
contribution, and the interval. The interval is propagated, not dropped: the output row carries the
point estimate and the lower bound, and the decision rule uses the lower bound.

Stage four is selection under the decision rule: choose the candidate with the highest lower bound;
if the held price's lower bound is not exceeded by at least one grid step of value, recommend the
held price. This rule is deliberately conservative and it converts the wide uncertainty into a
no-change recommendation automatically, which is the correct behaviour for a programme that cannot
yet resolve the question.

Stage five is write-out: a recommendation table, an exclusion report by reason, and a summary that a
merchant can read in thirty seconds. The merchant does not read the model. The merchant reads the
recommendation, the reason code, and the interval.

### 8b. Online publication path

Publication is a separate service with its own controls. It accepts an approved artefact, validates
that the artefact's constraint check passed at the same version as the recommendation, writes the
price to the catalogue, and invalidates the cache for the affected product pages.

Three properties matter and each has a specific test. *Atomicity across surfaces:* the listing page,
the product page, the cart and the checkout must serve the same price. The test is a synthetic
purchase that records all four values and fails the publication if any differ. *Immutability of
accepted orders:* an order placed before a price change keeps its price; the test is to place an
order, publish a change, and complete checkout, then reconcile the receipt. *Bounded blast radius:*
publication is chunked by category with a kill switch per chunk, so a defect found in one category
does not require reverting the whole catalogue.

The service refuses to publish any version that has not passed the constraint check, and it refuses
to publish a price more than 10% away from the last published price regardless of what the
recommendation says. That duplication is intentional. The constraint check in the batch job protects
against a modelling error; the check in the publication service protects against a plumbing error
that bypasses the batch job.

### 8c. Key design decisions

| Decision | Alternative rejected | Reason |
|---|---|---|
| Recommend, never auto-publish, in the pilot | Full automation from day one | An unreviewed price error is customer-visible within minutes and unrecoverable |
| Interval-based selection | Point-estimate argmax | The interval is what makes the programme refuse to act on noise |
| Write filtered candidates to the output | Write only the winner | An empty feasible set must be observable, not inferred from a missing row |
| Duplicate the constraint check in publication | Trust the batch job | Separates modelling error from plumbing error |
| Version-bind model, cost basis and constraint set | Version the model alone | A recommendation without its cost basis cannot be reproduced or defended |
| Exclude SKUs with cost disputes rather than defaulting | Substitute a category-average cost | A synthetic cost produces a profitable-looking recommendation on a loss-making SKU |
| Chunk publication by category | Publish the whole catalogue at once | Limits rollback cost and makes the incident review tractable |

### 8d. Operational readiness

Five controls gate the first publication. Each has an owner and a test, and the pilot does not start
until all five are recorded as passed.

The *rollback* control requires that a single command restores the previous published price for a
category within a stated time, and that the restore has been rehearsed on the staging catalogue.
The *audit* control requires that every published price is traceable to a recommendation row, a
policy version, an approver identity and a timestamp. The *exclusion* control requires a daily count
of SKUs that were eligible yesterday and are not eligible today, with a reason. The *guardrail*
control requires that the customer-complaint rate and the price-change frequency are both visible
daily, not weekly. The *kill switch* control requires a named on-call owner who can stop publication
without a code change.

None of these is a model control. That is the point: the model's failure modes are mostly absorbed by
review, while the system's failure modes are absorbed only by these five.

## 9. Evaluation & Validation

The evaluation must be specified before the pilot, because a metric chosen after the outcome is known
is not evidence.

**Numerator.** Incremental realised contribution: the sum over eligible product-days of realised
contribution under the new policy minus the counterfactual contribution those same product-days would
have produced under the held-price baseline, net of the cost of running the policy.

**Denominator.** Eligible product-days in the evaluation window, so the headline metric is
contribution per eligible product-day. The per-day normalisation prevents a policy that expands
eligibility from looking good purely by counting more days.

**Population.** Products eligible on day one and still eligible at the end of the window. Products
that become ineligible are reported separately with reasons; they are not silently dropped, because
dropping them is equivalent to conditioning on a post-treatment variable.

**Window.** A four-week evaluation window after a two-week burn-in. The burn-in exists because the
first days of a new policy produce merchant override behaviour that is not representative. The window
is fixed in advance and is not extended because the result is close to the threshold.

**Comparator.** Two comparators, both required. The first is the held price on the same product-days,
reconstructed from the audit log. The second is the randomised control arm, for the subset of
product-days that were assigned to a control price. The first answers "did we beat what we would
have done"; the second answers "did the policy cause the difference". Reporting only the first is the
standard error in this domain.

**Guardrails, each with a threshold fixed in advance.** Customer complaint rate attributable to price
display, per 10,000 orders. Share of product-days where the price moved twice within seven days.
Share of SKUs whose contribution fell by more than 20% against the held price. Count of publication
incidents where the cart price differed from the listing price, which has a threshold of zero.

**Uncertainty.** Every reported metric carries an interval. Differences are reported as intervals
from a cluster bootstrap at the product level, since product-days within a product are strongly
dependent. A point estimate without an interval is not reportable, because the decision rule in
section 8a depends on the lower bound.

**Executed or not executed, stated plainly.** No baseline replay was executed. No model was trained.
No pilot was run. No interval was computed from real data. The arithmetic in section 6 is closed-form
on constructed inputs, and the four-cell elasticity table is similarly constructed. This is the
honest status: designed and computed, not executed and observed.

**Why the executed/not-executed distinction matters here.** A reader who skims section 6 could take
away "the pilot returned 2.55 elasticity and 0.33% improvement". Both numbers were produced by
arithmetic in this document, not by a business. The distinction is not a legal nicety; it changes
what a reader should do next, because an observed 2.55 with a wide interval calls for a bigger test,
while a constructed 2.55 calls for a first test.

Four validation checks would be required before any number in section 6 could be treated as evidence.

*A/A test.* Split eligible product-days at random into two groups, apply the held price to both, and
confirm the estimated difference is centred on zero with the predicted interval width. A non-zero
centre indicates a pipeline defect, not a real effect, and the study stops.

*Placebo price.* Assign a price change that the rules service filters out at publication, so the
customer sees no change. Any measured difference is a pipeline artefact and bounds the minimum
detectable effect.

*Negative control outcome.* Track a metric that the policy cannot affect, such as the share of
sessions from a paid channel with a fixed budget. A change in it indicates that assignment altered
traffic composition rather than price response.

*Reconciliation.* Sum realised contribution from the ledger and compare against the sum computed by
the evaluation pipeline for the same product-days. A gap above a pre-agreed tolerance invalidates the
evaluation, and the gap is reported rather than adjusted away.

The power arithmetic from Step 8 is part of the validation design, because a test that cannot resolve
the decision boundary is not a cheap test, it is a wasted one.

| Design | Unit of randomisation | Design effect | Effective sessions | MDE at 4.5% base |
|---|---|---:|---:|---:|
| Session-level, naive | Session | 1 | 6,084 per arm | 1.05 pp |
| Market-level, 6 markets | Market-week | 201.0 | 3,980 | 1.30 pp |
| Switchback, 48 blocks | Market-block | 26.0 | 30,771 | 0.47 pp |
| Market-level, ICC 0.005 | Market-week | 667.7 | 1,198 | 2.37 pp |

The table is the argument for the switchback design. Session-level randomisation is not implementable
for a *shelf* price, because all customers in a market see the same price; the market-level design is
the naive implementable choice and its design effect of 201 destroys the test. Alternating the price
by block within market recovers most of the information, at the cost of requiring the blocks to be
time-balanced and the analysis to model the block structure.

## 10. Action Taken

The decision record, written for this teaching case, records four actions and two refusals.

**Action one: reject the blanket cut.** The pricing director rejects the proposed 10% reduction on
the grounds that it requires an elasticity of 2.52 and the observed response implies 2.19. The
rejection is documented with the arithmetic, so a future reader can see it was a calculation and not
a preference.

**Action two: authorise the scarcity rule for the constrained subset.** For SKUs whose sellable stock
is below the 15-day cover threshold and whose predicted demand exceeds available units, the batch job
may recommend a price up to the level that clears the stock, subject to the contractual ceiling and
the 10% move limit. The recommendation is reviewed by the merchant before publication in the first
four weeks and published directly thereafter if the guardrails hold. This subset is where the value
is, and the value does not depend on the elasticity magnitude.

**Action three: commission a properly powered price test.** The test uses the switchback design, four
price cells inside ±10%, a pre-registered primary metric of contribution per session, and the
decision rule fixed in advance: adopt the candidate only if the interval on the difference excludes
zero. Duration is set from the power calculation, not from the calendar.

**Action four: fix the guardrail definition.** The move limit is replaced by an anchor rule, so the
published price cannot drift more than 10% from an approved quarterly reference, regardless of how
many small moves occur. This closes the walk described in Step 8.

**Refusal one.** The team refuses to approve a revenue-based success metric, on the grounds that the
worked example shows revenue and contribution moving in opposite directions from the same data. The
refusal is recorded because the merchant organisation requested the revenue metric.

**Refusal two.** The team refuses to include the personalised-pricing variant in the pilot scope,
because C3 is not yet confirmed and a personalised price cannot be recalled once shown. Deferring
that variant costs nothing this quarter, because the value identified sits in the scarcity subset.

Two things were explicitly not done. The team did not run the cut for a week "to see what happens",
because the arithmetic already answers that question and the answer is a loss of about 0.01 per
session at the assumed traffic. And the team did not relax the 10% move limit to let the model reach
its preferred price, because the limit protects customers and the model has not earned the right to
override a customer protection.

## 11. Outcome & Results

There is no business outcome to report. The programme was not launched, no revenue was earned, no
contribution was realised, and no customer responded to anything. Every figure in this section is a
constructed projection from the section 6 arithmetic, labelled **ILLUSTRATIVE**, and exists to show
the scale of the decisions, not to claim a result.

| Quantity | Constructed value | Basis | Status |
|---|---:|---|---|
| Contribution margin at held price | 7.07 per unit | 0.915*18.00 - 9.40 | arithmetic |
| Contribution margin at -10% price | 5.42 per unit | 0.915*16.20 - 9.40 | arithmetic |
| Required elasticity for the cut | 2.52 | ln(7.07/5.42)/ln(18.00/16.20) | arithmetic |
| Observed elasticity, two cells | 2.19 | ln(0.0519/0.0412)/ln(18.00/16.20) | constructed input |
| Observed elasticity, four cells | 2.55 | OLS on four log points | constructed input |
| 95% interval on four-cell elasticity | [1.29, 3.81] | t(2) x standard error | arithmetic |
| Contribution change per session | -3.37% | 0.281454 vs 0.291284 | arithmetic |
| Contribution change with order cost | -8.78% | 0.224364 vs 0.245964 | arithmetic |
| Annualised at 200,000 sessions/week | -102,235 | -0.009830 * 200,000 * 52 | projection |
| Scarcity price uplift at 850 units | +17.93% | 8.3379 vs 7.0700 per unit | arithmetic |
| Sessions per arm for the planned test | 6,084 | two-proportion power, 80%, alpha 0.05 | arithmetic |
| MDE under the switchback design | 0.47 pp | effective n = 30,771 | arithmetic |

The annualised row is the one that would be quoted in a steering committee, and it is the one most
likely to be wrong. It multiplies a constructed per-session difference by a constructed traffic
figure and assumes 52 identical weeks. Real traffic is seasonal, real SKUs differ, and real
elasticities vary across the catalogue. The row is included to show the order of magnitude that makes
the decision consequential, not as a forecast.

Three sensitivities would change the sign or the size of the projection, and each is worth stating
because a decision maker needs to know what would flip it.

If the per-order cost w is 0 instead of 1.10, the treatment is still negative but by 3.37% instead of
8.78%. The sign survives, so the rejection of the cut is robust to this input. If the elasticity is at
the upper end of the interval, 3.81, the cut becomes contribution-positive and the rejection would be
wrong; the rejection is therefore a bet on the point estimate, and should be revisited as soon as the
interval narrows. If the inventory cap applies to most of the catalogue rather than a subset, the
scarcity rule dominates everything else and the elasticity question becomes second-order.

The one number that would most change the decision is not in the table. It is the share of eligible
SKUs whose sellable stock falls below demand. If that share is small, the scarcity rule captures a
small fraction of the value and the elasticity problem needs solving. If it is large, the programme
should be built around supply and the demand model is a secondary component. That share is not known
here, and it is the first thing a real programme should measure.

## 12. Failure Modes & What Went Wrong

This section catalogues the failure mechanisms that this decision is actually exposed to. Each entry
states the symptom an operator would observe, the competing causes that produce the same symptom, the
check that discriminates between them, and the correction. A mechanism without a discriminating check
is not a failure mode, it is a worry.

### F-01 — Promotion calendar confounds the price coefficient

*Symptom.* The observational regression returns an elasticity of 0.4 on a category where the merchant
insists demand is elastic, and adding a price-squared term changes the sign.

*Competing causes.* (a) A real, inelastic category; (b) promotional activity correlated with price, so
the price coefficient absorbs the promotion; (c) insufficient price variation, so the coefficient is
estimated from noise.

*Discriminating check.* Regress on the randomised cell only, where promotion is held fixed by design,
and compare with the observational coefficient on the same products and weeks. A large divergence
identifies (b). A wide interval on both identifies (c). A stable near-zero coefficient on the
randomised cells supports (a).

*Correction.* Estimate response from the experimental contrast, and use the observational model only
where no experiment exists, with an explicit bias bound.

*Prevention.* Log the promotion flag as a covariate at assignment time and forbid any analysis of
price response that omits it.

### F-02 — Stockout censoring is read as low demand

*Symptom.* A SKU sells out at 11:00 and its measured conversion is a third of the category average.
The model learns "high price, low demand" from a day on which the true statement is "high demand,
no stock".

*Competing causes.* (a) Genuine low demand; (b) stockout censoring; (c) a listing or search-placement
defect that suppressed impressions.

*Discriminating check.* Join the stockout timestamp and recompute conversion on sessions before the
stockout. If the pre-stockout conversion is close to the category average, (b) is confirmed. If the
impression count is also low, (c) is also present and must be excluded before (b) is sized.

*Correction.* Exclude product-days with a stockout inside the trading window, or fit a censored model
that treats the stockout as an unobserved upper bound on demand.

*Prevention.* Require the stockout timestamp as a non-null field on every product-day row, and fail
the build when it is missing on a day with zero fulfillable units.

### F-03 — Landed cost drifts between recommendation and ledger

*Symptom.* The batch job reports a 7.07 margin and the finance reconciliation reports 6.52. The gap
is 8% and it appears on the SKUs with the highest volume.

*Competing causes.* (a) A vendor price increase that took effect after the cost record was read;
(b) a currency revaluation on imported goods; (c) a freight surcharge applied at receipt but not at
the unit cost.

*Discriminating check.* Compare the cost version used in the recommendation against the effective
cost on the order date for the same SKUs. A version mismatch identifies (a). A stable mismatch for
imports only identifies (b). A mismatch that scales with receipt volume identifies (c).

*Correction.* Recompute the recommendation on the correct cost version and re-run the evaluation;
never adjust the evaluation to match the ledger.

*Prevention.* Bind the cost version to the recommendation row and assert at evaluation time that the
version matches the order date's effective cost.

### F-04 — The feasible set is empty and the failure is invisible

*Symptom.* A category shows no price changes for three weeks during a cost shock. Nobody notices,
because "no recommendation" looks identical to "recommendation equals the held price".

*Competing causes.* (a) Every candidate filtered by the move limit; (b) every candidate filtered by
the contractual floor; (c) an upstream failure that removed the SKU from eligibility.

*Discriminating check.* Count recommendation rows by reason code per day. A spike in move-limit
filtering identifies (a), a spike in floor filtering identifies (b), and a drop in eligible-SKU count
identifies (c).

*Correction.* Publish the held price and escalate the empty set to the merchant with the binding
constraint named, so a human can decide whether the constraint or the price is wrong.

*Prevention.* Make the filtered-candidate table a required output and alert on the empty-set rate
exceeding a threshold.

### F-05 — Interference through substitution

*Symptom.* A price increase on product A raises A's contribution and lowers B's, where B is A's
substitute, and the category total is flat or negative.

*Competing causes.* (a) True substitution; (b) a shared promotion that ended in the same week;
(c) a shared stockout that suppressed both.

*Discriminating check.* Randomise the price change at the category level in some markets and the SKU
level in others, then compare the category total contribution under both. A difference attributable
to the randomisation unit is (a). A contemporaneous promotion or stockout on both products points to
(b) or (c).

*Correction.* Evaluate at the category level when substitution is material, and set prices jointly
rather than per SKU.

*Prevention.* Maintain an effective-dated substitute map and declare the evaluation unit from it
before the test, not after.

### F-06 — Revenue substitutes for contribution in the success metric

*Symptom.* The pilot is declared a success on a 13.37% revenue-per-session improvement while
contribution per session fell 3.37%.

*Competing causes.* (a) The metric was never defined precisely; (b) the metric was defined but the
dashboard computed revenue; (c) contribution was computed before the per-order cost was added.

*Discriminating check.* Recompute both metrics from the same two cells. If revenue rises and
contribution falls, (a) or (b) is present and the direction of the difference identifies which. If
contribution is positive only in the version without the per-order cost, (c) is present.

*Correction.* Restate the primary metric as contribution per session with the cost basis named in the
definition, and re-issue the pilot verdict.

*Prevention.* Put the metric definition, its numerator, denominator and cost basis in the
pre-registration document, signed before the test starts.

### F-07 — Elasticity is extrapolated outside the tested range

*Symptom.* A model trained on prices between 16.20 and 18.90 recommends 24.00 for a slow-moving SKU,
and the recommendation is accepted because the arithmetic is internally consistent.

*Competing causes.* (a) Extrapolation of a constant elasticity; (b) a genuine regime change, such as
a competitor exiting; (c) a cost increase that legitimately raises the optimal price.

*Discriminating check.* Plot the fitted demand against the observed cells and flag any recommendation
more than one grid step outside the tested band. If the flag fires and no competitor or cost event is
recorded, (a) is present.

*Correction.* Cap the recommendation at the tested band plus one step, and require an explicit
merchant approval for anything outside it.

*Prevention.* Encode the tested band as a constraint, not as a comment, so the model cannot recommend
outside it without a recorded override.

### F-08 — Publication plumbing serves inconsistent prices

*Symptom.* The listing page shows 16.20 and the cart charges 18.00. Complaints arrive before the
monitoring dashboard updates, which is the worst possible order.

*Competing causes.* (a) Cache not invalidated on one surface; (b) two publication versions live at
once during a chunked rollout; (c) the cart reading a price snapshot rather than the live catalogue.

*Discriminating check.* Run the synthetic purchase journey against the affected product and record
all four prices with their serving timestamps and version identifiers. A single stale surface
identifies (a); two different version identifiers identify (b); a cart timestamp older than the
publication identifies (c).

*Correction.* Roll back the affected chunk, then re-publish after the surface is fixed. Do not
"re-publish and hope", because the stale surface will serve the same stale value.

*Prevention.* Make the four-surface consistency check a publication gate with a zero-incident
threshold, and alert on version divergence rather than on complaint volume.

### F-09 — Override behaviour is mistaken for model error

*Symptom.* Merchant overrides run at 40% of SKUs in the first weeks, and the measured policy effect is
small, which is read as "the model is weak".

*Competing causes.* (a) The model's recommendations are genuinely poor; (b) merchants override on
SKUs where they have private information the model lacks, such as a vendor negotiation; (c) the
override rule itself changed during the window.

*Discriminating check.* Compare realised contribution on overridden versus non-overridden product-days
at the same predicted margin. If overrides outperform, (b) is present and the model is missing a
feature. If overrides underperform, (a) is supported. A step change in the override rate at a date
identifies (c).

*Correction.* Analyse the pilot on an intention-to-treat basis, using the recommendation rather than
the published price as the assigned treatment, and report the per-protocol result separately.

*Prevention.* Log every override with an actor, a reason code and a timestamp, and report the override
rate as a first-class metric.

## 13. Transferable Lesson

The pattern is **causal confounding dressed as a measurement problem**. A team is asked to optimise a
number, the number is easy to compute from history, and the historical variation that would identify
the mechanism is entangled with the reasons the mechanism was invoked. The trap is that everything
computes. The elasticity, the optimum, the projected saving and the confidence interval are all
correct arithmetic on an input that does not mean what it appears to mean.

Four habits transfer to any decision of this shape.

*Convert the proposal into a falsifiable threshold before modelling anything.* "The cut requires an
elasticity of 2.52" took two logarithms and no data, and it immediately reframed the conversation
from "build a model" to "can we measure 2.52". This is the highest-leverage hour in the project.

*Normalise to the resource that is actually scarce.* The interesting number was not the optimal price
but the contribution per available unit, and once the inventory cap was recognised the answer changed
from "cut" to "raise". Every pricing, staffing and capacity decision has this structure: the objective
is per unit of a constrained resource, and the constraint, not the model, usually determines the
direction.

*Report the interval, and let it decide.* A point estimate of 2.55 against a boundary of 2.52 is not a
result, it is a coin flip with a decimal point. The interval said so, the decision rule respected it,
and the programme deferred rather than shipped.

*Distinguish arithmetic from evidence in the document itself.* Every number here is checkable; none
of it is evidence. Writing the two categories separately is what allows a reader to trust the
arithmetic without being misled about the world, and it is the difference between a worksheet and a
false claim.

The specific numbers are disposable. The structure — threshold first, scarce resource second, interval
third, evidence label fourth — is the transferable part.

## 14. Connections & Cross-References

**Sibling cases in this collection.** The airline and hotel revenue-management case
(02_Airline_And_Hotel_Revenue_Management.md (planned in `INDEX.md`; not yet authored)) shares
this case's core object — a price or a booking limit set against uncertain demand — and differs in
that its capacity is fixed per departure or per night, so the constraint binds every day rather than
on a subset. Reading the two together separates the elasticity question from the allocation question.

**Patterns.** The primary pattern is causal confounding: the price coefficient is identified from
variation that was not randomly assigned. The secondary patterns are evaluation mismatch, because
revenue and contribution diverge in sign on the same data, and constraint violation, because the
objective's optimum is regularly outside the feasible set.

**Prerequisites a reader should have.** Contribution margin arithmetic; the log-log demand form and
its elasticity; two-proportion power calculation; cluster randomisation and design effect; the
distinction between event time and processing time.

**What this case deliberately does not cover.** Multi-product joint optimisation with a substitution
matrix, competitive response, markdown optimisation over a season, auctions, and regulated pricing.
Each is a separate decision with a separate owner, and the unit's index carries them as planned
commitments rather than compressing them here.

**Where a reader should go next inside this unit.** The revenue-management case for capacity
allocation; the experimentation sheet for the power and design-effect arithmetic used in section 9;
the common-issues file for the numerical traps that recur across pricing cases.

### Interview pack

#### Q01 — A merchant proposes a 10% cut and reports a 26% conversion lift. Is that good?

**Answer.** Not necessarily, and on the constructed numbers in this case it is bad. Contribution per
session falls by about 3.37%, or 8.78% once a per-order cost is included.

**Reasoning.** Conversion is not the objective, contribution is. A price cut reduces the per-unit
margin, so the volume gain must exceed the margin loss. The margin ratio is 7.07/5.42 = 1.3037, so
the cut needs 30.4% more units; a 26% conversion lift delivers less than that, and the shortfall
widens once per-order costs are counted.

**Follow-up.** What conversion lift would make the cut contribution-neutral?

**Red flag.** Accepting the lift as evidence because the number is large and the direction is
intuitive.

**Scoring.** 3 points for naming contribution as the objective and computing the margin ratio;
2 more for the 30.4% requirement; 1 more for noticing that per-order cost moves the threshold.

#### Q02 — Compute the elasticity implied by a move from 18.00 with 4.12% conversion to 16.20 with 5.19%.

**Answer.** 2.19, using the log-log (constant-elasticity) form.

**Reasoning.** Price ratio 16.20/18.00 = 0.9, ln = -0.10536. Conversion ratio 5.19/4.12 = 1.2597,
ln = 0.23088. Elasticity is minus the ratio of the log changes: 0.23088/0.10536 = 2.191. The sign
convention is that elasticity is reported as a positive magnitude, so the answer is 2.19.

**Follow-up.** Why is this estimate less trustworthy than a four-cell OLS estimate?

**Red flag.** Using simple percentage ratios instead of log ratios, which gives 2.597 and overstates
the response.

**Scoring.** 3 points for the correct log computation; 1 for the sign convention; 2 for explaining
why two cells cannot detect curvature.

#### Q03 — Derive the optimal price for constant-elasticity demand.

**Answer.** p* = e·c / [(e−1)·k], with e > 1.

**Reasoning.** Contribution is (k p − c)·A p^(−e). Differentiate: A[−e p^(−e−1)(k p − c) + k p^(−e)]
= 0, so k p = e(k p − c), giving k p(e − 1) = e c. With e = 2.191, c = 9.40 and k = 0.915, p* =
20.595/1.08977 = 18.90.

**Follow-up.** What happens to p* as e approaches 1 from above, and what does that tell you about the
formula's range?

**Red flag.** Quoting the formula without checking e > 1, then applying it to an e of 1.05.

**Scoring.** 3 points for the derivation; 2 for the numeric answer; 2 for the e > 1 condition and the
divergence at the boundary.

#### Q04 — Is a shelf price the same as a booking limit? Where does the analogy break?

**Answer.** The analogy holds at the level of allocating a scarce unit; it breaks on the direction of
the control and on the observability of the constraint.

**Reasoning.** Both decisions choose how much of a fixed capacity to sell at a low price and how much
to reserve for a high one. With a shelf price the seller chooses a number that indirectly determines
quantity, and demand may exceed or fall short. With a booking limit the seller chooses quantity
directly and the price is fixed per class. The break matters because a shelf price cannot be revised
per customer, while a booking limit can be revised per booking, and because shelf-price demand is
censored by stockouts that the seller caused.

**Follow-up.** Which control would you prefer if you could forecast demand well but could not measure
elasticity?

**Red Flag.** Asserting they are the same problem because both are "revenue management".

**Scoring.** 2 points for the shared structure; 2 for the control-variable difference; 2 for the
censoring consequence; 1 for the preference argument.

#### Q05 — Compute the contribution of selling 850 units at 19.39 versus 850 at 18.00.

**Answer.** 7,087.23 versus 6,009.50, an improvement of 1,077.73 or 17.93%.

**Reasoning.** Margin at 19.39 is 0.915×19.39 − 9.40 = 17.742 − 9.40 = 8.342. Times 850 = 7,090.70
before rounding the price. At 18.00 the margin is 7.07, times 850 = 6,009.50. The difference is
1,081.20 on unrounded inputs, or about 1,079.50 on the displayed two-decimal price.

**Follow-up.** Why is this comparison more robust than the 0.33% full-supply gain?

**Red flag.** Multiplying the demand-model gain by the constrained quantity, which mixes two
different counterfactuals.

**Scoring.** 3 points for both contributions; 2 for the ratio; 2 for the robustness argument.

#### Q06 — The elasticity interval is [1.29, 3.81] and the decision boundary is 2.52. What do you do?

**Answer.** Do not act on the price question. The interval contains the boundary, so the test cannot
resolve the decision, and acting on the point estimate is a bet rather than a finding.

**Reasoning.** The decision needs the sign of the difference between observed and required elasticity.
An interval that spans the threshold gives no sign. Options are to increase power, to narrow the
price grid so the effect per step is larger relative to noise, or to defer — and deferral is
legitimate because the scarcity subset carries independent value.

**Follow-up.** Which of those three options changes the estimand, and which only changes precision?

**Red flag.** Reporting 2.55 as "the elasticity" without the interval, then declaring the cut viable.

**Scoring.** 3 points for refusing to act; 2 for the interval-spans-threshold reasoning; 2 for
distinguishing estimand changes from precision changes.

#### Q07 — Why can the design effect be 201 for a market-level test?

**Answer.** Because the unit of randomisation is a market-week containing about 133,000 sessions, and
sessions within a market are highly correlated in conversion.

**Reasoning.** The design effect is 1 + (m − 1)·ICC. With m = 133,333 and ICC = 0.0015, DE = 1 +
133,332 × 0.0015 = 201. The effective sample is the raw sample divided by the design effect, so 800,000
sessions per arm become about 3,980 effective. A switchback with 48 blocks reduces the block size and
the design effect to about 26, restoring 30,771 effective sessions.

**Follow-up.** What assumption must hold for the switchback estimate to be unbiased?

**Red flag.** Computing power from session counts without mentioning clustering.

**Scoring.** 3 points for the design-effect computation; 2 for the effective sample; 2 for the
switchback comparison; 1 for the time-balance assumption.

#### Q08 — Diagnosis: revenue per session rose 13.4%, contribution per session fell 3.4%. What is
wrong, and what do you check first?

**Answer.** Nothing is wrong arithmetically; the two metrics measure different things and the price
cut moved them apart. Check that the contribution definition includes all variable per-order costs,
then check that the session populations in the two arms are comparable.

**Reasoning.** Revenue per session is price times conversion; contribution per session is margin times
conversion. A cut raises the first and lowers the second whenever the margin elasticity exceeds the
volume elasticity. The two candidate explanations for a *further* divergence are an incomplete cost
basis and a traffic-composition change between arms.

**Follow-up.** What guardrail would have caught the traffic-composition problem before launch?

**Red flag.** Concluding that the measurement is broken because the metrics disagree.

**Scoring.** 2 points for the correct mechanism; 2 for the cost-basis check; 2 for the traffic check;
1 for naming a concrete guardrail.

#### Q09 — A SKU sold out at 11:00 and its conversion is a third of the category. What is the correct
inference?

**Answer.** Almost none about demand. The observation is censored: the SKU could not sell after 11:00,
so measured conversion measures availability, not willingness to pay.

**Reasoning.** Conversion is orders divided by sessions. Fulfilled orders are truncated at stock, while
sessions continue arriving. The correct estimator uses sessions before the stockout, or a censored
likelihood that treats demand above stock as unobserved.

**Follow-up.** What field must exist in the data for you to make that correction at all?

**Red flag.** Dropping the row entirely, which conditions the analysis on a post-price outcome.

**Scoring.** 3 points for naming censoring; 2 for the pre-stockout estimator; 2 for the required field.

#### Q10 — The feasible candidate set is empty for a category for three weeks. What happened?

**Answer.** A hard constraint is binding, and the programme has silently reverted to the baseline.

**Reasoning.** Candidates are filtered by the contractual floor and ceiling, the 10% move limit, and
the safety reserve. A cost shock raises the model's preferred price by more than 10%, so every
candidate is filtered and the held price is published. Because "no recommendation" and "recommend the
held price" look identical downstream, the failure is invisible unless reason codes are counted.

**Follow-up.** Which output would you add to make this observable on day one?

**Red flag.** Relaxing the move limit so that a recommendation appears.

**Scoring.** 3 points for naming a binding constraint; 2 for the reason-code mechanism; 2 for the
correct remediation.

#### Q11 — Justify rejecting a contextual bandit for this pilot.

**Answer.** Because the pilot's purpose is to measure a response, and adaptive allocation degrades
measurement to improve short-run reward.

**Reasoning.** A bandit concentrates exposure on the arm that currently looks best, which starves the
contrast needed to narrow the interval. It also interacts with the move limit: arm availability
depends on the previously published price, so the arm set changes over time and the estimator must
account for that. And the inference is harder to defend to a finance audience than a fixed
allocation.

**Follow-up.** Under what conditions would you reverse that judgement?

**Red flag.** "Bandits are more advanced, so use a bandit."

**Scoring.** 3 points for the exploration-versus-exploitation trade; 2 for the constraint interaction;
2 for the defensibility argument.

#### Q12 — Design the check that distinguishes a stale cache from a version conflict in publication.

**Answer.** A synthetic purchase journey that records price, serving timestamp, and artefact version
at each of the four surfaces, compared against the published artefact.

**Reasoning.** A stale cache shows one surface with an old timestamp and the correct version. A version
conflict shows two surfaces with different version identifiers but similar timestamps. Recording both
fields is what separates them; recording only the price cannot.

**Follow-up.** Which of the two failures would a complaint-volume alert catch first, and why is that
too late?

**Red flag.** Checking the catalogue table only, which is the one surface that is usually correct.

**Scoring.** 3 points for the journey and the two recorded fields; 2 for the discrimination logic;
2 for the alerting argument.

#### Q13 — The merchant overrides 40% of recommendations. How does that change your evaluation?

**Answer.** It forces an intention-to-treat analysis, with the per-protocol result reported separately
and labelled as such.

**Reasoning.** If the evaluation uses the published price, it measures the merchant's judgement
combined with the model, and the two cannot be separated. If it uses the recommendation, it measures
the policy as assigned. The difference between the two estimates is the override effect, which is
itself informative but is not the policy effect.

**Follow-up.** What would a large gap between the two estimates tell you about the model's features?

**Red flag.** Dropping overridden product-days to "clean" the data.

**Scoring.** 3 points for ITT; 2 for per-protocol as secondary; 2 for interpreting the gap.

#### Q14 — Where does the constant-elasticity form fail, and what replaces it?

**Answer.** It fails outside the tested price band, near e = 1, and wherever a competitor or an
assortment change shifts the demand curve rather than moving along it.

**Reasoning.** The form assumes the percentage response is the same at every price. Real demand
flattens at very low prices (the product substitutes for everything) and at very high prices (the
product is abandoned), so a spline or a semiparametric fit is better when the tested band is wide.
Near e = 1 the implied optimal price diverges, which is a symptom of using the form outside its range.
A curve shift changes A, not e, and no constant-elasticity model can represent it.

**Follow-up.** What would you plot to detect a curve shift rather than a movement along the curve?

**Red flag.** "Elasticity is a constant, so one estimate serves all prices."

**Scoring.** 3 points for the three failure regions; 2 for the replacement methods; 2 for the shift
detection.

#### Q15 — A guardrail is breached but contribution improved. Who decides, and on what basis?

**Answer.** The decision owner, on the basis of the guardrail's stated purpose, not on the size of the
contribution gain. Guardrails are pre-committed for the reason that they would otherwise always lose.

**Reasoning.** A guardrail encodes a cost the objective does not measure — customer trust, legal
exposure, or operational stability. Trading it against measured contribution at the moment of breach
is exactly the substitution the guardrail exists to prevent. The owner may suspend the policy while
investigating; the owner may not accept the breach in exchange for the gain without a recorded
policy change.

**Follow-up.** What would have to be true for the guardrail itself to be wrong rather than the policy?

**Red flag.** Comparing the breach cost against the contribution gain and accepting the net.

**Scoring.** 3 points for refusing the trade; 2 for the pre-commitment reasoning; 2 for the
distinction between suspending and amending.

#### Q16 — What single measurement would most change this recommendation?

**Answer.** The share of eligible SKUs whose sellable stock is below predicted demand, because it
determines whether the programme is a supply problem or a demand-response problem.

**Reasoning.** If that share is large, the scarcity rule captures most of the value and the
elasticity question is second-order. If it is small, the programme must solve the elasticity problem
and the current evidence cannot. The measurement is cheap — it needs an inventory snapshot and a
demand forecast, both of which already exist — and it is not in the case's arithmetic, which is why
it is the right question to ask.

**Follow-up.** How would you measure it before the demand model exists?

**Red flag.** Naming another model, another feature, or more data as the answer.

**Scoring.** 3 points for the share of constrained SKUs; 2 for the either/or argument; 2 for the
cheap measurement path.

## 15. Quick Reference Card

| Item | Value |
|---|---|
| Decision | One shelf price per eligible product-market-day for the next trading day |
| Owner | Pricing director; pricing science recommends, engineering publishes |
| Objective | Contribution per eligible product-day, not revenue and not conversion |
| Retention factor | k = 1 − 0.060 − 0.025 = 0.915 applied to revenue |
| Margin at 18.00 | 0.915×18.00 − 9.40 = 7.07 per unit |
| Break-even elasticity for a 10% cut | ln(7.07/5.42)/ln(18.00/16.20) = 2.52 |
| Observed elasticity | 2.19 two-cell; 2.55 four-cell; 95% interval [1.29, 3.81] |
| Contribution effect of the cut | −3.37% per session; −8.78% with 1.10 per-order cost |
| Scarcity price at 850 units | 19.39, worth +17.93% over the held price |
| Decision | Reject the blanket cut; authorise the scarcity rule; commission a switchback test |
