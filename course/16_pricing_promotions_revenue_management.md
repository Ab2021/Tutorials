# Module 16 — Pricing, Promotions, and Revenue Management

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 30 hours (11 lecture, 11 lab, 8 project)
> **Track fit:** Common core; deepest in Retail & E-commerce, required reading for Manufacturing commercial roles
> **Prerequisites:** Module 04 (causal inference, endogeneity); Module 09 (promotion lift, baseline decomposition); Module 10 (newsvendor, salvage); Module 01A (margins, units, currency)
> **Tools:** Python 3.12, pandas, statsmodels, linearmodels, NumPy, matplotlib; a spreadsheet for markdown grids; optional Pyomo or OR-Tools for the price-ladder optimisation
> **Industry question:** "If you cut price 10% and volume rises 25%, did you make money — and can you prove the volume rise was caused by the price?"

---

## 1. Why industry cares

Price is the highest-leverage number in the business and the one most often set by habit. A 1% price improvement at a 30% gross margin moves operating profit by roughly 3.3% if volume holds — more than most cost programmes deliver in a year. Yet in many firms the price file is updated annually, the markdown calendar is copied from last season, and the trade-spend budget is allocated by last year's share rather than by measured incrementality.

The decision has three faces. **Margin**: price times volume minus variable cost, and the volume response is the part everyone gets wrong. **Inventory**: for perishable and seasonal goods, price is the mechanism that clears stock, and the salvage value sets the floor. **Relationship**: price is the most visible thing a firm does, and a pricing model that is individually rational can be collectively destructive — to customers, to channel partners, and to the firm's legal position.

The 2026 context has two features. First, algorithmic pricing moved from airline and hotel revenue management into grocery, rental housing, and marketplaces, and with it came enforcement: the U.S. Department of Justice reached a proposed settlement with RealPage in November 2025 restricting the use of non-public, competitively sensitive data, and DOJ and FTC statements of interest through 2025–2026 argue that algorithmic price fixing is per se unlawful with no competitor-to-competitor communication required. Second, personalised pricing became technically trivial and reputationally expensive; the gap between what a model can compute and what a customer will tolerate is now a first-order design constraint, not an afterthought.

The characteristic failure is not a bad elasticity estimate. It is a **good elasticity estimate applied to a decision it does not describe**. An own-price elasticity estimated from historical promotions, where price moved together with display and feature, does not tell you what happens when you change the shelf price alone. A markdown model calibrated on full-price weeks does not describe clearance behaviour. Most pricing analytics fails at the mapping from estimate to intervention, not at the regression.

## 2. Learning outcomes

By the end of this module, you can:

- Distinguish price, value, and willingness to pay, and explain why cost-plus pricing ignores all three.
- Estimate own- and cross-price elasticity, state the identification assumption, and diagnose the direction of the endogeneity bias.
- Choose between OLS, fixed effects, and instrumental variables for a given pricing dataset, and defend the instrument.
- Apply the Lerner markup rule and explain why it does not solve a clearance problem.
- Build a markdown optimisation with salvage and inventory constraints, and compute the optimal clearance price.
- Decompose a trade promotion into incremental margin, subsidy on baseline volume, and forward buying, and compute the break-even lift.
- Apply Littlewood's rule to a two-fare capacity-allocation problem and state its assumptions.
- Design a price test — geo, switchback, or A/B — with adequate power and a pre-registered decision rule.
- Identify the antitrust risk in a pricing model: per se illegality, hub-and-spoke, and information sharing.
- Assess a pricing model for fairness, transparency, and customer-trust risk before it ships.

## 3. Mental model

### 3.1 The pricing decision stack

```
 willingness to pay (heterogeneous, unobserved)
            │
   value drivers: alternatives, reference price, urgency, switching cost
            ▼
   demand curve  D(p | segment, context)
            │
   own-price elasticity, cross-price elasticity, promotion response
            ▼
   objective: margin, revenue, volume, or clearance
            │
   constraints: inventory, capacity, legal, fairness, competitor reaction
            ▼
   price, promotion, markdown calendar
```

Every layer is a modelling choice, and the objective is not always profit. A category manager measured on revenue will price differently from one measured on margin, and both will differ from a clearance planner measured on units cleared.

### 3.2 The four pricing problems

| Problem | Time scale | Constraint | Core method |
|---|---|---|---|
| List price setting | Months–years | Competitive position, positioning | Willingness-to-pay research, elasticity |
| Promotional pricing | Weeks | Trade budget, retailer acceptance | Lift estimation, incrementality |
| Markdown and clearance | Days–weeks | Finite inventory, salvage value | Dynamic pricing with inventory |
| Revenue management | Days | Finite capacity, segmented demand | Littlewood, EMSR, dynamic programming |

These are different problems with different data and different failure modes. Applying a list-price elasticity to a clearance decision is the most common category error in retail analytics.

### 3.3 The margin arithmetic everyone should memorise

```
Δprofit ≈ Δprice × volume + price × Δvolume − Δcost × volume
At constant cost, a 1% price rise with no volume loss:
Δprofit% = 1% × price / profit = 1% / margin%
```

At a 30% margin, 1% price is 3.3% profit. At a 5% margin, 1% price is 20% profit. This ratio is why grocery pricing analytics is a different discipline from luxury pricing analytics.

## 4. Core concepts

### 4.1 Price, value, and willingness to pay

| Concept | Definition | How it is measured | Common error |
|---|---|---|---|
| Price | What is charged | The price file | Treating list price as transaction price |
| Value | The customer's perceived benefit | Conjoint, van Westendorp, choice modelling | Confusing value with cost |
| Willingness to pay (WTP) | The maximum a customer will pay | Choice experiments, auctions, field tests | Assuming one WTP per product, not per segment and occasion |
| Reference price | The price the customer expects | Historical prices, competitor prices, memory | Ignoring that promotions reset it downward |

Willingness to pay is a distribution, not a number. Two customers with the same WTP on average can have completely different price sensitivity, and a single price leaves money on the table with one and loses the sale with the other. Price segmentation — versioning, channel, timing, bundle — exists to charge different points of that distribution without charging different people different prices for the same thing at the same moment, which customers experience as unfair.

**Decision rule.** Before optimising a price, write down which of the four problems you are solving and which objective the business is measured on. If those two disagree, the model will be correct and the recommendation rejected.

### 4.2 Own-price and cross-price elasticity

```
Own-price:    ε_ii = (∂q_i / ∂p_i) × (p_i / q_i)
Cross-price:  ε_ij = (∂q_i / ∂p_j) × (p_j / q_i)
```

Constant-elasticity form, estimated in logs:

```
ln q_i = α + ε_ii ln p_i + Σ_j ε_ij ln p_j + β·X + u
```

| Value | Interpretation | Implication |
|---|---|---|
| `ε = 0` | Perfectly inelastic | Price has no volume effect (almost never true) |
| `−1 < ε < 0` | Inelastic | A price rise raises revenue |
| `ε = −1` | Unit elastic | Revenue is maximised |
| `ε < −1` | Elastic | A price cut raises revenue |
| `ε_ij > 0` | Substitutes | Raising j's price helps i |
| `ε_ij < 0` | Complements | Raising j's price hurts i |

Elasticities are not constants. They vary with the price point (demand curves are rarely isoelastic over a wide range), with the segment, with the channel, and with the presence of a promotion. A single number is a local approximation, and it should be reported with the range over which it was estimated.

Meta-analytic evidence in marketing reports average own-price elasticities in the roughly −2 to −3 range with enormous heterogeneity across categories, brands, and data sources (Bijmolt, van Heerde, and Pieters, *JMR*, 2005). The practical implication is not "use −2.5" but "expect your category estimate to be wrong if it comes from a different category, and validate it locally".

### 4.3 Estimating elasticity: identification and endogeneity

Three biases dominate price-elasticity estimation:

| Bias | Mechanism | Direction | Fix |
|---|---|---|---|
| Simultaneity | Firms raise price when demand is strong | Elasticity biased toward zero (too inelastic) | Instrument with a cost shifter |
| Omitted variable | Promotion, display, and feature move with price | Depends on the omitted variable's sign | Control for all marketing mix |
| Selection | Price changes are chosen where managers expect a response | Elasticity biased away from zero | Experiment, or model the selection |

The instrument must shift price and affect demand only through price. Candidate instruments: input cost indices (commodity, freight, FX), supplier price changes, and regulatory or tax changes. A competitor's price is *not* a valid instrument if it also shifts demand directly — which it usually does.

```python
"""Elasticity estimation: OLS versus IV on simulated weekly SKU data.
Run: pip install numpy pandas statsmodels linearmodels ; python elasticity.py
The DGP sets a true elasticity of -2.2 and makes price respond to demand shocks,
so OLS should be biased toward zero and IV should recover the true value.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

rng = np.random.default_rng(7)
rows = []
for sku in range(40):
    base = rng.uniform(3.0, 12.0)
    for week in range(104):
        season = 0.15 * np.sin(2 * np.pi * week / 52)
        log_cost = np.log(base * (1 + 0.06 * np.sin(2 * np.pi * week / 26))
                          + rng.normal(0, 0.02 * base))
        promo = int(rng.random() < 0.15)
        shock = rng.normal(0, 0.18)                    # unobserved demand shock
        log_p = (log_cost + 0.22 * shock - 0.06 * promo
                 + rng.normal(0, 0.03))                # price responds to shock
        log_q = (np.log(900) - 2.2 * (log_p - np.log(base)) + 0.35 * promo
                 + season + shock + rng.normal(0, 0.12))
        rows.append((sku, week, log_p, log_q, log_cost, promo, season))

df = pd.DataFrame(rows, columns=["sku", "week", "log_p", "log_q",
                                 "log_cost", "promo", "season"])
for c in ["log_q", "log_p", "log_cost", "promo", "season"]:
    df[c + "_w"] = df[c] - df.groupby("sku")[c].transform("mean")   # SKU fixed effects

ols = smf.ols("log_q_w ~ log_p_w + promo_w + season_w", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["sku"]})
iv = IV2SLS.from_formula(
    "log_q_w ~ 1 + promo_w + season_w + [log_p_w ~ log_cost_w]", data=df
).fit(cov_type="clustered", clusters=df["sku"])

print("true elasticity  -2.200")
print(f"OLS {ols.params['log_p_w']:+.3f} (se {ols.bse['log_p_w']:.3f})")
print(f"IV  {iv.params['log_p_w']:+.3f} (se {iv.std_errors['log_p_w']:.3f})")
```

Expected behaviour: OLS lands around −1.6 because price rises with the demand shock, which the regression attributes to price; IV, using the cost shifter, recovers something close to −2.2. If your IV estimate is *more* negative than OLS by a large factor, suspect a weak instrument — check the first-stage F statistic before believing the second stage.

### 4.4 From elasticity to price: the Lerner rule

For a monopolist with constant elasticity and constant marginal cost `c`, profit `(p − c)·D(p)` is maximised where:

```
(p − c) / p = 1 / |ε|        (Lerner condition)
p* = c · ε / (1 + ε)          with ε < −1
```

**Worked example.** `c = $40`, `ε = −2.0`:

```
p* = 40 × (−2) / (1 − 2) = 80
Check: (80 − 40) / 80 = 0.50 = 1/2 ✓
```

At `ε = −1.8`, `p* = 40 × 1.8 / 0.8 = $90`; at `ε = −3`, `p* = 40 × 3 / 2 = $60`.

| `ε` | Optimal markup on cost | Margin |
|---:|---:|---:|
| −1.2 | 6.0× | 83% |
| −1.5 | 3.0× | 67% |
| −2.0 | 2.0× | 50% |
| −3.0 | 1.5× | 33% |
| −5.0 | 1.25× | 20% |

**The rule assumes you can supply whatever demand appears.** That is exactly what a clearance problem violates. In a markdown, inventory is fixed, salvage is low, and the optimal price is set by the inventory constraint, not by the markup rule. Conflating the two is the single most common analytical error in retail pricing.

### 4.5 Markdown optimization with salvage

With `Q` units of inventory, price `p`, demand `D(p)`, and salvage `s` per unsold unit:

```
Revenue(p) = p · min(Q, D(p)) + s · max(0, Q − D(p))
```

**Worked example.** 800 units remain, reference price $60 with base demand 500, constant elasticity `ε = −1.8`, salvage $15.

```
D(p) = 500 × (p / 60)^(−1.8)
```

| Price | Demand | Units sold | Salvage units | Revenue |
|---:|---:|---:|---:|---:|
| $60 | 500 | 500 | 300 | $34,500 |
| $55 | 585 | 585 | 215 | $35,392 |
| $50 | 694 | 694 | 106 | $36,297 |
| $47 | 776 | 776 | 24 | $36,832 |
| **$46.50** | **791** | **791** | **9** | **$36,917** |
| $46 | 807 | 800 | 0 | $36,800 |
| $45 | 839 | 800 | 0 | $36,000 |

The optimum sits just above the sell-out price. Solve for the price at which demand equals inventory: `(800/500)^(−1/1.8) × 60 = 1.6^(−0.556) × 60 = 0.770 × 60 = $46.21`. At any price below that, extra demand is wasted and revenue falls linearly. The optimum is a little above it, because a small amount of salvage is better than giving away the last units.

Now compare with the Lerner rule. At `ε = −1.8` and `c = $40`, the unconstrained optimum is `p* = $90` — far above the current $60. The markup rule says *raise* price; the clearance model says *cut* it. Both are correct for their own problem. The $90 answer assumes you can supply demand you do not have; the $46.50 answer assumes you must dispose of inventory you do have. **The gap between them is the cost of the inventory decision, not the pricing decision.**

**Two-period extension.** With `T` periods remaining and inventory carried forward, the optimal price in each period solves a Bellman equation: the value of a unit sold now is `p`, and the value of holding it is the expected marginal value next period. The practical approximation used in most retail systems is to set a markdown ladder by weeks-to-clearance and re-optimise weekly as the inventory position and remaining time update.

### 4.6 Promotion optimization and trade spend

A trade promotion has three components that must be separated:

```
gross lift         = promoted volume − baseline volume
incremental margin = gross lift × margin at promoted price
subsidy            = baseline volume × (list price − promoted price)
net promotion profit = incremental margin − subsidy − fixed costs
```

**Worked example.** List price $5.00, COGS $2.80 (margin $2.20). Trade price $4.00 (margin $1.20). Baseline 1,000 units; promoted volume 1,800.

```
Gross lift = 800 units
Incremental margin = 800 × $1.20 = $960
Subsidy on baseline = 1,000 × $1.00 = $1,000
Net = $960 − $1,000 = −$40
```

The promotion lost money even before fixed costs, forward buying, and the post-promotion dip.

**Break-even lift.** The lift at which net profit is zero:

```
Required lift / baseline ≥ discount / margin at promoted price
                        = $1.00 / $1.20 = 83.3%
Actual lift = 800 / 1,000 = 80.0% → below break-even
```

This ratio is the most useful number in trade-spend analytics, because it is computable before the promotion runs. At a 20% discount and a 25% margin, you need an 80% lift just to break even on the incremental units; most promotions do not deliver it once the baseline is correctly measured.

| Trade-spend metric | Formula | What it catches |
|---|---|---|
| Lift | (promoted − baseline) / baseline | Raw volume response |
| Incrementality | incremental units / promoted units | Share of promoted volume that is new |
| Break-even lift | discount / margin at promoted price | Whether the mechanic can pay for itself |
| Forward buying | units bought above normal in the deal window | Retailer arbitrage, not demand |
| Post-promotion dip | units below baseline after the promotion | Borrowed rather than created demand |
| Trade spend ratio | spend / incremental gross margin | Efficiency of the spend |

**Forward buying and the dip are the two reasons promotions look better in the shipment data than in the consumer data.** Measure on POS or scan data where available; if you only have shipments, model the inventory pipeline explicitly.

### 4.7 Revenue management and Littlewood's rule

Two fare classes, capacity `C` remaining, discount fare `f_d`, full fare `f_f`, full-fare demand `D_f` with distribution `F`. Accept a discount request if:

```
f_d ≥ f_f × P(D_f > remaining capacity after accepting)
```

Equivalently, protect `y*` units for full fare where `P(D_f > y*) = f_d / f_f`, so `y* = F⁻¹(1 − f_d/f_f)`.

**Worked example.** `C = 100` seats, `f_f = $400`, `f_d = $200`, `D_f ~ Normal(60, 20)`.

```
f_d / f_f = 0.5 → y* = 60 seats protected, 40 released at the discount
```

| Protection level | Discount seats | Expected discount revenue | Expected full-fare revenue | Total |
|---:|---:|---:|---:|---:|
| 50 | 50 | $10,000 | $18,416 | $28,416 |
| **60** | **40** | **$8,000** | **$20,808** | **$28,808** |
| 70 | 30 | $6,000 | $22,418 | $28,418 |

Full-fare revenue uses `E[min(D, y)] = μ − σ·L(z)` with the standard normal loss function. At `y = 60`, `z = 0`, `L(0) = 0.3989`, so `E[min] = 60 − 20 × 0.3989 = 52.02` and revenue is `52.02 × 400 = $20,808`.

The objective is flat near the optimum — a ten-seat error costs about $390, or 1.4%. That flatness is why simple rules survive in revenue management, and why the modelling effort should go into the demand forecast and the fare fences rather than into the optimiser.

**Assumptions to state.** Demand for the two classes is independent; the full-fare demand distribution is known; there is no cancellation, no no-show, no overbooking, and no recapture of discount customers at the full fare. Every one of these is violated in practice, which is why production systems use EMSR-b or dynamic programming rather than plain Littlewood.

### 4.8 Dynamic and personalized pricing

| Approach | Price varies by | Data required | Main risk |
|---|---|---|---|
| Static list price | Nothing | Cost, competition | Leaves surplus on the table |
| Time-based dynamic | Time, inventory, remaining capacity | Demand forecast, inventory state | Customer learning and backlash |
| Segment-based | Segment, channel, cohort | Segment membership | Proxy discrimination |
| Personalized | Individual | Identity, behaviour, context | Fairness, privacy, legal exposure |

Dynamic pricing is legitimate and standard in airlines, hotels, and marketplaces, where the product is perishable capacity and the price is visibly time-varying. It becomes reputationally dangerous when the same product is priced differently to two customers at the same moment for reasons the customer cannot see — the classic example being a price that rises with inferred willingness to pay rather than with cost or scarcity.

**Decision rule.** Before deploying personalised pricing, answer three questions in writing: does the price vary with a cost or scarcity driver that can be explained, or with an inferred personal characteristic; would a customer accept the explanation if shown it; and is the segmentation variable a protected characteristic or a proxy for one. If the first answer is "personal characteristic", the third is usually "yes, indirectly".

### 4.9 Competitive response and price wars

A price cut invites a response. The analytical treatment is a repeated game: if firms interact indefinitely and can observe each other, cooperation at a high price can be an equilibrium; if one can defect undetected, it is not. Rotemberg and Saloner (1986) showed that price wars are more likely during demand booms, because the short-run gain from defecting rises while the future punishment stays the same.

| Situation | Likely response | Analytical implication |
|---|---|---|
| Transparent, concentrated market | Fast matching | Price cut gains share for days, not quarters |
| Differentiated product | Partial or no matching | Share gain persists; margin effect dominates |
| Capacity-constrained competitor | No response | You capture the spillover |
| Private-label entrant | Price war on the branded item | Margin destruction with no volume gain |

Model competitive response as a scenario, not a parameter. "Competitor matches within two weeks" and "competitor does not respond" are two different business cases, and the recommendation should state which one it assumes.

### 4.10 Price testing and causal measurement

| Design | Unit | Strength | Weakness |
|---|---|---|---|
| A/B test | Customer or session | Clean randomisation | Cannot price differently to similar customers without detection |
| Geo test | Store, region, market | Real shelf prices, real competitors | Few units, low power; spillover across regions |
| Switchback | Time block within unit | Controls for unit heterogeneity | Carryover, customer learning |
| Regression discontinuity | Price threshold | Strong identification | Requires a genuine threshold |
| Difference-in-differences | Treated vs control units | Handles common shocks | Parallel trends must hold |
| Synthetic control | Aggregate unit | Handles one treated market | Few donors, sensitive to fit |

Power is the binding constraint in retail price tests. Price effects are small relative to demand noise, and store-level tests often have fewer than 50 units per arm. Compute the minimum detectable effect before running the test; if it is larger than the effect you care about, the test cannot answer the question and a different design is needed.

**Pre-register the decision rule.** "If the 95% interval for the elasticity excludes −1, we will move to the new price; otherwise we keep the current one." Without this, every result becomes a debate about interpretation, and the test becomes advocacy.

### 4.11 Reference prices and psychological pricing

| Effect | Description | Practical implication |
|---|---|---|
| Reference price | Customers compare to an expected price, formed from memory and context | A sustained discount lowers the reference price and the future full-price demand |
| Anchoring | The first price seen shifts subsequent judgments | "Was $80, now $60" changes evaluation even when $60 is the true value |
| Nine-ending | Prices ending in 9 signal a bargain | Effective in low-involvement categories, not in luxury |
| Round-price | Round numbers signal quality and reduce cognitive cost | Common in premium and B2B |
| Decoy effect | A dominated option shifts choice toward the target | Bundle and tier design |
| Loss aversion | A surcharge hurts more than an equivalent discount pleases | Frame as "save" rather than "avoid fee" where honest |

The reference-price effect is the one with the largest P&L consequence, because it is a *stock*, not a flow: a promotion that trains customers to wait for discounts changes the elasticity of every future period. Measuring promotion lift without measuring the post-promotion reference-price effect systematically overstates the value of promotion.

### 4.12 Antitrust and algorithmic pricing

Price coordination is illegal per se under Section 1 of the Sherman Act; the per se rule was applied to horizontal price fixing in *United States v. Socony-Vacuum Oil Co.* (1940) and remains the framework for naked restraints.

| Theory | Mechanism | Why a model can create it |
|---|---|---|
| Per se price fixing | Competitors agree on price | A shared pricing tool with a common recommendation can function as the agreement |
| Hub-and-spoke | A common intermediary facilitates horizontal coordination | A vendor's algorithm used by competitors, with the vendor as hub |
| Information sharing | Exchange of competitively sensitive, non-public data | A benchmarking feed or a shared "market price" input |
| Algorithmic tacit collusion | Autonomous learning converges on supra-competitive prices without communication | Reinforcement-learning pricing agents in a repeated game |

The enforcement posture as of October 2026: DOJ and FTC statements of interest through 2025–2026 argue that algorithmic price fixing is per se unlawful and that no competitor-to-competitor communication is required; DOJ's proposed settlement with RealPage in November 2025 would restrict the use of non-public, competitively sensitive data. But the picture is not uniform — *Gibson v. Cendyn*, in the Ninth Circuit, dismissed claims against shared hotel revenue-management software, and the en banc rehearing was denied in November 2025. The OECD's October 2025 G7 report catalogues the mechanisms and notes that enforcement remains fact-specific.

**Practical controls.** Do not use a competitor's non-public price, occupancy, or supply as an input. Do not share your own forward pricing or capacity with competitors through a common vendor. Keep an auditable record of the inputs to each price recommendation. Test whether the model's output changes when a competitor's private signal is removed — if it does, the signal is in the model and the model is a legal exhibit waiting to happen.

### 4.13 Fairness, transparency, and trust

| Risk | Example | Control |
|---|---|---|
| Price gouging | Essentials priced up during a shortage | Cap on necessity categories during declared emergencies; state the rule |
| Proxy discrimination | ZIP code standing in for race or ethnicity | Audit segmentation variables; test disparate impact |
| Dark patterns | Drip pricing, fake scarcity, hard-to-cancel | Design review; disclosure of the full price early |
| Unexplained personalisation | Same product, different price, no reason given | Explainability requirement; opt-out |
| Reference-price deception | Inflated "was" price | Substantiate prior prices; keep records |

Two regulatory anchors matter. The EU Omnibus Directive (2019/2161) requires that a stated prior price be the lowest price in the 30 days before the reduction — a direct constraint on "was/now" labelling. In the United States, FTC Act Section 5 covers deceptive pricing, and the FTC's 2024 "click to cancel" rule and related actions target negative-option and drip-pricing practices; state-level algorithmic-pricing statutes took effect in California and New York in 2026 `[VERIFY: exact statute numbers, effective dates, and scope before citing in a compliance memo]`.

## 5. Data, standards, and workflow

| Data object | Grain | Source | Requirement |
|---|---|---|---|
| Transaction price | Order line | ERP / POS / OMS | Net of discount, rebate, and coupon; not list price |
| List and trade price | SKU × customer × period | Pricing system | Effective dates; off-invoice vs bill-back |
| Cost | SKU × period | Finance | Standard cost plus variances; do not use last year's |
| Promotion | SKU × store × week | TPM / retailer POS | Mechanic, feature, display, circular, coupon |
| Competitor price | SKU × store × day | Price scraping, audit | Collection method and accuracy; legal review |
| Inventory and sell-through | SKU × location × day | WMS / POS | For markdown and clearance models |
| Elasticity estimate | SKU or category × segment | Model output | With method, sample, and confidence interval |
| Salvage and disposal cost | SKU × channel | Finance / operations | Include disposal fees, not just recovery value |

**Standards.** GS1 GTIN for product identity across retailer and manufacturer records; ISO 8601 for periods; ISO 4217 for currency, with an explicit FX policy for cross-border pricing. For consumer-price display, the EU Omnibus Directive constrains prior-price claims. For promotions, the retailer's own vendor guide governs deduction and proof-of-performance requirements, and those rules differ by retailer — read them before designing a promotion.

**Workflow.** Reconcile the price file to the invoice before estimating anything; a model built on list price when the business runs on net price will be wrong by the average discount. Separate baseline from incremental volume with an explicit method, and state it. Log every price change with the reason code. And keep the experiment register: which prices were tested, where, when, and what was decided — because the second-most-common failure after endogeneity is re-litigating a test nobody recorded.

## 6. Industry case studies

### 6.1 American Airlines and the origin of yield management (1980s)

**Context.** After deregulation in 1978, U.S. airlines could set fares freely, and the incumbent carriers faced low-cost entrants on high-demand routes. American had a fixed seat inventory per departure and a booking curve that could be observed in near real time through its SABRE reservation system.

**Intervention.** American developed origin-and-destination revenue management, allocating seats between fare classes using booking forecasts and marginal-value rules of the Littlewood family. The system was integrated with SABRE and later extended to continuous re-optimisation as bookings arrived.

**Evidence.** The mechanism and its economics are documented in the revenue-management literature (Talluri and van Ryzin; Phillips) and in industry accounts. The revenue benefit figures are company claims and should be treated as such `[VERIFY: exact figure and year before quoting]`.

**Outcome.** Revenue management became a standard capability in airlines and hotels, and the discipline's core insight — that the right price depends on remaining capacity and time — generalised to any perishable-inventory business.

**What transferred.** The framing of a sale as consuming a scarce unit whose future value is uncertain, and the use of a demand forecast by fare class rather than a single aggregate forecast. **What did not.** The assumption of clean fare fences. In grocery and general retail there is no equivalent of a Saturday-night stay, so segmenting by price without segmenting by product or channel is difficult and often unfair.

### 6.2 J.C. Penney's "Fair and Square" pricing (2012–2013)

**Context.** J.C. Penney had run a heavy coupon and promotion calendar for years, and management concluded that customers were waiting for discounts and that the promotional machinery was expensive and confusing.

**Intervention.** In February 2012 the company eliminated most coupons and markdowns, moving to a three-tier structure of everyday prices, month-long "monthly values", and periodic "best prices", under the "Fair and Square" banner.

**Evidence.** Fiscal 2012 comparable-store sales fell by roughly 25% and the company reported a net loss of about $1 billion; the CEO departed in April 2013 and the strategy was reversed. These figures come from company reporting and contemporaneous coverage.

**Outcome.** The strategy failed commercially. The mechanism is legible in pricing terms: Penney removed the reference price against which its customers evaluated a purchase, and replaced a familiar discount with an unfamiliar everyday price that was not low enough to be a value position and not high enough to be a quality position.

**What transferred.** The lesson that reference prices are an asset with real value, and that removing a promotion is a price increase in the customer's frame even when the net price is unchanged. **What did not.** The inference that promotions are therefore always good. The correct conclusion is narrower: any change to the price architecture must be tested at scale before it is rolled out chain-wide, because the reference-price effect is not observable in a two-week pilot.

### 6.3 RealPage and algorithmic rent pricing (2021–2025)

**Context.** RealPage supplies revenue-management software to landlords, including a pricing recommendation engine that uses inputs from participating properties. ProPublica reporting in 2022 raised questions about whether the shared data and common recommendations facilitated coordination among otherwise independent landlords.

**Intervention.** The U.S. Department of Justice, joined by state attorneys general, filed suit in 2023 alleging that the algorithm functioned as a device for exchanging non-public, competitively sensitive information and for aligning rents. In November 2025 DOJ announced a proposed settlement that would restrict the use of such non-public data.

**Evidence.** The allegations and the proposed settlement are matters of public record in the litigation; the underlying conduct is contested, and this case study describes the allegations, not established facts. The Ninth Circuit's dismissal in *Gibson v. Cendyn* (en banc rehearing denied November 2025) shows that courts have not treated all shared pricing software alike.

**Outcome.** As of October 2026 the enforcement position is that algorithmic price fixing can be per se unlawful without direct competitor communication, and the settlement imposes data-use restrictions. The civil litigation continues.

**What transferred.** The design constraint: a pricing model's *inputs* determine its legal character as much as its outputs. A model that consumes a competitor's non-public price is a coordination mechanism regardless of how it is packaged. **What did not.** The inference that all algorithmic pricing is unlawful. Cost-based, inventory-based, and own-history-based dynamic pricing is ordinary commercial practice; the risk concentrates in competitor-data inputs and in common intermediaries.

### 6.4 Inditex and Zara's markdown discipline (2000s onward)

**Context.** Fashion retail carries high markdown risk: long design-to-shelf cycles, uncertain demand, and a short selling season mean that a large share of inventory is sold at a discount.

**Intervention.** Inditex built a model of short production runs, frequent replenishment from store sales data, and rapid design-to-shelf cycles, so that most inventory is committed late and in small batches. The consequence is fewer units that must be cleared at the end of the season, and a smaller markdown budget.

**Evidence.** Described in the operations literature, including Ferdows, Lewis, and Machuca's *Harvard Business Review* article "Rapid-Fire Fulfillment" (2004), and in company disclosures. Markdown percentages and cycle-time figures are company-reported or case-estimated, not independently audited.

**Outcome.** Inditex reports lower markdown levels and higher full-price sell-through than typical apparel peers. The mechanism is supply chain, not pricing: the company reduced the quantity that needed discounting rather than optimising the discount.

**What transferred.** The most important pricing lever is often upstream — the commitment decision. A markdown model applied to a bad buy optimises the recovery, not the outcome. **What did not.** The full model in categories with long lead times (appliances, furniture), where the buy must be committed before demand is observed and markdown optimisation is the only remaining lever.

## 7. Labs

### Lab 16.1 — Elasticity estimation and the endogeneity audit

**Objective.** Estimate own- and cross-price elasticity and show how identification changes the answer.

**Data.** Weekly SKU-store panel with price, units, promotion flags, display, feature, cost index, and a competitor price series.

**Steps.** (1) Estimate log-log OLS with SKU and week fixed effects. (2) Add promotion and display controls and compare. (3) Estimate with a cost-shifter instrument and report the first-stage F. (4) Compare the three estimates and explain the direction of each bias. (5) Estimate cross-price elasticities within the category and check for implausible signs. (6) Recommend which estimate to use for which decision.

**Deliverable.** A coefficient table across specifications, a bias explanation, and a decision memo.

**Acceptance criteria.** Fixed effects specified; instrument justified and first stage reported; bias direction explained, not just observed; cross-elasticity signs checked for consistency. **Stretch.** Estimate a random-coefficients logit demand model and compare the implied elasticities.

### Lab 16.2 — Markdown calendar optimisation

**Objective.** Build a markdown ladder with salvage and compare it with a rule-of-thumb calendar.

**Data.** A seasonal item with 12 weeks of selling, initial inventory, weekly demand curve, salvage value, and historical markdown behaviour.

**Steps.** (1) Estimate the demand curve. (2) Implement the single-period revenue model with salvage. (3) Compute the optimal clearance price and confirm the sell-out price arithmetic. (4) Extend to a multi-period dynamic programme with inventory carry-over. (5) Compare total revenue with the historical calendar. (6) Test sensitivity to elasticity and salvage.

**Deliverable.** A markdown schedule, a revenue comparison, and a sensitivity table.

**Acceptance criteria.** Salvage included; sell-out price computed and checked; multi-period recurrence correct; sensitivity reported. **Stretch.** Add a competitor's markdown response as a scenario.

### Lab 16.3 — Trade promotion incrementality

**Objective.** Separate incremental volume from baseline, forward buying, and the post-promotion dip.

**Data.** Retailer POS data with promotion calendar, plus shipment data for the same period.

**Steps.** (1) Build a baseline with a seasonal model excluding promoted weeks. (2) Compute gross lift and incrementality. (3) Compute break-even lift for each mechanic. (4) Estimate forward buying from the shipment-POS gap. (5) Estimate the post-promotion dip. (6) Rank the promotions by net profit and recommend a calendar.

**Deliverable.** A promotion scorecard, a break-even analysis, and a calendar recommendation.

**Acceptance criteria.** Baseline method stated; shipment and POS reconciled; break-even computed per mechanic; dip estimated rather than assumed. **Stretch.** Add a reference-price effect and show how it changes the ranking.

### Lab 16.4 — Price test design and legal review

**Objective.** Design a defensible price test and audit a pricing model for antitrust and fairness risk.

**Data.** A store-level sales panel, a candidate price change, and a fictional pricing-model specification with its input list.

**Steps.** (1) Compute the minimum detectable effect for a store-level test at the available store count. (2) Choose between geo test, switchback, and matched control, with a justification. (3) Pre-register the decision rule. (4) Audit the model specification's inputs against the antitrust tests in §4.12. (5) Audit the segmentation variables for proxy discrimination. (6) Write the controls you would require before deployment.

**Deliverable.** A test protocol and a legal-and-fairness risk memo.

**Acceptance criteria.** Power computed before design; decision rule pre-registered; every model input classified as public, own, or competitor; proxy risk assessed. **Stretch.** Run the audit on a real vendor specification sheet.

## 8. Assignment — Elasticity and markdown study

Produce an **elasticity and markdown study** for one category, of the quality a pricing committee would act on.

| Section | Content |
|---|---|
| 1. Decision | The pricing decision, the horizon, the objective, and who owns it |
| 2. Data | Price, cost, promotion, competitor, and inventory sources; net versus list price |
| 3. Elasticity | Own and cross estimates, specification, identification strategy, confidence intervals |
| 4. Validation | Holdout or test evidence; comparison with a naive baseline |
| 5. Markdown model | Demand curve, salvage, inventory, optimal ladder |
| 6. Promotion | Incrementality, break-even lift, forward buying, dip |
| 7. Competitive | Response scenarios and the profit consequence of each |
| 8. Legal | Input audit, information-sharing exposure, fairness assessment |
| 9. Recommendation | Price or calendar change, expected effect, and the evidence required to reverse it |
| 10. Reproduction | Code, data snapshot, parameter file, one-command rerun |

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Identification | 25% | OLS on raw price and quantity | Fixed effects, promotion controls | Instrument justified, first stage reported, bias direction explained |
| Elasticity interpretation | 15% | Reports a single number | Reports elasticity by segment | States the range of validity and the decision it supports |
| Markdown model | 20% | Applies a fixed discount rule | Single-period optimum with salvage | Multi-period dynamic programme with sensitivity |
| Promotion analysis | 15% | Gross lift only | Baseline and incrementality | Forward buying and dip measured; break-even per mechanic |
| Legal and fairness | 15% | Not considered | Risks named | Inputs classified, controls specified, proxy audit performed |
| Communication | 10% | Regression table | Clear memo | A pricing-committee document with a pre-registered decision rule |

## 10. Regulatory, ethical, and commercial lens

| Area | Requirement or risk | Practical control |
|---|---|---|
| Sherman Act §1 / EU Art. 101 | Horizontal price fixing is per se unlawful | No competitor non-public data in the model; no shared forward pricing |
| Hub-and-spoke | A common intermediary can facilitate coordination | Vendor due diligence; contractual limits on data use |
| EU Omnibus Directive (2019/2161) | Prior-price claims must reference the lowest price in 30 days | Keep a price history; substantiate "was" prices |
| FTC Act §5 | Deceptive pricing and drip pricing are unfair or deceptive | Full price disclosed early; no fake scarcity |
| GDPR / CCPA-CPRA | Personalised pricing uses personal data | Lawful basis; transparency; opt-out where required |
| Price gouging statutes | Essentials during declared emergencies | Pre-committed cap rule for necessity categories |
| Consumer protection (state) | Algorithmic pricing disclosure requirements | Monitor state law; `[VERIFY: 2026 California and New York statutes before relying]` |
| EU AI Act (Regulation 2024/1689) | Pricing that materially affects access to essentials may be high-risk in some readings | Document the pricing logic; human review; appeal path |

The ethical core is simple to state and hard to implement: a price should be explainable to the person paying it. Where the explanation is "the model found you would pay more", the firm has converted a data asset into a trust liability, and the legal exposure usually arrives after the reputational one.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| List price used as transaction price | Elasticity near zero | Discounts and rebates omitted | Model describes a price nobody pays | Use net price; reconcile to invoice |
| Endogeneity ignored | Implausibly inelastic estimate | Price set in response to demand | Under-pricing in strong demand | Instrument with a cost shifter |
| Promotion effects attributed to price | Large elasticity in promoted weeks only | Display and feature omitted | Over-optimistic price cuts | Control for full marketing mix |
| Lerner rule applied to clearance | Recommends raising price on dead stock | Supply constraint ignored | Inventory written off | Use the inventory-constrained model |
| Salvage ignored | Markdown too shallow | Overage treated as free | Excess disposal cost | Include salvage and disposal |
| Baseline built on promoted weeks | Lift overstated | Contaminated baseline | Chronic over-investment in promotion | Exclude promoted periods; seasonal model |
| Forward buying treated as demand | Shipments exceed POS | Channel inventory build | Production whiplash after the deal | Reconcile shipments to POS |
| Break-even lift not computed | Promotion approved on gross lift | Subsidy on baseline ignored | Negative-return promotions at scale | Compute break-even before approval |
| Competitor price in the model | Recommendation moves with a rival's private data | Information sharing | Antitrust exposure | Remove; audit inputs |
| Test underpowered | "No significant effect" on a small test | MDE larger than the effect | Wrong decision, false confidence | Compute MDE before designing |
| Reference price ignored | Full-price demand falls after every promotion | Customer learning | Structural margin erosion | Measure post-promotion demand |
| Personalised pricing without explanation | Support complaints about "different prices" | Opaque segmentation | Trust and legal damage | Explainability; opt-out; audit proxies |

## 12. Interview and discussion questions

| # | Question | Answer hints |
|---:|---|---|
| 1 | Why is cost-plus pricing usually wrong? | It ignores willingness to pay and competitor position; cost sets a floor, not a price. |
| 2 | Your OLS elasticity is −0.8. What do you suspect? | Endogeneity: price rises with demand, biasing toward zero; also omitted promotion variables and list-versus-net price. |
| 3 | What makes a valid instrument for price? | Correlated with price, uncorrelated with the demand error; cost shifters usually qualify, competitor price usually does not. |
| 4 | Derive the Lerner rule. | Maximise `(p−c)D(p)`; the first-order condition gives `(p−c)/p = 1/|ε|`. |
| 5 | Why doesn't the Lerner rule solve a markdown? | It assumes you can supply any demand; a markdown has fixed inventory and a salvage value. |
| 6 | Where is the optimal clearance price relative to the sell-out price? | Just above it: below it, demand is wasted; at it, you forgo salvage on the last units. |
| 7 | A promotion has 80% lift and a 20% discount at a 25% margin. Profitable? | Break-even lift is 80%; at exactly 80% it is marginal, and forward buying and the dip make it negative. |
| 8 | State Littlewood's rule. | Protect `y*` where `P(D_f > y*) = f_d/f_f`; accept a discount booking if the discount fare exceeds the expected full-fare value of the seat. |
| 9 | What is the difference between dynamic and personalised pricing? | Dynamic varies with time, inventory, and scarcity; personalised varies with the individual's inferred willingness to pay. |
| 10 | How would you test a price change in 40 stores? | Geo or matched-pair design, compute MDE first, pre-register the decision rule, watch for spillover and competitor response. |
| 11 | When does a pricing algorithm become an antitrust problem? | When it uses competitors' non-public data or a common intermediary facilitates coordination; per se treatment can apply without direct communication. |
| 12 | How do you measure the reference-price effect of a promotion? | Compare post-promotion full-price demand with a matched control that did not promote. |
| 13 | What would make you reject a vendor's pricing tool? | Competitor private data as an input, no audit trail, no explanation facility, or a black-box segmentation with protected-class proxies. |

## 13. Key terms

| Term | Definition |
|---|---|
| Willingness to pay | The maximum price a customer will accept for a product in a context |
| Reference price | The price a customer expects, formed from memory and context |
| Own-price elasticity | Percentage change in quantity for a percentage change in own price |
| Cross-price elasticity | Percentage change in quantity for a percentage change in another product's price |
| Endogeneity | Correlation between a regressor and the error term, usually from simultaneity or omission |
| Instrumental variable | A variable that shifts price but affects demand only through price |
| Lerner condition | `(p − c)/p = 1/|ε|` at the profit-maximising price |
| Markup rule | Optimal price `c·ε/(1+ε)` under constant elasticity |
| Salvage value | Recovery value of an unsold unit at the end of the season |
| Markdown optimisation | Choosing clearance prices subject to finite inventory and salvage |
| Trade spend | Manufacturer funding of retailer promotions: off-invoice, bill-back, slotting, display |
| Incrementality | Share of promoted volume that would not have occurred otherwise |
| Break-even lift | Lift required for a promotion to cover the subsidy on baseline volume |
| Forward buying | Retailer purchase beyond normal needs during a deal, to sell later at full price |
| Post-promotion dip | Demand below baseline after a promotion, reflecting borrowed purchases |
| Revenue management | Allocating finite capacity across segments and time to maximise revenue |
| Yield management | Revenue management in perishable-capacity services |
| Littlewood's rule | Two-fare protection level rule for capacity allocation |
| EMSR | Expected marginal seat revenue; multi-fare extension of Littlewood |
| Dynamic pricing | Prices that vary with time, inventory, and demand state |
| Personalised pricing | Prices that vary by individual or inferred willingness to pay |
| Price fence | A condition that separates segments, such as a Saturday-night stay |
| Geo experiment | A price test randomised at the store or region level |
| Switchback test | A test alternating treatment over time within the same unit |
| Minimum detectable effect | The smallest effect a test can reliably detect at a given power |
| Prior-price rule | Requirement that a "was" price reflect the recent actual selling price |
| Hub-and-spoke | Coordination facilitated by a common intermediary |
| Per se illegality | Conduct unlawful without inquiry into its effects |
| Price gouging | Raising prices on necessities during an emergency, regulated by state law |

## 14. Sources and further reading

**Standards and regulation**

- Sherman Antitrust Act, 15 U.S.C. §1. Cornell Legal Information Institute.
- Consolidated version of the Treaty on the Functioning of the European Union, Article 101. EUR-Lex.
- Directive (EU) 2019/2161 on better enforcement of consumer protection rules (Omnibus Directive). EUR-Lex. Prior-price provision, Article 6a of Directive 98/6/EC as amended.
- Federal Trade Commission Act, Section 5. 15 U.S.C. §45.
- Regulation (EU) 2024/1689 (Artificial Intelligence Act). EUR-Lex.
- Regulation (EU) 2016/679 (GDPR); California Consumer Privacy Act as amended by CPRA. Accessed 27 October 2026.
- U.S. Department of Justice, proposed settlement with RealPage and Greystar, November 2025. https://www.justice.gov/atr — accessed 27 October 2026. `[VERIFY: exact press release permalink and settlement date.]`

**Peer-reviewed and foundational**

- Littlewood, K. "Forecasting and Control of Passenger Bookings." *AGIFORS Symposium Proceedings*, 12, 1972.
- Gallego, G., and van Ryzin, G. "Optimal Dynamic Pricing of Inventories with Stochastic Demand over Finite Horizons." *Management Science*, 40(8), 1994.
- Rotemberg, J. J., and Saloner, G. "A Supergame-Theoretic Model of Price Wars during Booms." *American Economic Review*, 76(3), 1986.
- Chevalier, J. A., Kashyap, A. K., and Rossi, P. E. "Why Don't Prices Rise During Periods of Peak Demand? Evidence from Scanner Data." *American Economic Review*, 93(1), 2003.
- Bijmolt, T. H. A., van Heerde, H. J., and Pieters, R. G. M. "New Empirical Generalizations on the Determinants of Price Elasticity." *Journal of Marketing Research*, 42(2), 2005.
- Blattberg, R. C., Briesch, R., and Fox, E. J. "How Promotions Work." *Marketing Science*, 14(3), 1995.
- van Heerde, H. J., Leeflang, P. S. H., and Wittink, D. R. "Decomposing the Sales Promotion Bump with Store Data." *Marketing Science*, 23(3), 2004.
- Talluri, K. T., and van Ryzin, G. J. *The Theory and Practice of Revenue Management*. Springer, 2004.
- Phillips, R. L. *Pricing and Revenue Optimization*, 2nd ed. Stanford University Press, 2021.
- Ferdows, K., Lewis, M. A., and Machuca, J. A. D. "Rapid-Fire Fulfillment." *Harvard Business Review*, 82(11), 2004.

**Industry**

- OECD, *Algorithmic Pricing and Competition in G7 Jurisdictions*, October 2025. https://www.oecd.org/competition/ — accessed 27 October 2026. `[VERIFY: exact title, publication date, and permalink.]`
- U.S. DOJ and FTC statements of interest on algorithmic pricing, 2025–2026. Enforcement position, not settled law in all circuits.
- J.C. Penney Company, fiscal 2012 results and 2012–2013 proxy statements. Company reporting; comparable-store sales decline of about 25% is widely reported.
- Inditex, annual reports and the "Rapid-Fire Fulfillment" case literature. Company-reported markdown levels.
- Nucleus Research, *2025 Supply Chain Planning Technology Value Matrix*, July 2025. Analyst view on planning vendors, including pricing and promotion modules. https://nucleusresearch.com/ — accessed 27 October 2026. `[VERIFY: exact report title and permalink.]`

**Tools and datasets**

- `statsmodels` and `linearmodels` documentation for OLS, fixed effects, and IV estimation. https://www.statsmodels.org and https://bashtage.github.io/linearmodels/ — accessed 27 October 2026.
- Dunnhumby Complete Journey and Corporación Favorita datasets for promotion and elasticity labs. Accessed 27 October 2026.
- M5 (Walmart) dataset for price and promotion panels. https://www.kaggle.com/competitions/m5-forecasting-accuracy — accessed 27 October 2026.

> **Note:** An elasticity is a statement about a specific price change, in a specific context, for a specific population. Quoting one without those three qualifiers is not an estimate; it is a guess with decimal places.

---

> **Next:** [Module 17](17_assortment_category_retail_analytics.md) — from the price of one item to the composition of the shelf: assortment, basket affinity, space, and the economics of variety.