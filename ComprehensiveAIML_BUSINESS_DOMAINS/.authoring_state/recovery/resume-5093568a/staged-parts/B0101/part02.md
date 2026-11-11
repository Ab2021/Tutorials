## 5. Data, Tools & Method

### 5a Data Specification

**Unit of analysis.** The *treated store-week-item*, where item means the promoted UPC (not the brand, not the category) and week means the retailer's 4-5-4 fiscal week in which the shelf price changed. One row = one store × one fiscal week × one UPC. A store-week appears once per promoted item; a store with three promoted items contributes three rows per week.

**Primary key.** `store_id × fiscal_week × upc × price_zone_version`. The fourth component is necessary because a mid-week price-zone correction can produce two legitimate price states in one week; without it the join silently doubles or halves units depending on whether the correction is earlier or later than the transaction timestamp.

**Population, and its exact definition.** All store-weeks in the 13 planning stores plus 27 comparison stores, restricted to stores with at least 40 weeks of continuous item-level sales history for the UPC, and to weeks in which the UPC was authorised for sale. This yields, in the constructed example, 40 stores × 8 analytic weeks = 320 store-weeks per item; with three promoted items the panel has 960 rows. Stores below the 40-week history floor are excluded, not imputed: a store that opened during the window has no usable pre-period, and including it forces the parallel-trends assumption to carry a level difference it cannot carry.

**Time fields, all retained separately.**

| Field | Meaning | Why it must be separate |
|---|---|---|
| `event_start`, `event_end` | the dates the shelf price was actually the promoted price | defines the treated window; must come from the price file, not the deal sheet |
| `circular_drop` | the date the offer became visible to shoppers | anticipation contamination is measured against this, not against `event_start` |
| `txn_timestamp` | the register timestamp of the sale | the only timestamp that defines when demand occurred |
| `warehouse_load_ts` | when the transaction landed in the analytic store | explains late-arriving rows without changing event time |
| `funding_recognised_ts` | when the supplier settlement was booked | determines the maturity of the profit figure |
| `snapshot_date` | the extract date used by the analysis | the replay handle; without it no figure can be reproduced |

**Cut-off timestamp.** Decisions for the week-14 planning meeting are taken on the extract dated the Friday of fiscal week 11, 21 days before the earliest commitment date. Any row whose `warehouse_load_ts` is after that snapshot is invisible to the decision — including the last two days of the previous event, which is a real and quantifiable blind spot rather than a modelling flaw.

**Grain mismatches that must be declared.** Supplier funding is contracted per case of 12 units; the analysis works per unit; POS works per scanned unit including multi-packs of a different UPC. The conversion factors (12 units per case for the standard pack, 6 for the club pack) are inputs with their own error, and a 1% error in the case-pack conversion is a 1% error in the per-unit funding credit, which is 0.4% of the promo price. Small, but it must be stated because it changes the sign of nothing and the second decimal of everything.

**What is not available at decision time.** The following are genuinely unavailable and must never enter the rule: post-event restated cost of goods (final settlement lands 6–10 weeks later); the competitor's price in the two weeks after the event; the item's later markdown history; total market category volume from syndicated data (lagged 4 weeks); and any customer-level purchase sequence for non-loyalty transactions, which is roughly 62% of transactions in the constructed example. A decision rule that appears to need any of these is mis-specified.

**Units, all explicit.** Units in eaches (not cases, not pallets); prices and margins in USD per unit at two decimal places; volume per store-week, not per week total; discount depth as a fraction of the regular shelf price in the same store; statistical precision in units and, separately, as a percentage of the counterfactual.

### 5b Tools & Stack

The method needs four capabilities and nothing exotic:

1. **A versioned extract.** Snapshot tables with an immutable `snapshot_date`, a row count, and a checksum recorded in the decision log. Without this, replication is impossible and disputes cannot be settled.
2. **A deterministic transformation layer.** SQL or a dataframe script that maps raw POS to the store-week-UPC grain using explicit, version-controlled rules for returns, voids, and multi-pack normalisation. No manual spreadsheet steps: a manual step is an unfalsifiable step.
3. **A panel estimator.** Difference-in-differences on a matched panel, plus a synthetic-control fallback for the cases where no clean comparison group exists. Both are ordinary least squares with fixed effects; there is no need for anything more elaborate at this grain.
4. **A decision record.** A one-page artefact per event with numerator, denominator, window, comparator, confidence interval, threshold, and the decision taken. This is the deliverable; the model is plumbing.

Deliberately absent: a machine-learned uplift model. A causal-forest or meta-learner could be justified with many more treated units per event and a stable feature set, but with 13 treated stores the estimation error of a flexible learner exceeds the effect being measured. The honest statement is that the design, not the algorithm, limits what can be learned.

### 5c Method Summary

**Step 1 — freeze the contract.** Before looking at any outcome, write down: the metric (incremental gross profit per store-week, net of substitution and carryover), the window (event weeks plus two carryover weeks), the comparator (matched comparison stores, difference-in-differences), the threshold (positive at the 80% interval, and above the category's hurdle of $150 per store-week), and the decision rule that maps the outcome to one of the six options.

**Step 2 — build the counterfactual.** Estimate the treated stores' would-be sales as (treated pre-period level) × (comparison-group growth ratio over the same weeks). The growth ratio is the only thing borrowed from the comparison group; the level stays treated. This is why a level difference between treated and comparison stores is tolerable and a trend difference is not.

**Step 3 — estimate the effect and its uncertainty.** Point estimate, standard error clustered at the store level, and a pre-period placebo test. The placebo test is not decoration: if the same estimator run on the four pre-event weeks produces a "significant" effect, the design has failed and the post-period number is uninterpretable regardless of its size.

**Step 4 — convert units to money.** Apply the effective-dated cost and the funding split, then subtract substitution and add carryover. Order matters: converting to money before adjusting for substitution produces a number that is arithmetically consistent and economically wrong.

**Step 5 — compare against the pre-registered threshold and choose.** Report the decision, the number, and the interval. If the interval spans the threshold, the answer is "defer with a named test", not "repeat" and not "stop".

**Step 6 — archive.** Store the snapshot handle, the code version, the contract, and the outcome. The next decision on this item cites this record instead of recomputing from scratch, which is how the organisation accumulates knowledge rather than re-deriving it annually.

### 5d Provenance, Quality, Exclusions, Leakage

**Provenance.** Three source systems are joined: the price file (authoritative for the shelf price actually charged), the POS transaction feed (authoritative for units and revenue), and the deal sheet (authoritative for funding terms). When they disagree, the price file wins on price, POS wins on volume, and the deal sheet wins on funding — but the disagreement itself is logged as a quality signal, because a systematic disagreement means one of the three has a broken process.

**Quality gates, all measured rather than asserted.**

| Gate | Threshold | Action when breached |
|---|---|---|
| Item-level sales present in all 8 analytic weeks | ≥95% of store-weeks | exclude the item from the causal analysis; report descriptively |
| Out-of-stock rate during the event | ≤8% of store-days | adjust demand or exclude the store-week; report the rate |
| Duplicate transaction IDs | 0 | block; a duplicate inflates both numerator and denominator |
| Price file rows with a null regular price | ≤1% | quarantine; a missing regular price makes depth uncomputable |
| Comparison stores with any promotional price on the item | 0 | reclassify as treated or drop |
| Funding settlement lag | ≤10 weeks for the final figure | report provisional, marked provisional |

**Exclusions and their rate.** The starting panel is 40 stores × 8 analytic weeks = 320 store-weeks. Removing stores with insufficient history (2 stores, 16 store-weeks), store-weeks above the 8% out-of-stock gate (9), and one item repack (the 400g pack became 375g mid-window, a 6.25% size reduction that would read as a price increase, so that item leaves the causal analysis entirely) leaves 295 usable store-weeks — a 7.8% exclusion rate, computed as 25/320. The rate is reported because an 8% exclusion concentrated in low-volume stores is itself informative: it means the surviving sample over-weights large stores, and the per-store-week effect will not generalise to the small-format estate without a further adjustment.

**Leakage, three distinct kinds.**

- *Temporal leakage.* Using the post-event restated cost, or the supplier's final settlement, in a decision taken 21 days before the event. This is the most common and the most damaging, because it is invisible in the code — the table simply contains a column that did not exist at decision time.
- *Group leakage.* Comparison stores that share a price zone, a regional circular, or a distribution centre with treated stores. The contamination is not in the row, it is in the relationship between rows, so no per-row validation catches it.
- *Target leakage through selection.* Choosing which events to evaluate because their results are already known. An evaluation portfolio assembled after the outcomes is a biased sample no matter how good each individual estimate is.

**Maturity.** The profit figure has three legitimate states: *provisional* (POS complete, funding estimated), *settled* (funding reconciled), and *final* (returns and markdowns closed). Every reported number carries its state. Mixing states across events in one league table is the single most common way a promotion scorecard becomes untrustworthy.

### 5e Reproducibility & Method-Selection Ledger

| Decision | Chosen | Rejected alternative | Why rejected |
|---|---|---|---|
| Identification | matched difference-in-differences | pre/post on treated stores only | attributes seasonality and trend to the event; the bias is one-directional and large |
| Identification fallback | synthetic control from donor pool | interrupted time series on 13 stores | 13 stores give too few pre-periods for a stable seasonal model |
| Matching | price zone + store format + pre-period volume decile | propensity score on 30 covariates | with 13 treated stores, a 30-covariate score is overfit and unverifiable |
| Outcome | incremental gross profit per store-week | units sold | units ignore the margin rate and the funding split, so it cannot order the options |
| Window | event + 2 carryover weeks | event weeks only | pantry loading displaces post-event volume; a one-week window reports it as profit |
| Inference | store-clustered standard errors | transaction-level standard errors | transactions within a store are correlated; transaction-level errors are roughly 3× too narrow |
| Depth handling | realised price from POS | advertised depth from the circular | circular depth is wrong on any item with a mixed price ladder |

**Reproducibility package.** Snapshot handle, transformation script hash, cost table version, funding terms version, seed (for any bootstrap), and the contract text. A second analyst with these five artefacts must be able to reproduce the point estimate to the cent. If they cannot, the number is not a measurement, it is an opinion with decimal places.

**What this method cannot deliver, stated in advance.** It cannot attribute a national event with no untreated stores to a causal effect. It cannot measure a mechanic that always co-occurs with a price cut. It cannot detect an effect smaller than about 14.5% of baseline with 40 stores per arm (§6, D-5). It cannot separate retailer-funded from supplier-funded effects when the funding varies by account. Those four limitations cover the majority of real promotion questions, which is why the honest output is often "defer" — and why the deferral must name the test that would resolve it.

Takeaway: the method is a contract about what will be believed, written before the data can flatter it.

## 6. Analysis Walkthrough

Five derivations below, each recomputed by hand. All inputs are ILLUSTRATIVE. The item is a 500g pack with a regular shelf price of $4.00 and a landed cost of $2.20, so the regular unit margin is $1.80 on a 45.0% margin rate. The promoted price is $3.20, a 20% cut.

### 6.1 D-1: the counterfactual and the naive bias

```text
FORMULA [D-1]  uplift = (Q1 - k*Q0) / (k*Q0),  k = C1/C0
WHERE:
  Q0 [units/store-week] treated pre-period mean, event excluded
  Q1 [units/store-week] treated event-period mean
  C0, C1 [units/store-week] comparison-group means, same weeks
  k [-] growth ratio of the comparison group, dimensionless, k > 0
CONSTRAINT: k estimated on a comparison group with no promotional price on the item
            in any week of the window, and with pre-period parallel trends not rejected.
NUMERICAL EXAMPLE:
  Q0 = 1,000 ; Q1 = 1,600 ; C0 = 2,000 ; C1 = 2,240
  k = 2,240 / 2,000 = 1.12
  Q_cf = 1.12 * 1,000 = 1,120 units
  uplift = (1,600 - 1,120) / 1,120 = 480 / 1,120 = 0.428571 = 42.86%
INTERPRETATION: the raw sales increase is 1,600/1,000 - 1 = 60.00%. Of that, 12.00
  percentage points are the comparison group's movement, which has nothing to do with
  the event. The event-attributable effect is 42.86%, and the difference between 60.00%
  and 42.86% is not noise - it is a deterministic consequence of k > 1.
```

Arithmetic check by a second route: implied counterfactual growth in units is 120 units. 1,600 − 1,000 − 120 = 480 incremental units, and 480/1,120 = 42.857%. The two routes agree.

**Where D-1 breaks.** If k is estimated on comparison stores that themselves took a promotional price, k is inflated, Q_cf is inflated, and the uplift is understated — the error is conservative but unbounded. If the comparison group is in a declining region while treated stores are in a growing region, k < 1 and the uplift is overstated by the regional divergence. The discriminating check is a pre-period event-study plot: run D-1 on the four weeks before the event, treating the two most recent as "post". A correct design returns an effect indistinguishable from zero; a design with a trend problem returns a large spurious effect. In the constructed example the placebo estimate is +2.1 units with a standard error of 41, which is consistent with zero and supports the design.

### 6.2 D-2: the break-even uplift, and why the sales lift is the wrong number

```text
FORMULA [D-2]  delta_profit = dQ*m_p - Q_cf*(m_r - m_p)
               break-even dQ* = Q_cf*(m_r - m_p)/m_p
WHERE:
  m_r [USD/unit] regular unit margin  = p_r - c
  m_p [USD/unit] promoted unit margin = p_p - c + f, f = per-unit supplier funding
  dQ [units/store-week] incremental units = Q1 - Q_cf
  Q_cf [units/store-week] counterfactual level from D-1
CONSTRAINT: m_p > 0. The identity assumes the same cost per unit in both states and
  no fixed (lump-sum) funding; a lump-sum payment is additive and outside the formula.
NUMERICAL EXAMPLE (unfunded):
  p_r = 4.00 ; c = 2.20 ; p_p = 3.20 ; f = 0.00
  m_r = 4.00 - 2.20 = 1.80 ; m_p = 3.20 - 2.20 = 1.00
  Q_cf = 1,120 ; dQ = 1,600 - 1,120 = 480
  delta_profit = 480*1.00 - 1,120*(1.80 - 1.00) = 480 - 896 = -416.00 USD
  break-even dQ* = 1,120*(0.80)/1.00 = 896 units -> 896/1,120 = 80.00% uplift required
INTERPRETATION: the event sells 60% more units and loses $416 per store-week. The
  break-even requirement is an 80% incremental lift, exactly (m_r - m_p)/m_p, because a
  20% price cut on a 45% margin rate removes 44.4% of the unit margin and that must be
  recovered on every unit the store was going to sell anyway.
```

Second route to the same number, from the counterfactual rather than from the identity: no-promotion profit = 1,120 × 1.80 = $2,016. Promo profit = 1,600 × 1.00 = $1,600. Difference = −$416. The identity and the level comparison agree to the cent.

**A trap worth naming.** The period-over-period profit change is 1,600 − 1,000×1.80 = 1,600 − 1,800 = −$200. That is a different quantity: it includes the comparison group's 12% growth, which the promotion did not cause. Reporting −$200 as "the promotion's profit impact" understates the loss by $216 — precisely the margin that the 120 organically-grown units would have earned at the regular price. A promotion that loses $416 can show a period-over-period loss of only $200 in a growing category, and in a fast-growing category it can show a period-over-period *gain* while destroying value. This is the single most common way promotion scorecards flatter events.

**Now add funding.** The supplier agrees to fund 50% of the discount, i.e. $0.40 per unit sold during the event.

```text
FORMULA [D-2b]  m_p = p_p - c + f,  f = share_s * (p_r - p_p)
WHERE: share_s [-] supplier share of the discount, 0 <= share_s < 1.
NUMERICAL EXAMPLE: f = 0.5 * 0.80 = 0.40 ; m_p = 3.20 - 2.20 + 0.40 = 1.40
  delta_profit = 480*1.40 - 1,120*(1.80 - 1.40) = 672 - 448 = +224.00 USD
  break-even dQ* = 1,120*0.40/1.40 = 320 units -> 320/1,120 = 28.57% uplift required
INTERPRETATION: the same 42.86% observed uplift clears a 28.57% requirement, so the
  event is worth +$224 per store-week. The funding did not make the promotion better -
  it changed which promotion was being evaluated. The retailer gave away $0.40 of margin
  per unit instead of $0.80, so the hurdle halved.
```

Check: 672 − 448 = 224 ✓. Break-even 1,120 × 0.2857 = 320.0 ✓, and at dQ = 320 the profit is 320×1.40 − 448 = 448 − 448 = 0 ✓.

**Where D-2 breaks.** Set p_p = $2.00 with c = $2.20. Then m_p = −0.20 and the break-even formula returns 1,120 × 2.00 / (−0.20) = −11,200 units, a negative quantity that is not a quantity of anything. The formula's constraint m_p > 0 has been violated: when the promoted price is below cost there is no volume that recovers the loss, and the correct statement is "unprofitable at any volume", not a number. A second break: when funding is a lump-sum bill-back of $5,000 conditional on a display, that money is invisible to m_p, so D-2b reports the event as loss-making while the retailer's cash position improves by $5,000 − $416 = $4,584 per store-week-equivalent. The formula is not wrong; it is answering the per-unit question while the decision turns on the fixed-plus-variable question, and the decision record must say which.

### 6.3 D-3: substitution and cannibalisation

Promoted items do not sell into a vacuum. Some of the promoted unit's lift is volume transferred from a sibling SKU, from the same brand's larger pack, or from the retailer's own-brand equivalent. Transfer is not incremental: it moves margin around inside the category, and a per-item evaluation that ignores it will approve a portfolio of events that collectively shrink category profit.

```text
FORMULA [D-3]  delta_sub = sum_j (dQ_j * m_j)   over j in substitute set, dQ_j < 0
               delta_net = delta_profit + delta_sub
WHERE:
  dQ_j [units/store-week] effect on substitute j, estimated with the same D-1 design
  m_j [USD/unit] substitute j regular margin, valued at the margin the unit would
         have earned absent the promotion (not at its promoted margin, which is zero
         if j is not promoted)
CONSTRAINT: the substitute set is fixed before looking at the data. Searching the
  category for whatever declined and calling it cannibalisation manufactures a loss.
NUMERICAL EXAMPLE:
  premium 800g pack: dQ = -150 units, m = 2.40 -> -360.00 USD
  own-brand 500g:    dQ =  -90 units, m = 0.90 ->  -81.00 USD
  delta_sub = -360.00 + (-81.00) = -441.00 USD
  delta_net = 224.00 - 441.00 = -217.00 USD per store-week
INTERPRETATION: the funded event is +$224 on the promoted item and -$217 once the
  transferred volume is priced at the margin it would have earned. The category is
  essentially flat while the promoted item's scorecard reads positive - the classic
  portfolio illusion of item-level promotion evaluation.
```

**Why the substitute margin must be the regular margin.** Suppose the premium pack was also on a shallow promotion during the event, with m = $1.90. Valuing its 150 lost units at $1.90 gives −$285 instead of −$360, and the net becomes −$61 instead of −$217. The rule is to use the margin the unit would have earned *had the evaluated promotion not run*, which in the counterfactual means the pack's own regular margin unless it had an independent, separately justified promotion. Using the observed promoted margin double-counts the discount: the retailer loses the regular margin on the transferred unit, not the already-reduced margin.

**Where D-3 breaks.** If the substitute set is not fixed in advance, the estimate is a researcher degree of freedom with a large range: in the constructed category there are 11 plausible substitute SKUs, of which 4 declined. Choosing the 2 with the largest declines yields −$441; choosing the 4 yields −$512; choosing none yields $0. The pre-registration is what makes the number a measurement.

### 6.4 D-4: carryover, pantry loading, and the full-window answer

A price promotion shifts purchase timing. Households that buy 2 units a month buy 5 during the event and 0 for the next six weeks. The event window captures the forward-buy as incremental volume; the carryover window captures the offset. Evaluating on the event window alone is not a conservative approximation — it is a biased estimator whose bias grows with the depth of the discount and the storability of the item.

```text
FORMULA [D-4]  delta_window = delta_net + sum_t (dQ_t * m_u), t = 1..T post weeks
WHERE:
  dQ_t [units/store-week] post-event deviation from the D-1 counterfactual, negative
        under pantry loading
  m_u [USD/unit] the margin actually earned on the displaced unit. Use m_r if the
        displaced unit would have sold at the regular price after the event; use the
        post-event markdown margin if the category was marked down.
CONSTRAINT: T is fixed in the contract (2 weeks here). Extending the window after
  seeing the data lets the analyst find the window that produces the desired sign.
NUMERICAL EXAMPLE:
  post weeks 1-2: counterfactual 2,000 units, observed 1,600 units
  dQ_total = 1,600 - 2,000 = -400 units
  delta_carry = -400 * 1.80 = -720.00 USD
  delta_window = -217.00 - 720.00 = -937.00 USD per store-week-equivalent
INTERPRETATION: over the event plus two weeks, the promotion destroys $937 per
  store-week-equivalent of gross profit while increasing event-window sales by 60%.
  Per unit of discount given (1,600 * 0.80 = $1,280), the loss is 73.2%.
```

Cross-check the carryover direction with a physical quantity: the event sold 480 incremental units. If households are forward-buying, the post-event shortfall should be a fraction of the forward-bought volume, not larger. 400 displaced units versus 480 incremental units is a 83.3% displacement rate at the item level, which is high but internally consistent for a shelf-stable item with a two-week consumption cycle. A displacement rate above 100% would mean the calculation has a sign or grain error — that is a usable sanity gate, and it is the one to run first when a carryover estimate looks implausible.

**Where D-4 breaks.** The cleanest failure: the post-event weeks contain an independent event on the substitute item, which pulls demand back and masks the shortfall. The measured displacement then looks small, the full-window number looks good, and the error is in the direction that approves more promotions. The discriminating check is a calendar audit of every promoted item in the category across the carryover window, which is cheap and is skipped more often than any other check in this discipline.

### 6.5 D-5: precision, and whether the estate can answer the question

An effect estimate without its precision is a claim that the organisation can measure anything it likes. With store-level clustering, the relevant variance is between stores, and the achievable minimum detectable effect is set by the number of stores, not by the number of transactions.

```text
FORMULA [D-5]  SE_DiD = sigma * sqrt(2/n) ;  MDE = z * SE_DiD
               n_required = 2 * sigma^2 * z^2 / MDE^2
WHERE:
  sigma [units/store-week] between-store standard deviation of the outcome change
  n [stores] per arm
  z [-] 2.802 for 80% power at two-sided alpha = 0.05, for a normal approximation
  MDE [units/store-week] smallest true effect detected with that power
CONSTRAINT: equal arm sizes; sigma estimated from the pre-period change distribution,
  not from the post-period (which contains the treatment). If arms are unequal the
  formula uses 1/n_t + 1/n_c in place of 2/n.
NUMERICAL EXAMPLE:
  sigma = 260 ; n = 40
  SE_DiD = 260 * sqrt(2/40) = 260 * 0.223607 = 58.14 units
  MDE = 2.802 * 58.14 = 162.9 units -> 162.9/1,120 = 14.5% of the counterfactual
  observed effect 480 units = 8.3 standard errors -> detectable
  target effect 112 units (10%): n = 2*67,600*7.849/12,544 = 84.6 -> 85 stores per arm
INTERPRETATION: the estate (40 stores per arm) can detect a 14.5% uplift and cannot
  detect a 10% uplift. Any decision that hinges on distinguishing a 6% from a 10%
  effect is not answerable with this estate, and the honest output is "defer until a
  larger estate or a pooled multi-item test exists" - not a point estimate with a
  wide interval presented as a finding.
```

Verify the variance arithmetic: 260² = 67,600. 2/40 = 0.05. 67,600 × 0.05 = 3,380. sqrt(3,380) = 58.14 ✓. 2.802 × 58.14 = 162.9 ✓. For n = 85: 135,200/85 = 1,590.6, sqrt = 39.88, × 2.802 = 111.7, i.e. just under the 112-unit target — consistent with the closed form giving 84.6 and rounding up to 85.

**Where D-5 breaks.** sigma estimated from the post-period includes treatment heterogeneity, so it overstates the noise and produces a conservative MDE — acceptable but wasteful. sigma estimated from a single pre-period pair understates it, because a single difference has no averaging over weeks, and produces an MDE that is too small, which is the dangerous direction: it makes an underpowered test look adequate. The correct sigma uses the distribution of week-over-week changes across the whole pre-period, pooled across stores, and it is recomputed whenever the estate changes.

### 6.6 Boundary case: realised depth versus advertised depth

The circular says 20% off. In 30% of promoted store-weeks the item was already on a local markdown from $3.80 when the event started, so the realised cut was $3.80 → $3.20 = 15.79%, not 20%.

```text
FORMULA [D-6]  depth_realised = 1 - p_p/p_pre ;  breakeven = (m_pre - m_p)/m_p
WHERE: p_pre is the price in effect in the 4 weeks before the event in that store;
       m_pre = p_pre - c. This replaces m_r whenever the pre-period was not regular.
NUMERICAL EXAMPLE:
  marked-down stores: p_pre = 3.80 -> m_pre = 1.60 ; p_p = 3.20 -> m_p = 1.00 (f = 0)
  breakeven = (1.60 - 1.00)/1.00 = 60.00% (versus 80.00% in the clean stores)
  discount given = 1,600*0.60 = 960.00 USD (versus 1,280.00 USD)
INTERPRETATION: the shallower realised depth lowers the hurdle and lowers the cash
  given away. The formula is still valid because m_pre > 0; what has changed is the
  reference price, and using the advertised depth of 20% against the marked-down
  baseline would overstate both the hurdle and the discount by a third (0.60 vs 0.80
  per unit, 960 vs 1,280 per store-week).
```

This is the boundary that catches most automated scorecards: they compute depth as `1 - promo_price/regular_list_price`, where regular_list_price is the chain-wide file price rather than the price the store was charging. On items with active local markdowns, that overstates depth by exactly the markdown percentage, and the error is systematic because markdowns cluster on the items the chain is already trying to move.

Takeaway: the counterfactual, the hurdle, the transfers, the timing offset, and the precision are five separate numbers, and each one can flip the decision on its own.
