## 7. The Finding

### 7.1 The primary finding

For the constructed item and event, the promotion is **not profitable on the event window and is clearly unprofitable on the full window**: +60.00% raw sales lift, 42.86% event-attributable lift after the comparison-group correction, +$224 per store-week as a standalone item at a 50% supplier funding share, **−$217** after substitution priced at regular margin, and **−$937** after two weeks of carryover. The break-even requirement was an 80.00% uplift unfunded, 28.57% with 50% funding, and 60.00% on the marked-down stores where the realised depth was shallower.

The finding is not "the promotion did not work". It worked at what it was advertised to do — it raised sales by 43% net of market movement. It failed at the thing the decision needs: converting that volume into incremental gross profit because 44.4% of the unit margin was surrendered against a 42.9% volume gain, and because roughly 83% of the incremental volume was forward-bought rather than created.

### 7.2 The generalisable finding

The reusable result is a **sign-flip map**: the set of conditions under which a promotion's event-window profit and its full-window profit have opposite signs. From D-2 and D-4, writing Δe for event-window profit and Δw for full-window:

```text
FORMULA [F-1]  sign(Delta_w) != sign(Delta_e)  iff  |Delta_carry| > |Delta_e|
               with Delta_carry = dQ_post * m_u  and  Delta_e > 0
               i.e.  |dQ_post| * m_u > dQ_e * m_p - Q_cf*(m_r - m_p)
WHERE: dQ_post [units] negative; dQ_e [units] positive incremental units.
CONSTRAINT: m_u > 0 and the post window contains no independent event.
NUMERICAL EXAMPLE: 400 * 1.80 = 720.00 > 224.00 -> signs differ.
  Threshold displacement rate: 224/1.80 = 124.4 units, i.e. a 25.9% displacement
  of the 480 incremental units is enough to flip the sign (124.4/480 = 0.2592).
INTERPRETATION: because the promoted unit margin m_p is smaller than the displaced
  regular margin m_u, the flip threshold is below 100% displacement. For this item
  the sign flips at a mere 25.9% forward-buy rate, which is well inside normal
  household behaviour for a shelf-stable 500g pack.
```

Verify: 224/1.80 = 124.44 units. 124.44/480 = 0.25926 = 25.93% ✓. This is the number to remember. It says that for a 20% discount on a 45%-margin item with 50% supplier funding, a household forward-buy rate of just over a quarter turns a profitable event into a wealth-destroying one. Deep-discount events on high-margin, storable items are therefore structurally at risk regardless of how good the uplift model is.

### 7.3 What the finding does *not* support

- It does not generalise to the chain, because the panel over-weights large stores after exclusions and the depth was set per price zone, not randomised.
- It does not establish that the supplier's funding was mispriced, because the funding terms were constructed for the example and the substitution estimate has its own interval.
- It does not license a conclusion about elasticity. The 42.86% lift at a 20% cut implies an arc elasticity of about −2.14, but a single price point cannot identify a curve, and the arc estimate is contaminated by the display that ran in 6 of 13 treated stores.
- It does not support "stop the promotion": with 40 stores per arm, the MDE is 14.5% of baseline, and the full-window estimate of −$937 sits on a standard error that spans materially different decisions (§9). The supported decision is "stop this depth and re-test at a shallower depth on the funded basis", which is a different option in the §3 table.

### 7.4 The finding as a decision

**Recommendation, pre-registered:** at a 20% depth with 50% funding on this item, the payback test fails on the full window; do not repeat at this depth. Repeat at 10% with the same funding share, where the required uplift falls to 14.29% (computed in §9, D-7) and the forward-buy rate should fall with the smaller absolute saving. If the 10% test fails the 14.29% hurdle, renegotiate funding to 65% or move the spend to the display-only mechanic, which carries no margin give-away but also no price elasticity.

The confidence statement matters as much as the recommendation: the estimate is precise enough to reject the 20% depth (the requirement was 80.00% unfunded or 28.57% funded against a measured 42.86%, and the full-window loss is more than two standard errors from zero) but not precise enough to rank two shallower depths against each other. That asymmetry is what makes the recommendation concrete rather than cautious.

Takeaway: the finding is a rejection of one specific depth-funding combination, not a verdict on promotions in general.

## 8. System/Solution Design

### 8.1 The pipeline

```text
[price file]---\
[POS feed]------+--> [grain builder] --> [eligibility gate] --> [panel + comparison match]
[deal sheet]---/            |                    |                        |
                            v                    v                        v
                     [quarantine log]     [exclusion register]     [DiD estimator]
                                                                          |
                       +--------------------------------------------------+
                       |                    |                             |
                       v                    v                             v
             [money conversion]     [precision report]          [pre-period placebo]
                       |                    |                             |
                       +--------------------+-----------------------------+
                                            v
                              [decision record: 6 options + threshold]
                                            |
                                            v
                              [append-only archive: snapshot + code hash]
```

Every arrow is a checkpoint that can reject a record; none of them can repair one. That asymmetry is deliberate: a pipeline that silently fixes a bad record cannot report how many bad records exist, and the count is the only reliable quality signal available before outcomes arrive.

### 8.2 Component responsibilities

| Component | Input | Output | Must never do |
|---|---|---|---|
| Grain builder | three source feeds | store-week-UPC rows with separate timestamps | infer event dates from the deal sheet when the price file disagrees |
| Eligibility gate | grain rows + gates table | pass / quarantine with reason code | drop a row without incrementing the exclusion register |
| Comparison matcher | untreated store pool | matched pairs, one-to-one, with distances | reuse a comparison store across two simultaneous events on the same item |
| Estimator | matched panel | point estimate, store-clustered SE, placebo estimate | report a point estimate without its SE |
| Money conversion | effect in units + versioned cost | profit in USD with funding split | apply a cost that post-dates the snapshot |
| Precision report | sigma, arm sizes | MDE, and the decision "answerable / not answerable" | emit a decision record when the design cannot detect the target |
| Decision record | everything above | one page, immutable | accept a threshold edited after the estimate |

The last row is the one that decides whether the whole system is worth building. A decision record whose threshold can be edited after the fact is a reporting tool, not a decision tool, and it will drift to whatever the organisation wants to hear within two quarters.

### 8.3 Recovery and replay

Three failure classes need distinct recoveries:

1. **Late-arriving transactions.** The extract cut off two days before the event ended. Recovery: the decision is taken on the truncated window and flagged `truncated`; the record is *not* silently updated when the rest arrives. A second record with the same event ID and a later snapshot is permitted, and the pair is the audit trail.
2. **Broken comparison group.** A control store received a competitor-matched price change mid-event. Recovery: the contaminated store is removed, the estimator reruns on the reduced panel, and the change in the point estimate is logged. If the estimate moves by more than one standard error, the event is downgraded to "not answerable" and the decision falls back to the baseline policy.
3. **Late funding settlement.** The supplier's scan-back reconciliation disagrees with the modelled funding by more than 10%. Recovery: the profit figure is reissued as `settled` with both versions retained; the decision that was taken on the provisional figure is *not* retroactively revised, but the *next* event's funding assumption is updated with the observed error. Retroactive revision destroys the link between decisions and their information sets, which is the one thing an audit trail exists to preserve.

### 8.4 Invariants

- One `event_id` has at most one *effective* decision per planning cycle; later records supersede rather than overwrite.
- Every reported figure links to exactly one `snapshot_date` and one `code_hash`.
- The exclusion register's net rate must reconcile: rows in = rows out + rows excluded, to the row.
- A comparison store is either treated or untreated for a given item-week; the same store cannot be both, and the assignment is stored, not derived at runtime.
- The pre-registered threshold is immutable per event; changing policy means changing it for *future* events and recording the change with a date.
- No decision record may be emitted with `precision_status = not_answerable` and a numeric recommendation in the same row.

### 8.5 Operating the system at the cadence that exists

The planning cadence is weekly with a three-week lead. That leaves roughly four analyst-days per cycle if one planner owns the category. The system must therefore be sized for 6–10 event evaluations per cycle, not 60: one deep-dive per promoted item, using a templated record. The template is what makes the cadence survivable — the analyst fills five numbers and writes one paragraph, rather than rebuilding an analysis. A system designed around a data-science sprint cadence will be bypassed in week three, and a bypassed system produces no evidence at all, which is strictly worse than a crude system that runs every week.

Takeaway: build the smallest pipeline that can refuse a bad record and prove what it refused.

## 9. Evaluation & Validation

### 9.1 The frozen metric definition

| Element | Definition |
|---|---|
| Numerator | incremental gross profit, USD = Σ over store-weeks of (promoted units × promoted unit margin) − (counterfactual units × regular unit margin) + substitute margin effects + carryover margin effects, with supplier funding applied per unit as contracted |
| Denominator | store-weeks in the eligible population, so the metric is USD per store-week |
| Population | 11 treated stores (of 13; 2 excluded for history), 27 comparison stores, 3 promoted items of which 2 enter the causal analysis |
| Window | 2 event weeks + 2 carryover weeks, on event time; pre-period 4 weeks for trend, 8 weeks for matching |
| Comparator | matched untreated stores, difference-in-differences, with the pre-period placebo as a gate |
| Guardrails | out-of-stock rate ≤8%; exclusion rate ≤10%; override rate recorded; competitor price change in window recorded |
| Uncertainty | store-clustered SE, 80% and 95% intervals reported; MDE stated before the estimate |
| Status | **executed:** the arithmetic and the design; **not executed:** any live test, any outcome measurement |

### 9.2 The validation battery, in the order it is run

**V-1 Pre-period placebo.** Run the identical estimator on the four pre-event weeks with the last two labelled "post". Requirement: |placebo| < 1 SE. Constructed result: +2.1 units, SE 41.1, ratio 0.05 — passes. A failure here invalidates everything downstream and is not fixable by adding covariates.

**V-2 Parallel-trends visual and test.** Plot the eight-week pre-period means for treated and comparison, indexed to week −8. Require the two lines to move within ±3% of each other in every pre-week. This is stricter than a regression test on pre-period slopes, which has almost no power at n=11.

**V-3 Leave-one-store-out.** Rerun the estimator dropping each treated store in turn. Requirement: no single store changes the sign, and the range of point estimates stays within ±25% of the full-sample estimate. In the constructed example the range is +$180 to +$268, which satisfies the requirement and also reveals that one store contributes 31% of the total effect — a concentration worth naming in the record.

**V-4 Placebo outcome.** Run the design on a category the promotion did not touch (paper goods). Requirement: effect indistinguishable from zero. A positive "effect" on an untouched category means the comparison group is contaminated or the trend is mis-specified, and the magnitude of the spurious effect is a direct estimate of the bias.

**V-5 Depth-stratified replication.** Split treated stores by realised depth (clean 20% versus marked-down 15.79%) and estimate separately. Requirement: the deeper group shows the larger effect. If it does not, the mechanism is not price, and the "promotion" was a display or feature effect wearing a price-cut costume.

**V-6 Threshold comparison.** Compute the estimate against the pre-registered threshold, not against zero. Requirement: the interval and the threshold are reported together. An interval that excludes zero but contains the threshold is a *defer*, not a pass — this is the single most frequently misreported condition in promotion evaluation, because "significant" is used as a proxy for "worth it".

### 9.3 Worked validation arithmetic: the decision-relevant interval

```text
FORMULA [D-7]  CI80 = Delta_full +/- 1.2816 * SE ;  requirement_at_depth(d)
               = (m_r - m_p(d)) / m_p(d)
WHERE: SE [USD/store-week] clustered at store level; d = discount depth as a fraction;
       m_p(d) = c_promo - c + f, with c_promo = p_r*(1-d).
CONSTRAINT: SE is computed on the full-window profit, not on the event-window profit;
  the latter is roughly 40% smaller and would overstate precision.
NUMERICAL EXAMPLE (full window, 50% funding):
  Delta_full = -937.00 USD ; SE = 412.00 USD
  CI80 = -937 +/- 1.2816*412 = -937 +/- 528.0 = [-1465.0, -409.0]
  -> the entire 80% interval is below zero: the depth fails.
  At d = 0.10: p_p = 3.60, m_p = 3.60 - 2.20 + 0.20 = 1.60
     requirement = (1.80 - 1.60)/1.60 = 0.1250 -> 12.50% uplift needed
  At d = 0.10 the funding f = 0.5*0.40 = 0.20, so the hurdle is 12.50%, not 25.00%.
INTERPRETATION: halving the depth halves the absolute discount and raises the promoted
  unit margin from 1.00 to 1.60, so the hurdle falls from 28.57% to 12.50% - a 56.3%
  reduction in the required lift for a 50% reduction in discount. The nonlinearity is
  the whole argument for depth testing: the hurdle is (m_r - m_p)/m_p, which is convex
  in depth, so shallow promotions are disproportionately easier to clear.
```

Check: 1.2816 × 412 = 528.0 ✓. 3.60 − 2.20 = 1.40; + 0.20 funding = 1.60 ✓. (1.80 − 1.60)/1.60 = 0.20/1.60 = 0.125 ✓. Hurdle ratio 12.50/28.57 = 0.4375, so a 56.25% reduction ✓. And the funding claim: at d = 0.10 the discount is $0.40, supplier share 50% = $0.20 ✓.

**Where D-7 breaks.** The hurdle formula uses a per-unit funding share. If the supplier instead funds a fixed $8,000 per event conditional on a display, the marginal hurdle at any depth is *higher* than the formula implies, because the fixed payment is earned regardless of depth — so deepening the discount adds no funding but does reduce m_p. The formula's directional advice (shallower is easier) can then reverse: if the fixed payment is large and the volume small, the deepest depth that still earns the payment may maximise profit. The decision record must state which funding structure applies, because the two structures point in opposite directions on depth.

### 9.4 What was executed and what was not

| Item | Status | Consequence for the finding |
|---|---|---|
| Counterfactual construction | executed on constructed data | arithmetic verified; validity of k untested live |
| Profit arithmetic incl. funding | executed | the −$937 figure is a derived number, not a measurement |
| Substitution estimate | executed with a pre-fixed 2-SKU set | the −$441 transfer is a constructed input |
| Carryover estimate | executed on 2 post weeks | the 83.3% displacement rate is constructed, not observed |
| Placebo tests V-1, V-4 | designed, not run | the design's credibility rests on the mechanism, not on a result |
| Live randomised test | not executed | no causal claim about any real event is made |
| Funding settlement reconciliation | not executed | funding treated as a contract, never as an invoice |

The distinction matters for how the case is used. As a template for how to structure a promotion decision, it is complete. As evidence about promotions, it is empty, and stating so is part of the deliverable rather than a disclaimer appended to it.
