## 10. Action Taken

### 10.1 What was actually done

**Executed:** the decision contract was written and frozen before any outcome was examined; the panel was built at the store-week-UPC grain with the six time fields kept separate; the six derivations were recomputed by hand and reconciled by a second route; the precision calculation was run and returned `answerable` for the 14.5% MDE case and `not_answerable` for the 10% target.

**Not executed:** no live promotion was run, no store was randomised, no supplier was approached, no price file was touched. This section therefore describes a *plan of action* with its decisions, not a record of deployment. The distinction is load-bearing: every number in §11 is a projection from §6, not an observation.

### 10.2 The decision record as filed

| Field | Value |
|---|---|
| Event ID | IL-500G-W14-A (illustrative) |
| Decision | change depth (20% → 10%), keep mechanic, keep funding share pending renegotiation |
| Metric | incremental gross profit, net of substitution and 2-week carryover, USD per store-week |
| Threshold | > 0 at the 80% interval **and** ≥ $150 per store-week, pre-registered |
| Estimate | −$937 per store-week-equivalent on the full window; −$217 on the event window |
| Interval | 80% CI [−$1,465, −$409] |
| Precision status | answerable for a 14.5% effect; not answerable for a 10% effect |
| Reason code | `PAYBACK_FAIL_AFTER_CARRYOVER` |
| Supersedes | none (first record for this item) |
| Owner | promotion manager; countersigned by category buyer |

### 10.3 The three actions and their owners

**Action 1 — re-test at 10% depth on the same item, 2 event weeks, same 11 treated stores.** Owner: promotion manager. Due before the week-14 commitment. Why 10% and not 5%: at 5% the absolute saving per unit is $0.20, which is below the observed price-perception threshold for the category (shoppers in the constructed panel do not respond measurably below roughly 8% on this basket size), so the test would fail for the wrong reason — no response rather than no profit. Why not 15%: the requirement at 15% is 18.75%, and the estate cannot distinguish an 18.75% effect from a 33% effect, so the test's outcome would be ambiguous across the threshold.

**Action 2 — request the funding share be raised from 50% to 65% in exchange for the shallower depth.** Owner: category buyer. The offer is coherent for the supplier because the retailer's absolute discount falls from $1,280 to $640 per store-week, so the supplier pays more per dollar of retailer discount but less in total: at 50% of the 20% depth the supplier's exposure is $640; at 65% of the 10% depth it is $416. The supplier's total cost falls 35% while the retailer's hurdle falls further, which is the rare trade that both sides can defend internally.

**Action 3 — instrument the carryover window so displacement is measured rather than assumed.** Owner: analytics. Concretely: log household-level repeat purchase for the 38% of transactions with loyalty identification, and extend the post-event price audit to every promoted item in the category for the four weeks after the event. This is the action that makes the *next* decision cheaper, and its absence is why the current record had to assume an 83.3% displacement rate instead of measuring it.

### 10.4 What was deliberately not done

- **No chain-wide rollout decision.** The panel is 11 stores and excludes the small-format estate; a chain decision would extrapolate a per-store-week figure across a store mix whose response is unknown.
- **No change to the comparison group.** The contaminated control store discovered in V-3 was left in the panel and flagged rather than removed, because removing it after seeing the estimate is a degree of freedom, and the honest treatment is to report both figures. The record shows the estimate with and without it: +$224 and +$262.
- **No vendor or model procurement.** A flexible uplift learner is not the binding constraint; 11 treated stores are. Buying a model would not change the MDE.

Takeaway: the action is a narrower, better-instrumented re-test, not a bigger analysis of the same event.

## 11. Outcome & Results

### 11.1 Status

**not-measured.** No promotion was executed, no store was randomised, and no supplier settlement exists. The figures below are projections from the derivations in §6 using constructed inputs, and are labelled as such at every appearance. Any reader who needs an observed promotion effect must go elsewhere; this case contributes a decision structure and a set of checked identities.

### 11.2 The projected results table

| Quantity | Value | Basis | Unit |
|---|---|---|---|
| Raw sales lift | +60.00% | 1,600/1,000 − 1 | fraction of pre-period |
| Comparison-group growth | 12.00% | 2,240/2,000 − 1 | fraction |
| Event-attributable lift | 42.86% | D-1 | fraction of counterfactual |
| Incremental units | 480 | 1,600 − 1,120 | units/store-week |
| Discount cash given | $1,280 | 1,600 × $0.80 | USD/store-week |
| Break-even uplift, unfunded | 80.00% | D-2 | fraction |
| Break-even uplift, 50% funded | 28.57% | D-2b | fraction |
| Item-level profit, funded | +$224 | D-2b | USD/store-week |
| Substitution effect | −$441 | D-3 | USD/store-week |
| Carryover effect | −$720 | D-4 | USD/store-week |
| Full-window profit | −$937 | D-3 + D-4 | USD/store-week-equivalent |
| Sign-flip displacement rate | 25.93% | F-1 | fraction of incremental units |
| 80% interval, full window | [−$1,465, −$409] | D-7 | USD/store-week |
| MDE at 40 stores/arm | 14.5% | D-5 | fraction of counterfactual |
| Requirement at 10% depth | 12.50% | D-7 | fraction |

### 11.3 How to read the table without over-reading it

Three columns carry equal weight: the value, the basis, and the unit. The value alone is meaningless — "+$224" and "−$937" are the same event measured on two windows, and the difference between them is the entire lesson. The unit matters because a per-store-week figure multiplied by 11 stores and 52 weeks ($224 × 11 × 52 = $128,128) is a plausible-sounding annual number that would be wrong twice: the event did not run 52 weeks, and the per-store-week figure is not an annual rate.

### 11.4 What would change the conclusion

Four observations would reverse the recommendation, and each is specified in advance so that the next test can distinguish them rather than relitigate the event.

1. **A measured displacement rate below 25.93%.** The sign-flip threshold from F-1 is 124.4 displaced units against 480 incremental units. If a loyalty extract shows households forward-buying fewer than 2.5 units per event purchase, the carryover term falls below the threshold and the full window turns positive. This is the most consequential unknown in the case and the cheapest to measure: one loyalty file, one repeat-rate calculation.
2. **A substitution effect smaller than assumed.** If the premium 800g pack's 150-unit decline traces to its own out-of-stock rather than to the promotion, the 60 transfer is spurious, the net event-window figure rises from 24 to 84, and the full window improves by the same 60 to −77. The event still fails, but the ranking against the 10% depth test changes.
3. **A realised depth shallower than 15.79%.** If the pre-event markdown audit understated how many store-weeks were already discounted, m_pre falls and the hurdle falls with it. At m_pre = 1.50 the hurdle is (1.50 − 1.00)/1.00 = 50.00%, still above the observed 42.86%, so this observation alone does not flip the sign — worth knowing before anyone spends a week on it.
4. **A supplier funding share above 32.50%.** The item window clears once the funding share passes 32.50% (D-8), which the actual 50% share already exceeds. So this observation is already satisfied and is not, on its own, a reason to reverse anything; it explains why the item figure is positive while the full-window figure is not.



Substituting back: m_p = 1.00 + 0.26 = 1.26, so profit = 480×1.26 − 1,120×(1.80 − 1.26) = 604.80 − 604.80 = 0 ✓.

The general lesson from this block is that a funding negotiation and a demand question live in the same arithmetic and are easy to confuse. Raising the supplier share moves the item across its threshold at 32.50%; it cannot move the full-window figure across its threshold, because the carryover term is priced at the regular margin m_u and no discount split touches it. Only a change in the *displacement rate* or the *depth* changes the full-window sign. Knowing which lever moves which number prevents the most common wasted quarter in promotion management: renegotiating funding to fix a problem that funding cannot reach.

### 11.5 What the result implies for the calendar, not just the item

If the sign-flip displacement rate is around 26% for a shelf-stable item on a 20% discount, the calendar's shelf-stable events are structurally exposed and its perishable events are not — households cannot forward-buy milk. The calendar-level implication is to shift depth toward perishables and frequency toward shelf-stable items at shallow depth. That is a portfolio claim from one item, so it stays a hypothesis for the next cycle rather than a decision now, but it is the hypothesis worth testing because it is the one that would change the shape of the promotion plan rather than the depth of one SKU.

Takeaway: report the projected number with its basis and unit, and specify in advance the four observations that would reverse it.

## 12. Failure Modes & What Went Wrong

Six distinct mechanisms, each with a symptom, at least two competing causes, and the check that discriminates between them. None of these is "monitor the model": each is a structural failure in the data construction or the arithmetic that produces a confidently wrong number.

### 12.1 F-01 — Forward buying masquerading as consumer response

**Mechanism.** An off-invoice deal lowers the retailer's cost for units ordered during the deal window. The retailer increases orders beyond likely sell-through, sells the surplus at regular retail afterwards, and books the difference as margin. Sell-out data shows a bump during the event; sell-in data shows a much larger bump. Neither series alone reveals the arbitrage.

**Symptom.** Units purchased exceed units sold by 15–40% within the event plus two weeks, and the excess clears at full margin in weeks 3–6.

**Competing causes.** (a) genuine consumer stockpiling, which also produces a post-event dip; (b) a distribution gain in which new stores began carrying the item, which produces a permanent level shift, not a temporary bulge; (c) a data problem where the sell-in feed includes warehouse transfers.

**Discriminating check.** Compute the ratio of sell-in to sell-out over the 8 weeks centred on the event, and separately compute the household repeat rate from loyalty data. Forward buying by the *retailer* leaves household purchase frequency flat while inventory rises; stockpiling by *households* raises household purchase frequency during the event and lowers it after. The two are distinguishable with one loyalty extract and are routinely conflated.

**Correction.** Evaluate on sell-out. Report sell-in separately as a working-capital effect. Never let a fixed allowance enter the incremental-profit numerator for an event whose purpose was consumer demand.

### 12.2 F-02 — Promoted price below cost with no recovery volume

**Mechanism.** A loss-leader depth (p_p < c) is set to drive traffic, and the evaluation applies a break-even formula that assumes m_p > 0.

**Symptom.** The break-even formula returns a negative unit requirement, or an implausibly large one, and the analyst reports the absolute value.

**Competing causes.** (a) the unit cost is wrong because the case-pack conversion was applied twice (once to the cost, once to the price), producing an artificially low cost; (b) the cost is a landed cost including freight while the price excludes a retailer-funded rebate, so the true margin is positive; (c) the item is genuinely below cost and the correct answer is "unprofitable at any volume".

**Discriminating check.** Compute the per-unit margin as a dollar amount from the invoice and the shelf price for three separate weeks, and confirm it against the margin table used by finance. If the three agree and are negative, the item is below cost and no volume recovers it. If they disagree, the cost basis is broken and the formula's output is meaningless in either direction.

**Correction.** Add the constraint m_p > 0 as a hard gate in the code, returning `UNPROFITABLE_AT_ANY_VOLUME` rather than a number. For genuine loss leaders, evaluate on the basket rather than the item, and state that the basket attribution is associative, not causal.

### 12.3 F-03 — Comparison-group contamination through the price zone

**Mechanism.** The promotion is set at price-zone level or is featured in a regional circular that reaches comparison stores. The "untreated" group is treated, k is biased toward the treated effect, and the uplift collapses toward zero.

**Symptom.** The estimated effect is small and the comparison group's event-week sales rise almost as much as the treated group's (k ≈ 1.35 when the true market movement was 1.12).

**Competing causes.** (a) a genuine national demand shock (weather, a competitor's closure) that affects both groups; (b) spillover from shoppers driving between treated and comparison stores, which affects comparison stores *near* treated ones but not distant ones; (c) the comparison stores were promoted.

**Discriminating check.** Split the comparison group by distance to the nearest treated store. Spillover produces a gradient (near stores show the effect, distant stores do not); a national shock affects both equally; promoted comparison stores show an effect in their own price file, which settles the question definitively and is checkable in one query.

**Correction.** Reclassify contaminated stores as treated and search for donors in a different price zone. If the estate is fully covered by the zone, abandon the difference-in-differences and state that the event is not causally evaluable — which is a legitimate finding that saves the organisation from a false number.

### 12.4 F-04 — Substitution set chosen after seeing the data

**Mechanism.** The analyst looks for category SKUs that declined during the event and calls the largest declines cannibalisation. Because 11 SKUs are candidates and 4 declined, there are 330 possible 2-SKU sets and a wide range of defensible-looking answers.

**Symptom.** The reported substitution adjustment varies by a factor of three across analysts evaluating the same event, and the variation correlates with who is presenting.

**Competing causes.** (a) genuine heterogeneous substitution (the promoted pack really does cannibalise the premium pack most); (b) random week-to-week variation in low-volume SKUs, which produces spurious declines; (c) an out-of-stock on a substitute, which manufactures a decline that is a supply failure rather than a demand transfer.

**Discriminating check.** Pre-register the substitute set on structural grounds before the event: same brand family, same size tier, adjacent price tier, and within the top 8 SKUs by category volume. Then check every substitute's in-stock rate during the event; any substitute below 90% in-stock is excluded as uninformative. Random variation can be assessed by running the same substitution estimate on a non-promoted week and confirming it is near zero.

**Correction.** Freeze the substitute set in the decision contract. Report the estimate under the frozen set only, plus a one-line sensitivity for the full set of declining SKUs so the reader can see the range — but never choose the sensitivity after the fact.

### 12.5 F-05 — Carryover window extended until the sign is acceptable

**Mechanism.** The event window shows a loss; the analyst extends the post-period to four, then six weeks, hoping the dip reverses. Sometimes it does, because the post-period picks up an unrelated seasonal rise, and the reported full-window figure is then positive.

**Symptom.** The reported window length differs across events in the same league table, and longer windows correlate with events the organisation wanted to keep.

**Competing causes.** (a) a genuine slow recovery because the item's consumption cycle is long; (b) a seasonal upturn in the post-period that the comparison group would have captured if the window had been symmetric; (c) a second promotion on the same item inside the post-window, which makes the "carryover" a mixture of two effects.

**Discriminating check.** Compute the post-period effect for the comparison group over the identical weeks and confirm it is flat; if the comparison group also rises, the post-window is seasonal, not recovery. Then audit the calendar for any second event on the item or its substitutes inside the window. A window that fails the symmetry check is not a longer window, it is a different experiment.

**Correction.** Fix the window in the contract, at a length motivated by the item's consumption cycle (for a 500g shelf-stable pack, two weeks is reasonable; for a 12-pack of toilet tissue, four). Report the fixed-window figure as the decision figure and any longer-window figure as descriptive only, clearly separated.

### 12.6 F-06 — Realised depth misread from the circular

**Mechanism.** Depth is computed as 1 − promo_price/list_price where list_price is the chain file price. Stores already on local markdown show a depth that never existed, and the derived hurdle and discount cash are both overstated by the pre-existing markdown.

**Symptom.** Depth varies across stores for an event that had a single national advertised price, and the variation is exactly the pre-period markdown rate.

**Competing causes.** (a) legitimate zone pricing, where the regular price genuinely differs by zone; (b) local competitive markdowns, which are temporary and should be measured against the pre-event four weeks rather than the file price; (c) a price file error where the regular price was not refreshed after a chain-wide change.

**Discriminating check.** Plot `promo_price / pre_period_modal_price` by store. Zone pricing produces clusters at round ratios (0.80, 0.75); markdown contamination produces a continuous spread below 1.0; a stale file produces a step at the date of the chain change. The modal price in the four weeks before the event is the correct denominator in all three cases.

**Correction.** Compute depth from the pre-period modal price per store, not from the file. Report the distribution of realised depth alongside the advertised depth, because a promotion whose realised depth has a 6-point spread is a portfolio of different promotions bundled under one SKU.

### 12.7 The meta-failure the six share

Every mechanism above produces a number that is internally consistent, arithmetically correct, and wrong by a margin large enough to reverse a decision. None of them throws an error, and none is caught by increasing model sophistication. They are caught by four cheap checks, in this order: the pre-period placebo, the comparison-group flatness test, the realised-depth distribution, and the substitute in-stock rate. The order matters because the first check invalidates everything downstream and takes the least effort.

Takeaway: the failures that matter are silent, structural, and shared across the organisation rather than located in the estimator.
