## 13. Transferable Lesson

### 13.1 The lesson in one sentence

In any evaluation where a treatment changes the price of a thing, the decision-relevant number is the *incremental profit at the funding grain over a window fixed in advance*, and that number is almost never the number the business is already reporting.

### 13.2 The five transfers, stated so they can be applied elsewhere

**Transfer 1 — Convert to money before comparing to a threshold, but after adjusting for transfers.** The order is the lesson. Converting units to money first is necessary because a 43% unit lift on a 45%-margin item is not comparable to a 30% lift on a 60%-margin item. Adjusting for substitution before converting is necessary because transfer volume carries margin at a different rate than the promoted volume does. Getting the order wrong produces numbers that are arithmetically consistent and economically wrong, and there is no diagnostic that flags them.

Transfers outside retail: a discount on a subscription tier cannibalises the higher tier; a free trial converts some users who would have paid anyway; a paid search campaign captures brand queries that would have converted organically. In each case the "campaign-level" number is positive and the "portfolio-level" number is not, and the gap is exactly the transfer priced at its displaced margin.

**Transfer 2 — The break-even multiplier is (margin before − margin after) / margin after.** This single ratio decides more promotions than any model. For a price cut from p_r to p_p with constant cost, it equals d/(1 − d − m_rate) when expressed in rates, which grows faster than the discount: at a 45% margin rate, a 10% cut needs 12.5% more volume, a 20% cut needs 28.6%, a 30% cut needs 46.2%, and a 40% cut needs 80.0%. The convexity is the argument for shallow-and-frequent over deep-and-rare, and it holds for every product with a positive margin and a fixed cost.

Check the convexity arithmetic: at d = 0.10, m_p = 4.00×0.90 − 2.20 = 1.40, requirement = 0.40/1.40 = 28.57% — not 12.5%. The 12.5% figure belongs to the *unfunded, no-cost* case with m_rate = 1. So the table must be computed with the cost in it, and mixing rate-space and dollar-space formulas is a common and silent error. The correct dollar-space series for this item: d = 0.10 → 28.57%; d = 0.20 → 80.00%; d = 0.25 → 160.0%; d = 0.30 → 320.0%. In dollar space the hurdle explodes because the cost is a large share of a small price.

That correction is itself the transferable lesson: whenever a rule of thumb is quoted in margin-rate space, re-derive it in dollar space before using it on a low-margin, low-price item, because the two spaces diverge exactly where the cost is a large fraction of price.

**Transfer 3 — Timing displacement is a first-class term, and its threshold can be as low as a quarter.** The F-1 identity shows that when the promoted margin is smaller than the displaced margin, a displacement rate below 100% flips the full-window sign, and for a 20% cut with half funding the threshold was 25.93%. The general form: the flip threshold is Δe/m_u divided by dQ, and it falls as funding improves the promoted margin relative to the regular margin. Any promotion of a storable good, any subscription discount with annual prepay, any "buy one get one" on a durable, has a threshold in this range.

**Transfer 4 — Precision is an estate property, not an analysis property.** The MDE is set by the number of independent units (stores, regions, accounts) and the between-unit variance. No amount of transaction-level data or model flexibility reduces it. Teams routinely commission a better model when the correct intervention is more treated units, and the failure is invisible because the model does deliver a smaller standard error on the wrong variance.

**Transfer 5 — Freeze the contract before the data, and keep the record even when the answer is "defer".** The six-option decision table in §3 and the immutable threshold in §8 are the durable artefacts. They survive staff turnover, they make disagreements about policy rather than arithmetic, and they turn each event into a data point for the next. Without them, the organisation relearns the same lesson annually and calls it institutional knowledge.

### 13.3 Where the lesson does *not* transfer

- **To brand-building spend with no measurable short-term sales effect.** The window logic inverts: the effect arrives later than any fixed window, so a fixed-window conclusion is systematically negative. Different design needed, not this one.
- **To clearance and markdown of perishable or end-of-life inventory.** The counterfactual is write-off, not full-price sale, so the displaced margin is negative and the arithmetic flips: selling at a loss beats not selling.
- **To market-level media where no untreated geography exists.** The difference-in-differences has no comparison group, and the honest answer is a modelled counterfactual with a stated identification strategy, not this design.
- **To new-item launches.** The counterfactual is "the item did not exist", so the incremental question is undefined and the relevant question becomes cannibalisation of the existing range plus long-run repeat.

### 13.4 The diagnostic habit

If one habit can be carried out of this case, it is: before reporting any uplift number, state the denominator, the window, and the margin at which displaced volume is priced. Three sentences. In the constructed example those three sentences are "1,120 counterfactual units per store-week", "two event weeks plus two carryover weeks", and "displaced volume at $1.80 regular margin". Change any one of them and the decision can change; state all three and the reader can disagree with the input rather than with the analyst.

Takeaway: the transferable asset is the ordering of operations and the frozen contract, not the estimator.

## 14. Connections & Cross-References

### 14.1 Within this unit

| File | Relationship | What to take from it |
|---|---|---|
| [01_Markdown_Optimisation.md](./01_Markdown_Optimisation.md) | sibling case, permanent price reduction | markdown depth is permanent, so the counterfactual cannot revert to the regular price; the hurdle arithmetic (D-2) is identical but the window is not |
| [03_Retail_Media_Incrementality.md](./03_Retail_Media_Incrementality.md) | sibling case, media spend | the same counterfactual problem with an attribution-based comparator instead of matched stores; read it for the ghost-ads and holdout discussion |
| [04_Personalised_Offer_Controls.md](./04_Personalised_Offer_Controls.md) | sibling case, targeted offers | selection is the treatment there; the break-even hurdle here becomes a targeting-precision problem |
| [05_Price_Change_Incident.md](./05_Price_Change_Incident.md) | sibling case, incident | shows what happens when a price error runs live; the realised-depth audit from §12.6 is how such an error is detected downstream |
| [../Cheat_Sheets/02_Promotion_Experiment_Guide.md](../Cheat_Sheets/02_Promotion_Experiment_Guide.md) | theory sheet | the five-part formula anatomy used in §6 comes from there, as does the eligibility-gate framing in §5a |
| [../Cheat_Sheets/03_Retail_Media_Measurement.md](../Cheat_Sheets/03_Retail_Media_Measurement.md) | theory sheet | comparator construction when randomisation is impossible; the matching ledger in §5e mirrors its approach |
| [../Cheat_Sheets/01_Retail_Price_Guardrails.md](../Cheat_Sheets/01_Retail_Price_Guardrails.md) | theory sheet | the margin-floor constraint C-05 and the depth-approval workflow that makes C-06 necessary |

### 14.2 To adjacent units

- **`01_store_operations_and_merchandising`** supplies the display and shelf-availability facts that make A-06 (availability, not demand) testable; a promotion measured without an out-of-stock column is a promotion measured on an unknown quantity of product.
- **`02_omnichannel_customer_loyalty_and_fulfilment`** owns the loyalty identity layer that converts the 38% identified transactions into a displacement measurement rather than a displacement assumption. The single most valuable input to §11.4 is held in that unit.
- **`../../02_retail_media`** and the retail-media subtree carry the media-side version of the same measurement problem, where the comparator is a geographic holdout rather than a matched store.

### 14.3 To method topics outside retail

- **Difference-in-differences and synthetic control.** The estimator in §5c is the standard panel DiD; the synthetic-control fallback is what to use when no single comparison group satisfies parallel trends. The relevant literature property is that synthetic control's inference is placebo-based, which is why V-1 and V-4 are structured as placebos rather than as t-tests.
- **Clustered inference.** The 3× understatement from transaction-level standard errors (§5e) is the classic Moulton problem: intra-cluster correlation inflates the effective sample. The fix is to cluster at the level of treatment assignment, which here is the store.
- **Pre-registration and specification search.** The eight plausible substitute sets in §12.4 are a specification-search problem with a known solution: fix the set ex ante and report the range descriptively.
- **Power analysis.** The MDE closed form in D-5 is the two-sample normal approximation; the exact answer for small arms uses the t distribution, which for 40 stores per arm changes the multiplier from 2.802 to about 2.83 and the MDE by 1%, immaterial here but material at n = 8 per arm.

### 14.4 To the decision environment

The case sits inside a planning cycle with a three-week lead time, a central pricing authority, and four functions with different scorecards (§2.6). Those facts are not background colour: they eliminate randomisation of depth (C-06), bound the analysis capacity (C-04), and determine that the deliverable must be a one-page record rather than a report. A reader who transplants the method into an environment with store-level pricing authority should randomise depth instead of exploiting zone variation, because that design identifies the depth-response curve directly and removes the reliance on A-01.

### 14.5 Common confusions this case is designed to break

- **"Lift" versus "incremental"**: lift is relative to a counterfactual; incremental is relative to the margin the counterfactual would have earned. A 43% lift can be negative incremental profit.
- **"ROI" versus "incremental profit"**: an ROI dividing incremental profit by discount spend is a ratio whose denominator goes to zero for shallow events, so it explodes and cannot rank events across depths. Report profit per store-week for ranking and ROI only as context.
- **"Significant" versus "worth it"**: V-6. An interval that excludes zero and contains the threshold is a defer.
- **"Promotion" versus "price cut"**: a display or feature changes salience, not price; measuring it with a price elasticity is a category error with a plausible-looking output.
- **"Sell-in" versus "sell-out"**: F-01. Both are called "sales" in the source systems, which is why the error is so common.
