# Market Basket Analysis and Association Rules

> Module 17 — Assortment, Category, and Retail Analytics  
> Topic 04 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Module 04; Topic 01  
> Version 2026.5

## 1. Why this topic exists

A category manager needs to choose which products to stock, where to place them, and which offers to test.
The receipt is a useful observation because it shows products purchased together on an occasion.
It is not a direct record of the shopper's needs, all products considered, or the effect of a shelf change.
Association analysis turns receipts into candidate relationships that deserve commercial and operational scrutiny.

Suppose a retailer notices that shoppers who buy chilled pasta often buy a sauce on the same visit.
The manager may want to keep sauce available nearby, develop a meal solution, or test a joint offer.
Each action consumes space, labor, inventory, or promotional funding.
A strong observed association makes the action worth investigating; it does not establish its incremental value.

The expensive error is to treat co-purchase as evidence that one product causes another to sell.
Pasta and sauce may be on promotion together, share a meal occasion, or be purchased by customers who already planned both.
Conversely, a weak same-receipt relationship does not prove products are unrelated: shoppers may buy them on separate trips.
The decision is therefore not “find the highest lift and implement it.”
It is “identify stable, material, feasible candidates and test whether the proposed intervention improves the business outcome.”

**Who uses the result?**

- Category managers use it to identify complementary missions and possible assortment dependencies.
- Space planners assess whether proximity or cross-merchandising is physically feasible.
- Inventory teams check whether a proposal creates stockout or replenishment risk.
- Commercial teams calculate margin and funding rather than interpreting sales revenue alone.
- Analysts define the denominator, check stability, and design the measurement.
- Privacy and legal teams assess whether customer-level linkage or activation is permissible.
- Store operations decide whether the proposed execution is practical and auditable.

This topic focuses on transaction-level association rules.
The next topic develops attach rates and cross-selling decisions; this one explains what the underlying co-occurrence does and does not show.
Examples are synthetic unless explicitly attributed.
They illustrate a method, not an observed retailer result.

## 2. Learning outcomes

After this topic you can:

1. Define a valid basket population and convert receipt lines into deduplicated item-presence records.
2. Compute support, confidence, baseline prevalence, lift, and counts from the same denominator.
3. Explain the directional nature of confidence and the symmetric nature of two-item lift.
4. Screen a candidate using volume, margin, availability, and business constraints, not lift alone.
5. Diagnose confounding, selection, sparse data, multiple testing, and temporal instability.
6. Design a randomized test that measures incremental category profit and guards against adverse effects.
7. Explain privacy controls, data minimization, and why pseudonymized customer data can remain personal data.
8. Communicate a recommendation with assumptions, uncertainty, ownership, and a stopping rule.

Exercises 1–8 map in order to these outcomes.
Use the self-check answers to inspect the arithmetic and the design rubrics to assess judgment.

## 3. The core mental model

A basket is a defined purchase occasion, not necessarily a household, shopping mission, or individual customer.
You observe whether items were purchased together under the store's existing prices, stock, assortment, and layout.
You then compare co-purchase to an appropriate baseline before deciding whether the pattern merits a test.

```text
Decision and eligible shopping occasions
                 |
                 v
Receipt lines -> clean, deduplicate, define baskets and product concepts
                 |
                 v
Counts: N, n(A), n(B), n(A and B) -> support, confidence, lift
                 |
                 v
Materiality + availability + stability + statistical checks
                 |
                 v
Candidate action -> feasibility and privacy review -> randomized test
                 |
                 v
Incremental profit, shopper and operational guardrails -> decision
```

Read the arrows as dependencies, not proof of causation.
A receipt dataset reflects past store choices and shopper choices simultaneously.
The store may have placed the products near one another already.
It may have offered a discount to the same shoppers, or the products may share an occasion.
Any of these can produce an association without the proposed change producing additional profit.

### Three questions, three denominators

For rule A → B, **support** asks what fraction of all eligible baskets contain both A and B.
**Confidence** asks what fraction of A baskets also contain B.
**Lift** asks how that confidence compares with the fraction of all baskets containing B.
A confidence of 60% can be unremarkable if 60% of all baskets contain B.
A confidence of 10% can be notable if only 2% of all baskets contain B, but it may still affect few baskets.

Lift compares observed co-occurrence to the co-occurrence expected under marginal independence in the selected population.
It is a descriptive ratio, not a treatment effect, a profit ratio, or a causal probability.
The two-item lift of A → B equals the lift of B → A.
The confidences generally differ, because the conditioning populations differ.

### From pattern to action

A useful candidate needs a plausible decision, enough eligible occasions, acceptable margin, and operational feasibility.
The category role matters: a destination category might justify a traffic-oriented test, whereas a convenience category may justify a targeted placement.
Estimate incremental effects against a counterfactual, rather than crediting all existing co-purchases to the proposed action.
Predeclare whether the decision is to expand, revise, or stop the test.

A **forecast** is a prediction of likely future outcomes under stated conditions.
A **plan** specifies the intended action and resources.
A **target** is the desired performance threshold.
A **commitment** is an authorized obligation, such as a confirmed supplier order.
The association itself is none of these.

## 4. Definitions and vocabulary

All probabilities below use the same eligible-basket population and the same observation window unless noted.
A and B are product concepts, which may each contain multiple stock-keeping units (SKUs).

| Term | Definition and unit | Common misuse |
|---|---|---|
| Basket | One defined eligible transaction or purchase occasion; counted in baskets | Assuming one receipt equals one person or mission |
| Basket key | Identifier that uniquely selects a basket within the analysis scope | Using a receipt number that repeats across stores or days |
| Item | A SKU, brand, or category concept represented as present or absent | Mixing aggregation levels without a rule |
| Itemset | A set of distinct items present in the same basket | Treating repeated units of one item as distinct items |
| A → B | Descriptive rule with antecedent A and consequent B | Reading the arrow as a causal or chronological arrow |
| Antecedent | The condition A defining the confidence denominator | Interpreting it as an intervention |
| Consequent | The item or itemset B evaluated within A baskets | Assuming it was purchased after A |
| N | Number of eligible baskets, in baskets | Using all receipts for N but filtered receipts for item counts |
| n(A) | Number of eligible baskets containing A, in baskets | Counting scanned units instead of baskets |
| n(B) | Number of eligible baskets containing B, in baskets | Counting the same basket more than once |
| n(A,B) | Number of eligible baskets containing both, in baskets | Summing pairwise SKU counts that overlap |
| Prevalence | n(B)/N, fraction of all baskets containing B | Using category share of revenue as a basket baseline |
| Support | n(A,B)/N, fraction of eligible baskets containing both | Calling conditional confidence support |
| Confidence | n(A,B)/n(A), fraction of A baskets also containing B | Calling it the probability that an intervention sells B |
| Lift | Confidence divided by n(B)/N, dimensionless | Treating a high value as proof of incremental sales |
| Leverage | n(A,B)/N − [n(A)/N][n(B)/N], in basket-share points | Confusing an absolute difference with a ratio |
| Independence | Joint probability equals product of marginal probabilities | Assuming plausible complementarity implies independence |
| Minimum support | Screening threshold for joint frequency, in baskets or fraction | Treating a chosen threshold as a significance guarantee |
| Multiple testing | Examining many candidate rules creates a greater risk of false discoveries | Reporting only the most impressive lift without the search scope |
| Holdout | Data reserved from candidate selection for evaluation | Reusing it repeatedly until a desired rule appears |
| Experiment | A design assigning an intervention under a specified allocation mechanism | Calling a before-and-after comparison randomized |
| Incremental profit | Profit difference attributable to assignment, relative to a counterfactual, in currency | Counting baseline purchases as incremental |
| Cannibalization | Sales transferred from another product or category rather than added | Counting substitute losses as new demand |
| Availability | Product was purchasable in the relevant store and period | Treating an out-of-stock nonpurchase as evidence of no interest |
| Pseudonymization | Replacing direct identifiers while additional information can reconnect records | Calling all pseudonymized data anonymous |

The information model is binary at the selected level: a basket contains an item or it does not.
Quantity and spend can be retained separately for economics, but they do not change binary support.
A category definition must be versioned because a SKU can be reclassified over time.
If two concepts overlap, interpret the relationship carefully: overlapping definitions can force co-occurrence.
For standard A → B calculations, use disjoint A and B and require n(A) > 0 and n(B) > 0.
If A never occurs, confidence is undefined; if B never occurs, lift is undefined.

## 5. The mechanics, step by step

### Step 1 — Translate the business request into a decision

**Input:** a category decision, planning horizon, and decision owner.
**Action:** write down the action that could follow from a rule.
**Output:** a decision brief that can be approved or rejected without changing its purpose later.

For example, ask whether a chilled pasta endcap should include one sauce line during a four-week test.
Name the affected stores, available shelf space, eligible products, execution capacity, and expected review date.
Specify whether the proposed intervention is a placement, a price change, an offer, or a bundle.
Each changes a different mechanism and needs a different counterfactual.

A report of “interesting rules” is not yet a decision brief.
If no action is feasible, archive the exploration as descriptive learning rather than promising business value.
If the intervention requires customer targeting, involve privacy and legal reviewers before creating linked profiles.

### Step 2 — Define the population and purchase occasion

**Input:** transaction scope and decision brief.
**Action:** define stores, channels, period, and the eligible basket key.
**Output:** a population contract with a reproducible denominator.

An example key is `(store_id, business_date, register_id, receipt_id)`.
The actual key must match the point-of-sale system; a receipt identifier alone may not be globally unique.
Define the store's business date consistently around midnight and daylight-saving changes.
For an online order, decide whether amended orders and split deliveries are one order or several.
Do not silently combine online and physical store baskets if their assortment and fulfillment differ materially.

Exclude voided transactions according to documented source semantics.
Decide how to handle returns, cancellations, negative quantities, employee transactions, and wholesale orders.
A return-only receipt may be excluded, while an exchange may need separate treatment.
Record counts before and after each filter; do not quietly drop unmatched product lines.
If a transaction contains only non-merchandise lines, define whether it remains in N.
The answer affects all support and baseline values.

Choose a window that reflects the decision horizon and captures relevant trading days.
A four-week period may be suitable for a short tactical test, but can miss seasonal variation.
A holiday period should not silently be generalized to ordinary weeks.
Write down whether the selected stores represent a region, a format, or the entire chain.

### Step 3 — Construct item presence

**Input:** filtered receipt lines and a dated product hierarchy.
**Action:** map each eligible line to the intended product concept, then deduplicate `(basket, item)`.
**Output:** a binary basket-item table and a reconciliation report.

If a receipt has two units of the same sauce, the sauce is present once for support calculations.
If two SKUs both belong to the sauce category, the basket still contributes only one sauce presence.
A bundle SKU needs explicit treatment: decide whether its components count as purchased items and whether their prices can be separated.
Do not both expand a bundle into components and count a pre-existing component line without deduplication.

Retain original line quantities and net prices in a separate economic table.
This preserves the ability to assess margin without corrupting binary basket counts.
Review hierarchy coverage by baskets and by sales value, especially for new or discontinued SKUs.
Unknown mapping is missing information, not proof that no item was purchased.

If an A category contains B by definition, the rule may be tautological.
Use mutually exclusive item concepts where the decision is about genuine cross-category affinity.
A composite antecedent, such as pasta AND cheese, should be explicitly constructed as the intersection of its components.
Its denominator is the number of baskets containing both antecedent items, not the sum of their counts.

### Step 4 — Compute counts before ratios

**Input:** deduplicated basket-item presence and population N.
**Action:** count N, n(A), n(B), and n(A,B).
**Output:** a 2×2 contingency table in baskets.

| Eligible baskets | B present | B absent | Total |
|---|---:|---:|---:|
| A present | n(A,B) | n(A) − n(A,B) | n(A) |
| A absent | n(B) − n(A,B) | N − n(A) − n(B) + n(A,B) | N − n(A) |
| Total | n(B) | N − n(B) | N |

Every cell must be nonnegative and the four cells must sum to N.
The pair count cannot exceed either marginal count.
These checks catch duplicate joins, inconsistent filters, and denominator mistakes.
They do not prove that the eligible population is representative or that product mappings are correct.

### Step 5 — Calculate support, confidence, lift, and leverage

**Input:** the contingency table.
**Action:** calculate the following measures without premature rounding.
**Output:** interpretable counts and ratios for each candidate.

```text
support(A,B) = n(A,B) / N
confidence(A → B) = n(A,B) / n(A)
baseline(B) = n(B) / N
lift(A → B) = confidence(A → B) / baseline(B)
leverage(A,B) = support(A,B) − [n(A)/N × n(B)/N]
```

At lift 1, observed joint frequency equals the independence benchmark in this population.
Above 1, co-purchase is more frequent than that benchmark; below 1, it is less frequent.
These statements say nothing about statistical precision or whether changing one item affects the other.
Lift is unchanged when A and B swap, because both directions equal `P(A,B) / [P(A)P(B)]`.
Confidence changes with direction because its denominator changes.

Leverage measures the absolute gap between observed joint share and the independence benchmark.
Multiplying leverage by N yields an excess or deficit of baskets relative to that descriptive benchmark.
Those baskets are not incremental sales attributable to an intervention.
A high lift and low leverage can describe a rare pair; an apparently modest lift can cover many baskets.

### Step 6 — Filter for materiality and feasibility

**Input:** metrics, pair counts, item economics, and operational constraints.
**Action:** apply predeclared support and count thresholds, then evaluate feasible actions.
**Output:** a manageable candidate list with reasons for inclusion and exclusion.

A minimum pair count protects against making a decision from one or two coincidences.
A support floor helps avoid a long list of rare pairs, but a universal floor can hide valuable niche missions.
Use separate, predeclared rules for high-volume and niche decisions when their economics differ.
List the thresholds and the total number of candidate rules considered.

Estimate the number of potentially affected occasions, not merely the observed number of pairs.
Use contribution margin after relevant variable costs, funding, and waste.
Check shelf capacity, replenishment frequency, substitution, and whether the items are simultaneously available.
A high-affinity pair with insufficient stock may make the shopper experience worse if promoted.
Do not treat supplier funding as incremental profit without recording its conditions and duration.

### Step 7 — Check comparability and exposure

**Input:** candidate rules, store-week characteristics, prices, promotion, and availability.
**Action:** compare segments or periods that could plausibly differ in their shopping mix.
**Output:** a stratified descriptive view and a list of unresolved selection mechanisms.

Compute the same metrics by store format, region, week, and relevant price or promotion state when sample sizes permit.
Report counts in each segment; avoid publishing sparse cells.
Ask whether an apparent chain-wide association is driven by a few large stores or one special offer.
Consider basket size: large baskets mechanically create more opportunities for pairs to co-occur.
Comparing only larger baskets can itself condition on an outcome of shopping behavior, so describe the selection and avoid causal claims.

Availability matters for the interpretation of absence.
If a sauce SKU was unavailable in a store, absence from a receipt cannot be interpreted as a rejection of that SKU.
Restricting analysis to in-stock store-days can improve comparability but changes the target population and may introduce selection.
Document the in-stock measurement, its granularity, and missingness; test sensitivity to alternative definitions.

### Step 8 — Evaluate statistical stability

**Input:** screened rules and a predeclared evaluation plan.
**Action:** assess uncertainty and out-of-sample reproducibility at the appropriate sampling unit.
**Output:** interval estimates or stability summaries with explicit dependence and selection caveats.

A simple binomial interval for confidence treats A baskets as independent, identically distributed observations.
That can be optimistic when baskets cluster within stores, customers, or promotion weeks.
Consider a store-level or store-week cluster bootstrap when the independent units and sample size justify it.
Keep all rows from a sampled cluster together; resampling individual receipt lines is wrong.
For lift, recompute both the conditional probability and the baseline in every bootstrap replicate.

A wide interval can make a top-ranked rule indistinguishable from many alternatives.
For many tested rules, report the search size and choose an appropriate multiplicity procedure if making formal inferential claims.
An untouched later time window or held-out stores can check transportability, but cannot remove all bias.
Choose candidates on discovery data and evaluate the fixed list once on holdout data.
Repeatedly using the same holdout for selection makes it part of discovery.

Nonstationarity remains possible even when a rule replicates.
A promotion, assortment reset, weather event, competitor opening, or price change can alter behavior.
Use a rolling monitoring plan with a minimum count and meaningful change threshold.
Do not interpret a statistically significant tiny association as automatically worthwhile.

### Step 9 — Form an action hypothesis

**Input:** stable candidates, constraints, and a business objective.
**Action:** articulate mechanism, intervention, outcome, and counterfactual separately.
**Output:** a testable hypothesis and an economic decision rule.

Example: placing one sauce next to chilled pasta may reduce search friction and increase combined category contribution per eligible store-week.
The intervention is placement, not the historical co-occurrence.
The primary outcome should cover affected products and reasonable substitutes, not just the promoted SKU.
Define the time bucket, currency, exclusions, margin method, and costs before inspecting test outcomes.
Guardrails may include out-of-stocks, waste, customer complaints, execution compliance, and neighboring-category contribution.

A useful profit bridge is incremental gross contribution less incremental placement, labor, funding, waste, and fulfillment costs.
Avoid adding the margin of all test-period pair purchases as if they would not have happened in control.
Report the expected range and the sensitivity of the decision to uncertain costs.
A positive point estimate without enough precision or execution compliance may justify a longer or redesigned test, not rollout.

### Diagnostic drill 1 — Receipt identifiers and the population

The source team reports that receipt number 42 appears at two different stores on the same day.
One receipt contains pasta and the other contains sauce.
If the analyst joins on receipt number alone, the merged record falsely contains both items.
The erroneous result adds one pair basket even though neither actual receipt contains the pair.
The appropriate key depends on the point-of-sale system and might also require register and business date.
Verify uniqueness against the source transaction header, not just the item-line table.

Now suppose 10,000 completed receipts exist in a week.
Only 1,000 contain at least one of the two focal product groups.
The remaining 9,000 still belong in the all-basket denominator if they meet the eligibility contract.
If 200 contain both, support is 200 / 10,000 = 2.0% of eligible baskets.
Using 200 / 1,000 instead gives 20.0% of baskets containing at least one focal item.
The latter is a different conditional quantity, not the conventional support for the eligible population.
Both numbers can be computed, but they answer different questions.

Check whether zero-line headers are genuine receipts, technical records, or canceled transactions.
Consult source owners before including them merely to enlarge N.
Keep a record of the decision and of the volume it changes.
If a data feed drops all empty-focal-item receipts, the intended support cannot be recovered from that feed alone.
Obtain the eligible header population or explicitly relabel the narrower analysis.

### Diagnostic drill 2 — Unit counts versus basket counts

A customer buys three jars of sauce and one chilled pasta package on one receipt.
For binary presence, that receipt contributes one A basket, one B basket, and one pair basket.
It does not contribute three pair baskets.
Other receipts can contain different numbers of units, but support measures the share of occasions.

Suppose 100 receipts contain pasta and 30 of those also contain sauce.
If the 30 pair receipts contain two sauce jars each, the number of sauce units is 60.
The confidence A → B is still 30 / 100 = 30% of pasta baskets.
The ratio 60 / 100 = 0.60 sauce units per pasta basket is a different quantity.
It can be useful in replenishment planning, but should not be labelled confidence.
The monetary value of those units is another measure again.
Keep these measures separately labelled in dashboards and decision briefs.

If product returns are attached to the original receipt, determine whether the data represent gross or net units.
An item bought and immediately voided may not represent an actual purchase.
If a later return is on a separate receipt, the original shopping occasion may remain valid for a co-purchase question.
For net-margin measurement, the subsequent return may nevertheless matter.
The correct rule depends on the estimand and available linkage, not a universal “drop negatives” command.

### Diagnostic drill 3 — Aggregation level

Imagine a sauce category consisting of red, white, and pesto products.
The category-level question asks whether a basket contains at least one sauce product.
A basket containing both red and pesto sauce contributes one category-presence observation.
Adding the red-basket count and pesto-basket count double-counts that basket.
Use the union of basket identifiers when mapping to the parent category.

At SKU level, rare items may have very high lift because their baseline is small.
At category level, the same relationship can be diluted by products with different uses.
Choose the level at which a manager can actually act.
A category adjacency decision may use category presence, while a specific cross-merchandised SKU requires SKU economics.
Show the relationship at both levels when aggregation could change the decision.
Do not infer a strong relationship for each SKU merely from a category-level rule.

Changes in packaging and product hierarchy also matter.
If an item changes identifiers midway through a period, a naive exact-SKU analysis can split one product's history.
If a SKU is reassigned from one category to another, using only the current mapping can rewrite history.
Store the effective dates and explain whether reporting uses an as-sold or restated hierarchy.
The two choices answer different historical questions.

### Diagnostic drill 4 — High confidence without special affinity

Suppose 10,000 baskets are eligible.
Bread B occurs in 8,000 baskets, so its baseline prevalence is 80%.
Pasta A occurs in 1,000 baskets, of which 800 also contain bread.
The confidence A → B is 800 / 1,000 = 80%.
Lift is 80% / 80% = 1.0.
The 80% confidence sounds high, but bread is equally common in the population overall.

The support of the pair is 800 / 10,000 = 8%.
That is material volume, although the relationship is not above the independence benchmark.
If bread availability is operationally important, the volume still informs stock planning.
It does not identify a special pairing that a display will improve.
Baseline prevalence prevents the analyst from confusing popularity with relative affinity.
The decision to test should also consider reach and contribution, not just whether lift exceeds one.

### Diagnostic drill 5 — High lift with limited reach

Suppose 100,000 baskets are eligible.
A appears in 100 baskets, B appears in 200 baskets, and the pair appears in 20 baskets.
The pair support is 20 / 100,000 = 0.02%.
The confidence A → B is 20 / 100 = 20%.
The B baseline is 200 / 100,000 = 0.2%.
Lift is 20% / 0.2% = 100.

Only 20 observed occasions support that impressive ratio.
The pair may matter for a specialist shop, but it is unlikely to justify a chain-wide display without further evidence.
The result is also sensitive to small count changes or mapping errors.
Check whether A and B are rare because of short selling periods, limited distribution, or true low demand.
Compare the right stores and dates before attributing the pattern to shopper affinity.
If few stores stock both items, the all-store denominator may misrepresent the relevant choice set.
Report that narrower estimand rather than quietly switching denominators.

### Diagnostic drill 6 — Direction and timing

Consider 2,000 eligible baskets, with 200 containing A, 500 containing B, and 100 containing both.
The confidence A → B is 100 / 200 = 50%.
The confidence B → A is 100 / 500 = 20%.
The A baseline is 200 / 2,000 = 10%, and the B baseline is 500 / 2,000 = 25%.
Both directions have lift 2.0.

These values describe different conditional groups.
They do not say whether A was selected before B, whether B was planned first, or whether either influenced the other.
Receipt scan order is also not necessarily decision order: shoppers choose items before reaching the register.
Clickstream order can reveal a digital sequence, but even a genuine sequence can have shared causes.
Sequential association and intervention effect are separate estimands.
If the action is an offer shown after an online selection, randomized offer assignment can directly evaluate the intervention.

### Diagnostic drill 7 — Pooling can reverse a pattern

Consider two store formats with different shopping missions.
In the first format, 90 of 100 A baskets contain B, while 720 of 900 non-A baskets contain B.
Within this format, B occurs in 90% of A baskets and 80% of non-A baskets.
In the second format, 20 of 200 A baskets contain B, while 10 of 200 non-A baskets contain B.
Within that format, the corresponding rates are 10% and 5%.

Pooling both formats gives 110 B baskets among 300 A baskets, or about 36.7%.
It also gives 730 B baskets among 1,100 non-A baskets, or about 66.4%.
The pooled comparison reverses the within-format comparison because A baskets are distributed differently across formats.
This is a numerical illustration of Simpson's paradox, not a causal identification strategy.
It shows why store mix can distort a chain-wide summary.

Inspect relevant format-specific counts, baselines, and co-occurrence.
Avoid automatically splitting into dozens of sparse groups: decide which stratifications matter to the business and data-generating process.
When estimating a causal effect, selecting variables to adjust for requires a causal argument.
Conditioning on a mediator or collider can introduce bias rather than remove it.
A descriptive breakdown alone cannot resolve every confounding concern.

### Diagnostic drill 8 — Missing stock and opportunity to buy

Suppose 100 pasta baskets occur on days when sauce is in stock and 40 of them contain sauce.
Another 100 pasta baskets occur when sauce is unavailable and none contain sauce.
The pooled confidence is 40 / 200 = 20%.
The confidence on observed in-stock occasions is 40 / 100 = 40%.
Neither figure alone estimates the effect of guaranteeing sauce availability.

The in-stock subset is selected on a store-day attribute that may also reflect demand and replenishment.
Busy stores, promotions, delivery delays, and holidays can affect both stock and purchasing.
Availability may be measured at store-day level even though shelves were empty for only part of a day.
The apparent nonpurchase rate may therefore mix actual choice with lack of opportunity.
Report the data grain and uncertainty in stock measurements.
For a placement test, establish replenishment and monitor stockouts in both arms.

If the relevant question is the current shopper experience, include actual stockout exposure in the outcome.
If the question is affinity when both products are purchasable, define the eligible exposure accordingly.
Those populations differ and need not produce the same answer.
Do not claim that the in-stock association is an unbiased forecast of a guaranteed-stock policy.

### Diagnostic drill 9 — Promotion and common causes

Suppose a retailer discounts pasta and sauce in the same week.
More shoppers may buy both because the prices are lower, because an advertisement promotes a meal, or because the event coincides with a holiday.
Existing adjacency may also make the products easier to find.
The transaction table records the pair but does not separately identify those mechanisms.
Comparing promoted weeks with nonpromoted weeks can be informative, but time and shopper mix also change.

Record promotion type, timing, location, and discount depth for each product.
Break out ordinary and promoted periods when the counts permit.
Label these as descriptive strata rather than randomized treatment groups.
If promotions are targeted to selected stores or customers, the assignment process itself can create differences.
If multiple changes happen together, an experiment estimates the effect of that package unless it separates the components.
Do not claim a placement effect from a placement-plus-discount package without additional design.

### Diagnostic drill 10 — Candidate search and multiple testing

Suppose an analyst examines 1,000 product pairs.
Even when no genuine relationship exists, random fluctuations can make some observed measures look unusual.
The exact number of false discoveries depends on dependence, sampling, and the testing procedure.
It is not valid to treat the most extreme selected pair as though it were the only pair ever examined.

First define the candidate universe and restrictions, such as eligible categories and minimum counts.
Record whether the analysis includes both directions, multi-item antecedents, and many time windows.
Then reserve a later period or different stores for an evaluation of the fixed candidate list.
A holdout supplies a useful reproducibility check, but can differ systematically from the deployment population.
Formal false-discovery-rate procedures require valid input tests and applicable dependence assumptions.
Statistical adjustment cannot repair bad basket definitions or unmeasured causal confounding.

Report both the number of rules searched and the number advancing to review.
Explain whether thresholds were selected before or after seeing outcomes.
If exploratory thresholds change, label the results exploratory and obtain a fresh evaluation set.
Documenting failed candidates reduces the temptation to present discovery as a prespecified confirmation.

### Diagnostic drill 11 — Temporal validation

Suppose the discovery window contains a four-week summer holiday.
The candidate pair may be associated with outdoor meals rather than the ordinary store mission.
Recompute the same definitions in a later ordinary period without changing the candidate list.
Compare the pair count, both marginal prevalences, confidence, lift, and availability.

For example, discovery data might show 300 pair baskets out of 10,000 eligible baskets.
In the holdout, the same pair might appear in 90 of 10,000 baskets.
The support drops from 3.0% to 0.9%, but lift cannot be inferred from pair support alone.
The marginal counts for both periods are needed to determine whether relative affinity changed.
Even if lift is unchanged, the smaller volume may make the display uneconomic.

Time validation should match the deployment question.
A winter seasonal display needs relevant winter data or a transparent extrapolation.
A new item with no previous winter history requires additional uncertainty.
Changing the product definition between windows makes apparent drift difficult to interpret.
Preserve mapping versions and restate prior periods consistently when appropriate.

### Diagnostic drill 12 — Clustering and precision

Suppose a chain has 20 stores and 200,000 receipts.
If all receipts within a store share a price, layout, and inventory system, they are not 200,000 independent treatments.
A receipt-level standard error can understate uncertainty about a store-level rollout.
The effective information for an intervention may be much closer to the number and variation of independently assigned stores.

For descriptive uncertainty, a cluster bootstrap can sample stores with replacement and recompute the entire statistic.
Within each selected store, keep its receipts together and retain the correct duplicate-store multiplicity in the replicate.
Recalculate N, both marginal counts, the joint count, and lift each time.
Use the empirical replicate distribution for an interval only when the cluster sample and bootstrap assumptions are appropriate.
With very few stores, even cluster resampling can be unreliable.
Consult an appropriate small-sample method and report the limitation instead of presenting false precision.

For an experiment, follow the randomization design in the analysis.
If stores were paired before assignment, use an analysis consistent with that pairing.
If treatment was assigned by geographic region, a store-only analysis may not reflect the true independent assignment units.
Predeclare the method before unblinding results.

### Diagnostic drill 13 — Margin and substitution

A proposed bundle sells 100 extra promoted sauce units in a week.
Of those, assume 60 units replace purchases of another sauce and 40 are genuinely added at the category level.
Suppose each added unit contributes USD 1.50 before display cost.
Suppose the switch from the original sauce loses USD 0.20 contribution per switched unit.
The contribution change before display cost is 40 × USD 1.50 − 60 × USD 0.20 = USD 48 per week.
If display labor and materials cost USD 60 per week, the illustrative net change is −USD 12 per week.

Counting all 100 units as incremental would materially overstate value.
The assumed split between new and switched units is not identifiable from association rules alone.
Measure it with an appropriate comparison and cover relevant substitutes in the outcome.
If a discount also affects margin on units customers would have bought anyway, include that cost.
If funding expires after the test, evaluate the economics both with and without sustainable funding.
Revenue, gross margin, and contribution after variable costs are different outcomes.

### Diagnostic drill 14 — Action selection

Candidate A is a stable, high-volume pair with limited shelf capacity.
Candidate B is a moderate-volume pair with reliable supply and low display cost.
Candidate C is a rare but high-margin specialty pair concentrated in a few stores.
There is no universal ranking without the objective and constraints.

For A, the next step may be a space trade-off analysis rather than immediate display installation.
For B, a randomized store test may be feasible if the plausible contribution exceeds execution costs.
For C, a limited-format pilot may be appropriate if the target population and precision are adequate.
If the existing display already places the items together, another adjacency intervention may have little room to change behavior.
When the feasible action changes price instead of placement, the relevant causal question changes as well.

Document rejected candidates and reasons.
A rejection because of stock capacity is different from a rejection because the association failed to replicate.
The former may become feasible after supply changes; the latter may indicate instability.
Set a review date only when new evidence or a material constraint change could alter the decision.

### Diagnostic drill 15 — Analysis-ready reporting

A rule table should include a plain-language product definition, population, date window, and store coverage.
It should show N, n(A), n(B), n(A,B), support, confidence, baseline, lift, and leverage.
Add the candidate search scope and any predeclared filtering thresholds.
Show the corresponding counts for any segment or holdout, not percentages alone.

Separately identify the proposed intervention, primary outcome, forecast assumptions, and decision owner.
Mark historical metrics as observations, not forecasts or commitments.
Include availability and promotional context so that the result is not read outside its conditions.
Attach an uncertainty summary when the method and independent units support one.
Record known data gaps and the specific business decision they could change.

The final recommendation should say what will happen next and who authorizes it.
For example, “test a sauce placement in eligible stores after power and supply review” is a plan.
“The display will generate 410 extra baskets” misuses the independence-benchmark difference.
“Achieve positive net contribution with no material stockout deterioration” is a target.
“Order display materials for 20 stores” becomes a commitment only when authorized.

### Step 10 — Design the causal test

**Input:** proposed intervention, eligible stores, constraints, and primary metric.
**Action:** randomize at the unit where the intervention can be independently implemented.
**Output:** a documented experimental protocol and analysis plan.

For a shelf layout, stores are often the assignment units, not individual receipts.
Balance or stratify on pre-period store characteristics, then randomize stores to treatment and control.
Keep the rest of the promotion, price, and assortment as similar as practical.
If co-interventions are unavoidable, predefine them and interpret the effect as the combined package.
Allow for treatment spillovers, shoppers visiting multiple stores, regional marketing, and supply substitution.

Choose duration and number of stores using a prospective power or precision calculation based on historical store-level variability and the minimum economically important effect.
Do not use the number of receipts as if all receipts were independent when the treatment is assigned at store level.
Predefine exclusion rules, missing-data treatment, analysis population, and how to handle noncompliance.
An intention-to-treat comparison uses stores as assigned and preserves the randomized comparison under the design's assumptions.
Record actual execution separately to understand feasibility without claiming that compliant stores are an unbiased subgroup.

### Step 11 — Interpret, decide, and monitor

**Input:** trial outcomes, uncertainty, costs, guardrails, and compliance.
**Action:** compare the estimated incremental effect and uncertainty to the predeclared decision rule.
**Output:** an expand, revise, stop, or collect-more-evidence decision with an owner.

Report the assignment unit, number of stores, dates, primary outcome, estimated difference, uncertainty interval, and key deviations.
A result compatible with both a material gain and a material loss is inconclusive, not proof of no effect.
Monitor post-rollout availability, contribution, and execution; generalization to other formats remains an assumption.
Record the policy or data-access changes required to continue.
Archive the candidate generation process to avoid presenting exploratory findings as a prespecified test.

## 6. Worked example

### Business question and population

A regional grocery chain is considering whether to test a chilled pasta and pasta sauce meal display.
The synthetic dataset contains 18,240 eligible in-store baskets across eight comparable stores over four ordinary trading weeks.
The team uses one completed receipt as one basket and excludes void-only and return-only transactions.
Chilled pasta is A; shelf-stable pasta sauce is B.
Both concepts are mutually exclusive product groups, and presence is deduplicated within each basket.

The analyst counts 1,824 baskets containing A, 2,736 containing B, and 684 containing both.
Counts include all eligible baskets, not just baskets containing either product.
The retailer's decision concerns a new display, but the four-week data describe the current layout.
They are not a randomized evaluation of the new display.

### Reconstruct the table

| Eligible baskets | Sauce B present | Sauce B absent | Total |
|---|---:|---:|---:|
| Pasta A present | 684 | 1,140 | 1,824 |
| Pasta A absent | 2,052 | 14,364 | 16,416 |
| Total | 2,736 | 15,504 | 18,240 |

The pasta-only count is 1,824 − 684 = 1,140 baskets.
The sauce-only count is 2,736 − 684 = 2,052 baskets.
The neither count is 18,240 − 1,824 − 2,736 + 684 = 14,364 baskets.
These four counts sum to 18,240 baskets.

### Compute and interpret the metrics

The joint support is 684 / 18,240 = 0.0375, or 3.75% of eligible baskets.
The A prevalence is 1,824 / 18,240 = 0.10, or 10.0% of eligible baskets.
The B prevalence is 2,736 / 18,240 = 0.15, or 15.0% of eligible baskets.
The confidence for A → B is 684 / 1,824 = 0.375, or 37.5% of pasta baskets.
The lift is 0.375 / 0.15 = 2.5, a dimensionless ratio.

The reverse confidence is 684 / 2,736 = 0.25, or 25.0% of sauce baskets.
The reverse lift is 0.25 / 0.10 = 2.5.
A sauce basket is not the same conditioning population as a pasta basket.
The different confidence values do not imply a direction of influence.

Under the descriptive independence benchmark, expected joint support is 0.10 × 0.15 = 0.015, or 1.5% of baskets.
The benchmark corresponds to 18,240 × 0.015 = 273.6 baskets.
Observed leverage is 0.0375 − 0.015 = 0.0225, or 2.25 percentage points.
The observed-minus-benchmark difference is 684 − 273.6 = 410.4 baskets over the four-week population.
The fractional benchmark is an expectation, not a literal fractional transaction.
The 410.4-basket difference is not a forecast of extra purchases from a display.

### Check a comparison candidate

Suppose a second pair occurs in 24 of 18,240 baskets, with 80 baskets containing its antecedent and 120 containing its consequent.
Its support is 24 / 18,240 = 0.1316% of baskets.
Its confidence is 24 / 80 = 30.0% of antecedent baskets.
Its baseline is 120 / 18,240 = 0.6579% of baskets.
Its lift is 45.6, but the observed joint count is only 24 baskets.
The high ratio reflects a rare consequent and does not by itself establish meaningful scale or precision.
Both candidates need margin and feasibility checks; the larger pair may be more relevant to an eight-store display decision.

### Economics and proposed experiment

Assume, only for planning illustration, that the team expects the display to produce 100 additional combined-category purchase occasions across the test period.
Assume incremental contribution averages USD 2.40 per genuinely additional occasion after product costs and substitution.
Assume display materials, extra labor, and waste together cost USD 150 for the test period.
The planning estimate is 100 × USD 2.40 − USD 150 = USD 90 incremental contribution after those costs.
This is a hypothetical forecast, not an estimate derived from the 684 historical baskets.
If only 50 occasions are incremental, the same assumptions give USD 120 − USD 150 = −USD 30.
The break-even quantity is USD 150 / USD 2.40 = 62.5 incremental occasions, or at least 63 whole occasions under these simplified assumptions.

The plan is to randomize eligible stores to the display or the existing layout after a feasibility and sample-size review.
The target is positive net incremental contribution with no unacceptable out-of-stock or waste increase.
No purchase commitment is authorized by this example.
A proper trial must include more stores or a longer period if eight stores cannot provide useful precision.
Use store-level assignment and analysis; count all relevant substitute sales and costs.
The manager should not claim the display works until the randomized comparison is measured.

### Additional investigation: store-format mix

The following further counts are hypothetical and sum to the original four-week totals.
They illustrate what the category manager could ask the analyst to investigate before selecting stores.
The two groups represent different store formats, not experimental treatment and control.

| Four-week count | Urban format | Suburban format | All eight stores |
|---|---:|---:|---:|
| Eligible baskets | 9,120 | 9,120 | 18,240 |
| Pasta A baskets | 912 | 912 | 1,824 |
| Sauce B baskets | 1,368 | 1,368 | 2,736 |
| Both A and B | 456 | 228 | 684 |

In the urban format, pair support is 456 / 9,120 = 5.0%.
Confidence A → B is 456 / 912 = 50.0%.
The sauce baseline is 1,368 / 9,120 = 15.0%.
Lift is 0.50 / 0.15, approximately 3.33.
Urban leverage is 0.05 − (0.10 × 0.15) = 0.035, or 3.5 percentage points.

In the suburban format, pair support is 228 / 9,120 = 2.5%.
Confidence A → B is 228 / 912 = 25.0%.
The sauce baseline remains 15.0%.
Lift is 0.25 / 0.15, approximately 1.67.
Suburban leverage is 0.025 − 0.015 = 0.010, or 1.0 percentage point.

Both groups have lift above 1 in these illustrative data.
The larger urban lift may justify closer investigation of that format.
It does not prove that an urban display will have a larger treatment effect.
The formats may differ in meal occasions, existing layout, promotion, competition, or product availability.
The counts do not reveal those explanations.

The manager should first check whether the data include comparable trading periods and complete store feeds.
Next, inspect the product hierarchy and availability for each format.
Then compare pre-existing placement and promotions.
If the test aims to generalize to both formats, recruit adequate independent stores in each.
If only urban stores are included, the estimated effect applies most directly to that urban test population.
Do not claim a precise difference in effects between formats without designing and analyzing for that comparison.

### Additional investigation: benchmark versus intervention

Under independence, the whole-population expected pair count was 273.6 baskets.
The actual pair count was 684 baskets.
The difference of 410.4 baskets is a descriptive contrast between two frequency calculations.
It is not a count of shoppers persuaded by a display.
It is not the number of shoppers who would stop buying both if the products were separated.
It is not the number of additional baskets a new display would create.

The benchmark assumes the observed marginal rates but imposes independence.
A randomized display experiment instead compares outcomes under different assigned layouts.
The experiment's estimate includes effects mediated by behavior, availability, and execution under the tested policy.
If the display also changes price or adds advertising, its effect includes those components.
The answer depends on what was randomized and held fixed.

The distinction also affects the financial forecast.
Multiplying 410.4 by USD 2.40 would produce a number with no justified incremental interpretation.
The benchmark difference was not produced by a causal test and cannot supply the forecast's incremental occasion count.
The assumed 100 occasions in the planning calculation must be justified independently or used only as a sensitivity scenario.
If no credible prior estimate exists, report a range and identify the effect that makes the decision break even.

### Additional investigation: cost sensitivity

Retain the illustrative USD 2.40 contribution per additional occasion and USD 150 fixed test cost.
The table below is an arithmetic sensitivity analysis, not a probability distribution.
Each scenario assumes the cost and contribution remain unchanged as volume changes.

| Incremental occasions, test period | Contribution before fixed cost | Fixed test cost | Net contribution |
|---:|---:|---:|---:|
| 0 | USD 0 | USD 150 | −USD 150 |
| 25 | USD 60 | USD 150 | −USD 90 |
| 50 | USD 120 | USD 150 | −USD 30 |
| 63 | USD 151.20 | USD 150 | USD 1.20 |
| 100 | USD 240 | USD 150 | USD 90 |
| 150 | USD 360 | USD 150 | USD 210 |

If the incremental contribution is instead USD 1.50 per occasion, break-even rises to 100 occasions.
If the test cost rises to USD 240 and contribution remains USD 2.40, break-even is also 100 occasions.
If both change, break-even is USD 240 / USD 1.50 = 160 occasions.
This simple calculation omits uncertainty, taxes, fixed overhead allocation, capacity effects, and time value of money.
Its purpose is to show which assumptions matter before resources are committed.

### Additional investigation: interpreting an experiment

Assume a later randomized test assigns comparable stores to display and control.
Before seeing results, the team defines net contribution per store-week across the affected categories as the primary metric.
The metric includes product margin, display cost, and relevant substitution.
It also defines stockout rate and waste as guardrails.
The assignment unit, analysis period, and procedure for missing data are recorded in the protocol.

Suppose the treatment average is USD 1,250 per store-week and the control average is USD 1,200 per store-week.
The unadjusted difference is USD 50 per store-week in this illustrative calculation.
That difference is not automatically precise enough to justify rollout.
The sample size, store variation, randomization scheme, and resulting uncertainty interval are required.
If the interval covers both substantial loss and substantial gain, the outcome is inconclusive.

Suppose instead that the display is associated with USD 50 more contribution but a large stockout increase.
The team compares the result with its prespecified guardrail and operational capacity.
An apparently positive average may not justify expansion if the service loss is unacceptable.
Conversely, a small measured effect with a narrow interval entirely below the minimum worthwhile gain can support stopping.
These decisions depend on the protocol; the historical lift does not override it.

If several displays were tested and the team reports only the best one, selection affects interpretation.
Account for the number of comparisons or use independent confirmation before a large rollout.
If a few stores never installed the display, preserve the assignment-based comparison and disclose compliance.
An analysis restricted to compliant stores can be biased because compliance may depend on store characteristics.
The final record should distinguish what was planned, assigned, implemented, measured, and decided.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
It reconstructs the four-week synthetic example from basket-item records, with a duplicate scan and a noneligible return receipt.
The generated basket identifiers are illustrative and contain no personal information.
The code keeps empty eligible baskets in the denominator, even though they have no A or B line.
Replace the fixture and eligibility policy with audited point-of-sale data before real use.

```python
from collections import defaultdict

# N describes all eligible receipts, including those without either focal item.
basket_ids = {f"b{i:05d}" for i in range(18_240)}
lines = []

# 684 contain both; 1,140 contain only A; 2,052 contain only B.
for i in range(684):
    basket = f"b{i:05d}"
    lines.extend([(basket, "pasta"), (basket, "sauce")])
for i in range(684, 1_824):
    lines.append((f"b{i:05d}", "pasta"))
for i in range(1_824, 3_876):
    lines.append((f"b{i:05d}", "sauce"))

# Duplicate scan is still one presence; a noneligible receipt is ignored.
lines.append(("b00000", "pasta"))
lines.append(("return-only", "sauce"))

items_by_basket = defaultdict(set)
for basket, item in lines:
    if basket in basket_ids:
        items_by_basket[basket].add(item)

def rule_metrics(baskets, items, antecedent, consequent):
    if not baskets:
        raise ValueError("eligible basket population must not be empty")
    if antecedent == consequent:
        raise ValueError("use distinct, non-overlapping item concepts")
    n = len(baskets)
    n_a = sum(antecedent in items.get(b, set()) for b in baskets)
    n_b = sum(consequent in items.get(b, set()) for b in baskets)
    n_ab = sum(
        antecedent in items.get(b, set()) and consequent in items.get(b, set())
        for b in baskets
    )
    if not n_a or not n_b:
        raise ValueError("confidence or lift is undefined for a zero marginal")
    support = n_ab / n
    confidence = n_ab / n_a
    baseline = n_b / n
    lift = confidence / baseline
    leverage = support - (n_a / n) * baseline
    return n, n_a, n_b, n_ab, support, confidence, baseline, lift, leverage

n, n_a, n_b, n_ab, support, conf, baseline, lift, leverage = rule_metrics(
    basket_ids, items_by_basket, "pasta", "sauce"
)
print(f"N={n}, A={n_a}, B={n_b}, both={n_ab}")
print(f"support={support:.4f}, confidence={conf:.4f}, baseline={baseline:.4f}")
print(f"lift={lift:.2f}, leverage={leverage:.4f}")
```

Expected output:

```text
N=18240, A=1824, B=2736, both=684
support=0.0375, confidence=0.3750, baseline=0.1500
lift=2.50, leverage=0.0225
```

The example does not estimate uncertainty, identify causal effects, or solve a privacy review.
The sets implement binary item presence; the `basket_ids` set defines the denominator independently of item lines.
The two focal labels must represent non-overlapping concepts; the function cannot validate a product hierarchy.
For production-scale data, use a warehouse query or sparse representation, with equivalent deduplication and count reconciliation.
Do not export individual receipts merely to reproduce a summary if approved aggregate computation is available.

## 8. Data, standards, and regulatory context

### Data contract

Document the source system, refresh cadence, eligible dates, basket key, time zone, and store population.
Track transaction status, item identifier, quantity, price, promotion, return status, and mapping version where needed.
Capture whether a source is complete for all eligible channels and stores.
Retain a reconciliation between source transaction counts, excluded counts, mapped basket counts, and the analysis denominator.
A merchandising hierarchy is a business definition, not a universal scientific taxonomy.

A Global Trade Item Number (GTIN) is a GS1 identification key for trade items.
GS1 standards help identify products consistently, but a GTIN alone does not define a basket, category role, product substitution, or a valid causal analysis.
Map historical identifiers and package changes carefully; one shopper need can span several GTINs.
An internal SKU may differ by retailer and may not map one-to-one to a GTIN.
Consult current GS1 documentation for the applicable identification rules and implementation guidance.

### Privacy and data protection

Receipt-level analysis can sometimes be done without a loyalty identifier.
When linkage to a person or household is not needed, avoid collecting or joining it for this use.
A basket identifier may still be linkable through other tables, especially with precise time and store information.
Pseudonymization reduces some risks but is not automatically anonymization.
Aggregate tables can disclose rare behavior when small cells or overlapping queries allow reconstruction.
Apply appropriate access controls, minimum-cell rules, query controls, retention limits, and review of exports.
No universal small-cell threshold guarantees anonymity in every setting.

For processing within the scope of the European Union General Data Protection Regulation (GDPR), principles include data minimization, purpose limitation, storage limitation, and appropriate security under Article 5.
Article 6 addresses lawful bases, and Article 25 addresses data protection by design and by default.
Recital 26 and Article 4 are relevant to identifiability and pseudonymization.
Whether an organization has a lawful basis or must meet additional obligations depends on the facts and jurisdiction.
A business purpose alone is not a substitute for the applicable legal analysis.

California's privacy regime includes rights and obligations concerning personal information under the California Consumer Privacy Act as amended, subject to applicability and exceptions.
Targeting, sharing, and sensitive data can raise additional issues; use current regulator guidance and counsel for the specific workflow.
Do not assume that consent is always the only legal basis or that an opt-out is universally sufficient.
This educational discussion is not legal advice and is current in framing as of November 2026; verify current law before deployment.

### Governance and access

Separate model development from activation permissions.
An aggregate placement study and individualized offer targeting may require different reviews.
Limit role access to the minimum data and transformations necessary for the approved purpose.
Document deletion and retention schedules for intermediate basket tables, extracts, and backups.
Log product mapping versions and changes to eligibility criteria so results can be reproduced without retaining unnecessary identifiers indefinitely.
Assess whether rare product combinations could expose sensitive attributes or behavior even without names.

## 9. Industry practice and case evidence

Agrawal, Imieliński, and Swami's 1993 paper introduced association-rule mining in the context of large transaction databases.
It formalized support and confidence as measures for discovering relationships in transaction data.
Agrawal and Srikant's 1994 work on fast algorithms made frequent itemset mining a practical research direction.
Those papers establish methods, not proof that a particular modern retailer's display generates incremental profit.

The Instacart Market Basket Analysis competition released de-identified, sampled order data for a product-reordering prediction task.
Its published description distinguishes orders and products and provides a concrete example of a real online grocery data structure.
The competition's prediction objective is not a randomized evaluation of placement or the causal effect of co-purchase.
A model that predicts reorders is not, by itself, a justification for a targeted promotion.

In practice a category team should link any candidate rule to its category role, customer mission, margin, stock, and space constraints.
A destination category might be evaluated partly on traffic and downstream category contribution.
A convenience category might be judged on availability and friction reduction.
These are analytical decision choices, not claims that a named company achieved a particular outcome.

A common popular anecdote pairs beer and diapers and attributes the association to a specific retail discovery.
It is frequently repeated with inconsistent provenance.
Do not use it as evidence of a documented causal outcome or transfer its claimed pattern to a new population without primary evidence.
The credible practice is to reproduce the analysis in the retailer's own data and test a defined action.

**Evidence boundary:** published algorithm papers support the definitions and search approaches.
The Instacart dataset page supports the existence and structure of the public competition dataset.
Neither source validates the synthetic numbers or guarantees that a particular merchandising intervention will work.
The example's economic forecast and proposed experiment are instructional assumptions.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong basket key | Receipt IDs recur across stores | Unrelated lines merge | False pairs | Use source-verified composite key |
| Duplicated lines | Pair counts exceed marginals | Join multiplies rows | Inflated support | Deduplicate basket-item presence |
| Excluded empty baskets | N changes when focal items change | Denominator selected on outcome | Inflated prevalence | Build population independently |
| Returns mishandled | Negative quantities coexist with sales | Return-only and sale receipts mixed | Misleading presence | Define and audit transaction policy |
| Overlapping concepts | B always appears in A | Taxonomy embeds consequent | Tautological rule | Use disjoint concepts or disclose overlap |
| Out-of-stocks ignored | Affinity falls on low-availability days | Nonpurchase is not exposure | Wrong placement choice | Measure availability and sensitivity |
| Promotion confounding | Lift spikes during feature week | Common price or display changes | False causal attribution | Stratify descriptively and randomize action |
| Sparse high lift | Very few pair baskets | Rare baseline inflates ratio | Unstable ranking | Require counts and uncertainty |
| Multiple comparisons | Only the top rule is reported | Many chances for extremes | False discovery | Disclose search, adjust or hold out |
| Temporal drift | Holdout lift differs materially | Season or assortment changes | Poor generalization | Evaluate across periods and monitor |
| Cluster dependence | Narrow intervals from millions of receipts | Receipts share store conditions | Overconfidence | Use appropriate store-level inference |
| Revenue-only economics | Sales rise but margin falls | Discount and substitution omitted | Unprofitable rollout | Measure net incremental contribution |
| Treatment contamination | Control stores use the display | Assignment does not equal exposure | Attenuated or unclear effect | Audit execution and analyze assignment |
| Privacy leakage | Rare combinations appear in exports | Linkage or differencing identifies behavior | Harm and compliance risk | Minimize data and control queries |
| Simpson's paradox | Overall and segment trends differ | Mix of store or shopper populations | Misleading pooled comparison | Show relevant strata and their counts |
| Wrong intervention | Rule used to justify an unrelated discount | Association does not identify mechanism | Wasted test budget | Name mechanism and intervention |

A warning is a prompt to investigate, not automatic proof that a result is invalid.
Document which issues are observable and which remain assumptions.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to assess |
|---|---|---|---|
| Basket key identifies an occasion | Co-occurrence must refer to one transaction | Reused receipt IDs or split orders | Reconcile keys with source documentation |
| Population is complete | Support and baseline require correct N | Missing stores or empty receipts | Compare counts to source control totals |
| Product mapping is stable | A and B must mean consistent concepts | Reclassification or SKU changes | Version hierarchy and inspect coverage |
| Presence is accurate | Counts represent purchases | Duplicates, voids, missing lines | Audit raw-to-clean samples |
| Relevant availability is measured | Absence should be interpreted in context | Unrecorded stockouts | Compare inventory and shelf audit signals |
| Window is relevant | Historical pattern informs the planned period | Holiday, promotion, or regime shift | Compare comparable weeks and holdout |
| Enough independent clusters exist | Cluster-aware uncertainty needs information | Few stores or common shocks | Inspect cluster counts and variation |
| Holdout is untouched | Replication is not another selection step | Repeated tuning on holdout | Freeze candidate list and log accesses |
| Experiment assignment is valid | Treatment and control must be comparable in expectation | Nonrandom allocation or changes after assignment | Audit randomization and deviations |
| Outcome captures substitution | Profit should reflect the whole affected decision | Sales shift to another SKU | Include reasonable substitutes and costs |
| No material interference | One store's treatment should not alter control outcomes | Shopper switching or shared regional ads | Map exposure and consider cluster design |
| Data use is authorized | Access and activation must be lawful | Unapproved linkage or incompatible purpose | Obtain privacy and legal review |

The independence benchmark is a descriptive reference distribution, not an assertion that shopping decisions are naturally independent.
Even an interval excluding lift 1 does not establish causation.
Inference depends on sampling and design assumptions; the operational decision additionally depends on costs and risk tolerance.

## 12. Exercises

### Exercise 1 — Define the basket population

**Time:** 15 minutes.
A retailer has receipt lines from stores and online orders, including canceled orders, returns, and repeat scans.
**Deliverable:** a one-page population contract for a four-week store-layout decision.
**Acceptance criteria:** identify the basket key, included channels, exclusions, denominator, deduplication, and reconciliation checks.
Explain how you would treat eligible receipts with none of the focal products.

### Exercise 2 — Calculate a rule

**Time:** 15 minutes.
Among 4,000 eligible baskets, 400 contain A, 600 contain B, and 180 contain both.
**Deliverable:** a 2×2 table and support, confidence, B prevalence, lift, and leverage.
**Acceptance criteria:** use basket counts, label denominators, and show the arithmetic.

### Exercise 3 — Reverse the rule

**Time:** 10 minutes.
Use the counts in Exercise 2.
**Deliverable:** confidence and lift for B → A and a two-sentence interpretation.
**Acceptance criteria:** distinguish a directional conditional probability from a symmetric lift.

### Exercise 4 — Choose a candidate

**Time:** 20 minutes.
Candidate X appears in 500 of 20,000 baskets, has lift 1.4, and concerns a viable display.
Candidate Y appears in 10 of 20,000 baskets, has lift 8.0, and has unknown stock and margin.
**Deliverable:** a short recommendation and three additional measurements.
**Acceptance criteria:** do not equate lift with value; identify uncertainty, economics, and operational feasibility.

### Exercise 5 — Critique the inference

**Time:** 15 minutes.
A manager reports that a promoted pair has lift 3.0 and claims that moving A beside B will triple sales.
**Deliverable:** a critique naming at least three alternative explanations and one defensible descriptive statement.
**Acceptance criteria:** distinguish the observed comparison, promotion, causal effect, and statistical selection.

### Exercise 6 — Design the test

**Time:** 25 minutes.
A retailer can change displays in some stores but not independently for individual shoppers.
**Deliverable:** a trial outline with assignment, primary metric, guardrails, power or precision inputs, and decision rule.
**Acceptance criteria:** randomize at a feasible unit, include substitution and costs, and address execution and interference.

### Exercise 7 — Review privacy

**Time:** 15 minutes.
An analyst proposes exporting loyalty-linked baskets and exact timestamps to a shared folder for a store layout study.
**Deliverable:** a revised data flow and review checklist.
**Acceptance criteria:** minimize identifiers, restrict access and retention, address rare-cell disclosure, and flag legal review.

### Exercise 8 — Communicate the decision

**Time:** 20 minutes.
The trial estimated an increase of USD 100 per store-week, with an uncertainty interval from −USD 80 to +USD 280 per store-week.
The minimum worthwhile gain is USD 60 per store-week after costs.
**Deliverable:** a short decision note to the category director.
**Acceptance criteria:** state the inconclusive range, do not assert success or no effect, and propose a proportionate next step.

## 13. Self-check answers

### Exercise 1 rubric

A strong contract uses a source-verified store/date/register/receipt key where necessary.
It includes eligible store receipts even when neither focal item appears.
It excludes canceled or return-only transactions according to source semantics and records their counts.
It deduplicates item presence and reconciles the denominator to point-of-sale totals.
Online orders should be excluded or defined separately if they do not match the layout decision.
A weak answer uses only A and B line records to determine N.

### Exercise 2 calculation

| Eligible baskets | B present | B absent | Total |
|---|---:|---:|---:|
| A present | 180 | 220 | 400 |
| A absent | 420 | 3,180 | 3,600 |
| Total | 600 | 3,400 | 4,000 |

Support is 180 / 4,000 = 0.045, or 4.5% of baskets.
Confidence A → B is 180 / 400 = 0.45, or 45% of A baskets.
B prevalence is 600 / 4,000 = 0.15, or 15% of baskets.
Lift is 0.45 / 0.15 = 3.0.
Leverage is 0.045 − (0.10 × 0.15) = 0.03, or 3 percentage points.
The independence benchmark is 4,000 × 0.015 = 60 baskets; the descriptive excess is 120 baskets.
Neither the excess nor lift is an estimate of intervention-attributable purchases.

### Exercise 3 calculation

Confidence B → A is 180 / 600 = 0.30, or 30% of B baskets.
The baseline prevalence of A is 400 / 4,000 = 0.10.
Lift B → A is 0.30 / 0.10 = 3.0, the same as the forward lift.
Confidence changes because the conditioning population changes.
Neither rule specifies which product was selected first.

### Exercise 4 rubric

X is a plausible first candidate for economic investigation because its pair count is materially larger and the display is feasible.
Y's larger lift is based on only 10 pair baskets and could be unstable.
This is not enough to prove X has greater profit potential.
Request margin and incremental cost estimates, availability, and stability across stores or periods.
A strong answer may choose Y for a niche-specific investigation if it explains the objective and data needed.

### Exercise 5 rubric

Plausible alternatives include a joint promotion, a shared meal occasion, an existing display, seasonality, and basket-size differences.
Selection of the best rule from many candidates can exaggerate the observed lift.
A defensible statement is that, in the defined population and period, B was observed with A three times as often as the overall B baseline predicts under independence.
The statement does not say moving A caused anything or will triple sales.

### Exercise 6 rubric

Randomize eligible stores, ideally with justified pre-period stratification, and analyze the store-level assignment.
Prespecify the outcome as net incremental contribution across relevant categories, including costs and substitution.
Record availability, waste, and display compliance as guardrails or implementation measures.
Use historical store-level variance, store count, duration, and minimum worthwhile effect to plan precision or power.
Address regional advertising, shared customers, missing outcomes, and contamination.
Specify a rule for rollout, stopping, or further data before viewing outcomes.

### Exercise 7 rubric

Compute approved aggregates inside the controlled environment without loyalty linkage if individual history is unnecessary.
Use role-restricted access, retention and deletion rules, and disclosure controls for rare combinations.
Do not equate a hashed loyalty identifier with anonymity.
Escalate purpose, legal basis, and any targeted activation to appropriate privacy and legal reviewers.
A strong answer records who may access raw data and why the data are required.

### Exercise 8 rubric

The point estimate exceeds the USD 60 per store-week threshold, but the interval also includes material losses and gains below the threshold.
The evidence is therefore insufficient to claim the minimum worthwhile effect has been established.
Depending on costs and preregistered rules, continue or redesign the test for more precision, or stop if risk is unacceptable.
Report the assignment unit, interval method, execution, and guardrails before making a rollout decision.

## 14. Interview questions

1. **What is the difference between support and confidence?**
   A strong answer names the all-basket and antecedent-basket denominators; a weak answer treats them as synonyms.
2. **What does lift 1 mean?**
   A strong answer describes the selected population's independence benchmark; a weak answer says the products have no possible causal relationship.
3. **Why can a rare pair have high lift?**
   A strong answer discusses the low baseline, joint counts, and uncertainty; a weak answer ranks on lift alone.
4. **Why do the two rule directions have different confidence?**
   A strong answer identifies different conditioning sets and equal two-item lift; a weak answer infers purchase order.
5. **How would you define a basket?**
   A strong answer uses source-specific identifiers and eligibility; a weak answer assumes any receipt ID is globally unique.
6. **What does a promotion do to interpretation?**
   A strong answer explains common causes and separate causal testing; a weak answer assigns all co-purchase to placement.
7. **How would you handle many candidate rules?**
   A strong answer states the search scope and holdout or multiplicity plan; a weak answer reports only the largest lift.
8. **What would you randomize for a shelf display?**
   A strong answer considers store assignment, clustering, contamination, and power; a weak answer treats every receipt as an independently assigned treatment.
9. **Which economic outcome matters?**
   A strong answer includes incremental contribution, substitution, and cost; a weak answer totals observed basket revenue.
10. **Can hashed loyalty identifiers be treated as anonymous?**
    A strong answer discusses linkability and privacy review; a weak answer assumes hashing removes all personal-data obligations.

## 15. Key terms

- [Basket](#4-definitions-and-vocabulary): defined purchase occasion.
- [Basket key](#5-the-mechanics-step-by-step): source-specific transaction identifier.
- [Itemset](#4-definitions-and-vocabulary): set of distinct items in a basket.
- [Antecedent](#4-definitions-and-vocabulary): rule condition.
- [Consequent](#4-definitions-and-vocabulary): item evaluated within the condition.
- [Support](#5-the-mechanics-step-by-step): joint basket share.
- [Confidence](#5-the-mechanics-step-by-step): conditional basket share.
- [Prevalence](#5-the-mechanics-step-by-step): unconditional item share.
- [Lift](#5-the-mechanics-step-by-step): confidence relative to prevalence.
- [Leverage](#5-the-mechanics-step-by-step): absolute joint-share departure from independence.
- [Independence benchmark](#6-worked-example): product of marginal shares.
- [Contingency table](#6-worked-example): four joint presence and absence counts.
- [Deduplication](#5-the-mechanics-step-by-step): one presence per item and basket.
- [Availability](#5-the-mechanics-step-by-step): ability to purchase a product.
- [Confounding](#10-failure-modes-and-red-flags): shared causes of observed association.
- [Multiple testing](#5-the-mechanics-step-by-step): selection risk across many rules.
- [Holdout](#5-the-mechanics-step-by-step): data reserved for evaluation.
- [Cluster](#5-the-mechanics-step-by-step): group of dependent observations or assignment units.
- [Randomization](#5-the-mechanics-step-by-step): allocation by a specified chance mechanism.
- [Counterfactual](#3-the-core-mental-model): outcome under the alternative condition.
- [Incremental profit](#6-worked-example): attributable profit difference after relevant costs.
- [Cannibalization](#4-definitions-and-vocabulary): transferred rather than added sales.
- [Pseudonymization](#8-data-standards-and-regulatory-context): replacement of identifiers with potential relinkage.
- [GTIN](#8-data-standards-and-regulatory-context): GS1 trade-item identification key.

## 16. Sources

- Agrawal, R., Imieliński, T., and Swami, A. (1993), “Mining Association Rules between Sets of Items in Large Databases,” *Proceedings of the ACM SIGMOD International Conference on Management of Data*, pp. 207–216. [DOI: 10.1145/170035.170072](https://doi.org/10.1145/170035.170072). Original association-rule formulation.
- Agrawal, R., and Srikant, R. (1994), “Fast Algorithms for Mining Association Rules,” *Proceedings of the 20th International Conference on Very Large Data Bases*, pp. 487–499. [VLDB proceedings](https://www.vldb.org/conf/1994/P487.PDF). Frequent-itemset algorithms.
- Benjamini, Y., and Hochberg, Y. (1995), “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing,” *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300. [DOI: 10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x). A formal multiple-testing approach subject to its assumptions.
- GS1, [GTIN identification](https://www.gs1.org/standards/id-keys/gtin). Trade-item identifier overview.
- European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Primary legal text; consult current consolidated text and legal advice for application.
- California Privacy Protection Agency, [California privacy law](https://cppa.ca.gov/regulations/). Current regulations and rulemaking information.
- Instacart and Kaggle, [Instacart Market Basket Analysis](https://www.kaggle.com/competitions/instacart-market-basket-analysis). Public competition and dataset description; prediction task, not causal evidence.
- Hernán, M. A., and Robins, J. M., [*Causal Inference: What If*](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/), Chapman & Hall/CRC, 2020. Counterfactual reasoning and causal design.

## 17. Links

- [Module 17 index](./_index.md).
- [Assortment decision stack](./01_assortment-decision-stack.md).
- [Category roles and decision trees](./03_category-roles.md).
- [Module 17 survey and its lab sections](../../../course/17_assortment_category_retail_analytics.md).
- [Labs and electives index](../../05_labs_electives/_index.md).
