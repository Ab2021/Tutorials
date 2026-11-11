## 4. Core Formulas with Full Derivation

Every result in this section is an algebraic identity, not a fitted relationship. The five-part anatomy is: FORMULA, WHERE (symbol, unit, range), CONSTRAINT (the condition under which the identity is a valid statement about the store), NUMERICAL EXAMPLE (arithmetic shown to the cent), INTERPRETATION (what a practitioner may and may not conclude). Each subsection closes with an explicit validity condition.

The running dataset for all examples is a single store, two consecutive fiscal weeks. Week 0 is the baseline, week 1 the comparison. Values are ILLUSTRATIVE but internally consistent: every ratio below is recomputed from these four counts and one currency total, and the totals reconcile to the cent.

| Quantity | Symbol | Week 0 | Week 1 | Unit |
|---|---|---|---|---|
| Footfall | E | 8,600 | 8,400 | visits |
| Transactions | X | 1,806 | 1,789 | baskets |
| Units | U | 6,140 | 6,046 | units |
| Net sales | S | 25,788.00 | 25,242.05 | currency |
| Labour hours | H | 318 | 312 | hours |

### 4.1 Base case

```text
FORMULA [K-01]: S = E x c x A, with c = X/E and A = S/X
WHERE:
  S  [currency / store-week]  net sales, tax excluded, returns netted, range > 0
  E  [visits / store-week]    de-duplicated footfall, integer >= 0
  c  [fraction]               conversion, X/E, range (0, 1]
  A  [currency / basket]      average transaction value, S/X, range > 0
  X  [baskets / store-week]   completed transactions, integer >= 0
CONSTRAINT:
  (i) all four quantities cover the same seven trading days and the same opening hours;
  (ii) every transaction in X is attributable to a footfall event in E that week;
  (iii) no transaction is counted twice (split tenders, re-rung voids);
  (iv) the counter and the POS are cut at the same local midnight.
NUMERICAL EXAMPLE:
  c0 = 1806 / 8600   = 0.210000
  A0 = 25788.00 / 1806 = 14.279070
  check: 8600 x 0.210000 x 14.279070 = 1806 x 14.279070 = 25788.00  (rounds to the cent)
  week 1: c1 = 1789 / 8400 = 0.212976;  A1 = 25242.05 / 1789 = 14.109586
  check: 8400 x 0.212976 x 14.109586 = 1789 x 14.109586 = 25242.05
  change: 25242.05 - 25788.00 = -545.95, i.e. -2.1171%
INTERPRETATION:
  The identity is exact by construction, because E and X cancel: E x (X/E) x (S/X) = S.
  Therefore it carries no information until E, c and A are measured independently.
  Its use is to locate a move: sales fell 545.95, and the identity says the fall must be
  explained by traffic, conversion, basket, or an interaction among them.
```

The cancellation is the first thing to notice and the first thing to forget under pressure. A node tree built from this identity is *definitionally* true, so it can never be falsified by data; what can fail is the independence of the three measurements. If footfall is derived by dividing transactions by a modelled conversion rate, then the identity is circular and the "traffic effect" is a restatement of the conversion assumption. Before using K-01, ask how E was obtained. A beam counter, a video count, a Wi-Fi probe estimate, and a modelled figure are four different instruments with four different biases.

A second structural point: the identity fixes the *product* of the three factors but not their individual values. Two stores can post identical sales from very different combinations — 8,600 visits at 21.0% and 14.28 per basket versus 12,000 visits at 15.05% and 14.28 per basket — and the tree cannot tell you which store is healthier without the cost side. Section 9 handles the cost side; here the point is only that locating a move is not the same as evaluating a store.

Validity condition for K-01: valid whenever all four quantities are measured over the same population and window; invalid the moment one of them is modelled from another, because the identity then has fewer than three degrees of freedom and the decomposition is arithmetically exact but evidentially empty.

### 4.2 General case

The base case treats the store as one homogeneous population. Real stores trade through sub-populations — formats, day-parts, categories — and the correct general statement is a sum over strata, which reveals the weighting problem that produces most false conversion conclusions.

```text
FORMULA [K-02]: S = SUM_k ( E_k x c_k x A_k )   for strata k = 1..K
WHERE:
  E_k [visits / store-week]  footfall in stratum k
  c_k [fraction]             conversion within stratum k
  A_k [currency / basket]    average transaction value within stratum k
  w_k = E_k / SUM_j E_j      [fraction] footfall mix weight, SUM_k w_k = 1
CONSTRAINT:
  strata partition the footfall: every visit belongs to exactly one k, and
  SUM_k E_k = E.
NUMERICAL EXAMPLE (two strata, footfall split 70/30 then 50/50):
  week 0: E_A = 6020 at c_A = 0.2100, E_B = 2580 at c_B = 0.2100
          pooled c0 = (6020 x 0.2100 + 2580 x 0.2100) / 8600 = 1806 / 8600 = 0.210000
  week 1: E_A = 4200 at c_A = 0.2150, E_B = 4200 at c_B = 0.2110
          pooled c1 = (4200 x 0.2150 + 4200 x 0.2110) / 8400
                    = (903.00 + 886.20) / 8400
                    = 1789.20 / 8400 = 0.213000
  unweighted mean of the two stratum rates in week 1: (0.2150 + 0.2110)/2 = 0.213000
  same answer here only because the split is 50/50; recompute week 0 as an unweighted mean
  of 0.2100 and 0.2100 = 0.210000, also equal, so this example isolates the weighting
  rule rather than the mix. A mix that is unequal is shown in Section 4.4.
INTERPRETATION:
  The store rate is a footfall-weighted average of stratum rates. It moves when a stratum
  rate moves and when the weights move, and the two cannot be separated without the
  per-stratum table. Reporting the pooled rate alone conflates merchandising performance
  with who walked through the door.
```

The operational consequence is that the conversion node must be stored at stratum grain, not only at store grain. Storing only the pooled rate makes the mix effect invisible: a week where a high-converting stratum shrinks will look like a conversion collapse even if every stratum rate rose. Section 4.4 shows exactly that reversal with numbers, because it is the single most common misreading in retail performance packs.

The same structure applies to availability and to labour, with different weights. Store availability is a position-weighted average of route-section availability, weighted by planogram positions rather than by visits; weighting it by visits instead produces a number that rises when the least-visited sections are worst, which is precisely backwards from a customer-experience standpoint. Sales per labour hour is a hours-weighted harmonic mean of task-level productivity, so it is *not* the arithmetic average of department rates — an error that appears whenever a store reports department SPLH and then averages them.

Two identities follow from K-02 and are worth memorising because they justify the whole tree. First, ATV itself decomposes: A = UPT x ASP, since (U/X) x (S/U) = S/X. Second, contribution is additive in its cost terms: contribution = S x m − H x r − shrink − occupancy, where m is the gross margin rate and r the loaded hourly labour rate. The first says the basket node has two children that can be individually owned; the second says the cost side cannot be decomposed multiplicatively and must not be forced into the same tree shape.

Check the ATV identity against the running data: UPT0 = 6140/1806 = 3.399779 and ASP0 = 25788.00/6140 = 4.200000, so UPT0 x ASP0 = 3.399779 x 4.200000 = 14.279070 = A0. For week 1, UPT1 = 6046/1789 = 3.379542 and ASP1 = 25242.05/6046 = 4.175000, giving 3.379542 x 4.175000 = 14.109586 = A1. Both reconcile, which also confirms that the units count and the sales total were constructed consistently rather than independently rounded.

Validity condition for K-02: valid only if the strata partition footfall without overlap and the per-stratum rates are estimated from the same window as the weights. If the weights come from a loyalty panel and the rates from till data, the two are measured on different populations and the pooled number is a blend of incompatible samples.

### 4.3 Variants

Three attribution rules are defensible, and they answer different questions. Choosing one *before* looking at the numbers is the difference between analysis and motivated arithmetic.

```text
FORMULA [K-03a]: sequential (order E, then c, then A)
  step 1: d_E = E1 x c0 x A0 - S0
  step 2: d_c = E1 x c1 x A0 - E1 x c0 x A0
  step 3: d_A = E1 x c1 x A1 - E1 x c1 x A0
  identity: d_E + d_c + d_A = S1 - S0 exactly
WHERE:
  each step re-prices the running mix at the new value of exactly one factor, holding
  previously updated factors at their new value and later factors at their old value.
CONSTRAINT: the order must be fixed a priori; every order gives a different split and the
  same total.
NUMERICAL EXAMPLE, order E -> c -> A:
  step 1: 8400 x 0.210000 x 14.279070 = 1764.00 x 14.279070 = 25188.28
          d_E = 25188.28 - 25788.00 = -599.72
  step 2: 8400 x 0.212976 x 14.279070 = 1789.00 x 14.279070 = 25545.26
          d_c = 25545.26 - 25188.28 = +356.98
  step 3: 8400 x 0.212976 x 14.109586 = 1789.00 x 14.109586 = 25242.05
          d_A = 25242.05 - 25545.26 = -303.21
  total: -599.72 + 356.98 - 303.21 = -545.95  (equals S1 - S0)
INTERPRETATION:
  Sequential answers: what would sales have been if traffic had moved and nothing else had?
  It is the right rule when the business decision sequence is literally sequential, e.g.
  traffic is exogenous and basket is a response to the assortment chosen for that traffic.
```

```text
FORMULA [K-03b]: Shapley (average marginal contribution over all 6 orderings)
  phi_f = SUM over subsets T not containing f of [ |T|! (2-|T|)! / 3! ] x ( v(T u f) - v(T) )
WHERE:
  f in {E, c, A}; v(T) = product of factors with new values for members of T and old
  values otherwise; |T| is the size of subset T; weights 1/3, 1/6, 1/6, 1/3 by size.
CONSTRAINT: same total as K-03a; requires the multiplicative form to be exact.
NUMERICAL EXAMPLE, computed from the six orderings of the running data:
  phi_E = -600.38
  phi_c = +359.09
  phi_A = -304.66
  total = -545.95
INTERPRETATION:
  Shapley answers: on average across every order, how much did each factor contribute?
  It removes order dependence, which is what you want when no factor has a defensible
  causal priority. It is the fairest default for a performance pack.
```

```text
FORMULA [K-03c]: log-share (exact for the ratio, approximate for the level)
  ln(S1/S0) = ln(E1/E0) + ln(c1/c0) + ln(A1/A0)
  share_f   = ln(f1/f0) / ln(S1/S0)
  dollar_f  = share_f x (S1 - S0)
WHERE:
  all ratios positive; ln is natural log; shares sum to 1 by construction.
CONSTRAINT: dollar_f is a pro-rata assignment of the realised currency change, so it equals
  the log share of the *percentage* change, not of the level.
NUMERICAL EXAMPLE:
  ln(8400/8600)  = ln(0.976744) = -0.023530
  ln(0.212976/0.210000) = ln(1.014173) = +0.014073
  ln(14.109586/14.279070) = ln(0.988129) = -0.011940
  sum = -0.021398;  exp(-0.021398) - 1 = -0.021171  (matches -2.1171% to 4 decimals)
  dollar_E = (-0.023530 / -0.021398) x (-545.95) = +1.0997 x (-545.95) = -600.36
  dollar_c = (+0.014073 / -0.021398) x (-545.95) = -0.6577 x (-545.95) = +359.06
  dollar_A = (-0.011940 / -0.021398) x (-545.95) = +0.5580 x (-545.95) = -304.65
INTERPRETATION:
  Log-share answers: how much of the percentage change belongs to each factor, and it is
  the only one of the three that is invariant to the order in which factors are written and
  additive on the log scale. It is the safest when factors span different magnitudes,
  e.g. a 40% traffic shock next to a 0.3% price move.
```

Compare the three on the same data. Sequential with traffic first puts -599.72 on traffic; Shapley puts -600.38; log-share puts -600.36. The spread is under a dollar here because the moves are small, but the reason to prefer Shapley or log-share is not the size of the difference — it is that sequential's answer *changes* if you reorder. Reordering to A, then E, then c gives A = -306.09, E = -592.60, c = +352.74, which moves 7.12 of traffic effect onto basket purely by relabelling. An analyst who has not fixed the order can produce a 7-currency difference at will, and on a bigger move the same freedom is worth proportionally more.

A fourth variant is sometimes used and should be rejected: dividing the change by the old value to get percentage contributions, then multiplying each by the new total. That double-counts the change and will not sum to S1 − S0 for moves larger than a few percent. Its failure is not subtle — at a 20% move it over-allocates by roughly the square of the move — and its only advantage is that it is easy to compute in a spreadsheet cell, which is exactly how it propagates.

Validity condition for all of K-03: the factors must be measured at the same grain and window, and the multiplicative identity must hold exactly. If sales include a channel not present in footfall — for example click-and-collect where the customer never enters — then S ≠ E·c·A and none of the three rules is defined until the unmatched sales are removed from the numerator.

### 4.4 Limits and counterexamples

**Counterexample 1: the pooled-rate reversal.** Suppose the store's footfall splits between two formats. In week 0, format A supplies 6,000 visits at 20% conversion and format B supplies 3,000 visits at 30%. Pooled conversion is (6,000×0.20 + 3,000×0.30)/9,000 = (1,200 + 900)/9,000 = 0.233333. In week 1 the mix shifts: A supplies 3,000 visits at 19%, B supplies 6,000 at 29%. Pooled is (570 + 1,740)/9,000 = 2,310/9,000 = 0.256667. Pooled conversion rose 2.33 percentage points while *both* strata fell by one point. No arithmetic error occurred; the weights moved. Any bonus scheme or review that pays on the pooled rate pays for a mix shift the store did not create, and any investigation that follows the pooled move will search for a service improvement that does not exist.

**Counterexample 2: ratio of sums versus mean of ratios.** Daily conversion over a week is not the average of seven daily rates unless daily footfall is equal. Take three days with footfall 1,000, 4,000, 500 and transactions 250, 600, 100. The mean of daily rates is (0.250 + 0.150 + 0.200)/3 = 0.200. The ratio of sums is 950/5,500 = 0.172727. The gap of 2.7 points is a weighting artefact: the low-conversion day carried 57% of the visits but only a third of the weight in the average. The ratio of sums is correct for the week's sales question; the mean of ratios answers "what did a typical day look like", which is a different question. Publishing the wrong one is the most common cause of a "conversion doesn't match sales" dispute.

**Counterexample 3: percentage changes do not add.** Week-over-week, traffic fell 2.3256% and ATV fell 1.1869%, yet sales fell 2.1171%, not 3.5125%. The shortfall is the interaction term of roughly 0.5%: for small moves the cross product is negligible, for a 10% traffic fall and a 5% basket fall the interaction is 0.5% of sales, and for a 30% shock it is several percent. Any pack that adds percentage changes without an interaction row is quietly mis-stating the bridge, and the missing row always lands on whichever factor the reader least wants to blame.

**Boundary: zero or missing denominator.** If footfall is 0 — a closed day, a counter outage, a store with a broken door sensor — conversion is undefined, not zero. Writing 0 into the database converts an outage into a catastrophic performance result, and because 0 × anything is 0 the sales identity still "reconciles". The correct encoding is a null with a reason code, excluded from the pooled rate, and reported as a coverage percentage alongside it. The same rule applies to transactions of 0 (ATV undefined) and units of 0 (ASP undefined). A tree with three undefined ratios on one day must say so on the face of the report.

**Boundary: availability cannot exceed its audit route.** An availability of 100% means 100% of *audited positions were filled at the audit moment*. It does not mean the store had no gaps. If the route covers 150 of 420 positions, the estimate applies to 150 positions with a sampling error given in Section 4.1 of the measurement card, and to the remaining 270 positions it says nothing at all. Extrapolating to store level requires assuming the route is representative, which is an assumption about route design and must be stated as one.

**Boundary: labour productivity is not additive across channels.** If store hours include 40 hours of online order picking but online sales are booked to a digital channel, then SPLH for the store is understated by roughly the ratio of picked revenue to total store-attributed revenue. At 40 of 312 hours, a store with 25,242.05 of sales shows 80.90 per hour; reallocating 40 hours and an illustrative 3,200 of picked sales out of the store gives (25,242.05 − 3,200)/(312 − 40) = 22,042.05/272 = 81.04 per hour, and the online channel shows 3,200/40 = 80.00. The total is unchanged, but the store's number moved 0.17% purely by reclassification.

Validity condition for the whole of Section 4: every identity holds exactly when the inputs are measured independently at a common grain and window; every identity becomes circular, and every decomposition becomes an assumption dressed as arithmetic, when one input is derived from another.
