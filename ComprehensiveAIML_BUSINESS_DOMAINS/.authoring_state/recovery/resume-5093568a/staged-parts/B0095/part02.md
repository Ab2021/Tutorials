
## 6. Analysis Walkthrough

The walkthrough answers one question in four steps: what is the cheapest defensible answer threshold, does that threshold actually beat the incumbent process, which stage of the pipeline limits coverage, and what does the effective-date rule buy when it is priced rather than asserted. Every count below is constructed; every arithmetic step is shown so the reader can substitute their own inputs.

### 6.1 Deriving the answer threshold from costs, not from taste

**FORMULA**

```text
tau* = (C_e - C_a) / C_w
```

**WHERE**

| Symbol | Meaning | Units | Valid range |
|---|---|---|---|
| `tau*` | break-even probability that a delivered answer is unauthorised | dimensionless | 0 to 1 |
| `C_e` | incremental cost of escalating one question to a human | USD per question | > 0 |
| `C_a` | cost of delivering one answered question, verification included | USD per question | > 0 |
| `C_w` | expected cost of one unauthorised commitment reaching the customer | USD per event | > 0 |

Cost components, each built from a time study or an audit rather than asserted:

```text
C_a = skim_min * loaded_rate + compute_per_question
    = 1.5 * 0.60 + 0.20 = 1.10 USD

C_e = handle_min * loaded_rate + downstream_per_escalation
    = 13.0 * 0.60 + 1.40 = 9.20 USD

C_w = honoured_commitment + remediation + review
    = 96 + 28 + 16 = 140 USD
```

**CONSTRAINT**

The derivation holds only when four conditions are met: costs are linear in question count, so a thousand escalations cost a thousand times one; `C_e > C_a`, otherwise no question should ever be escalated; `C_w` is estimated from breaches that were actually detected, which makes it a lower bound; and the model's confidence is calibrated, so that a stated probability of 0.04 means four in a hundred across the population rather than four in a hundred within a convenient slice.

**NUMERICAL EXAMPLE**

```text
tau* = (9.20 - 1.10) / 140
     = 8.10 / 140
     = 0.05786
```

Check the break-even directly: an answer with unauthorised probability 0.05786 costs `0.05786 * 140 + 1.10 = 8.10 + 1.10 = 9.20`, which equals the escalation cost exactly. Below that probability answering is cheaper; above it, escalating is cheaper.

**INTERPRETATION**

The threshold is a property of the operating economics, not of the model. If the loaded agent rate rises from $0.60 to $0.72 per minute, `C_e` becomes `13.0 * 0.72 + 1.40 = 10.76` and `C_a` becomes `1.5 * 0.72 + 0.20 = 1.28`, so `tau* = 9.48 / 140 = 0.0677`. The threshold moves by a full percentage point because labour got 20 percent dearer, with no change to the model or the corpus. A design that fixes the threshold at 0.05 or 0.10 and never revisits it is therefore tracking the wrong object.

### 6.2 Does the threshold beat the incumbent process?

**FORMULA**

```text
T(theta) = N_a(theta) * C_a + N_e(theta) * C_e + N_w(theta) * C_w
```

**WHERE**

| Term | Meaning | Units |
|---|---|---|
| `T(theta)` | total cost of the answer policy at threshold `theta` | USD per quarter |
| `N_a(theta)` | questions answered under the policy | questions |
| `N_e(theta)` | questions escalated under the policy | questions |
| `N_w(theta)` | answered questions whose answer is unauthorised | questions |
| `N` | total questions, with `N_a + N_e = N` | questions |

The marginal form is the one a practitioner actually applies to a score band `b` with observed wrong-answer rate `p_b`:

```text
accept band b  <=>  p_b * C_w + C_a < C_e
               <=>  p_b < 0.05786
```

**CONSTRAINT**

`N_w` is the sum of per-band wrong rates weighted by band volume, so the total cost is a piecewise-linear, generally non-convex function of the threshold. It is non-convex whenever bands are not sorted by wrong rate, which happens when the confidence score is miscalibrated across surfaces: the store-handheld slice can have a high score and a high wrong rate at the same time. Non-convexity means the marginal rule can accept a band and reject a cheaper one, so bands must be sorted by observed wrong rate, not by score.

**NUMERICAL EXAMPLE**

Band structure over the questions that survive retrieval, verification, and authority checks — 36,734 of 48,000:

| Band | Questions | Observed unauthorised rate | Accept at tau*? | Contribution to `N_w` |
|---|---|---|---|---|
| 0.90 to 1.00 | 8,400 | 0.006 | yes | 50 |
| 0.80 to 0.90 | 10,600 | 0.039 | yes | 413 |
| 0.70 to 0.80 | 9,300 | 0.124 | no | 1,153 |
| below 0.70 | 8,434 | 0.270 | no | 2,277 |

Three policies compared against the baseline of full human handling for all 48,000 questions:

```text
Baseline:     48,000 * 9.20                                    = 441,600
tau* = 0.0579: 19,000 * 1.10 + 29,000 * 9.20 + 463 * 140
               = 20,900 + 266,800 + 64,820                    = 352,520
tau  = 0.0100:  8,400 * 1.10 + 39,600 * 9.20 +  50 * 140
               =  9,240 + 364,320 +  7,000                    = 380,560
tau  = 0.1300: 28,300 * 1.10 + 19,700 * 9.20 + 1,616 * 140
               = 31,130 + 181,240 + 226,240                   = 438,610
```

Marginal check on the middle band, which is the band the threshold decision actually turns on:

```text
accept band 2: 10,600 * 1.10 + 413 * 140 - 10,600 * 9.20
             = 11,660 + 57,820 - 97,520 = -28,040   (saves money)
accept band 3:  9,300 * 1.10 + 1,153 * 140 -  9,300 * 9.20
             = 10,230 + 161,420 - 85,560 = +86,090  (loses money)
```

**INTERPRETATION**

The saving over baseline at `tau*` is `441,600 - 352,520 = 89,080`, or 20.2 percent of the baseline cost. The saving is not proportional to the answer rate: answering 39.6 percent of questions produces a 20.2 percent saving, because the answered questions still carry a 2.4 percent wrong rate and because the escalated 60.4 percent is where most of the money stays. A pitch that quotes the answer rate as if it were the saving is overstating the case by roughly a factor of two, in either direction depending on which rate and which cost component is chosen.

The asymmetry between the two marginal tests is the transferable result. Accepting the 10,600-question band saves $28,040 per quarter; accepting the 9,300-question band below it loses $86,090. Two adjacent bands that look alike in score differ by more than $114,000 in consequence, which is why the threshold has to be located on measured band economics rather than interpolated.

### 6.3 Which stage actually limits coverage?

**FORMULA**

```text
A/N = (1 - z) * r5 * v * a * g
```

**WHERE**

| Factor | Meaning | Units | Illustrative value |
|---|---|---|---|
| `z` | share of questions whose candidate set is empty after the authorization predicate | fraction | 0.065 |
| `r5` | share of remaining questions whose controlling clause is in the top five | fraction | 0.912 |
| `v` | share whose intended quotation appears verbatim in the cited clause version | fraction | 0.964 |
| `a` | share whose implied commitment sits inside the acting tier | fraction | 0.931 |
| `g` | share of survivors whose band is accepted at `tau*` | fraction | 19,000 / 36,734 = 0.5172 |

**CONSTRAINT**

The product is valid only if the stages are conditionally independent given the question. They are not: a question whose controlling clause is a long nested exception has both low retrieval recall and low verbatim-quote pass rate, because the exception text is unusual and the model paraphrases it. Positive dependence makes the product an upper bound on true coverage, and the bound loosens as dependence grows. The rule of thumb is that the product is usable for prioritisation and not usable as a forecast when any pairwise correlation between stage residuals exceeds roughly 0.3.

**NUMERICAL EXAMPLE**

```text
A/N = 0.935 * 0.912 * 0.964 * 0.931 * 0.5173
    = 0.935 * 0.912        = 0.85272
    0.85272 * 0.964        = 0.82202
    0.82202 * 0.931        = 0.76530
    0.76530 * 0.5173       = 0.39583

questions answered = 0.39583 * 48,000 = 19,000
```

Stage counts, checked by successive multiplication:

```text
48,000 * (1 - 0.065) = 44,880   have a candidate set
44,880 * 0.912       = 40,931   controlling clause retrieved
40,931 * 0.964       = 39,457   quotation verified
39,457 * 0.931       = 36,734   commitment inside authority
36,734 * 0.5173      = 19,000   band accepted
```

Elasticity, derived by differentiating the logarithm of the product: because `ln(A/N) = ln(1-z) + ln(r5) + ln(v) + ln(a) + ln(g)`, a relative gain of 1 percent in any stage yields a relative gain of 1 percent in coverage. Converting to the absolute movement needed for each additional percentage point of coverage:

```text
needed relative gain = 0.01 / 0.39583 = 0.02526
recall:    0.02526 * 0.912  = 0.0230   -> +2.30 points
verifier:  0.02526 * 0.964  = 0.0244   -> +2.44 points
authority: 0.02526 * 0.931  = 0.0235   -> +2.35 points
gate:      0.02526 * 0.5173 = 0.0131   -> +1.31 points
```

**INTERPRETATION**

The gate is the cheapest place to buy coverage, because it is the only stage sitting near one half while the others sit above nine tenths. That conclusion is uncomfortable, because raising the gate acceptance means accepting a higher wrong-answer rate, which is exactly what the cost derivation forbids at `tau*`. The resolution is that gate coverage can only be improved by better calibration, not by a looser threshold: if the confidence score separated bands more sharply, the acceptable mass would grow at constant `tau*`. Retrieval improvements, by contrast, raise coverage without touching the wrong-answer rate, but they cost 1.8 times as much per point because recall starts high.

### 6.4 What the effective-date predicate is worth, priced rather than asserted

**FORMULA**

```text
E = N * s * (w / L) * C_w
```

**WHERE**

| Symbol | Meaning | Units | Illustrative value |
|---|---|---|---|
| `E` | expected quarterly cost of answering from a clause that is published but not yet effective | USD per quarter | computed |
| `N` | questions per quarter | questions | 48,000 |
| `s` | share of questions whose controlling clause family is amended during the quarter | fraction | 0.042 |
| `w` | gap between publication and effective date | days | 9 |
| `L` | amendment cycle length for that family | days | 182.5 |
| `C_w` | cost of one unauthorised commitment | USD per event | 140 |

**CONSTRAINT**

Valid when amendments are published before they take effect and the interval `w/L` is small relative to 1, so that at most one pre-effective window is open per clause at a time. It also assumes the index refresh does not matter, which is true only if the predicate is evaluated against the clause's own validity fields at query time. If instead the predicate is baked into a nightly index rebuild, the relevant interval is the rebuild lag `delta`, not `w`, and the formula must be re-derived with `delta` in place of `w`.

**NUMERICAL EXAMPLE**

```text
E = 48,000 * 0.042 * (9 / 182.5) * 140
  = 2,016 * 0.049315 * 140
  = 99.4 * 140
  = 13,919 USD per quarter
```

The same formula with a seven-day index rebuild lag and no query-time predicate, taking `w = 0` because amendments are effective on publication:

```text
E' = 48,000 * 0.042 * (7 / 182.5) * 140
   = 2,016 * 0.038356 * 140
   = 77.33 * 140
   = 10,826 USD per quarter
```

**INTERPRETATION**

Two things follow, and the second is the one that changes architecture. First, the dollar value of the predicate is small — about $13,900 per quarter, which is under a sixth of the gate's saving — so pricing it as an economic optimisation would lose the argument against a cheaper alternative. It is justified as a compliance invariant under constraint C-01, not as a return on investment, and mixing the two framings lets a cost-benefit review retire a control that exists to make a class of error impossible.

Second, the comparison of `E` and `E'` is misleading on its own, because the two quantities are different kinds of failure. A stale index that still contains the prior clause version, with that version's own interval, will have the prior clause excluded by a query-time predicate: correctness is preserved and the loss is recall, which shows up as a smaller candidate set. A stale index whose text has been updated while the interval has not is the dangerous case, because the predicate then admits the wrong wording while the interval test still passes. The invariant is therefore not "filter by date" but "version the text and the interval together, and never refresh one without the other".

### 6.5 Where the formulas break

Each derivation above has a stated constraint; three concrete counterexamples show how the break is observed rather than argued.

**The heavy-tail break in 6.1.** The break-even threshold assumes `C_w` is an average. If one answer in twenty thousand is screenshot and forwarded to a regulator, the cost distribution is heavy-tailed, the mean is unstable, and the expected-cost comparison understates the loss. The diagnostic is the ratio of the 99th-percentile commitment cost to the mean: under 5, the linear model holds; above 20, no threshold computed from averages is defensible and the gate must be replaced by a hard rule that prohibits the class of clause involved.

**The dependence break in 6.3.** Recompute recall separately for questions whose controlling clause contains a nested exception list. In this constructed case their share is 0.22 of questions and their recall is 0.771 against 0.951 for the rest, giving a weighted recall of `0.22 * 0.771 + 0.78 * 0.951 = 0.1696 + 0.7418 = 0.9114`, which matches the aggregate 0.912. The aggregate is right and the product is still an upper bound, because verifier pass on the same subset is 0.884 rather than 0.964. Multiplying the subset's own four factors gives `0.935 * 0.771 * 0.884 * 0.931 * 0.5172 = 0.307`, against 0.396 for the population, so exception questions get answered at about 78 percent of the population rate. The aggregate funnel never reveals this; a sliced funnel does.

**The interval break in 6.4.** If `valid_to` is stored as an empty string for open-ended clauses, a comparison written as `valid_to > t` silently drops every open-ended clause, which in this corpus is 61 percent of the clause count. The predicate still runs, still returns rows, and still looks healthy: the only symptom is that `z` jumps from 0.065 to 0.41. A corpus with an unexpectedly high zero-candidate share is therefore an interval-encoding bug until proven otherwise, and the check is to count clauses by whether `valid_to` parses as a date.

Takeaway: every number here is a decision input with a stated validity condition, and the conditions fail in ways that produce plausible-looking output.
