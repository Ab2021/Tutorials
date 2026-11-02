# A/B, A/B/n, and Cluster Randomisation — the three workhorse designs and when each is legal

> Module 05 — Experimentation, Pilots, and Evidence
> Topic 03 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 04 topics 03–05 and 07; Module 05 topics 01–02
> Version 2026.5

---

## 1. Why this topic exists

Topic 01 turned a claim into an estimand. Topic 02 chose the unit the estimand attaches to. This
topic is about the small set of designs you are actually allowed to run, and why the set is smaller
than the textbooks suggest.

A design is **legal** here in three senses, and all three must hold at once. **Statistically
legal**: the analysis the design demands is available — enough independent units, enough traffic per
arm, no unmodelled dependence between arms. **Operationally legal**: the treatment can reach some
units without reaching others, which shared inventory, a shared fleet, a shared sales force, and a
shared system configuration each prevent. **Commercially legal**: the varied attribute is not fixed
by contract, tariff, or a most-favoured-customer (MFN) clause.

Most design arguments in industry are conducted in the first sense and settled by the third. Four
failures show the shape.

**One. The four-variant test reported as one finding.** A team runs control plus three challengers
at 25% traffic each, sees p = 0.04 on the best, and reports "variant C wins with 95% confidence".
Under a true null, the chance that at least one of three comparisons crosses 0.05 is
`1 − 0.95³ = 14.3%`. Worse, the reported effect is the maximum of three noisy estimates, biased
upward by construction. The design was legal; the claim was not.

**Two. The store test analysed as an order test.** A cluster design does not merely need a wider
interval; it demands a variance component — the intra-cluster correlation (ICC) — that the analyst
must know before launch, and that decides whether the pilot is worth running at all.

**Three. The price test that legal stopped.** A subscription business designs a customer-level
discount test across two regions. Legal blocks it: the largest account has a most-favoured-customer
clause and one product line has a published tariff. The design was statistically immaculate and
never runnable. Discovering that after the power calculation costs a quarter of analyst time.

**Four. The factorial that could not be delivered.** A team wants to test a pick-path rule and a
slotting rule in a 2×2. The slotting rule changes only per site; the pick-path rule is per zone. The
four cells cannot exist inside one site, so the factorial collapses into a split-plot with two error
strata and a handful of degrees of freedom. Nobody noticed until the analysis.

> **Note:** None of the four is an arithmetic error. Each is a mismatch between the design chosen
> and the constraints the business operates under. That is the failure this topic prevents.

**Why the family is small.** There are three workhorses: the two-arm A/B test, the multi-arm A/B/n
test, and the cluster-randomised design. Factorial designs are a structured special case of A/B/n —
the arms are combinations rather than alternatives. Everything else in this module is a choice of
unit (Topics 04, 05) or of stopping rule (Topic 08).

**The claim of this topic.** Design choice in supply chain work is usually forced, not free. Your
job is to identify which constraint binds, price the design that constraint leaves you, and say
plainly what it cannot answer. An analyst who argues for the ideal design against a binding
operational constraint is not being rigorous. They are being unhelpful.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a described test into the design family and **state** the analysis obligation that
   classification creates.
2. **Apply** the three legality tests and **name** the binding constraint, whether statistical,
   operational, or commercial.
3. **Compute** the precision cost of moving from two arms to `k` at fixed traffic, in percentage
   points and relative terms.
4. **Distinguish** a main effect from an interaction, **explain** why the interaction costs roughly
   four times the sample, and **decide** which one the decision needs.
5. **Design** a 2×2 or 2³ factorial, **state** its resolution, and **identify** aliased effects.
6. **Compute** design effect, effective sample size, and minimum detectable effect (MDE) for a
   cluster-assigned test, and **choose** among analysis routes on stated grounds.
7. **Justify** a refusal: given a design that is legal but cannot detect the hoped-for effect,
   **write** the sentence that says so without proposing an impossible alternative.
8. **Critique** a reported multi-arm or cluster result for multiplicity, allocation, analysis level,
   and compliance handling, and **rewrite** the claim so that it survives.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Two questions decide the design. Ask them in order, because the second is constrained by the first.

```text
   QUESTION 1: HOW MANY VERSIONS MUST THE TEST CARRY?

                        2 versions              more than 2
                            |                        |
                        +-------+          +---------+----------+
                        |       |          |                    |
                      A / B   A/B/n     FACTORIAL            (bandit)
                     (1 comp) (k arms)  (2^k cells,        adaptive, not a
                                        structured)        fixed design
                            |                |
                            |                +--> main effects cheap
                            |                     interactions expensive
                            |
                            +--> one comparison, no correction,
                                 maximum precision per arm

   QUESTION 2: WHAT RECEIVES THE COIN?

              individual unit                   group of units
        (session, order, customer,        (store, DC, branch, lane,
         SKU, shipment, invoice)           sales territory, market)
                    |                                |
             standard A/B or A/B/n            CLUSTER RANDOMISATION
             analysis at the same unit        analysis must carry a
             or a variance correction         variance component
```

The first branch is about ambition: how many alternatives must the decision distinguish? The second
is about delivery: what is the smallest object the treatment can be confined to? The design is the
intersection, and only four intersections matter.

### 3.1 The three legality gates

```text
        PROPOSED DESIGN
              |
              v
   +--------------------+   no   +----------------------------------+
   | GATE 1  STATISTICAL|------->| different design, or declare the  |
   | enough independent |        | test descriptive only             |
   | units and traffic? |        +----------------------------------+
   +--------------------+                       ^
              | yes                             |
              v                                 |
   +--------------------+   no   +----------------------------------+
   | GATE 2  OPERATIONAL|------->| coarsen the unit, or hold the     |
   | treatment confined |        | shared resource fixed             |
   | to treated units?  |        +----------------------------------+
   +--------------------+                       ^
              | yes                             |
              v                                 |
   +--------------------+   no   +----------------------------------+
   | GATE 3  COMMERCIAL |------->| vary another attribute, or run a  |
   | attribute free of  |        | policy pilot instead of a test    |
   | contract, tariff?  |        +----------------------------------+
   +--------------------+
              | yes
              v
        RUNNABLE DESIGN
```

The gates are ordered by cost of discovery, not importance. Gate 1 is cheap and usually checked.
Gate 3 is cheap and almost never is, and it wastes the most calendar time.

### 3.2 Why the variance splits, in three facts

Every design here partitions variance, and the analysis of variance (ANOVA) is that partition's
accounting. You do not need the derivation.

**Fact one.** Total variance separates into what the design's structure explains and what is left
over. The ratio of the two, scaled by degrees of freedom, is the F statistic. Any design change
alters one of those two quantities.

**Fact two.** The design decides which variance counts as "within". In a two-arm test on 1.2 million
sessions, within-arm variance is session noise. In a cluster test on 240 branches, it includes
branch-to-branch differences, which are large. The design did not change the data; it changed what
counted as noise.

**Fact three.** Each factor added to a factorial removes variance from the residual at the cost of
a degree of freedom. That is why a well-chosen factorial beats several separate tests, and why an
unlucky factorial with many cells and few units per cell is worse than two clean A/B tests.

> **Note:** If a report gives an effect and a standard error but not the degrees of freedom, you
> cannot tell which variance was in the denominator, and so cannot tell whether the design was
> analysed as designed.

### 3.3 The factorial picture

A 2×2 factorial is four cells, and the interesting quantity is usually not in any cell.

```text
   2 x 2 FACTORIAL: PROMISE BANNER (A) x FREE-RETURNS BADGE (B)

                        B = off              B = on
                    +-------------+      +-------------+
      A = off       |   cell 00   |      |   cell 01   |
                    |  control    |      |  badge only |
                    +-------------+      +-------------+
                    +-------------+      +-------------+
      A = on        |   cell 10   |      |   cell 11   |
                    | banner only |      |  both on    |
                    +-------------+      +-------------+

   MAIN EFFECT of A  =  mean(cell10, cell11) - mean(cell00, cell01)
   MAIN EFFECT of B  =  mean(cell01, cell11) - mean(cell00, cell10)
   INTERACTION A x B = (cell11 - cell10) - (cell01 - cell00)

   The interaction asks: does the banner do something different when the
   badge is also present? It is a difference of differences, which is why
   its variance is four times a main effect's and its MDE twice as wide.
```

The interaction is the reason to run a factorial. Two separate A/B tests give the same precision for
main effects; you run the factorial because a rollout that turns both on is a bet on cell 11.

### 3.4 What practitioners argue about

**Fixed-horizon versus adaptive allocation.** A multi-armed bandit (MAB) shifts traffic toward the
arm that looks better as data arrives, minimising regret — revenue lost by serving inferior variants
during the test — at the cost of the clean inference a fixed split provides. The question is whether
you need a defensible estimate of the difference or a defensible policy for the next four weeks.

**Whether factorials are worth the complexity.** The classical online-experimentation position is
that strong interactions are rare and concurrent univariate tests usually beat a factorial. That is
wrong more often in supply chain than in web product work, because supply chain interventions
routinely interact: a routing rule with a slotting rule, a price with a service promise.

**Whether a cluster test with twenty clusters is worth running.** It is legal and usually
uninformative. The honest answer is that it is a descriptive exercise with an interval attached.

### 3.5 What each design can and cannot estimate

| Design | Can estimate | Cannot estimate | Typical supply chain use |
|---|---|---|---|
| Two-arm A/B | One average effect, precisely | Any interaction; any segment effect at useful precision | A single copy, layout, or policy change delivered per unit |
| A/B/n | `k − 1` effects against control, each less precisely | All pairwise differences at equal precision; interactions | Choosing among three carrier options or four slot offers |
| Factorial | Main effects at A/B precision; interactions at ~4× the sample | Effects in cells not run (fractional designs) | Two process changes that might interact |
| Cluster | One average effect, with a variance component | Anything below cluster level; anything needing many clusters | Store rollouts, distribution centre system changes, carrier policy |
| Bandit (contrast) | A policy; a biased estimate of arm means | An unbiased effect estimate without reweighting | Live optimisation, not evidence generation |

## 4. Definitions and vocabulary

The fourth column names the unit or coding convention. A definition without one is the defect this
module exists to prevent.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| A/B test | A two-arm randomised comparison of a control and one treatment | Used generically for any randomised test | Arm label, 0/1 |
| A/B/n test | A comparison of one control against two or more treatments | Called a "multivariate test" | Arm label, 0..k |
| Assignment key | The field whose hash decides the arm | Confused with the user identifier | Column name, e.g. `session_id` |
| Hash salt | A constant mixed into the assignment hash so tests do not collide | Omitted, so two tests assign identical units | Constant string per experiment |
| Stratum | A pre-treatment partition within which randomisation is balanced | Used for any segment | Stratum identifier |
| Blocking | Randomising separately within strata | Confused with analysis-time covariate adjustment | Block identifier |
| Factor | An attribute varied by the experiment | Called "the treatment" when there are several | Factor name |
| Level | One value of a factor | Confused with an arm | Factor name plus value |
| Cell | One combination of factor levels, hence one factorial arm | Called "a variant" loosely | Cell identifier |
| Factorial design | A design in which factors vary together so every combination occurs | Used for any multi-arm test | Factor list |
| Fractional factorial | A deliberate subset of combinations preserving certain effects | Called "a shortcut" | `2^(k−p)` cells |
| Main effect | Average change from moving one factor, averaged over the others | Reported as if it held in every cell | Outcome units |
| Interaction effect | Extent to which one factor's effect depends on another's level | Called "a segment effect" | Outcome units |
| Aliasing | Confounding of one effect with another in a fractional design | Discovered after data collection | Effect pair |
| Resolution | Which orders of effect a fractional design can separate, III to V | Quoted without saying what it forbids | Roman numeral |
| Multivariate test (MVT) | A test varying several page attributes at once | Used for A/B/n, which varies one attribute across versions | Design label |
| Multi-armed bandit (MAB) | Adaptive allocation shifting traffic toward better arms | Reported with fixed-horizon p-values | Algorithm name and parameters |
| Regret | Cumulative outcome lost by serving inferior arms during a test | Confused with the effect size | Outcome units, cumulative |
| Fixed-horizon test | A test whose sample size and stopping point are set in advance | Assumed to forbid any monitoring | Sample size plus end date |
| Sequential test | A test with a pre-specified valid early-stopping rule | Implemented as repeated uncorrected looks | Alpha-spending function |
| Family-wise error rate (FWER) | Probability of at least one false positive across a family | Quoted for a single comparison | Probability, 0 to 1 |
| False discovery rate (FDR) | Expected proportion of false positives among declared findings | Used when any false positive is unacceptable | Proportion, 0 to 1 |
| Bonferroni correction | Divide alpha by the number of comparisons | Applied to comparisons you never intended | Adjusted alpha |
| Holm procedure | Step-down FWER control, uniformly more powerful than Bonferroni | Assumed to be a different guarantee | Adjusted p-value |
| Benjamini–Hochberg | Step-up procedure controlling FDR | Used for confirmatory claims | Adjusted p-value |
| Guardrail metric | Metric that must not degrade, used to veto a winner | Chosen after the result is known | Definition plus threshold |
| Overall evaluation criterion (OEC) | The single pre-declared metric the decision is made on | Left implicit | Metric definition |
| Sample ratio mismatch (SRM) | Observed allocation inconsistent with the planned split | Treated as a minor operational issue | Chi-square p-value |
| Cluster randomisation | Randomising groups rather than the units inside them | Called "stratified randomisation" | Cluster identifier |
| Cluster | A group assigned together, sharing one arm | Used for any segment | Cluster identifier |
| Intra-cluster correlation (ICC) | Share of total variance lying between clusters | Assumed zero when unmeasured | Dimensionless, 0 to 1 |
| Design effect (DEFF) | Variance under clustering divided by variance under individual assignment | Reported as a count | Dimensionless, at least 1 |
| Effective sample size | Row count divided by the design effect | Reported as the row count | Count of units |
| Cluster-level analysis | One summary per cluster, analysed as `k` observations | Used with very unequal sizes without weighting | One row per cluster |
| Cluster-robust standard error | Variance estimate allowing arbitrary within-cluster correlation | Used below about 20 clusters unadjusted | Cluster identifier |
| Intention-to-treat (ITT) | Effect of assignment, regardless of compliance | Reported as the effect of treatment | Arm as assigned |
| Per-protocol | Effect among units receiving treatment as designed | Preferred because it looks larger | Arm as delivered |
| Compliance | Degree to which assigned units received the treatment | Assumed to be 100% | Proportion |
| Most-favoured-customer clause | Contractual promise that a buyer receives no worse terms than anyone else | Ignored when designing price tests | Contract clause |
| Tariff | A published regulated price schedule that must apply uniformly | Treated as a price like any other | Tariff schedule identifier |

Six of these carry most of the practical damage.

**A/B/n versus multivariate test.** An A/B/n varies *one* attribute across `k` versions: four
subject lines, four slot offers. A multivariate test varies *several* attributes at once and is a
factorial by another name. The distinction matters because the obligation differs: A/B/n obliges a
multiplicity correction, an MVT obliges an interaction model. Teams routinely run an MVT, analyse it
as an A/B/n, and report main effects that were never separately randomised.

**Fixed-horizon versus sequential.** A fixed-horizon test does not forbid monitoring. It forbids
*deciding* on an unadjusted interim look. You may watch the dashboard; you may not act on it without
an alpha-spending plan agreed at launch. Topic 08 gives the mechanics.

**FWER versus FDR.** FWER is the probability of *any* false positive; FDR is the expected
*proportion* among the claims you make. A confirmatory rollout decision wants FWER, because one
false rollout is expensive. An exploratory screen of forty segments wants FDR, because you will
follow up the survivors. Using FDR for a confirmatory claim is the commonest multiplicity error in
commercial analytics.

**Cluster versus stratified randomisation.** Stratification partitions units and randomises *within*
partitions. Cluster randomisation randomises *groups*. They are unrelated, are often used together,
and are confused constantly because both involve grouping. A stratified A/B test is still an A/B
test.

**ITT versus per-protocol.** ITT estimates the effect of *offering*; per-protocol estimates the
effect of *receiving*. Per-protocol looks larger because compliers are the motivated ones. For a
rollout decision ITT is right, because the organisation rolls out an offer, not a guaranteed
receipt. Report per-protocol beside ITT, never instead of it.

**Design effect versus effective sample size.** The design effect is a variance ratio; the effective
sample size is a count derived from it. Reporting "we had 1.2 million sessions" when the effective
sample is 11,900 is not a rounding difference. It is a claim about evidence that the design cannot
support.

## 5. The mechanics, step by step

Six phases, twenty-five steps. Each step names the check that fails if it is skipped.

### Phase A — Establish legality before designing anything

**Step 1. Write the decision as a sentence with a role, an action, a unit, and a date.**

"The trading director will switch the delivery promise to next-day for all e-commerce sessions, or
not, from the start of the third quarter (Q3)." That fixes the decision, its unit, and the deadline.
*Fails if skipped:* the test answers a question nobody is deciding, and the result is filed.

**Step 2. Enumerate every attribute the treatment touches.**

Not "the new promise" but: the product-page banner, the checkout estimate, the carrier selection
rule, the branch reserve-stock policy, and the service script. Five attributes, not at the same
level. *Fails if skipped:* you discover in week three that the treatment also changed the carrier
rule, a shared resource, and the design is invalid.

**Step 3. Apply the commercial-uniformity test to each attribute.**

| Question | If yes | Consequence |
|---|---|---|
| Is the price fixed by a published tariff or schedule? | The price cannot vary by unit | Vary the service, bundle, or communication |
| Does a contract contain a most-favoured-customer clause? | That buyer must receive the best terms | Exclude the account, or test a non-price attribute |
| Does a franchise or distributor agreement require uniform terms? | The attribute is fixed network-wide | Test at network level, or not at all |
| Does public procurement equal treatment apply? | Tenderers must be treated alike | No test on that population |
| Does the attribute touch a protected characteristic? | Assignment on it may be unlawful | Re-specify the intervention |

Twenty minutes with legal, and it saves quarters. Note that it constrains *what you vary*, not
*whether you may experiment*: a tariff-bound price can still be tested as a bundle or a service.

**Step 4. Apply the shared-resource and capacity tests.**

| Shared resource | Where it bites | Design consequence |
|---|---|---|
| Inventory | Two arms draw from one stock pool | Assign at the pool, or hold the pool fixed |
| Fleet and equipment | Carriers reallocate across lanes | Assign at the carrier, or contract capacity |
| Sales force | One rep serves accounts in both arms | Assign at the rep's span or territory |
| Warehouse labour | Pickers work both arms' orders | Assign at shift, zone, or site |
| System configuration | A setting is global per site | The site is the unit, whatever you wanted |
| Marketing budget | Spend is reallocated across arms | Hold budget fixed, or assign at budget holder |
| Customer service | Agents see both arms and learn | Stagger, or measure and report the leak |

Then count the units that survive. Session-level A/B typically buys detection of a fraction of a
percentage point; customer-level, a low single-digit change; store-level, a double-digit change;
distribution centre (DC) level, a very large change or nothing. If the feasible unit yields "nothing
smaller than 25%", the design is legal and the pilot is not worth running as a confirmatory test.

### Phase B — Two-arm A/B: assignment mechanics

**Step 5. Choose the assignment key and a per-experiment salt.**

```text
   arm = bucket( hash( salt || unit_id ) mod 100 )

   salt  = "promise-2026q3-v1"        (frozen at launch, never reused)
   unit  = session_id, customer_id, order_id, or store_id
   arm   = 0 if bucket < 50 else 1    (50/50 split)
```

The function is deterministic, so a unit's arm never changes on refresh. It is computed at exposure,
so a unit that never arrives is never assigned, which is why assigned and observed counts differ.
*Fails if skipped:* the assignment is recomputed from a mutable field and arms drift.

**Step 6. Fix the allocation ratio, and know what imbalance costs.**

Fifty-fifty maximises precision at a fixed total. Unbalanced splits limit exposure to a risky change
and are legitimate, but not free: precision is proportional to `4r(1−r)` where `r` is the control
share.

Fifty-fifty maximises precision at a fixed total; a 70:30 split keeps 92% of that
precision, 90:10 keeps 60%, and 95:5 keeps 43%, which is a canary rather than a test.

**Step 7. Pre-register the primary metric, the guardrails, and the analysis.**

One primary metric, two to four guardrails each with a direction and threshold, one stated analysis,
one decision rule. A page, signed, before launch. *Fails if skipped:* the primary metric becomes
whichever moved.

**Step 8. Ramp without re-randomising.**

Increasing traffic from 10% to 50% to 100% is fine if the assignment function is unchanged and the
ramp is applied by eligibility, not by re-hashing. Re-hashing at each ramp changes arms mid-test and
produces an artifact that looks like an effect. *Check:* a unit exposed in week one has the same arm
in week four; verify by joining the exposure log to itself.

**Step 9. Run the sample ratio mismatch check before reading any result.**

Compare observed arm counts with the planned split by chi-square. A p-value below 0.001 means the
assignment, logging, or eligibility rule is broken and the result is uninterpretable regardless of
what it says.

> **Note:** SRM is not a nuisance check. It is the only routine diagnostic that catches a broken
> experiment before the estimate is believed. Treat it as a gate.

**Step 10. Freeze, analyse at the assignment unit, and write the decision down.**

"The director approved rollout on 14 August, on an estimate of +0.11 pp (percentage points)
conversion (95% CI, confidence interval, +0.01 to +0.21), against a rule requiring the lower bound to
exceed the incremental fulfilment cost." That sentence is the artifact.

### Phase C — A/B/n: multiplicity and its price

**Step 11. Choose `k` for a reason, not for curiosity.**

Every arm buys one alternative and costs precision on all the others. The dishonest reason for a
fourth arm is that traffic is available; the honest one is four options the business must choose
between.

**Step 12. Decide the comparison family before seeing the data.**

| Family | Comparisons | When it is right |
|---|---|---|
| Each treatment versus control | `k − 1` | Default; the decision is "does this beat the status quo" |
| All pairwise | `k(k−1)/2` | Rare; the decision is "which is best", and you must defend the cost |
| Ordered dose-response | 1 | The arms are doses and a trend test is the question |

The family determines the correction. Choosing the family after the data arrives is the error, not
choosing a particular correction.

**Step 13. Choose and state the multiplicity control.**

| Method | Controls | Cost | Use when |
|---|---|---|---|
| None | Nothing | Inflated false positives | One pre-declared comparison only |
| Bonferroni | FWER | Conservative, `alpha/m` | Few comparisons, any false positive costly |
| Holm | FWER | Same first threshold as Bonferroni, better after | Default for FWER |
| Benjamini–Hochberg | FDR | More power, weaker guarantee | Exploratory screening of many candidates |
| Hierarchical gate | FWER | Requires a pre-ordered primary | One primary, then secondaries only if it wins |

Holm's first step uses `alpha/m`, identical to Bonferroni, so it helps only from the second
comparison onward — which surprises teams expecting a rescue. And a gate-keeping procedure is
usually the most powerful defensible option when the business genuinely has one primary question.

**Step 14. Price the precision loss in the units the business uses.**

At fixed total traffic, moving from two arms to `k` divides per-arm sample by roughly `k/2`, raises
the standard error by `sqrt(k/2)`, and raises the MDE by the same factor before correction. Section
6.3 computes it: 2.44% relative becomes 3.46%, and 3.99% once Bonferroni is applied.

> **Note:** An A/B/n at fixed traffic does not take longer than the A/B it replaces. It produces a
> wider interval in the same calendar time.

### Phase D — Factorial designs: main effects and interactions

**Step 15. Choose factors and levels so that all cells are deliverable.**

Before designing, verify each combination can physically exist. Two settings that are global per
site cannot be crossed inside a site, and a price bundled by contract cannot be crossed with a
service promise. If any cell is undeliverable, this is not a factorial.

**Step 16. Decide full or fractional, and state the resolution.**

| Resolution | What it means | What it forbids |
|---|---|---|
| III | Main effects clear of each other; aliased with two-factor interactions | Interpreting any interaction |
| IV | Main effects clear of two-factor interactions; interactions aliased with each other | Separating interactions from one another |
| V | Main effects and two-factor interactions all clear | Very little, at the cost of many runs |

Resolution III is a screening tool: which factors matter, not how they combine. Resolution IV says
which pairs interact but not which interaction is which unless you fold over. Resolution V is the
design you want when the interaction is the decision.

**Step 17. Write the alias structure down and read it.**

For a `2^(k−p)` design, list every effect and what it is confounded with. If the effect you care
about is aliased with one you also care about, no analysis will fix it. Aliasing is knowable in
closed form before launch and unfixable after it.

**Step 18. Analyse as one model with interaction terms, then decide what the decision needs.**

Fit the outcome on the factor indicators and their products. Each coefficient is a main effect
averaged over the other factor's levels; the product coefficient is the interaction; the residual is
within-cell variation and every effect is tested against it. Then ask the decision question: a
rollout that switches both factors on is a decision about cell 11, not about the average. If the
interaction is material, report the four cell means with intervals. If negligible, the main effects
are the right summary and the factorial was an efficient way to get two answers.

### Phase E — Cluster randomisation: assignment, variance, analysis

**Step 19. Set the assignment unit to the smallest object that contains the treatment.**

Not the smallest object you can measure — the smallest the treatment cannot escape. A store layout is
contained by the store. A carrier contract is contained by the carrier, not the lane, because the
carrier reallocates equipment.

**Step 20. Block on pre-period predictors of the outcome.**

Pair or block clusters on prior volume, region, format, or prior outcome, then randomise within
blocks. With few clusters, blocking is the cheapest precision available and often the difference
between an informative test and an uninformative one. *Check:* blocks must be defined on pre-period
data only; defining them on the test-period outcome absorbs part of the effect.

**Step 21. Estimate or bound the ICC before launch.**

Three sources, in order of preference: the same metric in a pre-period across the same clusters; the
same metric in a comparable category or region; a plausible range, reported as a range. Never a single
borrowed number with no provenance.

**Step 22. Compute the design effect and the MDE at cluster level.**

```text
   DEFF              = 1 + (m - 1) * ICC           m = mean cluster size
   effective n       = n / DEFF
   var(cluster mean) = ICC * var + (1 - ICC) * var / m
   SE(diff)          = sqrt( 2 * var(cluster mean) / k )    k = clusters per arm
   MDE               = z * SE(diff)
```

The limit worth memorising: as `m` grows, effective `n` tends to `k / ICC`. Adding rows inside
clusters cannot exceed that ceiling.

**Step 23. Choose the analysis route on stated grounds.**

| Route | What you compute | Best when | Cost |
|---|---|---|---|
| Cluster-level summary | One mean per cluster, then a two-sample test on `k` values | Very few clusters | Loses within-cluster precision |
| Mixed model | Row-level model with a cluster random effect | Many clusters; cluster-specific estimand wanted | Assumes a distribution for the random effect |
| Cluster-robust standard error | Row-level model, variance corrected at cluster level | Many clusters, few parameters | Biased downward below about 20–40 clusters |
| Randomisation inference | Reassign treatment across clusters many times | Very few clusters; any statistic | Heavier computation; exact under the sharp null |

Below about 20 clusters per arm, the honest choices narrow to cluster-level summaries or
randomisation inference. A cluster-robust standard error with 12 clusters and no correction is not
conservative; it is wrong.

**Step 24. Handle compliance by reporting ITT, with the delivery rate beside it.**

If only 70% of assigned stores implemented the change, the ITT estimate is diluted by about 30%, and
that is the estimate the rollout decision needs. Report compliance and ITT, and per-protocol only if
you can defend the exclusion.

### Phase F — Report the design, not just the result

**Step 25. Report six fields, and state what the design could not have found.**

```text
  DESIGN      which family, how many arms or cells, assignment unit
  ALLOCATION  planned split, observed split, SRM p-value
  ANALYSIS    unit of analysis, model, degrees of freedom, correction applied
  EFFECT      estimate with interval, in outcome units and in relative terms
  POWER       the MDE the design could detect, stated before the result
  LIMIT       what this design cannot answer, in one sentence
```

"At 120 branches per arm and an ICC of 0.02, this design could not detect anything smaller than a 25%
relative change. A null result is consistent with a true effect of 20%." That sentence is the
difference between a decision and a guess.

## 6. Worked example

Northmoor Supply is a trade distributor with 240 branches and an e-commerce channel. All figures are
constructed to be internally consistent and checkable by hand.

### 6.1 The decision and the arithmetic base

The trading director will switch the product-page delivery promise to next-day for all e-commerce
sessions from the start of Q3, or not. Four weeks of traffic is 1,200,000 eligible sessions; current
conversion is 4.2%; contribution per order is GBP 11.20; the promise costs GBP 0.34 per order.

```text
  p     = 0.042                    baseline conversion
  var   = p(1 - p) = 0.040236      variance of a binary outcome

  break-even lift, treating cost as accruing on every order:
      delta* = (0.34 / 11.20) x 0.042 = 0.001275 = 0.1275 pp = 3.04% relative
```

The approximation ignores cost on the incremental orders; netting that out gives 0.1315 pp. Use
0.1275 pp and state the approximation. Every design below is judged by one question: can it detect a
0.1275 pp change, the smallest change that pays for itself?

### 6.2 Design 1 — two-arm A/B at session level

All three gates pass: the promise renders server-side per session, consumes no shared inventory at
rendering, and touches no price.

```text
  sessions per arm   600,000
  SE(diff) = sqrt(2 x 0.040236 / 600,000) = 0.0003662 = 0.03662 pp
  MDE      = 2.8016 x 0.0003662            = 0.001026 = 0.1026 pp = 2.44% relative
```

The MDE sits below the 0.1275 pp break-even, with headroom. That is what an adequately powered test
looks like.

> **Note:** The comparison is 0.1026 pp against 0.1275 pp, not 2.44% against 3.04%. Mixing the two
> framings is how a factor-of-40 error enters a business case.

### 6.3 Design 2 — A/B/n with four arms

The team also wants three copy variants. With a control that is four arms; traffic is fixed, so each
gets 300,000.

```text
  SE(diff) = sqrt(2 x 0.040236 / 300,000) = 0.0005179 = 0.05179 pp
  uncorrected MDE = 2.8016 x 0.0005179 = 0.001451 = 0.1451 pp = 3.46% relative
  Bonferroni alpha = 0.05/3 = 0.01667;  z = 2.394 + 0.8416 = 3.2356
  corrected MDE   = 3.2356 x 0.0005179 = 0.001676 = 0.1676 pp = 3.99% relative
```

The four-arm test cannot detect a change that pays for itself, even before correction. **The test did
not get longer. It got blunter.** Holm would not rescue it: its first threshold equals Bonferroni's.

**What the team did.** Ran the two-arm test for the decision, and ran the three copy variants as a
separately labelled descriptive exercise with no winner declared. One confirmatory question, one
exploratory question, different claims, different reporting.

### 6.4 Design 3 — a 2x2 factorial

A second question: does a free-returns badge change the promise's effect? Two sequential A/B tests,
or one 2x2 with 300,000 per cell.

```text
  main effect pools two cells = 600,000 per level
  SE(main effect)  = sqrt(2 x 0.040236 / 600,000) = 0.0003662
  MDE(main effect) = 0.1026 pp = 2.44% relative   (identical to the A/B)

  interaction is a difference of differences over four cells
  SE(interaction)  = 2 x 0.0003662 = 0.0007324
  MDE(interaction) = 2.8016 x 0.0007324 = 0.2052 pp = 4.89% relative
```

Two main effects at the precision of a dedicated A/B, plus one interaction at half the precision,
for the same 1.2 million sessions. If the decision is "turn both on", the interaction is the quantity
that matters and the one the design measures worst.

### 6.5 Design 4 — cluster randomisation, forced by shared inventory

To honour a next-day promise reliably, branches must hold reserve stock. Reserve stock is a shared
pool: a control session fulfilled from a branch holding reserve is partly treated. The promise
cannot be delivered honestly at session level once the inventory change is part of the treatment.
The smallest unit that contains the treatment is the branch: 240 branches, 120 per arm, about 5,000
sessions each.

```text
  ICC = 0.02 (assumed from pre-period branch-level conversion variation)
  var(branch mean) = 0.02 x 0.040236 + 0.98 x 0.040236 / 5,000
                   = 0.00080472 + 0.00000789 = 0.00081261
  SE(diff)         = sqrt(2 x 0.00081261 / 120) = 0.0036801 = 0.36801 pp
  MDE              = 2.8016 x 0.0036801 = 1.0310 pp = 24.5% relative

  DEFF = 1 + 4,999 x 0.02 = 100.98;  sqrt(DEFF) = 10.05
  effective n = 1,200,000 / 100.98 = 11,884
```

The design is legal, the arithmetic is right, and the test cannot see a change one-eighth the size
of the one that would pay for itself. The ICC decides this, and it is worth seeing how much.

| ICC | DEFF | sqrt(DEFF) | MDE (relative) | Verdict |
|---:|---:|---:|---:|---|
| 0.005 | 26.00 | 5.10 | 12.4% | Still far above break-even |
| 0.020 | 100.98 | 10.05 | 24.5% | Uninformative |
| 0.050 | 250.95 | 15.84 | 38.7% | Descriptive only |

To detect a 5% relative effect at ICC 0.02 would need 2,893 branches per arm — 5,786 in total. The
network has 240. No analysis changes that. The correct response is a different design, a stepped
rollout in time (Topic 05), or no test at all with the decision taken on operational judgement and
reported as such.

### 6.6 The finding

Four questions, four designs: the promise is tested at session level; the copy variants are a
descriptive exercise with no winner declared; the promise-and-badge question is a 2x2 whose main
effects are usable and whose interaction is not; and the reserve-stock question is not run at all.

That last line is the finding. A team that ran the cluster test anyway and reported a null would have
taught the business that reserve stock does not work.

## 7. Code, query, or calculation

One block, reproducing sections 6.2 to 6.5 so the parameters can be changed.

```python
# MDE arithmetic for one decision across four designs. Binary outcome, so var = p(1-p).
p, z = 0.042, 2.8016                  # baseline conversion; 80% power, 5% two-sided
v = p * (1 - p)                       # 0.040236

def mde(n_arm, zc=z, v=v):            # n_arm = independent observations per arm
    return zc * (2 * v / n_arm) ** 0.5

rel = lambda x: 100 * x / p
print("A/B          ", f"{mde(600_000)*100:.4f} pp  {rel(mde(600_000)):.2f}% rel")
print("A/B/n raw    ", f"{mde(300_000)*100:.4f} pp  {rel(mde(300_000)):.2f}% rel")
zb = 2.394 + 0.8416                   # alpha/3 = 0.0167, power 80%
print("A/B/n Bonf.  ", f"{mde(300_000, zb)*100:.4f} pp  {rel(mde(300_000, zb)):.2f}% rel")
print("factorial ME ", f"{mde(300_000)*100:.4f} pp  {rel(mde(300_000)):.2f}% rel")
print("factorial INT", f"{mde(150_000)*100:.4f} pp  {rel(mde(150_000)):.2f}% rel")

for icc in (0.005, 0.02, 0.05):
    m, k = 5000, 120
    vbm = icc * v + (1 - icc) * v / m
    se = (2 * vbm / k) ** 0.5
    print(f"cluster icc={icc:<5}", f"{z*se*100:.4f} pp  {rel(z*se):.2f}% rel",
          f"DEFF {1+(m-1)*icc:7.2f}")
```

Expected output: `A/B` 0.1026 pp 2.44%; `A/B/n raw` 0.1451 pp 3.46%; `A/B/n Bonf.` 0.1676 pp 3.99%;
`factorial ME` 0.1026 pp 2.44%; `factorial INT` 0.2052 pp 4.89%; cluster at ICC 0.005, 0.02, and
0.05 giving 12.45%, 24.55%, and 38.70% relative, with DEFF 26.00, 100.98, and 250.95.

Four checks before trusting any such block.

- **The factorial main-effect MDE must equal the A/B MDE at the same total traffic.** A balanced 2x2
  estimates each main effect over 600,000 observations per level. If they differ, the cell count or
  the traffic total is wrong.
- **The interaction MDE must be exactly twice the main-effect MDE**, because its variance is four
  times a main effect's. 0.2052 / 0.1026 = 2.00.
- **The square root of the design effect must equal the ratio of cluster MDE to A/B MDE.** 1.0310 /
  0.1026 = 10.05 = `sqrt(100.98)`. This identity is what makes the design effect useful.
- **The A/B/n MDE must scale as `sqrt(k/2)` before correction.** With `k = 4`, 0.1451 / 0.1026 =
  1.414. Linear scaling in `k` means per-arm sample was computed on total traffic.

> **Note:** `mde(150_000)` serves the interaction because `4v/300,000 = 2v/150,000`. That is a
> convenience, not a claim that 150,000 sessions sit in any arm; the four cells hold 300,000 each.

## 8. Data, standards, and regulatory context

### 8.1 What the data must contain before the design can be chosen

| Field | Why the design depends on it | What goes wrong without it |
|---|---|---|
| A stable assignment identifier | The hash is computed on it; the analysis groups by it | Arms drift between refreshes |
| The planned allocation, recorded at launch | The SRM check compares observed to planned | Every allocation looks fine in hindsight |
| An immutable assignment table with a timestamp | The design must be auditable | Arm membership recomputed from a changed field |
| An exposure log keyed to the assignment identifier | Compliance and contamination are measured, not assumed | Delivery rate unknown; ITT inseparable from dilution |
| The cluster identifier for every unit | Cluster designs need the nesting relation | Clusters reconstructed from geography after the fact |
| A pre-period outcome on the same definition | Blocking, ICC estimation, balance checks | No way to estimate the variance component pre-launch |
| Guardrail metrics defined before launch | Guardrails chosen after the result are not guardrails | Every result passes or fails on which metric was picked |
| The intervention cost per unit | MDE must be compared with break-even, not zero | A significant but unprofitable change is rolled out |

The exposure-log row fails most often. Without it you cannot distinguish "the treatment did not
work" from "the treatment was not delivered", and those imply opposite decisions.

### 8.2 Standards and rules that constrain what may be varied

| Requirement | Jurisdiction or body | What it constrains | Grade |
|---|---|---|---|
| Robinson–Patman Act, 15 U.S.C. §13 | United States | Price discrimination between competing buyers of commodities of like grade and quality where it injures competition; statutory defences include cost justification and meeting competition | B |
| Article 101, Treaty on the Functioning of the European Union (TFEU), and national competition law | European Union and member states | Agreements distorting competition; relevant where a test is coordinated with a supplier or distributor | B |
| Public procurement equal-treatment principle | EU directives and national implementations | Treating tenderers differently during a procurement | B |
| Sector tariff regulation | National regulators | Published tariffs that must apply uniformly to a class of customer | B |
| Franchise and distributor agreements | Contract | Uniform pricing, presentation, or service across a network | Contract |
| Most-favoured-customer clauses | Contract | Any term worse than the best term offered elsewhere | Contract |
| Regulation (EU) 2016/679, the General Data Protection Regulation (GDPR) | European Union | Processing personal data for assignment and measurement | B |

> **Note:** Robinson–Patman has a specific scope, several statutory defences, and a large
> interpretive literature. Nothing here is legal advice. The operational point is narrow: price is
> the attribute most likely to engage a legal constraint, so review it before you power it.

The pattern is the same across all seven rows: they constrain *what may differ between units*, not
*whether an experiment may run*. A tariff-bound price can still be tested as a bundle, a
communication, a service level, or a contract term.

### 8.3 Standards that define the unit and the report

| Standard | Governs | What it forces you to declare | Grade |
|---|---|---|---|
| CONSORT 2010 extension to cluster randomised trials (Campbell et al., 2012) | Health and public-health trials | Number of clusters, unit of randomisation, ICC, whether analysis allowed for clustering | A |
| ICH E9 and E9(R1) | Regulated clinical development | Pre-specified primary variable and analysis; estimand definition | B |
| ISO 3534-1:2016 | Statistical vocabulary | What "cluster", "sample", and "population" mean | B |
| ISO 8601-1:2019 | Date and time representation | What a test window is measured on | B |
| GS1 Global Location Number (GLN) and Global Trade Item Number (GTIN) | Location and trade-item identity | Whether a store or stock-keeping unit (SKU) is the same object across systems | B |
| AEA RCT Registry | Social-science field experiments | Pre-registration of unit and analysis plan | B |

None binds a commercial pilot, and each is a template. A retailer that adopted the CONSORT cluster
checklist for store tests would eliminate most of section 10's failures.

### 8.4 What is convention rather than requirement

- **"Always split 50/50."** Optimal for precision at fixed total, not optimal when exposure must be
  limited. A 90:10 split is legitimate with a computable cost.
- **"Never run more than four arms."** A vendor convention born of traffic limits. The real
  constraint is the MDE you can tolerate, which is a calculation.
- **"Twenty clusters is too few."** Directionally right for cluster-robust inference; below about
  20, change the analysis route rather than abandon the design.

An analyst who can separate a requirement from a convention can challenge the convention.

## 9. Industry practice and case evidence

### 9.1 Where the designs came from

| Contribution | Source | What it established | Grade |
|---|---|---|---|
| Randomised field experiments and the analysis of variance | Fisher (1926), "The Arrangement of Field Experiments," *Journal of the Ministry of Agriculture of Great Britain*, 33 | Randomisation, replication, blocking; factorial designs as efficient use of plots | A |
| Industrial factorial and fractional factorial practice | Box, Hunter and Hunter, *Statistics for Experimenters* (2nd ed., 2005); Montgomery, *Design and Analysis of Experiments* | Resolution, aliasing, and screening designs as standard | A |
| Bandit formulation | Robbins (1952), "Some Aspects of the Sequential Design of Experiments," *Bulletin of the AMS*, 58 | Adaptive allocation as a decision problem distinct from estimation | A |
| Probability matching | Thompson (1933), *Biometrika*, 25 | The ancestor of most modern bandits | A |

The commercial point: the factorial design is a hundred years old, was developed to extract more
information from a fixed number of units, and solves the problem a 240-branch network has.

### 9.2 What online experimentation established about multi-arm designs

| Finding | Source | Implication | Grade |
|---|---|---|---|
| Multi-variable tests can run as a full factorial (`2^5 = 32`), fractional factorial, or Plackett–Burman design (a two-level screening design) | Kohavi, Longbotham, Sommerfield and Henne (2008), "Controlled experiments on the web," *Data Mining and Knowledge Discovery* | The factorial machinery transfers directly | A |
| Running several independently randomised univariate tests concurrently is often preferred to a factorial | Same | A reasoned argument against factorial, from the field with most test volume | A |
| "Strong interactions are rare in practice," cited to van Belle | Same | The claim that justifies the convention; note it is a claim about web product attributes | A |
| Multiplicity, sequential testing, and metric definition are the dominant sources of invalid inference | Larsen et al. (2023), "Statistical Challenges in Online Controlled Experiments," *The American Statistician* | The correction problem is not a corner case | A |

The second row is the strongest argument against the design this topic spends a section on. It is a
good argument where attributes are independent and a 32-cell test is operationally fragile. It is
weaker in supply chain work, where factors are process settings that genuinely interact.

### 9.3 The multiplicity arithmetic, which is checkable

Under a global null with independent comparisons, the probability of at least one false positive is
`1 − (1 − α)^m`.

| Comparisons (`m`) | FWER at α = 0.05 | Bonferroni threshold |
|---:|---:|---:|
| 1 | 5.0% | 0.0500 |
| 3 | 14.3% | 0.0167 |
| 4 | 18.5% | 0.0125 |
| 6 | 26.5% | 0.0083 |
| 10 | 40.1% | 0.0050 |

These are exact arithmetic, verifiable by hand. Vendor glossaries publish the same figures (Grade D,
undated marketing page, retrieved 2026). The point is that a four-arm test has a **one in five**
chance of producing a spurious winner when nothing is happening — roughly the base rate at which
organisations run tests.

### 9.4 Bandits: measured rather than asserted

| Finding | Source | Detail | Grade |
|---|---|---|---|
| Thompson sampling raises false positives from 5% to about 13% and cuts power from about 80% to about 56% versus uniform randomisation | Williams et al. (2021), "Challenges in Statistical Analysis of Data Collected by a Bandit Algorithm," arXiv:2103.12198 | Two-arm binary experiments, naive analysis | C |
| Inverse-probability weighting and algorithm-induced critical values restore validity but cut power sharply, to roughly 17% in the authors' illustration | Same | Validity is purchasable, at a price most pilots cannot pay | C |
| Bandit and A/B designs differ in power as a function of traffic split; hybrids can improve both | Xiang et al. (2022), KDD, "Multi Armed Bandit vs. A/B Tests in E-commerce" | The choice is a frontier, not a binary | C |
| Commercial tools running repeated uncorrected significance tests create "the worst form of the multiple-testing problem" | Burtini, Loeppky and Lawrence (2015), arXiv:1510.00757 | Peeking and adaptivity compound | C |
| Online FDR control for bandit data is achievable with always-valid p-values | Yang, Ramdas, Jamieson and Wainwright (2017), NeurIPS | The methods exist; dashboards do not implement them | C |

All five are preprints or conference papers, graded C for venue. The transferable claim is
directional and robust: **adaptive allocation invalidates the fixed-sample test most dashboards run
on the resulting data.** If you cannot state which estimator and critical value you used, do not
report a p-value from a bandit.

> **Note:** Do not call the 13%-versus-5% result "bandits triple your false positives". Quote the
> simulation: naive analysis of Thompson-sampled two-arm data produced about 13% against a nominal 5%.

### 9.5 Cluster designs and shared inventory in supply chain settings

| Finding | Source | Detail | Grade |
|---|---|---|---|
| Districts were randomised rather than health facilities because facilities shared a commodity planner and district office | World Bank / ISRCTN86533387, essential-drugs supply chain trial, Zambia | 400+ facilities, 24 districts; stratified randomisation, difference-in-differences, randomisation inference | B |
| Cluster randomisation limits contamination but carries costs in sample size, recruitment, and power | Torgerson (2001), "Contamination in trials: is cluster randomisation the answer?", *BMJ* | The classic statement of the trade-off | A |
| Field experiments in operations face contagion between work groups and should randomise at group or facility level; non-compliance needs explicit treatment | Ibanez and Staats (2019), "Field Experiments in Operations Management," *INFORMS TutORials* | The methods review closest to this module's setting | A |
| Retail tests at scale face interference, imperfect adherence, and non-random assignment because field agents have different incentives | Costa et al., "Generalized Synthetic Control for TestOps at ABI" | Real deployment across thousands of stores; company-authored | C |
| Shared inventory between variants violates independence and can produce arbitrarily high false positive and false negative rates | Bohle, Marynych and Meiners (2021), *Applied Stochastic Models in Business and Industry* | In one ranking example, identical algorithms appeared significantly different about 20% of the time | A |
| Allocating limited test inventory trades number of stores tested against observation quality under stock-outs | Wang, Mersereau and Chen (2017), "Optimal Merchandise Testing with Limited Inventory," *Operations Research* | With unobservable transaction timing, consolidate inventory in fewer stores | A |

The Bohle et al. result is the formal version of Gate 2. When two arms draw from one inventory pool,
the arms are not independent, and the dependence grows as inventory tightens. **Any test whose arms
compete for the same stock is not an A/B test, whatever the assignment table says.**

The ABI paper names the failure mode Step 4 predicts: field agents with differing incentives left
some control stores partly treated and some test stores untreated. The authors report conclusions on
65 of 99 experiments where conventional difference-in-differences supported 6 — a company claim
about its own platform (Grade C), directionally consistent with the rest of the literature.

### 9.6 Heterogeneity: why segment findings are usually illusions

| Finding | Source | Detail | Grade |
|---|---|---|---|
| A trial powered at 80% for an overall effect has roughly 29% power to detect an interaction of the same size; detecting it needs about four times the sample | Brookes et al. (2004), *Journal of Clinical Epidemiology*, PubMed 15066682 | The canonical quantification | A |
| Subgroup-specific tests produced a "significant in one subgroup only" result in 7% to 64% of simulations under a constant effect | Same | The mechanism behind most segment stories | A |
| Interaction tests have roughly four times the variance of an overall effect test; pre-specification and multiplicity control are required for confirmatory claims | Wang, Piantadosi, Le-Rademacher and Mandrekar, *Journal of Thoracic Oncology*, PMC7920926 | Confirmatory versus exploratory subgroup analysis | A |
| Power for treatment-effect heterogeneity in cluster trials depends on both outcome ICC and covariate ICC | Baumann et al. (2025), *International Journal of Epidemiology*, doi:10.1093/ije/dyag069 | Cluster designs make heterogeneity analysis more expensive still | A |

So in a two-arm test sized for a 2.4% relative change, any segment finding is exploratory. Report it
with an interval, label it, and do not build a rollout rule on it.

### 9.7 What is missing from the public record

There is no public audit of design choice in commercial supply chain tests. Retailers do not publish
assignment tables, ICCs, or MDEs, and the failures in section 10 appear in conversation far more
often than in print. What the record supports is narrower: the arithmetic is settled, the cluster
methods are documented in clinical literature, and no evidence suggests the design family has
changed since about 2010. The tooling has. The three workhorses have not.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| A/B/n reported without correction | "The best variant hit p = 0.04"; the slide shows no adjusted threshold | Three or more comparisons at α = 0.05 inflate FWER | A spurious winner is rolled out | Declare the family and correction before launch |
| Winner's curse on the maximum arm | The reported effect equals the largest of `k` estimates | The maximum of noisy estimates is biased upward | Rollout underdelivers; the effect does not replicate | Shrink the estimate, or confirm in a follow-up A/B |
| A/B/n read as a stronger test | "Same test, four options, same four weeks" | Per-arm sample falls by `k/2`; MDE rises by `sqrt(k/2)` | The test cannot see the effect that pays for itself | Compute the corrected MDE against break-even |
| Factorial analysed as separate A/B tests | Four cells reported as three independent comparisons | Interaction omitted; residual wrong | Main effects misestimated when the interaction is real | Fit one model with interaction terms |
| Fractional factorial with an aliased effect of interest | The alias table is absent from the design document | Resolution III used to answer an interaction question | The interaction cannot be separated from a main effect | State resolution and alias structure at design time |
| Cluster design analysed at row level | Degrees of freedom equal the row count, not the cluster count | Within-cluster correlation ignored | Interval too narrow by `sqrt(DEFF)` | Aggregate to cluster, or model the clustering |
| Cluster design too small to be worth running | The MDE exceeds the hoped-for effect, and nobody has said so | Between-cluster variance dominates | A null result is read as evidence of no effect | Compute the cluster MDE before launch |
| ICC borrowed with no provenance | "Industry standard ICC = 0.05" | ICC is category-, metric-, and geography-specific | Power wrong by a factor of two or more | Estimate from pre-period, or report a range |
| Shared inventory between arms | Both arms draw from one stock pool | Arms not independent; availability depends on the other arm's demand | False positives and false negatives, both potentially large | Assign at the pool, or hold the pool fixed |
| Price varied against an MFN clause | Legal is consulted after the design is approved | The design is not runnable | Quarter lost; test abandoned | Gate 3 review in week one |
| ITT replaced by per-protocol | The effect uses compliant stores only, with compliance in a footnote | Compliers differ from non-compliers | Effect overstated; rollout underdelivers | Report ITT first, per-protocol beside it |
| Bandit analysed with fixed-sample tests | A p-value reported from an adaptively allocated test | Adaptive allocation breaks the sampling distribution | Inflated false positives, invalid intervals | Use always-valid inference, inverse-probability weighting (IPW), or report no p |

Two of these survive review most easily. **A/B/n without correction** looks like every other number
in a deck; the tell is structural — count the arms, count the thresholds, and ask what the threshold
would have been at `0.05/3`. **Shared inventory** is the failure most likely to be missed: warehouse
stock, branch reserve stock, and carrier capacity are pools, and no randomisation restores
independence once two arms compete for one pool.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Units are independent across arms (SUTVA, the stable unit treatment value assumption) | Arm means must be unbiased | Shared inventory, fleet, staff, budget, or customers | Exposure log; shared-resource map; control-trend check |
| Assignment is independent of potential outcomes | Randomisation must be the only systematic difference | Assigning on a variable that also affects the outcome, or on a mutable field | Audit the assignment function; compare pre-period outcomes by arm |
| The allocation ratio holds in practice | SRM invalidates the test | Broken logging, eligibility rules differing by arm, bot traffic | Chi-square SRM check before reading results |
| The primary metric is defined identically in both arms | Like must be compared with like | Metric change mid-test; different measurement latency by arm | Freeze the metric dictionary; compare measurement coverage |
| No concurrent test on the same units | Two treatments on one unit cannot be separated | Overlapping pilots from different teams | Registry with unit-level collision checks |
| The ICC is stable over the test window | Cluster power depends on it | Seasonality, competitor entry, supply disruption | Estimate in pre- and test period; compare |
| Clusters are exchangeable within arm after blocking | Cluster means must be comparable | Voluntary participation; selection on the outcome | Balance table on pre-period covariates |
| Cluster sizes are roughly equal | `1 + (m−1)ICC` assumes it | A few very large stores or branches | Compute `sum(m_i²)/sum(m_i)`; correct if far above the mean |
| Factors do not interact when a main-effects design is used | A resolution III or IV design cannot separate some effects | Real interaction in a process setting | Run a 2x2 first, or use resolution V |
| Compliance is high enough for ITT to be informative | Low compliance dilutes toward zero | Field agents with different incentives; partial rollout | Exposure log; delivery rate beside ITT |
| The break-even effect is known and stable | MDE is judged against it, not against zero | Cost inputs changing during the test | Recompute break-even at readout; state the inputs |
| Segment findings are exploratory unless pre-specified and powered | Interaction tests need about four times the sample | Post-hoc slicing at low power | Count pre-specified interactions; label the rest |

**What no design fixes.** A treatment that cannot be delivered differentially, a decision that
changes after the result, and a metric that does not measure the behaviour. Each has an organisational
cause.

**Three hard limits.** The number of independent units: 240 branches support at most 240 independent
observations of a branch-level intervention, which is why the field developed paired designs,
switchbacks, and stepped rollouts. The commercial constraint: not negotiable by the analyst, usually
invisible in the data, and a design that ignores it is hypothetical, not conservative. And
break-even: a significant effect smaller than the cost of delivery is a loss.

## 12. Exercises

### Exercise 1 — Classify the design (recall, 20 minutes)

For each scenario, name the family, the assignment unit, the analysis obligation, and whether all
three gates pass. (a) A grocer tests a shelf label in 40 of 80 stores, analysing store-week revenue.
(b) A marketplace tests four commission rates, assigning sellers at random. (c) A 3PL, a third-party
logistics provider, tests two routing rules and two driver incentives across 60 depots. (d) A
distributor tests a confirmation email on 50% of orders. (e) A retailer tests a price in 12 of 30
categories, in all stores. (f) A manufacturer tests a scorecard with 9 buyers, analysing 12,000
purchase orders.

**Deliverable:** a six-row table with family, unit, obligation, and pass/fail per gate.
**Acceptance criteria:** each obligation names a specific analysis; each failure names the gate.

### Exercise 2 — Price the A/B/n (calculation, 30 minutes)

A test has 900,000 eligible sessions, baseline conversion 3.1%, and a break-even lift of 4.0%
relative. Compare a two-arm and a four-arm design, uncorrected and Bonferroni-corrected at α = 0.05,
80% power.

**Deliverable:** four MDEs in percentage points and relative terms, plus a verdict.
**Acceptance criteria:** per-arm sample stated for each; correction uses `α/(k−1)`; verdict compares
the corrected MDE with 4.0%.

### Exercise 3 — Design the factorial (design, 45 minutes)

A retailer wants to know whether a new delivery-fee structure and a new returns policy interact.
Both are set per market; there are 80 markets.

**Deliverable:** a 2x2 with cell sizes, main-effect and interaction MDEs, and what the design cannot
answer.
**Acceptance criteria:** the interaction MDE is shown to be twice the main-effect MDE; all four
cells are argued to be deliverable; one sentence states the decision the interaction would change.

### Exercise 4 — Read the alias structure (application, 35 minutes)

A `2^(4−1)` design is proposed with factors A, B, C, D and generator `D = ABC`. State the resolution
and list what each main effect is aliased with.

**Deliverable:** the resolution, the complete alias list, and a recommendation.
**Acceptance criteria:** the generator is used correctly; the resolution is justified from the
aliases, not asserted; the recommendation names which effects cannot be separated.

### Exercise 5 — Legality audit (critique, 40 minutes)

A subscription business proposes a customer-level test of a 10% discount across two countries. One
country has a published tariff; the largest account has a most-favoured-customer clause; the discount
is funded from a shared marketing budget reallocated monthly.

**Deliverable:** a gate-by-gate audit naming the binding constraint, plus three alternatives.
**Acceptance criteria:** all three gates assessed; each alternative names what it varies and what it
gives up; at least one is a non-experimental design.

### Exercise 6 — Cluster MDE (calculation, 35 minutes)

A test assigns 60 branches, 30 per arm. Each contributes 4,000 orders. Baseline on-time rate 91%,
ICC 0.04. Compute design effect, effective sample size, standard error of the difference, and MDE at
80% power and 5% two-sided.

**Deliverable:** the four quantities with arithmetic shown.
**Acceptance criteria:** variance of a cluster mean computed from the two-component formula; MDE
stated in percentage points and in relative terms.

### Exercise 7 — Write the six-field report (design, 30 minutes)

Take the result in section 6.2 and write the six-field report of Step 25 for a trading director.

**Deliverable:** six fields, one to three sentences each.
**Acceptance criteria:** MDE and break-even both appear; the limit states what the design could not
answer; the decision rule is stated before the result.

### Exercise 8 — Argue against a bandit (critique, 40 minutes)

A team proposes replacing a four-arm fixed-horizon test with a Thompson-sampling bandit, arguing it
"wastes less traffic on losers". Write the response.

**Deliverable:** a one-page memo with the strongest version of their argument, the statistical
objection, and a compromise.
**Acceptance criteria:** the objection names the estimator and critical value, not just "bias"; the
compromise specifies what is reported and to whom; one condition is stated under which the bandit is
right.

## 13. Self-check answers

### Exercise 1

| Scenario | Family | Assignment unit | Obligation | Gates |
|---|---|---|---|---|
| (a) Shelf label | Cluster | Store (40 of 80) | Variance component; cluster-level or mixed model | Pass, pass, pass |
| (b) Commission rates | A/B/n (4) | Seller | Multiplicity across 3 comparisons | Pass; pass unless sellers share buyers; pass unless rate is fixed |
| (c) Routes x incentives | Factorial 2x2 | Depot (60) | Interaction model; two error strata if either factor is depot-level | Pass; fails if a depot factor is crossed with a driver factor |
| (d) Confirmation email | A/B | Order | Single comparison, no correction | Pass, pass, pass |
| (e) Category price | A/B on 30 units | Category (12 of 30) | Clustering by category; substitution check | Pass; fails Gate 2 if categories share a fixture; likely fails Gate 3 |
| (f) Buyer scorecard | Cluster | Buyer (9) | Cluster-level summary or randomisation inference | Fails Gate 1: 9 units supports only very large effects |

### Exercise 2

```text
  p = 0.031, v = 0.030039, z = 2.8016
  two-arm, 450,000 per arm:  SE 0.03654 pp -> MDE 0.1024 pp = 3.30% relative
  four-arm, 225,000 per arm: SE 0.05168 pp -> MDE 0.1448 pp = 4.67% relative
  Bonferroni, alpha 0.01667, z 3.2356: MDE 0.1672 pp = 5.39% relative
```

The two-arm test detects 3.30%, below the 4.0% break-even. The four-arm test detects 4.67%
uncorrected and 5.39% corrected, both above it, so it cannot find a profitable change. Run the two-arm
test, drop to three arms, or extend the window.

### Exercise 3

Eighty markets, 40 per cell. Cell size 40 is a cluster, so the MDE depends on the market-level ICC:
at ICC 0.03 and 20,000 orders per market, `DEFF = 1 + 19,999 × 0.03 = 600.97` and effective `n` per
cell is about 1,331. Eighty markets cannot detect an interaction unless it is very large, and the
deliverable should say so with the arithmetic. Running the factorial at order level with the market
attribute held fixed is a different design; say which one you mean.

### Exercise 4

With `D = ABC` the defining relation is `I = ABCD`, so the design is resolution IV: main effects are
clear of each other and of two-factor interactions, but two-factor interactions are aliased with one
another.

```text
  main effects:  A-BCD   B-ACD   C-ABD   D-ABC
  2-factor:      AB-CD   AC-BD   AD-BC
```

Acceptable for screening four main effects; not acceptable if any two-factor interaction is the
question. To separate them, fold over — run the complementary fraction with signs reversed — giving
resolution V at the cost of all 16 runs.

### Exercise 5

Gate 1 passes: customers are numerous and independent enough. Gate 2 fails on the budget, because a
budget reallocated monthly couples treated and control spend. Gate 3 fails twice: the published
tariff forbids a customer-level discount on that product, and the MFN clause forbids offering the
largest account worse terms than the test arm. Alternatives: vary the communication rather than the
price; run in the country without the tariff, excluding the MFN account, accepting no generalisation
across the tariff boundary; or run a matched before-and-after, weaker but legal, reported as such.

### Exercise 6

```text
  p = 0.91, v = 0.91 x 0.09 = 0.0819;  ICC = 0.04;  m = 4,000;  k = 30
  var(branch mean) = 0.04 x 0.0819 + 0.96 x 0.0819 / 4,000 = 0.00329566
  SE(diff) = sqrt(2 x 0.00329566 / 30) = 0.014822 = 1.4822 pp
  MDE      = 2.8016 x 0.014822 = 4.1526 pp = 4.56% relative
  DEFF = 160.96;  effective n = 240,000 / 160.96 = 1,491
```

The design detects a 4.15 percentage point change in on-time rate. Whether that is adequate depends on
the value of a point. The common wrong answer reports 0.073 pp, the individual-level figure, wrong by
`sqrt(160.96) = 12.7`.

### Exercise 7

```text
  DESIGN      Session-level two-arm A/B, 50/50, 600,000 sessions per arm, 4 weeks.
  ALLOCATION  Planned 50/50; observed 600,412 and 599,588; SRM p = 0.46.
  ANALYSIS    Session-level comparison of conversion; two-sample z, 1,199,998 df.
  EFFECT      +0.118 pp conversion (95% CI +0.015 to +0.221), +2.81% relative.
  POWER       MDE at 80% power was 0.103 pp; break-even was 0.128 pp.
  LIMIT       Cannot detect a change below 0.103 pp, and says nothing about
              branch-level reserve-stock effects, which are a different design.
```

A weak answer omits the break-even, reports "+2.81%" without the percentage-point figure, or states
the limit as "more testing is needed".

### Exercise 8

The strongest version of their case: a fixed four-arm test sends 25% of sessions to each arm for the
whole window, including arms clearly worse after a week; a bandit reallocates and recovers revenue
while testing.

The objection: adaptive allocation biases arm means and breaks the sampling distribution the
fixed-sample test assumes. In the published simulation, naive analysis of Thompson-sampled two-arm
data produced about 13% false positives against a nominal 5% and power near 56% against a nominal
80%; restoring validity with inverse-probability weighting or always-valid inference cost power down
to roughly 17% in the illustrated case.

The compromise: run the bandit for the operational decision and a small fixed-horizon A/B or
fixed-allocation holdout for the evidential claim. Report the bandit as a policy result and the A/B
as an effect estimate. The bandit alone is right when nobody will ever need an unbiased estimate.

## 14. Interview questions

**1. "When would you run an A/B/n rather than a sequence of A/B tests?"**
Strong: when one decision picks among `k` alternatives, the corrected MDE stays below break-even, and
sequential tests would outlast it. States the `sqrt(k/2)` precision cost and the multiplicity cost.
Weak: "when you have more than two ideas" — a reason to want one, not to run one.

**2. "Your four-arm test produced a winner at p = 0.03. What do you report?"**
Strong: the corrected threshold `0.05/3 = 0.0167`; 0.03 does not clear it; an interval for the
selected arm; and the upward bias from selecting the maximum. Proposes a confirmatory two-arm test.
Weak: reports the winner at 95% confidence, the error the question exists to detect.

**3. "Why does an interaction cost four times the sample of a main effect?"**
Strong: it is a difference of differences over four cells, so its variance is four times a main
effect's, its standard error twice as large, and the MDE doubles.
Weak: "interactions need more data", without the factor.

**4. "What is the resolution of a fractional factorial, and why do you care?"**
Strong: III separates main effects from each other but aliases them with two-factor interactions; IV
separates main effects from those but aliases interactions with each other; V separates both. The
alias structure is knowable before launch and unfixable after.
Weak: calls resolution "how good the design is", which says nothing about aliasing.

**5. "You have 240 branches and an ICC of 0.02. Can you test a 3% effect?"**
Strong: computes `DEFF = 1 + (m−1)ICC`, converts to a cluster-level MDE, concludes no — about 24%
relative at 5,000 sessions per branch — and proposes a stepped rollout or switchback.
Weak: says yes, because 1.2 million sessions is a large sample.

**6. "What is the difference between the design effect and the effective sample size?"**
Strong: the design effect is a variance ratio; the effective sample is the row count divided by it,
converging to `clusters / ICC` as cluster size grows. The ceiling is the planning number.
Weak: uses the two terms interchangeably.

**7. "A team wants to test a price change on half their customers. What do you ask first?"**
Strong: whether the price is fixed by tariff, contract, or an MFN clause, and whether the arms share
an inventory or budget pool. Legal and operational before statistical.
Weak: asks about sample size, which is the third question.

**8. "What is wrong with analysing bandit data using a t-test?"**
Strong: adaptive allocation biases arm means and breaks the sampling distribution; published
simulations show false positives well above nominal and power well below; valid alternatives are
always-valid inference, inverse-probability weighting, or a fixed-allocation holdout.
Weak: "bandits are biased", without naming the estimator or consequence.

**9. "When is a factorial better than two concurrent A/B tests?"**
Strong: when the interaction is decision-relevant — a rollout turns both factors on, so the business
needs cell 11 — or when the factors are process settings likely to interact. For main effects only,
two concurrent A/B tests give the same precision with less operational risk.
Weak: "factorials are always more efficient", true only for effects they can separate.

**10. "Your pilot found a significant effect in the North and none in the South. What do you do?"**
Strong: treats it as exploratory, tests the region-by-treatment interaction rather than two subgroup
tests, notes that an interaction of the same size needs about four times the sample, and proposes a
pre-specified regional test or no regional claim.
Weak: proposes a regional rollout, which is the decision the finding cannot support.

## 15. Key terms

- **A/B test** — two-arm randomised comparison of a control and one treatment.
- **A/B/n test** — one control against two or more treatments; carries a multiplicity obligation.
- **Arm / allocation ratio** — one version in a test, and its planned share of units.
- **Assignment key and salt** — the field and constant whose hash decides the arm.
- **Stratum / blocking** — pre-treatment partition within which randomisation is balanced.
- **Factor / level / cell** — an attribute varied, its values, and one combination of values.
- **Factorial design** — factors varied together so every combination occurs.
- **Main effect** — average change from one factor, averaged over the others.
- **Interaction effect** — the extent to which one factor's effect depends on another's level.
- **Aliasing** — confounding of one effect with another in a fractional design.
- **Resolution** — which orders of effect a fractional design can separate; III to V.
- **Multi-armed bandit (MAB)** — adaptive allocation shifting traffic toward better arms.
- **Fixed-horizon test** — sample size and stopping point set in advance.
- **Sequential test / alpha spending** — pre-specified rules for valid early stopping.
- **Family-wise error rate (FWER)** — probability of at least one false positive in a family.
- **False discovery rate (FDR)** — expected proportion of false positives among claims.
- **Bonferroni / Holm / Benjamini–Hochberg** — three multiplicity procedures, three guarantees.
- **Guardrail metric / OEC** — what must not degrade, and what the decision is made on.
- **Sample ratio mismatch (SRM)** — observed allocation inconsistent with the planned split.
- **Cluster randomisation** — randomising groups rather than the units within them.
- **Intra-cluster correlation (ICC)** — share of variance that lies between clusters.
- **Design effect (DEFF)** — variance inflation from clustering, `1 + (m−1)ICC`.
- **Effective sample size** — row count divided by the design effect; ceiling `clusters / ICC`.
- **Cluster-level analysis / cluster-robust SE** — two routes to valid cluster inference.
- **ITT and per-protocol** — effect of assignment versus effect of receipt.
- **Break-even effect** — the smallest true effect that pays for the intervention's cost.

## 16. Sources

**Grading convention.** A peer-reviewed or audited; B regulatory, official, or registered; C company
case, investor claim, preprint, or conference paper; D vendor or consultancy marketing; E press or
anecdote. Preprints and conference papers are graded C regardless of quality.

**Peer-reviewed methodology**

- Fisher, R. A. (1926). "The Arrangement of Field Experiments." *Journal of the Ministry of
  Agriculture of Great Britain*, 33. Grade A. Page range not verified for this file.
- Robbins, H. (1952). "Some Aspects of the Sequential Design of Experiments." *Bulletin of the
  American Mathematical Society*, 58. Grade A. Page range not verified for this file.
- Thompson, W. R. (1933). "On the Likelihood that One Unknown Probability Exceeds Another..."
  *Biometrika*, 25. Grade A. Page range not verified for this file.
- Box, G. E. P., Hunter, J. S., and Hunter, W. G. (2005). *Statistics for Experimenters*, 2nd ed.
  Wiley. Grade A.
- Montgomery, D. C. *Design and Analysis of Experiments*; *Statistical Quality Control*, 7th ed.,
  ch. 13. Grade A.
- Kohavi, R., Longbotham, R., Sommerfield, D., and Henne, R. M. (2008). "Controlled experiments on
the web: survey and practical guide." *Data Mining and Knowledge Discovery*.
doi:10.1007/s10618-008-0114-1. Grade A.
- Larsen, K., et al. (2023). "Statistical Challenges in Online Controlled Experiments." *The
  American Statistician*. doi:10.1080/00031305.2023.2257237. Grade A.
- Brookes, S. T., et al. (2004). "Subgroup analyses in randomized trials: risks of subgroup-specific
analyses; power and sample size for the interaction test." *Journal of Clinical Epidemiology*.
PubMed 15066682. Grade A.
- Wang, R., Piantadosi, S., Le-Rademacher, J., and Mandrekar, S. "Statistical Considerations for
  Subgroup Analyses." *Journal of Thoracic Oncology*. PMC7920926. Grade A.
- Baumann, et al. (2025). "A tutorial on ... treatment effect heterogeneity in cluster randomized
  trials." *International Journal of Epidemiology*. doi:10.1093/ije/dyag069. Grade A.
- Torgerson, D. J. (2001). "Contamination in trials: is cluster randomisation the answer?" *BMJ*.
  PMC1119583. Grade A.
- Ibanez, M. R., and Staats, B. R. (2019). "Field Experiments in Operations Management." *INFORMS
  TutORials in Operations Research*. doi:10.1287/educ.2019.0197. Grade A.
- Bohle, D., Marynych, A., and Meiners, M. (2021). "A fundamental problem of hypothesis testing with
finite inventory in e-commerce." *Applied Stochastic Models in Business and Industry*.
doi:10.1002/asmb.2574. Grade A.
- Wang, Z., Mersereau, A. J., and Chen, L. (2017). "Optimal Merchandise Testing with Limited
  Inventory." *Operations Research*. doi:10.1287/opre.2017.1596. Grade A.

**Official, regulatory, and registered**

- Robinson–Patman Act, 15 U.S.C. §13. Grade B. Scope and defences require legal review.
- Regulation (EU) 2016/679 (GDPR); TFEU Article 101. Grade B.
- CONSORT 2010 extension to cluster randomised trials (Campbell et al., *BMJ*, 2012). Grade A.
- ICH E9 and E9(R1); ISO 3534-1:2016; ISO 8601-1:2019; GS1 General Specifications. Grade B.
- World Bank / ISRCTN86533387, essential-drugs supply chain cluster trial, Zambia. Grade B for the
  registered design.

**Company, technical, and unrefereed**

- Williams, G., Gentzkow, M., bin Zafar, S., and Choudhury, S. (2021). arXiv:2103.12198. Grade C.
- Burtini, G., Loeppky, J., and Lawrence, R. (2015). arXiv:1510.00757. Grade C.
- Xiang, et al. (2022). KDD. doi:10.1145/3534678.3539144. Grade C.
- Yang, F., Ramdas, A., Jamieson, K. G., and Wainwright, M. J. (2017). NeurIPS; arXiv:1706.05378.
  Grade C.
- Costa, et al. "Generalized Synthetic Control for TestOps at ABI." Company-authored. Grade C.

**Note on citations.** Findings are reported as the cited works state them. The worked example is
constructed for teaching and its arithmetic was checked against the code in section 7. Where a
bibliographic detail could not be verified — notably page ranges for the 1926, 1933, and 1952
references — the entry says so.

## 17. Links

- Module map and topic sequence: [`_index.md`](_index.md)
- Previous topic, which chose the unit this topic randomises: [`02_randomisation-units.md`](02_randomisation-units.md)
- The estimand that precedes every design choice: [`01_claim-hypothesis-design.md`](01_claim-hypothesis-design.md)
- Market-level units when even a cluster is too fine: [`04_geo-experiments.md`](04_geo-experiments.md)
- Time as the unit when the cluster count is too small: [`05_switchback-pilots.md`](05_switchback-pilots.md)
- Turning the MDE arithmetic into a full power calculation: [`06_power-mde.md`](06_power-mde.md)
- Interference treated formally, including shared inventory: [`07_interference-spillover.md`](07_interference-spillover.md)
- Why peeking and adaptive allocation break fixed-sample tests: [`08_sequential-testing.md`](08_sequential-testing.md)
- Hypothesis tests and the variance an F statistic divides:
  [`../04_statistics_causal_inference/03_hypothesis-tests.md`](../04_statistics_causal_inference/03_hypothesis-tests.md)
- Quasi-experimental designs when no assignment is possible:
  [`../04_statistics_causal_inference/07_experiments-quasi.md`](../04_statistics_causal_inference/07_experiments-quasi.md)
- Estimands and question types, the vocabulary this topic assumes:
  [`../04_statistics_causal_inference/01_estimands-question-types.md`](../04_statistics_causal_inference/01_estimands-question-types.md)
- The three-flow model that tells you which resources are shared:
  [`../01_supply_chain_foundations/01_three-flows-one-network.md`](../01_supply_chain_foundations/01_three-flows-one-network.md)
- Module survey file: [`../../../course/05_experimentation_pilots_evidence.md`](../../../course/05_experimentation_pilots_evidence.md)
- Promotion and cannibalisation measurement, a common source of multi-arm tests:
  [`../../../course/16_pricing_promotions_revenue_management.md`](../../../course/16_pricing_promotions_revenue_management.md)
- Return to the part map: [`../_index.md`](../_index.md)

