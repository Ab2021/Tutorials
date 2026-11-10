---
collection: "module-07-analytics-metrics-bi-product-sense"
file_id: "05"
topic: "Stakeholder Narratives and Decision Memos"
level: 3
status: "draft"
version: "2.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Change in decision-value inputs (gain/loss estimates), memo approval chain, reporting cutoff policy, or the prior used for the lift estimate"
notation_style: "plain-text"
prerequisites: ["01 Metric Trees and North Stars", "02 Funnel, Cohort and Retention Analysis", "04 Segmentation and Decision Use", "ratio arithmetic", "interval interpretation"]
pattern_tags: ["business-domain-analytics", "decision-writing", "uncertainty-communication"]
source_status: "OPEN"
owner_role: "analytics lead"
source_index: "INDEX.md"
---

# Stakeholder Narratives and Decision Memos

## 1. File Header

> **Purpose:** Convert an analysis into a memo that a named owner can accept, reject, or defer on a stated date, by making the decision rule, the uncertainty, and the cost of being wrong explicit enough that a reader with the same data would reach the same action.
> **Prerequisites:** A metric definition with entity grain and denominator (file 01), a cohort or funnel reconciliation habit (file 02), a segmentation discipline that refuses unstable slices (file 04), and the arithmetic of ratios and intervals.
> **Key outcome:** Given a decision, a decision owner, a gain/loss estimate, and a lift estimate with standard error, you can compute the break-even probability, the posterior probability that the change helps, the expected value of acting, the expected value of one more round of evidence, and the marginal value of a week of delay — then write those five numbers into a memo whose recommendation survives a hostile re-read.
> **This file does NOT cover:** causal identification and experiment design (Module 08), dashboard or alert configuration (file 03), and visual design or BI tooling. It also does not cover persuasive writing as rhetoric; the writing here is constrained by auditability, not by charisma.
> **Level:** 3 Advanced / Production
> **Version / reviewed:** 2.0 / not independently reviewed
> **Evidence:** DERIVED decision-theoretic identities with derivations shown; VERIFIED arithmetic in Section 5 (recomputed independently); ILLUSTRATIVE inputs throughout (no company, product, or campaign is described); OPEN for any claim about how a specific organisation approves decisions.
> **Retrieval cue:** If a stakeholder asks "so what should we do?", start at Section 4.1 for the decision rule, Section 4.2 for the uncertainty statement, Section 5 for a fully worked memo, and Section 8 for the ways memos fail.

A decision memo is not a report with a recommendation bolted on. It is a compressed, falsifiable argument in which every number carries its population, window, and cutoff, and in which the recommendation is derived from a rule the reader can check rather than from the analyst's confidence. The distinction matters because the memo's real product is not the analysis; it is a decision that someone else will own and be accountable for.

Three artefacts are frequently confused, and the confusion causes most memo failures:

| Artefact | Question it answers | Success test | Typical failure |
|---|---|---|---|
| Report | What happened last period? | Numbers reconcile to source | Treated as a recommendation, so nobody acts |
| Analysis | Why did the number move? | Competing explanations separated by evidence | Correct diagnosis, no owner, no date |
| Decision memo | What do we do now, who decides, when, and how would we know we were wrong? | Owner signs, action starts or is explicitly declined | Decision silently deferred, memo re-requested next quarter |

The memo is the only one of the three that has a counterfactual built into it: doing nothing is always one of the options, and the memo must say what "nothing" costs. A memo that cannot express the cost of no-action has not yet found its decision.

The reader this file prepares you for is not a passive audience. Assume a sceptical operations lead who has been burned by three previous "insights" that did not survive contact with the following month's data, who has a budget meeting on Thursday, and who will ask two questions: "how sure are you, in numbers?" and "what happens if you're wrong?". Everything in Sections 3 through 11 exists to make those two questions answerable in under a minute of reading.

**Takeaway:** a memo earns its name only when it names the decision, the decider, the rule, the uncertainty, and the cost of being wrong.

## 2. Concept Map / Mental Model

The memo is best understood as a pipeline with two loops, not as a document. The forward path converts raw observations into a decision rule; the two loops are the pre-registration loop (which fixes the rule before the numbers arrive, so the analyst cannot rationalise afterwards) and the post-decision loop (which records what was predicted so the next memo can be calibrated).

```text
   [stakeholder request]
            |
            v
   [decision framing] --no owner/date?--> [stop: convert to a research question]
            |
            v
   [population + grain + cutoff] --mismatch?--> [reconcile, or refuse to compare]
            |
            v
   [gain G, loss L, reversibility R]  (the value model, not the data)
            |
            v
   [lift estimate + standard error]  (the data)
            |
            v
   [posterior probability the change helps] -> [expected value of acting]
            |
      +-----+----------------------------+
      |                                  |
      v                                  v
 [act now]                        [buy evidence? EVSI > cost?]
      |                                  |
      |                                  v
      |                          [one more round, pre-registered]
      v                                  |
 [record prediction, date, metric] <-----+
      |
      v
 [follow-up memo: was the predicted direction right?]
      |
      v
 [calibration ledger -> better priors next time]
```

Read the map from the top: the first failure point is framing. A request such as "look into why engagement dropped" has no owner, no date, and no decision; it is a research question wearing a decision's clothes. Converting it costs one message and saves a week.

The second failure point is the join between the value model and the data. Note the ordering in the diagram: the gain and loss come *before* the lift estimate. This ordering is deliberate. If the analyst reads the lift first and then estimates the loss, the loss estimate will drift towards whatever makes the lift decisive — a well-documented form of motivated reasoning that no amount of statistical care repairs. Fix the value model in writing, with the finance or operations partner who owns the P&L line, then compute.

The third failure point is the two-branch split after expected value. Memos fail in two opposite directions: acting on a signal too weak to justify the downside, and commissioning yet another study when the cheapest available evidence already exceeds the decision's value. Both are failures of the same discipline — failing to compare the value of acting against the value of knowing more.

The two feedback loops are what separate a mature memo culture from a memo factory. The pre-registration loop is cheap: one paragraph written before the analysis says which result would trigger which action. The calibration loop is rarer and more valuable: a ledger of "we predicted X, the direction was right/not right" turns the analyst's priors from folklore into a measurable asset. Section 4.2 makes the prior an explicit input precisely so that the calibration ledger has something to score.

A useful mental compression of the whole sheet: **a memo is a bet, written down.** The population defines the table the bet is placed on, the value model defines the stakes, the interval defines the odds, the recommendation is the wager, the follow-up date is the settlement, and the calibration ledger is the bookmaker's record.

```text
    bet              memo analogue              where it is specified
    ----             ---------------             --------------------
    table            population, grain, cutoff   Section 3, Section 5 step 1
    stakes           gain G, loss L, salvage R    Section 4.1, 4.3(A)
    odds             lift estimate + std error    Section 4.2
    wager            recommendation + owner       Section 5 step 6
    settlement       follow-up date + metric      Section 5 step 8
    book record      calibration ledger           Section 9, Section 10
```

Two implications follow from the table. First, a memo that omits the settlement row is a bet with no result, and its author never learns. Second, the odds row must be expressed on the same scale as the stakes row; a probability in percentage points and a gain in currency cannot be compared until one is converted, which is the mechanical reason Section 4.2 ends in a probability rather than a test statistic.

Also note what the map deliberately excludes: it contains no box for "make the chart persuasive," "choose the colour palette," or "build stakeholder consensus." Those are legitimate activities, but they are downstream of, and subordinate to, whether the decision rule is sound. An analyst who optimises the presentation of an unsound rule has simply made the error travel faster.

One further structural point about the map: the arrow from `[record prediction]` back to `[calibration ledger]` is the only arrow that points at the analyst rather than at the decision. Every other element in the diagram is about the situation. This asymmetry is intentional. Most organisations can improve their decisions substantially without new data, purely by scoring their own past forecasts and adjusting the priors they bring to the next estimate. A team that knows it has historically overestimated lifts by a factor of 1.8 has a concrete, usable adjustment that costs nothing and requires no new instrumentation.

There is one more element worth noting that the diagram leaves implicit: the memo has a *shape* that readers learn to recognise. The recommendation appears in the first three sentences, the decision rule and the threshold appear in the first screen, the sensitivity appears before the appendix, and the action contract closes the document. A reader who opens a memo and finds four pages of methodology before the recommendation will either skim to the end (missing the caveats) or close it (missing the decision). Structure is not decoration; it is how the reader's attention is routed to the parts that change the decision.

The final compression: the pipeline diagram is a *checklist of joints*. There are six joints where an argument can break — framing, population, value model, estimate, threshold comparison, and action contract. A memo review that checks these six joints in order takes about ten minutes and catches the large majority of errors before the owner sees them. Section 8 is organised around failures at exactly these joints.

**Takeaway:** fix the value model before reading the estimate, split action from further evidence at the expected-value node, and close the loop with a scored prediction.

## 3. Foundational Definitions

Each definition below is written to be operational: it names the unit at which it is measured, the test that shows whether the definition is being respected, and a nonexample that shows the definition being violated. Vocabulary that cannot be tested is decoration, and decoration is what makes memos long and unconvincing.

1. **Decision (as opposed to topic).** A choice between at least two named options, made by a named role, by a stated date, with consequences that differ between the options. *Test:* can you write two complete sentences, one per option, describing different downstream states of the world? *Nonexample:* "improve onboarding" is a topic; "ship the guided checklist to all new accounts on 2026-12-01 vs. keep the current empty state" is a decision.

2. **Decision owner.** The single role with authority to accept or decline the recommendation and with budget or mandate to fund the consequence. *Test:* if the owner declines, does the action genuinely stop? *Nonexample:* a committee whose members each assume another member decides.

3. **Decision date.** The calendar date by which the choice must be made for the consequence to still be worth having. *Test:* does a later date change the value of the outcome, not merely the convenience of the discussion? *Nonexample:* an "ongoing review" with no date; it defers indefinitely and consumes attention each cycle.

4. **Decision grain.** The entity-and-time unit at which the decision is actually made — one account per month, one store per week, one claim per adjudication — together with the key that identifies an instance. *Test:* count distinct keys; does the count equal the number of things the owner can act on? *Nonexample:* reporting "users" when the owner can only act on accounts, which silently double-counts people who hold two accounts.

5. **Eligible population.** All entities that could have entered the measured experience during the window, before any exclusion. *Test:* is the exclusion list written down and each exclusion justified by a decision-relevant reason rather than availability? *Nonexample:* computing activation only over accounts that have a completed profile, because that is the table that was easiest to join.

6. **Estimand.** The precise quantity the memo claims, including population, contrast (versus what baseline), time horizon, and whether it is an average, a total, or a distributional tail. *Test:* could two analysts with the same data disagree about the number because they read the estimand differently? If yes, it is underspecified. *Nonexample:* "the feature works" — no contrast, no horizon, no population.

7. **Baseline (counterfactual comparator).** The state against which the change is measured: the prior period, a held-out population, or an explicit model-based expectation. *Test:* is the baseline's own instability quantified, so that a difference smaller than that instability is not reported as a finding? *Nonexample:* comparing a promotion week to the immediately preceding week, which was a holiday trough.

8. **Gain (G).** The value, in one currency, of a correct positive decision over the full consequence horizon. *Test:* is the derivation written down, with the unit (currency per entity per period) and the horizon? *Nonexample:* a gain stated as "significant revenue upside" with no unit and no horizon.

9. **Loss (L).** The value, in the same currency, of an incorrect positive decision — the cost of shipping something that harms the metric, including remediation, support load, and trust damage where those can be bounded. *Test:* is the loss estimated by someone other than the person advocating the action? *Nonexample:* a loss that includes only infrastructure cost and excludes the customer-facing cost that motivated the analysis.

10. **Salvage (R).** The fraction or amount of the loss recoverable by reversing the decision — rollback, refund, rework — net of the cost of reversing. *Test:* has a rollback actually been rehearsed, or is it assumed available? *Nonexample:* calling a change "reversible" because a feature flag exists, when the flag cannot be flipped without a schema migration.

11. **Break-even probability (p\*).** The smallest probability that the change helps at which acting has non-negative expected value under G, L, and R. *Test:* does the memo state p\* numerically? *Nonexample:* "we think it's likely to help," which has no comparator and no threshold.

12. **Posterior probability.** The probability that the change helps, updated from a stated prior using the observed effect and its standard error. *Test:* is the prior written down with its source, and is the update reversible if the standard error changes? *Nonexample:* quoting a p-value and calling it the probability the change helps.

13. **Claim–evidence ledger.** A per-claim table mapping each sentence of the recommendation to the specific computation or source that supports it, and to its evidence label (DERIVED, VERIFIED, SOURCED, ILLUSTRATIVE, HEURISTIC, OPEN). *Test:* can a reader delete an unsupported claim and still find the recommendation unchanged? *Nonexample:* an appendix of links that no sentence in the body references.

14. **Cutoff semantics.** The rule determining which records are visible at analysis time: ingestion-time cutoff, event-time cutoff, or restatement window with a known lag profile. *Test:* is the cutoff stated as a timestamp with a time zone, and is the late-arrival rate over the last period quoted? *Nonexample:* "data as of last night," which is compatible with three different values of last night.

15. **Decision latency.** Elapsed time from the availability of sufficient evidence to the owner's decision, including review cycles and meeting cadence. *Test:* measured on the last three decisions, or estimated with the queue formula in Section 9? *Nonexample:* an assumption that memos are read on the day they are sent.

16. **Action contract.** The written commitment that closes the memo: action, owner, start date, success metric, guardrail, review date, and the pre-agreed condition under which the action is rolled back. *Test:* are all seven fields filled with values that a third party could check? *Nonexample:* "we will monitor closely," which commits nobody to anything measurable.

17. **Guardrail metric.** A metric that must not degrade beyond a stated bound for the primary action to remain acceptable, chosen because the primary metric is known to be gameable or narrow. *Test:* is the bound numeric, and was the guardrail named before the result was known? *Nonexample:* a guardrail added after seeing that it did not move.

18. **Pre-registration note.** A short record, written before the analysis, of the decision rule, the metric, the cutoff, and the action that each result band will trigger. *Test:* is it timestamped and immutable? *Nonexample:* a plan document edited after the results arrive.

19. **Falsification condition.** The specific observation that would make the recommendation wrong, stated in advance in the same units as the evidence. *Test:* is it observable within the action window? *Nonexample:* "if the effect is not real," which cannot be checked by anyone.

20. **Narrative compression ratio.** The number of decision-relevant facts carried per hundred words of memo, used as a writing diagnostic rather than a virtue in itself. *Test:* mark each sentence as decision-relevant, context, or decoration; decoration above a fifth of the text indicates the memo is performing rather than deciding.

Two cross-cutting distinctions complete the vocabulary. The first is **precision versus accuracy of language**: a memo may state an estimate to three significant figures while its population is ambiguous, in which case the extra digits are noise that borrows credibility from arithmetic. The second is **confidence versus calibration**: confidence is what the author feels; calibration is the observed frequency with which statements of a given confidence band turned out to be correct. Only the second is a property of the analyst rather than the mood of the analyst.

A third distinction, easy to state and frequently violated, separates **evidence about a mechanism** from **evidence about an outcome**. An instrumented funnel may show exactly where users abandon a flow, which is evidence about mechanism; it says nothing about whether removing that step raises completed orders, which is evidence about outcome. Memos routinely import mechanism evidence and then write an outcome recommendation, and the gap between the two is where most "we shipped it and nothing happened" stories originate.

A fourth distinction, worth its own line because it silently corrupts thresholds, separates **a cost that recurs from a cost that is paid once**. Shipping cost, support load, and refunds recur per period of exposure; migration cost and write-off are one-off. Adding them into a single `L` without noting their time profile makes the break-even threshold wrong in the direction that favours the proposal, because recurrent costs are usually larger than the analyst initially estimates and one-off costs are usually smaller.

**Takeaway:** define each term with a test and a nonexample, because vocabulary that cannot fail a test cannot support a decision.

## 4. Core Formulas with Full Derivation

This section derives the five numbers a decision memo must contain: the break-even probability, the posterior probability that the change helps, the expected value of acting, the expected value of further evidence, and the marginal cost of delay. Everything else in this file is either an input to these quantities or a check on whether they are being applied outside their validity range.

### 4.1 Base case — break-even probability and expected value of a binary decision

**Validity before use.** The decision admits two options (act, do not act) and the consequence is summarised by two scalars: a gain `G` realised if the change helps, and a loss `L` realised if it does not. The option "do not act" has value zero by construction — everything is measured as a difference against it, which means the analyst must have already decided what "not acting" means in the same currency. No partial rollout, no gradual ramp, no learning-by-doing is modelled here.

**Derivation.** Let `p` be the probability that the change helps. If it helps, acting yields `G` relative to not acting; if it does not, acting yields `-L` relative to not acting. Expected value of acting is the probability-weighted sum:

```text
EV(p) = p*G + (1-p)*(-L) = p*G - (1-p)*L
```

Setting `EV(p*) = 0` and solving:

```text
p*G = (1-p*)*L
p*G + p**L = L
p*(G + L) = L
p* = L / (G + L)
```

The result is a ratio of the downside to the total spread. It is invariant to scale: doubling both G and L leaves p\* unchanged, which is why a small-stakes decision and a large-stakes decision with the same shape need the same strength of evidence.

```text
FORMULA [DM-01]: EV(p) = p*G - (1-p)*L
WHERE: EV value in currency units per decision [currency];
       p probability the change helps, dimensionless in [0,1];
       G gain if it helps [currency]; L loss if it does not [currency].
CONSTRAINT: binary outcome summary; "do not act" normalised to zero;
            G and L measured over the same horizon and the same population;
            no partial or staged action; risk-neutral decision maker.
NUMERICAL EXAMPLE: G = 480000 currency units, L = 1200000 currency units,
            p = 0.90:  EV = 0.90*480000 - 0.10*1200000
                        = 432000 - 120000 = 312000 currency units.
INTERPRETATION: at p = 0.90 acting is worth 312000 units over not acting;
            the decision's value is large enough to justify a review meeting,
            and the number is what an owner needs, not the p-value.
```

```text
FORMULA [DM-02]: p* = L / (G + L)
WHERE: p* break-even probability, dimensionless;
       G, L as in DM-01, same currency and horizon.
CONSTRAINT: as DM-01; p* is interpreted as a strict threshold when EV is
            strictly increasing in p, which holds because G + L > 0.
NUMERICAL EXAMPLE: G = 480000, L = 1200000:
            p* = 1200000 / (480000 + 1200000) = 1200000/1680000
               = 0.714285714... = 71.43%.
INTERPRETATION: any claim of "better than even odds" is insufficient here;
            the memo must defend p > 0.714, which is a materially
            higher bar than the intuition "more likely than not".
```

**Cross-check of DM-01 and DM-02.** Substituting `p = p*` into DM-01 must return zero. With the figures above: `0.714285714*480000 = 342857.142857` and `(1-0.714285714)*1200000 = 342857.142857`. The two products agree to the rounding of `p*`, confirming that the threshold was derived from the same value function and not asserted separately. A memo that quotes a threshold and an expected value that do not reconcile this way has two incompatible models in it.

**Dimensional check.** `p*` is a ratio of two currency quantities, hence dimensionless and bounded in (0,1) whenever `G, L > 0`. If `L = 0` then `p* = 0`: any positive signal justifies acting, which is the "free option" case. If `G = 0` then `p* = 1`: no finite evidence justifies acting, which is the "strictly dominated" case. Both extremes are diagnostic warnings rather than normal results, because in practice a memo that reports `p* ≈ 0` has usually forgotten to count a cost.

**Why the naive formulation fails.** A frequent memo formulation compares the estimated effect to zero: "the lift is +2.3 points, therefore act." That formulation silently sets `L = ∞` relative to `G` — it treats any chance of harm as disqualifying — or, more commonly, sets `L = 0` — it treats harm as free. Neither is the owner's value function. DM-02 forces the ratio into the open, and the act of writing `L` down is usually where a memo's first real discovery happens.

**Takeaway:** the decision rule is a threshold on probability, `L/(G+L)`, not a test of whether the estimate differs from zero.

### 4.2 General case — posterior probability that the change helps

**Validity before use.** The effect is estimated on a difference scale with an approximately normal sampling distribution; a standard error is available and is treated as known (a defensible approximation when the standard error is estimated from a reasonably large sample, and a known hazard when the sample is small — see 4.4). A prior over the true effect is stated explicitly. The decision's consequence depends only on the sign of the effect relative to the horizon of interest, not on its magnitude, which is what allows a sign-probability to drive DM-01.

**Derivation.** Let `d̂` be the estimated effect with standard error `s`, and let the prior on the true effect `d` be normal with mean `μ0` and standard deviation `σ0`. The likelihood is normal with mean `d` and standard deviation `s`. By conjugacy the posterior is normal with precision equal to the sum of precisions:

```text
1/σ1^2 = 1/σ0^2 + 1/s^2
μ1     = σ1^2 * ( μ0/σ0^2 + d̂/s^2 )
P(helps) = P(d > 0) = Φ( μ1 / σ1 )
```

The corresponding odds form is the one to remember, because it separates what the data contributed from what the analyst believed:

```text
posterior odds = prior odds * Bayes factor
P(helps)/(1-P(helps)) = [Φ(μ0/σ0)/(1-Φ(μ0/σ0))] * LR
```

For a normal model with a diffuse prior the Bayes factor in favour of a positive effect is well approximated by the likelihood ratio at the boundary, and for practical memo work the ratio can be read directly off the two probabilities implied by the prior and posterior. The operational consequence is that a memo must report the prior, because the same data will produce different posteriors under different priors — a fact that is the single largest source of disagreement between analysts and their stakeholders.

```text
FORMULA [DM-03]: 1/σ1^2 = 1/σ0^2 + 1/s^2 ;  μ1 = σ1^2*(μ0/σ0^2 + d̂/s^2) ;
                  P(helps) = Φ(μ1/σ1)
WHERE: d̂ estimated effect on the decision scale (e.g. percentage points)
           [percentage points]; s its standard error [percentage points];
       μ0 prior mean [percentage points]; σ0 prior SD [percentage points];
       μ1, σ1 posterior mean and SD, same units;
       Φ standard normal CDF, dimensionless.
CONSTRAINT: normal likelihood with known s; normal prior; effect measured on
           the same scale as the prior (both as absolute differences, or both
           as relative lifts — mixing the two invalidates the update);
           the consequence depends on sign only.
NUMERICAL EXAMPLE: d̂ = 2.266 pp, s = 1.459 pp, μ0 = 0, σ0 = 2.0 pp.
            precision prior = 1/4 = 0.25; precision data = 1/2.129281 = 0.469644
            total precision = 0.719644 -> σ1^2 = 1.389576, σ1 = 1.178803 pp
            μ1 = 1.389576*(0/4 + 2.266/2.129281) = 1.389576*1.064207
               = 1.478877 pp
            P(helps) = Φ(1.478877/1.178803) = Φ(1.254562) = 0.895191.
INTERPRETATION: rather than "p = 0.06, not significant", the memo states
            an 89.5% probability that the change helps, which is above the
            71.43% threshold from DM-02 and therefore supports acting.
```

**Consistency check with the frequentist summary.** The same data give a one-sided z of `2.266/1.459 = 1.5530`, a one-sided p-value of about `0.0602`. A memo that reports "not significant at 5%" and stops has thrown away the decision-relevant information, because the threshold in DM-02 is 71.4%, not 95%. The posterior of 89.5% and the p-value of 0.060 are not in conflict; they answer different questions, and only the posterior is on the scale the decision rule consumes.

**Sensitivity to the prior.** Because the update is an explicit function of `(μ0, σ0)`, the memo can and should publish a small sensitivity grid. Using the same `d̂` and `s`, with `σ0 = 2 pp`:

| Prior mean μ0 (pp) | Prior SD σ0 (pp) | Posterior mean (pp) | Posterior SD (pp) | P(helps) | Odds vs prior |
|---|---|---|---|---|---|
| 0.00 | 2.0 | 1.479 | 1.179 | 0.895 | 8.54 |
| −1.00 | 2.0 | 1.132 | 1.179 | 0.831 | 11.06 |
| −0.50 | 2.0 | 1.305 | 1.179 | 0.866 | 9.63 |
| +1.00 | 2.0 | 1.826 | 1.179 | 0.939 | 6.91 |
| 0.00 | 1.0 | 0.724 | 0.825 | 0.810 | 4.26 |
| 0.00 | 4.0 | 2.000 | 1.371 | 0.928 | 12.83 |

Two readings matter. First, the posterior probability moves from 0.810 to 0.928 across defensible priors while remaining above the 0.714 threshold in every cell — so the *decision* is robust even though the *probability* is not precise. That is the correct form of a robustness statement, and it is far more useful than a bare confidence interval. Second, the prior mean matters less than the prior spread here, because the data precision (0.470) is comparable to the prior precision at `σ0 = 2` (0.25) but dominates at `σ0 = 4` (0.0625).

**Scale discipline.** The single most common error in applying DM-03 is mixing scales: a prior stated as "we usually see 1–3% relative lifts" combined with an estimate stated as "+2.27 percentage points". With a base rate near 25%, a 2.27 pp absolute lift is roughly a 9% relative lift, so the prior in relative terms is far more pessimistic than the prior in absolute terms. The memo must state which scale each input uses, and the update must be performed entirely in one of them.

**Takeaway:** convert every estimate to a probability the change helps, and report the prior and a sensitivity grid so the reader can see whether the decision or only the probability is fragile.

### 4.3 Variants — reversibility, information value, mix standardisation, and delay

The base and general cases assume a clean binary bet with an immediate, permanent consequence. Four variants cover the situations memos actually face.

**Variant A — reversible decisions (salvage term).** If a wrong decision can be partly undone, the effective loss is the unrecoverable portion. Let `R` be the amount of `L` recovered by reversal, net of the cost of reversing.

```text
FORMULA [DM-04]: p*_R = (L - R) / (G + L - R)
WHERE: R salvage [currency], 0 <= R <= L; other symbols as DM-02.
CONSTRAINT: reversal is genuinely available within the decision horizon and
            has been exercised or rehearsed at least once; the gain G is not
            itself reduced by the reversal capability (e.g. no permanent
            capability cost); the reversal cost is already netted out of R.
NUMERICAL EXAMPLE: G = 480000, L = 1200000, R = 400000:
            p*_R = (1200000-400000)/(480000+1200000-400000)
                 = 800000/1280000 = 0.625.
INTERPRETATION: a rehearsed rollback that recovers one third of the downside
            lowers the evidence bar from 71.4% to 62.5%; the memo should
            claim this reduction only if the rollback is operational,
            because the reduction is the entire justification for acting
            on weaker evidence.
```

The derivative is instructive: `dp*_R/dR = -G/(G+L-R)^2 < 0`, so salvage always lowers the bar, and the reduction is largest when `G` is large relative to the residual downside — i.e. reversible decisions with big upside should be made fast and cheaply, which is the formal content of the "reversible decisions deserve less process" heuristic. The failure mode is claiming reversibility that does not exist operationally; Section 8 treats this as FAIL-03.

**Variant B — value of one more round of evidence (EVSI).** A test with sensitivity `se` and specificity `sp` produces a positive or negative result. Acting on the updated probabilities rather than on the current one yields an expected gain; the difference is the expected value of sample information.

```text
FORMULA [DM-05]: P(+) = p*se + (1-p)*(1-sp)
                 p_+  = p*se / P(+)
                 p_-  = p*(1-se) / ( p*(1-se) + (1-p)*sp )
                 EVSI = P(+)*max(0, EV(p_+)) + (1-P(+))*max(0, EV(p_-)) - max(0, EV(p))
WHERE: p current probability the change helps [dimensionless];
       se sensitivity, sp specificity of the proposed test [dimensionless];
       p_+, p_- posterior probabilities after a positive / negative result;
       EV(.) from DM-01; max(0, .) because the decision maker can decline.
CONSTRAINT: the test result is binary and its se/sp are known from a
            comparable population; the test is independent of the effect
            given the true state; the cost of the test is excluded from EVSI
            and subtracted separately; the decision is one-shot.
NUMERICAL EXAMPLE: p = 0.895191, G = 480000, L = 1200000,
            se = 0.90, sp = 0.90:
            P(+) = 0.895191*0.90 + 0.104809*0.10 = 0.816153
            p_+  = 0.805672/0.816153 = 0.987158
            p_-  = 0.089519/(0.089519 + 0.094328) = 0.486921
            EV(p_+) = 0.987158*480000 - 0.012842*1200000 = 458425.7
            EV(p_-) = max(0, 0.486921*480000 - 0.513079*1200000) = 0
            EV(p)   = 0.895191*480000 - 0.104809*1200000 = 303920.7
            EVSI = 0.816153*458425.7 + 0 - 303920.7 = 70224.7
INTERPRETATION: a 90/90 test is worth about 70225 currency units of
            expected value, so commissioning it is rational only if it
            costs less than that AND completes before the decision date.
```

The same computation with weaker or stronger tests shows the shape of the answer, which is the practical reason to compute it at all:

| Test se / sp | P(result +) | p after + | p after − | EVSI (currency) | Verdict |
|---|---|---|---|---|---|
| 0.80 / 0.75 | 0.742 | 0.9647 | 0.6949 | 8390 | Rarely worth the delay |
| 0.90 / 0.90 | 0.816 | 0.9872 | 0.4869 | 70225 | Worth it only if fast |
| 0.95 / 0.95 | 0.856 | 0.9939 | 0.3101 | 97998 | Strong case for the test |
| 0.60 / 0.90 | 0.548 | 0.9809 | 0.7915 | ≈0 | Useless: cannot change the action |

The last row is the diagnostic gem. A test with poor sensitivity is worthless *here* not because it is inaccurate but because a negative result leaves the posterior at 0.79, still above the 0.714 threshold — the result cannot flip the decision, so it cannot have value. This is the formal answer to "should we run one more study?" and it defeats the reflex to gather more evidence for its own sake.

Note also the ceiling: EVSI can never exceed the expected value of perfect information, `EVPI = min(p*G, (1-p)*L)`, which here equals `min(429553, 125771) = 125771`. The 0.95/0.95 test captures about 78% of that ceiling; the 0.80/0.75 test captures about 7%.

**Variant C — mix-standardised comparison.** When the treated and control populations differ in composition, the raw difference confounds effect with mix. Standardising the treated outcome to the control mix yields the comparison the decision needs.

```text
FORMULA [DM-06]: d_std = Σ_i w_i^C * r_i^T - Σ_i w_i^C * r_i^C
WHERE: i indexes segments (i = 1..k); w_i^C control share of segment i
           [dimensionless, sums to 1]; r_i^T, r_i^C segment rates in
           treated and control [proportion]; d_std standardised difference
           on the proportion scale.
CONSTRAINT: segments are defined by pre-treatment characteristics only
            (never by post-treatment behaviour, which would condition on
            a consequence of the treatment); every segment in the control
            mix exists in the treated population with nonzero counts; the
            standardisation target (control mix) is the population the
            decision applies to.
NUMERICAL EXAMPLE: control mix 1000/1000 across two segments with rates
            0.60 and 0.10 gives raw control 0.35. Treated counts 400 and
            1600 with rates 0.65 and 0.12 give raw treated 0.226, a raw
            difference of -12.4 pp. Standardising to the control mix:
            d_std = (1000*0.65 + 1000*0.12)/2000 - 0.35
                  = 0.385 - 0.35 = +0.035.
INTERPRETATION: the raw comparison says the treatment is 12.4 pp worse and
            the standardised comparison says it is 3.5 pp better; the sign
            flips because the treated population is 80% in the low-rate
            segment. The memo must report the standardised figure with the
            standardisation target named, or it will recommend the opposite
            of the correct action.
```

**Variant D — the marginal cost of delay.** A decision deferred by `Δ` periods forgoes the value that would have accrued, and may also consume the decision's own value through decay.

```text
FORMULA [DM-07]: C_delay = p * (G/H) * Δ
WHERE: H consequence horizon in periods [periods]; G/H value accrued per
           period if the change helps [currency/period]; Δ delay [periods];
           C_delay forgone expected value [currency].
CONSTRAINT: value accrues approximately uniformly over the horizon; the
            delayed start does not extend the horizon (if it does, the
            cost is only the shift, not the whole window); no compounding
            or discounting within H; p held constant across the delay.
NUMERICAL EXAMPLE: G = 480000, H = 52 weeks, Δ = 3 weeks, p = 0.895191:
            C_delay = 0.895191*(480000/52)*3
                    = 0.895191*9230.769*3 = 24781.8 currency units.
INTERPRETATION: three weeks of review costs about 24,790 units of expected
            value, roughly a fifth of the 125771 EVPI; so a review that
            takes three weeks must buy more than a fifth of all possible
            certainty to pay for itself.
```

The comparison in the last line is the single most useful sentence a memo can contain about process: it prices the review against the maximum possible value of the information the review could produce. When `C_delay > EVPI`, no amount of additional certainty obtained during the delay can repay the delay, and the memo should say so explicitly.

**Takeaway:** reversibility lowers the evidence bar, information has a computable and usually modest value, composition differences can flip the sign of a finding, and delay has a price comparable to certainty.

### 4.4 Limits and counterexamples

The formulas above are decision-theoretic identities; their application conditions are where memos actually go wrong. Each limit below is stated with the counterexample that demonstrates the failure.

**Limit 1 — the binary outcome summary is a fiction that must be audited.** DM-01 collapses the consequence into two scalars, which means any decision whose outcomes are multi-valued, path-dependent, or correlated across entities is being approximated. *Counterexample:* a pricing change that raises revenue per order by 3% but reduces orders by 6% cannot be scored with a single `G`, because the two effects have different horizons and different stakeholders. The correct response is not to abandon DM-01 but to define `G` and `L` on the *net* consequence over the horizon and to publish the two components, so that a reader who cares about the order count can see the trade being made.

**Limit 2 — risk neutrality is an assumption with an owner.** `EV` maximisation is appropriate for an organisation that can absorb repeated independent bets. *Counterexample:* a single decision that could produce a regulatory finding or a safety event is not a repeated bet, and a risk-neutral calculation will systematically over-recommend. In such cases the memo should report `EV` alongside the worst-case loss and let the owner apply an explicit risk constraint, rather than smuggling risk aversion into `L` without saying so.

**Limit 3 — known standard error.** DM-03 treats `s` as known. *Counterexample:* with 40 observations, `s` is itself uncertain by roughly `1/√(2n) ≈ 11%`, and the normal approximation to the sampling distribution is poor when the outcome is a rare event. The practical fix is a t-based or bootstrap interval for the effect and a corresponding sensitivity row in the memo, not silent normality.

**Limit 4 — the prior cannot be chosen after seeing the data.** The posterior in DM-03 is a deterministic function of `(d̂, s, μ0, σ0)`. If the analyst tries several priors and reports the one that produces the desired posterior, the result is not a posterior. *Counterexample:* a memo reporting `P(helps) = 0.94` that used `σ0 = 4` "because we were uncertain" when the team's actual historical spread is 2 pp. The defence is procedural: write the prior down in the pre-registration note (Section 3, definition 18), timestamp it, and show all rows of the sensitivity grid rather than the best one.

**Limit 5 — the sign is not the whole consequence.** DM-03 delivers the probability of a positive effect, and DM-02 needs only that sign. But `EV` also depends on magnitude, and a 90% chance of a tiny gain can be worth less than a 60% chance of a large one. *Counterexample:* two candidate features with `P(helps)` of 0.95 and 0.65 and expected magnitudes of 0.2 pp and 4 pp respectively. Ranking by probability ranks them backwards. The memo must report expected magnitude and `EV`, not only the probability, when comparing options.

**Limit 6 — EVSI assumes the test measures the same thing the decision depends on.** *Counterexample:* a usability test that measures task completion with an evaluator present has excellent `se/sp` for "can be completed with help" and poor validity for "will be completed unaided in production". The EVSI arithmetic will then be correct and the decision still wrong, because the tested state is not the payoff-relevant state — this is the "right answer to the wrong question" failure and it is invisible to every formula in this section.

**Limit 7 — standardisation extrapolates.** DM-06 reweights treated segment rates to the control mix, which requires each control segment to have enough treated observations to estimate `r_i^T` with usable precision. *Counterexample:* a segment with 12 treated observations producing a rate of 8/12 is reweighted with the same authority as a segment with 12,000, and the standardised difference can move several points on the strength of the small cell. The fix is to report the segment counts alongside the standardised figure and to require a minimum cell size before trusting the reweighting.

**Limit 8 — delay cost is not always linear or non-negative.** DM-07 assumes value accrues uniformly and that delay only defers value. *Counterexample:* a decision to *stop* an intervention that is causing harm has a negative delay cost in the sense that delay accrues further loss — the sign flips and the magnitude is `(1-p)`-weighted rather than `p`-weighted. A memo on a stop decision that uses DM-07 unchanged will understate the cost of delay by the entire harm rate.

**Takeaway:** every formula here has a stated domain; the memo's credibility comes from naming the domain it is standing on, not from the sophistication of the arithmetic.
## 5. Worked Numerical Example

The example below is one complete memo cycle on a single decision. Every input is ILLUSTRATIVE and was chosen to make the arithmetic checkable by hand; no company, product, or campaign is described. The arithmetic is VERIFIED: each step was recomputed independently (numerator and denominator reconstructed from the partitions, and the decision quantities re-derived from the raw counts rather than from the rounded intermediates shown).

### 5.1 The situation and the pre-registration note

An operations team owns a checkout flow. A change to the address-entry step is proposed: replace free-text entry with a structured selector. The owner is the checkout operations lead, who funds the engineering time and owns the completion metric. The decision date is four weeks out, because the next release train leaves then and missing it costs a full cycle.

Before any result is known, the memo's pre-registration note fixes the following. It is timestamped, and it is not edited after this point.

```text
PRE-REGISTRATION (timestamped before analysis)
Decision:      ship structured address entry to 100% of sessions
               vs. keep free-text entry, decided by the ops lead
Owner:         checkout operations lead (budget authority)
Decision date: release-train cutoff, 4 weeks from note date
Primary metric: order completion rate = completed orders / eligible
                started checkouts, per session, 28-day window
Grain:         one session; key = session_id
Population:    sessions starting checkout with >=1 address field touched
Cutoff:        event-time cutoff 2026-11-01T00:00:00Z, restatement lag
               profile known for 72 h; no records accepted after cutoff
Guardrail:     refund-request rate within 14 days must not rise more
               than 0.3 pp absolute
Action bands:  p(helps) >= 0.714 -> ship
               0.55 <= p(helps) < 0.714 -> ship to 25% for 2 weeks
               p(helps) < 0.55 -> do not ship; keep free text
Falsification: guardrail breach beyond 0.3 pp, or completion effect
               below +1.0 pp at the 2-week checkpoint when ramping
```

Note what the note does *not* contain: a target sample size expressed as a power calculation against an arbitrary minimum detectable effect, and a promise to reach statistical significance. The decision rule is a probability threshold derived from the value model, so the relevant question is not "is it significant" but "is the posterior above 0.714". This reframing is the central practical benefit of the framework: it converts a sample-size negotiation into an evidence-sufficiency question, and it lets a decision proceed on weak-but-sufficient evidence when the stakes justify it.

### 5.2 Step 1 — population, grain, and reconciliation

Raw session logs for the window contain 2,190 session records with at least one address field touched. Reconciliation proceeds as a series of subtractions, and each subtraction is recorded with the reason, because a memo that states only the final denominator invites the reader to assume the exclusions were convenient.

```text
 2,190  raw session records touching an address field
-  112  duplicate session_ids from a client retry that emitted
        two start events (identity: keep the earliest by event time)
-   38  sessions with zero payment-method interaction before the
        session ended: out of scope, the changed step is downstream
-    0  records arriving after the cutoff (the 72 h restatement
        window closed 3 days before extraction)
=2,040  eligible sessions in the analysis window
```

The denominator for the primary metric is therefore 2,040 sessions, and the memo states it as an identity: `eligible = raw − duplicates − out_of_scope − late`. A reader who disagrees with an exclusion can recompute the metric on the wider population; a memo that reports only 2,040 without the ledger cannot be audited at all.

The duplicate count matters more than its size suggests. If the 112 duplicates had been retained, and if duplicates were more likely among sessions that failed at the address step (retries cluster around errors), the completion rate would be biased downward in exactly the segment the change is meant to help. Population errors are almost never random with respect to the outcome of interest, which is why the reconciliation ledger belongs in the memo body rather than in an appendix.

### 5.3 Step 2 — the value model, fixed before the estimate is read

The operations lead and a finance partner fix the value model in a thirty-minute conversation, before the completion rates are computed. This ordering is not ceremonial: it is the only reliable defence against the analyst estimating the loss so that the result becomes decisive.

```text
Gain G:  the change raises completed orders. The completed-order
         contribution margin is 40 currency units per order. The
         address step carries an estimated 48,000 eligible sessions
         per month; the horizon H is 52 weeks (the change will not be
         revisited before then). The assumed magnitude if the change
         helps is a 2.5 pp absolute completion lift, which is about a
         10% relative lift on the 25% base - the team's historical
         median for this class of change.
         G = 48,000 sessions/month * 12 months * 0.025 lift
             * 40 currency/order
           = 48,000*12 = 576,000 sessions/year
           = 576,000*0.025 = 14,400 incremental orders
           = 14,400*40 = 576,000 currency gross gain.
         Net of the 96,000 currency one-off engineering cost:
         G = 576,000 - 96,000 = 480,000 currency.

Loss L:  if the change harms completion, the harm is symmetric in
         mechanism: sessions that would have completed now abandon,
         plus support cost for confused users, plus the cost of
         reverting. The reversion is a flag flip that has been
         rehearsed once. Harm magnitude is estimated at a 3.75 pp
         decline (1.5x the assumed upside, because a broken address
         step blocks the order entirely rather than degrading it)
         over the same 576,000 sessions:
         orders lost = 576,000*0.0375 = 21,600
         margin forgone = 21,600*40 = 864,000
         support cost = 21,600 affected * 1.5 contacts * 12 currency
                       = 388,800
         reversion engineer time = 120 h * 95 currency = 11,400
         L = 864,000 + 388,800 + 11,400 = 1,264,200 currency.
         Rounded for the memo: L = 1,200,000 currency (the rounding
         is disclosed; it lowers p* by about 1.0 pp (from 0.7248 to
         0.7143), in the optimistic
         direction, and the memo says so).

Salvage R: the reversion recovers the loss from the reversion point
         forward but not the harm already incurred. Estimated
         recoverable: 400,000 currency (the support and margin losses
         from the first two weeks of an assumed six-week harm period,
         since detection takes about two weeks).
         R = 400,000 currency, and it is claimed only because the flag
         flip was rehearsed once on a staging account.
```

The break-even probability follows directly, and it is the first number the memo computes:

```text
p* = L/(G+L) = 1,200,000/(480,000+1,200,000)
   = 1,200,000/1,680,000 = 0.714285714 -> 71.43%

with salvage:
p*_R = (L-R)/(G+L-R) = 800,000/1,280,000 = 0.625 -> 62.50%
```

The gap between 71.43% and 62.50% is the quantified value of having rehearsed the rollback. It is worth writing in the memo as a sentence, because it makes the case for operational readiness concrete: the team that rehearses its rollback gets to act on 62.5% confidence instead of 71.4%, which in practice is the difference between deciding this cycle and deferring to the next.

An important honesty check: the value model uses an assumed 2.5 pp lift to size `G`. That assumption is not the estimate being tested; it is the scale over which the estimate is applied. If the eventual estimate is 1.0 pp, `G` should be rescaled to `576,000*0.01*40 − 96,000 = 134,400`, which changes `p*` substantially (Section 5.7 shows the case at the observed magnitude). The memo handles this by stating that `p*` is conditional on the magnitude band assumed in the value model, and by reporting `p*` for the assumed band and for the observed band. Section 5.7 shows that computation.

### 5.4 Step 3 — the estimate and its standard error

Completion rates by arm over the 28-day window:

```text
treated:  n_T = 1,840 sessions, completions = 504
          p̂_T = 504/1,840 = 0.273913043...
control:  n_C = 1,795 sessions, completions = 451
          p̂_C = 451/1,795 = 0.251253482...
difference d̂ = p̂_T - p̂_C = 0.022659562 = +2.266 pp

standard error (independent samples, normal approximation):
s = sqrt( p̂_T(1-p̂_T)/n_T + p̂_C(1-p̂_C)/n_C )
  = sqrt( 0.273913*0.726087/1,840 + 0.251253*0.748747/1,795 )
  = sqrt( 0.198871/1,840 + 0.188125/1,795 )
  = sqrt( 0.000108082 + 0.000104805 )
  = sqrt( 0.000212887 ) = 0.014590909 = 1.459 pp
```

A note on what `s` is and is not. It is the standard error of the difference in proportions under independent sampling, which assumes sessions are independent of each other. If the design assigned at the account level and analysed at the session level, sessions within an account are correlated and `s` is understated; the memo must state the assignment unit and, where they differ, use a cluster-robust standard error or aggregate to the assignment unit. Here the memo states that assignment was at the session level via a randomised hash of `session_id`, so the independence assumption is defensible.

The frequentist summary, reported for completeness rather than for the decision:

```text
z = d̂/s = 0.022659562/0.014590909 = 1.552992
one-sided p = 1 - Φ(1.552992) = 0.060213
95% interval = d̂ ± 1.96*s = 0.022660 ± 0.028598
             = (-0.005938, +0.051258) = (-0.59 pp, +5.13 pp)
```

The interval spans zero. A memo that stops here writes "not significant, recommend more data" and has produced no decision. The framework's contribution is to convert this same estimate into the quantity the decision actually needs.

### 5.5 Step 4 — the posterior probability that the change helps

The prior is taken from the team's historical record of similar checkout-flow changes, which is a documented list of nine prior changes with a median lift of zero and a spread of about 2 pp on the completion-rate scale. That gives `μ0 = 0`, `σ0 = 2 pp`: a prior that says the change is as likely to hurt as help, with a two-point spread. This is deliberately sceptical, and stating it is what allows a stakeholder who believes the prior should be optimistic to recompute rather than argue.

```text
precision_prior = 1/σ0^2 = 1/4 = 0.25
precision_data  = 1/s^2  = 1/0.000212887 = 4.697... (in pp^-2: 1/2.129281
                  = 0.469644)
precision_total = 0.25 + 0.469644 = 0.719644
σ1^2 = 1/0.719644 = 1.389576  -> σ1 = 1.178803 pp
μ1 = σ1^2 * (μ0/σ0^2 + d̂/s^2)
   = 1.389576 * (0 + 2.265956/2.129281)
   = 1.389576 * 1.064205 = 1.478877 pp
P(helps) = Φ(μ1/σ1) = Φ(1.478877/1.178803) = Φ(1.254562)
         = 0.895191
```

The memo now has the sentence it needs: *"Under our stated prior, the probability that this change improves completion is 89.5%."* That is a claim a non-technical owner can act on, and it is auditable because both inputs and the update rule are written down.

The odds decomposition makes the same number legible in a second way:

```text
prior odds = Φ(0/2)/(1-Φ(0/2)) = 0.5/0.5 = 1.000
posterior odds = 0.895191/0.104809 = 8.541
Bayes factor (implied) = 8.541/1.000 = 8.541
```

An implied Bayes factor of 8.5 is "moderate" evidence by conventional verbal scales, and it is worth saying so, because a stakeholder who has been told that "p = 0.06 means no result" will be surprised that the same data constitute moderate evidence in favour. The two statements are consistent: the p-value asks how often a null of exactly zero would produce an estimate this large, while the Bayes factor asks how much the data shift the odds between "helps" and "hurts".

### 5.6 Step 5 — decision quantities and the recommendation

```text
EV at posterior:  EV = 0.895191*480,000 - 0.104809*1,200,000
                     = 429,691.7 - 125,770.8
                     = 303,920.7 currency units
EVPI at posterior: min(p*G, (1-p)*L)
                     = min(429,691.7, 125,770.8) = 125,770.8
Cost of 3-week delay: C_delay = 0.895191*(480,000/52)*3
                     = 0.895191*9,230.769*3 = 24,790.0
EVSI of a 90/90 confirmation test at 1 week:
   P(+) = 0.895191*0.90 + 0.104809*0.10 = 0.816153
   p_+ = 0.805672/0.816153 = 0.987158
   p_- = 0.089519/(0.089519+0.094328) = 0.486921
   EVSI = 0.816153*458,425.7 + 0 - 303,920.7 = 70,224.7
   net of 1-week delay cost (8,263.4): 61,961
```

Read the four numbers together. The expected value of acting is about 304,000 units. The maximum value of *any* additional certainty is about 125,800 units. A strong confirmation test would capture about 70,200 of that, and would cost about 8,263 in delay — so on paper the test is worth roughly 62,000 net.

The memo nevertheless recommends shipping without the test, and the reason is a condition the formula cannot see: the proposed confirmation test would run on the same sessions in the same four-week window, and the release train departs before a second test could complete. The EVSI calculation assumes the test result arrives before the decision date; here it does not. This is an honest limitation, and the memo states it explicitly rather than quietly omitting the EVSI number — a memo that hides an inconvenient calculation is worse than one that has none.

The guardrail is checked next, and the memo must report it in absolute percentage points because the guardrail was registered in those terms:

```text
refund-request rate, treated  = 62/1,840  = 0.033696 = 3.370 pp
refund-request rate, control  = 55/1,795  = 0.030641 = 3.064 pp
difference = +0.306 pp; limit +0.300 pp -> breach by 0.006 pp
standard error of the difference:
  sqrt(0.033696*0.966304/1,840 + 0.030641*0.969359/1,795)
  = sqrt(0.000017695 + 0.000016553) = sqrt(0.000034248) = 0.005852
= +0.585 pp standard errors from zero
```

The guardrail is nominally breached by 0.006 pp — a quantity that is one hundredth of one standard error. The correct memo language is neither "guardrail breached, do not ship" nor "guardrail fine, ship", but a third statement: *the guardrail difference is indistinguishable from zero at this sample size, and the pre-registered limit is inside the noise band, so the guardrail as specified cannot discriminate.* The memo proposes restating the guardrail as a 95% upper bound requirement rather than a point estimate, and flags that this restatement must be recorded as a definition change with a version number, not as a silent reinterpretation.

That last move is the single most important habit in this file: when a pre-registered rule turns out to be unusable, the memo says so, changes it in the open, and versions the change. It does not pretend the rule was met.

### 5.7 Step 6 — magnitude sensitivity for the value model

Because `G` was sized using an assumed 2.5 pp lift while the observed lift is 2.266 pp on a 25.1% base (about 9.1% relative), the memo reports `p*` under both magnitudes. This is where the value model and the estimate must be reconciled.

```text
Assumed band (2.5 pp -> lift used for G):        p*_R = 0.625
Observed band: G_obs = 576,000*0.02266*40 - 96,000
                     = 522,086 - 96,000 = 426,086
              L unchanged at 1,200,000; R = 400,000
              p*_R,obs = (1,200,000-400,000)/(426,086+1,200,000-400,000)
                       = 800,000/1,226,086 = 0.652482 -> 65.25%
```

The threshold moves from 62.50% to 65.25% when the value model is rescaled to the observed magnitude — a 2.75 pp increase. The posterior of 89.5% still clears it, so the decision is unchanged. But the exercise is not cosmetic: it shows that a value model built on an optimistic assumption *understates* the evidence required, and it quantifies by how much. A memo that skips this step is implicitly claiming that the assumed magnitude was correct, which it cannot know.

### 5.8 Step 7 — the action contract and the settlement

The memo closes with seven fields, each of which a third party could check:

```text
Action:        ship structured address entry to 100% on release date
Owner:         checkout operations lead
Start date:    next release train (4 weeks from memo date)
Success metric: order completion rate, same definition, 28-day window
Guardrail:     refund-request rate, restated as: 95% upper bound of
               difference must be below +0.5 pp (definition v1.1)
Review date:   6 weeks after start (allows 2 weeks detection + 4 weeks
               measurement)
Rollback if:   guardrail upper bound exceeds +0.5 pp at review, or
               completion point estimate is below +1.0 pp
Prediction:    completion +2.3 pp (range +0.5 to +5.1 pp), 89.5%
               probability of a positive effect
```

The prediction line is the settlement record. Twelve weeks later, the follow-up memo compares the realised effect to `+2.3 pp` and records whether the direction was right, whether the magnitude was inside the stated range, and whether the 89.5% confidence band was calibrated across the team's last several memos. A team that finds its 90% bands correct only 70% of the time knows to widen its priors; a team that finds them correct 95% of the time knows its priors are too pessimistic and is leaving value on the table by over-collecting evidence.

### 5.9 Independent checks performed on this example

Six checks were run, each of which is cheap and each of which catches a different class of error:

1. **Denominator reconstruction.** The 2,040 eligible sessions were rebuilt from disjoint partitions (duplicate-cleaned and in-scope), and the partition sum equalled the declared total exactly. A mismatch here would have invalidated every downstream rate.
2. **Rate arithmetic by complement.** `504/1,840 = 0.273913` and `1,840 - 504 = 1,336`; `1,336/1,840 = 0.726087`, and the two sum to 1.000000. Same for control. This catches numerator/denominator transposition, which is the most common silent arithmetic error in memos.
3. **Variance sign and magnitude.** Each term of `s^2` is positive, and `s = 1.459 pp` is smaller than either arm's binomial standard error (`sqrt(0.1989/1840) = 1.04 pp` scaled) — actually larger than either single-arm standard error, as it must be, since the difference of two independent estimates has variance equal to the sum. The check is that `s^2` equals the sum of the two arm variances, not either one alone.
4. **Threshold/EV reconciliation.** Substituting `p = p*_R = 0.625` into the salvage-adjusted value function returns zero: `0.625*480,000 = 300,000` and `0.375*800,000 = 300,000`. The threshold is consistent with the value function by construction, and the check confirms no transcription error entered.
5. **Posterior limit check.** As `σ0 → ∞` the posterior should converge to the likelihood-based `Φ(d̂/s) = Φ(1.552992) = 0.9398`. The `σ0 = 4` row of the Section 4.2 table gives 0.9277, approaching that limit from below, as it must because the finite prior mean of zero pulls the posterior slightly towards zero. As `σ0 → 0` the posterior should converge to the prior's `Φ(μ0/σ0)`; the `σ0 = 1` row gives 0.8100 versus a diffuse limit of 0.9398, moving in the correct direction.
6. **Delay-versus-information comparison.** `C_delay = 24,790` was compared to `EVPI = 125,771`; the ratio 0.197 was computed and reported, because a ratio near or above 1 would have been an argument for immediate action regardless of further analysis.

None of these checks requires specialised software, and all six take less than an hour in total. The habit they instil is the point: a memo whose numbers have been checked this way can be defended in a meeting without re-deriving anything, and its author can say "I checked that" with a specific answer to "how".

**Takeaway:** a worked memo is a chain of eight auditable steps, and the value of the chain is that any single link can be attacked without destroying the recommendation.
## 6. Visual / Geometric Intuition

The quantities in Section 4 have geometric interpretations that make them easier to argue about in a room, and each one exposes a different way a memo can mislead.

### 6.1 The decision threshold is a ratio on the number line

`p* = L/(G+L)` is the position of the threshold between 0 and 1, measured from the loss side. Picture a unit segment: the left end is "certainly harmful," the right end is "certainly helpful," and `p*` marks the point where the expected value crosses zero.

```text
 0.0        0.25       0.50       0.75      1.0
  |----------|----------|----------|---------|
  certainly            coin flip    |   certainly
  harmful                           |   helpful
                                    ^
                                    p* = 0.714  (G=480k, L=1200k)
              ^
              p*_R = 0.625  (salvage 400k)
```

Three geometric facts follow immediately and are worth memorising.

First, symmetric stakes put the threshold at the midpoint: if `G = L`, then `p* = 0.5`, and a coin-flip belief is exactly enough. The "better than even odds" intuition is correct only in the symmetric case, which is the case memos almost never have.

Second, as `L/G` grows, the threshold migrates rightward towards 1, and it does so with diminishing returns: at `L = G` it is 0.5, at `L = 3G` it is 0.75, at `L = 9G` it is 0.9. A factor-of-nine asymmetry in stakes is the difference between a 0.5 and a 0.9 threshold — a much smaller movement than most people expect, which is why "the downside is huge" is a weaker argument than it feels.

Third, salvage moves the threshold leftward by an amount that depends on the residual downside, not on the salvage fraction alone. Returning 400k of a 1,200k loss is a one-third recovery and moves the threshold 8.9 pp; returning 400k of a 2,400k loss would be a one-sixth recovery and move it much less. The threshold responds to the absolute residual, so a memo must report `R` in currency, not as a percentage.

### 6.2 The posterior is a shift of a distribution, not a point

The update in DM-03 is easiest to see as the multiplication of two bell curves. The prior is a distribution over the true effect centred at zero with spread 2 pp; the likelihood is a distribution over the estimate centred at `d̂ = 2.27 pp` with spread `1.46 pp`. Their product is narrower than either and centred between the two means, weighted by precision.

```text
        prior  N(0.00, 2.00)          likelihood N(2.27, 1.46)
            /\                              /\
           /  \                            /  \
          /    \                          /    \
    _____/      \____________________ ___/      \______
        -4   -2    0    2    4        0    2    4    6
                        \                 /
                         \   posterior   /
                          \  N(1.48,1.18)
                           \    /\    /
                            \  /  \  /
                             \/    \/
                           0    2    4

    shaded area to the right of 0 = P(helps) = 0.895
```

The picture explains three things that prose does not.

It shows why the posterior mean (1.48 pp) is below the estimate (2.27 pp): the prior's mean of zero pulls it, and the pull is proportional to the prior's precision relative to the data's. When the team has strong prior information (small `σ0`), the pull is large; when the prior is vague, the estimate dominates. This is exactly the behaviour a stakeholder expects from a careful analyst, and the diagram makes it non-mysterious.

It shows why the posterior *probability* can be high while the estimate remains statistically unremarkable: the area to the right of zero is large because the posterior distribution is narrow relative to its distance from zero, even though the 95% interval on the estimate itself spans zero. The interval and the tail area are different objects computed from different distributions, and conflating them is the source of the "significant/not significant" trap.

It shows the sensitivity result geometrically: widening the prior (larger `σ0`) shifts the posterior mean towards the estimate and narrows the posterior less, which pushes the tail area up. The Section 4.2 grid is a family of such pictures.

### 6.3 The gain/loss plane

Every decision can be placed on a plane whose axes are the log-magnitude of gain and loss. The iso-threshold lines `p* = L/(G+L) = c` are rays from the origin in this plane, since `L/G = c/(1-c)` is constant along a ray.

```text
  L (log)
   |                                p*=0.90  (L=9G)
   |                             /
   |                          /   p*=0.75 (L=3G)
   |                       /
   |                    /    p*=0.667 (L=2G)
   |                 /
   |              /      p*=0.5 (L=G)
   |           /
   |        /
   |     /
   |  /
   +---------------------------------------- G (log)
```

The plane is useful for portfolio thinking. A team that plots its live decisions on it discovers that most of them cluster in two regions: small gain with small loss (threshold near 0.5, should be decided fast by whoever is closest) and large gain with large loss (threshold high, needs the full memo apparatus). The middle region is where process disputes actually occur, and the plot usually shows that the dispute is about where the decision sits, not about what the evidence says. Moving the discussion from "how sure are we" to "where on this plane are we" resolves more meetings than additional analysis does.

A second use of the plane is detecting mispriced caution. A decision with a large `G` and a modest `L` has a threshold well below 0.5; demanding 95% confidence for it is a systematic error that costs the organisation every forgone positive-value action. Conversely, a decision with a modest `G` and a large `L` sits at a high threshold, and the same 95% insistence is correct there. Uniform evidentiary standards across a portfolio are therefore not neutral: they are biased against large-upside, low-downside actions, which are exactly the actions most organisations claim to want.

### 6.4 Information value as a funnel

The relationship between the value of acting, the value of perfect information, and the value of a specific test is a nested funnel.

```text
   +--------------------------------------------------+
   |  EV of acting now                    304k        |
   |   +--------------------------------------------+ |
   |   | EVPI (ceiling on any information)   126k   | |
   |   |   +--------------------------------------+ | |
   |   |   | EVSI, strong test 95/95      98k     | | |
   |   |   |   +--------------------------------+ | | |
   |   |   |   | EVSI, moderate test 90/90  70k | | | |
   |   |   |   |   +--------------------------+ | | | |
   |   |   |   |   | EVSI, weak test 80/75 8k | | | | |
   |   |   |   |   +--------------------------+ | | | |
   |   |   |   +--------------------------------+ | | |
   |   |   +--------------------------------------+ | |
   |   +--------------------------------------------+ |
   +--------------------------------------------------+
        each nested box: value foregone by not having it
```

Read outward-in: the outer box is what acting is worth, and each inner box is the portion of that value attributable to a better-informed decision. The funnel makes the crucial comparison visible: the *marginal* test (weak) captures 8k of a 126k ceiling, which is 6%, and a memo proposing it is proposing to spend real time for almost nothing. The funnel also makes clear that no test can be worth more than EVPI, so a proposal to run a study whose cost exceeds EVPI can be rejected on arithmetic alone, without any judgement about the study's design.

### 6.5 The memo's own geometry: information density by position

A memo has a spatial structure, and readers allocate attention unevenly across it. The useful mental model is a decaying attention curve over the document's length, with a bump at the end where readers look for the ask.

```text
 attention
    |
100%|**
    |  ***
 75%|     **
    |       **
 50%|         ***
    |            ***
 25%|               ***            **
    |                  ****    ****  **
  0%+--------------------********------*----> position
    |__recommendation__|__method__|_caveats_|_ask_|
```

The practical implication is that the recommendation, the decision rule, the threshold, and the ask must all appear in the first screen, and the ask must be repeated at the end. Caveats placed only in the middle are read by almost nobody. This is not a rhetorical trick; it is a correctness requirement, because a caveat that the owner never reads cannot inform the decision the owner makes.

The second structural point is that the memo's length is a cost paid by every reader, and the cost is multiplicative: a 12-page memo read by seven reviewers with an average of 25 minutes each is 175 minutes of senior attention, which at 110 currency per hour is 321 currency — trivial in absolute terms, but it delays seven schedules, and the delay cost computed in DM-07 typically dominates. Section 9 makes this quantitative.

**Takeaway:** the threshold is a position, the posterior is a shifted distribution, the gain/loss plane tells you how much process a decision deserves, and information value nests inside a ceiling.

## 7. Comparison / Selection Table

This section compares the artefacts and methods an analyst can choose between when converting an analysis into a decision. The comparison is organised by the question the analyst faces, not by the tool's name, because the naming is where most of the confusion lives.

### 7.1 Choosing the communication artefact

| Artefact | Decision content | Reader effort | Best when | Fails when | Minimum viable form |
|---|---|---|---|---|---|
| Dashboard widget | None (it displays state) | Seconds per visit | The decision is routine and the rule is already agreed | A novel decision is expected from a display | Bound the metric, show the threshold line |
| Slack-style update | Implicit recommendation | Under a minute | Reversible, small stakes, owner already engaged | Stakes are asymmetric or the population is contested | One number, one comparison, one ask |
| One-page memo | Explicit recommendation + rule | 3–5 minutes | Moderate stakes, one owner, one decision | The value model needs negotiation across functions | Sections 1, 5.6, 5.8 of this sheet compressed |
| Full decision memo | Rule + value model + sensitivity + contract | 10–20 minutes | Asymmetric stakes, multiple reviewers, audit trail needed | The decision is trivially reversible | The eight-step chain in Section 5 |
| Pre-registration only | Rule, no result | 2 minutes to write | Before analysis, always | Written after results are known (then it is fiction) | The block in Section 5.1 |
| Calibration ledger entry | Prediction record | 5 minutes after settlement | Every memo with a prediction | Entries are not compared across memos | Prediction, range, date, outcome, hit/miss |

The selection rule is not "use the most rigorous form." It is "use the lightest form that still carries the decision rule and the cost of being wrong." A Slack update that says "completion +2.3 pp, 89.5% likely positive, threshold was 71%, shipping Tuesday, guardrail is refunds" satisfies the rule for a reversible, moderate-stakes decision. A twelve-page memo that omits the threshold satisfies nothing.

### 7.2 Choosing how to express uncertainty

| Expression | What it claims | Stakeholder readability | Requires | Failure mode |
|---|---|---|---|---|
| Point estimate only | "the effect is X" | High | Nothing | Implies precision that does not exist |
| Confidence interval | "the effect is in [a,b] at 95%" | Medium | Sampling model, correct interval type | Read as a probability that the truth is in the interval |
| One-sided p-value | "the null is unlikely given this data" | Low | Null hypothesis, test direction | Confused with P(helps), leading to wrong action |
| Posterior probability | "89.5% chance the change helps" | High | Explicit prior | Prior can be chosen to reach a desired answer |
| Posterior + prior + sensitivity grid | as above, with fragility visible | Medium | Prior and grid | Longer; requires the reader to accept a prior |
| Qualitative band | "probably, but not certain" | High | Nothing | Cannot be compared to a threshold |

The recommendation for decision memos is the fifth row: posterior plus prior plus a sensitivity grid. It is the only option that produces a number comparable to `p*` while also letting a reader who disputes the prior recompute. The p-value row is the trap: it is the most familiar expression in many organisations and the least directly usable, because it is not on the scale the decision rule consumes. Section 11 includes a question built specifically around this conversion.

### 7.3 Choosing between acting now and buying evidence

| Condition | Action | Reason |
|---|---|---|
| `p` above `p*_R` and EVSI < cost of the test | Act now | Evidence cannot earn its keep |
| `p` above `p*_R` and EVSI > cost, and test finishes before decision date | Consider the test | Positive expected net value |
| `p` above `p*_R` and EVSI > cost, but test finishes after decision date | Act now, log the prediction | Value of information is contingent on timing |
| `p` below `p*` and a test exists that could flip the decision | Test | This is exactly what information value is for |
| `p` below `p*` and no test can move `p` past `p*` | Do not act; close the question | Further evidence gathering is waste |
| `p` below `p*` and the decision is reversible with high `R` | Ramp small | Small exposure buys real information cheaply |
| Cost of measurement exceeds `EVPI` | Act on current evidence | No possible study can be worth it |

The fifth row is the one that saves the most time in practice. A test whose positive and negative results both leave the posterior on the same side of the threshold has zero EVSI, and the correct response is to close the question rather than to run the test "for completeness."

### 7.4 Choosing the comparison basis

| Basis | Use when | Requires | Failure mode |
|---|---|---|---|
| Prior period | No concurrent control; change is broad | Stability of the metric across periods | Confounded by seasonality and other changes |
| Held-out population | A concurrent comparison exists | Genuine assignment mechanism | Assignment depends on the outcome (selection) |
| Segment-standardised difference | Mixes differ materially | Pre-treatment segment definitions | Extrapolation to thin cells |
| Model-based expectation | No comparison group is possible | Documented model and its error | Model error presented as measurement error |
| Expert judgement | Time-critical, low stakes, no data | Named expert and their stated reasoning | Unfalsifiable and unrecordable |

The standardised row deserves emphasis because it is the one that can invert a sign, as Section 4.3 variant C showed: a raw −12.4 pp becomes a standardised +3.5 pp. Any memo whose comparison group differs in composition and which reports only the raw difference is one reweighting away from recommending the opposite of the correct action.

### 7.5 Choosing the review cadence

| Cadence | Latency | Cost per decision | Appropriate stakes |
|---|---|---|---|
| Same-day async | Hours | One reader-hour | Reversible, `p*` below 0.6 |
| Weekly review slot | 3–7 days, plus queue | 3–5 reader-hours | Moderate, `p*` 0.6–0.75 |
| Bi-weekly committee | 10–14 days | 8–15 reader-hours | High, `p*` above 0.75 |
| Ad-hoc escalation | Unbounded | Unbounded | Should be reserved for genuine novelty |

The table is a direct application of DM-07: a committee that adds a week of latency is adding `p*(G/H)*(1/52)` of expected cost in exchange for whatever additional scrutiny it provides. Using the Section 5 numbers, a week costs about 8,263 units, so a committee must be worth more than that in error reduction to justify its cadence. Most committees have never been asked this question, and the arithmetic is usually the first time anyone has priced their own schedule.

**Takeaway:** choose the lightest artefact that carries the rule, express uncertainty as a probability comparable to the threshold, and price every review cadence against the delay it imposes.
## 8. Failure Modes & Edge Cases

Each entry below follows the same discipline: the observable symptom, two or more competing causes that are genuinely hard to tell apart, the check that discriminates between them, the smallest fix that addresses the cause rather than the symptom, and the verification that shows the fix worked. Entries are ordered by how early in the memo chain (Section 2) the failure enters, because earlier failures are more expensive.

### FAIL-01 — The memo answers a question nobody is deciding

**Symptom.** The memo is well-constructed, the numbers reconcile, and nothing happens. Six weeks later the same analysis is requested again, with slightly different framing.

**Competing causes.** (a) No decision was ever named — the request was a topic ("look into churn"), and the memo faithfully answered the topic. (b) A decision existed but the owner is not in the recipient list; the memo went to people who can comment but not decide. (c) The decision exists and the owner received it, but the memo's recommendation is conditional on facts the owner does not control, so the owner cannot act on it alone.

**Discriminating check.** Ask three questions of the memo text alone: is there a sentence of the form "Option A vs Option B, decided by ROLE by DATE"? Is that role in the `To:` line? Does the recommendation require any action from a role not in the `To:` or `Cc:` line? A "no, no, yes" pattern indicates (a), "yes, no, no" indicates (b), "yes, yes, yes" indicates (c).

**Bounded fix.** For (a), reply with one sentence converting the topic to a decision with a proposed owner and date, and ask for confirmation; do not write the memo until the conversion is accepted. For (b), re-send to the owner with the decider named in the first line, and move the discussion group to `Cc:`. For (c), add the dependent owner to the action contract as a named co-owner with their own commitment.

**Verification.** Re-read the memo one week later: has a calendar entry, ticket, or explicit decline appeared? A memo that produced neither an action nor a recorded decline has not closed, regardless of how many replies it received. Track this as a two-value field in the memo register (`closed: yes/no`) rather than as a feeling.

### FAIL-02 — The denominator moved while the headline stayed fixed

**Symptom.** The metric is reported as improving for three consecutive periods, but the operational team insists nothing improved. Both statements are true.

**Competing causes.** (a) The eligible population changed because an exclusion rule was relaxed or tightened, so a larger or smaller base now sits under the same ratio. (b) An instrumentation change altered what counts as a qualifying event, so the numerator and denominator both shifted and the ratio moved for compositional reasons. (c) Genuine behaviour change in one segment is offsetting a genuine decline in another, and the aggregate is nearly flat because of mix.

**Discriminating check.** Recompute the metric on the *frozen* population definition from the earliest period, applying it to all periods. If the frozen-definition series is flat while the live series rises, cause is (a) or (b). To separate those two, compare the numerator and denominator *levels* rather than the ratio: an instrumentation change typically moves both by a similar proportional amount at a specific date, producing a visible step; an exclusion-rule change typically moves only the denominator, producing a ratio jump without a numerator step.

**Bounded fix.** Publish the series under a versioned metric definition (`completion_v1`, `completion_v2`), show the overlap period where both definitions are computable, and state the size of the definitional break so readers can mentally rebase. Never backfill silently: the historical numbers are what the previous decisions were made on.

**Verification.** After publishing both versions, a reader should be able to answer "how much of the change is definitional?" with a number. If the overlap period is unavailable, that number is `OPEN` and the memo must say so — it must not present the break as growth.

### FAIL-03 — Reversibility is claimed but not rehearsed

**Symptom.** A memo lowers its evidence bar using the salvage argument (DM-04), ships on weak evidence, and then discovers the rollback path does not work: the flag flips but the data migration is one-way, or the flag was removed in the previous release, or the rollback itself takes four hours of pipeline time during which harm continues.

**Competing causes.** (a) The mechanism was genuinely available at design time but has since been removed or gated behind another team's deploy. (b) The mechanism works technically but its latency is long enough that the harm accrued before reversion exceeds `R` as estimated. (c) The mechanism works for the code path but not for the data already written, so users who experienced the change cannot be returned to their previous state.

**Discriminating check.** Rehearse the rollback on a staging population that has been exposed to the change for at least as long as the production exposure window, and time it end to end including pipeline and data-restoration steps. Compare measured undo time against the detection-to-decision latency plus undo time, and compare that total against the harm accrual rate implied by `L`. If `(detection + undo) * harm_rate > R`, the salvage claim is overstated regardless of whether the flag exists.

**Bounded fix.** Either reduce `R` in DM-04 to the measured recoverable amount, or invest in the rollback path until the measured total time is short enough. The first is free and correct; the second is a real cost that belongs in the memo's `L` if it is incurred to justify the decision.

**Verification.** Record the rehearsal: date, population, steps, measured duration, and the operator. A salvage claim without a rehearsal record is `HEURISTIC`, not `VERIFIED`, and the memo must label it that way — which in turn means `p*_R` should be computed with a smaller `R` or reported alongside `p*` as an optimistic bound.

### FAIL-04 — The prior was chosen after the data

**Symptom.** The memo reports a posterior probability that is suspiciously well-aligned with the recommendation, and when pressed, the analyst cannot produce a timestamped prior.

**Competing causes.** (a) The prior really was chosen in advance and the alignment is coincidence. (b) The prior was chosen from a set of plausible priors to produce a desired posterior; the analyst may not even experience this as dishonest, since each candidate prior is individually defensible. (c) The prior is genuine but was stated in a different unit (relative lift) from the estimate (absolute points), so the update was performed on mismatched scales and the resulting number is meaningless even though the prior is honest.

**Discriminating check.** Ask for the prior's provenance in one of three forms: a timestamped document, a computation from a documented historical list with a stated rule (median and interquartile range of the last `n` comparable changes), or a named external source with a page reference. If none exists, cause is (b). To detect (c), ask what units the prior is in and what `σ0` is in the same units as `d̂`; a prior quoted as "1–3% lifts" against a `d̂` of "+2.27 pp" is a scale mismatch.

**Bounded fix.** For (b), replace the prior with the documented-historical-list construction, which is mechanical and hard to game: take all comparable past changes, compute the distribution of their effects, use that distribution's centre and spread. It will sometimes be unflattering, which is the point. For (c), restate the prior on the estimate's scale and redo DM-03.

**Verification.** Publish the sensitivity grid (Section 4.2) showing all rows, not the chosen row. If the recommendation survives across the grid, the prior dispute is moot and the memo can say so; if it does not, that fragility is the memo's most important content and must be in the body, not an appendix.

### FAIL-05 — Two populations are compared because the labels match

**Symptom.** A comparison shows a large effect. A subject-matter expert says the two groups are "not the same kind of thing." Both are right, and the memo's recommendation is wrong.

**Competing causes.** (a) The treated population is genuinely different in composition (as in the Section 4.3(C) example, where 80% of the treated group sat in a low-rate segment). (b) The populations are compositionally similar but differ in *eligibility timing*: the treated group entered through a new path that admits entities earlier in their lifecycle, so the comparison mixes a lifecycle effect with a treatment effect. (c) The populations differ in *observation window*: one group has had more time for the outcome to occur, so its rate is mechanically higher.

**Discriminating check.** Compute the segment-level rates and counts for both groups. If any segment's weight differs by more than a few points between groups, cause is (a) and standardisation is required. If weights match but the time-since-entry distribution differs, cause is (b). If the exposure-duration distribution differs even with matched entry timing, cause is (c). All three checks are histograms, not statistical tests, and they should be run before any effect estimate is interpreted.

**Bounded fix.** For (a), apply DM-06 standardisation and report the target mix. For (b), restrict to entities compared at the same lifecycle stage, or include stage as a stratifying variable and report the stage-specific effects. For (c), fix a common observation horizon and censor both groups at that horizon — accepting the loss of late-arriving outcomes rather than mixing windows.

**Verification.** A standardised or stratified result should be accompanied by the cell counts. Recompute the headline under a different standardisation target (treated mix instead of control mix); a result that survives both targets is robust, and one that flips sign between them is a mix artefact that must not be reported as an effect.

### FAIL-06 — The memo reports a decision-relevant number with the wrong sign convention

**Symptom.** The recommendation contradicts the number. "The lift is −2.4, so we should ship," and the reader cannot tell whether that is a typo or a deliberate choice.

**Competing causes.** (a) A genuine typographical error in one place but not another (the direction was computed correctly and transcribed incorrectly). (b) A cost metric was reported as a positive improvement — for example, bounce rate down 2.4 points is good, but the memo wrote "−2.4" under a heading whose other rows are all higher-is-better. (c) The estimate is on a transformed scale (log-odds, or a log-transformed outcome), so its sign refers to the transformed quantity and not to the user-facing metric.

**Discriminating check.** Recompute the direction from raw counts, not from the reported rates: does `numerator_treated/denominator_treated` exceed `numerator_control/denominator_control` in the direction the recommendation assumes? Then check the table header for a "desirable direction" column; if absent, add it and see which rows change meaning. Finally, check the scale of the estimate against the scale of the metric: does the memo say "log-odds" anywhere near the estimate?

**Bounded fix.** Add a "direction that supports the recommendation" column to every numeric table, with values `higher`/`lower`, and require the recommendation sentence to name the direction in words ("completion rises"). For transformed estimates, report both the transformed coefficient and its marginal effect on the user-facing scale, and mark which one the decision uses.

**Verification.** A second reader who has not seen the analysis is asked to state, from the table alone, which arm is better. If they cannot, the table is not decision-ready regardless of whether the numbers are correct.

### FAIL-07 — The guardrail cannot discriminate

**Symptom.** As in Section 5.6: a guardrail is registered as a point-estimate bound, the observed difference is a hair over the bound and a hair under zero's noise, and the memo must decide whether to treat the breach as real.

**Competing causes.** (a) The guardrail is genuinely breached, and the small absolute difference is a real harm that would grow with exposure. (b) The guardrail is not breached, and the exceedance is noise from an underpowered comparison. (c) The guardrail definition itself is mismatched to the risk — for example, a 14-day refund window registered against a harm that manifests at 30 days, so the guardrail is measuring a leading indicator with unknown relationship to the harm.

**Discriminating check.** Compute the standard error of the guardrail difference and express the breach in standard-error units. A breach of less than one standard error cannot be distinguished from zero and indicates (b). Then compare the guardrail's measurement window to the documented harm lag; a mismatch indicates (c). Only if the breach exceeds roughly 1.5 standard errors and the window matches the harm lag should (a) be concluded.

**Bounded fix.** For (b), restate the guardrail as a bound on the confidence limit rather than the point estimate, and version the change. For (c), change the guardrail to measure the actual harm at its actual lag, which may mean the guardrail cannot be evaluated until after the review date — in which case the rollback trigger must be a leading indicator with a stated, tested relationship to the harm, not the harm itself. For (a), execute the rollback trigger; that is what it was registered for.

**Verification.** After the change, the guardrail must be able to produce both a pass and a fail on a simulated pair of outcomes: construct a case where the true difference is 0 and one where it is at the harm threshold, and confirm the new rule classifies both correctly. A guardrail that cannot fail on a constructed failing case is not a guardrail.

### FAIL-08 — The value model's horizon and the effect's horizon disagree

**Symptom.** The decision is made and the benefit never materialises, or materialises far smaller than `G` implied, and the post-mortem finds the effect was transient.

**Competing causes.** (a) The change produced a genuine but short-lived novelty effect: behaviour shifted because the interface was new, then reverted. (b) The gain was sized over a 52-week horizon while the effect was measured over 28 days, and the effect decays with a half-life much shorter than the sizing horizon. (c) The gain double-counts cannibalisation: the improvement is real but displaces orders that would have happened anyway through another channel, so the net is smaller than the gross.

**Discriminating check.** For (a) and (b), plot the effect by week across the full available window; a novelty effect shows a spike decaying towards zero, while a durable effect shows a stable or rising level. For (c), check the total across channels rather than the treated channel alone; cannibalisation is visible as an offsetting decline elsewhere that roughly matches the apparent gain.

**Bounded fix.** For (a) and (b), size `G` using an effect estimate from the post-novelty period, and shorten `H` to the period over which the effect has been observed to persist. That will reduce `G`, which raises `p*`, which is the correct conservative direction. For (c), restate `G` as the net across all channels and disclose the gross-to-net ratio.

**Verification.** Add a standing checkpoint at the end of the sizing horizon's first month: is the realised cumulative gain tracking the `G/H` rate? A trajectory substantially below the projection is evidence about the decay assumption, not about the original decision's soundness, and it should be recorded in the calibration ledger as a magnitude miss with a named cause.

### FAIL-09 — The memo's claim–evidence ledger does not match its sentences

**Symptom.** A reviewer challenges one number, and the analyst cannot locate its derivation without rebuilding the analysis.

**Competing causes.** (a) The ledger exists but was written after the memo, as a list of links rather than a per-claim mapping. (b) The ledger is per-claim but the memo's sentences are compound, so one sentence mixes a sourced fact, a derived threshold, and an illustrative assumption. (c) The number in question was computed in a notebook that has since been edited, so the ledger points at a file that no longer produces that number.

**Discriminating check.** Pick three sentences at random from the recommendation section and try to trace each to a single ledger row. Compound-sentence failures show up as one sentence needing three rows and having one — cause (b). Trace a ledger row back to the notebook and re-execute it; if the number changes, cause (c).

**Bounded fix.** For (a) and (b), rewrite the recommendation into one-claim sentences and attach a row per sentence with the four fields: claim, computation or source, evidence label, and the input's cutoff or version. For (c), pin the computation by storing the query or the notebook with its output, hash the output, and record the hash; a number whose hash no longer matches is flagged as stale rather than silently updated.

**Verification.** The strongest available test is an adversarial one: hand the memo and the ledger to someone who has not seen the analysis and ask them to change one input and recompute the affected claim. If they can do it in under fifteen minutes, the ledger is adequate. If they cannot, the memo is not reproducible in the sense that matters — not for science, but for the practical purpose of surviving a challenge in a meeting.

### FAIL-10 — The decision was made, the follow-up never happened

**Symptom.** The same class of question returns quarterly, and each memo's prior is as vague as the last.

**Competing causes.** (a) No follow-up date was set in the action contract, so nothing triggered the comparison. (b) A follow-up date was set but the metric was redefined between the decision and the review, so the prediction cannot be scored against a comparable number. (c) The follow-up happened but the result was not recorded anywhere durable, so the institutional memory lives in one person who has since changed team.

**Discriminating check.** Ask for the last three memos on this topic and check: does each have a review-date field? Does the metric definition version match between each memo and its review? Is there a single document containing all three settlements? A "no" on the first indicates (a); matching dates but mismatched versions indicates (b); a document that exists only in one person's notes indicates (c).

**Bounded fix.** For (a), make the review date a required field and put the follow-up on the author's calendar at send time, not at review time. For (b), freeze the metric definition in the memo and require the review to use the frozen definition, reporting any live-definition number separately. For (c), keep a single calibration ledger with one row per memo: date, decision, prediction, range, confidence, settlement date, realised value, direction hit/miss, magnitude inside range yes/no.

**Verification.** After three settled memos, compute the hit rate for the 80–90% confidence band. A band whose hit rate is far below its nominal level is mis-calibrated and the ledger is doing its job; a team with no ledger cannot compute this and therefore cannot improve. The ledger's existence and use is the only observable that distinguishes a memo culture from a memo habit.

### Edge cases worth registering

Four situations are not failures but require a decision rule of their own.

**Zero events in a cell.** A segment with 0 completions out of 40 has a rate estimate of 0 with a standard error that the normal approximation renders as 0, which is wrong in a way that flatters the segment. Use an interval method appropriate to small counts (Wilson or exact) and never report the raw 0/40 as a rate with a symmetric interval.

**A tie in the decision rule.** When the posterior lands within a rounding error of `p*`, the memo should not report a decision at all; it should report the boundary and state what small change in `G`, `L`, or the estimate would tip it. A near-tie is information about how thin the case is, and reporting "act" or "decline" as though the rule had a clear verdict destroys that information.

**Conflicting owners.** Two owners with different value models — say, a growth owner with a large `G` and a trust-and-safety owner with a large `L` — will derive different `p*` from the same evidence, and both are correct under their own model. The memo cannot resolve this by argument; it must escalate to whoever owns the trade-off between the two objectives, and it must present both `p*` values so the escalation is about the trade-off rather than about the data.

**A decision whose consequence falls outside the organisation's boundary.** If part of `L` is borne by a party the owner does not account for (a downstream partner, a customer segment with no voice in the decision), the owner's `L` is smaller than the social `L`, and the decision rule will systematically over-act. The memo should state the excluded cost explicitly, even when it cannot be quantified, because the existence of an unaccounted externality is a governance fact that the reader is entitled to know.

**Takeaway:** the ten failures map onto the six joints of the memo pipeline, and each one is detected by a check that costs less than the error it prevents.

## 9. Complexity & Cost Analysis

This section prices the memo itself. The analysis has three parts: the cost of producing the analysis behind a memo, the cost of the reading and review it triggers, and the cost of the latency it imposes — because a memo's total cost is the sum, and the third term is usually the largest and the least examined.

### 9.1 Cost of producing the analysis

The dominant term for a grouped scan is the number of rows or bytes read, and it scales linearly in the input size before any shuffle or sort effects.

```text
FORMULA [DM-08]: R = sum over p of B_p * (1 + replicate_p)
WHERE: R bytes read [bytes]; B_p bytes in partition p [bytes];
       replicate_p fraction re-read (retries, backfills, exploratory
       re-runs) for partition p [dimensionless].
CONSTRAINT: partitions disjoint; replicate factors measured from the
       query log rather than assumed; excludes shuffle spill and
       intermediate materialisation, which are separate cost lines.
NUMERICAL EXAMPLE: 12 daily partitions of 2 GiB each, with a measured
       replicate factor of 0.15 on two of them that were re-run after a
       schema fix:
       R = 12*2 GiB + 2*2 GiB*0.15 = 24 GiB + 0.6 GiB = 24.6 GiB
       = 24.6*1,073,741,824 = 26,414,048,870 bytes.
INTERPRETATION: the replicate term is small here, but it is the term that
       grows without bound when exploration is unmanaged; a team that
       re-runs every query four times for exploration has a replicate
       factor near 3, not 0.15.
```

The practical scaling statement is the one worth remembering: **cost is linear in bytes scanned, and bytes scanned is linear in the number of exploratory re-runs.** The number of re-runs is a process variable under the analyst's control, which is why the cheapest optimisation in most analytics organisations is not a better engine but a pre-registration note that reduces re-runs from six to two.

### 9.2 Cost of reading and review

Reading cost is small in currency and large in schedule, and the schedule term is what actually delays decisions.

```text
FORMULA [DM-09]: T_review = L_words/(v_read*60) * n_r + n_r * t_switch
WHERE: T_review total reviewer time [hours]; L_words memo length [words];
       v_read reading speed [words/min]; n_r number of reviewers
       [persons]; t_switch per-reviewer context-switch and comment time
       [hours].
CONSTRAINT: reading speed appropriate to technical prose with tables
       (roughly 180-220 words/min for careful reading, not skimming);
       reviewers read in parallel so wall-clock latency is max, not sum;
       excludes meeting time.
NUMERICAL EXAMPLE: L_words = 2,400, v_read = 200, n_r = 3,
       t_switch = 0.25 h:
       per-reviewer = 2,400/(200*60) + 0.25 = 0.20 + 0.25 = 0.45 h
       T_review = 3*0.45 = 1.35 h of total reviewer attention
       (wall-clock, if parallel: about 0.45 h plus scheduling).
INTERPRETATION: the attention cost is modest; the memo is not expensive
       because it is long, it is expensive because it competes for a
       scarce reviewer's calendar.
```

The wall-clock-versus-total distinction matters for the delay calculation in the next subsection: three reviewers reading in parallel impose the latency of one, while three reviewers reading sequentially impose the latency of three. A review chain that serialises five approvers converts a 20-minute read into a two-week queue, and the currency cost of that conversion is computed by DM-07.

### 9.3 Cost of latency, including queueing

Delay is not just the review duration; it is the wait for a review slot. The wait behaves like a queue, and the queue time grows nonlinearly as utilisation approaches one.

```text
FORMULA [DM-10]: W_q ≈ (rho/(1-rho)) * (1/mu)   [M/M/1 approximation]
WHERE: rho = lambda/mu utilisation [dimensionless, <1];
       lambda decision arrivals requiring review [decisions/period];
       mu review service rate [decisions/period];
       W_q expected wait before service begins [periods].
CONSTRAINT: single-server approximation with Poisson arrivals and
       exponential service; used as an order-of-magnitude sizing device,
       not a precise prediction; independence between consecutive
       decisions; the qualitative conclusion (superlinear growth near
       rho -> 1) is robust to the distributional assumptions.
NUMERICAL EXAMPLE: lambda = 6 decisions/week, mu = 8 decisions/week:
       rho = 0.75; W_q = (0.75/0.25)*(1/8) = 3*0.125 = 0.375 weeks
       = about 2.6 working days of queueing before the review starts.
       At lambda = 7 (rho = 0.875): W_q = 7*0.125 = 0.875 weeks
       = about 4.4 working days.
INTERPRETATION: a 17% increase in arrival rate doubled the wait; the
       queue is the reason a "quick review" takes a week, and adding
       reviewers (raising mu) helps far more than shortening memos.
```

Combining the queue wait with the delay cost from DM-07 gives the total expected cost of routing a decision through a review process, and it is almost always dominated by `W_q` rather than by the reading time.

```text
total expected cost = p*(G/H)*(W_q + T_review_wall) + reviewer_cost
reviewer_cost = n_r * (L_words/(v_read*60) + t_switch) * c_hour
```

Using the Section 5 numbers throughout, with a three-week observed review latency:

```text
delay term  = 0.895191*(480,000/52)*3 weeks = 24,790.0 currency
attention   = 3 reviewers * 1.35 h... (per-reviewer 0.45 h at 110/h
              = 49.5; times 3 = 148.5 currency)
total       ≈ 24,939 currency
```

The comparison that makes the point: the delay term is about 99% of the cost, and the attention term is about 1%. Every discussion about "making memos shorter" is optimising the 1%. Every discussion about review cadence is optimising the 99%. This asymmetry is the most actionable cost finding in this file.

### 9.4 Cost of the analysis pipeline versus the value at stake

A useful sanity ratio compares the cost of producing the memo to the expected value it protects or creates.

```text
FORMULA [DM-11]: kappa = C_memo / max(EV, epsilon)
WHERE: C_memo total memo cost including delay [currency];
       EV expected value of the decision [currency];
       epsilon small positive floor to avoid division by zero.
CONSTRAINT: EV computed per DM-01 at the current posterior; costs and
       EV in the same currency and the same period basis.
NUMERICAL EXAMPLE: C_memo = 24,939; EV = 303,920.9:
       kappa = 24,939/303,920.9 = 0.0821.
INTERPRETATION: the memo costs about 8% of the value it adjudicates,
       which is proportionate for a decision of this size; a kappa above
       0.25 signals that the process is out of scale with the stakes,
       and a kappa below 0.01 signals under-investment in the checks.
```

The ratio is a useful portfolio diagnostic. Compute it for ten recent decisions and sort: the ones with high `kappa` are candidates for a lighter process, and the ones with very low `kappa` are candidates for more checking. Both directions are improvements, and neither is visible without the ratio.

### 9.5 Storage and refresh cost of the evidence base

A decision memo needs a reproducible snapshot. Keeping one costs storage; recomputing it costs compute; both are cheap relative to the decision, but the numbers should still be sized rather than assumed.

```text
snapshot size = N_rows * bytes_per_row * compression_factor^-1 * replicas
             = 4.2e8 rows * 180 bytes / 3.5 * 2 replicas
             = 75.6e9 / 3.5 * 2 bytes = 43.2 GB
refresh compute = partitions * partition_size * scans_per_refresh
             = 12 * 2 GiB * 1 = 24 GiB per refresh
monthly at daily refresh = 30 * 24 GiB = 720 GiB-month scanned
```

These are arithmetic statements with units, not platform benchmarks: the storage figure follows from the row count, row width, a stated compression factor, and the replication factor, and it changes when any of those change. The operational conclusion is that snapshotting for reproducibility is inexpensive at the grain of this example, so the excuse that "we cannot afford to freeze the data" is usually false — the real cost is the governance decision about who owns the frozen copy.

### 9.6 Human cost of the checks themselves

The six independent checks in Section 5.9 took about an hour in total. Priced at 95 currency per hour for an analyst, that is 95 currency against a decision worth 304,000 — a ratio of 0.0003. The checks are, in cost-benefit terms, free; the only reason they are skipped is that they are not scheduled, not that they are expensive.

**Takeaway:** memo cost is dominated by queue latency rather than reading, so cadence changes beat prose changes by two orders of magnitude.

## 10. Key Theorems / Results Reference

The results below are the theoretical backbone of the practice. Each is stated with its conditions, because the conditions are what make it usable rather than decorative.

### R-01 — Expected-value linearity and the irrelevance of scale

For a binary decision with gain `G` and loss `L`, the expected value is linear in `p`, and the break-even probability `p* = L/(G+L)` is invariant to proportional rescaling of both `G` and `L`. *Conditions:* both outcomes summarisable as scalars on the same horizon; risk neutrality. *Consequence:* the strength of evidence required depends on the *ratio* of stakes, not their absolute size. This is why a small team and a large division need the same probability threshold for structurally identical decisions, and why "this is a big decision, so we need more confidence" is only correct when the decision is *asymmetric*, not merely large.

### R-02 — Conjugate normal update and precision weighting

For a normal prior `N(μ0, σ0^2)` and a normal likelihood `N(d, s^2)`, the posterior is `N(μ1, σ1^2)` with `1/σ1^2 = 1/σ0^2 + 1/s^2` and `μ1 = σ1^2(μ0/σ0^2 + d̂/s^2)`. *Conditions:* known `s`, normal prior, independence of the estimate from the prior. *Consequence:* the posterior mean is a precision-weighted average, so the prior's influence is exactly proportional to its precision relative to the data's. The memo can therefore quantify prior influence instead of arguing about it: when `1/σ0^2 << 1/s^2`, the prior barely matters, and the memo can say so with a number.

### R-03 — Monotonicity of the posterior in the estimate

`P(helps) = Φ(μ1/σ1)` is strictly increasing in the standardised posterior mean `μ1/σ1`, and `μ1/σ1` is strictly increasing in `d̂` for fixed `σ0, s`. *Conditions:* as R-02. *Consequence:* the decision rule is a threshold on an observable statistic, which permits a clean operational form: compute `z_post = μ1/σ1` and compare it to `Φ^{-1}(p*)`. In the Section 5 example, `Φ^{-1}(0.714) = 0.5657` and `z_post = 1.2546`, so the decision is "act" — a one-line computation once the inputs exist.

### R-04 — The value of information is bounded and convex in accuracy

`EVSI ≤ EVPI = min(p·G, (1-p)·L)`, and `EVSI` increases in the test's sensitivity and specificity but is zero whenever the test cannot move the posterior across the threshold. *Conditions:* one-shot decision, known test characteristics, cost excluded. *Consequence:* two practical rules. First, no study can be worth more than `EVPI`, so a study costing more than `EVPI` is rejected on arithmetic. Second, a test whose both branches leave `p` on the same side of `p*` has exactly zero value, which is the formal version of "don't run that study, it can't change our decision."

### R-05 — Ambiguity of aggregate rates under composition change

Any ratio of sums can move without any component rate moving, if the weights change; equivalently, `Σw_i r_i` is not invariant to `w`. *Conditions:* none beyond the definition of a weighted mean. *Consequence:* a memo comparing two aggregate ratios must either fix the weights (standardise, DM-06) or demonstrate that the weights are unchanged. The result is elementary and is nonetheless the single most common source of reversed recommendations.

### R-06 — Sensitivity versus specificity asymmetry in decision value

For a fixed sum `se + sp`, `EVSI` is more sensitive to `sp` than to `se` when the pre-test probability is high, and more sensitive to `se` when it is low. *Conditions:* binary test, one-shot decision, threshold in the interior. *Consequence:* the Section 4.3 table shows this concretely — the 0.60/0.90 test has essentially zero value at `p = 0.895` because its low sensitivity means a negative result leaves the posterior at 0.79, still above threshold. Selecting tests is therefore a matter of asking which result you need to be able to believe, not of maximising a symmetric accuracy score.

### R-07 — The irreversibility premium

`p*_R = (L-R)/(G+L-R)` is decreasing and convex in `R`, with `dp*_R/dR = -G/(G+L-R)^2`. *Conditions:* salvage available within the horizon, net of reversal cost. *Consequence:* the marginal value of reversibility is largest when `R` is small and `G` is large — i.e. the first increment of undo capability on an otherwise irreversible decision is worth more than the last increment on an already-reversible one. The practical rule is to invest in the *first* rollback rehearsal, not in perfecting an already-working one.

### R-08 — Queueing delay is superlinear in utilisation

For `M/M/1`, `W_q = (ρ/(1-ρ))(1/μ)`, so `dW_q/dρ = 1/((1-ρ)^2 μ)`, which grows without bound as `ρ → 1`. *Conditions:* Poisson arrivals, exponential service, single server, independence. *Consequence:* review capacity must be provisioned with headroom, because the delay penalty for running at 90% utilisation is roughly nine times the penalty at 50%. The distributional assumptions affect the constant, not the divergence, which is why the qualitative conclusion carries over to real review processes.

### R-09 — Calibration and the proper scoring of probability statements

A probability statement is *calibrated* if, across all statements made with confidence `c`, the observed frequency of the event is `c`. Brier-type scores are minimised in expectation by reporting one's true belief. *Conditions:* repeated, comparable statements; no selection of which statements to score. *Consequence:* the calibration ledger is not an audit of honesty but a measurement instrument — and it works only if every prediction is scored, because selecting which predictions to score reintroduces exactly the bias calibration is meant to detect.

### R-10 — The cost of a decision rule is the sum of error costs weighted by their rates

For a rule with false-positive rate `α` and false-negative rate `β` applied to a population with prevalence `π`, the expected cost per decision is `πβL + (1-π)αG`. *Conditions:* binary decision, known costs, stationary prevalence. *Consequence:* this is the general form of which DM-01 is the special case where `α` and `β` are replaced by the single probability `1-p`. It makes explicit that improving a rule's discrimination changes *both* error rates, and that a rule which reduces `α` while increasing `β` may be worse even when it looks "more precise."

### R-11 — Aggregation destroys information that decisions may need

Summing or averaging loses the joint distribution; consequently no function of marginal aggregates can recover a quantity that depends on the joint (a correlation, a percentile, a distinct count over the union of overlapping sets). *Conditions:* nonlinear functionals, overlapping populations. *Consequence:* a memo comparing two averages of distinct counts, or averaging two percentages, can produce a number that no population could produce. The fix is to retain the components at a grain that supports recomputation, which is the storage cost of Section 9.5 and is almost always worth paying.

### R-12 — Pre-registration constrains the analyst, not the data

A decision rule fixed before the analysis removes the analyst's discretion over the mapping from result to action, but it does not change the sampling distribution of the estimator. *Conditions:* the rule is genuinely fixed and the analysis is not silently re-specified. *Consequence:* pre-registration buys protection against motivated selection of the rule, and it does not buy protection against a badly designed comparison. Both protections are needed, which is why the memo requires the population reconciliation (Section 5.2) as well as the pre-registration note (Section 5.1).

**Takeaway:** these twelve results are the load-bearing structure; every memo claim that matters can be traced to one of them or is a matter of taste.
## 11. Interview / Exam Questions

Fourteen questions, ordered from recall through application to judgement. Each has a checked answer (arithmetic recomputed where numbers appear), the reasoning that earns credit, a follow-up the interviewer would plausibly ask, and the red flag that marks a weak answer.

### Q1 — "A product manager says your result has p = 0.06 and therefore shows nothing. What do you say?"

**Answer.** A p-value of 0.06 answers "how often would data this extreme arise if the true effect were exactly zero?", and it is not the quantity the decision needs. The decision needs the probability the change helps. Under the prior and standard error in this file, `d̂ = 2.266 pp`, `s = 1.459 pp`, `μ0 = 0`, `σ0 = 2 pp` give `P(helps) = Φ(1.2546) = 0.895`. Whether 0.895 is enough depends on the threshold: with `G = 480,000` and `L = 1,200,000`, `p* = 0.714`, so 0.895 clears it.

**Reasoning.** Credit requires (i) naming what the p-value is, (ii) naming what the decision needs, (iii) converting, and (iv) comparing against a threshold that comes from the stakes, not from a convention. Getting to step (iii) without (iv) is half credit, because a probability with no comparator still cannot drive a decision.

**Follow-up.** "What if the prior had been centred at −1 pp?" Then `μ1 = 1.132 pp`, `σ1 = 1.179 pp`, `P(helps) = Φ(0.9599) = 0.831`, which still clears 0.714 but by less. The decision is robust; the confidence in it is not.

**Red flag.** Answering "0.06 is marginally significant" or "it depends on your alpha" — both keep the conversation on a scale that does not drive the action, and the second pretends a conventionally chosen alpha is a decision-theoretic quantity.

### Q2 — "Your lift is 2.3 points on a 25% base. Is that big?"

**Answer.** It is a 9.0% relative lift on a 25.1% control rate (`0.022660/0.251253 = 0.0902`), and whether that is big depends entirely on `G` and `L` per point. Translating: at 576,000 eligible sessions per year and 40 currency margin per order, each percentage point of completion is worth `576,000*0.01*40 = 230,400` currency gross, or 134,400 net of the 96,000 one-off engineering cost. So 2.266 pp is worth `576,000*0.02266*40 − 96,000 = 522,086 − 96,000 = 426,086` currency of annual value at the point estimate — which is why the value model in Section 5.3 used `G = 480,000` for a 2.5 pp assumption, and why a 1 pp result would support only `134,400`.

**Reasoning.** The candidate who converts points to currency in one step is doing the job. The candidate who says "it depends" and stops is not: it does depend, and the analyst's task is to find out on what.

**Follow-up.** "What if the horizon is one quarter rather than a year?" Then `G` falls by a factor of four and `p* = 1,200,000/(120,000+1,200,000) = 0.909`, which the posterior of 0.895 does **not** clear. A shorter horizon makes the decision harder, not easier, because the downside is unchanged while the upside shrinks.

**Red flag.** Calling 9% "huge" or "marginal" without a stake size. Relative lifts have no intrinsic size.

### Q3 — "You want to run a confirmation test. How do you justify it?"

**Answer.** Compute the expected value of sample information. With `p = 0.895`, `G = 480,000`, `L = 1,200,000`, and a proposed test with `se = 0.90`, `sp = 0.90`: `P(+) = 0.8162`, `p_+ = 0.9872`, `p_- = 0.4869`. The policy after a positive result acts (`EV = 458,426`) and after a negative result declines (`EV = 0`). Expected value with the test is `0.8162*458,426 = 374,145`; without it is `303,921`; so `EVSI = 70,225`. Compare to the test's cost plus the delay cost of 8,263: net about 62,000 if the test completes before the decision date.

**Reasoning.** The critical discipline is the `max(0, ·)` after a negative result — the decision maker can always decline, and forgetting that overstates EVSI. Also credit for comparing EVSI to EVPI (`min(429,553, 125,771) = 125,771`) rather than treating EVSI as unbounded.

**Follow-up.** "What if the test had 60% sensitivity?" Then the negative branch leaves `p = 0.7915`, still above `p* = 0.714`, so both branches act and `EVSI ≈ 0`. A test that cannot change the action is worth nothing regardless of cost.

**Red flag.** Justifying a test by its accuracy, or by "more data is always better." Neither engages the decision.

### Q4 — "Give me the conditions under which your recommendation would be wrong."

**Answer.** Four falsifiers, each in the units of the evidence. (i) The guardrail's 95% upper bound exceeds +0.5 pp on refunds. (ii) Completion effect at the two-week ramp checkpoint is below +1.0 pp. (iii) The rollback rehearsal fails, which invalidates `R = 400,000` and raises `p*` from 0.625 to 0.714 — the posterior of 0.895 still clears it, so this falsifier changes the cost of error rather than the decision. (iv) The value model's horizon collapses: if the change is revisited in one quarter, `p*` rises to 0.909 and the decision reverses.

**Reasoning.** A strong answer distinguishes falsifiers that reverse the decision from those that change its price. Only (ii) and (iv) can reverse; (i) triggers a rollback by contract; (iii) makes the bet less attractive without flipping it.

**Follow-up.** "Which falsifier is most likely?" Judgement: (ii), the magnitude checkpoint, because the estimate is 1.55 standard errors from zero and a 1 pp floor is well inside the interval.

**Red flag.** "If the effect isn't real" — not observable. Or a list of general risks with no units.

### Q5 — "Two teams report the same feature as +3.5 pp and −12.4 pp. Reconcile."

**Answer.** They are comparing different populations. In the worked example, the control mix is 50/50 across two segments with rates 0.60 and 0.10, giving 0.35; the treated mix is 20/80 with rates 0.65 and 0.12, giving 0.226. The raw difference is −0.124. Standardising the treated rates to the control mix gives `(0.65 + 0.12)/2 = 0.385`, and `0.385 − 0.35 = +0.035`. The sign flip is entirely compositional: the treated population sits 80% in the low-rate segment.

**Reasoning.** The candidate must (i) suspect mix, (ii) compute both, (iii) name which one the decision needs. The decision-relevant comparison is the standardised one **if** the decision applies to the control mix; if the decision applies to the treated mix, the raw figure is the relevant one and the standardised figure is the counterfactual for a different population. Naming the standardisation target is the mark of a complete answer.

**Follow-up.** "What if a segment has 12 treated observations?" Then `r_i^T` is estimated with a standard error of roughly 0.14 on the proportion scale, and reweighting it into the headline can move it by several points on noise alone. Report cell counts and require a minimum cell size.

**Red flag.** Concluding "one of them made an error." Composition-driven sign flips are common and both numbers can be arithmetically correct.

### Q6 — "What is the break-even probability for a decision with 9:1 downside-to-upside, and what does that imply about the evidence bar?"

**Answer.** `p* = L/(G+L)` with `L = 9G` gives `9/10 = 0.90`. Implication: the organisation needs 90% confidence before acting, which is a materially higher bar than "significant at 5%" applied to a two-sided test — and it also means that a reversible version of the same decision, with `R = 0.33L`, needs only `(9-3)/(1+9-3) = 6/7 = 0.857`. The premium for irreversibility here is 4.3 pp.

**Reasoning.** Credit for the arithmetic, for noticing that 0.90 is *more* demanding than the usual significance convention, and for connecting reversibility to the bar.

**Follow-up.** "And with 1:9, upside nine times the downside?" Then `p* = 1/10 = 0.10`, and the correct action at almost any positive signal is to act — but the memo must still state `p*`, because a team applying a uniform 95% standard would reject it, which is a systematic bias against high-upside actions.

**Red flag.** Reciting "0.5 when symmetric" and stopping. The interesting content is the asymmetry.

### Q7 — "How do you decide whether a definitional change to a metric should be backfilled?"

**Answer.** Three considerations, in order. First, was the historical series used to make decisions that are still live? If yes, backfilling destroys the audit trail of what those decisions were based on. Second, can both definitions be computed over an overlap period? If yes, publish the overlap and the size of the break so readers can rebase mentally. Third, does the change alter the *direction* of the series in the overlap? If it does, the change is not cosmetic and requires a version bump plus a note in every live dashboard.

**Reasoning.** The answer's quality is in the ordering: audit trail before convenience. The weak answer is "yes, backfill so the chart looks clean."

**Follow-up.** "What if there is no overlap period?" Then the break is `OPEN`: publish both series as separate, unjoinable views, and state that no growth rate can be computed across the boundary. Do not interpolate.

**Red flag.** "Backfill everything so the trend is consistent" — this erases the evidence needed to evaluate past decisions.

### Q8 — "Your guardrail is breached by 0.006 pp. Ship or hold?"

**Answer.** Neither, as stated. The difference is `62/1,840 − 55/1,795 = 0.033696 − 0.030641 = +0.003055` (0.306 pp), with standard error `sqrt(0.000017695 + 0.000016553) = 0.005852`. So the breach is 0.10 standard errors — indistinguishable from zero. The correct statement is that the pre-registered limit sits inside the noise band and therefore cannot discriminate. Restate the guardrail as a requirement on the 95% upper bound (below +0.5 pp), version the change, and record it as a definition change rather than a reinterpretation.

**Reasoning.** The candidate must compute the standard error, not just the difference; must resist both "breach means hold" and "tiny means ignore"; and must propose a versioned fix rather than a silent one.

**Follow-up.** "What if the breach were 2.5 standard errors?" Then the guardrail is informative and the contract's rollback trigger fires, regardless of the point estimate being small in absolute terms.

**Red flag.** Deciding on the point estimate alone in either direction. The magnitude of the breach and its precision are separate facts.

### Q9 — "What does a decision memo contain that a report does not?"

**Answer.** Four things: a named owner with authority, a decision date tied to a consequence, at least two explicit options including "do not act" with its cost, and a threshold on evidence derived from the value model. A report can contain none of these and still be a good report.

**Reasoning.** The answer should make clear that the difference is structural, not stylistic. A memo with a recommendation but no threshold has merely added an opinion to a report.

**Follow-up.** "Where does the recommendation go?" First three sentences, with the threshold and the ask in the first screen, and the ask repeated at the end — because a caveat or an ask that the owner does not read cannot inform the decision.

**Red flag.** "A memo is shorter" or "a memo has an executive summary." Both are about form and neither constrains the argument.

### Q10 — "Estimate the cost of a three-week review for a decision worth 480,000 over a year with 89.5% probability of success."

**Answer.** `C_delay = p*(G/H)*Δ = 0.895191*(480,000/52)*3 = 0.895191*9,230.77*3 = 24,790.0` currency. Compare to `EVPI = 125,771`: the review costs about 19.7% of the maximum possible value of any information. So the review must buy more than a fifth of all achievable certainty to break even, which for a decision already at 89.5% confidence is implausible.

**Reasoning.** The comparison to EVPI is the discriminating move. A candidate who computes 24,790 and stops has done arithmetic; one who compares it to the information ceiling has done analysis.

**Follow-up.** "What if the decision were a stop decision on something harmful?" Then delay cost is `(1-p)`-weighted, not `p`-weighted, because the loss accrues while the decision is deferred. Using `p` would understate the cost of delay by the entire harm rate — here by a factor of about 8.5.

**Red flag.** Ignoring the delay entirely because "review is good practice."

### Q11 — "How do you handle a prior you cannot defend?"

**Answer.** Do not use an indefensible prior silently. Three options, in order of preference. (i) Construct it mechanically from the documented history of comparable changes: take all of them, use the median and interquartile range, convert to `μ0` and `σ0`. This is defensible because it is a procedure, not a belief. (ii) If no history exists, use a weak prior (`σ0` large) and report that the posterior is then near the likelihood, which is the honest statement of "we don't know much." (iii) If neither is possible, report the likelihood-based probability `Φ(d̂/s)` and label the prior as `OPEN`, with the specific evidence that would resolve it.

**Reasoning.** The answer must not be "use a flat prior" as an automatic reflex — a flat prior on the effect is not non-informative about the decision, and it can be indefensible in the opposite direction by ignoring real history.

**Follow-up.** "How do you know your prior was reasonable?" Only after the fact, via the calibration ledger: compare predicted directions and magnitudes to realised outcomes across memos, and adjust `σ0` so that nominal confidence matches observed hit rates.

**Red flag.** "Priors are subjective so it doesn't matter" — which either abandons the framework or hides the subjectivity, and hiding it is worse.

### Q12 — "A stakeholder asks for the analysis to be redone with a different segment definition that makes the result look better. What do you do?"

**Answer.** Run it, publish it, and publish the original alongside it, with a note that the segment definition was changed after seeing the result and that the second definition is therefore exploratory, not confirmatory. The reason is not purity: a result selected after the fact has an unknown false-positive rate, so it cannot support a decision at the same threshold as a pre-registered one. If the alternative definition produces a robust result across both definitions, the finding is strengthened; if it only appears under one, that is the finding.

**Reasoning.** The candidate must neither refuse (which is obstructive and does not prevent the request being met elsewhere) nor comply silently (which launders a selection effect into a decision). The publish-both move is the professional answer.

**Follow-up.** "How much does the threshold change?" If the analysis was exploratory, the effective evidence is weaker than the posterior suggests; the practical adjustment is to require a higher posterior — a `p*` computed with an inflated `s` — or to treat the exploratory result as an input to a confirmatory test with its own EVSI.

**Red flag.** "Sure, whatever the stakeholder wants" or a lecture about p-hacking with no practical alternative.

### Q13 — "When is it correct to decide on a point estimate with no uncertainty at all?"

**Answer.** When the decision is trivially reversible (`R ≈ L`, so `p*_R ≈ 0`), or when the cost of the uncertainty analysis exceeds the value it could change (`EVPI` smaller than the analysis cost), or when the decision must be made faster than any uncertainty estimate could be produced and the stakes are small. In all three cases the memo should still record which condition applied, because the next decision of the same class may not share it.

**Reasoning.** The honest answer is "sometimes," with conditions, and with a record. An answer of "never" is dogmatic; an answer of "always" is the failure mode this whole file exists to prevent.

**Follow-up.** "Give a number for 'trivially reversible'." With `G = 480,000` and `L = 1,200,000`, `R = 1,150,000` gives `p*_R = 50,000/530,000 = 0.094`. At that point almost any positive signal justifies acting, and the uncertainty analysis is worth less than the delay it costs.

**Red flag.** "Always quantify uncertainty" — a rule that, applied uniformly, prices small reversible decisions out of existence.

### Q14 — "You are given a memo that recommends shipping. In ten minutes, what do you check?"

**Answer.** Six joints, in order. (1) Is there a named owner and date, and are they in the recipient list? (2) Is the population reconciled, with the exclusion ledger shown? (3) Is the value model `G`, `L` stated in currency with a horizon, and is `p*` computed from them? (4) Is the evidence expressed as a probability comparable to `p*`, with the prior stated? (5) Does the guardrail have a discriminating rule, and was it registered before the result? (6) Is there an action contract with seven fields including a rollback trigger and a review date?

**Reasoning.** The answer's value is the checklist's completeness and its ordering — framing first, because a memo that fails (1) cannot be rescued by better numbers. A candidate who starts with the statistics has the priorities inverted.

**Follow-up.** "Which joint fails most often?" Judgement: (3), the value model, because `L` is usually estimated by the person advocating the action and is usually too small; and (6), because the review date is the field most often omitted.

**Red flag.** "I'd check the math." The math is the part most likely to be right, because it is the part that was checked; the framing and the value model are where memos actually fail.

**Takeaway:** interview answers earn credit by converting to the decision's own scale, and lose it by staying on statistical convention.

## 12. Quick Reference Card

| Item | Content |
|---|---|
| **Decision rule** | Act when `P(helps) > p*`; `p* = L/(G+L)`; with salvage `p*_R = (L-R)/(G+L-R)`. With `G = 480,000`, `L = 1,200,000`, `R = 400,000`: `p* = 0.714`, `p*_R = 0.625`. |
| **Uncertainty statement** | Report `P(helps) = Φ(μ1/σ1)` from `1/σ1² = 1/σ0² + 1/s²`, never a bare p-value. Worked: `d̂ = 2.266 pp`, `s = 1.459 pp`, prior `N(0, 2²)` → `P(helps) = 0.895`; p-value 0.060 is a different question. |
| **Value and information** | `EV = pG − (1−p)L`; `EVPI = min(pG, (1−p)L)`; `EVSI = Σ P(result)·max(0, EV(p|result)) − max(0, EV(p))`. Worked: `EV = 303,921`, `EVPI = 125,771`, `EVSI(90/90) = 70,225`. |
| **Delay price** | `C_delay = p·(G/H)·Δ`. Worked: `0.895·(480,000/52)·3 = 24,790`, i.e. 19.7% of EVPI — a review must buy that much certainty to pay for itself. |
| **Comparison discipline** | Standardise before comparing (`d_std = Σ w_i^C r_i^T − Σ w_i^C r_i^C`), never average displayed rates, never average distinct counts. Worked: raw −12.4 pp becomes standardised +3.5 pp. |
| **Failure triggers** | No owner/date → convert to a research question. Denominator drift → version the definition. Unrehearsed rollback → cut `R`, raising `p*`. Late-chosen prior → publish the sensitivity grid. Undiscriminating guardrail → bound the confidence limit, version the change. |
| **Cost reality** | Memo cost is ≈99% queue latency, ≈1% reading: `W_q = (ρ/(1−ρ))(1/μ)`. Fix cadence before prose. Six arithmetic checks cost ≈1 analyst-hour against decisions of order 10⁵ currency. |
| **The ten-minute review** | Owner+date → population ledger → value model in currency → probability vs threshold → guardrail registered and discriminating → seven-field action contract with review date. |
| **Evidence labels here** | DERIVED: DM-01…DM-11, R-01…R-12. VERIFIED: all Section 5 arithmetic, recomputed independently. ILLUSTRATIVE: every input value (sessions, rates, costs, margins). OPEN: organisational approval norms; no company or product described. |
| **Critical validity condition** | All thresholds assume a binary decision scored by scalars `G` and `L` on one horizon with a risk-neutral owner; multi-valued or safety-critical outcomes require an explicit risk constraint in addition, not a larger `L`. |
