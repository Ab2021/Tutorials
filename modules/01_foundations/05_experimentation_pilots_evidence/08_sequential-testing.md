# Sequential Testing, Peeking, and False Discovery — always-valid inference and the cost of early stopping

> Module 05 — Experimentation, Pilots, and Evidence
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 04 topics 03 and 05; Module 05 topics 01–07
> Version 2026.5

---

## 1. Why this topic exists

A regional operations team runs a twelve-week pilot on a new dock-scheduling process. The dashboard
refreshes every hour. At the Monday stand-up in week three, the cost-per-drop tile is 4.1% better
than control and the p-value reads 0.03. The team lead says the pilot works, the steering committee
approves rollout, and the process is installed network-wide. By week twelve the effect has
disappeared. Nothing was fraudulent and nobody made a data error. The pilot did what the organisation
asked: it produced a significant result at the moment somebody looked.

This is the peeking problem, and in supply-chain analytics it is the default rather than an edge
case. Operational dashboards are the primary interface between analytics and management, they
refresh on the cadence of operations rather than the cadence of inference, and the people watching
them are paid to act. A pilot observed through a live dashboard is a sequential experiment whether or
not anybody designed it as one. The design question is not whether you will look repeatedly. It is
whether the repeated looking sits inside or outside the error rate you report.

**The mechanism in one sentence.** A p-value is calibrated for one test at one planned sample size.
Recompute it every day and stop when it crosses 0.05, and the probability that it crosses at least
once under a true null is no longer 5%. It is the probability that a random walk ever crosses a fixed
line, which is much larger and grows with how often and how early you look.

Four shapes recur.

**One. The stand-up as an unplanned interim analysis.** A daily or weekly meeting reviews the primary
metric. Each meeting is a look. Twelve weeks of daily stand-ups is 84 looks, and nobody wrote down
that there would be 84. The analysis was designed for one.

**Two. The asymmetric guardrail.** The team watches the primary continuously and stops early when it
looks good, but checks the guardrail only at the end, or checks it continuously and ignores alarms
that arrive before the primary is ready. Either way the stopping rule treats the metric that can hurt
you differently from the metric that can reward you. That is worse than plain peeking, because it
selects on the outcome the guardrail exists to protect.

**Three. The dashboard as a multiplicity machine.** Twelve key performance indicators (KPIs), 84
days, five regions and three channels is 15,120 tile-days. Under a true null, at a nominal 5%, a
significant cell somewhere on that dashboard is not a possibility. It is a certainty.

**Four. The correction abandoned because it costs power.** A team discovers sequential testing,
computes that it needs 30% more sample, cannot get the stores, and returns to peeking with a clearer
conscience. The correct response is to recognise that the pilot was underpowered before the
correction and more underpowered after it, and to change the decision the pilot is meant to support.

> **Note:** Correcting for peeking does not rescue an underpowered design. It lowers the error rate
> and raises the minimum detectable effect (MDE) at the same time. Both problems must be fixed, and
> they are fixed by different means: the stopping rule for the first, the design for the second.

**The claim of this topic.** A stopping rule is a design parameter, not an administrative detail. It
belongs in the pre-registration next to the MDE and the decision rule, fixed before the first order
is assigned. Fixing it is cheap: a five-look group sequential correction costs about 3% more sample
under an O'Brien–Fleming boundary and about 23% under a Pocock boundary. Not fixing it is expensive:
the same pilot, watched daily, converts a 5% error rate into something between 25% and 40%.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** mechanically why repeated looks inflate the Type I error rate, and **compute** the
   independent-look bound and the Bonferroni threshold for a stated number of looks.
2. **Distinguish** the three remedies — do not look, spend alpha over looks, use an always-valid
   boundary — and **choose** among them for a stated operational setting.
3. **Read** a group sequential boundary table, **convert** an information ratio into an MDE
   inflation, and **state** what the boundary implies for early stopping in a small-effect pilot.
4. **Compute** the sample or MDE cost of an always-valid procedure, and **judge** whether the pilot
   can afford it against its break-even effect.
5. **State** what a Bayesian stopping rule does and does not guarantee, and **name** the prior
   assumption on which the guarantee depends.
6. **Design** a guardrail rule that neither stops good pilots on noise nor truncates slow harms.
7. **Write** a stopping-rule clause that fixes either the analysis date or the spending function, and
   **record** which one was fixed.
8. **Recognise** the interaction between looks, metrics, and segments, and **construct** a defensible
   family for a dashboard that produces hundreds of cells a week.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold three pictures: the random walk and the line it must not cross, the alpha budget, and the family.

### 3.1 The walk and the line

```text
        Z
        ^
   4.5  |o   O'Brien-Fleming look-1 boundary = 4.56
        |    (an early stop needs an enormous effect)
        |
   3.0  |      . . . . . . . . . . . . . .   anytime-valid boundary
        |    .                            .
   2.0  |- -/- - - - - - - - - - - - - - - -\- - -   nominal 1.96 line
        |  /                                \
   1.0  | /   a path generated under H0       \___
        |/                                        \___
   0.0  +----+--------+--------+--------+--------+-->  information fraction t
        0.2      0.4      0.6      0.8      1.0

              ^                                  ^
        crosses at t = 0.4:                ends at t = 1.0
        "p = 0.03, stop, launch"           below the line: no effect
```

The test statistic, standardised at each look, behaves under the null like a random walk in
information time. Information time is the share of the planned sample that has arrived, not calendar
time. At the end of the walk the statistic is roughly normal with mean zero and standard deviation
one. At every earlier point it is also roughly normal, but the values are correlated, because they
are computed from overlapping data.

A fixed-horizon test asks one question: is the value at `t = 1` beyond 1.96? A sequential procedure
asks another: did the path ever cross a boundary between `t = 0` and `t = 1`? Those are different
questions, and the second is the one a manager with a dashboard is asking.

**Property 1 — the error rate is the probability of ever crossing.** At any single look under the
null the chance of exceeding 1.96 is 2.5% in each direction. Over many looks the chance that at least
one exceeds it is much larger, because the path wanders.

**Property 2 — inflation depends on how many looks, how early, and how correlated.** Two looks far
apart are nearly independent and their error rates nearly add. Two looks one order apart are almost
the same look and add very little. The worst case is looking often and early, which is what a
refreshed operational dashboard does.

**Property 3 — the boundary must widen to compensate.** An anytime-valid boundary rises with
information time, typically like the square root of `log(t)` over `t` in the statistic's natural
units. The widening is the price of permission to look whenever you like.

### 3.2 The alpha budget

```text
   cumulative alpha spent
   0.050 |                                   *
         |                              *
         |                        *
         |                  *                  O'Brien-Fleming:
         |            *                        almost nothing spent
         |      *                              before t = 0.7
         |  * * * * * * * * * * * * * * *      Pocock: roughly even
   0.000 +----------------------------------->  information fraction t
         0                                  1
```

Alpha is a budget, not a threshold. A group sequential design decides in advance how much to spend at
each look. Spend it all at the end and you have a fixed-horizon test. Spend it evenly and you have a
Pocock design, which buys easier early stopping at the price of a stricter final boundary and a larger
maximum sample. Spend almost nothing early and most of it at the end and you have an
O'Brien–Fleming design, which makes early stopping nearly impossible for small effects but costs
almost nothing if you do not stop early.

The practical consequence is uncomfortable. When the effect is small, an efficacy boundary at an
early look is unreachable, so the ability to stop early for success is decorative. What the design
actually buys is the ability to stop early for futility, which saves sample and costs nothing in
alpha if it is specified correctly.

### 3.3 The family

```text
                        looks
                          |
          +---------------+---------------+
          |                               |
       metrics                        segments
          |                               |
   12 dashboard KPIs              5 regions x 3 channels
          |                               |
          +---------------+---------------+
                          |
              tile-tests per pilot = 12 x 84 x 15
                                   = 15,120

   Under a global null at alpha 0.05, the probability of at least one
   "significant" tile is indistinguishable from 1.
```

The multiple-looks problem is one axis of a three-axis family. The others are the metrics on the
dashboard and the segments a manager can filter by. Module 04 Topic 03 treated multiplicity across
metrics and hypotheses. This topic adds the time axis, which scales fastest: a dashboard with a fixed
metric set generates a new test on every refresh, without anyone deciding to run one.

## 4. Definitions and vocabulary

The fourth column names the unit or coding convention. A sequential quantity without an information
fraction attached is a number that cannot be interpreted.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Peeking | Inspecting a test before the planned analysis and letting it influence the decision | Used for any dashboard view, including one with no decision attached | Count of decision-relevant looks |
| Optional stopping | Choosing when to stop on the basis of accumulating data | Treated as synonymous with peeking whether or not a stop occurred | Indicator per look |
| Fixed-horizon test | One analysis at a pre-specified sample size | Assumed to be what a p-value always means | One look, at `t = 1` |
| Information fraction (t) | Share of planned statistical information accrued | Computed as calendar time elapsed | Proportion, 0 to 1 |
| Information time | The scale on which looks are spaced, proportional to variance accrued | Confused with calendar time when traffic is uneven | Proportion of planned variance |
| Interim analysis | Any analysis of accumulating data before the final analysis | Used only for formal monitoring committee meetings | Look index, with date |
| Look | One decision-relevant inspection of the test statistic | Counted once per dashboard refresh regardless of who looked | Integer count |
| Group sequential design | A design with a small pre-specified number of analyses and boundaries | Treated as equivalent to continuous monitoring | Number of looks, boundary per look |
| Alpha spending function | A rule allocating total Type I error across information time | Confused with the boundary values it implies | Cumulative alpha as a function of `t` |
| O'Brien–Fleming boundary | A spending rule that is strict early and near-nominal at the end | Chosen for its reputation without checking the MDE cost | Z per look; information ratio |
| Pocock boundary | A spending rule with a nearly constant boundary across looks | Chosen for easy early stopping without noting the sample cost | Z per look; information ratio |
| Lan–DeMets method | Applying a spending function when look timing is not fixed in advance | Treated as a licence to look whenever, including on results | Spending function plus `t` values |
| Hwang–Shih–DeCani | A one-parameter family spanning OBF-like to Pocock-like spending | Used without stating the parameter | Gamma parameter |
| Information ratio | Sequential maximum sample divided by fixed-horizon sample for equal power | Reported as a percentage rather than a multiplier | Dimensionless, at least 1 |
| Futility boundary | A lower boundary permitting a stop when the effect is unpromising | Confused with an efficacy boundary | Z boundary; binding status |
| Binding futility | A futility rule the design is obliged to follow | Assumed by default | Indicator |
| Non-binding futility | A futility rule that may be overridden without inflating alpha | Treated as identical to binding | Indicator |
| Always-valid inference | Inference whose error guarantee holds at every stopping time | Used as a synonym for "correct" | Error rate, uniform in `t` |
| Anytime-valid p-value | A p-value whose probability of ever falling below alpha is at most alpha | Interpreted as a fixed-horizon p-value | Probability, uniform over stops |
| Confidence sequence (CS) | A sequence of intervals with joint coverage over all `t` | Read as a single confidence interval | Coverage, uniform in `t` |
| mSPRT | Mixture sequential probability ratio test; mixed likelihood ratios forming a martingale | Treated as assumption-free | Mixture and its scale parameter |
| Martingale | A process whose expected next value equals its current value | Used as a synonym for "random" | Process, with filtration |
| Ville's inequality | The bound that a nonnegative martingale exceeds `1/alpha` with probability at most alpha | Applied without checking the martingale property | Probability bound |
| Test martingale | A martingale built from likelihood ratios and used as a test statistic | Confused with the statistic itself | Ratio, starting at 1 |
| E-value | A test martingale's value; its reciprocal is a p-value | Converted without stating the mixture | Ratio, at least 0 |
| Mixing parameter (tau) | The scale of the prior over effect sizes in an mSPRT | Left at a software default | Effect units |
| Tuning parameter (N*) | The sample at which a confidence sequence is calibrated to be tightest | Optimised for a sample the pilot will not reach | Count of units |
| Type I error rate | Probability of rejecting a true null, per experiment | Reported per look rather than per experiment | Probability |
| Family-wise error rate (FWER) | Probability of at least one false rejection in a family | Reported per comparison | Probability, 0 to 1 |
| False discovery rate (FDR) | Expected share of false rejections among those made | Interpreted as the probability a given rejection is false | Proportion, 0 to 1 |
| False positive risk | Share of significant results that are false, given a base rate | Confused with the Type I error rate | Proportion |
| Multiplicity | Error inflation from testing many hypotheses | Applied only across metrics, not across time | Count of tests in the family |
| Stopping rule | A pre-specified mapping from accumulating data to a stop decision | Written after the pilot, from memory | Rule text plus boundary values |
| Analysis date | The single date fixed in advance for the primary analysis | Moved when the result is borderline | Date, with version |
| Look log | A record of when the metric was inspected, by whom, and what was seen | Not kept, so peeking cannot be measured | Timestamped rows |
| Pre-registration | A dated document fixing hypothesis, metric, MDE, stopping rule, decision rule | Treated as bureaucracy by teams that intend to peek | Document with version |
| Garden of forking paths | Error inflation from many defensible choices made after seeing data | Reduced to "p-hacking", which implies intent | Count of defensible paths |
| Conditional exaggeration | Upward bias in the estimate among results that stop early | Reported as the effect size | Ratio of stop-time estimate to truth |
| Guardrail metric | A metric that must not degrade, used to veto a winner | Checked only at the end, or only when convenient | Definition plus margin |
| Non-inferiority margin | The largest degradation the business will tolerate | Set to zero, which requires infinite sample | Outcome units |

## 5. The mechanics, step by step

Eighteen steps in six phases. The order matters: the regime before the data, the boundary before the
dashboard, the readout before the result.

### Phase A — Decide the regime before the data

**Step 1. Write the decision rule and the analysis regime in one sentence.**

"The category director will roll the slot-booking change out to all 14 distribution centres (DCs)
from the start of the next quarter if the primary metric clears the pre-specified boundary at the
final look, with no guardrail degradation beyond the stated margin; the pilot will be analysed at
five equally spaced looks using an O'Brien–Fleming spending function." That sentence names the
decision, the decision-maker, the metric, the boundary, and the regime.

*Input:* the decision from Topic 01. *Output:* one sentence in the pre-registration. *Fails if
skipped:* there is no rule to violate, so any stopping time can be described as consistent with the
plan.

**Step 2. Count the looks the organisation will actually take.**

List every recurring meeting, dashboard view, and ad hoc request that will touch the primary metric.
Include the executive who asks for "a quick read" on Friday. The count is usually three to ten times
the number of formal interim analyses. A pilot with one planned analysis and a daily stand-up has 84
looks in practice.

*Input:* the meeting calendar and the dashboard access log. *Output:* an honest look count per week.
*Fails if skipped:* the correction is sized for a design nobody is running.

**Step 3. Fix either the analysis date or the spending function, and record which.**

There are two admissible regimes. **Regime A, fixed date:** one analysis, on a stated date, at a
stated sample, with no inspection of the primary by anyone with decision authority before then.
**Regime B, sequential:** a spending function or boundary rule fixed in advance, with looks at
information times that are pre-specified or driven by a rule independent of the observed effect.
Write down which one, with version and date.

*Input:* the look count and the operational constraints. *Output:* a regime statement. *Fails if
skipped:* the organisation drifts into Regime B without a boundary, the worst of the three options.

**Step 4. Separate decision metrics from monitoring metrics.**

The primary and the guardrail family are decision metrics and get a stopping rule. Operational
metrics — pick rate, dwell time, backlog — are monitoring metrics: they may be watched continuously
and acted on operationally, but they do not enter the inferential claim. Label every tile with its
role.

*Input:* the metric hierarchy from Topic 06. *Output:* a labelled dashboard specification. *Fails if
skipped:* a monitoring metric becomes the reason for a decision whose error rate was never computed.

### Phase B — Quantify the exposure

**Step 5. Compute the independent-look bound.**

If looks were independent, the probability of at least one false positive at level alpha over `k`
looks would be `1 - (1 - alpha)^k`. At alpha = 0.05 that is 22.6% for five looks, 46.0% for twelve,
70.8% for twenty-four, and 98.7% for eighty-four. This is an upper bound, because looks on the same
data are positively correlated, and it is loose. It is still the first number to show, because it is
arithmetic anyone can check.

*Input:* the look count. *Output:* a bound per look count. *Fails if skipped:* nobody believes the
problem is real, because 5% feels small.

**Step 6. Correct the bound for correlation using the actual schedule.**

Looks far apart in information time are closer to independent; looks one order apart are nearly the
same look. Simulation studies of daily peeking report false positive rates of roughly 14% at five
looks, 19% at ten, 25% at twenty, 32% at fifty, and 37% at one hundred, against a nominal 5%. These
figures depend on the statistic, the spacing, and whether the variance is estimated at each look;
treat them as a range, not a constant. The mechanism is stable: inflation grows with the number of
looks and is worst when early looks have little data.

*Input:* the look schedule. *Output:* a realistic error-rate range. *Fails if skipped:* the
correction is sized to the bound, which overstates the problem at large look counts, and the team
concludes correction is impossible.

**Step 7. Add metrics and segments to the family.**

Multiply looks by the metrics that carry a decision and the segments that will be inspected. Twelve
metrics, 84 days, and fifteen segment combinations is 15,120 tests. The family is not the pilot; it
is the dashboard times the calendar.

*Input:* the metric and segment lists. *Output:* a family size and a decision about which members are
confirmatory. *Fails if skipped:* the correction covers one axis and the other two generate the
discoveries.

### Phase C — Apply a classical fix

**Step 8. Choose the number and timing of looks in information time.**

Looks should be spaced by information, not calendar. If traffic is uniform the two coincide. If the
pilot ramps up, a look at week two may be at `t = 0.05`, which is nearly worthless and expensive in
alpha. State the information fractions, not only the dates.

*Input:* the traffic forecast. *Output:* a look schedule in `t`. *Fails if skipped:* boundary values
computed for equal spacing are applied to unequal spacing.

**Step 9. Choose the spending function and justify it against the decision.**

O'Brien–Fleming is the confirmatory default: strict early, so an early claim needs overwhelming
evidence, and cheap if you do not stop early. Pocock suits cases where early stopping is genuinely
valuable and the maximum sample can grow. Hwang–Shih–DeCani interpolates with one parameter. Choose
by asking what an early stop is worth operationally, not by copying a boundary from a paper.

*Input:* the value of an early decision. *Output:* a named spending function and parameter. *Fails if
skipped:* the boundary is conservative in a way nobody chose, and the design is criticised for a
property it did not need.

**Step 10. Derive the boundary, the information ratio, and the adjusted MDE.**

For five equally spaced looks at a two-sided 5% level, the O'Brien–Fleming boundaries are Z = 4.562,
3.226, 2.634, 2.281 and 2.040, with a maximum information ratio of 1.028. The Pocock boundary is a
constant Z = 2.413 with an information ratio of 1.229. The MDE at the original sample inflates by the
square root of the information ratio: 1.4% for OBF and 10.9% for Pocock. Report the adjusted MDE.

*Input:* the spending function and look schedule. *Output:* boundary values, information ratio,
adjusted MDE. *Fails if skipped:* the pre-registration quotes an MDE the design cannot deliver.

**Step 11. Add a futility boundary and say whether it binds.**

A futility boundary stops the pilot when the effect is too small to matter. It saves sample and, if
non-binding, costs nothing in alpha. In small-effect supply-chain pilots it is the boundary that will
actually be used, because the early efficacy boundary is usually unreachable. State whether the rule
binds: a binding rule must be followed and its alpha cost accounted for; a non-binding rule may be
overridden.

*Input:* the break-even and the smallest effect size of interest. *Output:* a lower boundary and its
binding status. *Fails if skipped:* the pilot runs to completion for an effect ruled out at week
three, or stops for futility and restarts when the result improves.

### Phase D — Or choose always-valid inference

**Step 12. Choose the method and state its assumptions.**

The mixture sequential probability ratio test (mSPRT) builds a martingale from a mixture of likelihood
ratios over a prior on the effect, and Ville's inequality gives a time-uniform error bound. Confidence
sequences invert the same idea into intervals with joint coverage. The original mSPRT guarantee is
parametric or asymptotic; nonparametric confidence sequences exist but are wider. State which you use
and what it assumes about the outcome distribution.

*Input:* the outcome type and tolerance for assumptions. *Output:* a named method. *Fails if
skipped:* "always valid" is treated as a property of the software rather than of a model.

**Step 13. Tune the mixing parameter to the decision horizon.**

The mSPRT mixture has a scale, often written tau, encoding where you expect the effect to be.
Confidence sequences have an analogous parameter, sometimes written `N*`, calibrated to the sample at
which decisions are made. A platform default of a few thousand units is wrong for a pilot that
decides at 500,000.

*Input:* the MDE and the expected decision sample. *Output:* a documented tuning value. *Fails if
skipped:* the procedure is calibrated for a different experiment and its width is uninterpretable.

**Step 14. State the power cost and check it against the break-even.**

Always-valid inference is wider than fixed-horizon inference at the same sample. Published
implementations report needing roughly two to three times the sample for the same power when the
effect is correctly anticipated, with empirical runtimes of one to four times a well-calibrated
fixed-horizon design and a smaller premium when the true effect is smaller than assumed. At a fixed
sample, a two- to three-fold sample requirement inflates the MDE by a factor of 1.41 to 1.73. If that
pushes the MDE above the break-even, the pilot cannot afford continuous validity and should choose a
small number of pre-specified looks instead.

*Input:* the fixed-horizon MDE and the break-even. *Output:* go or no-go on the always-valid route.
*Fails if skipped:* the pilot buys a guarantee it cannot afford and loses the power it needed.

### Phase E — Or choose a Bayesian rule

**Step 15. Choose the prior, the stopping rule, and what "stop" means.**

A Bayesian rule stops when the posterior probability that the effect exceeds a threshold is high
enough, or when the expected loss of continuing exceeds the expected loss of stopping. The posterior
is a martingale, so posterior odds are not distorted by the stopping rule in the way a p-value is.
That property depends on the prior being a genuine representation of belief rather than an improper
default chosen for convenience. State the prior's source and the threshold's business justification.

*Input:* historical effect sizes from comparable changes. *Output:* a prior with provenance and a
posterior-based rule. *Fails if skipped:* "the Bayesian result is significant" becomes the rule,
which is a threshold with no decision content.

**Step 16. Check the frequentist operating characteristics by simulation.**

Even if you do not adopt a frequentist guarantee, know the false positive rate your rule produces
under a true null and the power it produces at the break-even. Simulate under the chosen prior and
under a point null, and report both. A rule calibrated under its prior and badly calibrated under the
null will produce surprises when the prior is wrong, which it will be.

*Input:* the prior and the rule. *Output:* a two-row table of error rate and power. *Fails if
skipped:* the Bayesian framing avoids a calculation rather than making a decision.

### Phase F — Operate, read out, and govern

**Step 17. Put the boundary on the dashboard and give guardrails a symmetric rule.**

A tile showing only the effect and its interval invites the reader to judge significance by eye. A
tile showing the effect, the interval, the information fraction, and the boundary at that fraction
invites judgement of the design. It is harder to build and much harder to misread. Decide in advance
whether the guardrail is monitored continuously with a stop rule, checked at the pre-specified looks,
or assessed once at the end, and apply the same schedule to the primary and the guardrail unless you
can justify the difference. A rule that stops for a good primary but not for a bad guardrail is a
selection rule, not a safety rule.

*Input:* the boundary table and the guardrail margin. *Output:* a dashboard and monitoring
specification. *Fails if skipped:* the pilot stops early on a win and observes the guardrail for a
shorter window precisely when the intervention is most aggressive.

**Step 18. Read out with the bias adjustment, then log what happened.**

An estimate taken at a stopping time is biased away from zero. Report the observed estimate, the
boundary crossed, and a bias-adjusted or median-unbiased estimate where available, with the interval
from the sequential procedure rather than a fixed-horizon interval recomputed at the stopping sample.
Then record every inspection of the primary metric by anyone with decision authority: timestamp,
person, information fraction, value seen, action taken. If the log shows eleven unplanned looks, say
so and correct the analysis accordingly, even approximately.

*Input:* the look log and the final data. *Output:* a readout with three numbers, a stopping
statement, and an appendix log. *Fails if skipped:* the most flattering number becomes the headline,
and the organisation cannot learn from its own pilots.

## 6. Worked example

Ravensbourne Retail runs the click-and-collect promise pilot from Topic 06: 96 stores, 48 per arm,
twelve weeks, about 18,000 eligible sessions per store, baseline conversion 3.60%, store-level
intra-cluster correlation (ICC) 0.012. The fixed-horizon design has a standard error of the
difference of 0.41751 percentage points and an MDE of 1.1697 points at 80% power and a two-sided 5%
test. The break-even effect is 0.1076 points. The company installs a live dashboard that refreshes
hourly, because the promise is an operational change and operations need to see it working.

All figures are constructed to be internally consistent and checkable by hand. The fixed-horizon
inputs are carried from Topic 06 unchanged.

### 6.1 The design as written and as operated

The pre-registration says: "The primary analysis will be conducted at the end of week 12." It does
not say who may look before then, what they may look at, or what they may do with what they see. That
omission is the whole problem.

| Field | Value |
|---|---:|
| Stores per arm; weeks | 48; 12 |
| Sessions per store; baseline conversion | 18,000; 3.60% |
| Store-level ICC; SE of the difference | 0.012; 0.41751 pp |
| MDE, one look; break-even | 1.1697 pp; 0.1076 pp |
| Guardrail; margin | On-time pick, 97.8%; 0.5 pp |

In operation, four look schedules coexist:

```text
   dashboard refresh     2,016 times over 12 weeks (hourly)
   stand-up inspection      84 times (daily, weekdays)
   regional filters       ~180 times (3 directors x 5 regions x 12 weeks)
   pre-specified analysis    1 time (week 12)
```

The correction that matters is for the stand-up inspections, because those are the looks attached to a
decision. The regional filters matter for a different reason: they multiply the family by fifteen,
and a regional "win" is exactly the kind of result a director will act on.

### 6.2 What the looks cost in error rate

The independent-look bound is the first number to show, because it is checkable:

```text
   k looks    bound = 1 - 0.95^k    Bonferroni z at alpha 0.05
      1             0.0500                 1.960
      5             0.2262                 2.576
     12             0.4596                 2.865
     24             0.7080                 3.078
     84             0.9865                 3.434
```

The bound assumes independence and is therefore too high. Simulation studies report lower figures
because looks are correlated:

| Looks | Independent bound | Reported simulation range | Nominal alpha |
|---:|---:|---:|---:|
| 1 | 5.0% | 4.9–5.2% | 5% |
| 5 | 22.6% | about 14% | 5% |
| 10 | 40.1% | about 19% | 5% |
| 20 | 64.2% | about 25% | 5% |
| 50 | 92.3% | about 32% | 5% |
| 84–100 | 98.7% | 35–38% | 5% |

> **Note:** The simulation figures are secondary reproductions from several independent simulators,
> graded D in section 16. They vary with the statistic, the spacing, and whether the variance is
> estimated at each look. The stable finding is direction and order of magnitude, not the second
> decimal. The bound is exact arithmetic and is the number to show a sceptical audience first.

At 84 looks the honest statement is: "If the promise does nothing, the probability that this
dashboard shows a significant conversion gain at some point in twelve weeks is somewhere between
about one in three and one in one, depending on how looks are counted. It is not 5%."

### 6.3 What an early stop would have reported

Suppose the true effect is zero. The standardised statistic behaves like a random walk in information
time, and conditional on stopping at the first crossing of a 1.96 boundary the reported effect is
about 1.96 standard errors at the information fraction where the crossing occurred. Because the
standard error shrinks as information accrues, an early crossing reports a much larger effect than a
late one.

```text
   t = 0.25   SE = 0.41751 / sqrt(0.25) = 0.8350 pp
              reported effect ~ 1.96 x 0.8350 = 1.64 pp  = 45.5% relative

   t = 0.50   SE = 0.41751 / sqrt(0.50) = 0.5904 pp
              reported effect ~ 1.96 x 0.5904 = 1.16 pp  = 32.1% relative

   t = 1.00   SE = 0.41751 pp
              reported effect ~ 1.96 x 0.41751 = 0.82 pp = 22.7% relative
```

Every one of these is between eight and fifteen times the break-even effect of 0.108 points. A team
that stops at week three and reports a 1.6-point gain, or a 45% relative lift, has reported the
boundary, not the effect. The number is arithmetically correct and evidentially empty. The overshoot
makes it worse: with discrete looks the path stops at the first look that exceeds the boundary, not
at the boundary, so the reported value is the boundary plus an overshoot that is larger for a sparse
schedule. The damage is worst in exactly the weekly-meeting regime most pilots use.

### 6.4 What a pre-specified sequential design would have cost

Suppose the team had fixed five equally spaced looks in information time, at 20%, 40%, 60%, 80%, and
100% of planned sessions, at a two-sided 5% level.

| Design | Looks | Boundary | Information ratio | MDE (pp) | Relative | Times break-even |
|---|---:|---|---:|---:|---:|---:|
| Fixed horizon | 1 | 1.960 | 1.000 | 1.1697 | 32.5% | 10.9 |
| O'Brien–Fleming | 5 | 4.562, 3.226, 2.634, 2.281, 2.040 | 1.028 | 1.1860 | 32.9% | 11.0 |
| Pocock | 5 | 2.413 at every look | 1.229 | 1.2967 | 36.0% | 12.0 |
| Always-valid, 2x sample | continuous | confidence sequence | 2.000 | 1.654 | 45.9% | 15.4 |
| Always-valid, 3x sample | continuous | confidence sequence | 3.000 | 2.026 | 56.3% | 18.8 |

The group sequential arithmetic is one line each: `1.1697 x sqrt(1.028) = 1.1860` and
`1.1697 x sqrt(1.229) = 1.2967`. The correction for five looks is cheap: 1.4% more MDE under
O'Brien–Fleming, 10.9% under Pocock. Permitting five pre-specified looks costs about as much as a
rounding error in the store count.

**The uncomfortable finding.** Look at the first boundary. To stop at the first look under
O'Brien–Fleming the observed effect must exceed 4.562 standard errors at `t = 0.2`, where the standard
error is 0.9336 points. That is an effect of 4.26 points, or a 118% relative lift on a 3.60% baseline.
Under Pocock the first boundary is 2.413, requiring 2.25 points, or 63% relative. For a pilot looking
for a tenth of a point, neither boundary is reachable. The early-stopping option is decorative.

What the design actually buys is the futility boundary. If the promise is not working, a lower
boundary at week five can stop the pilot and free the stores, at no cost in alpha if the rule is
non-binding. That is the operational value of the sequential design, and it is the part most pilots
leave out.

### 6.5 What continuous validity would have cost

If the team insists on watching the dashboard and acting whenever it moves, the honest design is an
always-valid one. In this pilot the two- to three-fold sample premium would take the MDE from 1.17
points to between 1.65 and 2.03 points, and the design would need 192 to 288 stores for the same
power. Ravensbourne has 96. The always-valid route is not available at this scale unless the pilot is
redesigned around more units or a longer window.

### 6.6 The guardrail

The guardrail is the on-time pick rate, baseline 97.8%, ICC 0.03, margin 0.5 points. From Topic 06
the standard error of the difference is 0.5256 points at the final sample.

```text
   one-sided 95% bound on degradation, one look      = 1.645 x 0.5256 = 0.865 pp
   one-sided bound at the final look of a five-look
     O'Brien-Fleming design                          = 2.040 x 0.5256 = 1.072 pp
```

Both exceed the 0.5-point margin the business tolerates. The guardrail was underpowered at one look;
the sequential design makes it more so. Now add the operational behaviour: the team watches it daily
and has agreed to "pause if it degrades". Under a true null a daily-watched guardrail crosses a 5%
degradation threshold at some point in a large share of pilots, by the arithmetic of section 6.2. So
the rule produces two errors at once: it pauses good pilots on noise, and when it does not pause it
has observed the guardrail for a shorter window than planned, which is where slow harms appear.

That second effect is specific to supply chains. Damage, returns, short-ships, and late deliveries
materialise weeks after the order. A pilot stopped at week four because the primary looked good has
observed guardrails for four weeks, not twelve. The harms it would have caught are not missing at
random; they are missing precisely because the intervention was rolled out early.

### 6.7 The verdict

Two errors are in this pilot and they are independent.

The first is the sequential error. As operated, the nominal 5% is not 5%, and every significant result
is suspect. Fixing it costs 1.4% to 11% more MDE, which is cheap.

The second is the power error, carried from Topic 06. Even at one look the MDE is 10.9 times the
break-even, and fixing the sequential error does not touch it. A perfectly valid sequential design at
96 stores still cannot detect the effect that would pay for the promise.

The recommendation is not "add a spending function and proceed". It is: fix the stopping rule because
it is cheap and the dashboard will otherwise produce a false winner; and separately decide whether a
pilot this size can answer the question. Here it cannot, and the honest output is a feasibility pilot
with a stated limit, not a confirmatory test with a boundary.

## 7. Code, query, or calculation

One block, arithmetic only. It computes the independent-look bound, the Bonferroni threshold, the
group sequential MDE inflation, the effect reported at an early stop, and the always-valid premium.
It uses only the standard library.

```python
from statistics import NormalDist
nd = NormalDist()
alpha, se, base = 0.05, 0.0041751, 0.036

for k in (1, 5, 12, 24, 84):
    print(f"k={k:3d}  bound {1-(1-alpha)**k:6.3f}  "
          f"Bonferroni z {nd.inv_cdf(1-alpha/(2*k)):5.3f}")

mde = (nd.inv_cdf(0.975) + nd.inv_cdf(0.80)) * se
print("fixed MDE pp", round(100 * mde, 4))
for name, ratio in (("OBF", 1.028), ("Pocock", 1.229), ("AV2", 2.0), ("AV3", 3.0)):
    adj = 100 * mde * ratio ** 0.5
    print(f"{name:7s} ratio {ratio:5.3f}  MDE {adj:6.3f} pp  "
          f"{adj/base:5.1f}% relative")

for t in (0.25, 0.50, 1.00):
    print(f"t={t:4.2f}  SE {100*se/t**0.5:6.4f} pp  "
          f"stop-time effect {196*se/t**0.5:5.3f} pp")
```

Expected output:

```text
k=  1  bound  0.050  Bonferroni z 1.960
k=  5  bound  0.226  Bonferroni z 2.576
k= 12  bound  0.460  Bonferroni z 2.865
k= 24  bound  0.708  Bonferroni z 3.078
k= 84  bound  0.987  Bonferroni z 3.434
fixed MDE pp 1.1697
OBF     ratio 1.028  MDE  1.186 pp   32.9% relative
Pocock  ratio 1.229  MDE  1.297 pp   36.0% relative
AV2     ratio 2.000  MDE  1.654 pp   45.9% relative
AV3     ratio 3.000  MDE  2.026 pp   56.3% relative
t=0.25  SE 0.8350 pp  stop-time effect 1.637 pp
t=0.50  SE 0.5904 pp  stop-time effect 1.157 pp
t=1.00  SE 0.4175 pp  stop-time effect 0.818 pp
```

Three checks before trusting the block. First, the fixed MDE must equal `2.8016 * SE`, the identity
used in Topic 06. Second, the OBF adjusted MDE must sit only slightly above the fixed MDE, because
the information ratio is 1.028, while Pocock must be visibly worse. Third, the stop-time effect must
fall as information accrues and must sit between eight and fifty times the break-even at every look.

> **Note:** The stop-time effect uses 1.96 as the threshold. Under a group sequential boundary the
> threshold differs at each look, and the first OBF boundary of 4.562 at `t = 0.2` implies a
> stop-time effect of 4.26 points. The block shows the naive-peeking case, which is what an
> uncorrected dashboard does.

## 8. Data, standards, and regulatory context

### 8.1 The data a sequential design requires

The extra requirement over a fixed-horizon analysis is a record of time and of looking.

| Field | Why the design depends on it | What goes wrong without it |
|---|---|---|
| Assignment timestamp | Defines when information starts accruing | Pre-launch units contaminate the first look |
| Exposure timestamp | Defines the information fraction | Look timing computed on intent, not delivery |
| Metric event timestamp | Defines when an outcome is observable | Long-lag outcomes make early looks optimistic |
| Planned sample and information fractions | The `t` scale for the boundary | Boundary values applied at the wrong `t` |
| Look log with timestamps and viewers | Evidence of how many looks occurred | Peeking cannot be measured or corrected |
| Dashboard refresh and access logs | The de facto look schedule | The plan is used and the reality ignored |
| Freeze record and data cut date | Reproducibility of each look | The look cannot be reconstructed |
| Metric definition change log | Comparability across looks | A definition change looks like an effect |
| Tuning parameter for always-valid methods | Width and calibration | Interval computed for another horizon |
| Stopping decisions and reasons | Governance and learning | The organisation cannot tell if the rule held |

The most important is the look log, and it is the one almost nobody keeps. The dashboard access log
is a reasonable proxy and is often already collected. If it shows that fourteen people opened the
primary metric on the day the result turned, the pilot was sequential in fact regardless of the
pre-registration.

### 8.2 Standards and regulatory guidance

Sequential designs are mature in clinical research and almost entirely unregulated in commercial
pilots. The clinical guidance is still the best statement of what a defensible design contains.

| Instrument | What it requires or endorses | Status for a commercial pilot | Grade |
|---|---|---|---|
| FDA guidance, *Adaptive Designs for Clinical Trials of Drugs and Biologics* (2019) | Pre-specified adaptation rules; O'Brien–Fleming and Pocock boundaries; Lan–DeMets alpha spending with safeguards when look timing could depend on results | Not binding on a retailer; the closest thing to a regulatory standard | B |
| ICH E9, *Statistical Principles for Clinical Trials* | Pre-specification of interim analyses and treatment of multiplicity | Harmonised guideline for trials; not binding commercially | B |
| ICH E20 (in development) | Adaptive design principles, including estimands and simulation evidence | Check current status before citing; not final at the time of writing | B, status caveat |
| Company pre-registration template | Hypothesis, metric, MDE, stopping rule, spending function, decision rule | Internal governance artefact | None |

The FDA guidance makes three points that transfer. First, group sequential boundaries control Type I
error when there are multiple interim looks at a conventional level. Second, alpha spending is
acceptable because it specifies how the error budget is spent over time while allowing flexibility in
the number and timing of analyses. Third, if the timing of an interim analysis is chosen on the basis
of comparative results, the error rate can inflate even under a spending function. The FDA's
safeguard is that the schedule should be driven by information statistically independent of the
treatment effect, such as enrolment rate or calendar logistics.

That third point is the one commercial teams violate most often. "We will look again when the result
gets closer" is a timing rule that depends on the result. So is "we will extend the pilot if it is
borderline". Both are data-dependent schedules, and neither is covered by a spending function chosen
in advance.

> **Note:** None of these instruments binds a commercial pilot, and none is enforced. Their value is
> that they encode the fields a reviewer needs. A pre-registration borrowing the FDA's list —
> adaptation rule, boundary, timing rule, and the information on which timing depends — would
> eliminate most of the failures in section 10.

### 8.3 What is convention rather than requirement

- **Five percent and eighty percent.** Conventions with no theoretical privilege. A pilot deciding at
  a 1% error rate and 90% power is not wrong; it is making a different trade.
- **A small number of looks.** Three to five is usual. The number is a cost decision: more looks means
  more flexibility and a wider boundary.
- **O'Brien–Fleming as the default.** Inherited from confirmatory drug trials. A pilot with a cheap
  rollback might prefer Pocock.
- **"Do not peek" as a rule.** An organisational convention that is usually unenforceable and often
  dishonest. The enforceable version is "if you look, the boundary accounts for it".
- **Reporting a correction at all.** Not required by any regulator for a commercial pilot, but
  required by anyone who wants the pilot to survive a post-mortem.

An analyst who can separate a requirement from a convention can challenge the convention. One who
cannot will treat the five-look O'Brien–Fleming table as a physical constant.

## 9. Industry practice and case evidence

### 9.1 Where the methods come from

| Contribution | Source | What it established | Grade |
|---|---|---|---|
| Repeated significance testing on accumulating data | Armitage, McPherson and Rowe (1969) | The error rate of repeated tests and the boundaries required to control it | A |
| Group sequential boundaries with a fixed number of looks | Pocock (1977); O'Brien and Fleming (1979) | Constant and conservative-early boundary families, with sample-size implications | A |
| Alpha spending functions | Lan and DeMets; reviewed in DeMets and Lan (1994) | Flexible allocation of alpha over information time, tolerant of unplanned timing | A |
| Comprehensive treatment of group sequential methods | Jennison and Turnbull (2000) | Boundaries, bias adjustment, sample-size re-estimation, and the theory linking them | A |
| Always-valid inference for online experiments | Johari, Koomen, Pekelis and Walsh (2022) | Always-valid p-values and intervals from an mSPRT, valid at any stopping time | A |
| Nonparametric confidence sequences | Howard, Ramdas, McAuliffe and Sekhon (2021) | Time-uniform, nonasymptotic intervals without a parametric likelihood | A |
| Asymptotic confidence sequences for production | Waudby-Smith et al., preprint | A CLT-scale bridge making confidence sequences practical at platform scale | C |

### 9.2 What platforms actually ship

| Platform or source | Default behaviour | Stated guarantee | Grade and date |
|---|---|---|---|
| Optimizely Stats Engine | Sequential is the default; frequency-based and Bayesian modes are alternatives | Always-valid p-values with false discovery rate control; mandatory two-tailed tests | D, retrieved 2026-11 |
| Optimizely white paper, "The New Stats Engine" | Likelihood-ratio martingale with a normal prior; boundary proportional to `sqrt(log n / n)` | Type I error controlled at any stopping time; the always-valid p-value overstates the fixed-horizon one | D, vendor technical paper |
| GrowthBook | Bayesian is the default; sequential testing is available for the frequentist engine only | Asymptotic confidence sequences with a tuning parameter `N*` defaulting to 5,000 | D, retrieved 2026-11 |
| GrowthBook documentation on Bayesian results | States that Bayesian results remain valid when stopping early | Adds that early stopping can still inflate false positive rates, calling it "a difference without a distinction" | D, retrieved 2026-11 |
| Adobe Journey Optimizer | Anytime-valid confidence sequences in production | Reports 2–3x sample for equal power, 1–4x runtime, and 15% of 2,089 tests significant under fixed-horizon peeking but not under the confidence sequence | D for the product page; C for the paper |
| General-purpose BI tools | No experiment concept, no information fraction, no boundary | None; tiles are recomputed fixed-horizon statistics at each refresh | Inference, not a sourced product claim |

### 9.3 What the platform evidence shows about size

The Adobe paper is the most useful public calibration because it reports both a simulation and a
production comparison. In simulation, fixed-horizon testing with peeking produced a cumulative Type I
error approaching one in A/A tests, while the confidence sequence and the mSPRT stayed below 5%. In
production, across 2,089 real experiments, the two approaches agreed 85% of the time; 308 tests, or
15%, were significant under peeking but not under the confidence sequence, and three went the other
way.

The 15% figure is the one to remember. It is the share of "wins" that a peeking organisation would
have banked and an always-valid organisation would not. In a company running fifty pilots a year, that
is seven or eight rollouts that were noise.

> **Note:** The 2,089 experiments are online product experiments at a software company, not supply
> chain pilots. The arithmetic transfers; the base rates do not. Supply-chain pilots typically have
> fewer units, longer outcome lags, and larger costs of a wrong rollout, which make peeking worse and
> the always-valid premium harder to afford.

### 9.4 What the public record does not contain

We found no public audit of sequential practice in commercial supply-chain pilots. Retailers,
carriers, and distributors do not publish look schedules, spending functions, or dashboard access
logs. The evidence here comes from clinical trials, online experimentation, and platform
documentation, and it transfers because the arithmetic is the same.

What does not transfer is the governance. Clinical trials have monitoring committees, registered
protocols, and regulators who read them. Software platforms have engines that apply a boundary
whether or not the experimenter understands it. A supply-chain pilot has a dashboard, a stand-up, and
a manager with stop authority. The machinery is available and the enforcement is absent, which means
the analyst has to supply the governance personally: the written stopping rule, the boundary on the
tile, and the look log in the appendix.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Unplanned continuous peeking | Primary metric on a dashboard refreshing faster than the analysis plan | Every inspection is a test; nominal alpha applies to none | False positive rate of 25–40% instead of 5% | Fix the date or the spending function before launch |
| Stopping on the first significant look | "It hit significance on Thursday, so we stopped" | The stopping time is chosen by the data | The reported effect is the boundary, not the effect | Pre-specify the boundary; report the bias |
| Asymmetric stopping | The pilot stops early for a win but never for a loss | The rule is applied in one direction only | Winners truncated, losers run to completion | Apply the same schedule both ways |
| Guardrail checked only at the end | Guardrail tile blank until week 12 | The harm window is unmonitored at peak exposure | A real degradation is found after rollout | Pre-specify a guardrail schedule and margin |
| Guardrail watched daily with a stop rule | "We paused three pilots last quarter on the damage metric" | Daily monitoring of a low-base-rate metric alarms on noise | Good pilots stopped; alarms ignored | Use a sequential guardrail rule, or monitor without stopping |
| Analysis date moved | Pre-registration says week 12; readout dated week 15 | The date moved after a borderline result | Alpha spent twice; the second analysis is invalid | Version the pre-registration and log date changes |
| Spending function copied across designs | Five-look OBF boundaries applied to a three-look uneven design | Boundary values depend on timing and `t` | Nominal alpha is not actual alpha | Recompute for the actual schedule |
| Timing rule depends on results | "We will add a look if it is close" | Data-dependent timing inflates error even under spending | The spending guarantee does not hold | Fix the schedule or drive it by independent information |
| Dashboard multiplicity | 12 KPIs x 84 days x 15 segments, no correction | The family is the dashboard times the calendar | A significant cell is certain under a global null | One primary, a guardrail family, everything else descriptive |
| Segment shopping after a null | Overall null; a regional cut is presented | The segment was chosen after seeing the data | The reported effect is selected, not estimated | Pre-specify segments or label the cut exploratory |
| Always-valid method used without its power cost | "We use a sequential engine, so we can stop whenever" | The method is valid but wider; the design was not resized | Underpowered pilot; null misread as no effect | Compute the sequential MDE before launch |
| Bayesian stop read as a free lunch | "Bayesian doesn't have a peeking problem" | Posterior validity is not calibrated decision error | A noisy win rolled out with a posterior attached | State the prior; simulate the null error rate |
| Stopping-time estimate reported as the effect | The headline is the week-4 number | Conditional exaggeration at a boundary | Effect overstated by 2–10x | Report the bias-adjusted estimate and interval |
| Pilot extended after a borderline result | "We need a few more weeks to be sure" | A data-dependent schedule change | The final analysis is neither fixed nor valid sequential | Pre-specify the extension rule and its alpha cost |
| Restarting a pilot that dipped | The experiment resets and the clock restarts at week 0 | The reset is triggered by an unfavourable path | Selection on the path, plus discarded data | Reset only for technical faults, never for results |
| Novelty read as durability | The effect is largest in week 1 and decays | Early stops capture the novelty peak | A durable null rolled out on a transient lift | Pre-specify a minimum duration and a washout |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The look schedule is known in advance | The boundary depends on the number and timing of looks | Ad hoc checks, executive requests, dashboard views | Compare the plan with the access log |
| Information accrues as planned | The `t` scale converts looks into boundary values | Traffic ramps, seasonality, supply disruption | Plot cumulative information against plan weekly |
| The statistic is approximately a random walk | Boundary values are derived under this model | Heavy tails, ratio metrics, few clusters | Simulate the null distribution at each look |
| The variance is stable over the pilot | Boundaries assume known or slowly changing variance | Variance grows with the mean, or the population shifts | Compare variance by look |
| Outcomes are observed with a short, constant lag | Early looks must compare like with like | Returns, damage, complaints observed weeks later | Measure the lag; define the primary on a short-lag metric |
| Units are independent | The variance and martingale property depend on it | Shared inventory, shared fleet, interference | See Topic 07; use cluster-aware sequential methods |
| The spending function matches the design | Boundary values are function-specific | Copying values from a paper with different spacing | Recompute from the function and the actual `t` |
| The Bayesian prior is genuine | The calibration claim depends on it | A default improper prior chosen for convenience | Sensitivity analysis across priors |
| The tuning parameter matches the decision horizon | Confidence sequence width depends on it | A platform default calibrated elsewhere | Recompute at the planned sample and state it |
| The stopping rule is followed | The guarantee is a property of the rule, not the intention | Overrides, extensions, restarts | Compare the look log with the rule; report deviations |

**Three hard limits.** Every correction costs sample or MDE, and always-valid corrections cost a
multiple rather than a percentage. A pilot needing two to three times the sample may need two to
three times the duration, which crosses seasons and changes the baseline. And a boundary nobody
follows is decoration: the only evidence that it was followed is a look log somebody kept.

## 12. Exercises

### Exercise 1 — The look-count arithmetic (calculation, 25 minutes)

A pilot runs 16 weeks. The team holds a weekly review, checks the dashboard ad hoc about twice a week,
and runs one formal interim at week 8. Compute the independent-look bound and the Bonferroni
threshold for the formal count and for the practical count.

**Deliverable:** a table of look counts, bounds, and thresholds, with counts justified.
**Acceptance criteria:** formal and practical counts distinguished; the bound labelled as an upper
bound; thresholds stated in Z and p-value units.
**Estimated time:** 25 minutes.

### Exercise 2 — Reading a boundary table (calculation, 30 minutes)

Five equally spaced looks at a two-sided 5% level. O'Brien–Fleming boundaries are 4.562, 3.226, 2.634,
2.281, 2.040 with an information ratio of 1.028. The fixed-horizon MDE is 0.80 points on a 12%
baseline, and the standard error at `t = 0.2` is 0.90 points. Compute the adjusted MDE, the relative
MDE before and after, and the effect required to stop at the first look.

**Deliverable:** three numbers with arithmetic and a one-sentence interpretation.
**Acceptance criteria:** information ratio converted with a square root; first-look effect computed
against the `t = 0.2` standard error; interpretation states whether early stopping is available.
**Estimated time:** 30 minutes.

### Exercise 3 — Design the stopping rule (design, 45 minutes)

A 14-DC pilot of a slot-booking change runs 20 weeks, reviewed weekly. Propose a stopping rule with a
regime statement, a look schedule in information time, a spending function, a futility boundary, a
guardrail rule, and a readout plan.

**Deliverable:** a one-page clause suitable for a pre-registration.
**Acceptance criteria:** regime named; schedule in information fractions; spending function justified
against the value of an early stop; futility binding status stated; guardrail schedule and margin
stated.
**Estimated time:** 45 minutes.

### Exercise 4 — The always-valid premium (calculation and judgement, 35 minutes)

A pilot has a fixed-horizon MDE of 2.0% relative and a break-even of 2.4% relative. Published
implementations report a two- to three-fold sample premium for equal power. Compute the MDE under the
premium, compare with the break-even, and state the design decision.

**Deliverable:** adjusted MDE range, comparison, recommendation with a stated condition.
**Acceptance criteria:** square-root conversion shown; recommendation distinguishes "cannot afford
continuous validity" from "the pilot is underpowered"; one alternative design named.
**Estimated time:** 35 minutes.

### Exercise 5 — Bayesian stopping critique (critique, 40 minutes)

A team proposes: "We will use a Bayesian engine with a default uninformative prior and stop when the
posterior probability that the lift is positive exceeds 95%." Identify assumptions, failure modes, and
the simulation you would require before approving.

**Deliverable:** a numbered critique and a simulation specification.
**Acceptance criteria:** the prior's role explained, not merely named; posterior validity distinguished
from decision error; simulation specifies null, prior, rule, and reported metric.
**Estimated time:** 40 minutes.

### Exercise 6 — Guardrail optional stopping (design, 45 minutes)

A guardrail is a damage rate with a 0.6% baseline and a 0.15-point margin, observed with a two- to
four-week lag. The primary is observed same-day. Design the guardrail rule and state what the final
analysis can conclude.

**Deliverable:** a guardrail clause plus a three-sentence limit statement.
**Acceptance criteria:** lag explicit and exposure window stated; monitoring distinguished from
stopping; limit statement says what degradation the design can exclude.
**Estimated time:** 45 minutes.

### Exercise 7 — The family (calculation, 30 minutes)

A dashboard shows 9 metrics, refreshes daily for 90 days, and filters by 4 regions and 3 channels.
Compute the family size and the independent-bound probability of at least one 5% result under a global
null, then propose a family structure a team could defend.

**Deliverable:** the arithmetic and a one-page family structure.
**Acceptance criteria:** family size multiplies all three axes; bound labelled; structure names one
primary, a guardrail family, and a descriptive tier.
**Estimated time:** 30 minutes.

### Exercise 8 — Write the pre-registration clause (design, 30 minutes)

Write the stopping-rule section of a pre-registration for a pilot you know or can invent, under 250
words.

**Deliverable:** the clause, with a version line and a date.
**Acceptance criteria:** regime named; boundary or date stated; schedule in information time;
guardrail rule stated; readout plan says what is reported at a stopping time; the clause states which
of the two regimes was fixed.
**Estimated time:** 30 minutes.

### Exercise 9 — Critique the meeting transcript (critique, 40 minutes)

A transcript reads: "Week 3: cost per drop is down 3.1%, p = 0.04. Let's keep going but I think we
have our answer. Week 5: p = 0.11, but that's the bank holiday. Week 9: p = 0.03, so we were right.
Let's roll it out and keep measuring." Identify every sequential error and rewrite the readout.

**Deliverable:** a numbered list of errors and a corrected three-paragraph readout.
**Acceptance criteria:** at least seven errors; each names the mechanism; rewrite states look count,
error rate, estimate with bias, and what the pilot can support.
**Estimated time:** 40 minutes.

### Exercise 10 — The dashboard redesign (design, 50 minutes)

Redesign an operational dashboard for a live pilot so it can be watched hourly without destroying the
inference. Specify what each tile shows, what it does not show, and what the viewer may do with it.

**Deliverable:** a tile-by-tile specification and a one-paragraph rationale.
**Acceptance criteria:** primary tile shows information fraction and boundary; operational tiles
labelled monitoring-only; early action routed to the pre-specified rule; at least one tile removed
rather than redesigned.
**Estimated time:** 50 minutes.

## 13. Self-check answers

### Exercise 1

```text
   formal looks    = 1 interim + 1 final = 2
   bound, 2 looks  = 1 - 0.95^2 = 0.0975      Bonferroni z = 2.241

   practical looks = 16 weekly + 32 ad hoc = 48
   bound, 48 looks = 1 - 0.95^48 = 0.9146     Bonferroni z = 3.238
```

The formal count understates exposure by a factor of twenty-four. The practical count governs the
decision, because the ad hoc checks are the ones that produce the "let's keep going" conversation.
The bound is an upper bound; a simulation with the actual spacing would give a lower number,
probably in the 30–40% range for 48 correlated looks.

### Exercise 2

```text
   adjusted MDE    = 0.80 x sqrt(1.028) = 0.8111 pp
   relative before = 0.80 / 12  = 6.67%
   relative after  = 0.8111 / 12 = 6.76%
   first-look effect = 4.562 x 0.90 = 4.11 pp = 34.2% relative
```

Early stopping is not realistically available: a 34% relative lift at the first look will not appear
if the true effect is 5–10%. O'Brien–Fleming costs 1.4% in MDE and buys almost nothing in early
stopping at this effect size. Its real value is the futility boundary.

### Exercise 3

A strong clause names a regime, gives information fractions rather than only dates, justifies the
spending function against the value of an early stop, and includes a futility boundary with its
binding status. It states the guardrail schedule and margin, and says what is reported at a stopping
time, including a bias-adjusted estimate. A weak clause names a spending function, stops there, and
leaves the guardrail and readout unspecified.

### Exercise 4

```text
   MDE at 2x sample = 2.0 / sqrt(2) = 1.41% relative
   MDE at 3x sample = 2.0 / sqrt(3) = 1.15% relative
```

Both are below the 2.4% break-even, so the pilot can afford continuous validity in principle. The
decision is not "always-valid is too expensive" but "always-valid is affordable here, and the design
should use it because operations will watch the dashboard regardless". At a break-even of 1.2%, only
the 2x case clears it. State which premium you assume and why.

### Exercise 5

1. A "default uninformative prior" is not uninformative for a decision rule; it implies a specific
   posterior shape and stopping behaviour.
2. Posterior validity under optional stopping is a statement about belief updating, not about the
   frequency of false claims across repeated pilots.
3. The 95% posterior threshold has no necessary relationship to a 5% error rate; its frequentist
   error rate depends on the prior and the true effect distribution.
4. A prior centred on implausibly large effects stops early on noise; one centred near zero runs long.
5. The rule needs a guardrail clause and a maximum sample, or a null runs indefinitely.

Simulation: true effects from the prior and from a point null, 10,000 pilots per scenario, the actual
look schedule; report the null error rate, power at the break-even, and the median sample.

### Exercise 6

A strong clause separates monitoring from stopping. Monitoring: the damage rate is displayed daily
with its lag stated, and no decision is taken before the pre-specified looks. Stopping: at each
pre-specified look, a one-sided non-inferiority test against the 0.15-point margin; if exceeded, the
pilot pauses pending review. The final analysis covers the full exposure window plus a four-week tail.

Limit statement: "The pilot can exclude a damage-rate increase larger than X points at the final look.
It cannot exclude a smaller increase, and the early looks cover a shorter exposure window, so an early
stop is not evidence of guardrail safety."

### Exercise 7

```text
   family size = 9 x 90 x (4 x 3) = 9,720
   bound       = 1 - 0.95^9720  (indistinguishable from 1)
```

Structure: one primary metric with a pre-specified boundary; a guardrail family of no more than three
metrics with a stated correction and a shared margin; everything else descriptive, with no p-values on
the tiles. A reader who wants to promote a descriptive metric to a decision must say so before the
next look, not after this one.

### Exercise 8

A strong clause is under 250 words, names the regime explicitly, gives the boundary or date, gives the
schedule in information time, states the guardrail rule, and states what is reported at a stopping
time. It carries a version and a date. A weak clause says "we will monitor results and stop if
significant", which describes peeking rather than ruling it out.

### Exercise 9

1. No pre-specified look schedule, so the week-3 p-value is one of an unknown number of tests.
2. The week-3 result is treated as an answer while the pilot continues, a decision on a peek.
3. The week-5 p-value is explained away by a bank holiday, a post hoc exclusion.
4. The week-9 result is treated as confirmation, when it is a re-test on overlapping data.
5. Rollout is decided at week 9, before the planned end.
6. "Keep measuring" after rollout is not a controlled comparison; the control is gone.
7. The reported effect is a stopping-time estimate and biased upward.
8. Three p-values are presented as independent evidence when they are three looks at one path.
9. No guardrail is mentioned.

Corrected readout: the pilot had no pre-specified stopping rule; the metric was inspected at least nine
times; the nominal 5% does not apply; the week-9 estimate is X with a sequential interval of Y; the
pilot supports a feasibility decision and not a claim about effect size.

### Exercise 10

A strong design shows the primary tile with the estimate and its sequential interval, the information
fraction, and the boundary at that fraction. It labels operational tiles monitoring-only and routes a
viewer who wants to act early to the pre-specified rule. It removes at least one tile, typically a
segment breakdown that invites shopping, and says why. A weak design adds a warning label to every
tile and changes nothing about what is shown.

## 14. Interview questions

**1. "What is the peeking problem?"**

Strong: a p-value is calibrated for one test at a planned sample; inspecting repeatedly and stopping
when it crosses a threshold converts the error rate from the probability of one exceedance to the
probability that a random walk ever crosses a line, which grows with the number and earliness of
looks, and gives a number — 5% at one look, roughly 25% at twenty daily looks. Weak: "looking too
often biases the result", which names neither the mechanism nor the magnitude.

**2. "Our result is p = 0.03 on day four. Why isn't that a 3% chance we are wrong?"**

Strong: the p-value was computed under a fixed-horizon model, and the day it was computed was chosen
by the data; the correct statement is about the probability the path would have crossed at some
point, and about the estimate at the crossing being biased away from zero. Weak: "p-values are often
misunderstood", which is true and useless.

**3. "O'Brien–Fleming or Pocock?"**

Strong: OBF is strict early and near-nominal at the end, so it costs little if you do not stop early
and makes early claims expensive; Pocock has a nearly constant boundary, so early stopping is easier
but the maximum sample and final boundary are worse; for a small effect neither early boundary is
reachable, so the futility boundary is the one that matters. Weak: "OBF is more conservative", with
no statement about cost or when it is wrong.

**4. "What is an always-valid confidence interval and what does it cost?"**

Strong: a sequence of intervals with joint coverage over all sample sizes, so the guarantee survives
any stopping rule; the cost is width, and published implementations report roughly two to three times
the sample for equal power when the effect is correctly anticipated, with a smaller premium when the
effect is smaller than assumed. Weak: "a confidence interval you can check anytime", with no statement
about width or power.

**5. "Is Bayesian optional stopping safe?"**

Strong: the posterior is a valid summary of belief at any stopping time, but a rule that stops when
the posterior crosses a threshold can still produce a high rate of false claims, especially with a
default or misspecified prior; the guarantee depends on the prior being genuine and the model
approximately right. Weak: "yes, Bayesians can stop whenever", which is the claim the literature
spent a decade qualifying.

**6. "A manager will check the pilot every morning. What do you do?"**

Strong: design for it — fix a spending function or an always-valid boundary before launch, put the
boundary on the dashboard next to the estimate, separate operational tiles from the decision tile, and
write the rule into the pre-registration. It does not try to stop the manager looking, because that
instruction is unenforceable. Weak: "tell them not to peek", an organisational wish presented as a
statistical control.

**7. "How do you handle guardrails under continuous monitoring?"**

Strong: guardrails are a non-inferiority question with their own margin, base rate, and lag; they need
their own schedule and boundary, and the schedule should match the primary or the asymmetry should be
justified; early stopping truncates the window in which slow harms appear. Weak: "we check the
guardrails at the end", which leaves the highest-exposure period unmonitored.

**8. "How does peeking interact with testing twelve metrics?"**

Strong: the family is looks times metrics times segments; correcting one axis leaves most of the error
in place; the practical structure is one primary, a small guardrail family, and a descriptive tier,
with FDR control if a family-level guarantee is needed. Weak: "we correct for multiple metrics", with
no statement about the time axis.

**9. "What goes in the pre-registration about stopping?"**

Strong: the regime, the schedule in information time, the boundary values or the function generating
them, the futility rule and its binding status, the guardrail rule and margin, what is reported at a
stopping time, and a version and date. Weak: "the sample size and the metric", which leaves the
stopping rule — the thing that determines the error rate — unspecified.

**10. "The pilot is significant at week four and the committee wants to stop. What do you say?"**

Strong: it depends whether the boundary at week four was pre-specified and crossed — if so, the stop
is valid and the estimate should be reported with its bias adjustment; if not, the result is one look
in an unknown sequence, the estimate is the boundary plus noise, and the committee is about to bank a
number that will not replicate. It offers the pre-specified analysis and the look log. Weak: "let's
run to the end", correct advice delivered without the mechanism and therefore likely to be overruled.

## 15. Key terms

- **Peeking** — inspecting a test before the planned analysis and letting it influence a decision.
- **Optional stopping** — choosing when to stop on the basis of accumulating data.
- **Fixed-horizon test** — one analysis at a pre-specified sample size.
- **Information fraction** — share of planned statistical information accrued.
- **Information time** — the scale on which looks are spaced, proportional to variance accrued.
- **Look log** — the timestamped record of who looked, when, and what they saw.
- **Group sequential design** — a small pre-specified number of analyses with boundaries.
- **Alpha spending function** — a rule allocating Type I error across information time.
- **O'Brien–Fleming boundary** — strict early, near-nominal at the end, small sample cost.
- **Pocock boundary** — nearly constant across looks, easier early stopping, larger sample cost.
- **Lan–DeMets method** — applying a spending function when look timing is not fixed in advance.
- **Information ratio** — sequential maximum sample divided by fixed-horizon sample for equal power.
- **Futility boundary** — a lower boundary permitting a stop when the effect is unpromising.
- **Non-binding futility** — a futility rule that may be overridden without inflating alpha.
- **Always-valid inference** — inference whose guarantee holds at every stopping time.
- **Anytime-valid p-value** — a p-value whose probability of ever falling below alpha is at most alpha.
- **Confidence sequence** — a sequence of intervals with joint coverage over all sample sizes.
- **mSPRT** — mixture sequential probability ratio test; mixed likelihood ratios forming a martingale.
- **Martingale** — a process whose expected next value equals its current value.
- **Ville's inequality** — the bound that a nonnegative martingale exceeds `1/alpha` with probability at most alpha.
- **Type I error rate** — probability of rejecting a true null, per experiment.
- **Family-wise error rate** — probability of at least one false rejection in a family.
- **False discovery rate** — expected share of false rejections among those made.
- **Multiplicity** — error inflation from testing many hypotheses across metrics, segments, or time.
- **Conditional exaggeration** — upward bias in the estimate among results that stop early.
- **Stopping rule** — a pre-specified mapping from accumulating data to a stop decision.
- **Analysis date** — the single date fixed in advance for the primary analysis.
- **Pre-registration** — a dated document fixing hypothesis, metric, MDE, stopping rule, decision rule.
- **Guardrail metric** — a metric that must not degrade beyond a margin.
- **Non-inferiority margin** — the largest tolerated degradation.

## 16. Sources

**Grading convention.** A peer-reviewed or audited; B regulatory, official, or registered; C company
case, investor claim, preprint, or conference paper; D vendor or consultancy marketing or
documentation; E press, anecdote, or course notes. Vendor and documentation pages are dated where a
date is visible and marked as retrieved in November 2026 where it is not.

**Peer-reviewed methodology**

- Armitage, P., McPherson, C. K., and Rowe, B. C. (1969). "Repeated significance tests on accumulating
  data." *Journal of the Royal Statistical Society, Series A*. Grade A.
- Pocock, S. J. (1977). "Group sequential methods in the design and analysis of clinical trials."
  *Biometrika*. Grade A.
- O'Brien, P. C., and Fleming, T. R. (1979). "A multiple testing procedure for clinical trials."
  *Biometrics*. Grade A.
- DeMets, D. L., and Lan, K. K. G. (1994). "Interim analysis: the alpha spending function approach."
  *Statistics in Medicine*. doi:10.1002/sim.4780131308. Grade A.
- Jennison, C., and Turnbull, B. W. (2000). *Group Sequential Methods with Applications to Clinical
  Trials*. Chapman and Hall/CRC. Grade A.
- Johari, R., Koomen, P., Pekelis, L., and Walsh, D. (2022). "Always Valid Inference: Continuous
  Monitoring of A/B Tests." *Operations Research*, 70(3). doi:10.1287/opre.2021.2135. Grade A.
- Howard, S. R., Ramdas, A., McAuliffe, J., and Sekhon, J. (2021). "Time-uniform, nonparametric,
  nonasymptotic confidence sequences." *Annals of Statistics*, 49(2); arXiv:1810.08240. Grade A.
- Rouder, J. N. (2014). "Optional stopping: No problem for Bayesians." *Psychonomic Bulletin and
  Review*, 21, 301–308. Grade A.
- Sanborn, A. N., and Hills, T. T. (2014). "The frequentist implications of optional stopping on
  Bayesian hypothesis tests." *Psychonomic Bulletin and Review*, 21, 283–300. Grade A.
- Tendeiro, J. N., Kiers, H. A. L., and van Ravenzwaaij, D. (2022). "Worked-out examples of the
  adequacy of Bayesian optional stopping." *Psychonomic Bulletin and Review*, 29, 70–87. Grade A.

**Standards and regulatory**

- U.S. Food and Drug Administration (2019). *Adaptive Designs for Clinical Trials of Drugs and
  Biologics: Guidance for Industry*. November 2019; Federal Register notice 2 December 2019.
  Grade B.
- ICH E9. *Statistical Principles for Clinical Trials*. International Council for Harmonisation.
  Grade B.
- ICH E20 on adaptive designs. In development at the time of writing; check current status before
  relying on any provision. Grade B, with status caveat.

**Company, conference, and preprint**

- Johari, R., Pekelis, L., and Walsh, D. (2017). "Peeking at A/B Tests: Why It Matters, and What to Do
  About It." KDD. Grade C.
- Kohavi, R., and Chen, N. (2024). "False Positives in A/B Tests." KDD.
  doi:10.1145/3637528.3671631. Grade C.
- Waudby-Smith, I., Arbour, D., Sinha, R., Kennedy, E. H., and Ramdas, A. "Time-uniform central limit
  theory and asymptotic confidence sequences." arXiv:2103.06476. Preprint. Grade C.
- Maharaj, A., et al. (2023). "Anytime-Valid Confidence Sequences in an Enterprise A/B Testing
  Platform." WWW '23 Companion; arXiv:2302.10108. Grade C.
- Deng, A., Lu, J., and Chen, S. (2016). "Continuous Monitoring of A/B Tests without Pain: Optional
  Stopping in Bayesian Testing." IEEE DSAA. doi:10.1109/dsaa.2016.33. Grade C.
- de Heide, R., and Grünwald, P. D. "Why optional stopping is a problem for Bayesians."
  arXiv:1708.08278. Preprint. Grade C.

**Vendor and platform documentation**

- Optimizely. "Statistical analysis methods overview." docs.optimizely.com. Grade D, retrieved
  November 2026.
- Optimizely. "How and why statistical significance changes over time in Optimizely Experimentation."
  docs.optimizely.com. Grade D, retrieved November 2026.
- Pekelis, L., Walsh, D., and Johari, R. "The New Stats Engine." Optimizely white paper. Grade D,
  undated in the retrieved copy.
- GrowthBook. "Sequential Testing." docs.growthbook.io. Grade D, retrieved November 2026.
- GrowthBook. "GrowthBook's Statistics." docs.growthbook.io. Grade D, retrieved November 2026.
- Adobe. "Statistical calculations used by Adobe Journey Optimizer Experimentation." Experience
  League documentation. Grade D, retrieved November 2026.
- StataCorp. "gsbounds — Group sequential designs." *Stata Manual*. Grade B, official software
  documentation; no publication date in the retrieved copy.

**Secondary simulations and teaching material**

- Bell Statistics. "The Peeking Problem" and "Always-Valid Inference and mSPRT Explained." Glossary
  pages. Grade D, undated, retrieved November 2026.
- livevariant.ai. "How much does peeking at an A/B test cost? A simulator, and a correction to my own
  page." Grade D, undated, retrieved November 2026.
- Socratopia Library. "Statistics at Work V: Experiments and A/B Testing," chapter 13. Course notes.
  Grade E, retrieved November 2026.

**Note on citations.** Figures attributed to simulators are reported as those sources state them and
are graded D for that reason. The worked example is constructed for teaching, its fixed-horizon inputs
are carried from Topic 06, and its arithmetic was checked against the block in section 7. Where a
bibliographic detail could not be verified, the entry says so rather than supplying a plausible volume
or page range.

## 17. Links

- Module map and topic sequence: [`_index.md`](_index.md)
- The claim and decision rule this stopping rule protects: [`01_claim-hypothesis-design.md`](01_claim-hypothesis-design.md)
- The unit whose independence the sequential variance assumes: [`02_randomisation-units.md`](02_randomisation-units.md)
- Cluster designs, where looks and clusters interact: [`03_ab-cluster-designs.md`](03_ab-cluster-designs.md)
- Geo experiments, where units are few and looks are expensive: [`04_geo-experiments.md`](04_geo-experiments.md)
- Switchbacks, where time is the unit and autocorrelation matters: [`05_switchback-pilots.md`](05_switchback-pilots.md)
- The fixed-horizon MDE this topic corrects: [`06_power-mde.md`](06_power-mde.md)
- Interference, which changes the variance a boundary assumes: [`07_interference-spillover.md`](07_interference-spillover.md)
- Incrementality without an experiment: [`09_mmm-incrementality.md`](09_mmm-incrementality.md)
- Appraising a vendor's claims about its sequential engine: [`10_vendor-evidence-appraisal.md`](10_vendor-evidence-appraisal.md)
- Hypothesis tests and the error rates this topic makes robust to repeated looking: [`../04_statistics_causal_inference/03_hypothesis-tests.md`](../04_statistics_causal_inference/03_hypothesis-tests.md)
- Autocorrelation, which matters when looks are close in time: [`../04_statistics_causal_inference/05_autocorrelation-robust-errors.md`](../04_statistics_causal_inference/05_autocorrelation-robust-errors.md)
- The three-flow model, which explains why some outcomes are observed late: [`../01_supply_chain_foundations/01_three-flows-one-network.md`](../01_supply_chain_foundations/01_three-flows-one-network.md)
- Module survey file: [`../../../course/05_experimentation_pilots_evidence.md`](../../../course/05_experimentation_pilots_evidence.md)
- Statistics and causal inference survey, including multiplicity: [`../../../course/04_statistics_causal_inference.md`](../../../course/04_statistics_causal_inference.md)
- Return to the part map: [`../_index.md`](../_index.md)
