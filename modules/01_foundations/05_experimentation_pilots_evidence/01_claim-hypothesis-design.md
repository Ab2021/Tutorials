# From Claim to Hypothesis to Design

> Module 05 — Experimentation, Pilots, and Evidence
> Topic 01 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 04, topics 01–03
> Version 2026.5

---

## 1. Why this topic exists

A business claim is a sentence someone believes. A testable hypothesis is a sentence that can be
wrong. The distance between them is crossed on paper, in about an hour, before anyone opens a
notebook.

Two claims arrive in the same week. The commercial director says **"The promotion worked."**
Everyone nods. Nobody can say what would have happened without it, what "worked" means in currency,
over what window, or for which stores. The analytics team is asked to "validate" the claim. What
follows is a before-and-after chart that goes up, and a decision to repeat the promotion.

The operations director says **"Our service level is world class."** This is not a claim about a
quantity. It is a claim about rank, and the rank has no reference set. World class compared with
whom, on which definition of service, over which period, at which level of aggregation? A company
can hold 97% on-time in-full (OTIF) at order level, 71% at item-location level, and 99% on the
orders it chose to measure, and all three numbers can be correct.

Neither sentence can be tested. Both will be acted on. That is the problem this topic exists to
solve.

### 1.1 Three ways a claim fails to become evidence

**Untestable.** "World class" has no comparator, no unit, and no threshold. No amount of data will
resolve it, because there is nothing to resolve. The repair is a conversation that produces a
comparator and a number.

**Testable but undecidable at the available sample size.** "The new slotting policy raised picks
per hour by 2%." Precise, at last. The pilot has 18 pickers over 4 weeks, and the minimum
detectable effect (MDE) at 80% power is 7%. The experiment cannot answer its own question and will
produce a point estimate someone acts on anyway. The repair is a power calculation before the
pilot, which is Topic 06.

**Decidable but unactionable.** The pilot shows a 4.1 percentage point OTIF improvement with an
interval excluding zero. Everyone celebrates. Nobody asks whether 4.1 points pays for the extra
freight runs, and the answer, computed later, is that it does not. The repair is a pre-specified
decision rule with a cost-weighted threshold, which is section 5.7.

The three are ordered by cost. The first wastes a meeting. The second wastes a pilot. The third
wastes a rollout.

### 1.2 Why the gap exists

Business language is about intent and momentum; statistical language is about quantities and
comparisons. Four mismatches recur.

| Mismatch | Example | What it hides |
|---|---|---|
| Verbs carry unstated counterfactuals | "drove", "lifted", "unlocked", "delivered savings" | A causal claim with no comparison group |
| Adjectives carry unstated reference sets | "world class", "best in class", "leading" | A ranking with no peer group, metric, or period |
| Nouns carry unstated definitions | "service level" | At least four different measures with different denominators |
| Numbers carry unstated bases | "we improved by 30%" | Whether the base is last year, the plan, or a control group |

The mismatches are not a communication problem that clearer slides will fix. They are category
errors, and each has to be resolved explicitly or it is resolved silently and wrongly.

### 1.3 What the discipline buys

The chain this topic teaches — claim, question, estimand, design, metric, decision rule — removes a
way for the analysis to be wrong at every joint. It also makes the analysis falsifiable in advance.
An organisation that can say no to its own pilot learns roughly twice as fast, because half its
pilots are secretly approved projects with an evaluation attached.

> **Note:** Falsifiability here is operational, not philosophical: a sentence a number could
> contradict, plus a written statement of which number and which direction would do it.

### 1.4 Provenance worth knowing

In clinical drug trials, two studies with the same drug, endpoint, and data could reach opposite
conclusions depending on how they handled patients who stopped treatment. That is a disagreement
about the question, not the arithmetic. The International Council for Harmonisation (ICH) responded
with E9(R1), an addendum to its statistical principles guideline, adopted at Step 4 on 20 November
2019 and effective at European Medicines Agency Step 5 on 30 July 2020; it requires an estimand to
be defined before analysis, through five attributes.

Separately, a replication problem in applied social science produced the same shape of response:
write the analysis plan before seeing the outcome data. Neither binds a supply chain analyst. Both
are the most carefully worked-out public statements of a problem commercial analytics rediscovered
on its own.

## 2. Learning outcomes

By the end of this topic you can:

1. **Convert** a stated business claim into a written question by naming the population, the
   comparison, the verb's counterfactual, and the window, and **identify** which was missing.
2. **Write** a falsifiable hypothesis with a direction, a magnitude, and a null, and **state** what
   observation would refute it.
3. **Produce** an estimand in six slots for a supply chain pilot, and **distinguish** the unit of
   assignment from the unit of analysis.
4. **Select** a design that can estimate the stated estimand, and **reject** at least one that
   cannot, giving the reason in terms of the estimand rather than the sample size.
5. **Derive** a cost-weighted decision threshold from a break-even calculation and a loss ratio,
   and **explain** why a significance level of 0.05 is not a decision rule.
6. **Classify** an analysis as confirmatory or exploratory, and **write** a pre-analysis plan short
   enough to be used and specific enough to constrain the result.
7. **Diagnose** which link in the chain broke when a completed pilot produced an unusable answer.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a chain of six links running from a sentence someone said to an action someone
takes, with a specific way to break at every joint.

```text
   CLAIM
   "The promotion worked."  /  "Our service level is world class."
     |
     |  LINK 1 - interrogate: which population, which comparison,
     |           which counterfactual does the verb imply, which window?
     v
   QUESTION
   One sentence, with a comparator and a period. Still not a quantity.
     |
     |  LINK 2 - make it refutable: direction + magnitude + null.
     |           What number, in which direction, would prove this wrong?
     v
   HYPOTHESIS
   "Twice-weekly replenishment raises OTIF by at least 3 percentage points."
     |
     |  LINK 3 - quantify: population, unit, outcome, time, intervention,
     |           comparison. Six slots, each computable by a stranger.
     v
   ESTIMAND
   The exact quantity to be estimated, fixed before data are touched.
     |
     |  LINK 4 - build the comparison: assignment mechanism, who is
     |           compared with whom, over what window, at what grain.
     v
   DESIGN
   A structure that can produce a credible substitute for the missing half.
     |
     |  LINK 5 - measure: the outcome variable, at the estimand's grain,
     |           with a stated denominator and a stable definition.
     v
   METRIC
   The number that will appear in the report, defined the same way twice.
     |
     |  LINK 6 - convert to action: threshold from costs, not from convention.
     v
   DECISION RULE
   "Roll out if the lower bound exceeds 3.3 pp; extend if it does not."
     |
     v
   ACTION
```

The picture encodes one claim: **each link constrains the next, and the chain is built backwards
from the decision.** You start at the bottom, with the action and its costs, and work upward until
you reach the sentence someone said. Analysts who build it forwards produce designs that cannot
answer the question, which is why so many pilots end with "the results were mixed".

Six properties follow from the picture.

| Property | What it means in practice |
|---|---|
| Link 1 does most of the work | Most claims are not wrong, they are underspecified: "worked" becomes testable once someone names the margin, the stores, and the four weeks |
| The hypothesis must be able to lose | No magnitude and no direction means nothing can count against it, and it will still feel rigorous because it is written down |
| The estimand is a contract; the design is the delivery | If the unit is the lane and assignment is at the supplier, only cluster-aware designs qualify |
| The metric is where silent substitution happens | Revenue for margin, shipments for demand, order-level for line-level: the adjacent metric was easier to extract |
| The decision rule is where statistics becomes management | A confidence interval is not a decision; converting it needs the cost of acting wrongly and the cost of not acting |
| Everything above the line is written before the data | Claim through decision rule are pre-specified; estimate, interval, and sensitivity analyses come afterwards |

### 3.1 The two-track model

The chain describes confirmatory work. Real projects also explore, and the two must be kept apart
without suppressing the second.

```text
                        ALL ANALYST ACTIVITY ON A PROJECT
                                     |
              +----------------------+----------------------+
              |                                             |
     CONFIRMATORY TRACK                            EXPLORATORY TRACK
     planned before data                           unplanned, data-driven
              |                                             |
     has a pre-analysis plan                       has no plan, by definition
     one primary estimand                          many candidate estimands
     error rate controlled                         error rate uncontrolled
              |                                             |
     supports "we should roll out"                 supports "we should test this"
              |                                             |
     reported first, in full                       reported separately, labelled
              |                                             |
              +----------------------+----------------------+
                                     |
                          FEEDS THE NEXT CYCLE
                    (exploratory finding becomes the next
                     confirmatory hypothesis, on new data,
                     with a new plan)
```

The arrow is the whole point. An exploratory finding is not a defective confirmatory finding; it is
the input to the next one. The rule that makes the system work is that the same data cannot play
both roles for the same claim. A subgroup discovered in the data and then reported with its own
p-value is a confirmatory claim computed on exploratory evidence, and its error rate is unknown.

### 3.2 The chain as sentences you say out loud

Ask, in order: "Compared with what, over what period, for whom?"; "What number would prove you
wrong?"; "What is one observation, and what is one unit of assignment?"; "What would have to be true
for this comparison to be fair?"; "Is that the same thing the estimand named, at the same grain?";
and "What does it cost if we are wrong in each direction?" An analyst who can say these without
notes will outperform one who can run a regression and cannot.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Claim | A sentence asserting something about the business that the speaker believes | Treated as a finding because it was stated confidently | Free text, usually with an unstated comparator |
| Business question | A claim rewritten so that an answer could change a decision | Treated as a restatement of the claim | Free text with a named decision and owner |
| Hypothesis | A statement about a quantity that an observation could contradict | Used as a synonym for "guess" or "assumption" | Directional statement with a magnitude |
| Null hypothesis | The statement that the effect is at or below a stated value, usually zero | Always set to zero, even when the decision threshold is not zero | Effect value on the outcome scale |
| Alternative hypothesis | The complement of the null, including the direction of interest | Stated as "there is an effect", with no direction | Effect value or region |
| Falsifiability | The property that some observable result would count against the statement | Confused with "we can compute something" | Property of a statement |
| Minimum worthwhile effect | The smallest true effect that would justify changing the decision | Confused with the MDE the design can detect | Outcome units |
| Estimand | The precise quantity to be estimated, fixed before data are analysed | Used as a synonym for "metric" or "KPI" | Structured statement, six slots |
| Estimator | The procedure that produces an estimate of the estimand | Named when the estimand was requested | Method name |
| Estimate | The numerical value an estimator produces on a dataset | Presented as if it were the estimand | Same unit as the estimand |
| Decision rule | The mapping from an estimate to an action, fixed in advance | Invented after seeing the estimate | If-then specification |
| Decision threshold | The effect size above which the action is taken | Confused with the significance level | Outcome units |
| Break-even effect | The effect at which the intervention exactly pays for its running cost | Omitted entirely | Outcome units |
| Loss function | The cost, in currency, of each possible error as a function of its size | Assumed symmetric, or omitted | Currency per unit of error |
| Cost of a false positive | What you lose by acting when the effect is not there or is too small | Assumed to equal the cost of a false negative | Currency per decision |
| Cost of a false negative | What you lose by not acting when the effect is real | Ignored, because it does not appear in a P&L | Currency per decision |
| Loss ratio (k) | Cost of a wrong action divided by cost of a wrong inaction | Left implicit inside a significance level | Dimensionless ratio |
| Significance level (alpha) | The pre-specified probability of rejecting a true null | Used as if it were a decision threshold | Probability, one- or two-sided |
| Power | The probability of detecting an effect of a stated size, if it is true | Quoted without the effect size it refers to | Probability at a stated effect |
| Minimum detectable effect (MDE) | The smallest true effect a design detects with stated power and alpha | Reported as if smaller effects do not exist | Outcome units |
| Pre-analysis plan (PAP) | A document fixing the estimand, primary outcome, analysis, and decision rule before outcome data are examined | Written after the pilot starts, or after the data arrive | Document with a timestamp |
| Preregistration | Public, time-stamped registration of a study and its primary outcomes | Confused with a PAP; registration is thinner | Registry record identifier |
| Confirmatory analysis | An analysis planned in advance, whose error rate is controlled | Applied to any analysis with a p-value | Analysis label |
| Exploratory analysis | An analysis generated by the data, whose error rate is not controlled | Reported with confirmatory language | Analysis label |
| HARKing | Hypothesising after the results are known | Assumed to be deliberate; usually it is not | Behaviour, not a statistic |
| Specification search | Trying many model specifications and reporting the preferred one | Defended as "robustness" | Count of specifications tried |
| Multiple comparisons | Testing many hypotheses without adjusting the error rate | Ignored when the tests sit in different slides | Count of tests |
| Family-wise error rate (FWER) | The probability of at least one false rejection across a family of tests | Confused with the per-test alpha | Probability |
| False discovery rate (FDR) | The expected proportion of rejections that are false | Confused with FWER | Proportion |
| Primary outcome | The single outcome the decision rule is written against | One of eleven outcomes, all reported | Named variable |
| Secondary outcome | A pre-specified supporting outcome, with a stated role | Promoted to primary when it is significant | Named variable |
| Subgroup analysis | A pre-specified analysis within a defined stratum | Run across every cut of the data | Stratum definition |
| Intercurrent event | A post-assignment event that changes what the outcome means | Treated as missing data | Event definition plus date |
| Pilot | A small-scale implementation used to estimate feasibility or effect | Assumed to produce a decision-grade estimate | Study label |
| A/B test | A randomised comparison of two or more variants | Used to describe any comparison of two periods | Design label |
| Cluster | A group of units randomised together | Ignored when computing standard errors | Assignment identifier |
| Contamination | Treatment reaching units assigned to control | Discovered after the result | Proportion of control units exposed |
| Stopping rule | The pre-specified rule for when data collection ends | Set by the calendar or by the result | Date or information fraction |
| External validity | The extent to which an estimate transfers to another population or period | Assumed from a significant result | Scope statement |
| Net benefit | A single currency-scale summary combining benefits and harms at a stated weight | Confused with net profit | Currency per unit |

Three pairs carry most of the misunderstanding.

**Minimum worthwhile effect and minimum detectable effect.** The first comes from the business: the
smallest true effect that justifies acting. The second comes from the design: the smallest true
effect the sample can distinguish from the null at stated power. A pilot is informative only if the
MDE is below the minimum worthwhile effect; otherwise it cannot separate a good effect from a
trivial one. The two numbers have different owners.

**Significance level and decision threshold.** Alpha concerns the probability of a false rejection
under a null that is usually zero. A decision threshold concerns the effect size at which the action
pays for itself. They coincide only if the break-even effect is zero and the loss ratio happens to
match the one alpha implies, and in supply chain work the break-even effect is rarely zero.

**Confirmatory and exploratory.** The distinction is not about quality or sample size but about
whether the analysis was planned before the data were seen. A confirmatory analysis of a bad design
is still confirmatory and still bad; the error is claiming one's evidential status for the other.

## 5. The mechanics, step by step

This section walks the chain from the bottom up, because that is the direction that works. Each
step states its input, its output, and the way it usually breaks.

### 5.1 Step 0 — Name the decision, the owner, and the deadline

Before the claim, establish what will be done differently. Ask three questions and write the
answers: what action is contingent on the answer (roll out, renew, keep, stop); who owns that action
(a role, not a department); and by when the answer must exist, since an estimate delivered after the
decision has the same value as no estimate.

The output is one line: "The logistics director will decide on 15 March whether to extend
twice-weekly replenishment from 24 pilot lanes to 240 lanes." That line supplies three estimand
slots for free: population, intervention, horizon. The decision date constrains the design far more
than analysts expect.

**Where it breaks.** The owner cannot be named because the decision is collective. That is a
finding: collective decisions produce analyses that satisfy everyone and change nothing.

### 5.2 Step 1 — Interrogate the claim

A claim becomes a question through four interrogations, run in this order.

**Which comparator?** Every claim implies a comparison, usually unstated. "The promotion worked"
could compare the promoted period with the same stores before; with unpromoted stores in the same
weeks; with a forecast; with the plan; with last year's equivalent weeks; or with the counterfactual
margin. These are six questions. Force the choice.

**Which population?** "Our service level is world class" does not say whose. All customers,
customers above a spend threshold, all lanes, one region? Naming the population also names the
exclusions, and the excluded group is often where the problem lives.

**Which counterfactual does the verb carry?** Classify the verb. This takes ten seconds and settles
the evidence standard for the whole project.

| Verb family | Implied counterfactual | Question type | Evidence required |
|---|---|---|---|
| Drove, caused, lifted, improved, impacted, unlocked | A world without the intervention | Causal | A design with a credible comparison group |
| Will, expect, forecast, plan for | An unobserved future under a stated plan | Predictive | A validated model and a stated plan |
| Totalled, reached, recorded, ran at | None | Descriptive | A defined population, period, and counting rule |
| Is better than, leads, outperforms | A named peer set | Descriptive comparison | A peer set, a metric, and a period |
| Should, recommend, choose | A choice among outcomes that occur anyway | Predictive plus a loss | Predicted outcomes and a cost model |

**Which window?** A promotion measured over four weeks and over thirteen weeks answer different
questions, because the second includes the post-promotion trough. A supplier policy measured over a
quarter and over a year answer different questions, because the second includes the contract
renewal.

The output is one sentence: "Did [intervention], compared with [comparator], change [outcome] for
[population] over [window]?"

**Where it breaks.** Stakeholders agree to all four answers individually and disagree when they see
them together. That is the most valuable moment in the project: a disagreement about the comparator
is a disagreement about what the business is trying to learn.

### 5.3 Step 2 — Write the hypothesis with a magnitude

A hypothesis has three parts, all required. **Direction**: higher, lower, more than, less than; a
two-sided hypothesis with no direction has no decision content. **Magnitude**: a number in the
outcome's units, equal to the minimum worthwhile effect, not a round number chosen for convenience.
**Null**: the value the test is run against, usually zero and often it should not be.

Weak: "Twice-weekly replenishment improves OTIF."

Usable: "Twice-weekly replenishment raises line-level OTIF by at least 3.3 percentage points on the
pilot lanes, against a null of no change, over 26 weeks."

The 3.3 is the break-even effect derived in section 5.7. Hypothesis and decision rule use the same
number deliberately: a magnitude that differs from the decision threshold produces a test that
answers a question nobody is deciding.

**Where it breaks.** The magnitude is set by the design ("what we can detect") rather than by the
business ("what would be worth doing"). This inverts the logic and produces pilots designed to
detect effects that do not matter.

### 5.4 Step 3 — Write the estimand in six slots

| Slot | The question it answers | The test of a good answer |
|---|---|---|
| Population | Which units, meeting which criteria, in which period? | You can count them in the data before the pilot starts |
| Unit | What is one observation, and what is the unit of assignment? | Both are stated, even when they coincide |
| Outcome | Which variable, measured how, aggregated how, over what window? | A second analyst computes the same number |
| Time | When does follow-up start, and how long does it run? | Time zero is a decision point, not a data-availability accident |
| Intervention | What is done, to whom, at what dose, for how long, and what is the comparator? | The comparator is described as precisely as the treatment |
| Comparison | Which summary measure, on which scale, contrasting which conditions? | The scale is stated with the number |

Three slots deserve extra attention.

**The unit slot has two values, and both are mandatory.** Replenishment frequency is assigned to a
lane and measured on purchase-order lines; a slotting policy is assigned to a distribution centre
(DC) and measured on pick transactions. When the two differ, observations within an assignment unit
are correlated, and standard errors computed at the measurement grain are too small. State both
always, because the act of stating them is what catches the error.

**The time slot is where pilots quietly change their own question.** Define time zero as the moment
of assignment and the horizon as the period over which the decision applies. If the decision is an
annual contract, a four-week pilot answers a different question, and answers it optimistically.

**The comparison slot must name a scale.** "OTIF rose by 4.1 percentage points" and "OTIF rose by
4.7%" are different numbers from the same data, and a reader not told which is quoted assumes the
larger.

**Where it breaks.** One slot cannot be filled because the data do not exist. That is not a reason
to proceed with five slots: the estimand is not estimable, and the project needs new instrumentation
or a different question.

### 5.5 Step 4 — Choose a design that can estimate this estimand

Design selection is a feasibility test against the estimand, not a preference. Run five
eliminations.

1. **Can the analyst assign the intervention?** If not, you must argue identification.
2. **At what grain?** It must be at or finer than the grain at which the decision is made.
3. **How many assignment units are there?** This number, not the row count, sets the precision: a
   million order lines nested in twenty-four lanes is still twenty-four units of standard error.
4. **What is the smallest effect the design can detect?** If the MDE exceeds the minimum worthwhile
   effect, no analysis will fix the design.
5. **What could contaminate the comparison?** Shared suppliers, carriers, and demand, plus planners
   who apply the new rule to control lanes. Each needs a design response or a diagnostic.

The output is four lines: assignment mechanism, assignment grain, number of assignment units, and
the identification assumption. Topics 02 to 05 own the specific designs; this step only requires
that the design be selected against the estimand rather than against convenience.

| Design | Requires | Can estimate | Cannot estimate |
|---|---|---|---|
| Individual randomisation | Unit-level assignment, no interference | Average effect across individuals | Anything about a decision at a coarser grain |
| Cluster randomisation | Enough clusters, known intra-cluster correlation | Cluster-level effect, with wider intervals | Small effects, with few clusters |
| Stratified or paired | A credible pre-treatment matching variable | Effect within strata, with less variance | Effects that vary by the matching variable |
| Switchback | A treatment that can be alternated over time | Effect within unit over time | Effects with long carryover |
| Stepped wedge | A rollout that must be sequential | Effect, confounded with time unless modelled | Clean separation of time and treatment without assumptions |
| Difference-in-differences | Parallel trends and an untreated group | Effect on the treated, under parallel trends | Effects when trends already differ pre-treatment |
| Before-and-after | Nothing | A descriptive change | Any causal effect, without an explicit argument |

**Where it breaks.** The design is chosen first because the team has a template, and the estimand is
adjusted to fit it. The inversion is hard to see from inside because the resulting document looks
coherent.

### 5.6 Step 5 — Specify the metric as the outcome, not near it

The metric is the operational definition of the estimand's outcome slot. It must share the
estimand's construct (margin, not revenue), its grain (lane-week, not order), and its denominator
(order lines, not orders). "OTIF of 91%" is meaningless without the denominator, and each choice of
denominator gives a number large enough to change a decision.

Two requirements are specific to pilots. The metric must be **computable from the systems that will
exist during the pilot**: one needing a feed that goes live in week three of a six-week pilot will
be computed differently for the first half. And it must be **stable across the pilot**: a mid-pilot
change to the warehouse management system's timestamp convention creates a definitional break no
statistical adjustment repairs.

The output is four lines — numerator, denominator, grain, window — copied verbatim into the plan
and the report. **Where it breaks:** the metric is defined by analytics, approved by nobody, and
differs from the definition operations uses, so the number cannot be compared with the baseline the
business already believes.

### 5.7 Step 6 — Derive the decision rule from costs

This is the step most often skipped and the one that separates an experiment from a piece of
research. A decision rule has a statistic, a threshold, and a mapping to actions, all fixed before
the data.

**Part A — the break-even effect.**

```text
break-even effect = running cost of the intervention per unit per period
                    ---------------------------------------------------
                    value of one unit of the outcome per unit per period
```

In the section 6 case, the intervention costs €310 per lane-week and one percentage point of OTIF is
worth €95 per lane-week, so the break-even effect is `310 / 95 = 3.26` percentage points. Below 3.26
points the programme destroys value even when it works.

**Part B — the loss ratio and the required probability.** Break-even says what success means. It
does not say how confident you must be, because acting and not acting have different costs when you
are wrong. Let `B` be the benefit if you act and the effect is real, and `H` the harm if you act and
it is not. The standard threshold result, which comes from clinical decision analysis and is derived
from a two-by-two table with utility weights, is:

```text
act if   P(effect > break-even)  >  H / (H + B)
```

In odds form, act when the odds of a real effect exceed `H / B`. With `H = B` the threshold
probability is 0.5; with `H = 4B` it is 0.8. The formula is exact for a binary act/do-not-act
decision and is the cleanest way to make the asymmetry explicit.

**Part C — at least three branches.** A two-branch rule ("roll out" / "do not") has no place for an
inconclusive result and therefore forces one. A usable rule has act, extend, and stop.

- If the lower bound of the 95% interval exceeds 3.3 percentage points, roll out to all 240 lanes.
- If the interval includes 3.3 points and the point estimate is above zero, extend to 12 additional
  matched pairs and re-evaluate at 26 weeks.
- If the point estimate is at or below zero, stop and revert.

**Part D — why alpha is not the threshold.** A two-sided 5% test with a one-directional decision
implies a threshold probability of 0.975, so `H/B = 39`; a one-sided test implies `H/B = 19`. A
safety stock change reversible in a week has a loss ratio near 1, and 0.05 is far too strict; an
irreversible contract change with twelve months' notice has a ratio well above 39, and 0.05 is too
permissive. The point is not that 0.05 is wrong, but that it is a number someone else chose.

**Where it breaks.** The threshold is set after the result, by finding the break-even that makes the
result significant. This is the most damaging form of post-hoc reasoning in commercial analytics,
because it is invisible: the threshold appears once, as a considered judgement, and nobody sees the
three versions discarded.

### 5.8 Step 7 — Write the pre-analysis plan

The plan is what makes the chain binding. It does not need to be long. A two-page plan people use
beats a twenty-page plan nobody reads, and the evidence from applied economics is that detailed
plans matter most when heterogeneity analysis matters or when a partner has a vested interest in the
result.

A working plan has nine sections: decision and owner; claim and interrogated question; hypothesis;
six-slot estimand; design and identification assumption; primary metric with numerator, denominator,
grain, and window; secondary metrics and their roles; analysis, including estimator, clustering
level, covariates, and intercurrent events; and the decision rule with its three branches.

Two further items belong in an appendix, and they separate a plan from a wish. **The deviations
log** is a blank table with columns for date, deviation, reason, and approver: deviations are
normal, undisclosed deviations are not. **The exploratory boundary** is a sentence stating which
analyses are exploratory and how they will be labelled, which protects exploratory work rather than
suppressing it.

> **Note:** A plan written after the data are collected but before they are analysed is worth much
> less than one written before collection, because the analyst has already seen the distribution and
> possibly the result. The timestamp is the substance, not the ceremony.

### 5.9 Step 8 — Run, then separate the two tracks

Report the confirmatory result in full: every pre-specified outcome, significant or not, with the
decision rule applied mechanically, because a rule that can be overridden by judgement was never a
rule. Report deviations with their reason and their likely effect on the estimate. Report
exploratory findings separately and labelled, in the registered-report convention of a dedicated
"Exploratory analyses" section: an exploratory finding has an unknown error rate, and presenting it
beside a confirmatory finding transfers credibility it has not earned.

**Where it breaks.** The confirmatory result is weak and the exploratory section contains the story,
so the report leads with the exploratory finding and buries the primary outcome.

### 5.10 The chain audit

Before committing budget, run the audit. A "no" on any row is a stop.

| # | Check | Pass condition | Fail signal |
|---:|---|---|---|
| 1 | Decision named | A role acts differently depending on the answer | "It will inform the roadmap" |
| 2 | Claim interrogated | Comparator, population, verb, and window written | "Everyone knows what we mean" |
| 3 | Hypothesis has a magnitude | A number in outcome units, traceable to a cost | "An improvement" |
| 4 | Both units stated | Assignment and analysis grains named and reconciled | Treatment at DC level, inference at order level |
| 5 | Design fits the estimand | Selected after the estimand, not before | "We always run A/B tests" |
| 6 | MDE below minimum worthwhile effect | The arithmetic is shown | "We will see what the data say" |
| 7 | Threshold derived from costs | Break-even effect and loss ratio shown | "We will use p < 0.05" |
| 8 | Plan timestamped | Written before outcome data were examined | A plan with no date |

## 6. Worked example

### 6.1 The claim

A European consumer goods manufacturer runs a central replenishment team that orders from tier-1
suppliers once a week, on a fixed Tuesday cycle. The logistics director says, in a steering meeting:
**"Twice-weekly replenishment would fix our OTIF problem."** The analytics team is asked to "prove
it". All figures are illustrative and internally consistent; none is drawn from a named company.

### 6.2 Step 0 — the decision

The action is to extend twice-weekly replenishment from 24 pilot lanes to all 240 lanes in the
category, or to stop the programme. The owner is the logistics director, who holds the freight
budget and the supplier relationships. The deadline is 15 March, ahead of the annual carrier tender.
That fixes the population, the intervention, and the horizon.

### 6.3 Step 1 — interrogating the claim

| Interrogation | As stated | What the team writes down |
|---|---|---|
| Comparator | "Fix" implies a comparison with the current cycle | Weekly ordering, same lanes, same suppliers, same 26 weeks |
| Population | "Our OTIF problem" implies all lanes | 24 lanes, tier-1 suppliers, annual spend above €2m, no planned supplier change in the window |
| Verb | "Would fix" is a causal claim about a future state | Causal: the effect of the ordering frequency, not a forecast of OTIF |
| Window | Unstated | 26 weeks from the switch, because the decision covers a contract year and the effect is expected to decay after the honeymoon |

The interrogated question: **"Does twice-weekly ordering, compared with weekly ordering, raise
line-level OTIF on these 24 lanes over 26 weeks?"** The interrogation did not become more precise
about the mechanism, and it did not ask why OTIF is low. It converted a belief into a question and
left the mechanism for later, which is correct: the pilot tests whether the lever works, not why.

### 6.4 Step 2 — the hypothesis

The magnitude comes from cost, not from statistics. Finance confirms a running cost of €310 per
lane-week: the additional freight run (€240), planner time (€45), and supplier administration (€25).
The cost-to-serve model values one percentage point of line-level OTIF on a lane at €95 per
lane-week: avoided expedite freight (€52), avoided penalties (€28), and reduced safety stock (€15).
The team notes that the safety-stock component is a stock-flow conversion sensitive to the
carrying-cost rate, and states it as an assumption rather than a measurement.

```text
break-even effect = 310 / 95 = 3.26 percentage points
```

The hypothesis: "Twice-weekly ordering raises line-level OTIF by at least 3.3 percentage points on
the pilot lanes over 26 weeks, against a null of no change." The team keeps 3.26 for the arithmetic
and rounds the threshold up to 3.3, because rounding it down would make the programme easier to
declare a success, and the direction of rounding is itself a decision that should be stated.

### 6.5 Step 3 — the estimand

| Slot | Value |
|---|---|
| Population | 24 lanes, tier-1 suppliers, annual spend above €2m, no supplier change or lane redesign in the window, weeks 1–26 of 2026 |
| Unit | Lane-week for measurement; **lane** for assignment |
| Outcome | Line-level OTIF, defined as the proportion of order lines delivered on or before the confirmed date and in full quantity, rolled to lane-week |
| Time | Time zero is the week of the switch; 26-week follow-up |
| Intervention | Twice-weekly ordering on a 3-day frozen window, versus weekly ordering on a 7-day frozen window, same suppliers and lanes |
| Comparison | Difference in mean lane-week OTIF, in percentage points, 95% interval clustered at lane level |

Two choices inside the table change the number and are surfaced deliberately. **Line-level rather
than order-level OTIF**: order-level treats a 40-line order with one late line as a failure, so it is
dominated by large orders, while the decision concerns lane performance and line level is the grain
at which late deliveries are experienced. **Confirmed date rather than requested date**: the
confirmed date is what the supplier committed to, whereas measuring against the requested date
measures the manufacturer's own planners as much as the supplier.

### 6.6 Step 4 — the design

The team cannot randomise at order-line level, because ordering frequency is a lane-level policy and
a lane cannot run two frequencies at once without contaminating both. The design is a **matched-pair
stepped rollout**. The 24 lanes are paired on prior-year OTIF, spend, and supplier country, giving 12
pairs. Within each pair, one lane switches at week 1 and the other at week 14. The estimate is the
within-pair, within-period contrast.

| Design element | Value | Why |
|---|---|---|
| Assignment | Lane, within matched pair | The intervention is a lane policy |
| Assignment units | 24 lanes in 12 pairs | This, not the order count, sets precision |
| Identification assumption | No differential time trend within a pair beyond the switch | Tested on the 8 pre-period weeks |
| Contamination control | Separate planners for switch and hold lanes; no cross-lane substitution | Planners would otherwise apply the better rule everywhere |
| Interference risk | Shared carrier capacity on two pairs | Flagged; a spillover check on the paired lane is pre-specified |

The design has a known weakness and the team writes it down: the second lane in each pair switches
at week 14, so the estimate is partly identified from a period when the first lane has run for 13
weeks. If the effect decays, the two halves disagree, and the pre-specified analysis reports them
separately.

### 6.7 Step 5 — the metric

The numerator is order lines delivered on or before the confirmed date, in full quantity. The
denominator is all order lines scheduled for delivery in the lane-week. The grain is the lane-week
and the window is 26 weeks from switch. Exclusions are limited to lines cancelled by the
manufacturer and lines on lanes with a supplier-initiated allocation. The stability check is that
the timestamp convention is unchanged for the pilot duration, verified against the warehouse
management system release log. The exclusion list is short on purpose: every exclusion is a place
where the metric can drift.

### 6.8 Step 6 — the decision rule

**Break-even:** 3.26 percentage points, from `310 / 95`.

**Loss ratio.** The team asks finance and operations two questions. What does a wrong rollout cost?
The programme is reversible, but reversing it means renegotiating 240 supplier collection schedules,
reconfiguring the transport management system, and a one-off cost of about €600,000, plus the annual
running cost if the effect is below break-even. What does a wrong no-rollout cost? The forgone
margin, at €95 per point per lane-week, with no offsetting cost.

At an assumed effect of 4.1 points, on 240 lanes over 52 weeks:

```text
value per point = 240 x 52 x 95 = EUR 1,185,600/yr
B = (4.1 - 3.26) x 1,185,600 = EUR 995,904
H = 3.26 x 1,185,600 + 600,000 = EUR 4,465,000
H / (H + B) = 4,465,000 / 5,460,904 = 0.818
```

The harm figure is pessimistic because it assumes the effect is zero rather than merely below
break-even, and the team states that choice. The implied rule requires roughly an 82% probability
that the true effect exceeds break-even before rolling out.

**The three branches, written before the data:**

- Roll out to 240 lanes if the lower 90% bound on the effect exceeds 3.26 points.
- Extend by 12 further matched pairs if the point estimate is above break-even but the bound is not.
- Stop if the point estimate is at or below zero.

> **Note:** Using a lower confidence bound as a stand-in for a posterior probability is an
> approximation, and it is not exact. It is used because most commercial teams have a frequentist
> interval and no prior. The approximation is stated in the plan so a reader can disagree with it.

### 6.9 The result

The pilot runs 26 weeks. The primary estimate is a +4.1 percentage point difference in lane-week
OTIF, 95% interval 2.1 to 6.1 points, from a paired analysis clustered at lane level, with a
standard error of 1.02 points.

| Question | Statistic | Answer |
|---|---|---|
| Is the effect different from zero? | `t = 4.1 / 1.02 = 4.02`, `p < 0.001` | Yes |
| Is the effect above break-even? | `t = (4.1 - 3.26) / 1.02 = 0.82`, one-sided `p = 0.21` | Not established |
| Does the decision rule say roll out? | 21% probability mass above 3.26 against an 82% requirement | No |
| Expected annual value at the point estimate | `0.84 x 1,185,600` | €996,000 |
| Expected annual value at the lower bound | `(2.1 - 3.26) x 1,185,600` | −€1,375,000 |

The last two rows are why the rule is demanding. The point estimate is worth about €1.0m a year. The
lower bound is worth about −€1.4m, because at 2.1 points the programme runs at a loss on every lane.
A decision made on the point estimate bets a million on a distribution whose lower tail costs more
than the upper tail gains.

### 6.10 What the team reports, and the conventional version

The team reports the decision (the rule applied mechanically: extend, do not roll out); the primary
result; two deviations, both retained and flagged, with the pre-specified sensitivity analysis
moving the estimate to +3.9 points; and two labelled exploratory findings: the effect is larger on
single-country than on cross-border lanes, and larger in weeks 1–8 than in weeks 19–26. The
recommendation is to extend to 12 additional pairs, with a revised hypothesis that the effect
decays after week 12. That decay finding is the most commercially interesting result in the study
and is not allowed to change the decision.

Run the conventional analysis and the team reports `p < 0.001`, a 4.1 point improvement, and a
recommendation to roll out because the result was significant. That recommendation is wrong by the
decision's own cost arithmetic, and wrong in a way no reviewer can see.

## 7. Code, query, or calculation

This section is arithmetic, not code, because the calculations that decide the outcome are a few
lines long and should be done on paper in the meeting.

```text
INPUTS
  c        = 310        running cost of intervention, EUR per lane-week
  v        = 95         value of one percentage point of OTIF, EUR per lane-week
  n_lanes  = 240        rollout population
  weeks    = 52         decision horizon
  C_rev    = 600,000    one-off reversal cost, EUR
  se       = 1.02       standard error of the effect, percentage points
  effect   = 4.10       observed effect, percentage points
STEP 1  tau        = c / v              = 310 / 95            = 3.26 pp
STEP 2  lane_weeks = n_lanes * weeks    = 240 * 52            = 12,480
        value_pp   = lane_weeks * v     = 12,480 * 95         = 1,185,600 EUR/yr
STEP 3  B          = (effect - tau) * value_pp = 0.84 * 1,185,600    = 995,904
        H          = tau * value_pp + C_rev    = 3,865,000 + 600,000 = 4,465,000
STEP 4  p_star     = H / (H + B)        = 4,465,000 / 5,460,904 = 0.818
STEP 5  z_tau      = (effect - tau) / se = 0.84 / 1.02          = 0.82
        P_above    = 1 - Phi(0.82)                              = 0.21
STEP 6  P_above (0.21) < p_star (0.818)  ->  extend, do not roll out
```

Expected output: `tau = 3.26`, `p_star = 0.818`, `P_above = 0.21`, decision `extend`.

Two sanity checks before you trust a calculation of this shape. **Check the units:** `tau` is in
percentage points, `value_pp` is currency per point per year, and `B` is currency per year; if `B`
comes out in percentage points, a missing multiplication by `value_pp` is the usual cause. **Check
the direction of the rounding:** rounding `tau` from 3.26 down to 3.2 lowers the required
probability and makes the programme easier to approve, so rounding decisions in a threshold
calculation belong in the plan, before the result is known.

## 8. Data, standards, and regulatory context

No standard governs the specification of a supply chain pilot. Three bodies of practice constrain
parts of it, and knowing which part each constrains prevents two opposite errors: claiming a
compliance obligation that does not exist, and inventing a definition where a good one is available.

### 8.1 The clinical-trials framework and its scope

| Instrument | Status and date | What it establishes | Applies to supply chain? | Grade |
|---|---|---|---|---|
| ICH E9(R1) addendum | Step 4, 20 November 2019; EMA Step 5 effective 30 July 2020 | Estimand defined by five attributes; five strategies for intercurrent events; alignment of primary analysis, sensitivity analysis, and reporting | No; analogy only | B |
| ICH E9 (original) | 1998 | Statistical principles for clinical trials, including intention-to-treat | No; historical context | B |
| U.S. FDA E9(R1) guidance | May 2021 | U.S. adoption of the addendum for medicinal product applications | No | B |
| CONSORT 2010 and extensions | 2010 onward | Reporting of randomised trials, including assigned versus analysed populations | No; a checklist of what a causal report must disclose | B |

The addendum's five attributes are treatment, population, variable, population-level summary, and
the strategy for handling intercurrent events. The six-slot form splits time and unit out of the
clinical population and variable attributes, because supply chain analyses get those two wrong more
often than they get the summary measure wrong.

> **Note:** The addendum governs medicinal product trials. Nothing in it binds a retailer, a
> manufacturer, or a logistics operator, and a process should not be described as "ICH-compliant"
> unless it is a clinical trial. Its value here is that it is the most carefully worked-out public
> statement of this problem, written by people for whom ambiguity was expensive and visible.

### 8.2 Registration and pre-specification practice

| Instrument | Owner | What it requires | Transfer to a commercial pilot | Grade |
|---|---|---|---|---|
| AEA RCT Registry | American Economic Association | Public registration with primary outcomes | A timestamped internal registry entry is the minimum viable version | B |
| OSF Registries | Center for Open Science | Registrations and pre-analysis plans across disciplines | A hosted plan with a DOI and a date | B |
| AsPredicted | Wharton Credibility Lab | Short structured preregistration | A one-page template usable for a pilot | B |
| Registered Reports | Multiple journals | Review of the plan before data collection; exploratory analyses in a separate section | The two-section reporting convention transfers directly | B |

The transferable content is not the registry. It is three conventions: the timestamp, the separation
of confirmatory from exploratory reporting, and the disclosure of deviations. All three can be
implemented in a shared drive with version history, and none requires an external body.

### 8.3 Governance that touches the process without defining it

Model risk management guidance (for example U.S. Federal Reserve SR 11-7, 2011, grade B) requires
documentation, validation, and review of models used in decisions; internal control regimes over
financial reporting require evidence trails for figures reaching the statements; pharmacovigilance
practice requires adverse event capture during a study; industry measurement guidelines (grade C)
supply evidence hierarchies. None defines an estimand. They make it more likely that someone will
ask what a model estimates, and an analyst with a six-slot estimand and a written decision rule can
satisfy any of them in an afternoon.

### 8.4 What the data must contain

| Artefact | Corresponds to | What it must contain | Common absence |
|---|---|---|---|
| Pre-period history | Population | At least 8 weeks at the assignment grain, before time zero | Only the pilot period is extracted |
| Assignment log | Design | Which unit was assigned to which condition, when, by whom | Assignment reconstructed from treatment data |
| Exposure log | Intervention | When the treatment was actually live, per unit | Planned dates used as exposure dates |
| Outcome extract | Outcome | The metric at the analysis grain, with the denominator | Metric computed at a different grain |
| Cost parameters | Decision rule | Running cost and value per outcome unit, with an owner and a date | Costs estimated by the analyst from public benchmarks |
| Deviation log | All slots | Every departure from the plan, dated | Deviations discovered during write-up |

The cost parameters are the most frequently missing and the most consequential. A pilot with no cost
model can produce a number but not a decision, and the number will be pressed into service as a
decision by whoever is most senior in the room.

### 8.5 Requirement versus convention

Nothing in 8.1 to 8.4 binds a commercial pilot. p < 0.05, 80% power, two-sided tests, and reporting
every secondary outcome are conventions, and an analyst who can tell a convention from a requirement
can argue with it. One who cannot will defend 0.05 as though it were a physical constant.

## 9. Industry practice and case evidence

### 9.1 How often business experiments produce false positives

| Finding | Source | Grade |
|---|---|---|
| Across 2,766 experiments and 4,964 effects on a commercial testing platform, the false discovery rate was 18–25% at a 5% threshold and 28–37% at 10% | Berman and Van den Bulte (2021), *Management Science* | A |
| The high rate is driven mainly by a high proportion of true nulls (about 70%), not by low power (average about 65–70%) | Same | A |
| Even at 100% power, the false discovery rate would remain in the range of 11–20% in that sample | Same | A |

The third row is the one to carry into a meeting. Most teams believe false positives are a power
problem and a bigger sample fixes them. In this sample the dominant cause was that most tested
changes do nothing. A larger sample makes each null result more precisely null, and does not reduce
the proportion of rejections that are false by as much as people expect.

> **Note:** The sample is one platform, dominated by digital product tests with many small effects.
> Supply chain pilots differ: more expensive per unit, fewer units, and changes with a plausible
> mechanism. The direction transfers; the magnitudes should not be quoted as if they described a
> warehouse pilot.

### 9.2 What pre-analysis plans actually buy

| Finding | Source | Grade |
|---|---|---|
| With many outcomes and no true effect, roughly 5% of outcomes appear significant at 5%; pre-specification is the response | Olken (2015), *Journal of Economic Perspectives* | A |
| Full pre-specification of a complex analysis tree is infeasible: one worked example yields 59,049 possible regressions | Same | A |
| Detailed plans matter most when heterogeneity analysis matters or a partner has a vested interest | Banerjee, Duflo, Finkelstein, Katz, Olken and Sautmann (2020), NBER working paper | B |
| Deviations should be disclosed and are not, by themselves, evidence of misconduct | Same | B |

The 59,049 figure is the argument against exhaustive pre-specification, and it is worth
internalising. The plan is not a promise to run one regression. It is a promise to say which
regression is primary and to label the rest.

### 9.3 Decision-theoretic thresholds in practice

| Finding | Source | Grade |
|---|---|---|
| Hypothesis testing imposes a 0/1 loss in which every wrong decision costs the same; a profit-maximising test uses actual opportunity costs | Feit and Berman (2019), *Marketing Science* | A |
| Under that framing a conventional 0.05 threshold over-tests, and profit-maximising sample sizes are far smaller than null-hypothesis-testing recommendations | Same | A |
| Regret from a hypothesis-testing rule grows linearly in population size, against square-root growth for a decision-theoretic rule | Same | A |
| A decision-theoretic rule for feature adoption selects the p-value cutoff from a loss function rather than fixing it at 0.05 | Goldberg and Johndrow, arXiv preprint | B |
| Regret weights each mistake by its economic size, so a false positive deploying a harmful treatment costs the magnitude of the harm and a false negative costs the benefit forgone | Farrell, Korganbekova and Misra, arXiv preprint | B |

The convergence is the substantive point: the fixed threshold is a special case of the cost-weighted
rule, corresponding to one particular loss ratio, and it is rarely the right ratio for a business
decision.

### 9.4 What happens when pilots are run without this discipline

| Finding | Source | Grade |
|---|---|---|
| In a documented retail inventory pilot, forecast records were not saved during the pilot and the sample shrank mid-study; positive results nonetheless led to broad rollout | Ibanez and Staats (2019), *INFORMS TutORials in Operations Research* | B |
| Operations field experiments are characterised by high cost, execution risk, small samples, Hawthorne effects, and irreproducibility; repeat opportunities are rare | Same | B |
| A systematic review of supply chain experiments found under-reporting of power analyses and manipulation checks, and over-reliance on convenience samples | Carter et al. (2024), *Journal of Business Logistics* | A |
| In a large physical-retail experimentation programme, conventional inference found significance in roughly 6% of experiments; a synthetic-control approach raised the number of significant findings about tenfold | Costa, Farias et al. (2023), *INFORMS Journal on Applied Analytics* | A |
| Reviews of operations field experiments identify pre-analysis plans, attrition, covariate balance, and heterogeneous effects as commonly overlooked | Gao, Li and Sun (2023), *Journal of Operations Management* | A |

The first row is the most instructive and it is not a story about fraud. It is a pilot that produced
an unusable measurement and a decision that was made anyway. The forecasts were not saved because
nobody had written down that they would be needed. That is a pre-analysis plan failure, not a data
failure. The fourth row calibrates expectations: a 6% significance rate is not evidence that retail
experiments are hopeless, but evidence that effects are small relative to noise, and that a design
ignoring the correlation structure finds almost nothing.

## 10. Failure modes and red flags

| Failure | Red flag you can see in a meeting or a dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Claim never interrogated | The word "worked" or "world class" survives into the project charter | Comparator, population, and window all implicit | The analysis answers a question nobody asked | Run the four interrogations in writing, in the meeting |
| Hypothesis with no magnitude | "We expect an improvement" in the plan | Any positive number confirms it | The pilot cannot fail, so it teaches nothing | Require a number in outcome units, traceable to a cost |
| Estimand missing the unit slot | The plan says "per order" but the treatment is assigned per lane | Assignment and analysis grains differ | Standard errors too small; false positives | State both units in every document |
| Design chosen before the estimand | The plan opens with "we will run an A/B test" | Template-driven design | The design cannot estimate the quantity that matters | Write the estimand first, then eliminate designs |
| MDE above the minimum worthwhile effect | The power calculation is absent, or uses a default effect | Sample sized for convenience | The pilot cannot separate a good effect from a trivial one | Compute MDE against the break-even effect |
| Metric substituted for the outcome | Revenue in the report, margin in the plan | The adjacent metric was easier to extract | A profitable-sounding result that loses money | Metric dictionary fixed in the plan, quoted verbatim in the report |
| Threshold set after the result | The break-even analysis is dated after the data extract | Post-hoc threshold search | Any result can be made decisive | Date the cost model and the threshold in the plan |
| Significance used as the decision rule | "The result was significant, so we recommend rollout" | Alpha mistaken for a cost threshold | Rollouts that are significant and unprofitable | Publish the loss ratio and the required probability |
| Two-branch decision rule | "If significant, roll out" | No place for an inconclusive result | Inconclusive pilots are forced into a decision | Require act, extend, stop |
| No exposure log | Treatment dates taken from the plan, not the system | Planned and actual exposure conflated | Effect diluted by units never treated | Log exposure per unit per week |
| Exploratory findings presented as results | A subgroup result appears in the executive summary | No labelling convention | Unknown error rate advertised as a finding | Separate section, labelled, in every report |
| Deviations undisclosed | The analysis sample is smaller than the assignment log | Silent exclusions | Unverifiable estimate | Deviations log, reported with the primary result |

Three deserve a note. **Significance used as the decision rule** is the most consequential because
it is the most respectable: the analyst has run a correct test and drawn an inference it does not
support. Section 6.9 gives the arithmetic of a case where `p < 0.001` and the correct decision is
not to roll out. **Two-branch decision rules** are how inconclusive pilots become rollouts, because
"do not roll out" sounds decisive and is a decision to stop learning. **Metric substitution** is
invisible because both numbers are in the pack: the reader sees a rising line labelled with the
right words and cannot tell the underlying series is a different construct.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| The decision, owner, and deadline can be named before the pilot | The estimand and threshold are derived from them | The pilot answers a question nobody owns and the result is filed | Ask who will act, and by when; if nobody, stop |
| The claim can be interrogated into a single question | The chain needs one question to carry | Multiple questions collapse into one test, and the primary becomes whichever is significant | Split into separate questions, each with its own plan |
| A minimum worthwhile effect can be computed | The hypothesis and threshold both depend on it | The magnitude is set by what the design can detect, not what matters | Check that the cost model has an owner and a date |
| The outcome is measurable at the estimand's grain | The metric must equal the outcome slot | The estimate is for a different quantity than the one stated | Build the metric before the pilot and reconcile it to the operational report |
| The assignment grain is feasible | Precision depends on the number of assignment units | Too few clusters, and the MDE is far above the target | Count assignment units, not rows |
| The identification assumption is credible | Every non-randomised design needs one | The estimate is confounded and the interval is meaningless | Pre-trend test, negative control, spillover test |
| Costs are stable over the decision horizon | The threshold is computed from current costs | The break-even effect moves and the decision reverses | Sensitivity of the threshold to cost inputs, reported |
| Interference is absent or bounded | Standard errors and the estimand assume no spillover | The effect is overstated, sometimes doubled | Buffer zones; spillover test at distance; paired-lane check |
| The exploratory boundary is respected | Confirmatory claims need controlled error rates | Exploratory findings carry unearned credibility | Check that the primary outcome appears first in every report |

Two limits are worth stating plainly.

**The chain disciplines the question; it does not supply the answer.** A perfectly specified
estimand with a design that cannot identify it yields a precise statement of an unanswerable
question. That is an improvement, because the failure becomes visible while it is still cheap, but
it is not a result.

**The cost-weighted threshold is only as good as the cost model.** The `H/(H+B)` rule converts
uncertainty into a decision using numbers that are usually estimates themselves. When the cost model
is uncertain, compute the threshold at both ends of the plausible range and check whether the
decision changes. If it does, the cost model is the binding constraint and the pilot should wait.

> **Note:** There is a real tension between pre-specification and learning. A plan rigid enough to
> be credible can be too rigid to accommodate what the pilot reveals. The resolution used in
> practice is a thin primary plan plus an explicit exploratory track, not a thick plan that tries to
> anticipate everything.

## 12. Exercises

### Exercise 1 — Interrogate two claims (analysis, 35 minutes)

Run the four interrogations on: (1) "Our new carrier improved on-time delivery." (2) "Inventory
accuracy is now best in class."

**Deliverable:** for each claim, a table with comparator, population, verb classification, and
window, plus the interrogated question in one sentence.
**Acceptance criteria:** at least three distinct candidate comparators named for claim 1; the verb
in claim 2 classified as a ranking claim rather than a causal one; both questions contain a
comparator and a period; one sentence per claim stating what is still unknown.

### Exercise 2 — Write a falsifiable hypothesis (application, 25 minutes)

A warehouse manager claims a new put-away rule will "reduce mispicks". Labour cost of the change is
€0.04 per unit put away. The cost of a mispick, including rework and customer credit, is €28.

**Deliverable:** the break-even mispick reduction per 1,000 units, the hypothesis with direction,
magnitude, and null, and the observation that would refute it.
**Acceptance criteria:** the break-even arithmetic is shown with units; the hypothesis states a
number on a stated scale; the refuting observation is a specific data pattern.

### Exercise 3 — Fill the six slots (design, 40 minutes)

For the put-away question, write the estimand. Assume 6 distribution centres, 14 weeks, and a rule
that can be applied to a subset of aisles.

**Deliverable:** the six-slot table, with the unit of assignment distinguished from the unit of
analysis.
**Acceptance criteria:** population countable in the data; time zero tied to a decision point;
comparator described as precisely as the intervention; comparison stated with a scale.

### Exercise 4 — Eliminate designs (critique, 30 minutes)

Using the estimand from Exercise 3, eliminate at least three candidate designs, giving the reason in
terms of the estimand rather than the sample size.

**Deliverable:** a table of candidate designs, the elimination criterion each fails, and the one
design that survives with its identification assumption stated.
**Acceptance criteria:** each elimination names a slot the design cannot serve; the surviving design
states its number of assignment units; the identification assumption is testable.

### Exercise 5 — Derive a decision rule (calculation, 45 minutes)

For the put-away pilot: the intervention costs €0.04 per unit, the mispick rate is 1.8 per 1,000
units, and each of the six distribution centres handles 90,000 units per week. Assume a reversal
cost of €120,000 and a one-year decision horizon across all six sites.

**Deliverable:** break-even effect, annual value per unit of effect, benefit and harm at a stated
effect, required probability, and a three-branch rule.
**Acceptance criteria:** every intermediate number shown with units; the required probability
derived rather than asserted; all three branches specified.

### Exercise 6 — Find the broken link (critique, 35 minutes)

A pilot report states: "The new slotting policy improved picks per hour by 6.2% (p = 0.03). We
recommend rollout." The appendix shows the metric was total warehouse picks divided by total labour
hours, the policy was applied to two of five zones, and the comparison is with the same warehouse
last year.

**Deliverable:** a diagnosis naming which links broke, in chain order, and a corrected version of
each broken link.
**Acceptance criteria:** at least three links identified; each diagnosis quotes the specific text or
appendix item; each correction is written, not described.

### Exercise 7 — Write a two-page pre-analysis plan (design, 60 minutes)

Write a pre-analysis plan for the put-away pilot using the nine-section structure in section 5.8.

**Deliverable:** a plan of no more than two pages, plus a blank deviations log and a one-sentence
exploratory boundary.
**Acceptance criteria:** all nine sections present; the primary metric has numerator, denominator,
grain, and window; the decision rule has three branches; the plan is dated.

### Exercise 8 — Separate the tracks and stress the cost model (analysis, 55 minutes)

Take a report with one pre-specified primary outcome (non-significant), four secondary outcomes (one
significant), and a subgroup finding (significant). Rewrite the executive summary, then recompute
the Exercise 5 break-even effect and required probability at plus and minus 25% on the cost of a
mispick.

**Deliverable:** a revised summary of no more than 150 words, a note on what changed and why, and a
three-row table (low, base, high) with a one-paragraph conclusion.
**Acceptance criteria:** the primary outcome appears first and is described as not established; the
subgroup finding is labelled exploratory and hypothesis-generating; no sentence transfers
credibility between tracks; the conclusion states whether the decision is robust and which input
binds.
## 13. Self-check answers

### 13.1 Exercise 1 model answer

For claim 1 the candidate comparators are: the previous carrier on the same lanes; the same carrier
on comparable lanes elsewhere; the contractual target; the carrier's own bid commitment. Four
different questions. The verb "improved" is causal and implies a comparison with what the previous
carrier would have done in the same period, which is not observable. The population is unstated (all
lanes, or the lanes that moved) and the window is unstated.

For claim 2 the verb is a ranking claim, not a causal one. It needs a peer set, a metric, and a
period: best in class against which warehouses, on which accuracy definition (location count, unit,
or value), measured how (cycle count, wall-to-wall, or perpetual adjustment)? The interrogated
question is descriptive: "What is our location-count inventory accuracy, compared with the peer set
defined as [named set], over [period]?"

### 13.2 Exercise 2 arithmetic

```text
labour cost of the change   = 1,000 x 0.04  = EUR 40.00 per 1,000 units
cost per mispick            = EUR 28
break-even mispicks avoided = 40.00 / 28    = 1.43 per 1,000 units
```

Against a base rate of 1.8 per 1,000, that is a reduction of `1.43 / 1.8 = 79%`, which is
implausibly large. The immediate finding is that the economics do not work at these numbers: the
intervention costs more per unit than the errors it prevents, unless the base rate is far higher or
the mispick cost far larger. That is a legitimate output.

A usable hypothesis at a plausible base rate of 6 mispicks per 1,000: "The put-away rule reduces
mispicks by at least 1.5 per 1,000 units, against a null of no change." The refuting observation is
a confidence interval whose lower bound is below 1.43 per 1,000.

### 13.3 Exercise 3 model answer

| Slot | Value |
|---|---|
| Population | 6 distribution centres, 240 aisles, 14-week window, no racking reconfiguration |
| Unit | Aisle-week for measurement; **aisle** for assignment |
| Outcome | Mispicks per 1,000 units put away, from the adjustment log and the put-away transaction count |
| Time | Time zero is the week the rule goes live in that aisle; 14-week follow-up |
| Intervention | New put-away rule (nearest available location with a family constraint), versus the existing fixed-location rule |
| Comparison | Difference in mean mispicks per 1,000 units, 95% interval clustered at aisle level |

The two units differ, and that is the point. The rule is assigned per aisle and the outcome measured
per aisle-week. A naive analysis at aisle-week level would treat 3,360 observations as independent
when there are 240 assignment units.

### 13.4 Exercise 4 model answer

| Candidate design | Eliminated because | In estimand terms |
|---|---|---|
| Individual randomisation of put-away tasks | The intervention is a rule, not a task treatment | The intervention slot describes an aisle-level policy; task-level assignment would be a different intervention |
| Before-and-after at all 6 DCs | No comparison condition | The comparison slot requires two conditions; there is only one |
| Randomise 3 DCs versus 3 DCs | Only 3 assignment units per arm | Precision is set by assignment units, and 3 per arm cannot detect the target effect |
| Paired-aisle randomisation | Survives | Identification: aisles within a pair share SKU profile and labour pool |

The surviving design's assumption is that paired aisles would have followed the same mispick trend
absent the rule, testable on the pre-period.

### 13.5 Exercise 5 arithmetic

```text
units per week, all sites     = 6 x 90,000                = 540,000
base mispicks per week        = 540,000 x 1.8 / 1,000     = 972
cost of mispicks per week     = 972 x 28                  = EUR 27,216
labour cost of change / week  = 540,000 x 0.04            = EUR 21,600
break-even reduction / week   = 21,600 / 28               = 771.4 mispicks
break-even effect per 1,000   = 771.4 / 540               = 1.43 per 1,000 units
break-even as share of base   = 1.43 / 1.8                = 79.4%
value per point per year      = 52 x 540,000 x 28 / 1,000 = EUR 786,240
```

A 79% reduction in mispicks is not achievable by a put-away rule, so the correct decision is to stop
before piloting: at 0.5 per 1,000, `B = (0.5 - 1.43) x 786,240` is negative. When the observed
effect is below break-even the threshold formula does not apply in its simple form, because it
assumes a world in which acting can be right. If break-even is unreachable at any plausible effect,
the answer is no and no probability is needed.

### 13.6 Exercise 6 model answer

Three links broke.

**Link 3, unit slot.** The policy applied to two of five zones, so the population is those zones.
The metric uses total warehouse picks over total labour hours, mixing treated and untreated zones.
Population and unit were both violated.

**Link 4, comparison.** The comparison is the same warehouse last year: a before-and-after with no
control for demand mix, labour market, or SKU profile. The design cannot estimate a causal contrast.

**Link 5, metric.** Picks per labour hour is an aggregate ratio. It rises when low-productivity
zones are under-staffed, which is a scheduling effect, not a slotting effect.

A corrected version: "Among the 2 zones assigned the new rule, the change in picks per labour hour
over 8 weeks, compared with the 3 zones on the existing rule, was X (95% CI Y to Z), clustered at
zone level." Zone-level clustering gives 5 units, very likely too few for the target effect, and the
honest correction says so.

### 13.7 Exercise 7 rubric

A strong plan is two pages, names the decision and its owner on line one, and carries the break-even
effect from the cost model into both the hypothesis and the decision rule with the same number. It
states the metric's numerator, denominator, grain, and window, and the analysis's clustering level.
It has three branches and a blank deviations log. A weak plan is long, contains no cost model, and
has a decision rule reading "roll out if the results are positive".

### 13.8 Exercise 8 model answer

"The pre-specified primary outcome, mispicks per 1,000 units, did not differ between the new and
existing put-away rules: the estimate was 0.2 per 1,000 (95% CI −0.6 to 1.0), against a
pre-specified break-even of 1.43. The decision rule is therefore to stop. One secondary outcome,
put-away time per unit, improved by 4%, and a post-hoc comparison suggested a larger effect in
aisles with high SKU turnover; that comparison was not pre-specified and is reported as exploratory
and hypothesis-generating."

The rewrite changes three things: the primary appears first and is not established; the secondary is
not promoted; the subgroup is labelled and given a next step.

```text
mispick cost 21.00  ->  break-even 40.00/21 = 1.90 per 1,000  (106% of base rate)
mispick cost 28.00  ->  break-even 40.00/28 = 1.43 per 1,000  ( 79% of base rate)
mispick cost 35.00  ->  break-even 40.00/35 = 1.14 per 1,000  ( 64% of base rate)
```

The decision does not change: at every point in the range the break-even reduction exceeds 60% of
the base rate, which no put-away rule achieves. The result is robust to the cost model, and the
binding input is the labour cost of the change, not the cost of a mispick. That is the finding to
report, because it tells operations where to focus if they want the economics to work.

## 14. Interview questions

**1. "A stakeholder says the promotion worked. What do you ask first?"**
Strong: what decision depends on the answer, then the comparator, then the window and the
population; notes that "worked" is ambiguous between "sold more", "sold more than otherwise", and
"made money", and the three need different evidence. Weak: asks for the sales data, which accepts
the claim's framing.

**2. "What makes a hypothesis falsifiable in practice?"**
Strong: a direction, a magnitude in outcome units, a stated null, and the data pattern that would
count against it. Weak: "it must be testable", which repeats the definition rather than applying it.

**3. "What is an estimand, and why write it before the design?"**
Strong: the precise quantity to be estimated, in six slots; writing it first makes design selection
a feasibility test rather than a preference, and the unit slot alone can eliminate two designs.
Weak: "the thing you want to measure", which has no purchase on a decision.

**4. "Why is p < 0.05 not a decision rule?"**
Strong: alpha is a probability under a null of usually zero, while a decision needs a threshold at
break-even and a loss ratio; 0.05 implies a particular and rarely appropriate cost asymmetry. Weak:
"because statistical significance is not practical significance", which does not say what to do.

**5. "How would you set the threshold for a rollout decision?"**
Strong: break-even from running cost and value per outcome unit, required probability from the
harm-to-benefit ratio, then three branches, with the reversal cost inside the harm. Weak: picks a
threshold by judgement or by "what we usually use".

**6. "What is the difference between confirmatory and exploratory analysis?"**
Strong: whether the analysis was planned before the data were seen, which determines whether the
error rate is controlled; the error is claiming one's status for the other, and exploratory findings
feed the next confirmatory hypothesis on new data. Weak: says exploratory is lower quality.

**7. "What goes in a pre-analysis plan, and what does not?"**
Strong: decision, question, hypothesis, estimand, design, primary metric, secondary metrics,
analysis, decision rule, deviations log, exploratory boundary. Not a promise to run exactly one
regression, or an exhaustive specification tree. Weak: treats it as a form written after the pilot.

**8. "A pilot is significant but below break-even. What do you recommend?"**
Strong: against rollout, explaining that the test answered a question about zero rather than
break-even, and checking whether the extend branch applies. Weak: recommends rollout because the
result was significant, which is the error the question is testing for.

**9. "How do you decide the unit of assignment?"**
Strong: at or finer than the grain at which the decision is made, and a grain at which the
intervention can be applied without contaminating other units; precision comes from the number of
assignment units, not the row count. Weak: says "the customer" without reference to the decision.

**10. "Why do so many business experiments produce false positives?"**
Strong: most tested changes do nothing, so even at high power a substantial share of rejections are
false; design and pre-specification address this better than sample size alone. Weak: "because
people p-hack", which is one mechanism and not the main one in large samples.

## 15. Key terms

- **Claim** — a sentence someone believes, usually without a comparator or a window.
- **Hypothesis** — a directional statement with a magnitude that an observation could contradict.
- **Null hypothesis** — the effect value the test is run against, often but not always zero.
- **Falsifiability** — the property that some observable result would count against the statement.
- **Minimum worthwhile effect** — the smallest true effect that justifies acting; owned by the business.
- **Minimum detectable effect (MDE)** — the smallest true effect the design can distinguish; owned by the design.
- **Estimand** — the precise quantity to be estimated, in six slots, fixed before analysis.
- **Estimator** — the procedure that produces an estimate of the estimand.
- **Estimate** — the number the estimator produces on a dataset.
- **Decision rule** — the pre-specified mapping from an estimate to an action.
- **Decision threshold** — the effect size above which the action is taken, derived from costs.
- **Break-even effect** — the effect at which the intervention exactly pays for its running cost.
- **Loss function** — the cost of each error as a function of its size, in currency.
- **Loss ratio (k)** — the cost of a wrong action divided by the cost of a wrong inaction.
- **Required probability** — `H/(H+B)`, the confidence needed before acting.
- **Power** — the probability of detecting a stated effect, if it is true.
- **Pre-analysis plan (PAP)** — a timestamped document fixing the estimand, analysis, and decision rule.
- **Preregistration** — public registration of a study and its primary outcomes, thinner than a PAP.
- **Confirmatory analysis** — planned in advance, with a controlled error rate.
- **Exploratory analysis** — generated by the data, with an uncontrolled error rate, labelled as such.
- **False discovery rate (FDR)** — the expected proportion of rejections that are false.
- **Intercurrent event** — a post-assignment event that changes what the outcome means.

## 16. Sources

**Peer-reviewed and academic — Grade A**

- Olken, B. A. (2015). "Promises and Perils of Pre-Analysis Plans." *Journal of Economic
  Perspectives*, 29(3). DOI: 10.1257/jep.29.3.61. The 5%-of-outcomes argument and the 59,049
  specification count.
- Berman, R. and Van den Bulte, C. (2021). "False Discovery in A/B Testing." *Management Science*.
  False discovery rates of 18–25% at 5% and 28–37% at 10% across 2,766 experiments.
- Feit, E. M. and Berman, R. (2019). "Test & Roll: Profit-Maximizing A/B Tests." *Marketing
  Science*. Hypothesis testing as a 0/1 loss; profit-maximising sample sizes; regret growth.
- Pauker, S. G. and Kassirer, J. P. (1975). "Therapeutic Decision Making: A Cost-Benefit Analysis."
  *New England Journal of Medicine*, 293(5), 229–234. DOI: 10.1056/NEJM197507312930505.
- Pauker, S. G. and Kassirer, J. P. (1980). "The Threshold Approach to Clinical Decision Making."
  *New England Journal of Medicine*, 302(20), 1109–1117. DOI: 10.1056/NEJM198005153022003.
- Gao, H., Li, M. and Sun, S. (2023). "Field experiments in operations management." *Journal of
  Operations Management*. DOI: 10.1002/joom.1240.
- Carter, C. R. et al. (2024). "Experiments in supply chain management research: A systematic review
  and future directions." *Journal of Business Logistics*. DOI: 10.1111/jbl.12382.
- Costa, A., Farias, V. et al. (2023). "Generalized Synthetic Control for TestOps at ABI: Models,
  Algorithms, and Infrastructure." *INFORMS Journal on Applied Analytics*.
  DOI: 10.1287/inte.2023.0028.

**Working papers and preprints — Grade B**

- Banerjee, A., Duflo, E., Finkelstein, A., Katz, L. F., Olken, B. A. and Sautmann, A. (2020). "In
  Praise of Moderation: Suggestions for the Scope and Use of Pre-Analysis Plans for RCTs in
  Economics." NBER Working Paper 26993. DOI: 10.3386/w26993.
- Goldberg, L. and Johndrow, J. "A Decision Theoretic Approach to A/B Testing." arXiv:1710.03410.
- Farrell, M., Korganbekova, M. and Misra, K. "Robust A/B Decisions." arXiv:2609.07633.
- Ibanez, M. R. and Staats, B. R. (2019). "Field Experiments in Operations Management." *INFORMS
  TutORials in Operations Research*. DOI: 10.1287/educ.2019.0197.

**Standards, regulators, and official guidance — Grade B**

- ICH Harmonised Guideline E9(R1), *Addendum on Estimands and Sensitivity Analysis in Clinical
  Trials*. Step 4, 20 November 2019; EMA Step 5 effective 30 July 2020.
- U.S. Food and Drug Administration, *E9(R1) Statistical Principles for Clinical Trials: Addendum:
  Estimands and Sensitivity Analysis in Clinical Trials*, guidance, May 2021.
- CONSORT 2010 statement and extensions.
- American Economic Association RCT Registry; Center for Open Science OSF Registries; AsPredicted.
- Nature Portfolio, "Registered Reports: author guidelines" (exploratory analyses reported separately).
- U.S. Federal Reserve SR 11-7 (2011), *Supervisory Guidance on Model Risk Management*. Cited by
  analogy for documentation only.

**Industry, practitioner, and vendor — Grade C and D**

- LaunchDarkly, "An Alternative to Statistical Significance for Making Decisions with Experiments,"
  engineering blog, 2022. Grade D, dated 2022: the framing is useful, the figures are the vendor's.

**Press and anecdote — Grade E**

- No press summary or anecdote supports any quantitative claim in this topic. All figures in the
  worked example are constructed arithmetic, labelled as illustrative, internally consistent, and
  not drawn from a named company.

**Note on citations.** Volume and page numbers appear where the sources consulted confirmed them and
are omitted where they did not. The Berman and Van den Bulte figures are reported as the search
summary gave them and were not checked against the published tables. The `H/(H+B)` threshold is
presented as the standard clinical decision-analysis result; the derivation in section 5.7 is
arithmetic, and readers needing the full utility-theoretic derivation should go to Pauker and
Kassirer (1975).

## 17. Links

- Module map: [`_index.md`](_index.md)
- [`02_randomisation-units.md`](02_randomisation-units.md) — the unit the intervention is assigned to; see 5.5
- [`03_ab-cluster-designs.md`](03_ab-cluster-designs.md) — the design effect that 5.5's elimination 3 implies
- [`06_power-mde.md`](06_power-mde.md) — the arithmetic behind the MDE comparison in 5.5
- [`08_sequential-testing.md`](08_sequential-testing.md) — applying the decision rule continuously, not at a horizon
- [`10_vendor-evidence-appraisal.md`](10_vendor-evidence-appraisal.md) — the same chain applied to someone else's study
- Prerequisite, written: [`01_estimands-question-types.md`](../04_statistics_causal_inference/01_estimands-question-types.md)
  — the six slots and the three question types this topic builds on
- Prerequisite, written: [`02_probability-bootstrap.md`](../04_statistics_causal_inference/02_probability-bootstrap.md)
  — the intervals that section 6.9 reads
- Prerequisite, written: [`03_hypothesis-tests.md`](../04_statistics_causal_inference/03_hypothesis-tests.md)
  — the p-values that section 6.9 uses and then does not rely on
- Related: [`10_sensitivity-omitted-bias.md`](../04_statistics_causal_inference/10_sensitivity-omitted-bias.md)
  — how much confounding would overturn an estimate, which is the sensitivity analysis a plan should
  pre-specify
- Survey file: [`../../../course/05_experimentation_pilots_evidence.md`](../../../course/05_experimentation_pilots_evidence.md)
- Related course files: [`16_pricing_promotions_revenue_management.md`](../../../course/16_pricing_promotions_revenue_management.md),
  [`26_quality_evaluation_model_risk.md`](../../../course/26_quality_evaluation_model_risk.md)
- Return to the foundations map: [`../_index.md`](../_index.md)
