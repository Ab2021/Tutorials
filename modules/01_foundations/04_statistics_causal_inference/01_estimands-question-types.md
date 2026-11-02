> Module 04 — Statistics and Causal Inference
> Topic 01 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01A, 03

# Estimands and Question Types

---

## 1. Why this topic exists

Most analytics failures in supply chain are not modelling failures. They are failures of the
question. A model is fitted, validated, and shipped against an objective that nobody wrote down,
and the disagreement surfaces months later in a meeting where two competent people are quoting
two correct numbers that answer two different questions.

The pattern is stable enough to name. Someone asks "did the promotion work?" The team builds a
forecast, measures it against actual sales, reports 97% accuracy, and concludes that the promotion
worked. Every step is defensible and the conclusion does not follow. A forecast made after the
promotion was planned already contains the expected lift. Measuring it against the outcome tells
you whether the planning process was calibrated. It cannot tell you whether the lift would have
happened anyway, and it cannot tell you whether the promotion made money. In the worked example
in section 6, the forecast is 97.8% accurate and the promotion destroys €3,120 of margin.

Three question types are in play, and they are not a hierarchy:

- **What happened?** A descriptive question. It asks for a fact about a defined population over a
  defined window. It has no counterfactual and needs none.
- **What will happen?** A predictive question. It asks for a statement about an unobserved future
  observation, conditional on information available now and on a stated plan of action.
- **What would happen if we intervened?** A causal question. It asks for a comparison between two
  states of the world that cannot both be observed, in the same units, over the same window.

Each tolerates different data and different errors. A descriptive estimate can survive noisy
measurement, because noise that is unbiased cancels across a census and can be reconciled; it
cannot survive a change in the definition of what is being counted. A predictive model can survive
confounding, because a confounded feature is still a useful feature; it cannot survive a change in
the distribution it was trained on. A causal estimate can survive a good deal of measurement error
in the outcome, which attenuates the estimate without reversing its sign; it cannot survive
confounding, because no amount of data repairs a comparison between groups that were never
comparable.

The organising concept is the **estimand**: a precise statement of the quantity to be estimated,
fixed before the data are touched. The word is not a synonym for "metric" or "target". It names the
population, the unit, the outcome, the time window, the intervention, and the comparison, and it
is prior to any choice of estimator. An estimator is a procedure. An estimate is a number. The
estimand is the question, made specific enough to be wrong.

This precision has a provenance worth knowing. In 2019 the International Council for Harmonisation
(ICH) adopted an addendum to its statistical principles guideline, E9(R1), which requires clinical
trials to state an estimand before analysis and to align the analysis, the sensitivity analyses,
and the reporting with it. The addendum was written because two trials with the same drug, the same
endpoint, and the same data could reach opposite conclusions depending on how they handled patients
who stopped treatment — a disagreement about the question, not about the statistics. That is the
same disagreement that runs through commercial analytics, wearing different clothes: a store that
did not execute the display, a supplier that substituted a part, a promotion that ran out of stock
in week three.

> **Note:** The ICH addendum governs clinical trials. It does not govern supply chain analytics,
> and importing its vocabulary is analogy rather than compliance. The analogy is useful because
> the underlying problem — defining the question before estimating it — is identical, and because
> the addendum is the most carefully worked-out public statement of that problem.

This topic is first in the module because every later topic assumes its vocabulary. Topic 02
builds uncertainty around an estimate; you cannot say what an interval is an interval *for* until
you have said what is being estimated. Topic 06 draws causal graphs; a graph is a claim about which
comparisons are valid, which is an estimand decision. Topic 07 chooses between experiments and
quasi-experiments; that choice is determined by the estimand, not by convenience. Topic 10 asks how
fragile an estimate is; fragility is only meaningful relative to a stated target.

The practical goal is narrower than the philosophy. By the end you should be able to take a
sentence a business partner has said, decide which of the three question types it is, write the
estimand in six slots, and say what would have to be true for the answer to be right — in about
ten minutes, before anyone opens a notebook. That skill prevents more wasted modelling than any
technique in the rest of this program.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a stated business question as descriptive, predictive, or causal, and **justify**
   the classification by naming the counterfactual the question does or does not require.
2. **Write** an estimand in six slots — population, unit, outcome, time, intervention, comparison —
   precisely enough that a second analyst could compute it without asking you a question.
3. **Map** the six-slot statement onto the five attributes of the ICH E9(R1) framework, and
   **explain** where the two framings differ and why.
4. **Run** the target trial thought experiment for a proposed intervention and **identify** the
   point at which the observational data cannot emulate the trial.
5. **Distinguish** an estimand from an estimator and an estimate, and **diagnose** which of the
   three a given project description has actually specified.
6. **Detect** prediction-versus-causation confusion in a written claim, including the common
   supply chain form in which forecast accuracy is offered as evidence that an intervention
   worked.
7. **Apply** a mismatch checklist to a project proposal and **reject or repair** it before any
   model is fitted.
8. **State** the identification assumptions a causal estimand requires and **name** the diagnostic
   that would show each one failing.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture in your head: a single sentence at the top, three doors below it, and behind each
door a different object. The sentence is the same. The objects are not interchangeable.

```text
                        ONE BUSINESS QUESTION
                   "Did the Q3 promotion work?"
                                |
            +-------------------+-------------------+
            |                   |                   |
            v                   v                   v
     DESCRIPTIVE           PREDICTIVE            CAUSAL
   "What happened?"     "What will happen?"   "What would happen if?"
            |                   |                   |
     ESTIMAND              ESTIMAND              ESTIMAND
     mean units and        expected units        E[Y(promo) - Y(no promo)]
     margin per store      in the next promo     in the same 40 stores,
     in weeks 14-17        under the same plan   same 4 weeks
            |                   |                   |
     NEEDS                 NEEDS                 NEEDS
     complete,             representative      exchangeability or
     consistently          future conditions,  randomisation, overlap,
     defined records       stable relationship consistency, no
     of what occurred                            interference
            |                   |                   |
     CAN SAY               CAN SAY               CAN SAY
     "18,400 units,        "plan 18,000,         "+4,000 units
      EUR 71,760 margin"   80% PI 16.9k-19.3k"  (95% CI +1.9k to +6.1k)"
            |                   |                   |
     CANNOT SAY            CANNOT SAY            CANNOT SAY
     why, or what          what would have       what will happen next
     would have happened   happened without      quarter, or why the
     otherwise             the promotion         effect differs by store
```

The picture encodes one claim: **the question type determines the estimand, and the estimand
determines what evidence counts.** It does not encode a hierarchy. A descriptive number is not a
failed causal estimate; it is a different object, and it is often the right one. A retailer asking
"how many units did we ship last quarter" wants a census total, and offering a confidence interval
for a counterfactual would be an error.

Four properties follow, and they explain the rest of the topic.

**Property 1 — Only the causal question requires a counterfactual, and the counterfactual is
missing by construction.** For any unit, you observe the outcome under the treatment it received
and not the outcome under the alternative. Holland called this the fundamental problem of causal
inference, and it is not a data-quality problem that better instrumentation will solve. It is
structural. Everything in causal design — randomisation, matching, instruments, discontinuity — is
a way of constructing a credible substitute for the missing half.

**Property 2 — Each question type has a different failure hierarchy, so "more data" does not fix
all of them.** Descriptive errors are dominated by coverage and definition. Predictive errors are
dominated by distribution shift and leakage. Causal errors are dominated by confounding and
positivity. Adding rows improves precision in all three and bias in none. This is why a team that
responds to every problem by acquiring more data will keep being surprised.

**Property 3 — A prediction can be excellent and causally empty, and the reverse.** Shmueli (2010)
makes the first half formal: explanatory and predictive modelling have different objectives,
different validation, and different preferred models, and an underspecified model can have lower
expected prediction error than the correctly specified one. The second half is equally important
and less often said: a causally correct model can forecast badly, because causal structure does
not determine the values of the causes. Knowing that price affects demand does not tell you what
next month's price will be.

**Property 4 — The estimand is fixed before the data, and the estimator is chosen after.** This
ordering is the whole discipline. If you choose the estimator first — "let's run a
difference-in-differences" — you have let the available technique define the question. The
sequence is: decision, question type, estimand, identification strategy, design, estimator,
sensitivity analysis. Each step constrains the next, and skipping to step six is the most common
professional failure in this module.

### 3.1 What each question type will and will not tolerate

| Question type | What "wrong" means | Data it tolerates | Error it cannot tolerate | Typical design |
|---|---|---|---|---|
| Descriptive | The number does not match the defined population, period, or definition | Unbiased measurement noise; partial records if coverage is known | Non-coverage; definitional drift; double counting | Census, reconciliation, audit sample |
| Predictive | The forecast is wrong on the deployment distribution, or right for the wrong reason | Confounding; collinearity; proxy variables | Distribution shift; leakage; target constructed after the fact | Rolling-origin backtest, holdout, decision-loss evaluation |
| Causal | The contrast is between groups that were not comparable, or the intervention is not well defined | Classical measurement error in the outcome; modest overlap problems | Confounding; positivity violations; interference; inconsistent treatment definitions | Experiment, difference-in-differences, instrument, regression discontinuity |

Read the third column twice. It says that the three types accept different imperfections. A
descriptive estimate tolerates noise but not a shifted boundary. A predictive model tolerates a
confounded feature but not a shifted population. A causal estimate tolerates a noisy ruler but not
a rigged comparison. An analyst who applies one standard of rigour to all three will either
over-engineer the descriptive work or under-engineer the causal work, and the second is the
expensive mistake.

### 3.2 Why "should we" is not automatically causal

A common misreading of the three-way split is that decision questions are causal questions. They
are not, always. Consider "which of these two suppliers should we use next quarter?" If the choice
does not change the suppliers' behaviour, the required object is a prediction of each supplier's
lead time and defect rate, combined with a loss function over the consequences. No counterfactual
is needed. Now consider "should we change the receiving process?" The choice changes the process,
so the required object is causal: what would throughput be under the new process.

The test is mechanical. **Does the action change the outcome, or only select among outcomes that
will happen anyway?** If it selects, you need prediction plus a loss. If it changes, you need
causation. Confusing the two produces the strange failure in which a team runs an expensive
experiment to answer a question that a good forecast would have settled, or ships a forecast to
answer a question that only an experiment could settle.

### 3.3 The three doors are not a sequence

A project often needs all three, in a specific order, and the order matters. A demand planning
system needs a descriptive estimate of history, a predictive model of future demand, and — if it
is to recommend a price change — a causal estimate of the price response. These are three
deliverables with three estimands, and they should be documented separately. When they are merged
into one "model", the causal component quietly inherits the predictive component's validation
standard, which is how a system ends up optimising a price against a correlation.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Estimand | The precise quantity to be estimated, defined before data are analysed | Used as a synonym for "metric" or "KPI" | Structured statement, no unit of its own |
| Estimator | The procedure or algorithm that produces an estimate of the estimand | Named when the estimand was requested | Method name |
| Estimate | The numerical value produced by an estimator on a dataset | Presented as if it were the estimand | Same unit as the estimand |
| Question type | Descriptive, predictive, or causal; determined by whether a counterfactual is required | Assumed from the job title of the requester | Categorical |
| Descriptive question | Asks for a fact about a defined population, period, and definition | Treated as a trivial precursor to "real" analysis | Categorical |
| Predictive question | Asks for a statement about an unobserved observation, conditional on stated information | Confused with a causal question when phrased as "will X cause Y" | Categorical |
| Causal question | Asks for a contrast between potential outcomes under two or more well-defined interventions | Confused with a descriptive comparison of groups | Categorical |
| Counterfactual | The outcome that would have occurred under an intervention not received | Treated as an unobserved fact that better data would reveal | Potential outcome value |
| Potential outcome | The value of the outcome for a unit under a specified treatment condition, written `Y(a)` | Confused with the observed outcome | Same unit as the outcome |
| Intervention | A well-defined action, including dose, timing, duration, and delivery | Described as a slogan ("improve collaboration") | Protocol-level specification |
| Comparator | The alternative condition against which the intervention is contrasted | Left implicit, or assumed to be "nothing" | Protocol-level specification |
| Unit of analysis | The entity whose outcome is measured | Confused with the unit of assignment | Entity key |
| Unit of assignment | The entity to which the intervention is applied | Assumed to equal the unit of analysis | Entity key |
| Target population | The set of units to which the estimand is intended to generalise | Left as "our customers" | Eligibility criteria |
| Time zero | The moment at which eligibility is assessed and follow-up begins | Set to the first observed transaction | Date or timestamp |
| Follow-up horizon | The period over which the outcome is measured | Chosen to fit available data | Duration, with a start rule |
| Outcome variable | The measured quantity, including its instrument and aggregation | Named without its measurement rule | Unit plus definition |
| Population-level summary | The summary measure that makes two conditions comparable (mean difference, ratio, quantile difference) | Replaced by "the effect" | Contrast with a scale |
| Contrast | The comparison between summary values under two conditions | Omitted, leaving a naked number | Difference, ratio, or other functional |
| Intercurrent event | An event after time zero that affects the interpretation of the outcome, such as non-execution or a competing event | Ignored, or handled ad hoc in the analysis | Event definition plus date |
| Treatment policy strategy | Analyse outcomes regardless of whether the intervention was received as planned | Confused with "intention to treat" in all respects | Strategy label |
| Hypothetical strategy | Estimate the outcome that would have been observed had the intercurrent event not occurred | Assumed to be obtainable from the observed data without assumptions | Strategy label |
| Composite strategy | Incorporate the intercurrent event into the outcome definition | Used without stating the weighting | Strategy label |
| Principal stratum strategy | Target the subpopulation in which the intercurrent event would or would not occur | Confused with a subgroup analysis | Strategy label |
| While-on-treatment strategy | Use the outcome observed up to the time of the intercurrent event | Treated as a full-horizon effect | Strategy label |
| Intention-to-treat (ITT) | Analysis by assigned condition, regardless of adherence | Used loosely as "we analysed everyone" | Analysis principle |
| Per-protocol | Analysis restricted to units that adhered to the assigned condition | Assumed to be unbiased without adjustment | Analysis principle |
| Target trial | The hypothetical randomised trial that the observational analysis is attempting to emulate | Skipped entirely | Protocol specification |
| Exchangeability | The condition that, within covariate strata, treated and untreated units would have had the same outcomes absent treatment | Asserted rather than argued | Assumption |
| Positivity | The condition that every covariate pattern has a nonzero probability of each treatment condition | Ignored in high-dimensional matching | Assumption |
| Consistency | The observed outcome equals the potential outcome under the treatment actually received | Broken by vague treatment definitions | Assumption |
| Stable unit treatment value assumption (SUTVA) | No interference between units, and no hidden versions of the treatment | Assumed without checking for shared supply | Assumption |
| Interference | One unit's outcome depends on another unit's treatment | Ignored in marketplace and allocation settings | Assumption violation |
| Confounder | A variable that affects both the treatment and the outcome | Confused with any correlate of the outcome | Causal-graph node |
| Collider | A variable affected by both treatment and outcome; conditioning on it induces association | Controlled for because it "seems relevant" | Causal-graph node |
| Mediator | A variable on the causal path from treatment to outcome | Controlled for when the total effect is wanted | Causal-graph node |
| Selection on observables | The assumption that all confounders are measured | Asserted from data availability | Assumption |
| Prediction interval | An interval for a future observation, wider than a confidence interval for a mean | Reported as a confidence interval | Outcome units |
| Calibration | Agreement between predicted probabilities or intervals and observed frequencies | Replaced by a single accuracy score | Proportion, or reliability slope |
| Distribution shift | A change in the joint distribution of features and outcome between training and deployment | Detected only after performance falls | Divergence measure |
| Leakage | Information in the training data that would not be available at prediction time | Discovered only in production | Feature audit |
| Incrementality | The part of an outcome that would not have occurred without the intervention | Equated with observed lift | Outcome units |
| Baseline | The counterfactual level against which an intervention's outcome is compared | Taken as the pre-period actual | Outcome units |
| Lift | The difference between observed and baseline outcome | Reported without the baseline's definition | Outcome units or percent |
| Decision rule | The mapping from an estimate to an action, fixed in advance | Invented after seeing the estimate | If-then specification |
| Loss function | The cost of each possible error, used to choose among models or actions | Assumed symmetric | Currency per unit of error |
| Sensitivity analysis | An analysis that probes whether conclusions hold under departures from assumptions | Run only when the main result is doubted | Analysis label |
| Pre-specification | Fixing the estimand, analysis, and decision rule before seeing outcome data | Confused with "we discussed it internally" | Documented plan |

Four of these deserve expansion, because they carry most of the misunderstanding.

**Estimand, estimator, estimate.** The three words are used interchangeably in most project
documents and they name three different things. The estimand is the question, stated precisely
enough to be answerable: the average difference in units sold per store over four weeks between
running this promotion and not running it, in these 40 stores. The estimator is the procedure:
a difference-in-differences regression with store and week fixed effects. The estimate is the
number: 4,000 units, 95% interval 1,900 to 6,100. When a stakeholder asks "what method did you
use", they are asking about the estimator, and answering with the estimate is a category error.
When a regulator or an auditor asks "what are you estimating", answering with the estimator is a
more serious one, because it hides the choice that mattered.

**Counterfactual and baseline.** A counterfactual is a potential outcome under a specified
intervention; it is a defined quantity that is missing. A baseline is whatever number the analyst
chose to compare against. They coincide only when the baseline is a credible estimate of the
counterfactual. In practice, "baseline" is often the pre-period actual, which conflates the
intervention's effect with everything else that changed between the two periods. The whole of
causal design is the business of making a baseline into a counterfactual.

**Intercurrent event.** The ICH term for anything after time zero that affects the interpretation
of the outcome: a patient stopping treatment, a store not executing a display, a supplier
substituting a part, a promotion running out of stock. It is not missing data and it is not a
protocol deviation in the ordinary sense. It is a fork in what the question means, and the
addendum's contribution is to force an explicit choice among five strategies for handling it.
Supply chain analytics is full of intercurrent events and almost never names them.

**Confounder, collider, mediator.** These are three positions a variable can occupy relative to
treatment and outcome, and the correct analytic response differs for each: adjust for a
confounder, do not adjust for a collider, and adjust for a mediator only if you want a direct
rather than a total effect. Topic 06 treats the graph machinery. Here the point is that the
distinction is an estimand question before it is a modelling question, because "the effect of X on
Y" is ambiguous until you say whether you mean the total effect or the effect net of M.

## 5. The mechanics, step by step

This section turns a sentence into an estimand. The sequence is fixed: name the decision, classify
the question, write the six slots, reconcile with the ICH attributes, run the target trial,
specify the comparison, name the assumptions, choose the design, pre-specify the sensitivity
analysis. Nine steps, roughly ten minutes of thinking, and it must happen before any model.

### 5.1 Step 1 — Name the decision and the decision-maker

An analytics question that does not change a decision is a curiosity. The first step is not a
formality: it determines the population, the horizon, and the loss.

Ask: **what will someone do differently if the answer is A rather than B?** If the answer is
"nothing", stop. If the answer is "they will feel informed", stop. If the answer is "they will
renegotiate the supplier contract in March", you have a population (that supplier's lanes), a
horizon (through the contract term), and a loss (the cost of a wrong renegotiation).

Two follow-ups make the decision concrete. **Who acts?** A role, not a department. **When?** A
date, because a causal estimate delivered after the decision has the same value as no estimate.

> **Note:** "Name the decision" is not a demand that every analysis be immediately actionable.
> Descriptive reporting often supports a decision that is diffuse and slow. The demand is that
> you can *state* the decision, not that it is imminent.

### 5.2 Step 2 — Classify the question by its counterfactual

The classification is mechanical once you ask the right question: **does answering this require
comparing what happened with what would have happened under a different action?**

| The request sounds like | Question type | Because | What it needs |
|---|---|---|---|
| "How many units did we ship in Q3?" | Descriptive | No alternative state is invoked | A defined population, period, and counting rule |
| "What share of orders were late?" | Descriptive | Same | A denominator and a lateness rule |
| "How many units will we sell next month?" | Predictive | Asks about an unobserved observation under the current plan | A training distribution that resembles deployment |
| "What will service level be if we hold 12 days of cover?" | Predictive, conditional on a policy | It evaluates a plan, not an intervention that changes behaviour | A model of demand and lead time under the stated policy |
| "Did the promotion increase sales?" | Causal | Compares promoted with unpromoted potential outcomes | A counterfactual and an identification strategy |
| "What would happen if we cut lead time by a week?" | Causal | The action changes the outcome | Same, plus a mechanism |
| "Why did service fall in March?" | Causal (mechanism) | Asks for attribution among causes | A causal graph and, often, mediation analysis |
| "Which supplier should we use?" | Predictive plus a loss | The choice selects among outcomes that will occur anyway | Predicted outcomes and a cost model |
| "Should we change the receiving process?" | Causal | The choice changes the process | A target trial and an identification strategy |

The distinction between rows five and eight is the one most often missed. Both are phrased as
questions about suppliers and both sound like decisions. Only the fifth changes the world.

Beware three linguistic traps. **"Will X cause Y?"** is causal, not predictive, even though it is
in the future tense; the future tense describes when the outcome is measured, not what is being
estimated. **"What is the impact of X?"** is causal, and "impact" is doing the work that "effect"
should do. **"What drives Y?"** is ambiguous between prediction (which features are associated),
mechanism (which pathways), and intervention (what would change Y if we moved it). Ask which one
before answering.

### 5.3 Step 3 — Write the six slots

An estimand is written in six slots. Each must be specific enough that a second analyst would
compute the same quantity.

| Slot | The question it answers | Failure if vague |
|---|---|---|
| **Population** | Which units, meeting which eligibility criteria, in which period? | The estimate generalises to nobody in particular |
| **Unit** | What is one observation, and is it the same entity the intervention is applied to? | Standard errors computed at the wrong level; effects attributed to the wrong entity |
| **Outcome** | Which variable, measured how, aggregated how, over what window? | Two teams compute two different outcomes and call them the same |
| **Time** | When does follow-up start, and how long does it run? | Immortal time bias; effects contaminated by pre-period selection |
| **Intervention** | What exactly is done, to whom, at what dose, for how long, and what is the comparator? | The "effect" is an average over several different treatments |
| **Comparison** | Which summary measure, on which scale, contrasting which two conditions? | A naked number with no comparator |

Two slots carry disproportionate weight in supply chain work.

**Unit of analysis versus unit of assignment.** A promotion is assigned to a store and measured
on baskets. A supplier policy is assigned to a lane and measured on orders. A routing change is
assigned to a depot and measured on parcels. When the two differ, the outcome is correlated within
the assignment unit, and an analysis that treats baskets as independent will report standard
errors that are too small by a large factor. The estimand must state both, even when they coincide.

**Time zero.** In a trial, time zero is the moment of randomisation, and eligibility is assessed
just before it. In observational data, time zero is usually chosen by the analyst and is the
single most common source of a specific bias: if units are selected into the treated group because
of something that happened before follow-up began, the pre-follow-up period is "immortal" and the
treated group looks better by construction. The fix is to align time zero with the decision point
and to assess eligibility at that moment, not earlier.

### 5.4 Step 4 — Reconcile with the ICH E9(R1) attributes

The clinical-trials framework uses five attributes rather than six slots. The mapping is not
one-to-one, and understanding why is instructive.

| ICH E9(R1) attribute | Six-slot equivalent | What the clinical framing adds |
|---|---|---|
| Treatment | Intervention plus comparator | Requires the comparator to be specified as a condition, not as "no treatment" |
| Population | Population plus unit | Allows a principal stratum defined by an intercurrent event |
| Variable (endpoint) | Outcome plus time | Embeds the handling of intercurrent events into the variable itself |
| Population-level summary | Comparison | Names the summary measure explicitly, separate from the contrast |
| Strategy for handling intercurrent events | No direct equivalent | Forces one of five strategies to be chosen and defended |

The two framings differ in emphasis rather than substance. The six-slot version separates
**time** and **unit** because supply chain analyses routinely get them wrong in ways that clinical
trials do not: observational windows are chosen by the analyst, and assignment and measurement
often occur at different grains. The five-attribute version separates **summary measure** from
**contrast** because the choice between a difference and a ratio is a substantive decision that
analysts otherwise make by habit. A complete statement uses both: six slots for the substance, and
the ICH vocabulary for the intercurrent-event question.

The five strategies for intercurrent events are worth memorising, because they transfer directly.

| Strategy | Clinical form | Supply chain form |
|---|---|---|
| Treatment policy | Analyse outcomes regardless of whether the patient took the drug | Analyse sales regardless of whether the store executed the display |
| Hypothetical | What would the outcome have been had the patient not stopped treatment? | What would sales have been had every store executed the display? |
| Composite | Outcome is "response, or discontinuation counted as failure" | Outcome is "sales uplift, or non-compliance counted as no uplift" |
| While on treatment | Effect only during the period the patient took the drug | Effect only during the weeks the promotion was live on shelf |
| Principal stratum | Effect in the patients who would adhere | Effect in the stores that would comply if assigned |

The choice changes the number, sometimes the sign. A promotion analysed under a treatment-policy
strategy includes stores that never set up the display, so the estimated effect is diluted by
non-compliance. The same promotion analysed under a hypothetical strategy estimates the effect if
execution were perfect, which is the right number for a decision about whether the *mechanism*
works and the wrong number for a decision about whether to *run the programme* as it is actually
executed.

> **Note:** The ICH addendum separates sensitivity analyses, which probe robustness to departures
> from assumptions while targeting the same estimand, from supplementary analyses, which target
> different estimands. Mixing the two is a common reporting failure: a "sensitivity analysis" that
> quietly estimates a per-protocol effect is not a sensitivity analysis, it is a second study.

### 5.5 Step 5 — Run the target trial thought experiment

The target trial is a disciplined way to expose an ill-posed causal question. You specify the
randomised trial you would run if you could, in full protocol, and only then ask whether the
observational data can emulate it.

The protocol has seven components. Write them down even when the answer is "we cannot".

1. **Eligibility criteria.** Which units, at which moment, are eligible? Be strict: "stores
   carrying the item for at least 26 weeks, with no planned remodel in the window".
2. **Treatment strategies.** At least two, each well defined, including dose and duration. "Run a
   20% discount for four weeks" against "hold price". Not "promote" against "don't".
3. **Assignment.** How would units be assigned, and at what level? Individual, store, or market.
   This determines the unit of inference.
4. **Time zero.** The moment of assignment, with eligibility assessed immediately before.
5. **Follow-up.** Start, end, and the outcome measurement schedule.
6. **Outcome.** The variable, its measurement, and its aggregation.
7. **Causal contrast.** The summary measure and the population it refers to.

Then the second half of the exercise: **which component cannot be emulated?** The answer is
usually one of four. Eligibility may be unmeasurable, because the data do not record the criterion.
Assignment may be confounded, because managers promoted stores that were already growing. Time
zero may be ambiguous, because the promotion's start date is recorded inconsistently. Interference
may be present, because promoted stores drew demand from neighbouring stores that were not
promoted.

Naming the component that cannot be emulated is the deliverable. It converts "we can't do causal
inference" into "we can estimate this, but the estimate assumes no spillover between stores within
15 km, and here is the test for that".

### 5.6 Step 6 — Specify the comparison and its scale

Every estimand ends in a contrast, and every contrast has a scale. The same data can support a
difference, a ratio, or a quantile difference, and the three answer different questions.

| Summary measure | Answers | Scale | Common supply chain use |
|---|---|---|---|
| Difference in means | "How many more units?" | Outcome units | Promotion lift, policy effect on throughput |
| Ratio of means | "How many times as many?" | Dimensionless | Relative lift; comparability across products |
| Difference in proportions | "How many percentage points?" | Percentage points | Fill rate, on-time delivery, defect rate |
| Ratio of odds | "How much more likely?" | Odds ratio | Rare events: recall, stock-out, safety incident |
| Difference in quantiles | "How much does the tail move?" | Outcome units at level q | Lead-time p95, service-level tail |
| Expected loss difference | "How much money?" | Currency | Decisions under an explicit cost model |

Three rules. **State the scale with the estimate**, because "the promotion lifted sales by 30%"
and "the promotion lifted sales by 4,000 units" are different claims with different denominators.
**Do not switch scale after seeing the result**, because a ratio that looks large can be a small
absolute difference on a small base. **Report absolute and relative together** when the decision
depends on both, which it usually does.

### 5.7 Step 7 — Name the assumptions that make the comparison valid

Each question type carries a different assumption set. The point of naming them is not ritual; it
is to know which diagnostic would falsify the estimate.

| Question type | Core assumption | What it means | Diagnostic that would show it failing |
|---|---|---|---|
| Descriptive | Complete coverage of the defined population | Every unit that meets the definition is counted once | Reconciliation against an independent total; duplicate check |
| Descriptive | Definitional stability over the window | The counting rule did not change mid-period | Compare the data dictionary version across the period |
| Predictive | The deployment distribution resembles training | Feature and outcome relationships are stable | Backtest on the most recent period; drift monitor on feature distributions |
| Predictive | No information leakage | Every feature is available at prediction time | Rebuild features as of the prediction timestamp; audit for post-outcome fields |
| Causal | Exchangeability (no unmeasured confounding) | Within strata, treated and untreated would have had the same outcome absent treatment | Negative-control outcome; placebo test; sensitivity analysis for unmeasured confounding |
| Causal | Positivity | Every covariate pattern has nonzero probability of each condition | Overlap plots; count of units with extreme propensity scores |
| Causal | Consistency | The treatment is defined so that the observed outcome equals the potential outcome under it | Audit for multiple versions of the treatment |
| Causal | No interference (SUTVA) | One unit's outcome does not depend on another's treatment | Spillover test across geographic or network distance |

The exchangeability assumption cannot be tested directly, and this is the single most important
asymmetry in the table. Every other row has a diagnostic. For exchangeability, you can only argue
from design, measure the sensitivity of the conclusion to a plausible unmeasured confounder, and
look for a negative-control outcome that should show no effect. Topic 10 develops the sensitivity
machinery; here it is enough to know that the assumption is argued, not verified.

### 5.8 Step 8 — Choose the design, then the estimator

Design precedes estimator. The design is the comparison structure — who is compared with whom,
over what window, with what adjustment. The estimator is the arithmetic that produces the number
from that structure.

| Design | Estimand it can support | Requires | Fails when |
|---|---|---|---|
| Randomised experiment | Any well-defined causal contrast | Feasibility, ethics, and an assignment mechanism | Interference, non-compliance, or a population too narrow to generalise |
| Difference-in-differences | Effect of a policy on the treated, with a parallel-trends counterfactual | A credible untreated group with parallel pre-trends | The groups were on different trajectories before the intervention |
| Instrumental variable | Effect in the compliers | A variable that shifts treatment but affects the outcome only through it | The exclusion restriction is violated |
| Regression discontinuity | Effect at the threshold | A continuous assignment variable with a cutoff | Units manipulate the assignment variable |
| Matching on observables | Effect in the matched population | All confounders measured, overlap | An unmeasured confounder drives selection |
| Before-and-after | A descriptive change, not a causal effect | Nothing | Anything else changed at the same time |

The last row is included deliberately. A before-and-after comparison is a legitimate descriptive
statement and an illegitimate causal one. It appears in causal clothing constantly, usually
because nobody asked what else changed.

### 5.9 Step 9 — Pre-specify the sensitivity analysis and the decision rule

Two commitments make the analysis honest, and both must be made before the outcome data are seen.

**The sensitivity analysis** asks: how strong would an unmeasured confounder have to be to overturn
this conclusion? State the answer as a magnitude, not a verdict. "An unmeasured factor would need
to raise promoted-store sales by more than 9% relative to control to explain the entire estimate"
is a usable statement; "the result is robust" is not.

**The decision rule** maps the estimate to an action, including the action "do nothing". Write it
as an if-then: "if the lower bound of the interval exceeds the €2,000 break-even margin, roll out;
if the interval spans zero, run a second test in a different market; if the point estimate is
negative, stop the programme." A decision rule fixed in advance prevents the most common form of
analytical dishonesty, which is choosing the threshold after seeing the number.

### 5.10 The mismatch checklist

This is the artefact the topic exists to produce. Run it before any model is fitted. A "no" on
items 1, 2, 4, 6, or 9 is a stop.

| # | Check | Pass condition | Fail signal |
|---:|---|---|---|
| 1 | The decision is named | A role will act differently depending on the answer | "It will inform strategy" |
| 2 | The comparison is stated | There are two named conditions or a named benchmark | One number with no comparator |
| 3 | The question type is stated | Descriptive, predictive, or causal, in writing | The word "impact" or "driver" appears unexamined |
| 4 | The six slots are filled | Each slot is specific enough to compute from | "All customers" or "recently" |
| 5 | The unit of assignment is stated | It matches, or is explicitly distinguished from, the unit of analysis | Store-level treatment, basket-level inference, no clustering |
| 6 | Time zero is defined | Follow-up starts at a decision point, not at first observation | "We used all available history" |
| 7 | The comparator is a condition | It is described with the same specificity as the intervention | "Business as usual", undefined |
| 8 | Intercurrent events are named | At least the obvious ones, with a handling strategy | Non-execution discovered only in the robustness check |
| 9 | The identification argument is stated | One paragraph on why the comparison is credible | "We controlled for everything we had" |
| 10 | The loss function is stated | The cost of each error direction is named or bounded | Symmetric accuracy assumed for an asymmetric decision |
| 11 | The sensitivity analysis is pre-specified | A named departure and a threshold for concern | Robustness checked only if the result is doubted |
| 12 | The decision rule is pre-specified | An if-then mapping to action | The recommendation is written after the result |

Items 2, 5, and 7 are the ones that catch prediction-versus-causation confusion, because a
predictive question can pass all three only if the "comparison" is a benchmark forecast rather
than a counterfactual condition. Item 9 catches the opposite error, in which a causal claim is
made with no identification argument at all.

## 6. Worked example

### 6.1 Setting

A distributor sells one product through 120 stores. In weeks 14 to 17 it ran a four-week
price promotion in 40 stores. The stores were not chosen at random: the regional manager chose
stores where the product was already selling well and where a competitor had recently withdrawn a
competing item. The remaining 80 stores continued at full price.

The analytics team is asked three questions in the same meeting, and the answers are produced from
the same table. All figures are illustrative and internally consistent.

### 6.2 The three questions and their estimands

| | Question A | Question B | Question C |
|---|---|---|---|
| As asked | "What did we sell during the promotion?" | "What should we plan for the next promotion?" | "Did the promotion work?" |
| Type | Descriptive | Predictive | Causal |
| Population | The 40 promoted stores | The same 40 stores, next comparable promotion | The 40 promoted stores |
| Unit | Store-week | Store-week | Store |
| Outcome | Units sold; gross margin at promo price | Units sold | Gross margin per store over four weeks |
| Time | Weeks 14–17 | Weeks 14–17 of the next cycle | Weeks 14–17, time zero at week 14 |
| Intervention | None stated | The planned promotion, identical terms | 20% discount for four weeks, versus full price |
| Comparison | None; a fact | Point forecast plus 80% interval | Promoted versus counterfactual full-price margin |

The three questions share a dataset and nothing else. Question A cannot be wrong in the way
Question C can be wrong, because it makes no claim about a world that did not occur.

### 6.3 Question A: the descriptive answer

| Measure | Weeks 10–13 | Weeks 14–17 | Change |
|---|---:|---:|---:|
| Units, 40 stores | 13,100 | 18,400 | +5,300 (+40.5%) |
| Gross margin per unit | €5.20 | €3.90 | −€1.30 |
| Gross margin, total | €68,120 | €71,760 | +€3,640 (+5.3%) |

This is a complete and correct description: 18,400 units, €71,760 of gross margin, in 40 named
stores, over four named weeks. It is the right answer to Question A and it supports no causal
claim. The comparison with weeks 10–13 is offered as context, not as a counterfactual, and the
distinction matters because the category was growing in the same period.

### 6.4 Question B: the predictive answer

The plan for the next promotion, built from the same terms and the same stores, is 18,000 units
with an 80% prediction interval of 16,900 to 19,300. The model was validated by rolling-origin
backtest on the previous six promotions in comparable stores, with a mean absolute percentage
error of 4.1%.

Two things this answer is and one thing it is not. It is a conditional expectation given the plan,
useful for buying and staffing decisions. It is a calibrated interval, useful for a service-level
commitment. It is not evidence that the promotion caused anything, because the promotion is an
input to the forecast, not a treatment being tested.

### 6.5 Question C: the target trial

The team writes the trial it would have run.

- **Eligibility:** stores carrying the item for 26 weeks or more, no remodel in the window, no
  competing promotion in the prior four weeks.
- **Strategies:** 20% discount for four weeks, versus full price for four weeks.
- **Assignment:** randomised at store level, 40 treated and 80 control.
- **Time zero:** the first day of week 14.
- **Follow-up:** four weeks, margin measured weekly and summed.
- **Outcome:** gross margin per store, in euros, at the price actually charged.
- **Contrast:** difference in mean margin per store, with a 95% interval.

Then the emulation check. Eligibility is measurable. Assignment is not random, and the selection
was explicitly on prior sales and a competitor exit — that is the emulation failure. Time zero is
recoverable from the promotion calendar. Interference is possible if promoted stores drew demand
from nearby unpromoted stores, which would bias the estimate upward.

The design that follows is a difference-in-differences on matched control stores, with store and
week fixed effects, plus a spillover check comparing control stores within 15 km of a promoted
store against those farther away.

### 6.6 The arithmetic

The control group is 80 stores. To compare like with like, express it per 40 stores by halving its
totals.

| Group | Weeks 10–13 | Weeks 14–17 | Change |
|---|---:|---:|---:|
| Promoted (40 stores) | 13,100 | 18,400 | +5,300 |
| Control (80 stores, halved) | 13,000 | 14,300 | +1,300 |
| Difference-in-differences | | | **+4,000 units** |

```text
naive before-and-after change      = 18,400 - 13,100        = +5,300 units
control change (per 40 stores)     = 14,300 - 13,000        = +1,300 units
difference-in-differences          = 5,300 - 1,300          = +4,000 units
counterfactual volume              = 13,100 + 1,300         = 14,400 units
incremental lift                   = 4,000 / 14,400         = 27.8%
overstatement from the naive comparison = 1,300 / 4,000    = 32.5%
```

The naive comparison overstates the effect by 32.5%, because the category was growing and the
promoted stores were selected for growth. The DiD estimate is 4,000 units, with a 95% interval of
1,900 to 6,100 units from a cluster-robust standard error at store level.

### 6.7 Why 97.8% forecast accuracy is not evidence

The plan for the promoted stores was 18,000 units; the actual was 18,400. The weighted absolute
percentage error is `400 / 18,400 = 2.17%`, so the reported accuracy is 97.8%.

| Quantity | Value | What it measures |
|---|---:|---|
| Plan | 18,000 units | A conditional expectation given that the promotion would run |
| Actual | 18,400 units | The outcome in the world where it ran |
| WAPE | 2.17% | Calibration of the planning process |
| Causal effect | +4,000 units | Difference from a world where it did not run |

The forecast and the actual are both measured in the same world. The comparison contains no
counterfactual, so it cannot identify an effect. Worse, the forecast was built by a model that
included the promotion as a feature, so its accuracy is partly a measure of how well the model
encoded the promotion's expected effect. A model that perfectly predicts the outcome of a
value-destroying promotion scores 100%.

Three specific versions of this error appear in practice:

1. **Accuracy reported to the steering committee as "the promotion performed to plan."** True and
   irrelevant. Performing to plan is consistent with a plan that should not have been made.
2. **Accuracy computed after the fact with a model that includes a promotion dummy.** The dummy
   absorbs the effect, and the residual is small by construction. This is leakage wearing a
   validation report.
3. **Accuracy of a counterfactual baseline model used as validation of the uplift.** The baseline
   model is validated on non-promotional periods, where its target is observed. The uplift is the
   difference between two predictions in a period where neither potential outcome is observed.
   Fitting the baseline well says nothing about the difference.

### 6.8 The margin decomposition

The causal answer is not "more units". It is whether the units were worth having. At full price
the margin is €5.20 per unit; at the promotional price it is €3.90. The counterfactual is 14,400
units at €5.20.

| Component | Calculation | Value |
|---|---|---:|
| Actual margin | 18,400 × €3.90 | €71,760 |
| Counterfactual margin | 14,400 × €5.20 | €74,880 |
| Causal effect on margin | 71,760 − 74,880 | **−€3,120** |
| Margin gained on incremental units | 4,000 × €3.90 | +€15,600 |
| Margin forgone on baseline units | 14,400 × €1.30 | −€18,720 |
| Net | 15,600 − 18,720 | −€3,120 |

The decomposition is the useful part. The promotion gained €15,600 on units that would not
otherwise have sold, and gave away €18,720 on units that would have sold anyway. The second term
is larger, and it is invisible in every volume-based report.

Break-even requires the incremental units to compensate for the discount on the baseline:

```text
break-even incremental units = (14,400 x 1.30) / 3.90 = 4,800 units
break-even lift              = 4,800 / 14,400        = 33.3%
observed lift               = 4,000 / 14,400         = 27.8%
```

The promotion missed break-even by 5.5 percentage points of lift. That is the decision-relevant
number, and it took the causal estimate to produce it. A volume report would have shown +40.5%
and a margin report +5.3%; both are correct and both point the wrong way.

> **Note:** The example assumes the discount applies to all units sold in the window, that no
> forward buying occurs, and that the control group's trend is a valid counterfactual. Each is
> testable: forward buying shows up as a post-promotion trough, and the parallel-trends assumption
> is checked on weeks 1–13. The arithmetic isolates the estimand question; it does not substitute
> for those checks.

### 6.9 What the three answers permit you to say

| Claim | Permitted by A | Permitted by B | Permitted by C |
|---|---|---|---|
| "We sold 18,400 units in weeks 14–17" | Yes | No | No |
| "We expect 18,000 units next time, ±1,200" | No | Yes | No |
| "The promotion raised volume by about 4,000 units" | No | No | Yes, under stated assumptions |
| "The promotion was profitable" | No | No | Yes, and the answer is no |
| "Stores with the promotion sold more" | Yes, descriptively | No | Yes, but it is not the effect |
| "The forecast was accurate" | No | Yes | No |

A single meeting can legitimately contain all three columns. What it cannot contain is a sentence
that borrows the credibility of one column to support a claim from another.

## 7. Code, query, or calculation

This topic is about specification, so the artefact is a specification, not a program. The block
below is the estimand template applied to two supply chain questions. It is deliberately short
enough to copy into a project charter by hand.

```text
SLOT            PROMOTIONAL LIFT                SUPPLIER OTIF POLICY
--------------  -----------------------------  -----------------------------
Population      40 stores, item carried 26+    180 lanes, tier-1 suppliers,
                weeks, no competing promo      spend > EUR 1m/yr, 2025-2026
Unit            store-week (assignment:        lane-month (assignment:
                store)                         supplier)
Outcome         gross margin EUR per store-    OTIF proportion, line-level,
                week, at price charged         rolled to lane-month
Time            time zero = week 14 start;     time zero = contract start;
                4-week follow-up              12-month follow-up
Intervention    20% discount, 4 weeks, vs      new allocation rule with
                full price, same stores        2-week frozen window, vs
                                               existing weekly rule
Comparison      difference in mean margin      difference in OTIF proportion,
                per store, 95% CI              percentage points, 95% CI
ICE strategy    treatment policy (analyse      principal stratum (lanes that
                all assigned stores)           would comply with the rule)
```

Read the last row of each column. The promotion is analysed under a treatment policy, because the
decision is whether to run the programme as it is actually executed, non-compliant stores
included. The supplier policy is analysed under a principal stratum, because the decision is
whether the rule works when followed, and lanes that cannot follow it would dilute the answer.

Change either choice and the number changes. That is the point of writing it down.

Two calculations belong beside the template, and both are one line.

```text
incremental effect (DiD) = (Y_treat_post - Y_treat_pre) - (Y_ctrl_post - Y_ctrl_pre)
margin effect            = incremental units x promo margin - counterfactual units x discount
```

The first is the comparison. The second is the decision. Neither is a model, and both require the
estimand to have been fixed first.

## 8. Data, standards, and regulatory context

There is no standard that governs estimands in supply chain analytics. There is a very carefully
written standard that governs them in clinical trials, a set of reporting conventions in
observational research, and a small and uneven body of industry guidance on measurement. Knowing
which is which prevents the two common errors: claiming a compliance obligation that does not
exist, and inventing a definition where a good one is available.

### 8.1 The clinical-trials standard and its scope

| Instrument | Issued | What it establishes | Applies to supply chain? | Grade |
|---|---|---|---|---|
| ICH E9(R1) addendum | Adopted 20 November 2019; EMA effective 30 July 2020; FDA guidance May 2021 | Estimand defined by five attributes; five strategies for intercurrent events; alignment of analysis, sensitivity analysis, and reporting | No; analogy only | B |
| ICH E9 (original) | 1998 | Statistical principles for clinical trials, including intention-to-treat | No; historical context | B |
| CONSORT 2010 | 2010, updated extensions | Reporting of randomised trials, including assigned versus analysed populations | No; useful as a checklist of what a causal report must disclose | B |
| PICO / PICOTS | Evidence-based medicine convention | Population, Intervention, Comparator, Outcome, plus Timing and Setting | No; the closest existing template to the six slots | B |

The addendum's structure is worth stating because it is the model for everything in section 5. It
defines an estimand through five attributes — treatment, population, variable, population-level
summary, and strategy for handling intercurrent events — and requires that the primary analysis,
the sensitivity analyses, and the reported conclusions all target the same estimand. It also
distinguishes sensitivity analyses, which probe robustness to departures from assumptions, from
supplementary analyses, which address different questions.

> **Note:** The addendum is a regulatory document for medicinal product trials. Nothing in it
> binds a retailer, a manufacturer, or a logistics operator. Its value here is that it is the
> clearest public statement of a general problem, written by people who were forced to resolve it
> because the cost of ambiguity was visible and litigated. Treat it as a reference model, and do
> not describe an analytics process as "ICH-compliant".

### 8.2 The transfer into analytics practice

The migration is real but partial, and the partiality is instructive. The sources returned in
research for this topic show three distinct patterns.

| Pattern | Example | What transferred | Grade |
|---|---|---|---|
| Full formal transfer | Clinical and real-world-evidence research using E9(R1) estimands, including target trial emulation | The complete framework, including intercurrent-event strategies | A |
| Partial transfer to online experimentation | Analysis of what A/B test metrics estimate, distinguishing cumulative from windowed metrics | The core insight that the metric definition determines the quantity estimated; exposure timing and stopping rules as estimand-defining choices | B |
| Vocabulary-only transfer | Vendor documentation and marketing-mix guidance referring to "causal estimands" | The word, sometimes without the intercurrent-event discipline | D |

The middle row is the most useful for this program. Ankargren, Frånberg and Schultzberg, writing
from an experimentation platform, show that a cumulative metric — total sales over the whole
experiment — has an estimand that depends on the exposure distribution and on when the experiment
was stopped. Two identical experiments run for different lengths therefore answer different
questions, even though both are reported as "the lift". A windowed metric, fixed in advance at a
defined horizon, has a stable estimand. They also report that for cumulative metrics, statistical
power can *decrease* as more data accumulate, which breaks the intuition that longer experiments
are always better.

That is the same lesson as the promotion example in section 6, arriving from a different
direction: the window is part of the question, and choosing it after seeing the data is choosing
the answer.

### 8.3 Reporting conventions that carry over

| Convention | Origin | What it requires | Transfer to analytics |
|---|---|---|---|
| Pre-registration / analysis plan | Clinical trials, and increasingly social science | Estimand, primary outcome, and analysis fixed before unblinding | Write the estimand and decision rule before the backtest, not before the data exists |
| Target trial emulation reporting | Observational epidemiology | Protocol components stated, including time zero and eligibility | State the trial you would have run, and which component cannot be emulated |
| Sensitivity versus supplementary analysis | ICH E9(R1) | Separate robustness checks from different questions | A per-protocol re-analysis is a different estimand, not a robustness check |
| Negative control outcome | Epidemiology | An outcome that should show no effect if the design is valid | A product unaffected by the promotion, tested with the same design |
| Placebo or sham comparison | Trials | A comparator that differs only in the active ingredient | An unpromoted matched store, or a promotion with no price change |

The negative control is the most underused of these in commercial analytics and the cheapest to
implement. If your promotion design shows a "lift" in a product that was not promoted and could
not have been affected, the design is picking up a trend, not an effect. The test costs one extra
column in a query.

### 8.4 What no standard supplies

No standard defines the following, and each is a decision the analyst must make and defend:

- Which population a commercial estimand should target. The trial population is defined by
  eligibility; a retailer's is defined by commercial strategy.
- What counts as the outcome when revenue, margin, and units point in different directions. The
  worked example in section 6 is exactly this case.
- How to handle interference in a network where promoted stores draw demand from unpromoted ones.
  Clinical trials assume no interference; supply chains rarely satisfy it.
- What loss function is appropriate. Trials have a primary endpoint; businesses have a P&L.

These gaps are not a deficiency of the standards. They are the part of the problem that is
specific to the domain, and they are why the six-slot statement exists.

### 8.5 Governance analogies, stated carefully

Listed companies operate internal controls over financial reporting, and forecast accuracy can
feed revenue guidance. Model risk guidance in banking, such as the Federal Reserve's SR 11-7,
provides a useful template for model documentation and validation. Neither governs the choice of
estimand.

The honest statement is: **these regimes constrain documentation and review, not statistical
definition.** They make it more likely that someone will ask what the model estimates, which is
useful. They do not tell you the answer, and citing them as though they did is a way of borrowing
authority for a choice that remains yours.

## 9. Industry practice and case evidence

### 9.1 What operations management research says about its own causal practice

The operations management literature has spent the last decade auditing its own causal claims,
and the results are a useful calibration for how hard this is.

| Source | Finding | Grade |
|---|---|---|
| Ho, Lim, Reza and Xia (2017), *M&SOM* | Reviews five econometric tools for endogeneity and selection bias, and argues that big data helps mainly through variety and velocity, not volume | A |
| Lu, Ding, Peng and Chuang (2018), *Journal of Operations Management* | Systematic review of 2012–2017 papers; documents recurring identification failures and instrument-quality problems | A |
| Mithas, Chen, Lin and De Oliveira Silveira (2022), *Production and Operations Management* | Distinguishes baseline bias from differential-treatment-effect bias, and shows that different designs address different ones | A |
| Yılmaz, Son, Shang and Arslan (2024), *Journal of Operations Management* | Frames matching and synthetic control explicitly as selection-on-observables designs, and compares them by simulation | A |
| Ketokivi and McIntosh (2017), *Journal of Operations Management* | Argues the endogeneity problem is a design problem, not an estimation problem | A |

The consistent message is that the design carries the identification and the estimator does not
repair it. This is the academic version of the rule in section 5.8: design before estimator.

### 9.2 The promotion-forecast problem, documented

The claim that a promotion's forecast accuracy says nothing about its effect is not a stylistic
preference. It is supported by evidence about how promotion forecasts actually behave.

Fildes, Goodwin and Önkal (2018), in the *International Journal of Forecasting*, report
experimental evidence on how people forecast promotional effects. Forecasters anchor on the
statistical baseline and adjust insufficiently, over-weighting the most recent promotion and
non-diagnostic narrative reasons while neglecting the base rate. In their study a situation with a
50% base-rate uplift was forecast at a median of about 30%. The direction is important: judgmental
adjustment *under*-estimates promotion effects on average, even while the baseline methods used to
measure them tend to *over*-credit them.

| Claim | Evidence | Grade |
|---|---|---|
| Judgmental promotion forecasts anchored on a baseline under-estimate the base-rate uplift | Fildes, Goodwin and Önkal (2018), experimental study | A |
| Baseline methods used to measure incremental sales tend to over-credit promotions | Vendor benchmark study comparing two baseline methods against synthetic ground truth; reports median absolute error of 32.8% and 42.0% for the two methods, both with positive bias | D |
| A counterfactual model can reach about 90% forecast accuracy on non-promotional periods and still not validate the uplift estimate | Practitioner case write-up; the authors state that there is no ground truth in counterfactual forecasting | D |
| Attribution and return-on-ad-spend figures show what happened, not whether the activity caused the result | Industry incrementality guidelines, November 2025 | C |

The first row and the second row point in opposite directions, and both can be true. A human
adjusting a baseline upward under-adjusts; a statistical baseline used as a counterfactual tends to
attribute too much to the promotion. The measurement error runs one way and the judgment error
runs the other, and neither cancels the other.

> **Note:** The vendor figures in rows two and three are graded D because they come from commercial
> suppliers of baseline and measurement tools, reporting benchmarks that favour their framing.
> They are included because the *direction* of the bias they report is consistent with the
> academic evidence and with the arithmetic in section 6, not because their magnitudes should be
> quoted.

### 9.3 The industry evidence hierarchy for incrementality

The November 2025 incrementality guidelines published by the digital advertising trade bodies
rank measurement methods by causal strength, and the ranking transfers to supply chain
measurement almost unchanged:

| Method | Causal strength as graded | Supply chain analogue |
|---|---|---|
| Randomised experiment, holdout, matched market | Strong | Store-level randomised rollout, matched-market test |
| Model-based counterfactual | Strong to moderate, prone to omitted-variable and selection bias | Difference-in-differences, synthetic control |
| Econometric or marketing-mix model | Moderate to weak | Regression on observational history with controls |
| Attribution or last-touch proxy | Weak | "Sales during the promotion" or "orders from the promoted region" |

The guidelines' central sentence — that attribution shows what happened, not whether the activity
caused the result — is the same distinction this topic draws between Questions A and C. That a
trade body found it necessary to write it down in 2025 is evidence of how persistent the confusion
is.

### 9.4 What practitioners actually do

| Observed practice | Frequency and source | Consequence |
|---|---|---|
| Promotion measured against a forecast or plan | Common; the default in most planning suites | Accuracy is reported as effect; value-destroying promotions pass review |
| Before-and-after comparison with no control group | Common in commercial reviews | Trend and seasonality attributed to the intervention |
| A/B test read at a fixed horizon chosen post hoc | Common in online experimentation | The estimand depends on the stopping rule; significant results are inflated |
| Target trial stated before analysis | Rare outside regulated research | Identification failures discovered after publication, if at all |
| Intercurrent events named and handled by a stated strategy | Rare | Non-execution silently dilutes or inflates effects depending on the direction |
| Sensitivity analysis pre-specified | Occasional | Robustness claims made only for results that survived |

The pattern is consistent across the sources: the technical machinery for credible causal
inference is available and taught, and the specification discipline that would make it usable is
not routinely practised. That gap, not the mathematics, is what this topic addresses.

### 9.5 A note on the evidence quality in this section

Four of the five tables above rest on peer-reviewed sources. The vendor benchmarks are graded D
and dated. The industry guidelines are graded C: they are a consensus document from trade bodies
with a commercial interest in measurement being taken seriously, which does not make them wrong
and does make them interested. No quantitative claim in this topic depends on a vendor figure; the
worked example is constructed arithmetic, labelled as such.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Forecast accuracy offered as evidence of effect | "The promotion performed to plan" in a review pack | Forecast and actual are both in the treated world; no counterfactual | Value-destroying interventions approved | Separate the descriptive, predictive, and causal estimates in the pack |
| Estimator named instead of estimand | "We used a DiD" in answer to "what are we estimating" | Method substitutes for question | The design choice silently defines the question | Require the six-slot statement before the method is chosen |
| Unit of assignment ignored | Treatment at store level, inference at basket level | Clustered outcomes treated as independent | Standard errors too small; false positives | State both units; cluster at assignment level |
| Time zero set to first observation | "We used all available history" | Pre-follow-up period is immortal | Treated group looks better by construction | Align time zero to the decision; assess eligibility then |
| Comparator left undefined | "Versus business as usual" | The alternative condition is never specified | The effect is relative to an unknown state | Define the comparator with the same specificity as the treatment |
| Intercurrent events unhandled | Non-execution discussed only in the limitations | The question silently shifts from policy to mechanism | Estimate answers a question nobody asked | Name the events; choose one of the five strategies |
| Metric window chosen after the result | Several horizons reported; the significant one is highlighted | The estimand depends on the window and stopping rule | Inflated effects, unreproducible findings | Pre-specify the window and the stopping rule |
| Scale switched post hoc | Percentages for small effects, absolute for large | Ratio and difference answer different questions | Misleading magnitude | Report absolute and relative together, as pre-specified |
| Confounding described as "controlled for" | A long covariate list, no graph | Adjustment is a claim about the graph, not a count of variables | Collider and mediator bias, with more confidence | Draw the graph; state which variables are confounders |
| Positivity unexamined | Propensity scores near 0 and 1, matched sample much smaller | Some units have no comparable counterpart | Estimate applies to a shrinking, unrepresentative subset | Report overlap; state the population the estimate covers |
| Interference ignored | Promoted and control stores in the same catchment | One unit's outcome depends on another's treatment | Effect overstated, sometimes doubled | Buffer zones; spillover test at distance |
| Sensitivity analysis absent or post hoc | "Results are robust" with no threshold | No specified departure or threshold for concern | Fragile conclusions presented as settled | Pre-specify one sensitivity analysis and its decision threshold |
| Decision rule invented after the estimate | The recommendation matches the point estimate exactly | Threshold chosen to fit | Analysis becomes advocacy | Write the if-then rule before the result |
| Three question types merged in one model | One "demand model" serves finance, planning, and pricing | Different estimands share one validation | Each user inherits the wrong validation | Document separate estimands and separate validation per use |
| Descriptive claim with a causal verb | "The promotion drove 40% growth" | "Drove" implies a counterfactual the analysis does not have | False attribution in reporting | Rewrite with "during the promotion, sales were 40% higher" |

Four deserve a note.

**Forecast accuracy offered as evidence of effect** is the signature failure of commercial
analytics, and it survives because it looks like rigour. Someone quantified something, compared it
with a target, and reported the comparison. The defect is that both sides of the comparison are in
the same world. Section 6.7 gives the arithmetic of a case where the forecast is 97.8% accurate and
the intervention loses money.

**Estimator named instead of estimand** is the failure that this topic exists to prevent, and it
is usually committed by competent analysts. "We ran a difference-in-differences" is an answer
about method. The estimand answer is "the average difference in margin per store over four weeks
between running and not running the promotion, in the 40 stores that ran it". The second sentence
can be wrong; the first cannot.

**Time zero set to first observation** is the most technical failure on the list and the easiest to
fix once named. When an analyst uses all available history and defines the pre-period as everything
before the intervention, units that entered the treated group because of an earlier event are
credited with survival that had nothing to do with the treatment. The fix is to define the
decision point and start follow-up there, discarding the earlier history for identification
purposes even though it remains useful for describing the population.

**Three question types merged in one model** is the most organisational failure. It happens when a
single "demand model" is asked to support financial planning, operational replenishment, and
promotional pricing. Each of those uses requires a different estimand, and the model's validation
is usually built for one of them. The other two inherit a validation that does not apply, and the
error surfaces as an argument about whose numbers are right.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The question type can be classified unambiguously | The whole routing depends on it | A question that mixes description and causation ("how much did we gain") | Split into two questions and answer each separately |
| The six slots can be filled before seeing outcome data | Pre-specification is what makes the estimate credible | Stakeholders cannot agree on the population or the outcome | If a slot cannot be filled, the question is not yet a question |
| The decision is stable over the analysis window | The estimand is tied to a decision | Strategy changes, reorganisation, budget cycle moves | Check the decision owner and date before starting |
| The population is identifiable in the data | The estimate must refer to real units | Eligibility criteria are not recorded | Count eligible units; compare with the operational total |
| The comparator is a real condition | A contrast needs two states | "Do nothing" is not observed anywhere | Find units that experienced the comparator |
| For predictive questions, the deployment distribution resembles training | Otherwise the estimate does not transfer | Promotions, new items, structural breaks, channel shift | Backtest on the most recent period; monitor feature drift |
| For causal questions, exchangeability holds conditional on measured covariates | Identification of the counterfactual | Unmeasured confounders, selection on unobservables | Negative control; sensitivity analysis; argue from design |
| Positivity holds in the target population | Every unit must have a possible counterpart | Deterministic assignment rules, extreme propensity scores | Overlap plots; trimmed-sample comparison |
| Consistency holds: the intervention is one thing | Otherwise the potential outcome is undefined | Several versions of "the promotion" (different depths, durations) | Audit treatment definitions across units and time |
| No interference between units | SUTVA underlies the standard contrast | Shared demand, shared supply, allocation across a network | Spillover test at distance; buffer design |
| The outcome measurement is stable over the window | Otherwise the change includes a definitional component | System change, restatement, new data source | Compare data dictionary versions; reconcile a sample |
| The loss function is known well enough to choose a decision | Otherwise the decision rule cannot be written | Cost of stock-out, margin, and goodwill not quantified | Bound the costs; test the decision at both bounds |

Two limits of the whole framework are worth stating.

**The framework disciplines the question; it does not generate the answer.** A perfectly specified
estimand with an unidentified design produces a precise statement of an unanswerable question.
That is an improvement over an imprecise statement of an unanswerable question, but it is not an
answer. The value is that the failure becomes visible early, when it is cheap.

**The three-way split is a simplification.** Real questions sit on a spectrum. "What is the effect
of our forecast on our inventory?" is causal. "What inventory will we hold next quarter?" is
predictive but depends on a causal model of how orders respond to stock-outs. "How much service
did we deliver?" is descriptive until someone asks why. The taxonomy is a routing device, not an
ontology, and the useful question is always "which of these is being asked right now", not "which
category does this project belong to".

## 12. Exercises

### Exercise 1 — Compute the causal estimate and the margin consequence (calculation, 50 minutes)

A retailer ran a four-week display test in 30 stores with 40 control stores.

| Group | Units, weeks 1–4 before | Units, weeks 5–8 during |
|---|---:|---:|
| Treated (30 stores) | 9,000 | 11,700 |
| Control (40 stores, scaled to 30) | 8,700 | 9,150 |

Full-price margin is €4.00 per unit; promotional margin is €3.10 per unit. The promotion applied to
all units sold in the treated stores during the window.

**Deliverable:** the naive change, the control change, the difference-in-differences estimate, the
counterfactual volume, the percentage lift, the margin effect, and the break-even lift.
**Acceptance criteria:** every intermediate number shown; the naive overstatement expressed as a
percentage of the causal estimate; the margin decomposed into gained and forgone components; a
one-sentence decision recommendation.
**Time:** 50 minutes.

### Exercise 2 — Classify six requests (analysis, 30 minutes)

Classify each as descriptive, predictive, causal, or predictive-plus-loss, and state the
counterfactual each does or does not require.

1. "What was our fill rate in Q2?"
2. "How many pallets will we need in December?"
3. "Did the new routing rule reduce cost per stop?"
4. "Which carrier should we tender this lane to?"
5. "Why did OTIF drop after the system migration?"
6. "What would our service level be at 14 days of cover instead of 10?"

**Deliverable:** a six-row table with type, counterfactual required (yes/no), and one sentence of
justification.
**Acceptance criteria:** at least one item is predictive-plus-loss rather than causal, with the
reason stated; "why" is classified as causal with a note on mechanism.
**Time:** 30 minutes.

### Exercise 3 — Write the estimand for a supplier question (design, 40 minutes)

A procurement team asks whether a new allocation rule reduced late deliveries. The rule assigns
volume by a two-week frozen window instead of weekly reallocation, and applies to 180 tier-1 lanes.

**Deliverable:** the six-slot estimand, the ICH five-attribute mapping, and the intercurrent-event
strategy with one sentence of justification.
**Acceptance criteria:** population and unit distinguished; time zero tied to the contract start;
comparator specified as a condition; the strategy choice defended against at least one alternative.
**Time:** 40 minutes.

### Exercise 4 — Run the target trial (design, 45 minutes)

For the allocation-rule question in Exercise 3, write the target trial protocol and identify the
component that cannot be emulated.

**Deliverable:** the seven protocol components, the emulation failure, and the diagnostic that
would detect it.
**Acceptance criteria:** eligibility measurable or explicitly flagged; assignment level stated;
interference considered; the diagnostic is specific enough to run.
**Time:** 45 minutes.

### Exercise 5 — Find the confusion in a memo (critique, 40 minutes)

Write, or obtain, a one-page internal memo that claims an intervention worked. Identify every
sentence that borrows causal credibility from a descriptive or predictive result.

**Deliverable:** an annotated memo with each problematic sentence rewritten in its correct
question type, plus a list of the estimands the memo would need to support its conclusion.
**Acceptance criteria:** at least four sentences annotated; rewrites preserve the true claim;
the missing estimands are named in six-slot form.
**Time:** 40 minutes.

### Exercise 6 — Audit a forecast-accuracy-means-success claim (critique, 35 minutes)

A team reports 96% forecast accuracy during a promotion and concludes the promotion was
successful. Reconstruct what the 96% measures.

**Deliverable:** the numerator and denominator of the accuracy figure, the two worlds compared,
and a corrected claim.
**Acceptance criteria:** the absence of a counterfactual is stated explicitly; one alternative
explanation for high accuracy is given (the promotion was in the feature set); the corrected claim
is a descriptive or predictive statement, not a causal one.
**Time:** 35 minutes.

### Exercise 7 — Map the intercurrent events (analysis, 40 minutes)

For a promotion programme in 200 stores, list the intercurrent events that could occur between
assignment and outcome measurement, and write the effect under each of the five strategies.

**Deliverable:** an event table and a strategy table, with the direction of bias for each strategy
against the treatment-policy estimate.
**Acceptance criteria:** at least five events; each strategy described in one sentence; the
direction of bias stated, not just the existence.
**Time:** 40 minutes.

### Exercise 8 — Pre-specify the sensitivity analysis and decision rule (design, 45 minutes)

For Exercise 1, write the sensitivity analysis and the decision rule before looking at the answer.

**Deliverable:** one sensitivity analysis with a named departure and a numeric threshold for
concern, and an if-then decision rule covering rollout, retest, and stop.
**Acceptance criteria:** the threshold is a magnitude, not a verdict; the decision rule covers all
three branches; both are written as though the result were unknown.
**Time:** 45 minutes.

### Exercise 9 — Apply the mismatch checklist (application, 45 minutes)

Take a real or realistic project proposal and run the twelve-item checklist from section 5.10.

**Deliverable:** the completed checklist with pass/fail per item, the two most serious failures,
and a repaired estimand statement.
**Acceptance criteria:** every item answered, not skipped; the repairs are specific; at least one
item is failed and the consequence explained.
**Time:** 45 minutes.

### Exercise 10 — Build the question-type router for your own work (design, 60 minutes)

Take the last three analytics requests you received. Classify them, write the estimand each
actually required, and note where the delivered work diverged.

**Deliverable:** a three-row table with the request as stated, the estimand required, and the gap.
**Acceptance criteria:** at least one request turns out to have been misclassified; the gap is
described in terms of the six slots; the corrected estimand is computable.
**Time:** 60 minutes.

## 13. Self-check answers

### 13.1 Exercise 1 arithmetic

```text
naive change, treated      = 11,700 - 9,000          = +2,700 units
change, control            = 9,150 - 8,700           = +450 units
difference-in-differences  = 2,700 - 450             = +2,250 units
counterfactual volume      = 9,000 + 450             = 9,450 units
causal lift                = 2,250 / 9,450           = 23.81%
naive overstatement        = 450 / 2,250             = 20.00%
```

The naive comparison overstates the effect by 20%. The counterfactual volume is 9,450 units, not
9,000, because the control group grew by 5.2% over the same period and the treated stores would
have shared some of that growth.

```text
margin, actual             = 11,700 x EUR 3.10       = EUR 36,270
margin, counterfactual     =  9,450 x EUR 4.00       = EUR 37,800
margin effect              = 36,270 - 37,800         = EUR -1,530

decomposition:
  margin gained on incremental units = 2,250 x 3.10 = EUR +6,975
  margin forgone on baseline units   = 9,450 x 0.90 = EUR -8,505
  net                                                = EUR -1,530

break-even incremental units = 8,505 / 3.10           = 2,743.5 units
break-even lift              = 2,743.5 / 9,450        = 29.03%
observed lift                = 23.81%
shortfall                    = 5.22 percentage points
```

The recommendation is to stop or redesign: the display test raised volume by 23.8% and needed
29.0% to pay for the discount. Three correct summary numbers point in three directions — units up
30%, naive margin up, causal margin down €1,530 — and only the third answers the decision
question.

Two wrong answers are common. Using the post-period actual as the baseline for the percentage
(2,250 / 11,700 = 19.2%) understates the lift, because the denominator should be the counterfactual,
not the observed. Computing the margin effect as incremental units times the promo margin alone
(€6,975) omits the discount given away on units that would have sold at full price, which is the
larger term in this example.

### 13.2 Exercise 2 answers

| # | Request | Type | Counterfactual | Justification |
|---:|---|---|---|---|
| 1 | Fill rate in Q2 | Descriptive | No | A fact about a defined period and population |
| 2 | Pallets needed in December | Predictive | No | A statement about unobserved future demand |
| 3 | Did the routing rule reduce cost per stop | Causal | Yes | Compares the rule with the alternative that would have applied |
| 4 | Which carrier to tender | Predictive plus loss | No | The choice selects among outcomes that will occur anyway; a cost model is required |
| 5 | Why OTIF dropped after migration | Causal, mechanism | Yes | Attribution among candidate causes; needs a graph and often mediation |
| 6 | Service level at 14 days of cover | Predictive, conditional on a policy | No, if the policy is evaluated as a plan | It asks what would be observed under a stated inventory policy, not what an intervention would change |

Item 6 is the one that divides people. If the question is "what service would we deliver if we set
cover to 14 days", and setting cover does not change supplier behaviour or demand, a predictive
model of demand and lead time answers it. If the change would alter supplier behaviour — for
example by changing order frequency enough to affect fill — it becomes causal. The test from
section 3.2 decides: does the action change the outcome, or select among outcomes.

### 13.3 Exercise 3 rubric

A strong answer fills all six slots with specifics: 180 named lanes, tier-1 suppliers, spend above
a stated threshold; lane-month as the unit with supplier as the assignment unit; OTIF as a
line-level proportion rolled up; time zero at contract start with a 12-month horizon; the rule
described with its frozen window and its comparator described as the existing weekly rule; and a
difference in proportions in percentage points. It maps to the five ICH attributes and notes that
the six-slot "time" and "unit" are absorbed into population and variable in the clinical framing.
It chooses a principal-stratum strategy because the decision concerns whether the rule works when
followed, and it notes that a treatment-policy analysis would answer a different and also
legitimate question.

A weak answer says "compare OTIF before and after the rule change" and calls that the estimand. It
has an estimator and a period, and no population, comparator, or unit.

### 13.4 Exercise 4 rubric

A strong protocol states eligibility as measurable criteria, two strategies with dose and
duration, assignment at supplier level, time zero at contract start, a 12-month follow-up, OTIF
measured at line level, and a difference in proportions. It then identifies the emulation failure:
assignment was not random, and lanes were moved to the new rule partly because their suppliers
were already improving, which violates exchangeability. The diagnostic it proposes is a
pre-trend comparison over the 12 months before the change, plus a negative-control outcome such
as invoice accuracy, which the rule should not affect.

A weak protocol lists the components but concludes that the analysis is "observational, so causal
claims are not possible". That is the failure the target trial exists to prevent. The correct
conclusion is that a causal claim is possible under stated assumptions, and here is the assumption
that is doing the work.

### 13.5 Exercise 5 rubric

A strong annotation distinguishes three sentence types: descriptive sentences that are true but
carry no causal weight ("sales rose 40% during the promotion"), predictive sentences presented as
evidence ("the model predicted the lift, so the promotion worked"), and causal sentences with no
identification ("the promotion drove growth"). For each it supplies the missing estimand in
six-slot form. It notes that the memo's conclusion may still be correct — the exercise is about
what the evidence supports, not about which conclusion is right.

A weak annotation corrects the grammar of the claims without naming the missing estimand, or
declares the whole memo wrong without separating the parts that are true.

### 13.6 Exercise 6 rubric

A strong answer reconstructs the accuracy figure as `1 − |actual − forecast| / actual`, notes that
both terms are measured in the world where the promotion ran, and states that no counterfactual
appears anywhere in the calculation. It offers the leakage explanation: the promotion was a
feature in the forecasting model, so the model was told the promotion would happen and learned its
typical effect; high accuracy shows the model encoded the effect, not that the effect was caused.
The corrected claim is predictive — "we can plan the next promotion to within about 4%" — and the
memo should say so.

A weak answer says the accuracy figure is "biased" without saying what it is biased relative to,
or proposes a better forecasting model, which does not address the question.

### 13.7 Exercise 7 rubric

A strong event table includes at least: display not executed; out of stock during the promotion;
price overridden at the register; competitor promotion in the same weeks; store remodel; and
partial-week execution. The strategy table states the effect under each of the five strategies and
the direction of bias relative to the treatment-policy estimate: a hypothetical strategy that
assumes perfect execution will exceed the treatment-policy estimate when non-execution dilutes the
effect; a while-on-treatment strategy truncates the window and usually reduces the estimate; a
composite strategy that counts non-execution as failure will reduce it further; a principal-stratum
estimate applies to compliers and may be larger or smaller depending on whether compliers are
high- or low-response stores.

A weak answer lists events without strategies, or treats non-execution as missing data to be
imputed, which is a different and usually wrong move: the store did not fail to report, it did not
receive the treatment.

### 13.8 Exercise 8 rubric

A strong sensitivity analysis names one departure and quantifies it: "if promoted stores would
have grown 3 percentage points faster than control for reasons unrelated to the display, the
margin effect moves from −€1,530 to −€1,100; the conclusion reverses only if the differential
growth exceeds 8.7 percentage points, which is larger than any pre-period divergence observed."
The threshold is a magnitude and it is compared with something observable. The decision rule is
written as if-then, covers rollout, retest, and stop, and does not mention the actual result.

A weak answer says "we checked robustness by trying another control group" without stating what
would count as a failure, or writes a decision rule whose thresholds match the observed estimate.

## 14. Interview questions

**1. "What is an estimand, and how is it different from a metric?"**

A strong answer defines it as the precise quantity to be estimated, fixed before analysis, and
enumerates the slots: population, unit, outcome, time, intervention, comparison. It distinguishes
estimand from estimator and estimate, and notes that a metric is usually a measurement convention
without a comparison. A weak answer says "it's the thing you're trying to measure", which is true
and gives no purchase on any of the decisions the concept exists to settle.

**2. "A stakeholder asks whether the promotion worked. What do you ask first?"**

A strong answer asks what decision depends on the answer, then what the comparator is, then over
what window and in which stores. It notes that "worked" is ambiguous between "sold more", "sold
more than would have happened otherwise", and "made money", and that the three require different
estimands. A weak answer asks for the sales data.

**3. "Can a model be highly accurate and causally wrong?"**

A strong answer says yes, and gives the mechanism: accuracy is a statement about prediction on the
deployment distribution, and a model can predict well from proxies that are consequences of the
treatment, or from a feature that encodes the treatment itself. It gives the promotion example,
where the forecast includes the promotion as an input. A weak answer says "correlation is not
causation" without saying what accuracy does and does not measure.

**4. "What is the target trial, and why would you run one you cannot execute?"**

A strong answer describes the seven protocol components, explains that the exercise exposes
whether the causal question is well posed and which component cannot be emulated, and notes that
the emulation failure becomes a stated assumption with a diagnostic rather than a reason to
abandon the analysis. A weak answer describes it as a thought experiment with no practical
consequence.

**5. "What are intercurrent events, and give me a supply chain example."**

A strong answer defines them as post-time-zero events that change what the outcome means, gives
non-execution of a promotion or a stock-out during the promotion as examples, and names at least
three of the five handling strategies with their different estimates. A weak answer calls them
missing data.

**6. "Why does the unit of assignment matter if I am measuring baskets?"**

A strong answer explains clustering: outcomes within a store are correlated, so treating baskets
as independent understates standard errors, often by a large factor, and inflates significance. It
notes that the estimand should state both units and the analysis should cluster at the assignment
level. A weak answer says "we aggregate to store level", which may be right but does not explain
why it is necessary.

**7. "When is a before-and-after comparison legitimate?"**

A strong answer says it is a legitimate descriptive statement about change over time and an
illegitimate causal one unless nothing else changed, which is rare. It notes the specific threats:
seasonality, trend, concurrent events, and mean reversion. A weak answer says "never", which
overstates: the descriptive use is common and correct.

**8. "How would you tell whether a supplier scorecard is a causal tool?"**

A strong answer asks whether the scorecard is being used to predict (which supplier will be late)
or to intervene (whether changing the supplier changes lateness). It notes that a score built on
observed performance under the current allocation rule may not predict what happens under a
different rule, because the rule itself shaped the data. A weak answer assumes that a good
predictor is a good lever.

**9. "What does it mean for an analysis to be pre-specified, and why does it matter?"**

A strong answer says the estimand, the primary comparison, the window, and the decision rule are
fixed before outcome data are examined, and explains that this prevents selection on results —
choosing the metric, window, or subgroup that produces the desired answer. It distinguishes
pre-specification from having discussed the plan internally. A weak answer treats it as a
bureaucratic step.

**10. "What would make you abandon a causal analysis?"**

A strong answer names positivity failure, an intervention that cannot be defined consistently, and
interference that cannot be bounded. It notes that confounding alone is not a reason to abandon,
because it can be argued about and its sensitivity measured, whereas a positivity violation means
there is no comparable group at all. A weak answer says "when the data are not experimental".

**11. "How do you decide between reporting a difference and a ratio?"**

A strong answer says the choice depends on the decision and the audience: a difference is what an
operational plan needs, a ratio is what makes products comparable, and a ratio on a small base can
look large while being commercially trivial. It recommends reporting both when the decision
depends on both, and pre-specifying which is primary. A weak answer picks one by habit.

**12. "Your model says the promotion added 4,000 units. What else do you need before recommending a rollout?"**

A strong answer asks for the margin effect, the break-even threshold, the interval, the
generalisation population, and the cost of scaling. It notes that 4,000 units of volume with a 20%
discount can be margin-negative, so the volume estimate alone does not support a rollout. A weak
answer treats the volume estimate as the answer.

## 15. Key terms

- **Estimand** — the precise quantity to be estimated, stated before analysis.
- **Estimator** — the procedure that produces an estimate of the estimand.
- **Estimate** — the number an estimator produces on a dataset.
- **Descriptive question** — asks for a fact about a defined population and period, with no counterfactual.
- **Predictive question** — asks about an unobserved observation, conditional on stated information.
- **Causal question** — asks for a contrast between potential outcomes under well-defined interventions.
- **Counterfactual** — the outcome that would have occurred under a condition not received.
- **Potential outcome** — the outcome for a unit under a specified treatment condition.
- **Intervention** — a fully specified action, including dose, timing, duration, and delivery.
- **Comparator** — the alternative condition, specified as precisely as the intervention.
- **Unit of analysis** — the entity whose outcome is measured.
- **Unit of assignment** — the entity to which the intervention is applied.
- **Time zero** — the moment eligibility is assessed and follow-up begins.
- **Population-level summary** — the summary measure that makes two conditions comparable.
- **Contrast** — the comparison between summary values under two conditions.
- **Intercurrent event** — a post-time-zero event that changes the interpretation of the outcome.
- **Treatment policy strategy** — analyse outcomes regardless of adherence to the assigned condition.
- **Hypothetical strategy** — estimate the outcome had the intercurrent event not occurred.
- **Composite strategy** — incorporate the intercurrent event into the outcome definition.
- **Principal stratum strategy** — target the subpopulation in which the event would or would not occur.
- **While-on-treatment strategy** — use the outcome observed up to the event.
- **Target trial** — the hypothetical randomised trial an observational analysis emulates.
- **Exchangeability** — the condition that treated and untreated units would have had the same outcomes absent treatment, within strata.
- **Positivity** — every covariate pattern has a nonzero probability of each condition.
- **Consistency** — the observed outcome equals the potential outcome under the treatment received.
- **SUTVA** — no interference between units and no hidden versions of the treatment.
- **Interference** — one unit's outcome depends on another unit's treatment.
- **Confounder** — a variable affecting both treatment and outcome.
- **Collider** — a variable affected by both treatment and outcome; conditioning on it induces bias.
- **Mediator** — a variable on the causal path from treatment to outcome.
- **Incrementality** — the part of an outcome that would not have occurred without the intervention.
- **Baseline** — the level against which an outcome is compared; a counterfactual only if justified.
- **Leakage** — use of information at training time that is unavailable at prediction time.
- **Decision rule** — the pre-specified mapping from an estimate to an action.
- **Sensitivity analysis** — an analysis probing robustness to departures from assumptions, targeting the same estimand.

## 16. Sources

**Peer-reviewed and academic — Grade A**

- Shmueli, G. (2010). "To Explain or to Predict?" *Statistical Science*, 25(3), 289–310.
  DOI: 10.1214/10-STS330. Explanatory, predictive, and descriptive modelling as distinct goals;
  an underspecified model can have lower expected prediction error than the true one.
- Holland, P. W. (1986). "Statistics and Causal Inference." *JASA*, 81(396). The fundamental
  problem of causal inference; the potential-outcomes framing.
- Rubin, D. B. (1986). "Which Ifs Have Causal Answers?" *JASA*. SUTVA, and when a causal question
  is well posed.
- Hernán, M. A. and Robins, J. M. *Causal Inference: What If*. Chapman and Hall/CRC; free PDF from
  the authors. Chapter 3.6 on the target trial; Chapter 22 on emulation, time zero, and sustained
  strategies.
- Hernán, M. A. and Robins, J. M. (2016). "Using Big Data to Emulate a Target Trial When a
  Randomized Trial Is Not Available." *American Journal of Epidemiology*, 183(8), 758–764.
- Hernán, M. A., Dahabreh, I. J., Dickerman, B. A. and Swanson, S. A. (2025). "The Target Trial
  Framework for Causal Inference From Observational Data." *Annals of Internal Medicine*, 178(3),
  402–407. DOI: 10.7326/ANNALS-24-01871. Design problems versus data limitations.
- Kahan, B. C. et al. (2024). "The estimands framework: a primer on the ICH E9(R1) addendum."
  *BMJ*. The five attributes and five intercurrent-event strategies.
- Fildes, R., Goodwin, P. and Önkal, D. (2018). "Use and misuse of information in supply chain
  forecasting of promotion effects." *International Journal of Forecasting*.
  DOI: 10.1016/j.ijforecast.2017.12.006. Judgmental promotion forecasts anchor on the baseline and
  under-adjust.
- Mithas, S., Chen, Z.-L., Lin, M. and De Oliveira Silveira, A. (2022). "On the causality and
  plausibility of treatment effects in operations management research." *Production and
  Operations Management*, 31(12), 4558–4571. DOI: 10.1111/poms.13863.
- Ho, T.-H., Lim, N., Reza, S. and Xia, X. (2017). "OM Forum—Causal Inference Models in Operations
  Management." *M&SOM*, 19(4), 509–525. DOI: 10.1287/msom.2017.0659.
- Lu, G., Ding, X. D., Peng, D. X. and Chuang, H. H.-C. (2018). "Addressing endogeneity in
  operations management research." *Journal of Operations Management*, 64(1), 53–64.
  DOI: 10.1016/j.jom.2018.10.001.
- Yılmaz, E., Son, J., Shang, G. and Arslan, B. (2024). "Causal inference under selection on
  observables in operations management research." *Journal of Operations Management*, 70(5),
  831–859. DOI: 10.1002/joom.1318.
- Ketokivi, M. and McIntosh, C. N. (2017). "Addressing the endogeneity dilemma in operations
  management research." *Journal of Operations Management*, 52(1), 1–14.
- Rudin, C. (2019). "Stop explaining black box machine learning models for high stakes decisions."
  *Nature Machine Intelligence*. DOI: 10.1038/s42256-019-0048-x.
- Covert, I., Lundberg, S. and Lee, S.-I. (2021). "Explaining by Removing: A Unified Framework for
  Model Explanation." *Journal of Machine Learning Research*; arXiv:2011.14878. Removal-based
  explanations simulate interventions on features, which are not interventions on the world.

**Standards and regulatory — Grade B**

- ICH Harmonised Guideline E9(R1), *Addendum on Estimands and Sensitivity Analysis in Clinical
  Trials*. Adopted 20 November 2019; EMA Step 5 effective 30 July 2020.
- U.S. Food and Drug Administration, *E9(R1) Statistical Principles for Clinical Trials: Addendum:
  Estimands and Sensitivity Analysis in Clinical Trials*, guidance, May 2021.
- ICH E9, *Statistical Principles for Clinical Trials*, 1998. Historical context for
  intention-to-treat.
- CONSORT 2010 statement and extensions. Reporting of assigned versus analysed populations.
- Federal Reserve SR 11-7 (2011), *Supervisory Guidance on Model Risk Management*. Cited by analogy
  for documentation only.

**Preprints and working papers — Grade B**

- Ankargren, S., Frånberg, M. and Schultzberg, M. "It's About Time: What A/B Test Metrics Estimate."
  arXiv:2411.06150. Cumulative versus windowed metrics; estimands that depend on exposure
  distribution and stopping time; power that can fall as data accumulate. Preprint.
- Bojinov, I. and Shephard, N. "Time Series Experiments and Causal Estimands." Working paper,
  Harvard Business School faculty page.

**Industry and consensus — Grade C, with one vendor item at D**

- IAB and IAB Europe, *Guidelines for Incremental Measurement in Commerce Media*, November 2025.
  Evidence hierarchy for incrementality; attribution shows what happened rather than what was
  caused.
- Google Meridian documentation, "Causal estimands and estimation." Vendor documentation of
  incremental-outcome and ROI estimands for marketing mix modelling. Grade D.

**Vendor and practitioner — Grade D**

- Lift Math, "Accuracy." Vendor benchmark of two baseline methods against synthetic ground truth;
  reports median absolute error of 32.8% and 42.0%, both with positive bias. Dated 2026.
- Graphite Note, "How to Measure Whether a Promotion Actually Worked." Distinguishes observed from
  incremental lift.
- Artefact, "Forecasting something that never happened." Reports about 90% forecast accuracy on
  non-promotional periods and states that there is no ground truth in counterfactual forecasting.

**Press and anecdote — Grade E**

- No press summary or anecdote supports any quantitative claim in this topic. Figures without an
  academic or official source are constructed arithmetic in the worked example.

**Note on citations.** Volume and page numbers appear where the sources consulted confirmed them
and are omitted where they did not. The addendum's paragraph numbering is not reproduced, because
the versions consulted differed in their internal references. Vendor figures in section 9.2 are
reported as the vendors state them, are not independently verified, and are used only for the
direction of bias they describe.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Next topic, written: `02_probability-bootstrap.md` (planned)
- Later in this module: `06_dags.md` (planned) — the graph machinery behind the confounder,
  collider, and mediator distinctions in section 4
- Later in this module: `07_experiments-quasi.md` (planned) — design choice once the estimand is
  fixed
- Later in this module: `10_sensitivity-omitted-bias.md` (planned) — the quantitative version of
  the sensitivity analysis sketched in section 5.9
- Measurement background: [`../01A_foundations_units_time_assumptions/09_forecast-error-metrics.md`](../01A_foundations_units_time_assumptions/09_forecast-error-metrics.md)
  — what forecast accuracy does and does not measure, which section 6.7 depends on
- Experiment design and pilot evidence: [`../05_experimentation_pilots_evidence/_index.md`](../05_experimentation_pilots_evidence/_index.md)
- Survey file: [`../../../course/04_statistics_causal_inference.md`](../../../course/04_statistics_causal_inference.md)
- Return to the foundations map: [`../_index.md`](../_index.md)

