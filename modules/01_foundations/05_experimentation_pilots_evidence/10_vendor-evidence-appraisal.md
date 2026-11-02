# Vendor and Analyst Evidence Appraisal — sample, comparator, endpoint, duration, and conflict of interest

> Module 05 — Experimentation, Pilots, and Evidence
> Topic 10 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 05 topics 01–08; Module 02 topic 08; Module 28
> Version 2026.5

> **Note:** Topic 09 of this module covers marketing mix modelling and incrementality. It is a
> sibling, not a prerequisite: nothing here depends on it, and this file can be read without it.

---

## 1. Why this topic exists

A steering committee has forty minutes. A vendor's account director is on the call, sharing a deck.
Slide nine says that customers reduced excess inventory by 23% in ninety days, and that the average
return on investment (ROI) was 4.1 times. Six customer logos sit along the bottom. Someone asks
whether the number is real. The room looks at you, because you are the analyst.

You have eight minutes and no experiment to design. The evidence already exists, packaged, and your
job is not to produce it but to appraise it. That is a different skill from the eight topics before
this one, and it is the skill you will use most often. An analyst in a supply chain organisation
will design perhaps three experiments a year. The same analyst will be shown perhaps forty vendor
claims, six analyst reports, and a dozen internal pilot write-ups in the same period.

**Three asymmetries define the situation.**

*The repetition asymmetry.* The vendor has run this meeting hundreds of times and has refined the
deck against objections. You have seen this category perhaps twice. The vendor knows which questions
you will ask and has an answer prepared for each. Your advantage is not experience; it is method.

*The selection asymmetry.* The vendor chooses what to show you. You choose only whether to believe
it. Everything in the deck passed through at least three gates before it reached slide nine, and the
gates are not disclosed. Recovering them is most of the appraisal.

*The consequence asymmetry.* If you say yes and the claim is false, the cost lands on your balance
sheet; if you say no and the claim was true, it lands on the vendor's pipeline. The vendor is paid to
accept a higher false-positive rate than you are, which is why its confidence is not evidence.

**This topic sits last in the module for a reason.** Topics 01 to 08 teach you to produce evidence.
This topic teaches you to consume evidence produced by someone who did none of those things, or who
did them and does not have to tell you. The chain closes here: claim, estimand, design, power,
interference, sequential discipline, and appraisal of what you are told.

**The five artefacts you will appraise.** They differ in packaging and not much else.

| Artefact | Typical length | Who produced it | What it usually is |
|---|---|---|---|
| Vendor case study | 2–6 pages | Vendor marketing, sometimes with customer sign-off | A selected pre/post story with no comparator |
| Vendor ROI model or calculator | Spreadsheet or web form | Vendor value engineering | A projection from assumed inputs |
| Analyst report (Magic Quadrant, Wave, MarketScape) | 20–60 pages | Analyst firm, sometimes commissioned | A weighted opinion about vendors, not about your use case |
| Maturity model assessment | 5–40 pages | Analyst firm or consultancy | A rubric score on self-reported process adoption |
| Conference talk or pilot write-up | 20–40 minutes | A practitioner | A narrative with selected numbers and no control |

**The claim of this topic.** Appraisal is not cynicism. Most vendor claims are true statements about
a selected sample, computed correctly against a comparison that was never made. The work is to
recover the sample, name the missing comparison, and state what the number can and cannot support
for your operation. An analyst who can do that in a meeting, in ten minutes, with a scoring sheet and
two pieces of arithmetic, is worth more to a procurement decision than an analyst who can design a
perfect experiment nobody commissioned.

**Why the prior should be sceptical, and not because vendors lie.** Large operational improvements
are rare: most process and technology changes produce effects smaller than their business cases
assume, and many produce none. That is a statement about the distribution of operational change, not
about any particular vendor, and it means the correct starting point for an uncontrolled claim is
"probably smaller than stated", with the burden of proof on the claim.

## 2. Learning outcomes

By the end of this topic you can:

1. **Restate** a vendor claim as a testable sentence with a number, unit, denominator, and period,
   and **identify** whether it is an assertion, a projection, a pre/post measurement, or a
   controlled comparison.
2. **Apply** the six questions — sample, comparator, endpoint, duration, funding, reproducibility —
   to a case study, an analyst report, or a conference talk, and **record** an answer for each.
3. **Compute** the base-rate implication of a published result given a stated publication gate, and
   **explain** why a true average effect of zero can produce a published average of twenty per cent.
4. **Compute** the regression-to-the-mean expectation for a cohort selected on a bad baseline, and
   **state** how much of an observed improvement survives it.
5. **Distinguish** a measured outcome from a modelled projection, and **trace** an ROI to its inputs.
6. **Explain** what a Magic Quadrant position, a maturity level, and a hype-cycle phase actually
   measure, and **state** what they do not measure.
7. **Score** an evidence pack with a ten-item appraisal sheet in under fifteen minutes and
   **defend** the score in a meeting.
8. **Write** a procurement or investment memo with evidence and assertion in separate columns, and
   **state** the evidence that would change the recommendation.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold three pictures: the funnel that produced the number, the gate the six questions form, and the
ladder that tells you what kind of number you are holding.

### 3.1 The funnel

Every published customer result is the survivor of a sequence of selections. Draw the sequence and
the number stops being mysterious.

```text
   all deployments in the installed base                      180
        |
        |  gate 1 — the outcome was measured at all
        v
   deployments with a measured outcome                       ~120
        |
        |  gate 2 — the measured outcome was favourable
        v
   deployments with a result above the publication threshold  ~51
        |
        |  gate 3 — the customer agreed to be named
        v
   named reference customers                                   ~6
        |
        |  gate 4 — the vendor chose to publish this one
        v
   case studies in the deck                                    ~4
        |
        |  gate 5 — the account team chose to cite this one
        v
   the number on slide nine                                     1
```

The counts are illustrative and come from section 6. The shape is the claim: **five gates, each
selecting on the outcome or on something correlated with it.** Gate 3 is the one people forget. A
customer who had a bad experience does not agree to be named, so willingness is itself an
outcome-correlated selection, even when the vendor is scrupulous about not hiding failures.

### 3.2 The six questions

The questions are a gate, and the gate runs in a fixed order — sample, comparator, endpoint,
duration, funding, reproducibility — because earlier questions make later ones interpretable.

```text
   claim -> SAMPLE -> COMPARATOR -> ENDPOINT -> DURATION
         -> FUNDING -> REPRODUCIBILITY -> verdict

   sample           who, how many, out of what, selected how
   comparator       what would have happened otherwise
   endpoint         what was measured, defined by whom, pre-specified?
   duration         how long, what lag, does it persist?
   funding          who paid, who gains, what is disclosed
   reproducibility  could a third party rerun it from disclosed inputs?
```

Each question can end the appraisal on its own. A claim with no denominator cannot be interpreted
regardless of how good the rest is. A claim with an activity endpoint cannot support an outcome
decision regardless of its statistics. The order matters because a favourable answer to a later
question does not repair a failure on an earlier one. Good reproducibility of a pre/post study is
still a pre/post study.

### 3.3 The ladder

Not all numbers are the same kind of object. Place the claim on this ladder before you argue about
whether it is right.

```text
   5   replicated across independent sites, pre-registered,
       with a concurrent control ................ admit as a general claim
   4   one controlled concurrent comparison,
       endpoint pre-specified .................... admit for your context,
                                                   subject to local validation
   3   measured pre/post with a baseline
       constructed after the fact ................ directional only
   2   modelled projection from measured inputs .. a hypothesis with arithmetic
   1   asserted benefit, arithmetic not shown .... marketing
   0   anecdote .................................. a lead, not evidence
```

Three properties follow, and they are the ones to remember.

**Property 1 — the claim is the end of a selection process, not a random draw.** The number you are
shown is not a sample estimate of the vendor's average effect. It is the maximum, or near it, of a
distribution the vendor has seen and you have not. Treating it as an average is the single most
common appraisal error, and it is an error of interpretation rather than of arithmetic.

**Property 2 — every one of these numbers is a comparison, and the comparison is usually missing.**
"Reduced by 23%" means "lower than something". The something is either a concurrent control, a
historical baseline, a modelled counterfactual, or an implicit assumption that nothing else changed.
Ask which, and if the answer is the fourth, you have found the weakness.

**Property 3 — the denominator is the fact most often withheld and the one that determines the
interpretation.** Six out of six is not the same as six out of one hundred and eighty. The numerator
is on the slide because it is flattering. The denominator is in an appendix, or nowhere.

> **Note:** A case study is an existence proof. It answers "is this possible?" Procurement usually
> needs "is this likely, here?" Those are different questions, and the second one cannot be answered
> by any number of existence proofs, however honest.

## 4. Definitions and vocabulary

The fourth column names the unit or coding convention. An appraisal term without its unit attached
is a word rather than a measurement, and this topic is largely about attaching units to words.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Case study | A narrative account of one deployment, usually with selected metrics | Treated as a data point in a sample | One site, one period |
| Reference customer | A customer who has agreed to be named and contacted | Treated as a typical customer | Indicator plus name, with date |
| Survivorship bias | Distortion from observing only the units that survived a selection process | Used loosely for any favourable selection | Difference between observed and population mean |
| Self-selection | Selection arising because units choose whether to participate | Confused with randomisation failure | Participation indicator |
| Publication bias | Distortion because results are published selectively | Applied only to academic literature | Ratio of published to conducted studies |
| Base rate | The prior frequency of an outcome in the relevant population | Confused with the observed rate in a selected sample | Proportion per population |
| Regression to the mean (RTM) | The tendency of extreme measurements to be closer to the mean on re-measurement | Used as an explanation for any improvement | Expected change in outcome units |
| Reliability (rho) | Correlation between repeated measurements of the same unit under unchanged conditions | Assumed to be 1 | Dimensionless, 0 to 1 |
| Counterfactual | What would have happened to the same unit absent the intervention | Used as a synonym for baseline | Outcome, unobserved |
| Comparator | The group or period against which the treated group is compared | Assumed to be equivalent without evidence | Group or period definition |
| Concurrent control | A comparison group measured over the same period | Confused with a historical control | Group, same period |
| Historical control | A comparison period before the intervention | Treated as equivalent to a concurrent control | Period definition with dates |
| Pre/post design | A single group measured before and after | Described as a controlled study | One group, two periods |
| Endpoint | The quantity whose change constitutes the result | Used to mean any reported metric | Named metric with definition and unit |
| Primary endpoint | The single endpoint fixed in advance for the decision | Chosen after seeing the data | One named metric, dated |
| Surrogate endpoint | A metric standing in for the outcome of interest | Treated as the outcome | Named proxy and its validation status |
| Pre-specification | Fixing endpoint, analysis, and decision rule before data collection | Confused with having a plan | Dated document |
| HARKing | Hypothesising after the results are known | Used only for academic fraud | Count of post-hoc endpoints |
| Duration | The elapsed period over which the effect is measured | Reported without the outcome lag | Weeks or months, with lag |
| Persistence | Whether the effect remains after the initial period | Confused with statistical significance | Effect at 6 and 12 months |
| Novelty effect | A temporary improvement caused by attention rather than by the intervention | Treated as a durable gain | Difference between early and late periods |
| Modelled projection | A future value computed from assumptions | Presented as a measured result | Currency, with model and inputs |
| Measured outcome | A value observed in operation | Presented without its comparison | Outcome units |
| Return on investment (ROI) | Net benefit divided by cost over a stated period | Reported without the period or the cost base | Ratio, with period and cost scope |
| Total cost of ownership (TCO) | All costs of acquiring, operating, and exiting a system | Used as a synonym for licence price | Currency over a stated life |
| Payback period | Time until cumulative benefit equals cumulative cost | Confused with ROI | Months |
| Total Economic Impact (TEI) | A commissioned analyst methodology producing a risk-adjusted business case | Read as an independent audit | Commissioned study, four components |
| Composite organisation | A fictional customer built from several interviewed organisations | Described as a real customer | Fictional unit from 4–6 interviews |
| Magic Quadrant (MQ) | A two-axis vendor positioning chart published by Gartner | Read as a product ranking | Position in a 2x2, per market |
| Ability to Execute | One MQ axis, covering product, viability, sales, and operations | Treated as a measure of product quality | Ordinal position on an axis |
| Completeness of Vision | The other MQ axis, covering strategy and market understanding | Treated as a measure of fit | Ordinal position on an axis |
| Inclusion criteria | The rules determining which vendors appear in an analyst report | Assumed to cover the whole market | Rule set, revised annually |
| Maturity model | A staged rubric describing process adoption | Read as a performance measure | Level 1–5 against a rubric |
| Hype cycle | A five-phase curve describing technology expectations over time | Read as an adoption forecast | Phase label |
| Pay-to-play | The claim that analyst coverage depends on commercial spend | Asserted without evidence in either direction | Allegation, not a measurement |
| Conflict of interest | A financial or personal interest that could bias a claim | Treated as proof of falsity | Disclosure statement |
| Material connection | A relationship between endorser and advertiser that must be disclosed | Ignored in B2B contexts | Disclosure, per regulatory guide |
| Testimonial | A statement about experience used to promote a product | Treated as evidence of typical results | Statement plus typicality basis |
| Typicality | The requirement that an advertised result represent what customers generally achieve | Confused with "results not typical" disclaimers | Legal standard, not a statistic |
| Evidence grade | A letter A–E summarising the strength of a source | Assigned by vibes | Ordinal grade with stated basis |

Four of these deserve expansion, because they carry most of the analytical weight.

**Reliability and regression to the mean.** Reliability is the correlation between two measurements
of the same unit taken under unchanged conditions. It is not a property of the metric in the
abstract; it is a property of the metric, the measurement process, and the time gap. If reliability
is 0.55, then a unit measured 5.7 weeks above the mean is expected to be 0.55 times 5.7, or 3.1
weeks above the mean, on re-measurement, with no intervention at all. The expected fall is the
remaining 2.6 weeks. Every appraisal of a cohort selected for being bad must start here.

**ROI, TCO, and payback.** ROI is a ratio of net benefit to cost over a period. TCO is the full cost
base, including implementation, integration, internal labour, and exit. Payback is a duration. A
vendor can report a high ROI with a narrow cost base and a short period, and every individual number
can be correct. When you see 4.1x, ask: net of what, over what period, against which cost lines.

**TEI and the composite organisation.** A Total Economic Impact study is a commissioned business
case produced by an analyst firm. It is built from four to six customer interviews and a
risk-adjusted financial model, and the resulting "organisation" is a fictional composite explicitly
constructed to represent an ideal customer profile rather than a statistical average. That is a
defensible methodology for illustrating a business case. It is not a sample estimate, and Forrester's
own documentation says so.

**Typicality.** In United States advertising regulation, a testimonial about results carries an
implicit claim that the result is typical, and a "results not typical" disclaimer does not cure a
misleading impression. This is a legal standard about consumer understanding, not a statistical test.
It matters here because it means the question "is this customer typical?" is not merely good
practice; in the advertising context it is the question the regulator asks.

## 5. The mechanics, step by step

Fifteen steps in five phases. The order is deliberate: fix the claim before you evaluate it, run the
six questions before you do arithmetic, repair the comparison before you score, and write the memo
last.

### Phase A — Fix the claim

**Step 1. Write the claim as one sentence with a number, a unit, a denominator, and a period.**

"Six named customers reduced excess inventory by 23% within 90 days of go-live, measured against
their own pre-implementation baseline, out of 180 deployments in the installed base." That sentence
contains almost everything the appraisal needs, and the vendor's slide contains almost none of it.
Writing it out is the highest-value five minutes in the process, because half the failures in
section 10 are visible in the sentence itself.

*Input:* the slide, the case study, or the talk. *Output:* one sentence. *Fails if skipped:* you
spend the meeting arguing about whether 23% is impressive rather than about what it is 23% of.

**Step 2. Place the claim on the evidence ladder.**

Decide whether the number is an assertion, a modelled projection, a pre/post measurement, or a
controlled comparison. This determines what kind of question you can ask next. A projection invites
an audit of assumptions. A pre/post measurement invites a counterfactual. A controlled comparison
invites a check on the design.

*Input:* the sentence from step 1. *Output:* a ladder position, 0 to 5. *Fails if skipped:* you
apply the standards of a controlled trial to a projection and conclude the vendor is dishonest,
when the vendor is doing something else entirely.

**Step 3. Name the decision and the cost of both errors.**

State what you would do differently if the claim were true, and what it costs if you act on it and
it is false. A claim supporting a reversible €40,000 pilot deserves less scrutiny than one
supporting a five-year, €3m commitment with a two-year implementation. The rigour should match the
decision, and stating the decision first prevents both over- and under-scrutiny.

*Input:* the procurement context. *Output:* a decision statement and two cost figures. *Fails if
skipped:* every claim gets the same treatment, which means the important ones get too little.

### Phase B — Ask the six questions

**Step 4. Sample: who, how many, out of what, and how selected.**

Ask for four numbers: the number of named cases, the number of deployments they were drawn from,
the selection rule, and the number of deployments that failed or churned. Vendors usually have the
first two and rarely volunteer the fourth. If the denominator is refused, treat the claim as an
anecdote with a chart.

*Input:* the deck and the appendix. *Output:* numerator, denominator, selection rule. *Fails if
skipped:* the base rate is invisible and the number is uninterpretable in principle.

**Step 5. Comparator: what would have happened otherwise.**

Ask what the 23% is measured against. The answers form a hierarchy: a concurrent randomised control
is strongest; a concurrent matched comparison is next; a historical baseline is weaker; a modelled
counterfactual is weaker still; an implicit "nothing else changed" is not a comparison at all. Write
down which one you were given, in those words.

*Input:* the study description. *Output:* a named comparator type. *Fails if skipped:* a pre/post
result is read as a causal effect, and the entire improvement is attributed to the product.

**Step 6. Endpoint: what is measured, who defined it, and was it fixed in advance.**

Check three things. Is the endpoint an outcome (inventory, cost, service) or an activity (logins,
alerts, forecasts generated)? Who defined it, and would they define it the same way if the result
were unfavourable? Was it named before the deployment or chosen afterwards from the metrics that
moved?

*Input:* the metric definition and any protocol. *Output:* endpoint type and pre-specification
status. *Fails if skipped:* an activity metric becomes the basis for an outcome decision.

**Step 7. Duration: how long, with what lag, and does it persist.**

Ask when the clock started, how long the measurement window was, how long the outcome takes to
materialise, and whether anyone has measured at twelve months. A 90-day window on an inventory
metric captures the easy reductions — the slow-moving stock you were going to write down anyway —
and misses the replenishment failures that appear in the following two quarters.

*Input:* the timeline. *Output:* window length, outcome lag, persistence evidence. *Fails if
skipped:* a transient novelty effect is booked as a durable gain.

**Step 8. Funding and benefit: who paid, who gains, and what is disclosed.**

Identify who commissioned the study, who was interviewed, who approved the text, and what the
vendor gained. A commissioned study is not automatically wrong; it is automatically interested. The
useful question is not "is there a conflict?" — there always is — but "which direction would the
conflict push the number, and is there any check on it?"

*Input:* the cover page, the disclosure, the methodology note. *Output:* funder and beneficiary.
*Fails if skipped:* a commissioned study is read as independent research because it has a logo on it.

**Step 9. Reproducibility: could a third party rerun it.**

Could a competent analyst with access to the customer's data reproduce the 23% from the description?
If the answer requires the vendor's proprietary model, the claim is not reproducible even in
principle. Ask what data would be needed and whether the customer would release it under a
non-disclosure agreement (NDA). The answer tells you how much of the result is measurement and how
much is model.

*Input:* the methodology section. *Output:* reproducible, partly reproducible, or not reproducible.
*Fails if skipped:* a modelled number is treated as a measured one for the rest of the evaluation.

### Phase C — Repair the comparison

**Step 10. Reconstruct the selection at each gate.**

Walk the funnel in section 3.1 for this specific claim. Which gate did each case pass? Was the
customer selected for a good result, for a long relationship, for a strategic logo, or for
willingness to speak? Was the measurement window chosen after the result was known? Each gate that
selects on the outcome adds a bias you can describe even when you cannot quantify it.

*Input:* the customer list and the timeline. *Output:* a gate-by-gate description. *Fails if
skipped:* the bias is felt but not named, and cannot be communicated.

**Step 11. Compute the base-rate implication.**

If the publication gate is "a result above some threshold", and you have a view on the distribution
of true effects, you can compute what the published average would look like under a true average of
zero. Section 6 does this arithmetic. The point is not the specific number; it is that a published
average of 20% is entirely consistent with a true average near zero when the gate is at 15% and the
spread is wide.

*Input:* the gate and an assumed effect distribution. *Output:* expected published average under a
stated true mean. *Fails if skipped:* the number is treated as an estimate of the average, which it
is not.

**Step 12. Compute the regression-to-the-mean expectation.**

If the cohort was selected for a bad baseline, compute the expected re-measurement under no effect:
`expected post = population mean + rho x (baseline - population mean)`. Compare that expectation
with the observed post value. The difference is the part that requires an explanation. Section 6
carries the arithmetic. This is the single most useful calculation in the topic, because it converts
"maybe it was regression to the mean" from an objection into a number.

*Input:* population mean, baseline, reliability. *Output:* expected change under no effect. *Fails
if skipped:* the most common explanation for a pre/post win is never tested.

### Phase D — Decide

**Step 13. Score the pack and set the grade.**

Use the ten-item sheet in section 7. Score each item 0, 1, or 2. Apply the override rule: a zero on
sample denominator, comparator, or pre-specified endpoint caps the total regardless of the rest.
Convert the total to an evidence grade A–E for the source section of your memo.

*Input:* the answers from steps 4 to 12. *Output:* a score out of 20, a band, and a grade. *Fails if
skipped:* the appraisal stays in your head, and a colleague cannot check it or reuse it.

**Step 14. State what evidence would change your mind.**

Write one sentence naming the measurement that would move you. "A matched comparison across the
remaining 174 deployments, with the vendor's model held fixed, would change this from directional to
admissible." This does two things: it makes your scepticism falsifiable, and it gives the vendor a
constructive path, which is usually more productive than a no.

*Input:* the appraisal. *Output:* one sentence naming a specific study. *Fails if skipped:* the
appraisal reads as an attitude rather than a position.

### Phase E — Write

**Step 15. Write the memo with evidence and assertion in separate columns.**

The memo is the deliverable, and its structure is the argument. Two columns force the distinction
that the whole topic is about. A claim that cannot be placed in the evidence column goes in the
assertion column with its grade, and the recommendation states how much of the decision rests on it.
Section 6.8 gives a worked paragraph.

*Input:* the score, the arithmetic, the decision. *Output:* a one-page memo. *Fails if skipped:* the
recommendation inherits the vendor's framing and the committee cannot see where the uncertainty is.

## 6. Worked example

A vendor of an inventory optimisation product circulates a case study. The document is constructed
for teaching; the arithmetic is internally consistent and checkable by hand. It is not a description
of any real vendor or customer.

### 6.1 What the document claims

| Claim | Value | Unit | Stated basis |
|---|---:|---|---|
| Excess inventory reduction | 23 | % | Six named customers, 90 days after go-live |
| Average ROI | 4.1 | ratio | Net carrying-cost saving divided by annual subscription |
| Payback | 7 | months | Same model |
| Customers named | 6 | count | Reference programme |
| Deployments in installed base | 180 | count | Appendix, disclosed once |

The document is better than most: it discloses the installed base, names the six customers, states
the 90-day window, and shows the ROI arithmetic. That is why it is a useful teaching case. The
weaknesses are structural rather than rhetorical, and they survive a careful reading.

### 6.2 The sample

Six named customers out of 180 deployments. The document does not say how the six were chosen. A
footnote says they are "representative of customers who have completed implementation". That phrase
carries three selections at once: completed implementation (excludes failures), agreed to be named
(excludes the dissatisfied), and were chosen by the vendor (excludes the unflattering).

The six baselines are given: 11.4, 12.1, 12.8, 13.4, 14.7, and 15.8 weeks of supply. Mean 13.4,
standard deviation 1.6. The population mean, disclosed in the appendix as the installed-base average
at contract signature, is 7.5 weeks with a standard deviation of 3.0. The named cohort sits at
`(13.4 - 7.5) / 3.0 = 1.97` standard deviations above the population mean.

**The six customers were, on the metric being measured, among the worst in the base before the
product was installed.** That is not a scandal; it is a sensible sales motion, because those are the
customers with a reason to buy. It does mean the cohort is selected on the outcome's baseline, which
is the exact condition under which regression to the mean is largest.

### 6.3 The missing comparator

There is no comparison group. The 23% is measured against each customer's own pre-implementation
baseline, computed at go-live. Four things changed at the same time as the software: planners were told to reduce excess inventory,
because that is why they bought the tool; the implementation forced a data cleanup, which usually
improves any inventory metric; the measurement definition changed, because the vendor's tool computes
weeks of supply on its own demand forecast rather than the customer's prior forecast; and the 90-day
window coincides with the post-budget-season period in which many retailers reduce commitments anyway.

None of these is misconduct. All four are reasons why the counterfactual is not the pre-period.
A concurrent comparison across the 174 non-participating deployments would have absorbed three of
the four and left the definitional difference visible. It was not run, and the document does not say
why.

### 6.4 The endpoint

The endpoint is "excess inventory", defined as weeks of supply above the customer's target. Two
problems. First, it is a ratio whose denominator is a forecast: if the vendor's demand forecast is
smoother than the customer's previous one, measured weeks of supply fall even if physical inventory
is unchanged. That is not dishonest; it is what happens when the measurement instrument is also the
product. Second, the target is set by the customer with the vendor's guidance during implementation,
and a target that rises reduces measured excess without any inventory moving. The document does not
report the targets before and after.

A better endpoint would be physical inventory value at constant cost, or inventory turns at constant
demand, measured by the customer's own system of record. Neither is reported.

### 6.5 The base-rate arithmetic

Suppose the true effect of the product across the installed base is a reduction of 0.5 weeks of
supply, with a standard deviation across customers of 2.6 weeks. Suppose the publication gate is a
reduction of at least 2.0 weeks on a 13-week baseline, roughly 15%, plus willingness to be named.
Suppose 12% of eligible customers agree to be named.

The probability that a given customer clears the gate is:

`P(X >= 2.0)` where `X ~ Normal(0.5, 2.6)` = `1 - Phi((2.0 - 0.5) / 2.6)` = `1 - Phi(0.577)` = 0.282.

Out of 180 deployments, that is about 51 customers above the gate. At a 12% naming rate, about six
agree. **The vendor's six named customers are exactly what a true average effect of 0.5 weeks would
produce.** No dishonesty is required.

The average result among the published, given the same distribution, is the truncated mean:

`E[X | X >= 2.0] = 0.5 + 2.6 x (phi(0.577) / 0.282) = 0.5 + 2.6 x 1.198 = 3.61 weeks`.

On a 13-week baseline that is 27.4%. The claim on the slide is 23%. **The published average and the
true average differ by a factor of seven, and both are correct statements about different
populations.** The true effect in the model is 6.7% of the population mean; the published effect is
27.4% of the cohort baseline; the vendor's six customers are exactly what a 0.5-week true effect plus
a 12% naming rate would produce out of 180 deployments.

### 6.6 The regression arithmetic

Now compute what the six would have shown with no product at all. The reliability of a 90-day average
inventory measurement is not 1. Inventory positions are noisy, they are measured with error, and the
underlying process moves. Take reliability `rho = 0.55`, which is a plausible value for a
one-quarter average and should be estimated rather than assumed in a real appraisal.

For a cohort at 13.4 weeks against a population mean of 7.5:

`expected post = 7.5 + 0.55 x (13.4 - 7.5) = 7.5 + 3.245 = 10.745 weeks`.

`expected fall = 13.4 - 10.745 = 2.655 weeks`, which is 19.8% of the baseline.

The vendor's claim of 23% implies a post value of `13.4 x 0.77 = 10.318` weeks, a fall of 3.082
weeks. The part that regression does not explain is `3.082 - 2.655 = 0.427` weeks, or 3.2% of baseline.
So roughly 86% of the claimed improvement is what you would expect from re-measuring a group chosen
for being bad.

**Is the residual real?** With six customers and a standard deviation of the 90-day change of about
2.4 weeks, the standard error of the mean change is `2.4 / sqrt(6) = 0.98` weeks. The 95% confidence
interval for the residual is `0.427 +/- 2.571 x 0.98`, which runs from −2.09 to +2.95 weeks. It
contains zero comfortably. The study cannot distinguish the product's effect from no effect, and it
was never sized to.

### 6.7 What the ROI figure inherits

The 4.1x ROI is arithmetically consistent with the 23% claim. Take a customer with annual cost of
goods sold (COGS) of €48m, or a weekly COGS rate of €0.923m. A 3.08-week reduction releases
`3.08 x 0.923 = €2.84m` of inventory; at a 22% annual carrying cost that is €625,000 per year.
Against a €150,000 subscription, ROI is `625 / 150 = 4.17`, which rounds to 4.1. Internally coherent.

Now substitute the residual. A 0.427-week reduction releases `0.427 x 0.923 = €0.394m`, worth
€86,700 per year at 22%. Against the same subscription, ROI is `86.7 / 150 = 0.58`. Below 1: the
inventory released falls from €2.84m to €0.39m and the annual benefit from €625k to €87k.

The ROI is not wrong. It is a projection that inherits 100% of a contested effect size, and a fall of
76% in the effect estimate — from 3.08 to 0.74 weeks — takes the business case below break-even.
That sensitivity, not the arithmetic, is the finding.

### 6.8 The memo paragraph

> **Claim.** The vendor reports a 23% reduction in excess inventory across six named customers,
> implying a 4.1x ROI.
>
> **Evidence.** The six are drawn from 180 deployments and sit 1.97 standard deviations above the
> installed-base mean at baseline. There is no concurrent comparison. The endpoint is computed by
> the vendor's own forecast engine, and the target it is measured against is set during
> implementation. A regression-to-the-mean calculation with an assumed reliability of 0.55 explains
> 2.66 of the 3.08 claimed weeks; the residual of 0.43 weeks has a 95% interval of −2.09 to +2.95
> and is not distinguishable from zero. The figure is Grade C: a company case, directionally
> plausible, not a measured effect.
>
> **Assertion.** The vendor states that customers typically achieve 4.1x. This is a modelled
> projection conditional on the 23%, and it is Grade D.
>
> **Recommendation.** Proceed to a paid 12-week pilot in eight distribution centres with a matched
> comparison group of eight, a pre-registered primary endpoint of physical inventory value at
> constant COGS, and a kill criterion of less than 1.5% reduction. Budget €180,000 including
> internal labour. Do not sign the three-year agreement on the basis of the case study.

That paragraph is the deliverable. It separates claim, evidence, and assertion, assigns grades, and
converts the appraisal into a decision with a falsifiable test.

## 7. Code, query, or calculation

The scoring sheet is the primary artefact and it is a table, not code. It takes about two minutes per
item, so it fits a meeting. The one code block performs the two arithmetic moves from section 6.

### 7.1 The ten-item appraisal sheet

Score each item 0, 1, or 2. Maximum 20.

| # | Question | 0 | 1 | 2 |
|---:|---|---|---|---|
| 1 | Is the denominator disclosed and plausible? | No denominator | Denominator stated but selection rule not | Denominator and selection rule both stated |
| 2 | Is the selection process described? | Cases chosen by vendor, unexplained | Partly described | Rule stated and applied to the whole base |
| 3 | Is the comparator concurrent? | None, or implicit | Historical baseline | Concurrent control or matched comparison |
| 4 | Is the counterfactual stated? | No | Implied | Explicit and defended |
| 5 | Was the endpoint pre-specified? | No, chosen after | Stated but undated | Dated protocol or pre-registration |
| 6 | Is the endpoint an outcome rather than an activity? | Activity only | Mixed | Outcome, measured by the customer |
| 7 | Is the duration adequate and persistence shown? | Under one cycle, no follow-up | One cycle | Full cycle plus 12-month follow-up |
| 8 | Are funding and material connections disclosed? | Undisclosed | Named funder | Funder, role, and editorial control stated |
| 9 | Is the arithmetic reproducible from disclosed inputs? | No | Partly | Yes, by a third party |
| 10 | Is there independent corroboration? | None | Same vendor, different document | Independent source or replication |

Bands:

| Score | Band | What it supports |
|---:|---|---|
| 0–6 | Reject as evidence | A lead to investigate, not a basis for a decision |
| 7–11 | Directional only | Supports a small, reversible pilot |
| 12–15 | Usable with caveats | Supports a pilot with a pre-registered endpoint |
| 16–20 | Admissible | Supports a decision, subject to local validation |

**Override rule.** A score of 0 on item 1, 3, or 5 caps the total at 11 regardless of the other
items. Those three items are the ones whose absence cannot be compensated: without a denominator the
number is uninterpretable, without a comparator it is not causal, and without pre-specification the
endpoint was chosen by the data.

### 7.2 The two calculations

Twenty-two lines, standard library only. It reproduces the base-rate and regression figures from
section 6 and accepts different inputs.

```python
from math import sqrt, exp, pi, erf

def phi(z):                      # standard normal density
    return exp(-z * z / 2) / sqrt(2 * pi)

def upper_tail(z):               # P(Z > z)
    return 0.5 * (1 - erf(z / sqrt(2)))

mu, sd, rho = 7.5, 3.0, 0.55     # population mean, SD, reliability
base = 13.4                      # selected cohort baseline, weeks
observed = base * (1 - 0.23)     # vendor's claimed post value

expected_post = mu + rho * (base - mu)
print(f"expected post {expected_post:.3f} w")
print(f"reversion     {base - expected_post:.3f} w  ({(base - expected_post) / base:.1%})")
print(f"residual      {expected_post - observed:+.3f} w")

m, s, gate, named = 0.5, 2.6, 2.0, 0.12   # true effect, spread, gate, naming rate
p = upper_tail((gate - m) / s)
print(f"P(above gate) {p:.3f}  expected above {180 * p:.1f}  named {180 * p * named:.1f}")
print(f"E[effect | published] {m + s * phi((gate - m) / s) / p:.2f} w")
```

Expected output: `expected post 10.745 w`, `reversion 2.655 w (19.8%)`, `residual +0.427 w`,
`P(above gate) 0.282`, `expected above 50.8`, `named 6.1`, `E[effect | published] 3.61 w`.

Three checks. The residual must equal the claimed fall minus the expected reversion, to three
decimals. The expected named count must be close to the number actually named; if not, your assumed
distribution is wrong and the base-rate argument does not apply as stated. And if you set `rho =
1.0`, the reversion must fall to zero, which confirms you are computing a reversion, not an effect.

> **Note:** The reliability of 0.55 is an assumption, not a measurement, and the reversion scales
> with it. Estimate reliability from repeated measurements in a period with no intervention. If you
> cannot, present a range: at `rho` between 0.4 and 0.7 the reversion here runs from 2.36 to 4.13
> weeks.

## 8. Data, standards, and regulatory context

There is no standard that governs a vendor's ROI claim in the way that accounting standards govern a
financial statement. That absence is the context for everything in this section, and it should be
stated plainly rather than filled with borrowed authority.

### 8.1 What does not exist

| Claim type | Governed by | Consequence |
|---|---|---|
| Drug efficacy | Regulated trials, marketing authorisation | A claim requires evidence to a defined standard |
| Audited financial results | Accounting standards and audit | Figures are independently examined |
| Supply chain ROI from a software purchase | Nothing | The claim is advertising, or a commissioned study |
| Analyst vendor ranking | Nothing beyond general law | The ranking is opinion, as a court has held |

The asymmetry is worth sitting with. A pharmaceutical company cannot say "patients saw a 23%
improvement" without a regulated evidentiary basis. A software vendor can say exactly that on the
basis of six self-selected customers, and the statement is lawful. The appraisal method in this
topic is not a substitute for regulation; it is what you do because regulation is absent.

### 8.2 What does bind, narrowly: advertising and testimonials

In the United States, the Federal Trade Commission's (FTC) Guides Concerning the Use of Endorsements
and Testimonials in Advertising, at 16 CFR Part 255, constrain how customer results may be presented.
The guides were revised in 2023, with the current version effective 26 July 2023 and published at 88
FR 48102. Grade B: primary regulatory text.

Three provisions matter for vendor case studies.

**Honest opinion and advertiser responsibility.** An endorsement must reflect the endorser's honest
opinion, and the advertiser has a duty to guide, monitor, and remediate its endorsers. A vendor that
supplies a script to a reference customer is responsible for what that customer says.

**Material connection disclosure.** Section 255.5 requires disclosure of connections between the
advertiser and the endorser that would materially affect the weight of the endorsement — payment,
free product, employment, and similar. A customer who was paid to appear, or who receives ongoing
discounts, is a materially connected endorser.

**Typicality.** Section 255.2 addresses consumer testimonials about results. The guides take the
position that a "results not typical" disclaimer does not cure a misleading impression, because
consumers do not interpret it as intended. The operative question is whether the advertisement
conveys what customers generally achieve.

> **Note:** These guides are United States advertising law, enforced against advertisers, and are
> unevenly applied to business-to-business software. They do not require a controlled study and they
> do not create a private right of action for a disappointed buyer. Their value in appraisal is
> conceptual: they establish that "is this typical?" is the right question, and that a disclaimer is
> not an answer.

### 8.3 What binds through procurement rather than through regulation

Public-sector and large-company procurement functions impose their own requirements, which vary by
jurisdiction and organisation: a stated number of reference sites, a documented due-diligence
process, disclosure of conflicts by evaluators, and attributable benefits in the business case. These
are contractual rather than statutory. Treat them as leverage: a vendor who will not provide three
reference sites to a process that requires three is telling you something.

### 8.4 Analyst methodology, self-disclosed

Analyst firms publish their own methodology, and the published methodology is the best available
description of what a report measures. It is also marketing-adjacent and should be graded
accordingly.

| Document | What it discloses | What it does not | Grade |
|---|---|---|---|
| Gartner, Magic Quadrant FAQ | Axes, up to 15 criteria, weighting by high/medium/low, inclusion criteria, annual revision | The specific weights applied, the reference customers consulted, the interview notes | D |
| Gartner, "How Markets and Vendors Are Evaluated" | Criteria definitions and the evaluation process | Vendor-level scoring detail | D |
| Forrester, TEI methodology and policy | Four components: cost, benefit, flexibility, risk; commissioned status; no endorsement | The individual customer data behind the composite | D |
| Forrester, composite organisation guidance | Composite built from 4–6 interviews; explicitly not a statistical average | How the composite is weighted | D |

The most important disclosure in that table is Forrester's own: the composite organisation is
deliberately not an average. That is honest, and it means a TEI number is an illustration of a
business case rather than an estimate of a typical return.

### 8.5 The legal character of an analyst position

In *ZL Technologies, Inc. v. Gartner, Inc.*, 709 F. Supp. 2d 789 (N.D. Cal. 2010), affirmed at 433
F. App'x 547 (9th Cir. 2011), a vendor sued over its placement as a Niche Player in a Magic Quadrant.
The court dismissed the defamation and trade libel claims, holding that the placement was
non-actionable opinion rather than a statement of fact. Grade B: court record.

The reasoning is the useful part. The court noted that the axes are abstract and subjective, that the
report disclaimed the opinions as subject to change, that the criteria were qualitative, and that
there was no product testing. It also held that Gartner describing its model as rigorous did not
convert the ranking into a factual assertion, and that an alleged economic interest — vendors paying
for analyst time — did not remove the protection.

Read that as an appraisal instruction. The publisher of the ranking, in a court, characterised it as
opinion formed from subjective criteria and selective references. That does not make it useless; it
tells you what it is: structured expert judgement about vendor positioning, not a measurement of
fitness for your requirement.

### 8.6 What is convention rather than requirement

- **"Three customer references."** A procurement habit, not a rule. The number is not the point; the
  selection of the three is.
- **"A proof of concept before purchase."** Often theatre when the vendor defines success, the
  timeline is fixed, and there is no kill criterion.
- **"The business case must show a positive ROI."** A convention that produces optimistic modelling,
  because a case with a negative net present value does not get submitted.
- **"Twelve months is long enough."** A convention with no basis in any particular supply chain.
  Seasonality, contract cycles, and asset life all differ.

## 9. Industry practice and case evidence

### 9.1 How vendor case studies are produced

The reference programme is a marketing function with a sales purpose. Its output is shaped by four
filters, and knowing them tells you what to expect before you read the document.

| Filter | What it removes | Visible signature in the document |
|---|---|---|
| Customer must have a good result | The median and the failures | All cases show improvement; no range is given |
| Customer must agree to be named | The dissatisfied and the constrained | Named customers are large, long-standing, or reference-active |
| Legal and brand review | Specific numbers, contract terms | Rounded percentages, "significant", "double-digit" |
| Vendor editorial control | Unflattering framing | Customer quotes are generic; no costs are disclosed |

The fourth filter surprises people. Even a genuine customer quote has usually been drafted, approved,
and edited by the vendor's marketing team. That does not make it false; it makes it a marketing
artefact with a customer's signature on it, a different object from an interview transcript.

### 9.2 The composite and the modelled case

Two production techniques deserve names because they are frequently mistaken for measurement.

**The composite organisation.** A commissioned study interviews four to six customers and builds a
fictional organisation embodying their shared characteristics. Forrester's own guidance states that
the composite reflects an ideal customer profile and is deliberately not a statistical average. The
resulting ROI is a model output conditional on the composite's assumptions: a legitimate way to
structure a business case, but not a sample estimate, and not a number that can be averaged with
anything.

**The value calculator.** A spreadsheet in which the buyer enters headcount, order volume, and hourly
cost, and the tool returns an annual saving. Every step is arithmetically correct and every input is
an assumption. Its real function is to make the buyer co-author the number, which increases
commitment to it. Save the inputs; the output is your model, not the vendor's evidence.

### 9.3 What the analyst industry actually produces

The Magic Quadrant places vendors on two axes using up to fifteen criteria weighted high, medium, or
low. The criteria are published; the weights and the reference calls are not. Inclusion is governed
by market presence, product features, customer interest, and other characteristics, and the vendor
set is deliberately limited rather than exhaustive — Gartner's own FAQ states that exclusion does not
mean a vendor is not viable.

Criticism of the method is well documented and comes from three directions.

**From the analyst relations profession.** The Institute of Influencer and Analyst Relations (IIAR)
argues that market taxonomy drives inclusion and is subjective, that participation is often capped at
sixteen to eighteen vendors for reasons of analyst bandwidth rather than market structure, and that
combining variables through undisclosed weights is the method's most contested feature. Grade C.

**From academic study.** Pollock and Williams, in *Information and Organization* (2009), describe the
Magic Quadrant as performative: vendors reshape their products and messaging to fit the axes, so the
tool participates in constituting the market it claims to measure. Grade A.

**From litigation.** The ZL Technologies case in section 8.5 establishes the position as opinion in
law. Grade B.

None of this makes the reports worthless. They answer a narrow question — how a group of analysts,
using undisclosed weights and selected references, rank vendor viability and strategy in a market they
defined — which is adjacent to, and not the same as, "which product should we buy".

### 9.4 Maturity models and hype cycles

A maturity model scores an organisation against a rubric of process adoption, usually on a five-level
scale, usually from self-reported answers to a questionnaire. It measures **adoption of described
practices**, not performance. Two organisations at the same level can differ by a factor of two in
cost-to-serve, because the level describes what they do and not how well it works. The rubric's
levels are also ordinal: the difference between level 2 and level 3 is not the same size as the
difference between level 4 and level 5, and averaging them is meaningless.

The hype cycle has attracted direct methodological criticism. Steinert and Leifer (PICMET 2010)
tested the curve against longitudinal data and concluded it must be strongly criticised and needs an
underlying quantitative model. Dedehayir and Steinert (2016, *Technological Forecasting and Social
Change*) compared hype-cycle patterns against Gartner's own data and found incongruences. Grade A.
The phase label is a summary of analyst sentiment, not a forecast with a measurable error rate.

### 9.5 What the public record does not contain

We found no public audit of vendor ROI claims in supply chain software, no registry of failed
deployments, and no independent benchmark of realised versus claimed benefits across a category.
Buyers do not publish post-implementation reviews, and vendors have no incentive to. The absence is
the finding: the evidence base for a category-level claim is thinner than for any individual case
study, because the case studies are the only public artefacts and they are selected.

> **Note:** This is why the local pilot in section 6.8 is the recommendation rather than a better
> literature review. When the public evidence is a set of selected existence proofs, the only way to
> learn the base rate for your operation is to generate an observation of your own, with a comparison.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Denominator withheld | "Customers saw a 20% improvement" with no count | Selection invisible; numerator presented as rate | Claim read as an average when it is a maximum | Ask "out of how many" before anything else |
| Survivorship in the reference list | Logo slide with no churn, no failures, no range | Only survivors are nameable | Base rate overstated by an unknown factor | Ask for the count of deployments and the count still live |
| Pre/post read as causal | A before-and-after table with no control column | Counterfactual assumed to be the pre-period | Entire improvement attributed to the product | Ask what else changed at go-live |
| Regression to the mean read as effect | Cohort selected on worst baseline, large improvement | Extreme values revert on re-measurement | 50–90% of a pre/post win can be artefact | Compute expected reversion with reliability |
| Endpoint substitution | Headline metric is logins, alerts, or "decisions influenced" | Activity is easier to move than outcome | Decision made on a proxy with no established link | Require an outcome measured in the customer's system |
| Endpoint defined by the vendor's own tool | Metric computed inside the product being sold | Measurement instrument is also the intervention | Definitional improvement with no physical change | Ask for the customer's own system-of-record figure |
| Post-hoc endpoint | Headline metric does not appear in the pilot protocol | Endpoint chosen from what moved | Reported effect is selected, not estimated | Ask for the dated protocol; check the metric list |
| Duration too short | 90-day result, no 12-month data, no seasonality discussion | Early gains include easy wins and novelty | Effect decays after the case study is written | Ask for persistence at 12 months on the same cohort |
| Projection presented as measurement | "ROI of 4.1x" with no measured cost line | Model output inherits assumed inputs | Business case is a hypothesis dressed as a result | Trace every input; recompute at the low end |
| Composite described as a customer | "A leading retailer" with suspiciously round numbers | Composite of several organisations | Reader assumes a real single site | Ask how many organisations the composite represents |
| Analyst position read as fitness | "We are a Leader, so we are the safe choice" | Position measures analyst view of viability and strategy | Wrong shortlist, or wrong weighting for your context | Read the criteria and reweight for your requirement |
| Maturity level read as capability | "You are Level 2; leaders are Level 4" | Rubric measures described practice adoption | Spend on process theatre rather than outcome | Ask what outcome correlates with the level, and in which study |
| Conflict undisclosed | "Independent research" with no funder named | Commissioned study framed as independent | Reader gives it weight it has not earned | Look for "commissioned by" on the cover and in the footer |
| Pilot that cannot fail | No kill criterion, no comparison group, vendor defines success | Success is defined after the result | Pilot becomes a delayed purchase | Agree baseline, threshold, duration, and kill criterion in writing |

Two of these deserve expansion.

**Regression to the mean is the most common explanation for a pre/post win, and the least often
tested.** It is not a trick or a bias in the pejorative sense; it is arithmetic. Any cohort selected
because a metric was bad will improve on average, with no intervention. In supply chain settings the
selection is usually explicit — the customer bought the product because inventory was high — which
makes the reversion larger, not smaller. The calculation in section 6.6 takes four numbers and two
minutes, and it should be routine before any pre/post result is presented upward.

**The vendor-defined endpoint is the quietest failure.** When the metric is computed inside the
product, a definitional change and a real change are indistinguishable from outside. The remedy is
not accusation but a second measurement: ask the customer's finance or operations system for the same
quantity, computed the same way as before the deployment. If the two agree, the effect is probably
real. If they diverge, the divergence is the finding.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Published cases come from the same population as your operation | Otherwise the result does not transfer | Different scale, region, product mix, or maturity | Compare baseline distributions, not just the means |
| The installed base is representative of the market | The base rate is computed on it | Vendor sells to a selected segment | Ask for the base by segment and by contract size |
| The baseline measurement is reliable enough to estimate reversion | The RTM calculation depends on rho | No repeated measurement exists | Measure the same units twice in a quiet period |
| Reliability is stable between baseline and follow-up | A single rho is used for both | Process change, seasonality, data quality shift | Estimate rho separately in each period |
| The outcome definition did not change | Otherwise pre and post are different quantities | New tool, new forecast engine, new target-setting | Obtain both definitions in writing and compare |
| A historical baseline is a valid counterfactual | Many case studies rely on one | Concurrent shocks, trends, or policy changes | Add a concurrent comparison group; test for pre-trends |
| The decision is reversible at acceptable cost | Justifies a lighter appraisal | Multi-year lock-in, stranded integration | Price the exit before signing |
| The analyst's market definition matches your requirement | Otherwise the shortlist is drawn from the wrong set | Your use case sits across categories | Rewrite the inclusion criteria for your case |
| Your implementation capacity resembles the reference customer's | Transfer depends on it | Reference had dedicated staff and clean data | Compare team size, data quality, and timeline |

**Four hard limits.**

First, the arithmetic in this topic produces a *range of plausible explanations*, not an estimate of
the true effect. The base-rate and reversion calculations show that an observed number is consistent
with a much smaller true effect. They do not tell you what the true effect is. Only a comparison can
do that.

Second, the reliability parameter is usually unknown and the reversion scales linearly with it.
Presenting a single reversion figure without a sensitivity range overstates the precision of the
method. Section 7.2 gives the range for one plausible interval.

Third, the appraisal cannot fix a decision that has already been made. Many procurement processes
select the vendor before the evidence is examined, and the appraisal then functions as
rationalisation. If you are asked to appraise after the shortlist is fixed, say so in the memo; the
value of the exercise changes.

Fourth, the method is calibrated for claims about operational metrics and transfers poorly to claims
about capabilities — "our planners now make better decisions" — where there is no single endpoint.
For those, ask for a decision-level outcome: changed order quantities, changed expedite rates,
changed stock positions. If none exists, the claim is a capability assertion and belongs in the
assertion column with a D grade.

## 12. Exercises

### Exercise 1 — Rewrite the claim (recall and application, 20 minutes)

Take any vendor case study you can find publicly, or one supplied in your organisation. Extract every
quantitative claim and rewrite each as a sentence with a number, a unit, a denominator, and a period.

**Deliverable:** a claim table plus the missing fields. **Acceptance criteria:** at least five claims;
missing fields marked rather than inferred; one claim identified as a projection.
**Estimated time:** 20 minutes.

### Exercise 2 — The six questions (application, 45 minutes)

Apply the six questions to the same case study. Write one paragraph per question, and for each, state
what the document says, what it does not say, and what you would ask.

**Deliverable:** six paragraphs plus a one-line verdict. **Acceptance criteria:** disclosed and
undisclosed distinguished in each; the comparator named by type from the hierarchy in step 5; the
verdict says whether the document is an existence proof or an estimate.
**Estimated time:** 45 minutes.

### Exercise 3 — Base-rate arithmetic (calculation, 30 minutes)

A vendor reports an average 18% cost reduction, from an installed base of 240 customers. Assume true
effects are normal with mean 1% and standard deviation 9%, and that a customer is published only if
the observed reduction exceeds 12% and the customer agrees to be named (assume 15% of eligible
customers agree). Compute the probability of clearing the gate, the expected number above it, the
expected number named, and the expected published average.

**Deliverable:** four numbers with the arithmetic shown. **Acceptance criteria:** normal tail and
truncated mean computed correctly; one sentence stating what the published average does and does not
estimate. **Estimated time:** 30 minutes.

### Exercise 4 — Regression to the mean (calculation, 35 minutes)

A cohort of twelve sites was selected because their forecast error was in the worst decile. Baseline
mean error is 31%, population mean is 19%, population standard deviation is 8%, and the reliability
of the error measure is 0.6. The vendor claims a reduction to 24%. Compute the expected post value
under no effect, the expected fall, the claimed fall, and the residual. Then compute a 95% interval
for the residual assuming a standard deviation of change of 6 percentage points.

**Deliverable:** five numbers plus a verdict. **Acceptance criteria:** reversion formula applied
correctly; residual computed as claimed minus expected; interval uses the correct standard error for
n = 12; verdict says whether the residual is distinguishable from zero.
**Estimated time:** 35 minutes.

### Exercise 5 — Score the pack (application, 30 minutes)

Score the case study from Exercise 1 with the ten-item sheet. Apply the override rule. Convert the
score to a band and an evidence grade.

**Deliverable:** a completed sheet, total, band, and grade. **Acceptance criteria:** every item
scored 0, 1, or 2 with a reason; override rule applied and stated; grade consistent with the band.
**Estimated time:** 30 minutes.

### Exercise 6 — Read the methodology (critique, 40 minutes)

Obtain the published methodology for one analyst report (a Magic Quadrant, a Forrester Wave, or an
IDC MarketScape). Identify the axes, the criteria, the inclusion rules, and what is not disclosed.

**Deliverable:** one page on what the report measures and what it does not. **Acceptance criteria:**
axes and criteria listed; three undisclosed elements named; one sentence for the question it answers
and one for the question it does not. **Estimated time:** 40 minutes.

### Exercise 7 — Build the counterfactual (design, 50 minutes)

For a pilot or deployment you know, design the comparison that was missing. Specify the comparison
units, the matching variables, the outcome definition, and the pre-period.

**Deliverable:** a two-page design note with a matching table. **Acceptance criteria:** comparison
units named and justified; matching includes the baseline level of the outcome; outcome measured in
the customer's system; pre-period length justified against the outcome lag.
**Estimated time:** 50 minutes.

### Exercise 8 — Write the two-column memo (design, 45 minutes)

Write a one-page procurement memo for the decision in Exercise 1, with evidence and assertion in
separate sections, grades for each, and a recommendation that names the evidence which would change
it.

**Deliverable:** the memo, under 500 words. **Acceptance criteria:** every quantitative claim graded
A–E; at least one claim moved from evidence to assertion relative to the vendor's framing; the
recommendation includes a kill criterion and a cost. **Estimated time:** 45 minutes.

### Exercise 9 — Close the chain (critique, 30 minutes)

Take one claim from any source and trace it through the module: what is the claim, what estimand
would test it, what design would identify it, what power would be needed, what interference would
threaten it, what stopping rule would discipline it, and what appraisal it survives.

**Deliverable:** a one-page trace with seven labelled parts. **Acceptance criteria:** all seven
present; the estimand specific enough to compute; the design names a randomisation or comparison
unit; the verdict follows from the trace rather than preceding it.
**Estimated time:** 30 minutes.

## 13. Self-check answers

### Exercise 1

A strong answer produces a table in which at least one row has three of four fields missing. A typical
vendor case study supplies a percentage and a period and omits the denominator and the unit; the
projection is usually the ROI figure, computed from the percentage rather than measured. A weak answer
reproduces the vendor's sentences without decomposing them.

### Exercise 2

The comparator paragraph should name the type explicitly: "historical baseline, set at go-live,
computed by the vendor's tool" is a complete answer; "they compared before and after" is not. The
verdict should say "existence proof" unless a denominator and a selection rule are given. Most vendor
documents are existence proofs, and saying so is the point of the exercise.

### Exercise 3

```text
gate z            = (12 - 1) / 9 = 1.222
P(above gate)     = 1 - Phi(1.222) = 0.111
expected above    = 240 x 0.111 = 26.6
expected named    = 26.6 x 0.15 = 4.0
E[X | X >= 12]    = 1 + 9 x (phi(1.222) / 0.111)
phi(1.222)        = 0.1895
                  = 1 + 9 x 1.707 = 16.4%
```

The published average of about 16% is consistent with a true average of 1%. It estimates the mean of
a truncated distribution, not the mean of the population, and the difference is the selection.

### Exercise 4

```text
expected post   = 19 + 0.6 x (31 - 19) = 19 + 7.2 = 26.2%
expected fall   = 31 - 26.2 = 4.8 pp
claimed fall    = 31 - 24 = 7.0 pp
residual        = 7.0 - 4.8 = 2.2 pp

SE of mean change = 6 / sqrt(12) = 1.732 pp
t(11) 95%         = 2.201
interval          = 2.2 +/- 2.201 x 1.732 = (-1.61, +6.01) pp
```

The residual is 2.2 percentage points, 7.1% of the baseline. The 95% interval includes zero, so the
study cannot distinguish the product's effect from none; it also includes a large positive effect,
so it does not prove the product is useless, only that the study is uninformative at this sample
size. Those are different conclusions and the distinction matters in a meeting.

### Exercise 5

A strong sheet has a zero on the denominator item if no installed-base count is given, which caps the
score at 11 regardless of everything else, and the answer says so explicitly. A common weak answer
gives 1s and 2s for presentation quality — good charts, named customers, clear writing — which are
not evidence items.

### Exercise 6

A strong answer lists the two axes, notes that criteria are published but weights are not, identifies
that reference customers are selected by the vendor, and states that no product testing occurs. The
report answers "how do these analysts rank vendor viability and strategy in a market they defined";
it does not answer "which product fits our requirement".

### Exercise 7

A strong design matches on the baseline level of the outcome, because that determines reversion, and
on the variables that drive the outcome trend — volume, mix, seasonality, prior performance. It
specifies the outcome in the customer's system of record and a pre-period covering one full cycle. A
weak design matches on size and industry only and uses the vendor's metric.

### Exercise 8

A strong memo has a paragraph for evidence, a paragraph for assertion, a grade per claim, and a
recommendation with a cost and a kill criterion. It moves the ROI from the evidence column to the
assertion column, because the ROI is a projection. A weak memo reproduces the vendor's structure with
a caution paragraph appended.

### Exercise 9

A strong trace reads: claim "23% inventory reduction"; estimand the average effect on weeks of supply
across the installed base; design cluster-randomised by distribution centre with a matched
comparison; power sized for a 2-point reduction; interference through shared suppliers and
transferred demand; sequential discipline with three pre-specified looks; appraisal Grade C,
sufficient for a pilot and not for a three-year commitment. A weak trace names the seven words
without connecting them, which is a vocabulary exercise rather than a chain.

## 14. Interview questions

**1. "A vendor says customers saw a 20% improvement. What is your first question?"**

Strong: "Out of how many, and how were those customers selected?" — then explains that the numerator
is a maximum of a distribution and the denominator is what converts it into an estimate. Weak: "Can
I see the case study?", which accepts the framing.

**2. "Why is a case study not evidence of an average effect?"**

Strong: it is an existence proof selected on the outcome, and at least three gates — measurement,
favourability, willingness to be named — stand between the deployment and the document; it answers
"is this possible" and not "is this likely". Weak: "case studies are anecdotal", which is true but
does not explain the selection.

**3. "What is regression to the mean and why does it matter here?"**

Strong: units selected for extreme values are expected to be closer to the mean on re-measurement
even with no intervention; the expected reversion is `(1 - rho) x (baseline - mean)`; customers buy
because the metric is bad, so the cohort is selected on the baseline and the reversion is large.
Weak: "extreme results tend to normalise", which names the phenomenon and not the arithmetic.

**4. "How would you compute how much of a pre/post win is regression?"**

Strong: take the population mean, the cohort baseline, and an estimate of reliability; compute
`mean + rho x (baseline - mean)`; compare with the observed post; report the residual with an
interval and a sensitivity range over rho. Weak: "you can't really separate them", which is false
when the population parameters are available.

**5. "A vendor shows 4.1x ROI. What do you do?"**

Strong: trace the ROI to its inputs, identify which inputs are measured and which are assumed,
recompute at the low end of the effect estimate, and check whether the cost base includes
implementation, integration, and internal labour. Notes that the ROI usually inherits the contested
effect size entirely. Weak: "ROI numbers are usually inflated", which is an attitude rather than a
method.

**6. "What does a Magic Quadrant actually measure?"**

Strong: a structured analyst judgement about vendor viability and strategy on two weighted axes,
within a market the analyst defined and an included vendor set; not product testing, not fitness for
your requirement, and legally an opinion. Weak: "it ranks vendors", which treats a positioning chart
as a scorecard.

**7. "Is a Leader in the Magic Quadrant a safe choice?"**

Strong: it is a signal about market position and analyst perception, which correlates with vendor
viability and therefore with support risk, and does not correlate reliably with fit for a specific
requirement; the shortlist should be built from requirements and the report used to check viability.
Weak: "yes, Leaders are the safest", which outsources the requirement definition to the analyst.

**8. "How do you read a maturity model result?"**

Strong: it measures self-reported adoption of described practices against a rubric, on an ordinal
scale, with no necessary relationship to outcome performance; ask what outcome the level predicts and
in which study. Weak: "it tells you how advanced you are", which accepts the label.

**9. "What makes a vendor pilot credible?"**

Strong: a pre-registered baseline, a pre-specified primary endpoint measured in the customer's
system, a comparison group or a matched design, a stated duration covering the outcome lag, and a
kill criterion agreed before launch. Weak: "clear success metrics", which does not say who set them
or when.

**10. "How do you separate evidence from assertion in a memo?"**

Strong: two columns or two sections; every quantitative claim graded A–E with its basis; a
recommendation that states how much of the decision rests on the assertion column and what evidence
would move it. Weak: "I add a caveats section", which leaves the structure of the argument intact.

## 15. Key terms

- **Case study** — a narrative account of one deployment with selected metrics; an existence proof.
- **Reference customer** — a named customer who has agreed to be contacted; selected, not typical.
- **Survivorship bias** — distortion from observing only units that passed a selection process.
- **Publication bias** — distortion because results are published selectively.
- **Base rate** — the prior frequency of an outcome in the relevant population.
- **Regression to the mean** — the tendency of extreme measurements to move toward the mean on
  re-measurement, with no intervention.
- **Reliability (rho)** — correlation between repeated measurements of the same unit under unchanged
  conditions; the parameter that scales reversion.
- **Counterfactual** — what would have happened to the same unit absent the intervention.
- **Comparator** — the group or period against which the treated group is compared.
- **Concurrent control** — a comparison group measured over the same period.
- **Historical control** — a comparison period before the intervention; weaker than concurrent.
- **Pre/post design** — a single group measured before and after; not a controlled study.
- **Endpoint** — the quantity whose change constitutes the result.
- **Primary endpoint** — the single endpoint fixed in advance for the decision.
- **Pre-specification** — fixing endpoint, analysis, and decision rule before data collection.
- **Persistence** — whether an effect remains after the initial period.
- **Modelled projection** — a future value computed from assumptions, not observed.
- **Return on investment (ROI)** — net benefit divided by cost over a stated period.
- **Total cost of ownership (TCO)** — all costs of acquiring, operating, and exiting a system.
- **Total Economic Impact (TEI)** — a commissioned analyst methodology producing a risk-adjusted
  business case.
- **Composite organisation** — a fictional customer built from several interviewed organisations and
  explicitly not a statistical average.
- **Magic Quadrant (MQ)** — a two-axis vendor positioning chart; a structured opinion, not a test.
- **Inclusion criteria** — the rules determining which vendors appear in an analyst report.
- **Maturity model** — a staged rubric describing process adoption, not performance.
- **Material connection** — a relationship between endorser and advertiser that must be disclosed.
- **Typicality** — the requirement that an advertised result represent what customers generally
  achieve.
- **Evidence grade** — a letter A–E summarising source strength, with a stated basis.

## 16. Sources

**Grading convention.** A peer-reviewed or audited; B regulatory, official, or court record; C company
case, practitioner study, or preprint; D vendor, consultancy, or analyst marketing and
self-description; E press, blog, or anecdote. Analyst and vendor pages are dated where a date is
visible and marked as retrieved November 2026 where it is not.

**Peer-reviewed and academic**

- Barnett, A. G., van der Pols, J. C., and Dobson, A. J. (2005). "Regression to the mean: what it is
  and how to deal with it." *International Journal of Epidemiology*, 34(1), 215–220. Grade A.
- Streiner, D. L. (2001). "Regression toward the mean: its etiology, diagnosis, and treatment."
  *Canadian Journal of Psychiatry*, 46(1). Grade A.
- Furrow, R. E., and Hsu, J. L. (2019). "Regression to the mean in pre–post testing: using
  simulations and permutations to develop null expectations." *CBE—Life Sciences Education*, 18(2).
  Grade A.
- Pollock, N., and Williams, R. (2009). "The sociology of a market analysis tool: how industry
  analysts sort vendors and organize markets." *Information and Organization*, 19(2). Grade A.
- Dedehayir, O., and Steinert, M. (2016). "The hype cycle model: a review and future directions."
  *Technological Forecasting and Social Change*. Grade A.
- Steinert, M., and Leifer, L. (2010). "Scrutinizing Gartner's hype cycle approach." PICMET. Grade C.

**Regulatory and legal**

- 16 CFR Part 255, *Guides Concerning the Use of Endorsements and Testimonials in Advertising*.
  Revised guides effective 26 July 2023; 88 FR 48102. Grade B.
- *ZL Technologies, Inc. v. Gartner, Inc.*, 709 F. Supp. 2d 789 (N.D. Cal. 2010). Grade B.
- *ZL Technologies, Inc. v. Gartner Group, Inc.*, 433 F. App'x 547 (9th Cir. 2011), No. 10-16061.
  Unpublished memorandum; not precedential. Grade B.

**Analyst and methodology-owner documentation**

- Gartner. "FAQ: How markets and vendors are evaluated in Gartner Magic Quadrants."
  gartner.com/en/about/magic-quadrant-faq. Grade D, retrieved November 2026.
- Forrester. "Forrester Methodologies: Total Economic Impact." forrester.com/policies/tei. Grade D,
  retrieved November 2026.
- Forrester. "Your Guide to the TEI Composite Organization." Blog, January 2023. Grade D.

**Practitioner and critical commentary**

- Institute of Influencer and Analyst Relations (IIAR). "Gartner Magic Quadrant shortcomings, or how
  not to use the Gartner Magic Quadrant." 9 October 2018. Grade C.
- Chapple, D. "Is there a pay to play problem in the analyst industry?" Influencer Relations, 2014.
  Grade C.

**Consultancy and practitioner guidance on ROI appraisal**

- Baytech Consulting. "From marketing claims to measured outcomes: governing vendor ROI." Grade D,
  undated, retrieved November 2026.

**Evidence hierarchy**

- Center for Evidence-Based Management (CEBMA). "What are the levels of evidence?" Grade C.

**Note on citations.** The case study in section 6 is constructed for teaching, is not a description
of any real vendor or customer, and its arithmetic is internally consistent and reproducible with the
block in section 7.2. The four-to-six interview count and the "not a statistical average" character of
the TEI composite are reported as Forrester describes them and are not independently verified. The
IIAR's sixteen-to-eighteen vendor cap is reported as the IIAR states it. Where a figure could not be
sourced, the text describes the mechanism and does not supply a number.

## 17. Links

- Module map and topic sequence: [`_index.md`](_index.md)
- The claim and estimand this appraisal checks against: [`01_claim-hypothesis-design.md`](01_claim-hypothesis-design.md)
- The randomisation unit that determines whether a comparison is possible: [`02_randomisation-units.md`](02_randomisation-units.md)
- Power and minimum detectable effect, which tell you what a pilot could have found: [`06_power-mde.md`](06_power-mde.md)
- Interference, which is often the reason a vendor's comparison is not clean: [`07_interference-spillover.md`](07_interference-spillover.md)
- Sequential discipline, which vendor dashboards almost never have: [`08_sequential-testing.md`](08_sequential-testing.md)
- Marketing mix modelling and incrementality, the other route to a counterfactual: [`09_mmm-incrementality.md`](09_mmm-incrementality.md)
- Estimands and question types, the vocabulary used in step 1: [`../04_statistics_causal_inference/01_estimands-question-types.md`](../04_statistics_causal_inference/01_estimands-question-types.md)
- Quasi-experiments, which are the design most vendor studies approximate badly: [`../04_statistics_causal_inference/07_experiments-quasi.md`](../04_statistics_causal_inference/07_experiments-quasi.md)
- Difference-in-differences, the method behind a credible historical comparison: [`../04_statistics_causal_inference/08_difference-in-differences.md`](../04_statistics_causal_inference/08_difference-in-differences.md)
- Sensitivity to omitted variables, which is what an absent comparator leaves you with: [`../04_statistics_causal_inference/10_sensitivity-omitted-bias.md`](../04_statistics_causal_inference/10_sensitivity-omitted-bias.md)
- Data quality dimensions, which determine whether the baseline is reliable: [`../02_supply_chain_data_landscape/08_data-quality-dimensions.md`](../02_supply_chain_data_landscape/08_data-quality-dimensions.md)
- The three-flow model, which tells you which outcome a vendor is probably measuring: [`../01_supply_chain_foundations/01_three-flows-one-network.md`](../01_supply_chain_foundations/01_three-flows-one-network.md)
- Module survey file: [`../../../course/05_experimentation_pilots_evidence.md`](../../../course/05_experimentation_pilots_evidence.md)
- Evidence grading and case library: [`../../../course/28_domain_case_library.md`](../../../course/28_domain_case_library.md)
- Return to the part map: [`../_index.md`](../_index.md)
