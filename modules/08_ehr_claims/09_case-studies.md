# Case Studies

> **Module 08 — EHR, Claims, and Health System Operations**
> Topic 09 of 09
> Estimated study time: 3 hours
> Prerequisites: Topics 04, 05, and 08
> Version 2026.3

## 1. Why this topic exists

Case studies are the most persuasive and least reliable form of evidence in health-system analytics. A vendor publishes a story: a health system deployed its model, readmissions fell 39 percent, length of stay fell by a day, and the return on investment was 4:1. The story is selected because it worked, the metric is selected because it moved, and the comparison is selected because it flatters. The buyer cannot see the failures, the concurrent interventions, or the patients who were excluded.

Peer-reviewed evaluations are different but not perfect. They have a design, a comparison, a pre-specified outcome, and a review process. They also have publication bias, single-site limits, and implementation differences that make transferability uncertain. The two kinds of evidence answer different questions: a vendor case answers "can this work somewhere?", and a peer-reviewed evaluation answers "did this work here, compared with what?" Neither answers "will it work in my organisation?"

The cost of appraising badly is a purchased product that does not deliver. A hospital that adopts a readmission programme because a vendor reported a 39 percent reduction, when the peer-reviewed effect is closer to 9 percent, will budget for a return that does not exist. A hospital that dismisses a peer-reviewed evaluation because the effect is modest may miss a programme that is cost-effective at its own scale. The appraisal must separate the claim from the evidence, the evidence from the context, and the context from the local decision.

Three failure patterns recur.

**The vendor-selected metric.** A vendor case reports the metric that moved, not the metric that mattered. A readmission programme may report a 39 percent reduction in a selected high-risk subgroup while the overall rate is unchanged. A revenue-cycle tool may report recovered dollars while ignoring the cost to collect. The appraisal must ask what the denominator was, what the comparison was, and what was omitted.

**The survivorship sample.** Published cases are the ones that worked. The failures are not published, and the base rate of success is unknown. A case study cannot tell you the probability that your organisation will succeed; it can only tell you that success is possible.

**The transferability assumption.** A programme evaluated in an integrated system with a health plan, a mature data infrastructure, and a employed physician group may not transfer to a community hospital with a different payer mix and a different workflow. The appraisal must state the context and the mechanism, not only the effect.

This topic teaches a structured appraisal. The output is a one-page evidence assessment with the claim, the source, the design, the effect, the bias, the context, and the transferability judgement. The assessment is not a summary of the case; it is a decision about how much weight the case can bear.

A case appraisal is also a defence against two opposite errors: credulity and cynicism. Credulity adopts a programme because the story is compelling; cynicism rejects every case because none is definitive. The structured appraisal avoids both by asking what the case can support and what it cannot, and by recommending the next step accordingly.

> **Note:** A case study is existence proof, not an effect estimate. It can show that a programme is feasible and that a result was observed. It cannot show what would have happened without the programme, and it cannot show that the result will replicate.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** a vendor case study, a peer-reviewed evaluation, a preprint, and an internal report, and **state** the evidentiary weight of each.
2. **Extract** the population, intervention, comparator, outcome, design, and effect from a case.
3. **Check** the arithmetic of a claimed return on investment, including the denominator, the cost basis, and the time horizon.
4. **Identify** selection, survivorship, and publication bias in a case.
5. **Assess** transferability using context, mechanism, and implementation fidelity.
6. **Apply** a structured appraisal framework to a public case and **write** a recommendation.
7. **Compare** two cases with different designs and effects and **explain** why they differ.
8. **Design** a local pilot that tests the case's claim before a full commitment.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13. Outcomes 3, 5, and 6 are the core.

## 3. The core mental model

A case study is a chain from a claim to a decision. Each link can weaken the claim, and the appraisal tests each link.

```text
CLAIM
 "Readmissions fell 39%"
    |
    v
SOURCE
 vendor / peer-reviewed / preprint / internal
    |
    v
DESIGN
 before-after / ITS / DiD / trial / case series
    |
    v
EFFECT
 absolute, relative, NNT, cost per avoided event
    |
    v
CONTEXT
 population, payer, system, workflow, fidelity
    |
    v
TRANSFERABILITY
 mechanism, implementation, local capacity
    |
    v
DECISION
 adopt / pilot / reject / gather more evidence
```

The chain has four properties.

**The claim is not the evidence.** A claim is a sentence; the evidence is the design, the data, and the analysis behind it. The first task is to find the evidence behind the sentence.

**The effect has two scales.** A relative reduction of 39 percent can be a 2-point absolute reduction on a 5 percent baseline, or a 20-point reduction on a 50 percent baseline. The absolute effect and the NNT determine the cost per avoided event; the relative effect determines the headline.

**The context is part of the intervention.** A programme that works in an integrated system may depend on the system's data, staff, and payment model. The mechanism, not the brand, is what transfers.

**The decision is local.** The appraisal informs a decision under local constraints: capacity, capital, payer mix, and risk. A case that is strong evidence for one organisation may be weak for another.


### Evidence weight by source and design

| Source | Design | Weight | Claim supported |
|---|---|---|---|
| Peer-reviewed | Trial | High | Causal |
| Peer-reviewed | DiD or ITS | Moderate | Conditional causal |
| Peer-reviewed | Before-after | Low | Association |
| Preprint | Any | Low | Hypothesis |
| Vendor | Before-after | Very low | Feasibility |
| News | Any | Very low | Awareness |

The table is a guide, not a formula. A well-conducted before-after with a placebo can outweigh a poorly conducted DiD, and a vendor case with transparent data can be more useful than a preprint with a black box.

> **Note:** The evidence hierarchy is not a ranking of truth. A well-designed single-site study can be more informative than a poorly conducted multicentre trial. The hierarchy ranks designs by their ability to support causal claims, not by their prestige.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Case study | A descriptive account of an intervention and its results | Treated as an evaluation | Narrative |
| Vendor case | A case published by a product vendor | Treated as peer-reviewed | Marketing document |
| Peer-reviewed evaluation | A study reviewed by independent experts | Treated as universally transferable | Journal article |
| Preprint | A manuscript posted before peer review | Treated as peer-reviewed | Preprint server |
| Internal report | An organisation's own evaluation | Treated as independent | Report |
| Claim | A statement about an effect or result | Confused with evidence | Sentence |
| Design | The structure of the comparison | Ignored in favour of the result | Design label |
| Comparator | The group or period against which the intervention is compared | Omitted in vendor cases | Group or period |
| Counterfactual | The outcome without the intervention | Assumed to be the pre-period | Potential outcome |
| Effect size | The magnitude of the effect | Reported only as relative | Absolute or relative |
| Absolute effect | The difference in rates | Confused with relative | Percentage points |
| Relative effect | The ratio of rates | Confused with absolute | Percent |
| NNT | Number needed to treat | Ignored in business cases | Count |
| Cost per avoided event | Programme cost divided by events avoided | Reported without the denominator | Currency |
| ROI | Return on investment | Computed without the cost to collect | Ratio |
| Internal validity | Whether the study estimates the effect correctly | Confused with external validity | Judgement |
| External validity | Whether the effect applies elsewhere | Assumed from a single site | Judgement |
| Transferability | Whether the intervention can work in a new context | Confused with efficacy | Judgement |
| Selection bias | The sample differs from the population | Ignored in vendor cases | Bias |
| Survivorship bias | Only successful cases are visible | Ignored in case collections | Bias |
| Publication bias | Positive results are more likely to be published | Ignored in literature reviews | Bias |
| Confounding | Another factor explains the effect | Ignored in before-after | Bias |
| Secular trend | A change over time unrelated to the intervention | Attributed to the intervention | Trend |
| Concurrent intervention | Another change at the same time | Ignored in attribution | Confounder |
| Regression to the mean | Extreme values revert | Attributed to the intervention | Bias |
| Fidelity | The degree to which the programme was delivered as designed | Assumed to be 100 percent | Assessment |
| Implementation | The process of delivering the programme | Confused with the intervention | Process |
| Mechanism | The causal pathway | Ignored in favour of the brand | Explanation |
| Context | The setting, population, and system | Treated as background | Description |
| Scalability | The ability to expand the programme | Confused with efficacy | Judgement |
| Sustainability | The ability to maintain the effect | Confused with scalability | Judgement |
| Replication | A repeat of the study in a new setting | Confused with a subgroup analysis | Study |
| Generalisability | The extent to which findings apply broadly | Confused with transferability | Judgement |
| GRADE | A framework for rating evidence certainty | Treated as a study design | Framework |
| ROBINS-I | A tool for assessing bias in non-randomised studies | Confused with GRADE | Tool |
| Risk of bias | The likelihood that the estimate is distorted | Confused with precision | Judgement |
| Confidence interval | The uncertainty around an estimate | Ignored in vendor cases | Interval |
| Statistical significance | The probability of the data under the null | Confused with practical significance | p-value |
| Practical significance | Whether the effect matters | Ignored in favour of significance | Judgement |
| Cost basis | Whether cost is allowed, paid, or actual | Omitted in ROI claims | Currency basis |
| Time horizon | The period over which benefits accrue | Shortened in vendor cases | Months or years |
| Attribution | The assignment of the effect to the intervention | Assumed in vendor cases | Judgement |
| Benchmark | The comparison value | Chosen to flatter | Rate |
| Denominator | The population at risk | Omitted in relative claims | Count |
| Subgroup | A subset of the population | Selected after the result | Group |
| Headline effect | The effect reported in the abstract | Confused with the primary outcome | Effect |
| Primary outcome | The pre-specified outcome | Changed after the result | Variable |
| Secondary outcome | Other outcomes | Treated as primary | Variable |
| Sensitivity analysis | A re-analysis under different assumptions | Omitted in vendor cases | Analysis |
| Pre-registration | A plan written before the study | Rare in vendor cases | Document |
| Replication crisis | The failure of many published results to replicate | Ignored in case selection | Phenomenon |
| Due diligence | The process of verifying a claim | Confused with a site visit | Process |
| Pilot | A small-scale local test | Confused with a rollout | Study |
| Go/no-go | A decision rule for scaling | Written after the pilot | Rule |

## 5. The mechanics, step by step

The procedure below appraises a case study for a local decision. It applies to vendor cases, peer-reviewed evaluations, preprints, and internal reports.

### Step 1. Define the appraisal question and the decision

**Input:** the decision the case is meant to inform.

**Output:** a question naming the decision, the local population, and the evidence needed.

**Rationale:** a case can be appraised for many purposes: to decide whether to buy a product, to design a pilot, to estimate a budget, or to understand a mechanism. The purpose determines the depth and the standard. A budget decision needs the cost per avoided event and the scale; a mechanism question needs the implementation detail.

Write the question before reading the case. "Should we buy this readmission programme for our 9,000 annual discharges?" is a different question from "does the programme reduce readmissions?" The first needs local arithmetic; the second needs a causal estimate.

**Branch point:** if the decision is a purchase, the appraisal must include the contract, the implementation cost, and the exit cost, not only the clinical effect.

### Step 2. Classify the source

**Input:** the document, its author, and its publication venue.

**Output:** a source class: vendor, peer-reviewed, preprint, internal, or news.

**Rationale:** the source class determines the weight and the questions. A vendor case is marketing, even when the data are real; it is selected and framed to sell. A peer-reviewed evaluation has independent review but may still have publication bias. A preprint has not been reviewed and may change. An internal report may be honest but not independent.

Record the author, the funder, the venue, the date, and whether the data are available. A case with no author, no date, and no data is a claim, not evidence.

**Branch point:** if the vendor cites a peer-reviewed study, appraise the study, not the vendor's summary. The summary may overstate the effect or omit the comparator.

### Step 3. Extract the population, intervention, comparator, outcome, and design

**Input:** the case.

**Output:** a structured extract with each element and the page or paragraph where it appears.

**Rationale:** the five elements determine whether the case is relevant and whether it supports a causal claim. A case without a comparator cannot support a causal claim; a case with a selected subgroup cannot support a population claim; a case with a short follow-up cannot support a sustainability claim.

If an element is missing, record it as missing. A missing comparator is a finding, not a gap to fill with an assumption.

**Branch point:** if the population differs materially from the local population, the case may still be useful for mechanism or feasibility, but not for effect size.

### Step 4. Assess the design

**Input:** the design description.

**Output:** a design class and a judgement of internal validity.

**Rationale:** the design determines the counterfactual. A before-after design assumes no change without the intervention; an ITS uses the pre-trend; a DiD uses a control; a trial uses randomisation. The design class is the first determinant of how much weight the effect can bear.

Check whether the design is described accurately. A "pre-post with comparison" may be a DiD; a "time series" may be a simple before-after. Read the methods, not the label.

**Branch point:** if the design is a before-after with no control and a short series, the effect is descriptive. It can support a pilot, not a purchase.

### Step 5. Check the comparator

**Input:** the comparison group or period.

**Output:** a description of the comparator and its credibility.

**Rationale:** the comparator is the counterfactual. A vendor case often compares the post-period to the pre-period, or the treated subgroup to a historical baseline. A peer-reviewed DiD compares the treated group to a control. The comparator determines whether the effect is attributable to the intervention.

Ask whether the comparator is contemporaneous, similar, and untreated. A historical baseline is affected by secular trends; a contemporaneous control is affected by spillover; a matched control is affected by selection on unobservables.

**Branch point:** if the comparator is missing, the case cannot support a causal claim. It can support a feasibility claim.

### Step 6. Check the outcome and the denominator

**Input:** the outcome definition and the population.

**Output:** the outcome, the denominator, the time horizon, and the data source.

**Rationale:** a relative reduction without a denominator is uninterpretable. A 39 percent reduction on a 5 percent baseline is 2 points; on a 50 percent baseline it is 20 points. The denominator determines the absolute effect, the NNT, and the cost per avoided event.

Check whether the outcome is the same as the local outcome, whether the time horizon matches the decision, and whether the data source is credible. A readmission defined by claims may differ from a readmission defined by the EHR.

**Branch point:** if the outcome is a surrogate or a subgroup, the case cannot support a population claim without an assumption.

### Step 7. Verify the arithmetic

**Input:** the claimed effect, cost, and return.

**Output:** a recomputed absolute effect, NNT, cost per avoided event, and ROI.

**Rationale:** vendor cases often report a relative reduction and a return on investment without the intermediate arithmetic. Recompute from the reported counts or rates. If the counts are not reported, the claim cannot be verified.

The formulas are:

```text
Absolute effect = treated rate - comparator rate
NNT = 1 / absolute effect
Cost per avoided event = programme cost / (population x absolute effect)
ROI = (value of avoided events - programme cost) / programme cost
```

A claim that cannot survive these formulas is a marketing number.

**Branch point:** if the cost basis is unclear (allowed, paid, or actual), the ROI is not interpretable. State the basis or reject the claim.

### Step 8. Assess bias

**Input:** the design, the sample, and the publication.

**Output:** a bias assessment with the direction of each bias.

**Rationale:** the common biases in case studies are selection (the site or patients differ), confounding (another change explains the effect), survivorship (only successes are published), publication bias (positive results are more likely to appear), and measurement (the outcome definition changed). Each bias has a direction: selection can inflate or deflate, confounding usually inflates, survivorship inflates, publication bias inflates.

Use a structured tool such as ROBINS-I for non-randomised studies, or a simple checklist for a vendor case. Record the direction and the magnitude where possible.

**Branch point:** if the bias is likely to explain the entire effect, the case is not evidence for the intervention. It may still be evidence that the intervention is feasible.

### Step 9. Assess the context and mechanism

**Input:** the setting, the workflow, and the implementation.

**Output:** a description of the context and the proposed mechanism.

**Rationale:** the context is part of the intervention. A programme that depends on a mature data infrastructure, a employed physician group, or a risk-bearing contract may not work without them. The mechanism explains why the effect occurred and what must be replicated.

Ask what the programme actually did, who delivered it, how often, and what resources it required. A case that describes the brand but not the workflow cannot be replicated.

**Branch point:** if the mechanism is not described, the case cannot support a transferability claim. It can support a site visit or a pilot.

### Step 10. Assess transferability and implementation

**Input:** the local context and the case context.

**Output:** a transferability judgement with the conditions that must hold.

**Rationale:** transferability depends on the population, the payer mix, the workflow, the staff, the data, and the incentives. A case that worked in a integrated system with a health plan may not work in a standalone hospital. A case that worked with a dedicated team may not work with existing staff.

List the conditions that made the case work and test whether they hold locally. If a condition is missing, the expected effect is lower, and the pilot should test it.

**Branch point:** if a critical condition is missing and cannot be created, the case is not transferable. The decision is to reject or to test a different model.

### Step 11. Rate the certainty and synthesise

**Input:** the design, the bias assessment, and the transferability.

**Output:** a certainty rating (high, moderate, low, very low) and a summary.

**Rationale:** the certainty rating communicates how much confidence the decision-maker should place in the effect. A randomised trial with low bias and a relevant population is high certainty; a vendor before-after with no comparator is very low. The rating is not a verdict on the programme; it is a verdict on the evidence.

Use GRADE domains: risk of bias, inconsistency, indirectness, imprecision, and publication bias. A single case will usually be low or very low certainty, even when the effect is large.

**Branch point:** if the certainty is very low, the decision should be a pilot or a "no", not a full adoption.

### Step 12. Recommend and design the local test

**Input:** the appraisal and the local constraints.

**Output:** a recommendation (adopt, pilot, reject, gather more evidence) and a pilot design.

**Rationale:** the recommendation follows from the certainty and the stakes. A high-certainty, low-cost intervention can be adopted; a low-certainty, high-cost intervention should be piloted. The pilot should test the mechanism and the effect in the local context, with a pre-specified outcome and a go/no-go rule.

The pilot design should be the strongest feasible: a randomised pilot, a stepped wedge, or an ITS with a comparison. The appraisal tells you what to measure; the design tells you how to attribute it.

**Branch point:** if the pilot is not feasible, the decision should be based on the appraisal and the local arithmetic, with the uncertainty stated.

### Decision table: source hierarchy

| Source | Independence | Data access | Typical weight |
|---|---|---|---|
| Peer-reviewed trial | High | Variable | High |
| Peer-reviewed observational | Moderate | Variable | Moderate |
| Preprint | Low to moderate | Variable | Low |
| Internal evaluation | Low | High | Moderate for local |
| Vendor case | Low | Low | Low |
| News report | Very low | Very low | Very low |

### Decision table: design and claim

| Design | Causal claim | Feasibility claim | Transferability claim |
|---|---|---|---|
| Before-after | No | Yes | No |
| ITS | Conditional | Yes | Conditional |
| DiD | Conditional | Yes | Conditional |
| Stepped wedge | Yes, if randomised | Yes | Conditional |
| Trial | Yes | Yes | Conditional |
| Case series | No | Yes | No |

### Decision table: recommendation

| Certainty | Cost and stakes | Recommendation |
|---|---|---|
| High | Low | Adopt |
| High | High | Adopt with monitoring |
| Moderate | Low | Pilot or adopt with monitoring |
| Moderate | High | Pilot |
| Low | Low | Pilot or gather more evidence |
| Low | High | Reject or pilot with a strict rule |
| Very low | Any | Gather more evidence |

### Edge cases that invalidate an appraisal

**No comparator.** A case without a comparator cannot support a causal claim. The appraisal should say so and downgrade the certainty.

**Selected subgroup.** A case that reports a subgroup effect without the overall effect may be reporting the most favourable subset. Ask for the overall result and the pre-specification.

**Changed outcome.** A case that changes the primary outcome after the result is not a pre-specified evaluation. Downgrade the certainty.

**Missing cost basis.** A return on investment without the cost basis is not interpretable. Ask whether the cost is allowed, paid, or actual.

**Short horizon.** A case with a 3-month follow-up cannot support a sustainability claim. Ask for the 12- or 24-month result.

**Single site with a unique context.** A case in an integrated system with a health plan may not transfer to a standalone hospital. State the conditions and test them.

**Publication bias.** A collection of successful cases is not a representative sample. The base rate of success is unknown, and the appraisal should say so.

### Appraisal scorecard

| Domain | Question | Score |
|---|---|---|
| Population | Is the population relevant to the decision? | 0-2 |
| Intervention | Is the intervention described well enough to replicate? | 0-2 |
| Comparator | Is there a credible comparison? | 0-2 |
| Outcome | Is the outcome the same as the local outcome? | 0-2 |
| Design | Does the design support a causal claim? | 0-2 |
| Bias | Are the main biases addressed? | 0-2 |
| Arithmetic | Is the effect and cost arithmetic reproducible? | 0-2 |
| Context | Is the context described and relevant? | 0-2 |

A score of 0 to 5 is very low certainty; 6 to 10 is low; 11 to 13 is moderate; 14 to 16 is high. The score is a communication device, not a substitute for judgement.

### Bias direction

| Bias | Mechanism | Likely direction | Check |
|---|---|---|---|
| Selection | Treated patients are healthier | Inflates effect | Baseline characteristics |
| Confounding | Another change explains the effect | Inflates effect | Timeline and control |
| Survivorship | Only successes are published | Inflates effect | Base rate |
| Publication | Positive results are published | Inflates effect | Funnel plot or registry |
| Measurement | Outcome definition changed | Either | Version log |
| Regression to the mean | Extreme units selected | Inflates effect | Control or RD |
| Attrition | High-risk patients lost | Either | Follow-up completeness |

### Effect arithmetic examples

| Claim | Baseline | Relative effect | Absolute effect | NNT |
|---|---:|---:|---:|---:|
| A | 5% | 39% | 1.95 pp | 51 |
| B | 14% | 39% | 5.46 pp | 18 |
| C | 30% | 9% | 2.70 pp | 37 |
| D | 50% | 9% | 4.50 pp | 22 |

The same relative effect produces very different absolute effects and NNTs. A vendor that reports only the relative effect is hiding the denominator.

### Context checklist

| Condition | Why it matters | Local status |
|---|---|---|
| Risk-bearing contract | The organisation captures the savings | Yes or no |
| Data infrastructure | The model needs timely data | Yes or no |
| Staffing model | The programme needs dedicated staff | Yes or no |
| Payer mix | The savings depend on who pays | Describe |
| Workflow authority | The team can change the process | Yes or no |
| Executive sponsor | The programme survives turnover | Yes or no |

### Questions for a vendor

1. What was the comparator, and was it contemporaneous?
2. What was the denominator, and what was the absolute effect?
3. What was the cost basis for the return on investment?
4. Which sites were included, and which were excluded?
5. Was the result peer-reviewed, and where?
6. What was the implementation cost and timeline?
7. What happens if the effect is half of the reported effect?

A vendor that cannot answer these questions has not provided evidence; it has provided a story.

### A mini-appraisal

A vendor reports that a sepsis alert reduced mortality by 20 percent at a 400-bed hospital. The case does not report the comparator, the baseline mortality, or the denominator. The appraisal score is 4 out of 16: population 1, intervention 1, comparator 0, outcome 1, design 0, bias 0, arithmetic 0, context 1. The certainty is very low. The recommendation is to request the underlying study, not to purchase.

### Common case types

| Case type | Typical claim | Typical design | Main weakness |
|---|---|---|---|
| Vendor success story | 30-50% improvement | Before-after | No comparator |
| Peer-reviewed single site | 10-20% improvement | ITS or DiD | Transferability |
| Multi-site evaluation | 5-10% improvement | Stepped wedge | Heterogeneity |
| Internal quality report | Local improvement | Before-after | No control |
| Cautionary case | Failure or harm | Case series | Anecdote |

### What a case can support

| Claim | Minimum evidence |
|---|---|
| The programme is feasible | A case with implementation detail |
| The programme was associated with improvement | A case with a comparator and a plausible design |
| The programme caused improvement | A trial or a strong quasi-experimental design |
| The programme will work locally | A local pilot or a replicated multi-site evaluation |
| The programme is cost-effective locally | Local cost and effect data |

The table is the answer to the question "how much can this case bear?" A vendor case can support feasibility; it cannot support a local cost-effectiveness claim.

A case that is used for a claim it cannot support is not evidence; it is rhetoric with a chart.

That distinction is the whole discipline of appraisal.

It is also the discipline that protects the organisation from a costly mistake.

A good appraisal is cheap relative to a bad purchase.

## 6. Worked example

A health system is considering a licensed readmission programme. The vendor cites the Kaiser Permanente Northern California Transitions Program, published in *BMJ* in 2021, as proof that analytics-targeted transitional care reduces readmissions. The system has 9,000 annual discharges and a 14 percent readmission rate. The appraisal asks whether the case supports a purchase.

### Part A: extract the case

| Element | What the case reports |
|---|---|
| Population | 21 hospitals, 1.54 million discharges, 2010-2018 |
| Intervention | EHR risk score at discharge; patients at 25% or higher risk auto-enrolled in telephone case management |
| Comparator | Difference-in-differences across hospitals and time; regression discontinuity at the threshold |
| Outcome | 30-day readmission or death |
| Effect | Adjusted odds ratio 0.91 (95% CI 0.89-0.93); absolute risk reduction about 2.5 percentage points; NNT about 40 |
| Design | Observational, difference-in-differences with a regression-discontinuity cross-check |
| Source | Peer-reviewed, *BMJ* 2021;374:n1747 |

The case is unusually well documented for an operational programme. It reports the population, the intervention, the comparator, the outcome, and two designs. It also reports a mismatch between predicted risk and treatment effect: higher-risk patients benefited less.

### Part B: verify the arithmetic

The headline is an odds ratio of 0.91, which sounds like a 9 percent reduction. The absolute risk reduction is about 2.5 percentage points. On a 14 percent baseline, 2.5 points is a relative reduction of 2.5 / 14 = 17.9 percent, not 9 percent. The odds ratio and the relative risk are not the same, and the absolute effect is the one that determines the NNT.

```text
NNT = 1 / 0.025 = 40
```

Forty patients must be enrolled for one readmission to be avoided. If the programme costs $400 per enrolled patient, the cost per avoided readmission is 40 x $400 = $16,000. If the value of an avoided readmission is $16,800, the net is $800 per avoided event.

### Part C: scale arithmetic

The case covers 1.54 million discharges over nine years. If 20 percent are above the risk threshold, the eligible population is 308,000. At a 2.5-point absolute effect, the avoided readmissions are 308,000 x 0.025 = 7,700 over the period.

```text
Programme cost = 308,000 x $400 = $123,200,000
Savings = 7,700 x $16,800 = $129,360,000
Net = $129,360,000 - $123,200,000 = $6,160,000
```

The net is positive but thin: about 5 percent of the savings, over nine years. A 20 percent error in the cost per patient or the effect would wipe it out.

### Part D: appraise the evidence

| Domain | Assessment | Score |
|---|---|---|
| Population | Large, multihospital, relevant to adult readmissions | 2 |
| Intervention | Described, including the risk threshold and the call model | 2 |
| Comparator | DiD and RD, both credible | 2 |
| Outcome | 30-day readmission or death, relevant | 2 |
| Design | Observational but strong | 1 |
| Bias | Confounding by concurrent programmes; risk-effect mismatch | 1 |
| Arithmetic | Reproducible from the reported effect and NNT | 2 |
| Context | Integrated system with a health plan | 1 |

The score is 13 out of 16, which corresponds to moderate certainty. The case is strong evidence that the programme worked in that system. It is weaker evidence that it will work in a standalone hospital without a plan, a mature data infrastructure, and an employed physician group.

### Part E: transferability

The mechanism is the combination of risk stratification, early telephone contact, and case management. The conditions that made it work include:

1. A risk score available at discharge.
2. A centralized call team with capacity.
3. A health plan that shared the savings.
4. An integrated system that could coordinate post-acute care.
5. A data infrastructure that could measure 30-day readmissions across hospitals.

The local system has the data and the call capacity but not the plan and not the integrated post-acute network. Two of the five conditions are missing. The expected effect is lower, and the financial case is weaker.

### Diagnostic 1: the relative effect is applied to the wrong baseline

A local analyst applies the 9 percent odds ratio to the local 14 percent readmission rate and estimates a 1.26-point absolute reduction. The published absolute reduction is 2.5 points. The analyst has confused an odds ratio with a relative risk and has understated the effect by half. The diagnostic is to convert the odds ratio to a risk ratio using the baseline, or to use the published absolute effect and NNT.

### Diagnostic 2: the cost per avoided event uses the wrong denominator

A second analyst divides the programme cost by all 9,000 discharges rather than by the enrolled patients. The cost per discharge is $400 x 9,000 / 9,000 = $400, and the cost per avoided readmission is $400 / 0.025 = $16,000, which happens to match. If the programme enrols only 20 percent of discharges, the cost per enrolled patient is $2,000, and the cost per avoided readmission is $80,000. The denominator must be the enrolled population, not the discharge population.

### Varying one assumption

The case's absolute effect is 2.5 points. If the local effect is 1.0 point because two conditions are missing, the NNT is 100 and the cost per avoided readmission at $400 per enrolled patient is $40,000, well above the $16,800 value.

| Scenario | Absolute effect | NNT | Cost per avoided at $400 | Net per avoided | Decision |
|---|---:|---:|---:|---:|---|
| Published | 2.5 pp | 40 | $16,000 | $800 | Marginal |
| Local, two conditions missing | 1.0 pp | 100 | $40,000 | -$23,200 | Reject |
| Local, one condition missing | 1.8 pp | 56 | $22,400 | -$5,600 | Reject |
| Lower programme cost | 2.5 pp | 40 | $10,000 | $6,800 | Fund |

The decision is sensitive to the effect and the cost. The published case supports a pilot, not a purchase.

### Part F: the recommendation

The recommendation is to pilot the programme in one service line with a local risk threshold, a comparison unit, a pre-specified 12-month outcome, and a go/no-go rule. The pilot should measure the enrolled population, the cost per enrolled patient, the absolute effect, and the NNT. If the local absolute effect is below 1.5 points or the cost per avoided readmission exceeds $16,800, the programme should not be scaled.

The appraisal does not say the vendor is wrong. It says the case is moderate-certainty evidence from a different context, and the local decision needs local evidence. That is the correct use of a case study: as a reason to test, not as a reason to buy.

### Part G: a second case for comparison

The Geisinger post-discharge monitoring programme is the other canonical US case. It combined interactive voice response monitoring with nurse case management for Medicare Advantage patients. The published evaluation reported a 44 percent relative reduction in 30-day readmissions in a pre-post matched-control design with 875 programme patients and 2,420 controls, at a programme cost of about $25 per patient.

| Case | Design | Relative effect | Baseline | Absolute effect | NNT | Cost per patient | Cost per avoided |
|---|---|---|---:|---:|---:|---:|---:|
| Kaiser Transitions | DiD and RD | About 9% (OR 0.91) | 14% | 2.5 pp | 40 | $400 | $16,000 |
| Geisinger IVR | Pre-post matched control | 44% | 18% (assumed) | 7.9 pp | 13 | $25 | $325 |

The two cases differ by a factor of 50 in cost per avoided readmission. The difference is explained by the intervention (telephone case management versus IVR monitoring), the population, the design, and the cost. A buyer who treats "readmission programme" as a single category will misjudge both.

The Geisinger case is also weaker on design: a pre-post matched-control study is susceptible to confounding and regression to the mean. Its large relative effect may partly reflect the selection of a high-risk population and the pre-period baseline. The case supports the feasibility of low-cost monitoring, not a 44 percent effect in a new setting.

### Part H: vendor claim versus peer review

A vendor case reports a 39 percent relative reduction in all-cause 30-day readmissions at a university health system after a machine-learning and care-redesign programme. The case is a vendor document, not a peer-reviewed study. It does not report the comparator, the baseline rate, the absolute effect, the NNT, or the cost per avoided readmission.

| Element | Vendor case | Peer-reviewed Kaiser case |
|---|---|---|
| Comparator | Not reported | DiD and RD |
| Baseline | Not reported | Reported |
| Absolute effect | Not reported | 2.5 pp |
| NNT | Not reported | 40 |
| Cost per avoided | Not reported | Computable |
| Independent review | No | Yes |

The vendor case can support a site visit or a pilot. It cannot support a purchase on the basis of the 39 percent claim, because the denominator, the comparator, and the cost are missing.

### Part I: what the case cannot support

The Kaiser case cannot support a claim that the programme will reduce readmissions by 2.5 points in a standalone hospital. It cannot support a claim that the effect persists beyond the study period. It cannot support a claim that the mechanism is the telephone call rather than the attention of the care team. It can support a claim that the programme is feasible at scale, that a risk-targeted call model was associated with a reduction, and that the effect is modest.

The distinction is the difference between evidence and marketing. The case is good evidence for a pilot; it is not evidence for a purchase.

### Part J: the appraisal's own limits

The appraisal relies on the published report. It does not have access to the underlying data, the programme manual, or the implementation costs. The cost per patient of $400 is an assumption, not a published figure; the case reports the effect, not the cost. The scale arithmetic is illustrative and should be replaced with local numbers before a decision.

The appraisal score is a communication device. A different reviewer might score the context domain differently, and the total would change. The score is not a substitute for the reasoning; it is a summary of it.

### Part K: appraising the Geisinger case

| Domain | Assessment | Score |
|---|---|---|
| Population | Medicare Advantage, case-managed | 1 |
| Intervention | IVR plus case management, described | 2 |
| Comparator | Matched controls, pre-post | 1 |
| Outcome | 30-day readmission | 2 |
| Design | Quasi-experimental | 1 |
| Bias | Selection and regression to the mean | 1 |
| Arithmetic | Relative effect reported; absolute derivable | 1 |
| Context | Integrated system with risk | 1 |

The score is 10 out of 16, low to moderate certainty. The case is useful for feasibility and for the low cost per patient, not for a precise effect estimate. Its 44 percent relative effect should not be used as a local planning assumption.

### Part L: the decision memo

The memo to the operating committee has four paragraphs:

1. The claim: the vendor cites a peer-reviewed case with a 0.91 odds ratio and a 2.5-point absolute reduction.
2. The appraisal: moderate certainty, strong design, integrated-system context, two of five local conditions missing.
3. The arithmetic: NNT 40, cost per avoided $16,000 at $400 per patient, net $800 per avoided, scale net $6.2 million over nine years.
4. The recommendation: pilot in one service line, with a 12-month outcome, a comparison unit, and a go/no-go rule at a 1.5-point absolute effect.

The memo does not say "the vendor is wrong" or "the case is irrelevant". It says the case supports a test, and the test will produce the local evidence the purchase decision needs.

A memo that cannot be summarised in four paragraphs is not ready for a committee.

## 7. Code, query, or calculation

The Python script below reproduces the appraisal arithmetic for a local population of 9,000 discharges, a 20 percent eligibility share, a 2.5-point absolute effect, a $400 cost per enrolled patient, and a $16,800 value per avoided readmission. It also produces the scenario table.

Assumptions:

- The 2.5-point effect is the published absolute effect, applied without adjustment.
- The eligible share is 20 percent of discharges.
- The cost per enrolled patient is $400, an assumption, not a published figure.
- The value per avoided readmission is $16,800.

```python
baseline = 0.14
arr = 0.025
cost_per_enrolled = 400.0
value_per_readmission = 16800.0
discharges = 9000
eligible_share = 0.20

def case(arr, cost):
  nnt = 1 / arr
  cpa = nnt * cost
  net = value_per_readmission - cpa
  eligible = discharges * eligible_share
  avoided = eligible * arr
  programme = eligible * cost
  savings = avoided * value_per_readmission
  return nnt, cpa, net, avoided, programme, savings, savings - programme

nnt, cpa, net, avoided, programme, savings, scale_net = case(
  arr, cost_per_enrolled)
print(f"NNT={nnt:.0f} cost_per_avoided={cpa:,.0f} "
   f"net_per_avoided={net:,.0f}")
print(f"eligible={discharges*eligible_share:.0f} avoided={avoided:.0f} "
   f"programme={programme:,.0f} savings={savings:,.0f} "
   f"scale_net={scale_net:,.0f}")

print("\nSCENARIOS")
for a in (0.025, 0.018, 0.010):
  for c in (400.0, 600.0):
    n, cp, npv, av, pr, sv, sn = case(a, c)
    print(f"arr={a:.3f} cost={c:.0f} nnt={n:.0f} cpa={cp:,.0f} "
       f"net_per={npv:,.0f} scale_net={sn:,.0f}")

print("\nRELATIVE EFFECT FROM ABSOLUTE")
print(f"relative={arr/baseline*100:.1f}%")
```

Expected output:

```text
NNT=40 cost_per_avoided=16,000 net_per_avoided=800
eligible=1800 avoided=45 programme=720,000 savings=756,000 scale_net=36,000

SCENARIOS
arr=0.025 cost=400 nnt=40 cpa=16,000 net_per=800 scale_net=36,000
arr=0.025 cost=600 nnt=40 cpa=24,000 net_per=-7,200 scale_net=-324,000
arr=0.018 cost=400 nnt=56 cpa=22,222 net_per=-5,422 scale_net=-175,680
arr=0.018 cost=600 nnt=56 cpa=33,333 net_per=-16,533 scale_net=-535,680
arr=0.010 cost=400 nnt=100 cpa=40,000 net_per=-23,200 scale_net=-417,600
arr=0.010 cost=600 nnt=100 cpa=60,000 net_per=-43,200 scale_net=-777,600

RELATIVE EFFECT FROM ABSOLUTE
relative=17.9%
```

Read the output in three passes. The first line converts the 2.5-point absolute effect into an NNT of 40 and a cost per avoided readmission of $16,000, which is $800 below the $16,800 value. The second line scales to 1,800 eligible patients, 45 avoided readmissions, $720,000 of programme cost, $756,000 of savings, and a net of $36,000 per year. The scenario block shows that the net turns negative if the effect falls to 1.8 points or the cost rises to $600 per patient.

> **Note:** The scale in Section 6 used the published 1.54 million discharges over nine years, which produced 308,000 eligible patients and 7,700 avoided readmissions. The script uses the local 9,000 annual discharges, which produces 1,800 eligible and 45 avoided. The two are consistent applications of the same formula to different populations; the script is the local arithmetic.

A common error is to treat the NNT as a fixed property of the programme. The NNT depends on the baseline: the same relative effect produces a different NNT on a different baseline. The appraisal should report the baseline alongside the NNT, and should recompute the NNT for the local baseline.

### Relative versus absolute

The relative effect is 2.5 / 14 = 17.9 percent, not 9 percent. The odds ratio of 0.91 is not a relative risk, and applying it directly to the baseline understates the effect. The script prints the relative effect so that the two scales are visible in one place.

The script is deliberately simple: it makes the arithmetic auditable and shows how sensitive the conclusion is to the effect and the cost. A vendor case that reports only the headline cannot survive this arithmetic.

### A SQLite evidence table

The appraisal can be stored in a small table so that multiple cases can be compared:

```sql
CREATE TABLE case_appraisal (
 case_name TEXT, source TEXT, design TEXT,
 population INTEGER, baseline REAL, absolute_effect REAL,
 nnt REAL, cost_per_patient REAL, cost_per_avoided REAL,
 certainty TEXT
);
INSERT INTO case_appraisal VALUES
 ('Kaiser Transitions','BMJ 2021','DiD and RD',1540000,0.14,0.025,40,400,16000,'moderate'),
 ('Geisinger IVR','Medical Care 2011','pre-post matched',875,0.18,0.079,13,25,325,'low');
SELECT case_name, design, nnt, cost_per_avoided, certainty
FROM case_appraisal ORDER BY cost_per_avoided;
```

The table makes the comparison explicit: the two cases differ by a factor of 50 in cost per avoided readmission, and the certainty differs. A buyer who averages them is averaging incomparable interventions.

### PostgreSQL dialect: case evidence register

The following block is **PostgreSQL** syntax. It stores the appraisal with a JSON evidence field for the domain scores.

```sql
-- PostgreSQL 15+
CREATE TABLE case_evidence (
 case_id SERIAL PRIMARY KEY,
 case_name TEXT NOT NULL,
 source_class TEXT NOT NULL,
 design TEXT NOT NULL,
 absolute_effect NUMERIC,
 nnt NUMERIC,
 cost_per_avoided NUMERIC,
 domain_scores JSONB,
 certainty TEXT
);

INSERT INTO case_evidence
(case_name, source_class, design, absolute_effect, nnt, cost_per_avoided,
 domain_scores, certainty)
VALUES
('Kaiser Transitions', 'peer-reviewed', 'DiD and RD', 0.025, 40, 16000,
 '{"population":2,"intervention":2,"comparator":2,"outcome":2,"design":1,"bias":1,"arithmetic":2,"context":1}',
 'moderate');
```

The JSONB field allows the domain scores to be stored without a separate table. A query can filter on `certainty` and order by `cost_per_avoided` to produce a shortlist.

## 8. Data, standards, and regulatory context

Case appraisal is governed by evidence standards and by disclosure rules. The standards tell you how to judge a study; the disclosure rules tell you what the author must reveal.

### Evidence and reporting standards

| Standard | What it covers | Binding status | What it does not cover |
|---|---|---|---|
| GRADE | Certainty of evidence | Guideline | Study conduct |
| ROBINS-I | Risk of bias in non-randomised studies | Guideline | Randomised trials |
| CONSORT | Reporting of randomised trials | Journal requirement | Observational studies |
| TREND | Reporting of non-randomised evaluations | Guideline | Randomised trials |
| STROBE | Reporting of observational studies | Guideline | Intervention effects |
| SQUIRE | Reporting of quality improvement | Guideline | Causal inference |
| PRISMA | Reporting of systematic reviews | Journal requirement | Primary studies |
| Peer review | Independent expert review | Journal and funder policy | Data availability |

The standards are not interchangeable. GRADE rates the certainty of a body of evidence; ROBINS-I assesses the risk of bias in a single non-randomised study; CONSORT and TREND specify what a report must contain. A case appraisal should name the tool it uses and apply it consistently.

### Disclosure and marketing rules

**Vendor claims.** In the United States, the Federal Trade Commission Act prohibits deceptive advertising, and the Food and Drug Administration regulates promotional claims for regulated products. A vendor case that makes a false or misleading claim about a health outcome can be subject to enforcement. The analyst should not assume that a vendor case is independently verified; the disclosure rules are a floor, not a guarantee.

**Peer review.** Peer review is a quality filter, not a guarantee. It checks the methods and the reporting, not the data. A peer-reviewed case can still have a flawed design or an unreported conflict of interest.

**Conflict of interest.** Journals require authors to disclose funding and conflicts. A case written by the vendor's employees or funded by the vendor has a conflict that should be considered in the appraisal. The disclosure is not proof of bias, but it is a reason to scrutinise the comparator and the outcome.

> **Note:** Regulatory citations are dated to 25 October 2026. Advertising and promotional rules vary by jurisdiction and product. This section is educational and is not legal advice.


### Binding versus convention

| Element | Binding? | Source | Local choice |
|---|---|---|---|
| Trial reporting | Journal requirement | CONSORT | Design detail |
| Observational reporting | Guideline | STROBE | Design detail |
| Certainty rating | Guideline | GRADE | Domain scores |
| Risk-of-bias assessment | Guideline | ROBINS-I | Domains |
| Vendor advertising | Legal | FTC and FDA | Claim substantiation |
| Peer review | Journal policy | Publisher | None |

### What the standards do not settle

1. **The transferability judgement.** No standard says whether a case applies locally.
2. **The cost basis.** No standard requires a cost per avoided event.
3. **The base rate of success.** No standard reports the failures that were not published.
4. **The local decision rule.** No standard says when to adopt, pilot, or reject.
5. **The weight of a vendor case.** No standard assigns a numerical weight to marketing evidence.

## 9. Industry practice and case evidence

### Documented cases

**Kaiser Permanente Northern California Transitions Program.** Marafino et al., *BMJ* 2021;374:n1747. Across 21 hospitals and 1.54 million discharges, patients with an EHR-based 30-day readmission or death risk of 25 percent or more were auto-enrolled in telephone case management. The difference-in-differences estimate was an adjusted odds ratio of 0.91 (95 percent CI 0.89-0.93), an absolute risk reduction of about 2.5 percentage points, and an NNT of about 40; a regression-discontinuity analysis produced an absolute risk reduction of 2.7 points. The authors reported that higher-risk patients benefited less, which is a caution against assuming that the top-ranked patients are the highest-benefit patients.

**Geisinger post-discharge monitoring.** Graham et al., *Medical Care* 2011. An IVR telemonitoring programme layered on nurse case management for Medicare Advantage patients reported a 44 percent relative reduction in 30-day readmissions (95 percent CI 23-60 percent) in a pre-post parallel quasi-experimental design with 875 programme patients and 2,420 matched controls, at a cost of about $25 per patient. The design is weaker than the Kaiser DiD, and the effect is much larger, which suggests that the intervention, the population, or the design explains the difference.

**IBM Watson for Oncology.** STAT News, July 2018, based on internal IBM documents. The reporting described unsafe or incorrect treatment recommendations, training on a small number of synthetic cases, and recommendations that reflected the preferences of a few clinicians rather than guidelines. The case is a cautionary example of a product whose validation claims outran its evidence. It is not an operational case, but it is the canonical public record of an analytics product failing an evidence test.

**Discharge-before-noon programmes.** Artenstein et al., 2017, reported a 0.30-day reduction in length of stay and a 2.1-hour reduction in boarding. Kausar et al., 2023, reported an increase in the share of morning discharges but no independent association with length of stay or readmissions and a 41-minute increase in median boarding. Hodges et al., 2024, reported longer order-to-discharge times and increased boarding after a "discharge by 11 a.m." programme. The mixed results are a case study in why a single-site before-after cannot settle an operational question.

**Vendor-reported readmission reduction.** A Health Catalyst success story reports a 39 percent relative reduction in all-cause 30-day readmissions at the University of Kansas Health System after machine learning and process redesign. The document is a vendor case, not a peer-reviewed study, and it does not report the comparator, the baseline, the absolute effect, or the cost per avoided readmission. It is useful as an existence proof and as a description of a deployment, not as an effect estimate.


### Evidence table

| Case | Source | Design | Effect | Certainty |
|---|---|---|---|---|
| Kaiser Transitions | BMJ 2021 | DiD and RD | OR 0.91; ARR 2.5 pp | Moderate |
| Geisinger IVR | Medical Care 2011 | Pre-post matched | 44% relative | Low |
| Watson for Oncology | STAT 2018 | Investigative | Unsafe recommendations | Not applicable |
| Discharge before noon | Multiple | Pre-post | Mixed, some negative | Low |
| Vendor readmission case | Health Catalyst | Vendor | 39% relative | Very low |

The table separates the cases by source and design. The certainty column is the appraisal's summary, not the authors' claim.

### What the evidence means

The peer-reviewed cases support a modest effect for analytics-targeted transitional care (Kaiser) and a larger effect for a low-cost monitoring programme (Geisinger), with different designs and different contexts. The vendor cases support feasibility, not effect size. The cautionary cases (Watson, discharge-before-noon) show that a plausible intervention can fail or produce unintended consequences.

> **Inference:** A local business case should not use the largest published relative effect. It should use the most credible absolute effect, adjusted for the local context, and should test it in a pilot. The Kaiser case is the most credible for a readmission programme; the Geisinger case is the most credible for a low-cost monitoring programme.

### What mature organisations do

They maintain an evidence register with the case, the source class, the design, the effect, the context, and the appraisal score. The register is reviewed when a vendor proposes a programme, and it prevents the organisation from re-litigating the same case for every purchase.

They separate the vendor's claim from the peer-reviewed evidence. The vendor's claim is recorded as a claim; the peer-reviewed study is appraised on its own terms. A vendor that cites a peer-reviewed study is asked for the citation, and the study is read.

They pilot before they scale. The pilot has a pre-specified outcome, a comparison, and a go/no-go rule. The appraisal tells them what to measure; the pilot tells them whether the local effect is large enough.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Vendor case treated as evidence | No comparator or denominator | Marketing selection | Wrong purchase | Appraise the source |
| Relative effect only | "39% reduction" with no baseline | Denominator hidden | Effect overstated | Compute absolute effect and NNT |
| Survivorship sample | Only successes are cited | Failures unpublished | Base rate unknown | State the limitation |
| No cost basis | ROI without allowed/paid/actual | Cost basis omitted | ROI uninterpretable | Ask for the basis |
| Short horizon | 3-month follow-up | Early effect | Sustainability unknown | Ask for 12-24 months |
| Subgroup selected | Headline from a subgroup | Post-hoc selection | Effect overstated | Ask for overall result |
| Context ignored | Brand transferred without workflow | Mechanism missing | Implementation fails | Assess conditions |
| Arithmetic error | NNT inconsistent with effect | Denominator confusion | Cost per avoided wrong | Recompute |
| Publication bias | Literature all positive | Journals prefer positive | Effect overstated | Look for registries |
| Conflict of interest | Vendor-funded study | Sponsor influence | Bias | Disclose and scrutinise |
| Outcome changed | Primary outcome differs from protocol | Post-hoc choice | False positive | Check pre-registration |
| Scale assumed | Local population not tested | Transferability assumed | Budget wrong | Pilot |
| Regression to the mean | Site selected for poor performance | Extreme baseline | False improvement | Control or RD |
| Concurrent intervention | Other programmes at the same time | Confounding | Effect misattributed | Timeline |
| No exit plan | Contract locks in | Switching cost | Stranded spend | Negotiate exit |

Three failures deserve emphasis.

**The relative-effect headline.** A 39 percent reduction sounds large. On a 5 percent baseline it is 1.95 points and an NNT of 51; on a 30 percent baseline it is 11.7 points and an NNT of 9. The same headline describes very different programmes. The appraisal must recover the baseline.

**The missing denominator.** A vendor that reports recovered dollars without the number of claims, or a readmission reduction without the number of discharges, has not provided an effect. The denominator is the first question in the appraisal.

**The context transfer.** A programme that depends on a health plan, a data infrastructure, or an employed physician group may not work without them. The appraisal must list the conditions and test them locally.


### Pre-mortem for a case appraisal

1. What is the source class?
2. Is there a comparator?
3. What is the baseline and the absolute effect?
4. What is the cost basis?
5. What conditions made the case work?
6. What would make the local effect half as large?

A case that cannot answer all six is not ready for a purchase decision.
### What a decision committee should ask

| Question | Why it matters |
|---|---|
| What is the comparator? | The effect depends on it |
| What is the absolute effect? | The NNT depends on it |
| What is the cost basis? | The ROI depends on it |
| What conditions made it work? | Transferability depends on them |
| What would make the local effect half? | The pilot should test it |
| What is the exit cost? | The downside is not only the fee |

> **Note:** A case study is not disqualified by a weak design. It is disqualified from a causal claim. The same case can be useful for feasibility, for implementation detail, or for a pilot design.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The case is reported accurately | The appraisal depends on the report | Selective reporting | Request the underlying data |
| The comparator is credible | The effect depends on it | No comparator | Read the methods |
| The outcome is relevant | The local decision depends on it | Different definition | Compare definitions |
| The effect is absolute | The NNT depends on it | Relative only | Recompute from counts |
| The cost basis is stated | The ROI depends on it | Unclear basis | Ask finance |
| The context is described | Transferability depends on it | Brand-only case | Site visit |
| The population is similar | The effect may transfer | Different case mix | Compare characteristics |
| The implementation is feasible | The effect depends on fidelity | Staffing or data gaps | Pilot |
| The base rate of success is unknown | The decision needs it | Survivorship | State the limitation |
| The appraisal is independent | The judgement depends on it | Vendor authorship | Disclose and review |


### Assumption owners

| Assumption | Owner | Review cadence |
|---|---|---|
| Source classification | Analytics | At appraisal |
| Effect arithmetic | Analytics | At appraisal |
| Cost basis | Finance | At appraisal |
| Context conditions | Operations | At appraisal |
| Transferability | Clinical and operations | Before pilot |
| Pilot rule | Steering committee | At protocol |

The limits follow from the assumptions. A case study is a selected observation, and its evidence is bounded by the selection, the design, and the reporting. It can tell you that a result was observed somewhere; it cannot tell you the probability that you will reproduce it.

A second limit is that the appraisal itself is judgement. Two reviewers can score the same case differently, especially on context and transferability. The score is a communication device, and the reasoning should be recorded so that a disagreement can be located.

A third limit is that the evidence base is biased toward success. The failures are rarely published, and the base rate is unknown. A decision that relies on a collection of successful cases is relying on a selected sample.

## 12. Exercises

### Exercise 1. Classify the source (recall)

For each source, state whether it is a vendor case, peer-reviewed evaluation, preprint, internal report, or news, and the weight you would give it:

1. A product brochure with a customer quote and a 40 percent improvement claim.
2. A *BMJ* paper with a difference-in-differences design.
3. A medRxiv manuscript with no peer review.
4. A hospital's internal quality report with a before-after comparison.
5. A trade magazine article summarizing a vendor's case.

**Deliverable:** a five-row table with source class and weight.

**Acceptance criteria:** each source is classified; the weight is justified; at least one source is classified as low weight.

**Time:** 30 minutes.

### Exercise 2. Extract the case (application)

Using the Kaiser case in Section 6, extract the population, intervention, comparator, outcome, design, and effect. State one element that is missing or uncertain.

**Deliverable:** a six-row extract with the source location.

**Acceptance criteria:** all six elements are present; the missing element is identified; the effect includes both relative and absolute scales.

**Time:** 45 minutes.

### Exercise 3. Compute the absolute effect and NNT (application)

A case reports a 30 percent relative reduction in readmissions on a 12 percent baseline. Compute the absolute effect, the NNT, and the cost per avoided readmission at $500 per enrolled patient. If the value per avoided readmission is $16,800, compute the net per avoided event.

**Deliverable:** a calculation table.

**Acceptance criteria:** the absolute effect is computed from the baseline; the NNT is the reciprocal; the cost uses the NNT.

**Time:** 45 minutes.

### Exercise 4. Check a return on investment (application)

A vendor claims a 4:1 return: a $500,000 programme produces $2,000,000 in savings. The savings are denied dollars, and the collection rate is 60 percent. Recompute the net savings and the ROI. State whether the claim is supported.

**Deliverable:** a calculation and a one-paragraph judgement.

**Acceptance criteria:** the collection rate is applied; the ROI uses net savings; the judgement distinguishes gross from net.

**Time:** 45 minutes.

### Exercise 5. Appraise a vendor case (critique)

A vendor case reports a 39 percent reduction in readmissions, a 4:1 ROI, and a 6-month deployment. It does not report the comparator, the baseline, the absolute effect, or the cost basis. Write a structured appraisal with a score and a recommendation.

**Deliverable:** a one-page appraisal with the eight domains and a recommendation.

**Acceptance criteria:** at least three domains score 0 or 1; the recommendation is pilot or reject; the missing information is listed.

**Time:** 60 minutes.

### Exercise 6. Compare two cases (critique)

Compare the Kaiser Transitions case and the Geisinger IVR case on design, population, effect, cost, and transferability. Explain why the cost per avoided readmission differs by a factor of about 50.

**Deliverable:** a comparison table and a one-page analysis.

**Acceptance criteria:** at least five dimensions; the difference is attributed to intervention, population, design, and cost; the transferability judgement is explicit.

**Time:** 60 minutes.

### Exercise 7. Design a local pilot (design)

Design a 12-month pilot of a readmission programme in one service line. Specify the population, the risk threshold, the intervention, the comparator, the primary outcome, the sample size or power consideration, the go/no-go rule, and the cost per avoided event.

**Deliverable:** a two-page pilot protocol.

**Acceptance criteria:** the comparator is credible; the primary outcome is pre-specified; the go/no-go rule is explicit; the cost is included.

**Time:** 90 minutes.

### Exercise 8. Write a decision memo (design)

Write a one-page memo to an operating committee recommending pilot, adopt, or reject a programme based on a case appraisal. Include the claim, the evidence, the arithmetic, the context, and the recommendation.

**Deliverable:** a one-page memo with four paragraphs.

**Acceptance criteria:** the claim and the evidence are separated; the arithmetic is reproducible; the recommendation follows from the certainty and the stakes.

**Time:** 60 minutes.


### Assessment notes

The exercises are designed to be completed in sequence: classify, extract, compute, critique, design. A learner who can do Exercise 4 can spot a denominator error in a vendor deck; a learner who can do Exercise 7 can design a pilot that would settle the question.


Exercises 1 to 4 can be completed individually and checked against Section 13. Exercises 5 and 6 are critique tasks; Exercises 7 and 8 are design tasks assessed with the rubric. Add 20 minutes to Exercise 7 for learners who need to build the cost table.

## 13. Self-check answers

### Exercise 1

| Source | Class | Weight | Reason |
|---|---|---|---|
| Product brochure | Vendor | Low | Marketing, selected metrics |
| BMJ paper | Peer-reviewed | High | Independent review, DiD |
| medRxiv manuscript | Preprint | Low to moderate | Not reviewed |
| Internal quality report | Internal | Moderate for local | Honest but not independent |
| Trade article | News | Very low | Secondary summary |

The trade article is the weakest because it is a summary of a vendor case, with two layers of selection.

### Exercise 2

| Element | Value |
|---|---|
| Population | 21 hospitals, 1.54 million discharges, 2010-2018 |
| Intervention | EHR risk at discharge; 25% or higher risk auto-enrolled in telephone case management |
| Comparator | Difference-in-differences across hospitals and time; regression discontinuity |
| Outcome | 30-day readmission or death |
| Design | Observational, DiD with RD cross-check |
| Effect | OR 0.91 (0.89-0.93); ARR about 2.5 pp; NNT about 40 |

The uncertain element is the programme cost per enrolled patient; the paper reports the effect, not the cost. The cost used in this topic is an assumption.

### Exercise 3

```text
Absolute effect = 12% x 30% = 3.6 percentage points
NNT = 1 / 0.036 = 27.8, rounded to 28
Cost per avoided = 28 x $500 = $14,000
Net per avoided = $16,800 - $14,000 = $2,800
```

The NNT is 28, and the programme is marginally positive at a $500 cost per patient. A cost of $600 would make it negative.

### Exercise 4

```text
Gross savings = $2,000,000
Net savings at 60% collection = $1,200,000
Net benefit = $1,200,000 - $500,000 = $700,000
ROI = $700,000 / $500,000 = 1.4:1
```

The claimed 4:1 ROI is not supported. The gross savings are 4:1, but the net savings after the collection rate are 1.4:1. The vendor has used gross denied dollars as if they were cash.

### Exercise 5 rubric

A strong appraisal scores the case low on comparator, design, arithmetic, and context, and recommends a pilot or rejection. It lists the missing information: the comparator, the baseline, the absolute effect, the cost basis, the implementation cost, and the peer-reviewed citation. It does not reject the programme outright; it rejects the claim as evidence for a purchase.

A weak appraisal repeats the 39 percent and the 4:1 ROI without recomputing them.

### Exercise 6

| Dimension | Kaiser | Geisinger |
|---|---|---|
| Design | DiD and RD | Pre-post matched control |
| Population | 1.54M discharges, 21 hospitals | 875 programme, 2,420 controls |
| Intervention | Telephone case management | IVR plus case management |
| Effect | OR 0.91; ARR 2.5 pp | 44% relative |
| Cost per patient | Assumed $400 | About $25 |
| Cost per avoided | About $16,000 | About $325 |

The cost per avoided differs by a factor of about 50 because the Geisinger intervention is much cheaper per patient and the relative effect is much larger. The larger effect may reflect the high-risk population, the pre-post design, or the intervention intensity. The comparison shows that "readmission programme" is not a single category.

### Exercises 7 and 8 rubric

Exercise 7 is assessed on the population, the risk threshold, the intervention, the comparator, the primary outcome, the power consideration, the go/no-go rule, and the cost per avoided event. A strong pilot uses a comparison unit and a pre-specified absolute effect threshold.

Exercise 8 is assessed on the separation of claim and evidence, the arithmetic, the context, and the recommendation. A strong memo says what the case supports and what it does not, and recommends a pilot with a decision rule.


A note on precision: the NNT is rounded to the nearest whole patient, but the cost per avoided event should be computed from the unrounded NNT. At a 2.5-point effect, the NNT is 40.0; at 1.8 points, it is 55.6, and the cost per avoided at $400 is $22,222, not $22,400.

## 14. Interview questions

**1. How do you appraise a vendor case study?**
A strong answer classifies the source, extracts the population, intervention, comparator, outcome, and design, checks the arithmetic, assesses bias and transferability, and recommends based on certainty and stakes. A weak answer reads the headline and the ROI.

**2. Why is a relative reduction not enough?**
A strong answer explains that the absolute effect and NNT depend on the baseline, and gives an example where the same relative effect produces different NNTs. A weak answer accepts the percentage.

**3. What is survivorship bias?**
A strong answer explains that only successful cases are visible and the base rate is unknown. A weak answer confuses it with selection of patients.

**4. What is the difference between a peer-reviewed study and a vendor case?**
A strong answer explains independent review, disclosure, and data access, and notes that peer review is a filter, not a guarantee. A weak answer says peer review makes it true.

**5. How do you assess transferability?**
A strong answer lists the conditions that made the case work (data, staff, payer, workflow, incentives) and tests whether they hold locally. A weak answer says the programme is generic.

**6. How do you compute a cost per avoided event from a case?**
A strong answer uses the absolute effect and the population, or the NNT and the cost per patient, and states the cost basis. A weak answer divides the programme cost by the number of patients.

**7. What is a go/no-go rule?**
A strong answer is a pre-specified effect threshold and a decision to scale, redesign, or stop. A weak answer is a target set after the pilot.

**8. When would you reject a case outright?**
A strong answer rejects a causal claim when there is no comparator, but may still use the case for feasibility. A weak answer rejects the programme because the case is weak.

**9. What is publication bias?**
A strong answer explains that positive results are more likely to be published, and that a literature of successes overstates the effect. A weak answer confuses it with peer review.

**10. How do you compare two cases with different effects?**
A strong answer compares design, population, intervention, cost, and context, and explains the difference rather than averaging. A weak answer picks the larger effect.

**11. What would you ask a vendor before a site visit?**
A strong answer asks for the comparator, denominator, absolute effect, cost basis, peer-reviewed citation, implementation cost, and exit terms. A weak answer asks for a demo.

**12. What makes a case study useful despite its limits?**
A strong answer says it can show feasibility, implementation detail, and mechanism, and can justify a pilot. A weak answer says case studies are useless.


**13. What is the difference between internal and external validity?**
A strong answer says internal validity is whether the study estimated the effect correctly, and external validity is whether the effect applies elsewhere. A weak answer uses the terms interchangeably.

**14. Why is a vendor case still useful?**
A strong answer says it can show feasibility, implementation detail, and the shape of a deployment, and can justify a pilot. A weak answer says it is useless because it is marketing.

## 15. Key terms

- **Case study:** a descriptive account of an intervention and its results.
- **Vendor case:** a case published by a product vendor.
- **Peer-reviewed evaluation:** a study reviewed by independent experts.
- **Preprint:** a manuscript posted before peer review.
- **Internal report:** an organisation's own evaluation.
- **Claim:** a statement about an effect or result.
- **Design:** the structure of the comparison.
- **Comparator:** the group or period against which the intervention is compared.
- **Counterfactual:** the outcome without the intervention.
- **Effect size:** the magnitude of the effect.
- **Absolute effect:** the difference in rates.
- **Relative effect:** the ratio of rates.
- **NNT:** number needed to treat.
- **Cost per avoided event:** programme cost divided by events avoided.
- **ROI:** return on investment.
- **Internal validity:** whether the study estimates the effect correctly.
- **External validity:** whether the effect applies elsewhere.
- **Transferability:** whether the intervention can work in a new context.
- **Selection bias:** the sample differs from the population.
- **Survivorship bias:** only successful cases are visible.
- **Publication bias:** positive results are more likely to be published.
- **Confounding:** another factor explains the effect.
- **Secular trend:** a change over time unrelated to the intervention.
- **Concurrent intervention:** another change at the same time.
- **Regression to the mean:** extreme values revert.
- **Fidelity:** the degree to which the programme was delivered as designed.
- **Implementation:** the process of delivering the programme.
- **Mechanism:** the causal pathway.
- **Context:** the setting, population, and system.
- **Scalability:** the ability to expand the programme.
- **Sustainability:** the ability to maintain the effect.
- **Replication:** a repeat of the study in a new setting.
- **Generalisability:** the extent to which findings apply broadly.
- **GRADE:** a framework for rating evidence certainty.
- **ROBINS-I:** a tool for assessing bias in non-randomised studies.
- **Risk of bias:** the likelihood that the estimate is distorted.
- **Confidence interval:** the uncertainty around an estimate.
- **Statistical significance:** the probability of the data under the null.
- **Practical significance:** whether the effect matters.
- **Cost basis:** whether cost is allowed, paid, or actual.
- **Time horizon:** the period over which benefits accrue.
- **Attribution:** the assignment of the effect to the intervention.
- **Benchmark:** the comparison value.
- **Denominator:** the population at risk.
- **Subgroup:** a subset of the population.
- **Headline effect:** the effect reported in the abstract.
- **Primary outcome:** the pre-specified outcome.
- **Sensitivity analysis:** a re-analysis under different assumptions.
- **Pre-registration:** a plan written before the study.
- **Replication crisis:** the failure of many published results to replicate.
- **Due diligence:** the process of verifying a claim.
- **Pilot:** a small-scale local test.
- **Go/no-go:** a decision rule for scaling.
- **Site visit:** a visit to observe the programme in operation.
- **Reference call:** a call with a customer site.
- **Data room:** a repository of evidence for due diligence.
- **Claim substantiation:** the evidence a vendor must hold for an advertising claim.
- **Comparable:** a similar organisation used for benchmarking.
- **Base rate:** the proportion of similar programmes that succeed.
- **Funnel plot:** a graph used to detect publication bias.
- **Registry:** a database of studies, including unpublished ones.


## 16. Sources

All URLs were accessed on 25 October 2026.

### Peer-reviewed cases

1. Marafino, B. J. et al. Evaluation of an intervention targeted with predictive analytics to prevent readmissions in an integrated health system: observational study. *BMJ* 2021;374:n1747. https://www.bmj.com/content/374/bmj.n1747
2. Graham, J. et al. Postdischarge monitoring using interactive voice response system reduces 30-day readmission rates in a case-managed Medicare population. *Medical Care*, 2011. DOI 10.1097/MLR.0b013e318229433e
3. Artenstein, A. W. et al. Decreasing emergency department walkout rate and boarding hours by improving inpatient length of stay. *Western Journal of Emergency Medicine*, 2017. https://pubmed.ncbi.nlm.nih.gov/29085527/
4. Kausar, S. et al. Implementing and evaluating a discharge before noon initiative in a large tertiary care urban hospital. *Joint Commission Journal on Quality and Patient Safety*, 2023. https://www.jointcommissionjournal.com/article/S1553-7250(23)00212-X/abstract
5. Hodges, J. et al. Driving in the wrong direction: exploring the unintended consequences of an early discharge program. *Quality Management in Health Care*, 2024. https://doi.org/10.1097/qmh.0000000000000466
6. Kansagara, D. et al. Risk prediction models for hospital readmission: a systematic review. *JAMA*, 2011. https://doi.org/10.1001/jama.2011.1515

### Cautionary case

7. KFF Health News. Supercomputer Watson gave erroneous and unsafe cancer treatment advice for patients, July 2018. https://kffhealthnews.org/morning-breakout/supercomputer-watson-gave-erroneous-and-unsafe-cancer-treatment-advice-for-patients/
8. The Verge. IBM's Watson gave unsafe recommendations for treating cancer, July 2018. https://www.theverge.com/2018/7/26/17619382/ibms-watson-cancer-ai-healthcare-science

### Vendor case

9. Health Catalyst. Machine learning, predictive analytics, and process redesign reduces readmission rates by 50 percent. https://downloads.healthcatalyst.com/success-stories/Machine-Learning-Predictive-Analytics-and-Process-Redesign-Reduces-Readmission-Rates-by-50-Percent.pdf

### Appraisal standards

10. GRADE. Grading of Recommendations Assessment, Development and Evaluation. https://www.gradeworkinggroup.org/
11. ROBINS-I. Risk of Bias in Non-randomised Studies of Interventions. https://www.riskofbias.info/
12. CONSORT. Transparent reporting of trials. https://www.consort-statement.org/
13. TREND. Transparent reporting of evaluations with nonrandomized designs. https://www.cdc.gov/trendstatement/
14. STROBE. Strengthening the reporting of observational studies in epidemiology. https://www.strobe-statement.org/
15. SQUIRE. Standards for quality improvement reporting excellence. https://www.squire-statement.org/
16. PRISMA. Preferred reporting items for systematic reviews and meta-analyses. https://www.prisma-statement.org/

### Policy and context

17. Federal Trade Commission. Advertising and marketing basics. https://www.ftc.gov/business-guidance/advertising-marketing
18. US Food and Drug Administration. Prescription drug advertising and promotional labeling. https://www.fda.gov/drugs/information-drug-class/prescription-drug-advertising-and-promotional-labeling
19. CMS. Quality Measures. https://www.cms.gov/medicare/quality/measures
20. AHRQ. Healthcare Cost and Utilization Project. https://hcup-us.ahrq.gov/

> **Note:** Vendor cases are cited here as examples of the genre, not as evidence of effect. The Health Catalyst document is a marketing publication and has not been peer reviewed.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`08_evaluating-operational-interventions.md`](08_evaluating-operational-interventions.md) — ITS, DiD, and stepped-wedge designs.
- Related topic: [`04_operational-models.md`](04_operational-models.md) — the models these programmes deploy.
- Related topic: [`07_revenue-cycle.md`](07_revenue-cycle.md) — the cash arithmetic behind a business case.
- Root survey file: [`../../08_ehr_claims_operations.md`](../../course/08_ehr_claims_operations.md)
- Next module: [`../../09_clinical_nlp.md`](../../course/09_clinical_nlp.md) — clinical natural language processing.
- Labs: [`../../labs/`](../../labs/)

### Reading path

| If you need to | Read | Then |
|---|---|---|
| Appraise a vendor proposal | Sections 5 and 6 | The GRADE and ROBINS-I sources |
| Compare two cases | Sections 6 and 9 | The peer-reviewed studies |
| Design a pilot | Sections 5 and 12 | Topic 08 for the design |
| Write a decision memo | Sections 6 and 12 | Topic 01 for the cost basis |
| Build an evidence register | Sections 5 and 7 | The SQLite example |
| Judge a vendor claim | Sections 5 and 10 | The FTC guidance |
| Compare a case with a trial | Sections 3 and 9 | The GRADE handbook |

The case study is the last step in the module's arc: the operating model in Topic 01, the population in Topic 02, the cost in Topic 03, the operational model in Topic 04, the capacity in Topic 05, the measure in Topic 06, the cash in Topic 07, and the evaluation in Topic 08 all come together in the appraisal of a public case. The appraisal is where the analyst decides how much weight the evidence can bear, and the decision is where the organisation acts.

The final test of the module is whether the reader can read a case, separate the claim from the evidence, compute the local arithmetic, and recommend a pilot rather than a purchase.

The case is the beginning of the decision, not the end of it.

| Decision | Evidence needed | Local test |
|---|---|---|
| Adopt | High certainty, low cost | Monitoring |
| Pilot | Moderate or low certainty | 12-month comparison |
| Reject | High certainty of no effect | None |
| Gather more evidence | Very low certainty | Data request |

The table is the decision rule in its simplest form. The appraisal fills in the certainty, and the local constraints fill in the cost.

A committee that asks for the comparator and the absolute effect will make better decisions than one that asks for the headline.

That is the practical contribution of this topic.

It is also the contribution that survives a change of vendor, a change of leadership, and a change of fashion.

The arithmetic does not.

The arithmetic is the part a new leader can check.

That is why it belongs in the memo.
