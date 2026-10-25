# Evidence Synthesis

> Module 05 — Study Design and Evidence Generation
> Topic 09 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 05 and 08
> Version 2026.3

## 1. Why this topic exists

No single study is the evidence base. A regulator, payer, guideline panel, or clinician making a decision needs to know what the whole body of evidence says, how consistent it is, and how certain it is. Evidence synthesis is the discipline that answers those questions. It ranges from a systematic review of the literature to a meta-analysis, an indirect treatment comparison, or a network meta-analysis (NMA).

The commercial importance is direct. A payer's dossier usually includes an indirect comparison when head-to-head trials are absent. A guideline panel weighs the totality of evidence, not the largest trial. A regulator may ask for a meta-analysis of safety across trials. A product team that cannot synthesise the evidence is at a disadvantage in every one of those conversations.

The methodological risk is equally direct. A meta-analysis can produce a precise estimate of a biased effect if the included studies are biased. An indirect comparison can produce a plausible number from a network that is not connected or not consistent. A network meta-analysis can rank treatments on the basis of assumptions that the data cannot test. The appearance of quantitative rigour can be misleading.

The reporting standards are mature. The Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) 2020 statement provides a 27-item checklist. The Cochrane Handbook is the standard methods reference. The Grading of Recommendations Assessment, Development and Evaluation (GRADE) framework rates certainty. The Professional Society for Health Economics and Outcomes Research (ISPOR) has published good-practice guidance for indirect treatment comparisons and network meta-analysis. These are conventions and guidance, not statutes, but they define the review standard.

This topic teaches you to design a synthesis, choose the model, assess heterogeneity and inconsistency, rate certainty, and report the result so that a decision-maker can use it.

### What synthesis can and cannot do

Synthesis can:

- increase precision by combining studies,
- estimate an average effect across populations,
- explain heterogeneity through subgroup analysis or meta-regression,
- compare treatments that have not been compared head-to-head,
- rank treatments in a network,
- quantify the certainty of the evidence.

Synthesis cannot:

- remove bias from the included studies,
- create evidence where none exists,
- validate an inconsistent network,
- replace a well-designed trial for a specific question,
- guarantee that the average effect applies to a particular patient.

The limits are not a reason to avoid synthesis. They are a reason to be explicit about the assumptions.

### Why synthesis fails

- The search misses unpublished or negative studies.
- The included studies are biased, and the bias is not adjusted.
- The studies are clinically heterogeneous, and the pooled estimate is meaningless.
- The model is fixed when the effects vary.
- The network is inconsistent, and the ranking is unreliable.
- The certainty is overstated, and the recommendation is too strong.
- The review is not reproducible, and the reader cannot check it.

Each failure is preventable with a protocol, a systematic search, a risk-of-bias assessment, a heterogeneity analysis, and a certainty rating.

### What a synthesis is not

- It is not a literature review with a table of studies.
- It is not a vote count of significant versus non-significant trials.
- It is not a substitute for a trial when the question is causal and the effect is modest.
- It is not a way to create evidence where none exists.
- It is not a ranking exercise without uncertainty.
- It is not a substitute for clinical judgement.

The list is a warning against the common uses of "meta-analysis" in commercial documents.

### Synthesis method by question

| Question | Evidence structure | Method | Output |
|---|---|---|---|
| Is A better than B? | Multiple A vs B trials | Pairwise meta-analysis | Pooled effect |
| Is A better than B when no head-to-head trial exists? | A vs P and B vs P | Indirect comparison | Indirect effect |
| Which of A, B, C is best? | Connected network | NMA | Relative effects and ranking |
| Does the effect vary by dose? | Trials with different doses | Meta-regression | Dose-response |
| Is the evidence certain? | Any | GRADE | Certainty rating |
| What is the absolute effect? | Pooled relative effect and baseline risk | Risk difference | Absolute effect |

The table maps the question to the method. A question that cannot be mapped to a method is not yet a synthesis question.

### The decision context

Evidence synthesis is used for four decisions:

- **Regulatory:** a benefit-risk assessment across trials, a safety meta-analysis, or a supportive analysis for a submission.
- **Payer:** an indirect comparison or NMA to estimate relative effectiveness when head-to-head evidence is absent.
- **Guideline:** a systematic review and GRADE assessment to support a recommendation.
- **Clinical:** a rapid review or a meta-analysis to answer a point-of-care question.

Each decision has a different tolerance for uncertainty and a different requirement for rigour. A rapid review for a clinical question may accept a narrower search; a payer submission requires a systematic search and a reproducible analysis.

## 2. Learning outcomes

By the end of this topic you can:

1. **Formulate** a synthesis question with a defined population, intervention, comparator, outcome, and study design. (Exercise 1, self-check 1)
2. **Design** a search and selection process that is reproducible and reportable to PRISMA 2020. (Exercise 2, interview 2)
3. **Compute** a pairwise meta-analysis with fixed-effect and random-effects models, and interpret I-squared and tau-squared. (Exercises 3 and 4, self-check 3)
4. **Assess** publication bias and explain its limits. (Exercise 5, interview 5)
5. **Perform** an indirect comparison and check its consistency with direct evidence. (Exercise 6, self-check 5)
6. **Build** a network meta-analysis for a simple network and interpret the ranking. (Exercise 7, interview 6)
7. **Rate** the certainty of the evidence with GRADE domains. (Exercise 8, interview 8)

### How the outcomes connect

- Outcomes 1 and 2 define the question and the search.
- Outcomes 3 and 4 provide the pairwise synthesis and heterogeneity assessment.
- Outcomes 5 and 6 extend the synthesis to indirect and network evidence.
- Outcomes 7 and 8 provide the certainty rating and the decision.
- A learner who can perform outcomes 1 to 4 can run a meta-analysis.
- A learner who can also perform outcomes 5 to 8 can support a coverage or guideline decision.

## 3. The core mental model

Evidence synthesis is a chain from a question to a certainty rating. Each link has a method and an assumption.

```text
   QUESTION (PICO)
        |
        |  protocol and registration
        v
   SEARCH AND SELECTION
        |
        |  risk-of-bias assessment
        v
   INCLUDED STUDIES
        |
        |  effect measure and model
        v
   PAIRWISE META-ANALYSIS
        |
        |  heterogeneity and publication bias
        v
   SUMMARY EFFECT AND UNCERTAINTY
        |
        |  indirect comparison or NMA
        v
   RELATIVE EFFECTS ACROSS TREATMENTS
        |
        |  consistency and certainty
        v
   GRADE RATING AND DECISION
```

The chain is cumulative. A weak search undermines the synthesis regardless of the statistical method. A biased set of studies produces a biased summary regardless of the model. An inconsistent network produces unreliable rankings regardless of the software.

### Direct, indirect, and mixed evidence

**Direct evidence** comes from trials that compare two treatments head-to-head. It is the most credible evidence for a pairwise comparison.

**Indirect evidence** comes from trials that compare each treatment with a common comparator. If A is compared with placebo and B is compared with placebo, the A-versus-B effect can be estimated by subtracting the two effects, under the assumption that the trials are similar enough.

**Mixed evidence** combines direct and indirect evidence. A network meta-analysis does this for a network of treatments.

The key assumption is **transitivity**: the populations, interventions, and outcomes in the indirect comparisons must be similar enough that the subtraction is meaningful. Transitivity cannot be tested directly; it is assessed by comparing study characteristics and by checking consistency.

### Consistency

Consistency is the statistical agreement between direct and indirect evidence for the same comparison. It can be tested when a network has a loop. A loop with direct A-versus-B evidence and indirect A-versus-B evidence allows a test of the difference.

Inconsistency can arise from:

- differences in populations across trials,
- differences in doses or delivery,
- differences in outcome definitions,
- differences in study quality,
- chance.

A network with inconsistency should be interpreted cautiously, and the source should be investigated.

### Fixed versus random effects

A fixed-effect model assumes that every study estimates the same true effect. A random-effects model assumes that the true effects are drawn from a distribution, and it estimates the mean and the between-study variance.

The choice should be prespecified. A common approach is to use a random-effects model when heterogeneity is expected, and to report the fixed-effect estimate as a sensitivity analysis. A fixed-effect model with substantial heterogeneity produces a confidence interval that is too narrow.

### Certainty

GRADE rates certainty across five domains: risk of bias, inconsistency, indirectness, imprecision, and publication bias. The rating starts high for randomised evidence and low for observational evidence, and is downgraded for serious concerns in each domain.

The certainty rating is the bridge between the synthesis and the recommendation. A precise summary effect with low certainty is not a strong basis for a recommendation.

### Evidence structures

| Structure | What it looks like | Method | Key assumption |
|---|---|---|---|
| Pairwise | Several trials of A vs B | Meta-analysis | Comparability |
| Star | A vs P, B vs P, C vs P | Indirect comparison or NMA | Transitivity |
| Chain | A vs B, B vs C | Indirect comparison | Transitivity |
| Loop | A vs B, B vs C, A vs C | NMA with consistency test | Transitivity and consistency |
| Disconnected | A vs B, C vs D | Separate syntheses | None across components |
| Multi-arm | One trial with three arms | Meta-analysis with correlation | Within-trial correlation |

The structure determines what can be estimated. A chain can estimate A vs C indirectly; a disconnected network cannot estimate A vs C at all.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Systematic review | A review with a prespecified search, selection, and appraisal | Used for any literature review | Design |
| Meta-analysis | Statistical combination of study estimates | Used as a synonym for systematic review | Method |
| PICO | Population, intervention, comparator, outcome | Used without the study design | Framework |
| PRISMA 2020 | Reporting guideline for systematic reviews | Used as a quality assessment tool | Guideline |
| PROSPERO | International register of systematic review protocols | Confused with a trial registry | Registry |
| Risk of bias | The likelihood that a study's result is materially biased | Confused with quality | Judgement |
| RoB 2 | Cochrane risk-of-bias tool for randomised trials | Used for observational studies | Tool |
| ROBINS-I | Risk-of-bias tool for non-randomised studies | Used for randomised trials | Tool |
| Fixed-effect model | A model that assumes one true effect | Used when heterogeneity is present | Model |
| Random-effects model | A model that estimates the mean and between-study variance | Used without reporting tau-squared | Model |
| Heterogeneity | Real variation in effect across studies | Confused with sampling error | Variance |
| I-squared | The proportion of variability due to heterogeneity | Confused with the p-value | Percentage |
| Tau-squared | The between-study variance | Confused with the pooled estimate | Variance |
| Prediction interval | The range of effects expected in a new study | Confused with the confidence interval | Effect measure |
| Publication bias | Selective publication of positive results | Confused with reporting bias | Bias |
| Funnel plot | A plot of effect against precision | Used as proof of bias | Figure |
| Egger test | A regression test for funnel plot asymmetry | Used with few studies | Test |
| Trim-and-fill | A method to estimate and adjust for publication bias | Treated as a correction | Method |
| Direct evidence | Head-to-head comparison | Confused with any comparison | Evidence type |
| Indirect evidence | Comparison through a common comparator | Confused with direct evidence | Evidence type |
| Network meta-analysis (NMA) | A model combining direct and indirect evidence | Confused with pairwise meta-analysis | Method |
| Transitivity | The assumption that indirect comparisons are valid | Assumed without assessment | Assumption |
| Consistency | Statistical agreement between direct and indirect evidence | Confused with heterogeneity | Property |
| Inconsistency | Disagreement between direct and indirect evidence | Ignored in NMA | Property |
| Loop | A closed path in a network that allows a consistency test | Confused with a star network | Network feature |
| Star network | A network with one common comparator | Confused with a connected network | Network feature |
| SUCRA | Surface under the cumulative ranking curve | Interpreted as a probability of being best | Percentage |
| P-score | A frequentist analogue of SUCRA | Confused with SUCRA | Percentage |
| GRADE | A framework for rating certainty | Confused with risk of bias | Framework |
| Indirectness | Evidence does not match the question | Confused with heterogeneity | GRADE domain |
| Imprecision | The estimate is too uncertain for the decision | Confused with bias | GRADE domain |
| Network geometry | The structure of treatments and comparisons | Ignored in interpretation | Network feature |
| Lumping | Combining different interventions or populations | Confused with pooling | Error |
| Splitting | Separating interventions that should be combined | Confused with subgroup analysis | Error |
| Network plot | A graph of treatments and comparisons | Confused with a forest plot | Figure |
| Contribution matrix | The weight of each study in each NMA comparison | Ignored in interpretation | Matrix |
| Design-by-treatment interaction | A model-based consistency test | Confused with a loop-specific test | Test |
| Node-splitting | A local consistency test | Confused with a global test | Test |
| Bayesian NMA | An NMA with a prior distribution | Treated as assumption-free | Model |
| Frequentist NMA | An NMA with a likelihood-based estimator | Treated as assumption-free | Model |
| Effect modification | Variation in effect by a patient characteristic | Confused with heterogeneity | Concept |
| Meta-regression | A regression of effect on study-level covariates | Confused with subgroup analysis | Method |
| Ecological bias | Bias from using study-level covariates to infer patient-level effects | Ignored in meta-regression | Bias |
| Aggregation bias | Bias from combining different populations | Confused with heterogeneity | Bias |
| Reporting bias | Selective reporting of outcomes or analyses | Confused with publication bias | Bias |
| Outcome reporting bias | Selective reporting of outcomes within studies | Confused with publication bias | Bias |
| Language bias | Exclusion of non-English studies | Ignored in search limits | Bias |
| Time-lag bias | Delayed publication of negative studies | Ignored in rapid reviews | Bias |
| Citation bias | Preferential citation of positive studies | Ignored in snowball searches | Bias |
| Availability bias | Use of easily accessible data | Confused with systematic search | Bias |

Two distinctions deserve emphasis.

**Heterogeneity versus inconsistency.** Heterogeneity is variation across studies of the same comparison. Inconsistency is disagreement between direct and indirect evidence for the same comparison. A network can have low heterogeneity within each comparison and high inconsistency between them.

**Certainty versus precision.** Precision is the width of the confidence interval. Certainty is the confidence that the estimate is correct for the question, after considering bias, indirectness, and publication bias. A precise estimate can have low certainty.

## 5. The mechanics, step by step

The procedure has fourteen steps. Steps 1 to 5 define and assemble the evidence. Steps 6 to 9 analyse pairwise evidence. Steps 10 to 12 extend to indirect and network evidence. Steps 13 and 14 rate certainty and report.

### Procedure at a glance

| Step | Input | Output | Failure if skipped |
|---:|---|---|---|
| 1 | Decision and question | PICO and study design | Unfocused review |
| 2 | Question | Protocol and registration | Selective methods |
| 3 | Protocol | Search strategy | Incomplete evidence |
| 4 | Search results | Included studies | Selection bias |
| 5 | Included studies | Risk-of-bias assessment | Biased synthesis |
| 6 | Outcomes | Effect measure | Incompatible pooling |
| 7 | Study estimates | Pooled effect | Wrong model |
| 8 | Study estimates | Heterogeneity statistics | False precision |
| 9 | Study estimates | Publication-bias assessment | Overstated effect |
| 10 | Indirect comparisons | Indirect estimate | Invalid subtraction |
| 11 | Network | NMA estimates | Unsupported ranking |
| 12 | Network | Consistency assessment | Hidden disagreement |
| 13 | All evidence | GRADE rating | Overconfident recommendation |
| 14 | All methods | PRISMA report | Unreviewable review |

### Step 1: Formulate the question

**Input:** the decision and the clinical question.

**Output:** a PICO statement with the population, intervention, comparator, outcome, and eligible study designs.

**Rationale:** The question determines the search, the eligibility criteria, and the analysis. A broad question produces a heterogeneous review; a narrow question may produce no evidence.

The PICO should specify:

- the population, including any subgroup;
- the intervention, including dose and duration;
- the comparator, including usual care or placebo;
- the outcomes, with the primary outcome and the time point;
- the eligible designs, such as randomised trials or observational studies.

### Step 2: Write and register the protocol

**Input:** the PICO.

**Output:** a protocol with the search strategy, eligibility criteria, data extraction, risk-of-bias assessment, and analysis plan, registered in PROSPERO or an equivalent registry.

**Rationale:** The protocol prevents selective methods and selective reporting. Registration makes the review discoverable and allows comparison of the protocol with the published review.

The protocol should state the primary outcome, the effect measure, the model, the heterogeneity assessment, the subgroup analyses, and the certainty rating.

### Search sources and their roles

| Source | What it contributes | Limitation |
|---|---|---|
| MEDLINE | Biomedical journals | Indexing lag; English bias |
| Embase | Broader pharmacology coverage | Subscription |
| Cochrane CENTRAL | Trial records | Overlap with MEDLINE |
| ClinicalTrials.gov | Unpublished and ongoing trials | Incomplete results |
| WHO ICTRP | International registries | Variable data quality |
| Regulatory documents | Unpublished safety and efficacy data | Difficult to search |
| Conference abstracts | Emerging evidence | Limited data |
| Reference lists | Additional studies | Dependent on included studies |
| Citation searching | Studies citing a key paper | Dependent on the key paper |

The search should combine these sources and report the date and the number of records from each.

### Effect measures for common outcomes

| Outcome type | Effect measure | Interpretation | Assumption |
|---|---|---|---|
| Binary | Risk ratio | Relative risk | None |
| Binary | Odds ratio | Odds of the outcome | Approximates RR when rare |
| Binary | Risk difference | Absolute change | None |
| Continuous | Mean difference | Change in units | Same instrument |
| Continuous | Standardised mean difference | Change in SD units | Different instruments |
| Time-to-event | Hazard ratio | Relative hazard | Proportional hazards |
| Count | Rate ratio | Relative rate | Poisson or negative binomial |
| Ordinal | Proportional odds ratio | Shift in odds | Proportional odds |

The measure should be consistent across studies and appropriate for the decision. A risk difference is more interpretable for public health; a risk ratio is more interpretable for etiology.

### Step 3: Search

**Input:** the PICO and the protocol.

**Output:** a reproducible search strategy across bibliographic databases, trial registries, and grey literature.

**Rationale:** The search determines the evidence base. A search limited to one database or to English-language publications can miss relevant studies.

The search should include:

- at least two bibliographic databases, such as MEDLINE and Embase;
- a trial registry, such as ClinicalTrials.gov or the WHO International Clinical Trials Registry Platform;
- grey literature, such as conference abstracts and regulatory documents;
- reference lists of included studies;
- citation searching for key papers.

The strategy should be reported in full, with the date of the search and the number of records.

A reproducible search record includes:

- the databases and platforms searched;
- the search strings, with Boolean operators and limits;
- the date of the search;
- the number of records from each source;
- the deduplication method;
- the language and publication-year limits;
- the grey-literature sources;
- the hand-search and citation-search methods.

The record should be detailed enough for another reviewer to reproduce the search.

### Step 4: Select studies

**Input:** the search results.

**Output:** a set of included studies with a PRISMA flow diagram.

**Rationale:** Selection bias can arise if studies are included or excluded on the basis of their results. The selection should be done independently by two reviewers, with disagreements resolved by consensus or a third reviewer.

The flow diagram reports the number of records identified, screened, excluded, and included, with reasons for exclusion.

The selection process should record:

- the number of reviewers and their independence;
- the software used for screening;
- the resolution of disagreements;
- the reasons for full-text exclusion;
- the number of studies included in each analysis.

A review that does not report these cannot be reproduced.

### Step 5: Assess risk of bias

**Input:** the included studies.

**Output:** a risk-of-bias assessment for each study and each outcome.

**Rationale:** The synthesis is only as good as the studies. A meta-analysis of biased studies produces a precise biased estimate.

For randomised trials, use RoB 2, which has five domains. For non-randomised studies, use ROBINS-I, which has seven domains. The assessment should be outcome-specific, because a study can be at low risk for one outcome and high risk for another.

The assessment should record:

- the domain judgements for each study and outcome;
- the evidence for each judgement;
- the overall risk-of-bias judgement;
- the implications for the synthesis;
- the sensitivity analysis excluding high-risk studies.

A review that reports only the overall judgement, without the domain evidence, is not reproducible.

### Step 6: Choose the effect measure

**Input:** the outcomes and the study designs.

**Output:** an effect measure for each outcome, such as a risk ratio, odds ratio, risk difference, mean difference, or hazard ratio.

**Rationale:** The effect measure determines the interpretation and the pooling. A risk ratio is more interpretable for binary outcomes; a mean difference is more interpretable for continuous outcomes; a hazard ratio is appropriate for time-to-event outcomes if the proportional hazards assumption holds.

The measure should be consistent across studies. If studies report different measures, they may need to be converted, and the conversion should be reported.

### Step 7: Pool the studies

**Input:** the study estimates and their standard errors.

**Output:** a pooled estimate with a confidence interval, under a fixed-effect and a random-effects model.

**Rationale:** The pooled estimate is the summary of the evidence. The model determines what the summary means.

The fixed-effect estimate is a weighted average with weights equal to the inverse variance. The random-effects estimate adds the between-study variance to the weights. The random-effects model is more appropriate when heterogeneity is expected.

### Step 8: Assess heterogeneity

**Input:** the study estimates.

**Output:** Cochran's Q, I-squared, tau-squared, and a prediction interval.

**Rationale:** Heterogeneity determines whether the pooled estimate is meaningful. A pooled estimate with high heterogeneity is an average over different populations, not a generalisable effect.

The interpretation should consider:

- the magnitude and direction of the effects;
- the confidence intervals of the individual studies;
- the clinical and methodological differences;
- the prediction interval.

A common rule is that I-squared above 50 percent represents substantial heterogeneity, but the threshold is a convention, not a law.

### Step 9: Assess publication bias

**Input:** the study estimates and their precision.

**Output:** a funnel plot, an Egger test where appropriate, and a narrative assessment.

**Rationale:** Publication bias occurs when studies with positive results are more likely to be published. It can inflate the pooled effect.

The assessment is limited when there are fewer than 10 studies. A funnel plot with few studies is uninformative, and the Egger test has low power. The review should state the limitation rather than claim that no bias exists.

A publication-bias assessment should include:

- a funnel plot with the study labels;
- the Egger test, if at least 10 studies;
- a comparison of published and registered outcomes;
- a search of trial registries for unpublished studies;
- a statement of the limitations.

The assessment is a judgement, not a test. A symmetrical funnel plot does not prove the absence of bias.

### Step 10: Indirect comparisons

**Input:** two comparisons with a common comparator.

**Output:** an indirect estimate of the comparison of interest, with a confidence interval.

**Rationale:** Indirect comparisons allow a comparison when no head-to-head trial exists. The estimate is the difference between the two effects relative to the common comparator.

The assumption is transitivity: the trials must be similar in population, intervention, and outcome. The review should compare the trial characteristics and state the assumption.

### Step 11: Network meta-analysis

**Input:** a connected network of comparisons.

**Output:** relative effects for every pair of treatments, and a ranking.

**Rationale:** NMA combines direct and indirect evidence and can compare multiple treatments. It requires a connected network and the transitivity assumption.

The model can be fixed or random effects, and it must account for the correlation between multi-arm trials. The analysis should report the network geometry, the model, and the ranking method.

### Step 12: Assess consistency

**Input:** the network and the direct and indirect estimates.

**Output:** a consistency assessment, such as a loop-specific test or a design-by-treatment interaction model.

**Rationale:** Inconsistency means that the direct and indirect evidence disagree. It undermines the NMA estimates and the ranking.

The assessment should be prespecified, and the source of any inconsistency should be investigated. A network with material inconsistency should not be used for a confident ranking.

### Step 13: Rate certainty

**Input:** the synthesis and the risk-of-bias assessment.

**Output:** a GRADE rating for each outcome, with the reasons for downgrading.

**Rationale:** The certainty rating tells the decision-maker how much confidence to place in the estimate. It is the bridge between the synthesis and the recommendation.

The five domains are risk of bias, inconsistency, indirectness, imprecision, and publication bias. The rating can be high, moderate, low, or very low.

### Step 14: Report

**Input:** all of the above.

**Output:** a PRISMA 2020-compliant report with the search, the flow diagram, the study characteristics, the risk-of-bias assessment, the synthesis, the heterogeneity, the certainty, and the limitations.

**Rationale:** The report must be reproducible. A reader should be able to repeat the search and the analysis from the report.

The report should include the protocol registration, the search date, the software and version, and the data and code availability.

### Reporting checklist

- The PICO and the eligibility criteria.
- The protocol registration.
- The search strategy and date.
- The PRISMA flow diagram.
- The study characteristics.
- The risk-of-bias assessment.
- The effect measure and model.
- The forest plot.
- The heterogeneity statistics.
- The publication-bias assessment.
- The network plot and consistency assessment.
- The GRADE table.
- The limitations.
- The data and code availability.

### Edge cases

**Zero events.** A study with zero events in both arms cannot contribute to a risk ratio or odds ratio without a continuity correction. The review should state the method and test the sensitivity to the correction.

**Rare events.** A rare outcome produces imprecise estimates and makes publication bias hard to assess. The review should use a method appropriate for rare events, such as a Peto odds ratio or a beta-binomial model.

**Multi-arm trials.** A multi-arm trial contributes correlated comparisons. The meta-analysis must account for the correlation, and the NMA must model it.

**Cluster trials.** A cluster trial has a design effect that must be accounted for; otherwise the meta-analysis overstates precision.

**Overlapping reviews.** Several reviews may address the same question with different methods. The review should compare and reconcile them, not simply cite the most favourable.

**Rapid reviews.** A rapid review may limit the search or the number of databases. The limitations should be stated, and the certainty rating should reflect them.

**Network disconnection.** A network is disconnected if there is no path between two treatments. An NMA cannot estimate a comparison across disconnected components without additional assumptions.

### Edge-case checklist

- Zero events in one arm: state the continuity correction and test its sensitivity.
- Rare events: use a sparse-data method and report the event count.
- Multi-arm trials: account for the correlation between comparisons.
- Cluster trials: apply the design effect.
- Overlapping reviews: compare and reconcile, do not cherry-pick.
- Rapid reviews: state the limitations and downgrade certainty.
- Disconnected networks: do not estimate cross-component comparisons.
- Different outcome definitions: harmonise or analyse separately.
- Different doses: model dose as a moderator or split the intervention.
- Different follow-up: use the longest common follow-up or a time-to-event model.

### Decision table: synthesis method by evidence structure

| Evidence structure | Method | Key assumption | Main risk |
|---|---|---|---|
| Multiple trials, same comparison | Pairwise meta-analysis | Effects are comparable | Heterogeneity |
| Two comparisons, common comparator | Indirect comparison | Transitivity | Population differences |
| Connected network | NMA | Transitivity and consistency | Inconsistency |
| Star network | NMA or indirect comparison | Transitivity | Unconnected treatments |
| Single trial per comparison | Narrative synthesis | None | Imprecision |
| Disconnected network | Separate syntheses | None | No cross-component comparison |
| Rare events | Peto or beta-binomial | Sparse-data method | Bias with unbalanced arms |
| Observational studies | ROBINS-I plus synthesis | Confounding addressed | Residual confounding |

## 6. Worked example

### The setting

A payer is reviewing four treatments for migraine prophylaxis: placebo (P), drug A, drug B, and drug C. The company submitting drug A has provided a network meta-analysis. The payer's clinical team asks whether the ranking in the submission is credible.

The evidence base has seven randomised trials:

| Comparison | Trials | Log risk ratio (SE) |
|---|---:|---|
| A vs P | 3 | -0.55 (0.12), -0.20 (0.15), -0.30 (0.18) |
| B vs P | 2 | -0.20 (0.14), -0.25 (0.16) |
| C vs P | 2 | -0.45 (0.13), -0.38 (0.17) |
| A vs B | 1 | -0.10 (0.20) |

The outcome is the proportion of patients with at least a 50 percent reduction in monthly migraine days at 12 weeks. A negative log risk ratio favours the treatment.

### Step 1: Question

> In adults with episodic migraine, what is the relative efficacy of drugs A, B, and C compared with placebo and with each other for achieving at least a 50 percent reduction in monthly migraine days at 12 weeks?

The population is adults with episodic migraine; the interventions are A, B, and C; the comparator is placebo or another active drug; the outcome is the responder proportion at 12 weeks; the designs are randomised trials.

### Step 2: Protocol

The protocol is registered. It specifies the primary outcome, the effect measure (log risk ratio), the model (random effects), the heterogeneity assessment, the consistency assessment, and the GRADE rating.

### Search results in the example

The search covers MEDLINE, Embase, the Cochrane Central Register of Controlled Trials, and ClinicalTrials.gov, from inception to the search date. The search strategy is reported in full.

### Step 4: Selection

The PRISMA flow diagram shows 1,240 records, 86 full-text reviews, and 7 included trials. The reasons for exclusion are reported.

### Step 5: Risk of bias

All seven trials are randomised and double-blind. Two trials have unclear allocation concealment. The risk of bias is low for five trials and some concerns for two.

### Step 6: Effect measure

The effect measure is the log risk ratio. A negative value favours the active drug. The results are back-transformed to risk ratios for interpretation.

### Step 7: Pairwise pooling

| Comparison | Trials | Pooled log RR (95% CI) | Risk ratio (95% CI) | I-squared |
|---|---:|---|---|---:|
| A vs P | 3 | -0.37 (-0.60 to -0.14) | 0.69 (0.55 to 0.87) | 44.9% |
| B vs P | 2 | -0.22 (-0.43 to -0.02) | 0.80 (0.65 to 0.98) | 0% |
| C vs P | 2 | -0.42 (-0.63 to -0.22) | 0.65 (0.53 to 0.80) | 0% |

The pooled estimates show that all three drugs are superior to placebo. Drug C has the largest point estimate, followed by A and B. The confidence intervals overlap, so the pairwise comparisons do not establish a ranking.

### Step 8: Heterogeneity

The A-versus-P comparison has I-squared of 44.9 percent and tau-squared of 0.018. The heterogeneity is moderate. The prediction interval for a new trial is approximately -0.79 to 0.05, which includes no effect. The other comparisons have no heterogeneity.

The heterogeneity in A versus P is explained in part by a trial with a smaller effect (-0.20) and a trial with a larger effect (-0.55). The review explores the difference by dose and by trial duration, but the subgroup analysis is underpowered.

### Step 9: Publication bias

The funnel plot has only seven trials, which is too few for a reliable assessment. The Egger test is not reported because of the small number of studies. The review states that publication bias cannot be excluded.

### Step 10: Indirect comparison

The indirect estimate of A versus B is the difference between A versus P and B versus P:

```text
log RR(A vs B) = log RR(A vs P) - log RR(B vs P)
               = -0.37 - (-0.22)
               = -0.15
```

The standard error is the square root of the sum of the squared standard errors:

```text
SE = sqrt(0.115^2 + 0.105^2) = 0.156
```

The indirect estimate is -0.15 (95% CI -0.45 to 0.16), which is consistent with no difference.

### Network estimates

The network is connected through placebo. The NMA estimates are:

| Comparison | NMA log RR (95% CI) | Risk ratio |
|---|---|---|
| A vs P | -0.37 (-0.60 to -0.14) | 0.69 |
| B vs P | -0.22 (-0.43 to -0.02) | 0.80 |
| C vs P | -0.42 (-0.63 to -0.22) | 0.65 |
| A vs B | -0.15 (-0.45 to 0.16) | 0.86 |
| C vs B | -0.20 (-0.49 to 0.09) | 0.82 |
| C vs A | -0.05 (-0.36 to 0.25) | 0.95 |

The NMA estimates are close to the pairwise estimates because the network is simple and the direct evidence is limited.

### Step 12: Consistency

The only loop is A-B-P. The direct A-versus-B estimate is -0.10 (SE 0.20). The indirect estimate is -0.15 (SE 0.156). The difference is 0.05, with a standard error of the difference of 0.254, giving a z-score of 0.19 and a p-value of about 0.85.

There is no evidence of inconsistency. The network is consistent, but the confidence interval is wide.

### Step 13: Certainty

| Outcome | Certainty | Reason |
|---|---|---|
| A vs P | Moderate | Some concerns in two trials; moderate heterogeneity |
| B vs P | Moderate | Some concerns in one trial; few trials |
| C vs P | Moderate | Few trials; publication bias cannot be excluded |
| Ranking | Low | Imprecise comparisons; few trials per comparison |

The certainty for the ranking is low because the differences between treatments are small relative to the uncertainty. A ranking based on overlapping confidence intervals is not a reliable basis for a coverage decision.

### Reporting the synthesis

The report follows PRISMA 2020. It includes the search strategy, the flow diagram, the study characteristics, the risk-of-bias assessment, the forest plots, the network plot, the consistency assessment, the GRADE table, and the limitations.

### Two diagnostics that reveal a flawed synthesis

**Diagnostic 1: the overlap check.** The pairwise confidence intervals for A, B, and C overlap substantially. A ranking that places C first and B last without acknowledging the overlap overstates the precision. The diagnostic is to compare the confidence intervals of the NMA estimates, not only the point estimates.

**Diagnostic 2: the network geometry check.** The network has four treatments and seven trials, with only one direct A-versus-B trial and no direct A-versus-C or B-versus-C trials. The NMA estimates depend heavily on the transitivity assumption. The diagnostic is to compare the trial characteristics across comparisons; if the populations differ, the indirect estimates are not credible.

### Vary one assumption: the heterogeneity

If the A-versus-P heterogeneity is high (I-squared 80 percent), the random-effects estimate would be more uncertain, and the ranking would be less stable. The review should report the prediction interval and avoid a confident ranking.

### Vary one assumption: the network

If the A-versus-B trial were removed, the network would still be connected through placebo, but there would be no loop and no consistency test. The NMA estimates would rely entirely on the transitivity assumption. The review should state that consistency cannot be assessed.

### Vary one assumption: the effect measure

If the outcome were a continuous measure of migraine days rather than a responder proportion, the effect measure would be a mean difference, and the pooling would require means and standard deviations. The ranking might differ because the responder outcome discards information.

### The decision

The payer concludes that all three drugs are superior to placebo, that the differences between drugs are uncertain, and that the ranking is not reliable enough for a preferential coverage decision. It covers all three with prior authorisation and requests a head-to-head trial or a larger network.

The example shows the central lesson of evidence synthesis: a precise-looking ranking can rest on overlapping confidence intervals and untestable assumptions. The synthesis is useful for the comparison with placebo and for the uncertainty between drugs, not for a definitive ranking.

### Direct versus indirect evidence in this network

| Comparison | Direct evidence | Indirect evidence | Agreement |
|---|---|---|---|
| A vs P | 3 trials | None | Not applicable |
| B vs P | 2 trials | None | Not applicable |
| C vs P | 2 trials | None | Not applicable |
| A vs B | 1 trial (-0.10) | -0.15 | Consistent (p 0.85) |
| A vs C | None | -0.05 | Untestable |
| B vs C | None | -0.20 | Untestable |

The table shows that two of the six comparisons have no direct evidence. The NMA estimates for those comparisons depend entirely on the transitivity assumption.

### What the payer asked

- Are all three drugs better than placebo?
- Is one drug better than the others?
- How certain is the ranking?
- Does the evidence apply to our population?
- What would change the conclusion?

The review answers the first with moderate certainty, the second with low certainty, and the third with a warning about overlapping intervals. The fourth depends on the population in the trials, which was mostly adults with episodic migraine and no prior preventive treatment. The fifth would be a head-to-head trial or a larger network with more direct comparisons.

## 7. Code, query, or calculation

The code below implements a pairwise meta-analysis, heterogeneity statistics, an indirect comparison, and a simple ranking. It uses only the Python standard library.

```python
import math
from statistics import NormalDist

Z = NormalDist().inv_cdf
PHI = NormalDist().cdf


def pool(effects, ses, random_effects=True):
    """Fixed and random-effects pooling of log effects."""
    w = [1 / s ** 2 for s in ses]
    fe = sum(wi * e for wi, e in zip(w, effects)) / sum(w)
    q = sum(wi * (e - fe) ** 2 for wi, e in zip(w, effects))
    df = len(effects) - 1
    c = sum(w) - sum(wi * wi for wi in w) / sum(w)
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    wr = [1 / (s * s + tau2) for s in ses]
    re = sum(wi * e for wi, e in zip(wr, effects)) / sum(wr)
    se_re = math.sqrt(1 / sum(wr))
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0
    estimate = re if random_effects else fe
    se = se_re if random_effects else math.sqrt(1 / sum(w))
    return {
        "fixed": round(fe, 3),
        "random": round(re, 3),
        "se_random": round(se_re, 3),
        "q": round(q, 2),
        "i2": round(i2, 1),
        "tau2": round(tau2, 4),
        "ci": (round(estimate - 1.96 * se, 3),
               round(estimate + 1.96 * se, 3)),
    }


def indirect(e1, se1, e2, se2):
    """Indirect comparison: treatment 1 vs treatment 2 through a common comparator."""
    diff = e1 - e2
    se = math.sqrt(se1 ** 2 + se2 ** 2)
    return round(diff, 3), round(se, 3), (round(diff - 1.96 * se, 3),
                                          round(diff + 1.96 * se, 3))


def ranking(effects, ses):
    """Approximate probability each treatment is best (most negative effect)."""
    names = list(effects)
    probs = {}
    for name in names:
        p = 1.0
        for other in names:
            if other == name:
                continue
            diff = effects[other] - effects[name]
            se = math.sqrt(ses[name] ** 2 + ses[other] ** 2)
            p *= PHI(diff / se)
        probs[name] = p
    total = sum(probs.values())
    return {k: round(v / total, 2) for k, v in probs.items()}


if __name__ == "__main__":
    a = pool([-0.55, -0.20, -0.30], [0.12, 0.15, 0.18])
    b = pool([-0.20, -0.25], [0.14, 0.16])
    c = pool([-0.45, -0.38], [0.13, 0.17])
    print("A vs P:", a)
    print("B vs P:", b)
    print("C vs P:", c)
    print("indirect A vs B:", indirect(a["random"], a["se_random"],
                                      b["random"], b["se_random"]))
    effects = {"A": a["random"], "B": b["random"], "C": c["random"]}
    ses = {"A": a["se_random"], "B": b["se_random"], "C": c["se_random"]}
    print("ranking:", ranking(effects, ses))
```

Expected output:

```text
A vs P: {'fixed': -0.389, 'random': -0.37, 'se_random': 0.115, 'q': 3.63, 'i2': 44.9, 'tau2': 0.0179, 'ci': (-0.595, -0.145)}
B vs P: {'fixed': -0.222, 'random': -0.222, 'se_random': 0.105, 'q': 0.06, 'i2': 0.0, 'tau2': 0.0, 'ci': (-0.428, -0.016)}
C vs P: {'fixed': -0.424, 'random': -0.424, 'se_random': 0.103, 'q': 0.11, 'i2': 0.0, 'tau2': 0.0, 'ci': (-0.626, -0.222)}
indirect A vs B: (-0.148, 0.156, (-0.453, 0.157))
ranking: {'A': 0.34, 'B': 0.02, 'C': 0.65}
```

The output shows four things.

- The A-versus-P comparison has moderate heterogeneity (I-squared 44.9 percent), and the random-effects estimate (-0.37) is slightly smaller than the fixed-effect estimate (-0.389).
- The B-versus-P and C-versus-P comparisons have no heterogeneity.
- The indirect A-versus-B estimate is -0.148 with a confidence interval of -0.453 to 0.157, which includes no difference.
- The ranking places C first (0.65), A second (0.34), and B third (0.02), but the ranking is based on overlapping confidence intervals and should be interpreted cautiously.

The ranking function is approximate. A proper NMA would model the correlations and produce a joint posterior or a frequentist ranking with a P-score. The teaching point is that the ranking is a probabilistic summary of uncertain estimates, not a definitive ordering.

### Leave-one-out sensitivity analysis

A leave-one-out analysis shows how much each study influences the pooled estimate. The code below applies it to the A-versus-P comparison.

```python
import math


def pool_random(effects, ses):
    """Random-effects pooling with the DerSimonian-Laird estimator."""
    w = [1 / s ** 2 for s in ses]
    fe = sum(wi * e for wi, e in zip(w, effects)) / sum(w)
    q = sum(wi * (e - fe) ** 2 for wi, e in zip(w, effects))
    df = len(effects) - 1
    c = sum(w) - sum(wi * wi for wi in w) / sum(w)
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    wr = [1 / (s * s + tau2) for s in ses]
    re = sum(wi * e for wi, e in zip(wr, effects)) / sum(wr)
    se = math.sqrt(1 / sum(wr))
    return round(re, 3), round(se, 3)


if __name__ == "__main__":
    effects = [-0.55, -0.20, -0.30]
    ses = [0.12, 0.15, 0.18]
    for i in range(len(effects)):
        e = [x for j, x in enumerate(effects) if j != i]
        s = [x for j, x in enumerate(ses) if j != i]
        print(f"omit trial {i + 1}: {pool_random(e, s)}")
```

Expected output:

```text
omit trial 1: (-0.241, 0.115)
omit trial 2: (-0.461, 0.12)
omit trial 3: (-0.387, 0.175)
```

The output shows that omitting the trial with the largest effect (-0.55) moves the pooled estimate from -0.37 to -0.241, a reduction of about one third. Omitting the smallest-effect trial moves it to -0.461. The result is sensitive to the inclusion of individual trials, which is a reason to report the leave-one-out analysis and to interpret the pooled estimate cautiously.

### A GRADE summary

| Domain | A vs P | B vs P | C vs P |
|---|---|---|---|
| Risk of bias | Some concerns | Some concerns | Some concerns |
| Inconsistency | Moderate (I2 45%) | None | None |
| Indirectness | Direct | Direct | Direct |
| Imprecision | Adequate | Wide | Adequate |
| Publication bias | Undetected but possible | Undetected but possible | Undetected but possible |
| Certainty | Moderate | Moderate | Moderate |

The table is a summary of the GRADE assessment. The certainty is moderate, not high, because of the risk-of-bias concerns and the limited number of trials.

## 8. Data, standards, and regulatory context

### Binding requirements

**EU Clinical Trials Regulation and HTA regulation.** Regulation (EU) No 536/2014 governs clinical trials, and Regulation (EU) 2021/2282 on health technology assessment establishes a framework for joint clinical assessments. The latter requires a systematic review of the evidence for certain technologies. These are binding instruments.

**US regulatory context.** The FDA does not mandate a systematic review for every submission, but a benefit-risk assessment may rely on a meta-analysis, and the FDA's guidance on meta-analysis for safety and effectiveness describes the expectations. The guidance is not a statute, but it defines the review standard.

### Formal guidance

**PRISMA 2020.** The statement (BMJ 2021;372:n71) provides a 27-item checklist for reporting systematic reviews. It includes items on the search, the selection, the risk of bias, the synthesis, the certainty, and the data and code availability. It is a reporting convention, not a design requirement.

**Cochrane Handbook.** The Cochrane Handbook for Systematic Reviews of Interventions is the standard methods reference. It covers the search, the risk of bias, the synthesis, the heterogeneity, the reporting bias, and the certainty. It is a convention.

**GRADE.** The GRADE framework rates certainty across five domains. It is a convention used by guidelines and HTA bodies.

**ISPOR good practices.** ISPOR has published guidance on indirect treatment comparisons and network meta-analysis, including a questionnaire to assess relevance and credibility. The guidance is a convention, but it is widely used by HTA bodies.

**NICE methods guide.** The National Institute for Health and Care Excellence (NICE) in the UK publishes a methods guide for technology appraisal, including the evidence synthesis requirements. It is a binding requirement for NICE submissions, not for other jurisdictions.

### What the standards constrain and what they leave open

| Instrument | Constrains | Leaves open |
|---|---|---|
| PRISMA 2020 | Reporting content | The methods |
| Cochrane Handbook | Methods conventions | The specific model |
| GRADE | Certainty domains | The rating |
| ISPOR guidance | ITC and NMA credibility | The model and the network |
| NICE methods guide | Evidence requirements for NICE | The analysis |
| EU HTA regulation | Joint clinical assessment | The synthesis method |

The standards require transparency and a structured assessment. They do not prescribe the model, the network, or the certainty rating. Those depend on the evidence and the question.

### Questions a reviewer will ask

- What is the PICO, and was the protocol registered?
- Which databases and registries were searched?
- How were studies selected, and by how many reviewers?
- Was risk of bias assessed, and how?
- What model was used, and why?
- What was the heterogeneity, and how was it interpreted?
- How was publication bias assessed?
- Is the network connected, and was consistency assessed?
- How was the ranking calculated, and what is its uncertainty?
- What is the certainty for each outcome?

A review that answers these questions is reproducible and reviewable.

### Data standards

Systematic reviews increasingly share data and code. The Cochrane Handbook recommends sharing the extraction sheet, the analysis code, and the risk-of-bias assessments. Data sharing improves reproducibility and allows the review to be updated.

For network meta-analysis, the network geometry and the study-level data should be reported so that the analysis can be reproduced. A network plot without the underlying data is not reproducible.

## 9. Industry practice and case evidence

### Selective publication of antidepressant trials

Turner and colleagues compared the published literature on antidepressant trials with the FDA's registered data. They found that the published trials were predominantly positive, while the FDA data included several negative or null trials that were not published or were reported with a favourable spin. The published literature overestimated the benefit of the drugs.

The documented fact is the discrepancy between the published and regulatory data. The inference is that publication and reporting bias can inflate a meta-analysis even when the included studies are randomised and well conducted.

**Lesson.** A meta-analysis of the published literature can overstate the effect if negative studies are missing. Regulatory data and trial registries are essential for a complete synthesis.

### Rofecoxib and cardiovascular risk

Rofecoxib was withdrawn from the market in 2004 after the APPROVe trial showed an increased risk of cardiovascular events. Earlier observational studies and a meta-analysis had suggested a cardiovascular signal, but the evidence was contested and the trial was needed to confirm it.

The documented fact is the withdrawal and the trial result. The inference is that a synthesis of observational studies can generate a signal, but a randomised trial may be required to confirm it. The synthesis was useful for hypothesis generation, not for a definitive causal claim.

**Lesson.** Synthesis is a tool for weighing evidence, not a substitute for a designed study when the question is causal and the effect is modest.

### COVID-19 and rapid evidence synthesis

The COVID-19 pandemic produced an unprecedented volume of preprints, trials, and reviews. Many rapid reviews and meta-analyses were published, some with methodological limitations. The experience showed both the value of rapid synthesis and the risk of premature conclusions.

The documented fact is the volume and the methodological debate. The inference is that rapid synthesis requires explicit limitations and a cautious certainty rating. A rapid review that does not state its limitations can mislead.

**Lesson.** Speed and rigour are in tension. A rapid review should state what was sacrificed and how that affects the certainty.

### The RECOVERY trial and the evidence base

The RECOVERY trial randomised 2,104 participants to dexamethasone and 4,321 to usual care. Overall 28-day mortality was 22.9 percent versus 25.7 percent (age-adjusted rate ratio 0.83, 95 percent confidence interval 0.75 to 0.93). Among participants receiving invasive mechanical ventilation, mortality was 29.3 percent versus 41.4 percent (rate ratio 0.64, 95 percent confidence interval 0.51 to 0.81).

The documented fact is the trial result. The inference is that a single large trial can change practice faster than a meta-analysis of small trials, especially when the effect is large and the outcome is objective. The trial was designed to be large enough to answer the question, and it was embedded in routine care.

**Lesson.** A well-designed large trial can be more informative than a synthesis of small, heterogeneous studies. Synthesis is not always the fastest route to a decision.

### What companies actually do

The following practices are inferred from public dossiers, guidance, and published reviews.

1. **Systematic reviews are standard in HTA submissions.** Companies commission a systematic review and an indirect comparison for the payer dossier.
2. **NMA is used when head-to-head evidence is absent.** The NMA is often the basis for a relative effectiveness claim.
3. **Sensitivity analyses are expected.** The dossier includes fixed and random effects, alternative networks, and subgroup analyses.
4. **Certainty ratings are included.** GRADE or a similar framework is used to communicate uncertainty.
5. **Regulatory data are used.** Companies search trial registries and regulatory documents, not only the published literature.

### Where practice falls short

- **Search limited to published literature.** Negative trials and unpublished data are missed.
- **No risk-of-bias assessment.** The synthesis pools studies of different quality without adjustment.
- **Fixed-effect model with high heterogeneity.** The confidence interval is too narrow.
- **Ranking without uncertainty.** The SUCRA is presented without the confidence intervals.
- **Inconsistency ignored.** The NMA is presented without a consistency assessment.
- **Certainty overstated.** The GRADE rating is high despite serious concerns.

### What the cases have in common

- The published evidence looked more favourable than the complete evidence.
- The discrepancy was invisible from the published meta-analysis alone.
- Regulatory or registry data changed the conclusion.
- The correction required access to unpublished data.
- The commercial consequences were substantial.

The pattern suggests a rule: a synthesis that relies only on the published literature should state that limitation prominently, and a decision that depends on a small effect should seek regulatory or registry data.

### Practice comparison

| Practice | Mature reviews | Less mature reviews | Consequence |
|---|---|---|---|
| Protocol | Registered | Absent | Selective methods |
| Search | Multiple databases and registries | One database | Missing evidence |
| Selection | Dual review | Single reviewer | Selection bias |
| Risk of bias | Domain-level | Not assessed | Biased synthesis |
| Model | Prespecified; both fixed and random | Fixed only | False precision |
| Heterogeneity | I2, tau2, prediction interval | I2 only | Misleading summary |
| Publication bias | Funnel plot and registry search | Not assessed | Overstated effect |
| Consistency | Loop test or design-by-treatment | Not assessed | Unreliable ranking |
| Certainty | GRADE | Not rated | Overconfident recommendation |

The table is an inference from published reviews and methodological guidance, not a systematic survey.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Incomplete search | One database; English only | Publication and language bias | Missing evidence | Multiple databases; registries |
| Selection bias | Reviewers not independent | Selective inclusion | Biased synthesis | Dual review; consensus |
| No risk-of-bias assessment | All studies pooled equally | Quality differences | Biased summary | RoB 2 or ROBINS-I |
| Wrong model | Fixed effect with high I2 | Understated uncertainty | False precision | Random effects; prediction interval |
| Heterogeneity ignored | Pooled estimate without I2 | Different populations | Misleading average | Subgroup analysis; meta-regression |
| Publication bias ignored | Funnel plot omitted | Missing negative studies | Overstated effect | Funnel plot; registry search |
| Invalid indirect comparison | Populations differ | Transitivity violation | Biased comparison | Compare characteristics; sensitivity |
| Inconsistent network | Loop test omitted | Direct and indirect disagree | Unreliable NMA | Consistency assessment |
| Ranking overstated | SUCRA without intervals | Overlapping estimates | Wrong coverage decision | Report intervals; caution |
| Certainty overstated | High GRADE with concerns | No downgrade | Overconfident recommendation | GRADE domains |
| Overlapping reviews | Multiple reviews with different results | Different methods | Confusion | Compare and reconcile |
| Data not shared | No extraction sheet or code | Irreproducible | Unreviewable | Share data and code |
| Language bias | English-only search | Missing non-English trials | Biased evidence base | Translate or justify |
| Outcome reporting bias | Only significant outcomes reported | Selective reporting | Overstated effect | Compare protocol and report |
| Unclear dose | Different doses pooled | Lumping | Misleading average | Dose subgroup or meta-regression |
| Different follow-up | Short and long trials pooled | Incompatible horizons | Misleading effect | Use common horizon |
| Unclear comparator | Usual care varies | Lumping | Heterogeneity | Define comparator |
| Overlapping cohorts | Same patients in multiple studies | Double counting | Narrow CI | Audit study overlaps |
| Unpublished data missing | Registry results absent | Publication bias | Overstated effect | Registry search |
| Inconsistent definitions | Different outcome definitions | Measurement differences | Heterogeneity | Harmonise definitions |
| Different populations | Broad and narrow eligibility | Indirectness | Misleading average | Subgroup or meta-regression |
| Different eras | Standard of care changed | Secular trend | Misleading effect | Restrict by era |
| Different settings | Primary versus tertiary care | Indirectness | Limited applicability | Report setting |

### Red flags in a synthesis report

- The search is described in one sentence.
- Only one database is searched.
- The selection process is not described.
- Risk of bias is not assessed.
- The model is not stated.
- Heterogeneity is not reported.
- Publication bias is not discussed.
- The network is not displayed.
- Consistency is not assessed.
- The ranking is presented without uncertainty.
- Certainty is not rated.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The search is comprehensive | The evidence base is complete | Missing databases or languages | Search audit; registry search |
| The studies are comparable | Pooling is meaningful | Different populations or outcomes | Clinical review; subgroup analysis |
| The effect measure is consistent | Pooling is valid | Different definitions | Conversion; sensitivity analysis |
| The model is appropriate | The summary is interpretable | Wrong fixed/random choice | Report both; prediction interval |
| The studies are at low risk of bias | The summary is credible | Serious risk of bias | RoB 2; sensitivity analysis |
| Publication bias is absent or limited | The summary is unbiased | Missing negative studies | Funnel plot; registry search |
| Transitivity holds | Indirect comparisons are valid | Different populations | Compare characteristics |
| Consistency holds | NMA is valid | Direct and indirect disagree | Loop test; design-by-treatment |
| The ranking is stable | The decision is robust | Overlapping intervals | Report intervals; sensitivity |
| The certainty rating is appropriate | The recommendation is calibrated | Over- or under-downgrading | GRADE review |
| The network is connected | NMA estimates are estimable | Disconnected components | Network audit |
| The studies are independent | Variance is not double-counted | Overlapping cohorts | Study-level audit |
| The search is up to date | The evidence base is current | New trials | Update search |
| The outcome is patient-relevant | The recommendation matters | Surrogate outcome | Justify or downgrade |
| The effect measure is interpretable | The decision is informed | Odds ratio presented as risk | Report risk difference |
| The absolute effect is reported | Clinicians can act | Only relative effect | Add baseline risk |

### Limits of this topic

This topic covers the design and interpretation of systematic reviews, meta-analysis, indirect comparisons, and network meta-analysis. It does not cover the full statistical theory of NMA, the details of Bayesian hierarchical models, or the conduct of a full systematic review. Those require specialised training and software.

The worked example and code use a simple network and simulated effect sizes. The numbers illustrate the relationships among pooling, heterogeneity, indirectness, and ranking; they are not estimates from a real review.

## 12. Exercises

### Exercise 1 — Formulate a PICO (application, 25 minutes)

**Deliverable:** a PICO for a review of sodium-glucose cotransporter-2 inhibitors in heart failure, including the population, intervention, comparator, outcome, and eligible designs.

**Acceptance criteria:** all elements present; the outcome has a time point; the designs are specified.

**Time:** 25 minutes.

### Exercise 2 — Design a search (application, 30 minutes)

**Deliverable:** a search strategy for the PICO in Exercise 1, including at least two databases, a trial registry, and grey literature.

**Acceptance criteria:** databases named; search terms described; date range stated; limitations acknowledged.

**Time:** 30 minutes.

### Exercise 3 — Pairwise meta-analysis (calculation, 40 minutes)

**Deliverable:** a fixed-effect and random-effects pooled estimate for three trials with log risk ratios of -0.4 (SE 0.1), -0.3 (SE 0.15), and -0.5 (SE 0.2), including I-squared and tau-squared.

**Acceptance criteria:** formulas correct; estimates correct; heterogeneity interpreted.

**Time:** 40 minutes.

### Exercise 4 — Heterogeneity (critique, 30 minutes)

**Deliverable:** an interpretation of a pooled effect with I-squared of 70 percent and a prediction interval that includes no effect.

**Acceptance criteria:** heterogeneity identified; prediction interval interpreted; recommendation stated.

**Time:** 30 minutes.

### Exercise 5 — Publication bias (critique, 25 minutes)

**Deliverable:** an assessment of publication bias for a review with eight studies and an asymmetric funnel plot.

**Acceptance criteria:** the limitation of the small number of studies stated; the interpretation cautious; a registry search proposed.

**Time:** 25 minutes.

### Exercise 6 — Indirect comparison (calculation, 35 minutes)

**Deliverable:** an indirect estimate of A versus B from A versus P (-0.30, SE 0.12) and B versus P (-0.15, SE 0.10), with a confidence interval and a transitivity assessment.

**Acceptance criteria:** arithmetic correct; confidence interval correct; transitivity discussed.

**Time:** 35 minutes.

### Exercise 7 — Network meta-analysis (design, 45 minutes)

**Deliverable:** a network diagram and an analysis plan for four treatments with six trials, including the model, the consistency assessment, and the ranking method.

**Acceptance criteria:** network connected; model stated; consistency test specified; ranking method described.

**Time:** 45 minutes.

### Exercise 8 — GRADE rating (application, 35 minutes)

**Deliverable:** a GRADE rating for the primary outcome of the worked example, with the reasons for each domain.

**Acceptance criteria:** all five domains addressed; rating justified; reasons specific.

**Time:** 35 minutes.

## 13. Self-check answers

**Answer 1 (PICO).** Population: adults with heart failure and reduced ejection fraction. Intervention: SGLT2 inhibitors. Comparator: placebo or usual care. Outcome: cardiovascular death or heart failure hospitalisation at 12 months. Designs: randomised controlled trials, with observational studies as supportive evidence.

**Answer 2 (search).** MEDLINE, Embase, and the Cochrane Central Register of Controlled Trials, plus ClinicalTrials.gov and the WHO ICTRP. Search terms include the drug class, heart failure, and the outcomes. The date range is inception to the search date. Grey literature includes conference abstracts and regulatory documents.

**Answer 3 (meta-analysis).** The fixed-effect weights are 100, 44.4, and 25. The fixed-effect estimate is (-0.4*100 - 0.3*44.4 - 0.5*25) / 169.4 = -0.386. Q is about 0.74 on 2 degrees of freedom, so I-squared is 0 and tau-squared is 0. The random-effects estimate equals the fixed-effect estimate.

**Answer 4 (heterogeneity).** An I-squared of 70 percent indicates substantial heterogeneity. A prediction interval that includes no effect means a new study could show no benefit, even if the pooled estimate is significant. The recommendation should be cautious, and the source of heterogeneity should be investigated.

**Answer 5 (publication bias).** With eight studies, the funnel plot is unreliable. The asymmetry is suggestive but not conclusive. The review should search trial registries for unpublished studies and report the limitation.

**Answer 6 (indirect).** The indirect estimate is -0.30 - (-0.15) = -0.15. The standard error is the square root of 0.12^2 + 0.10^2 = 0.156. The 95 percent confidence interval is -0.456 to 0.156. Transitivity requires that the populations and outcomes in the two comparisons are similar.

**Answer 7 (NMA).** The network should be connected, with a common comparator or a loop. The model should be a random-effects NMA, with a consistency assessment using a loop-specific test or a design-by-treatment interaction model. The ranking should be reported with uncertainty, not as a definitive order.

**Answer 8 (GRADE).** Risk of bias: some concerns, downgrade one level. Inconsistency: moderate for A versus P, no downgrade or one level. Indirectness: direct, no downgrade. Imprecision: wide for B versus P, downgrade one level. Publication bias: possible, downgrade one level. The overall certainty is moderate to low.

## 14. Interview questions

**Question 1. What is the difference between a systematic review and a meta-analysis?**

A strong answer: a systematic review is a structured search and appraisal; a meta-analysis is a statistical combination. A review can be systematic without a meta-analysis. A weak answer treats them as synonyms.

**Question 2. Why is a protocol and registration important?**

A strong answer: they prevent selective methods and selective reporting, and they allow comparison of the planned and published review. A weak answer says "it is good practice."

**Question 3. What is heterogeneity, and how do you assess it?**

A strong answer: real variation in effect across studies, assessed with Q, I-squared, tau-squared, and a prediction interval, plus clinical review. A weak answer says "I-squared above 50 percent."

**Question 4. When would you use a random-effects model?**

A strong answer: when effects are expected to vary across studies and the goal is the mean of a distribution; report tau-squared and a prediction interval. A weak answer says "always" or "never."

**Question 5. What is publication bias, and how do you assess it?**

A strong answer: selective publication of positive studies; assess with a funnel plot, Egger test where appropriate, and a registry search; acknowledge the limits with few studies. A weak answer says "funnel plot" without the limits.

**Question 6. What is an indirect comparison, and what is its key assumption?**

A strong answer: a comparison through a common comparator, assuming transitivity of populations, interventions, and outcomes. A weak answer says "comparing two drugs through placebo" without the assumption.

**Question 7. What is network meta-analysis?**

A strong answer: a model that combines direct and indirect evidence across a network of treatments, requiring connectivity, transitivity, and consistency; it can rank treatments but the ranking must be interpreted with uncertainty. A weak answer says "a meta-analysis with many drugs."

**Question 8. What is GRADE, and why use it?**

A strong answer: a framework for rating certainty across risk of bias, inconsistency, indirectness, imprecision, and publication bias; it calibrates the recommendation. A weak answer says "a quality score."

**Question 9. Why can a precise meta-analysis be wrong?**

A strong answer: because precision addresses random error, not bias; a synthesis of biased studies can be precise and wrong. A weak answer says "because of heterogeneity."

**Question 10. How would you explain a ranking to a payer?**

A strong answer: as a probabilistic summary with overlapping intervals, not a definitive order; the decision should consider certainty, cost, and applicability. A weak answer presents the SUCRA as a probability of being best without qualification.

## 15. Key terms

- **Certainty:** confidence that an estimate is correct for the question.
- **Consistency:** agreement between direct and indirect evidence.
- **Direct evidence:** head-to-head comparison.
- **Egger test:** a regression test for funnel plot asymmetry.
- **Fixed-effect model:** a model that assumes one true effect.
- **Funnel plot:** a plot of effect against precision.
- **GRADE:** a framework for rating certainty.
- **Heterogeneity:** real variation in effect across studies.
- **I-squared:** the proportion of variability due to heterogeneity.
- **Imprecision:** an estimate too uncertain for the decision.
- **Indirect comparison:** a comparison through a common comparator.
- **Indirectness:** evidence that does not match the question.
- **Inconsistency:** disagreement between direct and indirect evidence.
- **Loop:** a closed path in a network that allows a consistency test.
- **Meta-analysis:** statistical combination of study estimates.
- **Network meta-analysis (NMA):** a model combining direct and indirect evidence.
- **Network geometry:** the structure of treatments and comparisons.
- **P-score:** a frequentist analogue of SUCRA.
- **PICO:** population, intervention, comparator, outcome.
- **Prediction interval:** the range of effects expected in a new study.
- **PRISMA 2020:** reporting guideline for systematic reviews.
- **PROSPERO:** international register of systematic review protocols.
- **Publication bias:** selective publication of positive results.
- **Random-effects model:** a model that estimates the mean and between-study variance.
- **Risk of bias:** the likelihood that a result is materially biased.
- **RoB 2:** Cochrane risk-of-bias tool for randomised trials.
- **ROBINS-I:** risk-of-bias tool for non-randomised studies.
- **Star network:** a network with one common comparator.
- **SUCRA:** surface under the cumulative ranking curve.
- **Systematic review:** a review with a prespecified search and appraisal.
- **Tau-squared:** the between-study variance.
- **Transitivity:** the assumption that indirect comparisons are valid.
- **Trim-and-fill:** a method to estimate and adjust for publication bias.

## 16. Sources

All URLs accessed 25 October 2026.

### Guidelines and standards

- Page MJ, McKenzie JE, Bossuyt PM, et al. The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. BMJ 2021;372:n71. https://doi.org/10.1136/bmj.n71
- Higgins JPT, Thomas J, Chandler J, et al. Cochrane Handbook for Systematic Reviews of Interventions. Version 6.5, 2024. https://training.cochrane.org/handbook
- Guyatt GH, Oxman AD, Vist GE, et al. GRADE: an emerging consensus on rating quality of evidence and strength of recommendations. BMJ 2008;336:924-926. https://doi.org/10.1136/bmj.39489.470347.AD
- Jansen JP, Fleurence R, Devine B, et al. Interpreting indirect treatment comparisons and network meta-analysis for health-care decision making. Value Health 2011;14:417-428. https://doi.org/10.1016/j.jval.2011.04.002
- Hoaglin DC, Hawkins N, Jansen JP, et al. Conducting indirect-treatment-comparison and network-meta-analysis studies. Value Health 2011;14:429-437. https://doi.org/10.1016/j.jval.2011.03.003
- NICE, Guide to the methods of technology appraisal, 2013 (updated). https://www.nice.org.uk/process/pmg9
- European Union, Regulation (EU) 2021/2282 on health technology assessment. https://eur-lex.europa.eu/eli/reg/2021/2282/oj

### Methods

- Sterne JAC, Savović J, Page MJ, et al. RoB 2: a revised tool for assessing risk of bias in randomised trials. BMJ 2019;366:l4898. https://doi.org/10.1136/bmj.l4898
- Sterne JAC, Hernán MA, Reeves BC, et al. ROBINS-I: a tool for assessing risk of bias in non-randomised studies of interventions. BMJ 2016;355:i4919. https://doi.org/10.1136/bmj.i4919
- DerSimonian R, Laird N. Meta-analysis in clinical trials. Control Clin Trials 1986;7:177-188. https://doi.org/10.1016/0197-2456(86)90046-2
- Higgins JPT, Thompson SG. Quantifying heterogeneity in a meta-analysis. Stat Med 2002;21:1539-1558. https://doi.org/10.1002/sim.1186
- Salanti G, Ades AE, Ioannidis JPA. Graphical methods and numerical summaries for presenting results from multiple-treatment meta-analysis. J Clin Epidemiol 2011;64:163-171. https://doi.org/10.1016/j.jclinepi.2010.03.016

### Case evidence

- Turner EH, Matthews AM, Linardatos E, et al. Selective publication of antidepressant trials and its influence on apparent efficacy. N Engl J Med 2008;358:252-260. https://doi.org/10.1056/NEJMsa065779
- Bresalier RS, Sandler RS, Quan H, et al. Cardiovascular events associated with rofecoxib in a colorectal adenoma chemoprevention trial. N Engl J Med 2005;352:1092-1102. https://doi.org/10.1056/NEJMoa050493
- RECOVERY Collaborative Group. Dexamethasone in Hospitalized Patients with Covid-19. N Engl J Med 2021;384:693-704. https://doi.org/10.1056/NEJMoa2021436

### Further reading

- Borenstein M, Hedges LV, Higgins JPT, Rothstein HR. Introduction to Meta-Analysis. 2nd edition. Wiley, 2021.
- Dias S, Ades AE, Welton NJ, et al. Network Meta-Analysis for Decision-Making. Wiley, 2018.
- Salanti G, Del Giovane C, Chaimani A, et al. Evaluating the quality of evidence from a network meta-analysis. PLoS One 2014;9:e99682. https://doi.org/10.1371/journal.pone.0099682
- Ioannidis JPA. The mass production of redundant, misleading, and conflicted systematic reviews and meta-analyses. Milbank Q 2016;94:485-514. https://doi.org/10.1111/1468-0009.12210
- Page MJ, Sterne JAC, Higgins JPT, Egger M. Investigating and dealing with publication bias and other reporting biases in meta-analyses of health research: a review. Res Synth Methods 2021;12:248-259. https://doi.org/10.1002/jrsm.1468
- Chaimani A, Caldwell DM, Li T, et al. Additional considerations are required when preparing a protocol for a systematic review with multiple interventions. J Clin Epidemiol 2017;83:65-74. https://doi.org/10.1016/j.jclinepi.2016.11.015
- Nikolakopoulou A, Higgins JPT, Papakonstantinou T, et al. CINeMA: an approach for assessing confidence in the results of a network meta-analysis. PLoS Med 2020;17:e1003082. https://doi.org/10.1371/journal.pmed.1003082
- Sterne JAC, Sutton AJ, Ioannidis JPA, et al. Recommendations for examining and interpreting funnel plot asymmetry in meta-analyses of randomised controlled trials. BMJ 2011;343:d4002. https://doi.org/10.1136/bmj.d4002
- Guyatt GH, Oxman AD, Kunz R, et al. What is "quality of evidence" and why is it important to clinicians? BMJ 2008;336:995-998. https://doi.org/10.1136/bmj.39490.551019.BE
- Schünemann HJ, Oxman AD, Brozek J, et al. Grading quality of evidence and strength of recommendations for diagnostic tests and strategies. BMJ 2008;336:1106-1110. https://doi.org/10.1136/bmj.39500.677199.AE
- Higgins JPT, Thompson SG, Deeks JJ, Altman DG. Measuring inconsistency in meta-analyses. BMJ 2003;327:557-560. https://doi.org/10.1136/bmj.327.7414.557
- Riley RD, Higgins JPT, Deeks JJ. Interpretation of random effects meta-analyses. BMJ 2011;342:d549. https://doi.org/10.1136/bmj.d549
- Veroniki AA, Jackson D, Viechtbauer W, et al. Methods to estimate the between-study variance and its uncertainty in meta-analysis. Res Synth Methods 2016;7:55-79. https://doi.org/10.1002/jrsm.1164

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`08_external-validation.md`](08_external-validation.md)
- Next topic: [`10_protocol-writing.md`](10_protocol-writing.md)
- Related topic: [`05_bias-taxonomy.md`](05_bias-taxonomy.md)
- Module survey file: [`../../05_study_design_evidence.md`](../../05_study_design_evidence.md)
- Labs: [`../../labs/`](../../labs/)
