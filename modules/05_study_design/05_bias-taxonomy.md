# Bias Taxonomy

> Module 05 — Study Design and Evidence Generation
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 02 and 03
> Version 2026.3

## 1. Why this topic exists

Bias is a systematic difference between the quantity a study estimates and the quantity it claims to estimate. It is not random error. Random error shrinks with sample size; bias does not. A study with a million participants can produce a precise estimate of the wrong quantity, and the precision makes the error more convincing, not less.

The commercial consequence is direct. A product launched on a biased estimate will underperform in the market, and the underperformance will be attributed to the product rather than to the study. A drug approved on a surrogate that was biased by attrition will fail in post-marketing studies. A model validated on a biased sample will fail at the site that bought it. Bias is therefore not an academic concern; it is a commercial risk that is paid for years after the study reports.

Regulators have codified the assessment. The Cochrane risk-of-bias tool for randomised trials (RoB 2) organizes bias into five domains. The ROBINS-I tool for non-randomised studies organizes it into seven domains and treats each study as an attempt to emulate a target trial. The GRADE approach uses risk of bias as one of five domains for rating certainty. The reporting guidelines, such as CONSORT and STROBE, require the information needed to assess bias. These instruments are conventions and guidance, not statutes, but they define the review standard.

The practical problem is that bias is easier to name than to detect. A protocol can list every bias in a textbook and still miss the one that matters, because the bias operates through a specific design choice: who was eligible, when follow-up began, how the outcome was measured, or which participants were excluded. This topic teaches a systematic assessment that connects each bias to a design feature and a diagnostic, so that the assessment produces a mitigation rather than a list.

### Bias, error, and uncertainty

Three quantities are often confused.

- **Random error** is the variability of an estimate around its expected value. It is quantified by the standard error and reduced by sample size.
- **Bias** is the difference between the expected value of an estimate and the true value. It is not reduced by sample size.
- **Uncertainty** is the combination of random error and the uncertainty about the assumptions that could produce bias. It is quantified by confidence intervals and sensitivity analyses.

A confidence interval addresses random error. It does not address bias. A study can report a narrow confidence interval around a biased estimate, and the interval will be narrow and wrong. This is why a sensitivity analysis is not optional: it is the only way to express uncertainty about bias.

### Why a taxonomy

A taxonomy is useful because it makes the assessment repeatable. Without one, reviewers assess the biases they remember. With one, they assess the biases that are relevant to the design, in a fixed order, with a diagnostic for each.

The taxonomy in this topic has six families:

1. **Selection bias:** who enters the study, and who does not.
2. **Confounding:** a common cause of exposure and outcome.
3. **Information bias:** measurement error in exposure, outcome, or covariates.
4. **Attrition bias:** who leaves the study, and why.
5. **Time-related bias:** immortal time, lead time, length time, and prevalent-user bias.
6. **Reporting and knowledge bias:** selective reporting, publication bias, automation bias, and spectrum bias.

The families overlap. Immortal time is a form of selection bias; spectrum bias is a form of selection bias that affects diagnostic performance. The taxonomy is a practical organization, not a set of mutually exclusive categories.

### The direction and magnitude question

Naming a bias is not enough. The assessment must state:

- the direction of the bias (toward the null, away from the null, or unpredictable),
- the plausible magnitude,
- the design feature that mitigates it,
- the sensitivity analysis that quantifies the residual.

A bias that is named but not quantified cannot be weighed against the benefit. A bias that is quantified but not mitigated cannot be fixed. The assessment must do both.

### The cost of bias

- A biased effect estimate can lead to a product launch that fails in the market.
- A biased validation can lead to a model that underperforms at the purchasing site.
- A biased safety analysis can delay a product or expose patients to harm.
- A biased economic model can lead to a coverage decision that is later reversed.
- A biased literature can mislead guideline developers and clinicians for years.
- A biased claim can damage the credibility of the team and the organisation.

The costs are not symmetrical. A false negative delays a useful product; a false positive can cause harm and destroy trust. The assessment should weight the two accordingly.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** random error, bias, and uncertainty, and explain why a confidence interval does not address bias. (Exercise 1, self-check 1)
2. **Draw** a causal diagram for a study and identify confounding, selection, and collider paths. (Exercise 2, self-check 2)
3. **Classify** a bias into the six families and map it to a design feature and a diagnostic. (Exercise 3, self-check 3)
4. **Assess** a study with ROBINS-I or RoB 2 and justify the domain judgements. (Exercise 4, interview 4)
5. **Quantify** the direction and plausible magnitude of a bias, and choose a sensitivity analysis. (Exercise 5, self-check 5)
6. **Detect** lead-time, length-time, and immortal-time bias in a survival or screening comparison. (Exercise 6, self-check 6)
7. **Recognize** automation bias and spectrum bias in AI evaluation, and propose a mitigation. (Exercise 7, interview 7)
8. **Write** a bias register for a protocol with residual risk and a monitoring plan. (Exercise 8, interview 8)

### How the outcomes build on each other

- Outcomes 1 and 2 establish the concepts and the causal model.
- Outcomes 3 and 4 provide the classification and the appraisal method.
- Outcomes 5 and 6 provide the quantification and the time-bias diagnostics.
- Outcomes 7 and 8 apply the taxonomy to AI and to protocol writing.

A learner who can perform outcomes 1 to 4 can assess a study. A learner who can also perform outcomes 5 to 8 can design a study that anticipates its biases.

## 3. The core mental model

Bias enters through a design choice. The model below traces the path from the target quantity to the observed estimate, with the points at which each bias family enters.

```text
   TARGET QUANTITY
   effect in the population of interest
              |
              |  selection: who is eligible, who is observed
              v
   STUDY POPULATION
              |
              |  confounding: common causes of exposure and outcome
              v
   COMPARISON GROUPS
              |
              |  information: measurement of exposure, outcome, covariates
              v
   MEASURED DATA
              |
              |  attrition: who leaves, and when
              v
   ANALYSIS SAMPLE
              |
              |  time: when follow-up starts, how time is counted
              v
   ESTIMATE
              |
              |  reporting: which analyses are reported
              v
   PUBLISHED CLAIM
```

Each arrow is a place where the estimate can diverge from the target. The design features that control each arrow are the subject of Section 5.

### The causal diagram

A causal diagram, or directed acyclic graph (DAG), represents the assumed causal relationships among variables. It is the most useful single tool for identifying confounding and selection bias, because it makes the assumptions visible.

The rules are simple:

- A **confounder** is a common cause of exposure and outcome. It creates a backdoor path that must be blocked by adjustment or design.
- A **collider** is a common effect of two variables. Adjusting for a collider opens a path and creates bias.
- A **mediator** lies on the causal path from exposure to outcome. Adjusting for it removes part of the effect.
- **Selection** into the study can be a collider if it is caused by both exposure and outcome.

A DAG is not a statistical model. It is a statement of assumptions. Two analysts can draw different DAGs for the same question, and the difference is the substance of the debate.

### The three families of bias in a comparison

For any comparison, bias can enter through three structural features.

**Who is compared.** If the exposed and unexposed groups differ in prognosis for reasons other than the exposure, the comparison is confounded. Randomisation addresses this by design; observational studies address it by adjustment, with residual uncertainty.

**How exposure and outcome are measured.** If the measurement differs between groups, the comparison is biased. Blinding addresses this for outcome assessment; validated instruments address it for exposure.

**Who remains in the analysis.** If the groups differ in who is lost to follow-up, the analysis sample is selected. Intention-to-treat and complete follow-up address this; imputation and sensitivity analysis quantify the residual.

### The bias register

A bias register is a table with one row per bias and the following columns: bias, mechanism, direction, magnitude, design mitigation, sensitivity analysis, residual risk, owner. It is the operational form of the taxonomy.

The register is more useful than a narrative limitations section because it forces the team to state the direction and magnitude, and it assigns a mitigation to each row. A register with ten rows and no mitigations is a list of worries, not an assessment.

### Why bias assessment fails

Four failure patterns recur.

**The textbook list.** The protocol lists every bias from a textbook, with no connection to the design. The reviewer cannot tell which biases matter.

**The directionless assessment.** The protocol says a bias "may affect" the result, without stating the direction. A directionless bias cannot be weighed.

**The unquantified residual.** The protocol says a bias was "addressed" by adjustment, without a sensitivity analysis. Adjustment reduces bias; it does not eliminate it.

**The post hoc discovery.** The bias is discovered after the result is known, and the analysis is changed to address it. The change may be correct, but it is not prespecified, and the result is vulnerable to the criticism that the analysis was chosen to produce a preferred answer.

### Bias families at a glance

| Family | Entry point | Typical design response | Typical diagnostic |
|---|---|---|---|
| Selection | Eligibility, sampling, time zero | Consecutive sampling; restriction | Comparison with target population |
| Confounding | Assignment | Randomisation; adjustment | Baseline balance; E-value |
| Information | Measurement | Blinding; validation | Validation substudy |
| Attrition | Follow-up | Retention; complete follow-up | Completer analysis |
| Time | Time origin and exposure definition | New-user design; mortality endpoint | Timeline; null cohort |
| Reporting | Analysis and publication | Registration; SAP | Protocol comparison |
| Knowledge | Workflow and interface | Human factors; training | Error injection; audit |

The table is a map, not a checklist. A study may have several families operating at once, and the assessment must consider their interaction.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Bias | A systematic difference between an estimate and the target quantity | Used as a synonym for error or uncertainty | Effect-measure units |
| Random error | Variability of an estimate around its expected value | Confused with bias | Standard error |
| Confounding | A common cause of exposure and outcome | Confused with selection bias | Bias category |
| Confounder | A variable that causes both exposure and outcome | Any variable associated with the outcome | Variable role |
| Collider | A common effect of two variables | Adjusted for as if it were a confounder | Variable role |
| Mediator | A variable on the causal path from exposure to outcome | Adjusted for as if it were a confounder | Variable role |
| Directed acyclic graph (DAG) | A diagram of assumed causal relationships | Treated as a statistical model | Diagram |
| Backdoor path | A non-causal path from exposure to outcome | Ignored in adjustment | Path |
| Selection bias | Bias from how participants enter or remain in the study | Confused with confounding | Bias category |
| Information bias | Bias from measurement error | Confused with selection bias | Bias category |
| Measurement error | A difference between the measured and true value | Assumed to be random | Variable-specific |
| Non-differential error | Measurement error unrelated to the other variable | Assumed to bias toward the null | Bias category |
| Differential error | Measurement error related to the other variable | Direction depends on the pattern | Bias category |
| Attrition bias | Bias from differential loss to follow-up | Confused with missing data | Bias category |
| Immortal time bias | Bias from person-time during which the outcome could not occur | Confused with confounding | Bias category |
| Lead-time bias | Apparent survival gain from earlier diagnosis | Confused with mortality benefit | Bias category |
| Length-time bias | Apparent survival gain from detecting slower-growing cases | Confused with mortality benefit | Bias category |
| Overdiagnosis | Detection of a condition that would never cause symptoms | Confused with early detection | Bias category |
| Prevalent-user bias | Bias from including people already on treatment | Confused with confounding | Bias category |
| Spectrum bias | Performance differences caused by case mix | Ignored in diagnostic validation | Bias category |
| Recall bias | Differential memory of exposure by outcome status | Confused with information bias generally | Bias category |
| Publication bias | Selective publication of positive results | Confused with reporting bias | Bias category |
| Reporting bias | Selective reporting of outcomes or analyses | Confused with publication bias | Bias category |
| Automation bias | Over-reliance on an automated recommendation | Treated as a user error, not a design issue | Bias category |
| Anchoring | Fixation on an initial value | Confused with automation bias | Cognitive bias |
| Confirmation bias | Seeking information that supports a prior belief | Confused with reporting bias | Cognitive bias |
| Risk of bias | The likelihood that a study's result is materially biased | Treated as a synonym for quality | Judgement |
| RoB 2 | Cochrane risk-of-bias tool for randomised trials | Used for non-randomised studies | Tool |
| ROBINS-I | Risk-of-bias tool for non-randomised studies of interventions | Used for randomised trials | Tool |
| GRADE | Framework for rating certainty of evidence | Treated as a risk-of-bias tool only | Framework |
| Sensitivity analysis | An analysis that tests the robustness of the result to an assumption | Confused with a subgroup analysis | Analysis |
| Quantitative bias analysis | A model-based correction for a specified bias | Confused with a sensitivity analysis | Method |
| E-value | The minimum confounding strength needed to explain away a result | Treated as a p-value | Ratio |
| Residual confounding | Confounding that remains after adjustment | Assumed to be zero after adjustment | Bias category |
| Healthy-user bias | Healthier people are more likely to receive treatment | Confused with confounding generally | Bias category |
| Informative missingness | Missingness related to the outcome | Treated as random | Bias category |
| Verification bias | Only some patients receive the reference standard | Ignored in diagnostic studies | Bias category |
| Review bias | The reference standard is interpreted with knowledge of the test | Confused with spectrum bias | Bias category |
| Incorporation bias | The test is part of the reference standard | Ignored in diagnostic studies | Bias category |

Three distinctions deserve emphasis.

**Confounder versus mediator.** A confounder must be adjusted for; a mediator must not be adjusted for if the total effect is the target. Adjusting for a mediator removes part of the effect and can reverse its sign. The distinction is a causal one, not a statistical one, and it must be made from subject-matter knowledge.

**Non-differential versus differential error.** Non-differential measurement error in a binary exposure usually biases the effect toward the null. Differential error can bias in either direction. The assumption of non-differential error is common and often untested; the sensitivity analysis should test it.

**Publication versus reporting bias.** Publication bias is the selective publication of whole studies. Reporting bias is the selective reporting of outcomes or analyses within a study. Both distort the literature, and both are addressed by registration and prespecified analysis plans.

## 5. The mechanics, step by step

The procedure has fourteen steps. Steps 1 to 4 set up the target and the causal model. Steps 5 to 11 assess each bias family. Steps 12 to 14 quantify, mitigate, and report.

### Procedure at a glance

| Step | Input | Output | Failure if skipped |
|---:|---|---|---|
| 1 | Claim and decision | Estimand and target trial | No reference for bias |
| 2 | Subject-matter knowledge | Causal diagram | Unjustified adjustment set |
| 3 | Design and data | Comparison and time zero | Immortal time or selection bias |
| 4 | Design | Bias register | Unsytematic assessment |
| 5 | Sampling | Selection judgement | Non-representative sample |
| 6 | DAG and covariates | Adjustment set and residual analysis | Confounding |
| 7 | Measurement instruments | Information-bias judgement | Misclassification |
| 8 | Follow-up | Attrition and missing-data plan | Differential loss |
| 9 | Timeline | Time-bias assessment | Lead, length, immortal time |
| 10 | Protocol and registration | Reporting-bias assessment | Selective reporting |
| 11 | Workflow and interface | Human-AI assessment | Automation bias |
| 12 | Diagnostics | Direction and magnitude | Unweighable bias |
| 13 | Register | Mitigation and sensitivity analyses | Unaddressed residual |
| 14 | All of the above | Residual-risk statement | Overconfident claim |

### Bias assessment worksheet

| Column | What to record | Example |
|---|---|---|
| Bias | Name and family | Confounding by indication |
| Mechanism | How it produces a difference | Sicker patients receive the drug |
| Direction | Toward null, away, or unpredictable | Away from null |
| Magnitude | Plausible range | 10 to 30 percent relative inflation |
| Design mitigation | What was done to prevent it | Active comparator |
| Analysis mitigation | What was done to quantify it | IPW and E-value |
| Diagnostic | Evidence that it was assessed | Baseline table; negative control |
| Residual risk | What remains after mitigation | Moderate unmeasured frailty |
| Owner | Who is accountable | Statistical lead |
| Review date | When it will be revisited | Before database lock |

The worksheet is the operational form of the register. A register with ten rows and no diagnostics is a list of concerns; a worksheet with diagnostics and residual risk is an assessment.

### Bias mitigation by design stage

| Stage | Mitigation | Bias addressed | Residual |
|---|---|---|---|
| Question | Define the estimand and target trial | All | Specification error |
| Eligibility | Consecutive, broad, explicit criteria | Selection | Non-participation |
| Allocation | Randomisation; concealed allocation | Confounding; selection | Chance imbalance |
| Blinding | Participant, clinician, assessor, analyst | Information; performance | Unblinding |
| Follow-up | Complete, protocolised, retention | Attrition | Informative loss |
| Outcome | Validated, objective, adjudicated | Information | Measurement error |
| Time zero | Aligned to treatment initiation | Immortal time | Grace-period error |
| Analysis | Prespecified; ITT; sensitivity analyses | Reporting; attrition | Residual confounding |
| Reporting | Registration; SAP; data sharing | Reporting; publication | Unreported analyses |

### Common errors in bias assessment

- Naming a bias without stating its direction.
- Adjusting for a mediator and calling it confounding control.
- Adjusting for a collider and creating bias.
- Assuming non-differential misclassification without evidence.
- Treating a completed-case analysis as ITT.
- Reporting survival from diagnosis in a screening study.
- Ignoring the possibility of overdiagnosis.
- Assessing automation bias without observing the workflow.
- Treating a nonsignificant sensitivity analysis as proof of no bias.
- Discovering the bias after the result and presenting the fix as prespecified.

### Quantitative bias analysis

A quantitative bias analysis models the bias explicitly and corrects the estimate. For misclassification, the analyst specifies the sensitivity and specificity of the outcome definition and computes the corrected effect. For unmeasured confounding, the analyst specifies the confounder-exposure and confounder-outcome associations and computes the bias-adjusted estimate.

The analysis is only as good as its assumptions. It should be presented as a range, not a point estimate, and the assumptions should be varied in a sensitivity analysis. The value of the analysis is that it converts an unquantified worry into a bounded range that a decision-maker can use.

### Assessing a published study

Use the following checklist when appraising a published study for bias.

- What is the target trial, and how does the study diverge from it?
- Who was eligible, and who was excluded?
- When did follow-up begin, and was it aligned across groups?
- How were exposure and outcome measured, and were the measurements blinded?
- What was the loss to follow-up, and was it differential?
- What confounders were measured, and what was the adjustment set?
- Was the analysis prespecified, and does the publication match the protocol?
- Are the survival, incidence, and mortality results consistent?
- Is there a human-factors assessment, if the intervention is AI?
- Does the conclusion match the design and the residual risk?

A study that fails several of these questions is not necessarily invalid, but the claim must be weakened to reflect the uncertainty.

### Step 1: State the estimand and the target trial

**Input:** the claim and the decision.

**Output:** a five-attribute estimand and a target trial specification.

**Rationale:** Bias is defined relative to a target. Without the target, the assessment has no reference point. The target trial makes the design choices explicit and shows where the study diverges.

The estimand should include the population, treatment conditions, variable, intercurrent events, and summary measure. The target trial should include eligibility, strategies, assignment, follow-up, outcome, contrast, and analysis. The divergence between the target trial and the actual study is the bias assessment.

### Step 2: Draw the causal diagram

**Input:** the estimand and subject-matter knowledge.

**Output:** a DAG with exposure, outcome, confounders, mediators, colliders, and selection variables.

**Rationale:** The DAG identifies the adjustment set and the variables that must not be adjusted for. It also makes the assumptions visible to reviewers.

The DAG should be drawn before the analysis plan is written. If the DAG is drawn after the results, it is likely to be shaped by the results. The protocol should include the DAG and the justification for the adjustment set.

### Step 3: Define the comparison and time zero

**Input:** the target trial and the data.

**Output:** a comparison definition with a single time zero and a follow-up rule.

**Rationale:** Many biases enter through the comparison definition. A comparison that starts follow-up at different times for the two groups creates immortal time. A comparison that includes prevalent users creates survivor bias. A comparison that defines exposure by a future event creates selection bias.

The time-zero definition should be the same for both groups, and it should be the moment at which the treatment strategies diverge. Any deviation should be documented and tested in a sensitivity analysis.

### Step 4: Enumerate the bias domains

**Input:** the design and the DAG.

**Output:** a bias register with the domains that are relevant to the design.

**Rationale:** The enumeration ensures that the assessment is systematic. It also prevents the common failure of assessing only the biases that are easy to measure.

The domains are:

| Domain | Core question | Typical design feature |
|---|---|---|
| Selection | Who enters the study, and who does not? | Eligibility, sampling, time zero |
| Confounding | What common causes are unbalanced? | Randomisation, adjustment |
| Information | How are exposure and outcome measured? | Blinding, validation |
| Attrition | Who leaves, and why? | Follow-up, retention |
| Time | When does follow-up start, and how is time counted? | Time zero, exposure definition |
| Reporting | Which analyses are reported? | Registration, SAP |
| Knowledge | How do users interact with the result? | Human factors, training |

### Step 5: Assess selection bias

**Input:** the eligibility criteria, the sampling method, and the time-zero definition.

**Output:** a judgement on selection bias with a direction and a diagnostic.

**Rationale:** Selection bias occurs when the study population differs from the target population in a way that is related to both exposure and outcome. It can arise from eligibility criteria, from the sampling frame, from differential participation, or from the definition of time zero.

Diagnostics include:

- comparison of the study population with the target population,
- comparison of participants and non-participants,
- a flow diagram with exclusion counts,
- a sensitivity analysis reweighting to the target population.

A study that enrolls volunteers is not necessarily biased, but it is biased if volunteering is related to both exposure and outcome. The assessment must state the mechanism, not just the label.

### Step 6: Assess confounding

**Input:** the DAG and the measured covariates.

**Output:** an adjustment set, a balance assessment, and a residual confounding analysis.

**Rationale:** Confounding is the most common threat to causal claims from observational data. It cannot be eliminated by adjustment, only reduced. The residual must be quantified.

Diagnostics include:

- a baseline table before and after adjustment,
- a propensity score overlap plot,
- an E-value or a bias analysis,
- a negative control outcome.

The adjustment set should be justified by the DAG. Adjusting for every available variable is not a solution; it can introduce collider bias and over-adjustment.

### Step 7: Assess information bias

**Input:** the measurement instruments and the data collection process.

**Output:** a measurement-error assessment with a direction and a validation plan.

**Rationale:** Information bias occurs when exposure, outcome, or covariates are measured with error that differs between groups or is related to the other variable. It is common in RWD, where codes and questionnaires are used for purposes other than research.

Diagnostics include:

- a validation substudy comparing the measure with a reference standard,
- a positive control outcome,
- a comparison of alternative definitions,
- a quantitative bias analysis for misclassification.

The assessment should distinguish non-differential from differential error and state the expected direction.

### Step 8: Assess attrition and missing data

**Input:** the follow-up procedures and the missingness pattern.

**Output:** an attrition assessment with a missing-data plan.

**Rationale:** Attrition bias occurs when loss to follow-up is related to both exposure and outcome. It is common in long trials and in RWD with discontinuous enrolment.

Diagnostics include:

- a comparison of completers and non-completers,
- the missingness rate by group and by reason,
- a sensitivity analysis under plausible missingness mechanisms,
- a tipping-point analysis.

Complete-case analysis is rarely adequate for a confirmatory claim. The plan should state the primary handling method and the sensitivity analyses.

### Step 9: Assess time-related biases

**Input:** the time-zero definition and the exposure definition.

**Output:** a time-bias assessment with a timeline diagram and a null-cohort check.

**Rationale:** Time-related biases are common and often invisible in the analysis. They include immortal time, lead time, length time, and prevalent-user bias.

Diagnostics include:

- a timeline diagram for exposed and unexposed participants,
- a null-cohort simulation to test for immortal time,
- a comparison of survival from diagnosis versus mortality,
- a check of the preclinical detectable phase and screening interval.

A study that reports improved survival after screening must show that mortality, not only survival, changed. A study that defines exposure by a future event must show that pre-exposure time is not counted as exposed.

### Step 10: Assess reporting and publication bias

**Input:** the protocol, the registration, and the publication record.

**Output:** a reporting-bias assessment with a comparison of prespecified and reported analyses.

**Rationale:** Reporting bias occurs when the reported analyses differ from the prespecified ones, or when studies are published selectively. It distorts the literature and the evidence base.

Diagnostics include:

- comparison of the protocol and the publication,
- a funnel plot or a selection model,
- a search of trial registries for unpublished studies,
- a comparison of the primary endpoint in the registry and the paper.

### Step 11: Assess knowledge and automation bias

**Input:** the user interface, the workflow, and the human factors.

**Output:** a human-AI interaction assessment with a mitigation plan.

**Rationale:** Automation bias occurs when users over-rely on an automated recommendation, accepting errors they would otherwise catch. It is a property of the socio-technical system, not only of the individual user.

Diagnostics include:

- a usability study with error injection,
- an audit of override and acceptance rates,
- a comparison of decisions with and without the model,
- a review of near misses.

A model that is accurate in isolation can be harmful in use if it induces over-reliance. The assessment should include the workflow, not only the algorithm.

### Step 12: Quantify direction and magnitude

**Input:** the bias register and the diagnostics.

**Output:** a direction and a plausible range for each material bias.

**Rationale:** A bias without a direction cannot be weighed. A bias without a magnitude cannot be compared with the effect.

The direction can be:

- toward the null,
- away from the null,
- unpredictable,
- dependent on the magnitude of another parameter.

The magnitude can be estimated from a validation study, a sensitivity analysis, or a quantitative bias analysis. When it cannot be estimated, the assessment should state the range over which the conclusion would change.

### Step 13: Design mitigation and sensitivity analyses

**Input:** the bias register and the quantified directions.

**Output:** a mitigation for each material bias and a sensitivity analysis for each mitigation.

**Rationale:** Mitigation can be by design or by analysis. Design mitigation is preferred because it removes the bias rather than adjusting for it. Analysis mitigation quantifies the residual.

| Bias | Design mitigation | Analysis mitigation |
|---|---|---|
| Selection | Consecutive sampling; broad eligibility | Reweighting; restriction |
| Confounding | Randomisation; active comparator | Adjustment; E-value; negative control |
| Information | Blinding; validated instruments | Bias analysis; alternative definitions |
| Attrition | Retention; complete follow-up | Imputation; tipping point |
| Immortal time | New-user design; time-zero alignment | Time-varying analysis |
| Lead time | Randomised screening; mortality endpoint | Lead-time adjustment |
| Length time | Randomised screening; incidence endpoint | Stage-specific analysis |
| Reporting | Registration; SAP lock | Comparison with protocol |
| Automation | Human factors; error injection | Audit; override analysis |

### Step 14: Report residual risk

**Input:** all of the above.

**Output:** a residual-risk statement with the claim's boundaries.

**Rationale:** The report must state what the evidence supports after the biases are considered. A claim that ignores residual risk is overconfident; a claim that refuses to make any statement is useless.

The statement should say:

- which biases are material,
- which are controlled and how,
- which remain and in what direction,
- what result would change the conclusion,
- what the claim does and does not support.

### Edge cases

**Non-inferiority trials.** A non-inferiority trial is especially vulnerable to information bias and attrition, because these biases move the estimate toward the null and can make an ineffective treatment appear non-inferior. The assessment should include assay sensitivity and a per-protocol analysis alongside ITT.

**Cluster trials.** Contamination between clusters, differential recruitment, and cluster-level attrition are additional biases. The assessment should include a contamination measure and a cluster-level flow diagram.

**Diagnostic accuracy studies.** Spectrum bias, review bias, and incorporation bias are the main threats. The assessment should include the sampling method, the reference standard, and the blinding of readers.

**AI validation studies.** Data leakage, spectrum bias, and automation bias are the main threats. The assessment should include the data split, the case mix, the calibration, and the human-AI interaction.

**Screening studies.** Lead-time, length-time, and overdiagnosis bias are the main threats. The assessment should include cancer mortality, not only survival, and an analysis of incidence.

### Decision table: observed pattern to likely bias

| Observed pattern | Likely bias | Diagnostic | Design response |
|---|---|---|---|
| Survival improves but mortality is unchanged | Lead time or length time | Mortality rate; stage shift | Randomised screening; mortality endpoint |
| Incidence rises with no mortality change | Overdiagnosis | Cumulative incidence; autopsy data | Report overdiagnosis; informed consent |
| Effect shrinks after adjustment | Confounding | Baseline table; E-value | Better confounders; active comparator |
| Effect reverses after adjustment | Collider or over-adjustment | DAG review | Remove collider; re-specify |
| Effect larger in completers | Attrition or adherence | Completer versus non-completer | ITT; imputation |
| Effect appears only in one site | Spectrum or implementation | Case-mix table | Multi-site validation |
| Effect disappears with objective outcome | Detection bias | Blinded assessment | Blinded adjudication |
| Effect appears only in small studies | Publication bias | Funnel plot | Registry search; trim-and-fill |

The table is a heuristic. A pattern can have more than one explanation, and the diagnostic must distinguish them.

## 6. Worked example

### The claim

A health system's marketing team says its prostate cancer screening programme "improves 5-year survival from 52 percent to 88 percent." The claim is based on a before-after comparison: 40,000 men in the pre-screening era and 40,000 men in the screening era, with 5-year survival measured from diagnosis. The programme wants to use the claim to justify a payer contract.

### Step 1: Estimand and target trial

The implicit estimand is the effect of the screening programme on 5-year survival among men diagnosed with prostate cancer. The target trial would randomise men to invitation to screening versus usual care and measure survival from a common time zero.

The actual study measures survival from diagnosis, not from randomisation, and compares two eras rather than two randomised groups. The divergence is the source of the bias.

### Step 2: Causal diagram

The relevant diagram is simple:

```text
   Screening programme ----> Earlier diagnosis ----> Longer survival from diagnosis
          |                                                    ^
          |                                                    |
          +------------------> True mortality ------------------+
```

The path from screening to longer survival from diagnosis goes through earlier diagnosis. The path from screening to true mortality is the causal effect of interest. The study measures the first path, not the second.

### Step 3: Comparison and time zero

The pre-screening cohort has time zero at diagnosis. The screening cohort also has time zero at diagnosis, but the diagnosis occurs earlier in the disease course. The comparison is therefore not aligned: the screening cohort has a longer observed survival because it is observed from an earlier point.

### Step 4: Bias domains

The relevant domains are lead time, length time, overdiagnosis, and confounding by era. Selection bias is also possible if the screening era includes a different population.

### Step 5: Selection

The screening era may include men who would not have been diagnosed in the pre-screening era, because screening detects asymptomatic cancers. This is overdiagnosis, which inflates the denominator with good-prognosis cases and improves survival.

### Step 6: Confounding

The two eras differ in treatment, diagnostic practice, and population. Treatment for prostate cancer changed substantially between eras, and the change may affect mortality independently of screening. The before-after comparison cannot separate screening from treatment.

### Step 7: Information

Survival is measured from diagnosis in both eras, but the diagnosis date is earlier in the screening era. The outcome is measured consistently, but the time origin is not.

### Step 8: Attrition

Men who are diagnosed by screening may be more likely to be treated and followed, which can affect survival. The before-after design does not control for this.

### Step 9: Time-related biases

The lead-time bias is the dominant one. The arithmetic is straightforward.

Assume:

- The preclinical detectable phase is 4 years.
- The time from clinical diagnosis to death is exponentially distributed with a mean of 8 years.
- Screening detects the cancer at the start of the preclinical phase.
- Screening does not change the time of death.

For the unscreened cohort, 5-year survival from diagnosis is:

```text
P(survival > 5 years) = exp(-5 / 8) = 0.535
```

For the screened cohort, diagnosis occurs 4 years earlier, so 5-year survival from diagnosis is the probability of surviving more than 1 year from the original clinical diagnosis:

```text
P(survival > 5 years from screen diagnosis)
  = P(survival > 1 year from clinical diagnosis)
  = exp(-1 / 8)
  = 0.882
```

The simulation in Section 7 gives 52.1 percent and 88.4 percent, consistent with the arithmetic. The number of deaths within 10 years is identical in both cohorts: 678 per 40,000. The survival gain is an artefact of the earlier diagnosis, not a mortality benefit.

### Step 10: Length-time bias

The example becomes worse when tumours grow at different rates. Slow-growing tumours have a longer preclinical phase and are more likely to be detected by screening. Fast-growing tumours may be missed between screening rounds.

Assume two tumour types:

- 60 percent slow: preclinical phase 8 years, survival mean 12 years.
- 40 percent fast: preclinical phase 1 year, survival mean 3 years.

Screening detects all slow tumours at the start of the preclinical phase and 50 percent of fast tumours. The simulation gives 5-year survival of 46.3 percent without screening and 68.6 percent with screening, with identical mortality (1,049 deaths per 40,000 in both). The survival gain is now a combination of lead time and length time.

### Step 11: Overdiagnosis

Screening also detects cancers that would never have caused symptoms. These cases are counted as survivors, inflating survival and incidence without affecting mortality. The magnitude depends on the tumour type and the screening intensity. A prostate cancer screening programme can have a substantial overdiagnosis rate; the exact rate is debated and varies by age and screening interval.

### Step 12: Direction and magnitude

The direction of the lead-time bias is away from the null: survival is overstated. The magnitude is approximately the lead time divided by the survival time, which in the example is 4/8 = 0.5, turning a 53.5 percent 5-year survival into 88.2 percent.

The direction of the length-time bias is also away from the null. The direction of the overdiagnosis bias is away from the null for survival and incidence, and neutral for mortality. The confounding by era is unpredictable.

### Step 13: Mitigation

The design mitigation is a randomised trial of invitation to screening, with cancer mortality as the primary endpoint and a common time zero at randomisation. The analysis mitigation for an observational study is a mortality-to-incidence ratio, a stage-shift analysis, and a lead-time adjustment.

A before-after study cannot support a mortality claim, regardless of the sample size. The correct claim from the before-after data is that survival from diagnosis improved, which is compatible with lead time, length time, overdiagnosis, and treatment change. It is not evidence that screening reduced mortality.

### Step 14: Residual risk

The report should state that the observed survival gain is consistent with lead-time and length-time bias, that the mortality data do not show a corresponding reduction, and that the claim of a mortality benefit is not supported. The residual uncertainty is the possibility of a small mortality benefit that the study is not powered to detect.

### What the data can and cannot support

| Statement | Supported by the before-after data? | Reason |
|---|---|---|
| 5-year survival from diagnosis improved | Yes | The observed survival is higher |
| Mortality from prostate cancer fell | No | The mortality count is unchanged |
| Screening caused the survival improvement | No | Lead time, length time, and treatment change are alternatives |
| Earlier diagnosis is beneficial | No | Earlier diagnosis without mortality benefit may be overdiagnosis |
| The programme should be reimbursed for mortality benefit | No | The mortality claim is unsupported |
| The programme may improve survival awareness | Not tested | The study did not measure that outcome |

The table is the discipline that a bias assessment imposes. It separates what the data show from what the programme wants to claim.

### Overdiagnosis

Overdiagnosis is the detection of a cancer that would never have caused symptoms or death. It is not the same as a false positive: a false positive is a test result that is later found to be negative, while overdiagnosis is a true cancer that would not have progressed.

Overdiagnosis has three consequences:

- it increases incidence without increasing mortality,
- it improves survival by adding good-prognosis cases to the denominator,
- it causes unnecessary treatment and anxiety.

The magnitude is difficult to estimate without a randomised trial or an autopsy series. A screening programme should report an overdiagnosis estimate and the assumptions behind it.

### Two diagnostics that reveal a flawed design

**Diagnostic 1: the mortality check.** The first diagnostic is to compare all-cause or cause-specific mortality between the two eras. If survival improves but mortality is unchanged, the most likely explanation is lead time, length time, or overdiagnosis. In the example, the mortality count is identical (678 deaths in both cohorts), which is the signature of a survival artefact.

**Diagnostic 2: the incidence check.** The second diagnostic is to compare cancer incidence. If incidence rises sharply after screening without a corresponding mortality decline, overdiagnosis is likely. A stage-shift analysis can distinguish a true stage shift from an artefact: if the proportion of early-stage cancers rises but the proportion of late-stage cancers does not fall, the shift may be overdiagnosis rather than early detection.

### Vary one assumption: the lead time

If the lead time is 2 years rather than 4, the screened 5-year survival is the probability of surviving more than 3 years from clinical diagnosis:

```text
P(survival > 5 from screen diagnosis) = exp(-3 / 8) = 0.687
```

The survival gain falls from 34.7 percentage points to 15.2 percentage points. A shorter lead time produces a smaller artefact, but the direction is unchanged. The mortality remains identical.

### Vary one assumption: the survival time

If the mean survival from clinical diagnosis is 4 years rather than 8, the unscreened 5-year survival is exp(-5/4) = 0.287, and the screened survival with a 4-year lead is exp(-1/4) = 0.779. The absolute gain is larger, even though the lead time is the same, because the survival distribution is steeper. The artefact is larger when the disease is more aggressive.

### Vary one assumption: the screening interval

If the screening interval is 2 years and the fast tumour's preclinical phase is 1 year, half of the fast tumours are missed. If the interval is 1 year, more fast tumours are detected, and the length-time bias is smaller. The interval is a design choice that affects both the benefit and the bias.

### The decision

The payer rejects the mortality claim and asks for a randomised or well-controlled study with cancer mortality as the endpoint. The programme can still claim that survival from diagnosis improved, but it must state that the improvement is compatible with earlier diagnosis and does not establish a mortality benefit.

The example shows the central lesson of the bias taxonomy: the outcome that is easiest to measure is often the one most vulnerable to bias. Survival from diagnosis is easy to measure and highly susceptible to lead time. Mortality is harder to measure and more robust.

## 7. Code, query, or calculation

The code below simulates a screening cohort with a fixed lead time and a fixed time of death. It shows that 5-year survival improves while mortality is unchanged. It then adds two tumour types to show length-time bias. It uses only the Python standard library.

```python
import math
import random


def make_cohort(n=40000, seed=5, lead=4.0, mean_survival=8.0, cancer_prob=0.10):
    """Generate cancer cases with onset and death times."""
    random.seed(seed)
    people = []
    for _ in range(n):
        if random.random() < cancer_prob:
            onset = random.uniform(0, 10)
            death = onset + lead + random.expovariate(1 / mean_survival)
            people.append((onset, death))
    return people


def survival_stats(people, screened, lead=4.0, horizon=10.0):
    """5-year survival from diagnosis and deaths within the horizon."""
    n = survived = deaths = 0
    for onset, death in people:
        diagnosis = onset if screened else onset + lead
        n += 1
        if death - diagnosis > 5:
            survived += 1
        if death <= horizon:
            deaths += 1
    return {
        "cases": n,
        "five_year_survival": round(survived / n, 3),
        "deaths_within_10y": deaths,
    }


def make_heterogeneous_cohort(n=40000, seed=6, cancer_prob=0.10):
    """Two tumour types with different preclinical and survival times."""
    random.seed(seed)
    people = []
    for _ in range(n):
        if random.random() < cancer_prob:
            slow = random.random() < 0.6
            lead = 8 if slow else 1
            mean_survival = 12 if slow else 3
            onset = random.uniform(0, 10)
            death = onset + lead + random.expovariate(1 / mean_survival)
            people.append((slow, lead, onset, death))
    return people


def heterogeneous_stats(people, screened, seed=99):
    """Screening detects all slow tumours and half of fast tumours."""
    random.seed(seed)
    n = survived = deaths = 0
    for slow, lead, onset, death in people:
        if screened:
            detected = slow or random.random() < 0.5
            diagnosis = onset if detected else onset + lead
        else:
            diagnosis = onset + lead
        n += 1
        if death - diagnosis > 5:
            survived += 1
        if death <= 10:
            deaths += 1
    return {
        "cases": n,
        "five_year_survival": round(survived / n, 3),
        "deaths_within_10y": deaths,
    }


if __name__ == "__main__":
    cohort = make_cohort()
    print("lead-time simulation, 4-year lead")
    print("unscreened:", survival_stats(cohort, screened=False))
    print("screened:  ", survival_stats(cohort, screened=True))

    hetero = make_heterogeneous_cohort()
    print("length-time simulation, two tumour types")
    print("unscreened:", heterogeneous_stats(hetero, screened=False))
    print("screened:  ", heterogeneous_stats(hetero, screened=True))
```

Expected output:

```text
lead-time simulation, 4-year lead
unscreened: {'cases': 3935, 'five_year_survival': 0.521, 'deaths_within_10y': 678}
screened:   {'cases': 3935, 'five_year_survival': 0.884, 'deaths_within_10y': 678}
length-time simulation, two tumour types
unscreened: {'cases': 4054, 'five_year_survival': 0.463, 'deaths_within_10y': 1049}
screened:  {'cases': 4054, 'five_year_survival': 0.686, 'deaths_within_10y': 1049}
```

The output is the central demonstration of this topic.

- In the lead-time simulation, 5-year survival rises from 52.1 percent to 88.4 percent, while deaths within 10 years are identical at 678. The survival gain is entirely an artefact of earlier diagnosis.
- In the length-time simulation, the survival gain is smaller because the fast tumours are less likely to be detected, but it is still 22.3 percentage points with identical mortality.
- The case count is identical in both arms of each simulation because the same people are analysed; only the diagnosis date changes.

A real screening evaluation would also model overdiagnosis, which would increase the case count in the screened arm and further inflate survival. The code does not model overdiagnosis because it requires assumptions about the fraction of cancers that would never progress, and those assumptions are the subject of debate.

### Calculating lead-time bias by hand

The arithmetic for the exponential case is:

```text
S_unscreened(5) = exp(-5 / mean_survival)
S_screened(5)   = exp(-(5 - lead) / mean_survival)
```

For mean survival 8 and lead 4:

```text
S_unscreened = exp(-0.625) = 0.535
S_screened   = exp(-0.125) = 0.882
```

The ratio of the survival probabilities is exp(lead / mean_survival) = exp(0.5) = 1.649. A longer lead time or a shorter survival time produces a larger artefact.

The arithmetic assumes that screening does not change the time of death. If screening also advances treatment and reduces mortality, the observed survival gain combines the true benefit and the lead-time artefact. Separating them requires a randomised design with a mortality endpoint.

### A quantitative bias analysis for misclassification

If the outcome is measured with known sensitivity and specificity, the observed counts can be corrected. For a binary outcome with sensitivity `se` and specificity `sp`, the corrected number of events is:

```text
true_positives = (observed_positives - (1 - sp) * n) / (se + sp - 1)
```

The correction is only valid if the misclassification is non-differential and the sensitivity and specificity are known. The code below applies the formula to a hypothetical validation study.

```python
def correct_events(observed_positives, n, sensitivity, specificity):
    """Correct an observed event count for non-differential misclassification."""
    denominator = sensitivity + specificity - 1
    if denominator <= 0:
        raise ValueError("sensitivity + specificity must exceed 1")
    return (observed_positives - (1 - specificity) * n) / denominator


if __name__ == "__main__":
    observed = 200
    n = 1000
    for se, sp in ((0.85, 0.98), (0.70, 0.99), (0.95, 0.95)):
        corrected = correct_events(observed, n, se, sp)
        print(f"sensitivity={se} specificity={sp} "
              f"corrected_events={corrected:.1f} "
              f"corrected_risk={corrected/n:.3f}")
```

Expected output:

```text
sensitivity=0.85 specificity=0.98 corrected_events=216.9 corrected_risk=0.217
sensitivity=0.70 specificity=0.99 corrected_events=275.4 corrected_risk=0.275
sensitivity=0.95 specificity=0.95 corrected_events=166.7 corrected_risk=0.167
```

The correction increases the event count when sensitivity is below 1, because the observed count misses true events. It can decrease the count when specificity is low, because false positives inflate the observed count. The direction depends on the trade-off between sensitivity and specificity, which is why the assumptions must be stated.

## 8. Data, standards, and regulatory context

### Binding requirements

**Clinical trial regulation.** The EU Clinical Trials Regulation (EU) No 536/2014 and the US investigational new drug and device regulations (21 CFR Parts 312 and 812) govern the conduct of trials and require the information needed to assess bias, including the protocol, the randomization plan, and the analysis plan. These are binding.

**Good clinical practice.** ICH E6(R3), adopted at Step 4 on 6 January 2025, requires the sponsor to manage risks to data quality and to describe the measures taken to prevent and detect bias. The guideline is binding where adopted.

**Device regulation.** The EU Medical Device Regulation (EU) 2017/745 requires clinical evaluation, and the US premarket approval and De Novo pathways require valid scientific evidence. Both require the manufacturer to address bias in the evidence.

### Formal guidance and tools

**RoB 2.** The Cochrane risk-of-bias tool for randomised trials has five domains: bias arising from the randomization process, bias due to deviations from intended interventions, bias due to missing outcome data, bias in measurement of the outcome, and bias in selection of the reported result. It is a convention used in systematic reviews and health technology assessment.

**ROBINS-I.** The tool for non-randomised studies of interventions has seven domains: confounding, selection of participants, classification of interventions, deviations from intended interventions, missing data, measurement of outcomes, and selection of the reported result. It treats each study as an attempt to emulate a target trial. It is a convention, not a regulation, but it is widely used by HTA bodies and guideline developers.

**GRADE.** The Grading of Recommendations Assessment, Development and Evaluation framework rates certainty across five domains: risk of bias, inconsistency, indirectness, imprecision, and publication bias. It is a convention used in guidelines and HTA.

**STROBE and CONSORT.** The reporting guidelines require the information needed to assess bias, including the flow of participants, the definitions of exposure and outcome, and the analysis plan. They are conventions enforced by journals.

### RoB 2 domains

| Domain | Question | Typical evidence |
|---|---|---|
| Randomisation process | Was allocation random and concealed? | Sequence generation; allocation concealment |
| Deviations from intended interventions | Were participants analysed as randomised? | Protocol deviations; ITT analysis |
| Missing outcome data | Were outcomes missing differentially? | Completeness by arm; reasons |
| Measurement of the outcome | Was outcome assessment blinded and valid? | Blinding; instrument validation |
| Selection of the reported result | Was the reported result prespecified? | Protocol; SAP; registry |

### ROBINS-I domains

| Domain | Question | Typical evidence |
|---|---|---|
| Confounding | Were all important confounders measured and adjusted? | DAG; baseline table; E-value |
| Selection of participants | Was selection related to exposure and outcome? | Sampling method; time zero |
| Classification of interventions | Was exposure defined without knowledge of outcome? | Exposure algorithm; validation |
| Deviations from intended interventions | Were deviations balanced and handled? | Adherence; per-protocol analysis |
| Missing data | Was missingness differential? | Completeness; imputation |
| Measurement of outcomes | Was outcome assessment biased? | Blinding; validation |
| Selection of the reported result | Was the result prespecified? | Protocol; registry |

### What the standards constrain and what they leave open

| Instrument | Constrains | Leaves open |
|---|---|---|
| ICH E6(R3) | Risk management and data quality | The specific biases and their magnitude |
| RoB 2 | The domains for randomised trials | The judgement within each domain |
| ROBINS-I | The domains for non-randomised studies | The target trial and the adjustment set |
| GRADE | The certainty domains | The rating within each domain |
| STROBE and CONSORT | The reporting content | The interpretation |

### GRADE certainty domains

| Domain | Question | Downgrade when |
|---|---|---|
| Risk of bias | Are the studies biased? | Serious or critical risk in key domains |
| Inconsistency | Do the results agree? | Unexplained heterogeneity |
| Indirectness | Do the studies match the question? | Different population, intervention, or outcome |
| Imprecision | Are the estimates precise? | Wide confidence interval; few events |
| Publication bias | Are negative studies missing? | Funnel plot asymmetry; registry gaps |

The table is a reminder that bias is one of five domains, not the whole assessment. A study can be at low risk of bias and still provide indirect or imprecise evidence.

The standards provide a structure, not a conclusion. The conclusion depends on the specific design and data.

### Data standards for bias assessment

Bias assessment depends on metadata: who was eligible, who was excluded, when follow-up began, how outcomes were ascertained, and who was lost. Standards such as the Observational Medical Outcomes Partnership (OMOP) Common Data Model and the Clinical Data Interchange Standards Consortium (CDISC) standards improve the availability of these metadata. They do not guarantee that the metadata are complete.

A bias assessment should cite the source of each metadata element. "Follow-up began at the first dispensing" is a design statement; "follow-up began at the first record" is a data statement. The difference matters.

## 9. Industry practice and case evidence

### Hormone replacement therapy and coronary heart disease

Observational studies in the 1980s and 1990s reported that postmenopausal hormone replacement therapy was associated with a lower risk of coronary heart disease. The Women's Health Initiative, a large randomised trial, found that combined hormone therapy increased the risk of coronary heart disease in the early years of treatment and did not provide the expected benefit.

The documented fact is the divergence between the observational and randomised results. The inference is that the observational studies were confounded by socioeconomic status, health-seeking behaviour, and baseline health: women who used hormone therapy were healthier and more likely to engage with preventive care. The adjustment in the observational studies did not capture the full difference.

**Lesson.** Confounding by indication and healthy-user effects can produce a large apparent benefit that randomisation does not confirm. Observational adjustment reduces but does not eliminate the bias.

### Prostate cancer screening

Randomised trials of prostate-specific antigen (PSA) screening have produced mixed results. The European Randomised Study of Screening for Prostate Cancer reported a modest reduction in prostate cancer mortality with substantial overdiagnosis; the Prostate, Lung, Colorectal and Ovarian Cancer Screening Trial reported no significant reduction. The differences are debated and depend on screening intensity, follow-up, and population.

The documented fact is the mixed trial evidence and the overdiagnosis concern. The inference is that survival from diagnosis is not a valid endpoint for screening evaluation, because it is inflated by lead time, length time, and overdiagnosis. Mortality is the appropriate endpoint.

**Lesson.** A screening programme must be evaluated with a mortality endpoint and an overdiagnosis analysis. A survival comparison is not sufficient.

### The Epic Sepsis Model

Wong and colleagues externally validated a widely implemented proprietary sepsis prediction model at Michigan Medicine. The cohort included 27,697 patients with 38,455 hospitalisations; sepsis was present in 2,552 hospitalisations (7 percent). The hospitalisation-level area under the receiver operating characteristic curve (AUC) was 0.63 (95 percent confidence interval 0.62 to 0.64), compared with the vendor's reported 0.76 to 0.83. At the alert threshold of 6 or more, 18 percent of hospitalisations would trigger an alert, with sensitivity 33 percent and positive predictive value 12 percent.

The documented fact is the performance gap. The inference is that the development and deployment populations differed in case mix, coding, and care processes, and that the vendor's performance claim was not reproduced. The study did not measure automation bias, but the alert burden suggests a risk of over-reliance and alert fatigue.

**Lesson.** Spectrum bias and case-mix differences can invalidate a performance claim. External validation in the deployment population is necessary.

### AI triage and automation bias

Studies of clinical decision support have documented automation bias: clinicians are more likely to accept an incorrect recommendation when the system presents it confidently, and less likely to notice an error. The effect is larger when the clinician is under time pressure or when the recommendation is presented as authoritative.

The documented fact is the existence of automation bias in decision support. The inference for AI products is that accuracy in isolation is not enough; the interface, the explanation, and the workflow determine whether the model improves or harms decisions.

**Lesson.** Evaluate the human-AI system, not only the algorithm. Include error-injection usability studies and audit override rates.

### Case comparison

| Case | Bias | Design feature | What changed the interpretation |
|---|---|---|---|
| HRT and CHD | Confounding by indication | Observational cohort | Randomised trial showed no benefit |
| PSA screening | Lead time, length time, overdiagnosis | Survival from diagnosis | Mortality endpoint showed mixed results |
| Epic Sepsis Model | Spectrum bias | External validation | Performance fell in deployment |
| AI triage | Automation bias | Human-AI workflow | Accuracy did not guarantee benefit |

The table shows that the bias is often invisible in the original design and becomes visible only when a different design or diagnostic is applied. The lesson is to apply the diagnostic before the claim, not after.

### What the cases have in common

- The original result was published and influential.
- The bias was not visible from the effect estimate alone.
- A different design or diagnostic changed the interpretation.
- The correction required a new study or a new analysis.
- The commercial or clinical consequences were substantial.

The pattern suggests a practical rule: when an effect is surprisingly large or surprisingly convenient, test it against a design that could overturn it before building a claim on it.

### What companies actually do

The following practices are inferred from public protocols, guidance, and published studies.

1. **Bias registers are used in mature teams.** The register maps each bias to a design feature and a sensitivity analysis, and it is reviewed at phase gates.
2. **DAGs are used to justify adjustment sets.** Teams that draw a DAG before the analysis are less likely to over-adjust or adjust for colliders.
3. **Quantitative bias analysis is used for material biases.** When a bias could change the conclusion, a model-based correction is preferred to a narrative limitation.
4. **External validation is used to address spectrum bias.** A model validated only in the development site is not considered deployment-ready.
5. **Human factors studies are used for AI products.** Error injection, override audits, and workflow analysis are part of the evaluation.

### Where practice falls short

- **Long lists without direction.** A limitations section that names biases but does not state their direction or magnitude.
- **Adjustment for everything.** A model that adjusts for mediators and colliders, producing a biased estimate.
- **Survival as a screening endpoint.** A claim of mortality benefit based on survival from diagnosis.
- **No attrition analysis.** A trial that reports the primary result without a comparison of completers and non-completers.
- **No automation-bias assessment.** An AI product evaluated on accuracy alone, without a human-factors study.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Confounding by indication | Treated patients are sicker | Treatment assigned by prognosis | Biased effect | Randomisation; active comparator; E-value |
| Collider adjustment | Adjusting for a variable caused by exposure and outcome | Opens a non-causal path | Bias toward or away from null | DAG review; remove collider |
| Over-adjustment | Adjusting for mediators | Removes part of the effect | Attenuated effect | DAG review; separate total and direct effects |
| Selection bias | Volunteers or convenience sample | Participation related to exposure and outcome | Non-representative estimate | Consecutive sampling; reweighting |
| Immortal time | Exposure defined by a future event | Pre-exposure time counted as exposed | Spurious benefit | New-user design; time-zero alignment |
| Lead-time bias | Survival improves, mortality unchanged | Earlier diagnosis extends observed survival | False mortality benefit | Mortality endpoint; randomised screening |
| Length-time bias | Screening detects slow-growing cases | Preferential detection of good prognosis | False survival benefit | Randomised screening; incidence analysis |
| Overdiagnosis | Incidence rises without mortality decline | Detection of non-progressive disease | Unnecessary treatment; inflated survival | Overdiagnosis analysis; informed consent |
| Attrition bias | Differential dropout | Loss related to exposure and outcome | Biased estimate | Retention; imputation; tipping point |
| Information bias | Differential measurement | Measurement differs by group | Biased effect | Blinding; validation; bias analysis |
| Recall bias | Case-control design | Differential memory | Biased exposure estimate | Prospective measurement; records |
| Publication bias | Small positive studies only | Selective publication | Overstated effect | Registry search; funnel plot |
| Reporting bias | Endpoint switched | Selective analysis | False claim | Registration; SAP |
| Automation bias | High acceptance of wrong advice | Over-reliance on the model | Clinical error | Human factors; audit; explainability |
| Spectrum bias | Validation in a different case mix | Non-representative sample | Misleading performance | Multi-site validation; case-mix table |
| Verification bias | Only some patients receive the reference standard | Selected reference standard | Inflated accuracy | Consecutive reference standard |
| Healthy-user bias | Healthier people receive treatment | Unmeasured health behaviour | Spurious benefit | Active comparator; E-value |

### Red flags in a bias assessment

- The limitations section lists biases without direction or magnitude.
- The adjustment set is not justified by a causal diagram.
- The analysis adjusts for variables that could be mediators.
- The survival endpoint is measured from diagnosis in a screening study.
- The exposure is defined by a future event.
- The analysis population excludes participants who did not adhere.
- The completer analysis is presented as the main result.
- The publication does not match the registered protocol.
- The AI evaluation reports accuracy without a human-factors assessment.
- The validation population is not compared with the deployment population.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The target trial is correctly specified | Bias is defined relative to it | Wrong target or wrong estimand | Protocol review; estimand audit |
| The causal diagram is correct | Adjustment set depends on it | Missing confounder; wrong path | Subject-matter review; sensitivity analysis |
| The measured confounders capture the important ones | Adjustment reduces bias | Unmeasured confounding | E-value; negative control |
| The exposure definition reflects the causal contrast | The estimate depends on it | Misclassification; immortal time | Validation; alternative definitions |
| The outcome definition is valid | The estimate depends on it | Coding error; differential ascertainment | Validation study; positive control |
| Missingness is non-informative or handled | Complete-case analysis is biased otherwise | Differential dropout | Tipping point; imputation |
| The time origin is aligned | Time-related biases depend on it | Different start times | Timeline audit; null cohort |
| The reporting is complete | The literature is unbiased | Selective reporting | Registration; protocol comparison |
| The human-AI interaction is safe | Automation bias can harm | Over-reliance; alert fatigue | Usability study; audit |
| The validation population matches deployment | Performance depends on it | Spectrum shift | Case-mix table; external validation |
| The bias analysis model is correct | Quantitative correction depends on it | Wrong assumptions | Sensitivity over assumptions |
| The report distinguishes bias from imprecision | The claim depends on the distinction | Conflating the two | Separate bias and CI reporting |
| The assessment is reproducible | Reviewers must verify it | Undocumented judgements | Register with evidence for each row |

### Limits of this topic

This topic provides a taxonomy and an assessment procedure. It does not provide a complete theory of causal inference, a full course in epidemiologic methods, or a substitute for subject-matter expertise. The choice of adjustment set and the interpretation of a DAG require clinical and biological knowledge that this topic cannot supply.

The case evidence is drawn from public sources and is not a systematic review. The inferences about company practice are labelled as inferences. The regulatory statements are educational summaries, not legal advice.

## 12. Exercises

### Exercise 1 — Error, bias, and uncertainty (recall, 20 minutes)

**Deliverable:** a one-page explanation, with an example, of the difference between random error, bias, and uncertainty, and why a narrow confidence interval does not rule out bias.

**Acceptance criteria:** the three concepts distinguished; the example correct; the role of sensitivity analysis explained.

**Time:** 20 minutes.

### Exercise 2 — Draw a DAG (application, 30 minutes)

**Deliverable:** a DAG for the question "does a new diabetes drug reduce cardiovascular events?" including age, obesity, renal function, adherence, and health-service use, with the adjustment set identified.

**Acceptance criteria:** confounders, mediators, and colliders labelled; the adjustment set justified; at least one variable explicitly excluded with a reason.

**Time:** 30 minutes.

### Exercise 3 — Classify biases (application, 30 minutes)

**Deliverable:** a table classifying ten biases from a published study into the six families, with the design feature that creates each and one diagnostic.

**Acceptance criteria:** each classification justified; the design feature specific; the diagnostic feasible.

**Time:** 30 minutes.

### Exercise 4 — ROBINS-I assessment (critique, 45 minutes)

**Deliverable:** a ROBINS-I assessment of a published non-randomised study, with a judgement for each of the seven domains and an overall judgement.

**Acceptance criteria:** each domain judged with evidence; the target trial stated; the overall judgement consistent with the domains.

**Time:** 45 minutes.

### Exercise 5 — Direction and magnitude (calculation, 35 minutes)

**Deliverable:** for a study with 20 percent differential dropout in the treated arm, a statement of the likely direction of bias, a plausible magnitude, and a sensitivity analysis.

**Acceptance criteria:** direction justified; magnitude bounded; sensitivity analysis specified; assumptions stated.

**Time:** 35 minutes.

### Exercise 6 — Detect time bias (calculation, 35 minutes)

**Deliverable:** for a screening programme with a 5-year lead time and a mean survival of 10 years, compute the 5-year survival in the screened and unscreened groups under the null, and explain the result.

**Acceptance criteria:** arithmetic correct; the null assumption stated; the interpretation correct; the mortality implication stated.

**Time:** 35 minutes.

### Exercise 7 — Automation bias (design, 35 minutes)

**Deliverable:** a human-factors evaluation plan for a sepsis alert, including an error-injection task, an override audit, and a workflow observation.

**Acceptance criteria:** the error-injection scenario described; the outcomes specified; the analysis plan stated; the ethics considered.

**Time:** 35 minutes.

### Exercise 8 — Bias register (design, 60 minutes)

**Deliverable:** a bias register for a protocol of your choice, with at least ten rows, direction, magnitude, mitigation, sensitivity analysis, and residual risk.

**Acceptance criteria:** each row specific to the design; direction stated; mitigation feasible; residual risk quantified or bounded.

**Time:** 60 minutes.

## 13. Self-check answers

**Answer 1 (error, bias, uncertainty).** Random error is variability around the expected value and shrinks with sample size. Bias is a systematic difference between the expected value and the truth and does not shrink with sample size. Uncertainty includes both random error and uncertainty about assumptions. A narrow confidence interval addresses only the first; a biased estimate can have a narrow interval. Sensitivity analysis is needed to address bias.

**Answer 2 (DAG).** Age and obesity are confounders if they affect both treatment choice and cardiovascular events. Renal function may be a confounder or a mediator depending on the drug's mechanism. Adherence is a mediator if the drug works only when taken. Health-service use may be a collider if both treatment and outcome affect it. The adjustment set should include age, obesity, and renal function at baseline; adherence should not be adjusted for if the total effect is the target; health-service use should not be adjusted for if it is a collider.

**Answer 3 (classification).** A convenience sample is selection bias. Sick patients receiving the drug is confounding by indication. A diagnosis code with poor sensitivity is information bias. Differential dropout is attrition bias. Exposure defined by a future prescription is immortal time. A survival endpoint in a screening study is lead-time bias. A published study with a switched endpoint is reporting bias. Over-reliance on an alert is automation bias. Validation in a tertiary centre is spectrum bias. Selective publication of positive studies is publication bias.

**Answer 4 (ROBINS-I rubric).** A strong assessment states the target trial, judges each domain with evidence from the study, and gives an overall judgement consistent with the domains. A weak assessment lists the domains and gives judgements without evidence.

**Answer 5 (direction and magnitude).** If the treated arm has 20 percent dropout and dropouts are more likely to have poor outcomes, the observed treated group is healthier than the assigned group, and the effect is biased away from the null. The magnitude depends on the outcome rate in dropouts; a tipping-point analysis can bound it. The sensitivity analysis should impute outcomes under plausible scenarios and report the range.

**Answer 6 (time bias).** With a 5-year lead and a mean survival of 10 years, unscreened 5-year survival is exp(-5/10) = 0.607. Screened 5-year survival is exp(0/10) = 1.000, because the 5-year window from screen diagnosis ends at the clinical diagnosis date, before any excess mortality. The mortality is unchanged. The result shows that survival from diagnosis is not a valid endpoint for screening.

**Answer 7 (automation bias rubric).** A strong plan includes a simulated error case, a measurement of whether the clinician follows the erroneous advice, an audit of override and acceptance rates, and an observation of the workflow. It states the outcomes, the analysis, and the ethics. A weak plan says "survey the clinicians."

**Answer 8 (bias register rubric).** A strong register has at least ten rows specific to the design, with direction, magnitude, mitigation, sensitivity analysis, and residual risk. A weak register lists generic biases without a link to the design.

## 14. Interview questions

**Question 1. What is the difference between bias and random error?**

A strong answer: bias is systematic and does not shrink with sample size; random error is variability and does shrink. A weak answer treats them as synonyms.

**Question 2. Why can a large study be more misleading than a small one?**

A strong answer: because a large study has a narrow confidence interval around a biased estimate, which makes the bias more convincing. A weak answer says large studies are always better.

**Question 3. What is confounding by indication, and how do you address it?**

A strong answer: treatment is chosen because of prognosis, so the treated and comparison groups differ in baseline risk; address it with randomisation, an active comparator, or adjustment with a sensitivity analysis. A weak answer says "adjust for confounders."

**Question 4. What are the domains of ROBINS-I?**

A strong answer: confounding, selection of participants, classification of interventions, deviations from intended interventions, missing data, measurement of outcomes, and selection of the reported result. A weak answer names fewer than five.

**Question 5. What is immortal time bias, and how do you detect it?**

A strong answer: person-time during which the outcome could not occur because exposure is defined by a later event; detect it with a timeline audit and a null-cohort simulation. A weak answer defines it without a diagnostic.

**Question 6. Why is survival from diagnosis a poor endpoint for screening?**

A strong answer: because lead time, length time, and overdiagnosis inflate survival without changing mortality; the appropriate endpoint is cause-specific or all-cause mortality. A weak answer says survival is "biased" without the mechanisms.

**Question 7. What is automation bias?**

A strong answer: over-reliance on an automated recommendation, leading users to accept errors they would otherwise catch; it is a socio-technical property, assessed with error injection and override audits. A weak answer calls it "user error."

**Question 8. How do you handle unmeasured confounding?**

A strong answer: you cannot eliminate it; you quantify it with an E-value or bias analysis, probe it with negative controls, and weaken the claim accordingly. A weak answer says "add more variables."

**Question 9. What is spectrum bias?**

A strong answer: performance differs because the case mix differs from the intended-use population; it is addressed by validation in the target population and by reporting case mix. A weak answer confuses it with selection bias generally.

**Question 10. Why is a bias register better than a limitations paragraph?**

A strong answer: it forces direction, magnitude, mitigation, and residual risk for each bias, and it is auditable. A weak answer says it is "more organized."

## 15. Key terms

- **Anchoring:** fixation on an initial value.
- **Attrition bias:** bias from differential loss to follow-up.
- **Automation bias:** over-reliance on an automated recommendation.
- **Backdoor path:** a non-causal path from exposure to outcome.
- **Bias:** a systematic difference between an estimate and the target quantity.
- **Collider:** a common effect of two variables.
- **Confounder:** a common cause of exposure and outcome.
- **Confounding:** bias from a common cause of exposure and outcome.
- **Differential error:** measurement error related to the other variable.
- **Directed acyclic graph (DAG):** a diagram of assumed causal relationships.
- **E-value:** the minimum confounding strength needed to explain away a result.
- **GRADE:** a framework for rating certainty of evidence.
- **Immortal time bias:** bias from person-time during which the outcome could not occur.
- **Information bias:** bias from measurement error.
- **Lead-time bias:** apparent survival gain from earlier diagnosis.
- **Length-time bias:** apparent survival gain from detecting slower-growing cases.
- **Measurement error:** a difference between the measured and true value.
- **Mediator:** a variable on the causal path from exposure to outcome.
- **Non-differential error:** measurement error unrelated to the other variable.
- **Overdiagnosis:** detection of a condition that would never cause symptoms.
- **Prevalent-user bias:** bias from including people already on treatment.
- **Publication bias:** selective publication of positive results.
- **Quantitative bias analysis:** a model-based correction for a specified bias.
- **Random error:** variability of an estimate around its expected value.
- **Recall bias:** differential memory of exposure by outcome status.
- **Reporting bias:** selective reporting of outcomes or analyses.
- **Residual confounding:** confounding that remains after adjustment.
- **Risk of bias:** the likelihood that a result is materially biased.
- **RoB 2:** Cochrane risk-of-bias tool for randomised trials.
- **ROBINS-I:** risk-of-bias tool for non-randomised studies.
- **Selection bias:** bias from how participants enter or remain in the study.
- **Sensitivity analysis:** an analysis that tests robustness to an assumption.
- **Spectrum bias:** performance differences caused by case mix.

## 16. Sources

All URLs accessed 25 October 2026.

### Guidelines and tools

- ICH, E6(R3) Guideline for Good Clinical Practice, Step 4, 6 January 2025. https://database.ich.org/sites/default/files/ICH_E6%28R3%29_Step4_FinalGuideline_2025_0106.pdf
- Sterne JAC, Savović J, Page MJ, et al. RoB 2: a revised tool for assessing risk of bias in randomised trials. BMJ 2019;366:l4898. https://doi.org/10.1136/bmj.l4898
- Sterne JAC, Hernán MA, Reeves BC, et al. ROBINS-I: a tool for assessing risk of bias in non-randomised studies of interventions. BMJ 2016;355:i4919. https://doi.org/10.1136/bmj.i4919
- Guyatt GH, Oxman AD, Vist GE, et al. GRADE: an emerging consensus on rating quality of evidence and strength of recommendations. BMJ 2008;336:924-926. https://doi.org/10.1136/bmj.39489.470347.AD
- von Elm E, Altman DG, Egger M, et al. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement. Lancet 2007;370:1453-1457. https://doi.org/10.1016/S0140-6736(07)61602-X

### Methods

- Hernán MA, Robins JM. Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available. Am J Epidemiol 2016;183:758-764. https://doi.org/10.1093/aje/kwv254
- Suissa S. Immortal Time Bias in Pharmacoepidemiology. Am J Epidemiol 2008;167:492-499. https://doi.org/10.1093/aje/kwm324
- Rothman KJ, Greenland S, Lash TL. Modern Epidemiology. 3rd edition. Lippincott Williams and Wilkins, 2008.
- Lash TL, Fox MP, Fink AK. Applying Quantitative Bias Analysis to Epidemiologic Data. Springer, 2009.

### Case evidence

- Rossouw JE, Anderson GL, Prentice RL, et al. Risks and benefits of estrogen plus progestin in healthy postmenopausal women: principal results from the Women's Health Initiative randomized controlled trial. JAMA 2002;288:321-333. https://doi.org/10.1001/jama.288.3.321
- Schröder FH, Hugosson J, Roobol MJ, et al. Screening and prostate-cancer mortality in a randomized European study. N Engl J Med 2009;360:1320-1328. https://doi.org/10.1056/NEJMoa0810084
- Wong A, Otles E, Donnelly JP, et al. External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients. JAMA Intern Med 2021;181:1065-1070. https://doi.org/10.1001/jamainternmed.2021.2626

### AI and human factors

- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. BMJ 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
- Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI. BMJ 2025;388:e082505. https://doi.org/10.1136/bmj-2024-082505
- Goddard K, Roudsari A, Wyatt JC. Automation bias: a systematic review of frequency, effect mediators, and mitigators. J Am Med Inform Assoc 2012;19:121-127. https://doi.org/10.1136/amiajnl-2011-000089

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`04_validation-designs.md`](04_validation-designs.md)
- Next topic: [`06_sample-size-and-power.md`](06_sample-size-and-power.md)
- Related topic: [`03_real-world-evidence.md`](03_real-world-evidence.md)
- Module survey file: [`../../05_study_design_evidence.md`](../../05_study_design_evidence.md)
- Labs: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)
