# Methods

> Module 22 — Sources and Further Reading
> Topic 04 of 09
> Estimated study time: 3 hours
> Prerequisites: 01_how-to-read-this-list
> Version 2026.3

## 1. Why this topic exists

Methods sources look like the most stable part of a bibliography. A randomised trial is a fixed object, a regression model is a mathematical statement, and a textbook on causal inference does not change every year. That stability is misleading. Three things move underneath it:

- **The recommendation changes.** A method that was standard practice in 2015 can be discouraged in 2024, not because the mathematics changed but because the empirical evidence about its performance changed. Propensity score matching and AUC-only reporting are two examples.
- **The implementation changes.** A paper describes an estimator; a software package implements it with defaults that are not the paper's defaults. The citation to the paper does not tell you what the code did.
- **The reporting standard changes.** TRIPOD 2015 was superseded by TRIPOD+AI in 2024 for prediction models. A methods section that follows the older checklist can be complete by its own standard and incomplete by the current one.

The failures that follow are expensive and quiet:

- A model is reported with an area under the receiver operating characteristic curve (AUC) of 0.82 and no calibration, and the clinical team discovers at deployment that the probabilities are systematically inflated.
- A propensity-score analysis is cited as if matching removed confounding, when the method only balances the covariates that were measured and correctly specified.
- A bootstrap confidence interval is computed with the wrong resampling unit, because the package default was rows rather than patients.
- A methods section cites a paper for an estimator that the software implements differently.
- A systematic review is cited as evidence for a treatment effect when it pooled studies with different estimands.

The people affected are the statistician who cannot defend the analysis, the reviewer who cannot reproduce it, the regulatory lead who cannot map it to a guidance, and the clinician who cannot tell whether the output is a probability or a ranking.

This topic treats methods sources as a specific source category with three objects per citation: the idea, the implementation, and the reporting standard. A citation that names only one of the three is incomplete, and the incompleteness is where most reproducibility failures begin.

A note on what this topic is not. It is not a statistics course and it does not teach the mathematics of any estimator. It teaches how to read and cite a method so that the reasoning can be checked. A reader who wants the mathematics should use the primary sources in section 16.

The distinction matters because the failure this topic addresses is rarely mathematical. It is a citation that points at the right paper and the wrong object, or a report that is complete by an outdated standard. Those failures are visible from the document, without re-deriving the estimator.

The topics that follow apply the same three-link logic to clinical background, where the third link is a guideline rather than a checklist, and to datasets, where the third link is a version.

> **Note:** A methods paper is authority for what the method is and what its assumptions are. It is not authority for what your software did. The implementation has to be checked separately, with its version, its defaults, and its own tests.

## 2. Learning outcomes

By the end of this topic you can:

1. **Identify** the three objects in a methods citation — idea, implementation, and reporting standard — and state which one a sentence depends on. Assessed by Exercise 1 and self-check item 1.
2. **Distinguish** a method's assumptions from its performance claims, and state what would falsify the use of the method in a given analysis. Assessed by Exercise 2.
3. **Evaluate** a prediction-model claim using discrimination, calibration, and net benefit, and explain why AUC alone is insufficient for a clinical decision. Assessed by Exercise 3 and the worked example.
4. **Trace** a software default to its documentation and to the paper it claims to implement, and identify any divergence. Assessed by Exercise 4.
5. **Check** a methods section against the current reporting standard, including version, and name the items that changed. Assessed by Exercise 5.
6. **Detect** the common methods citation failures: paper cited for software behaviour, estimator name without assumptions, and superseded reporting guidance. Assessed by Exercise 6.
7. **Compute** operating characteristics and net benefit from a contingency table and interpret the result at the threshold a clinic would use. Assessed by Exercise 7 and section 7.
8. **Build** a methods source register with versions for the methods, the software, and the reporting standard used in a project. Assessed by Exercise 8.

## 3. The core mental model

A methods claim is supported by a chain with three links. The first link is the idea: the paper or text that defines the estimator and its assumptions. The second is the implementation: the software, version, and defaults that produced the numbers. The third is the reporting standard: the checklist that defines what a complete methods section contains.

```text
                    METHODS CLAIM
        "we estimated X with method Y and report Z"
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
      IDEA              IMPLEMENTATION        REPORTING
   paper or text        software, version,    checklist,
   assumptions,         defaults, tests       version, items
   estimand
        |                     |                     |
        +---------------------+---------------------+
                              |
                              v
                    CONSISTENCY CHECK
        does the implementation match the idea,
        and does the report match the standard?
                              |
                              v
                    CITATION SET
        method paper | software citation | checklist
```

Read the diagram across. The three links are not substitutes. A perfect method paper does not tell you what the package did, and a compliant report does not tell you whether the method was appropriate.

**Assumptions are the load-bearing part.** Every estimator carries assumptions: independent observations, no unmeasured confounding, correct functional form, positivity, or a particular missing-data mechanism. The assumption is what makes the estimate interpretable, and it is usually stated in the paper's methods section rather than its abstract. A citation that does not name the assumption is a citation that cannot be assessed.

**Performance claims are empirical and conditional.** A method's performance in simulation studies depends on the data-generating process used in the simulation. A paper showing that an estimator performs well under a particular set of conditions is not evidence that it performs well under yours. The conditions are part of the claim.

**Defaults are part of the method.** Software defaults encode choices: the number of bootstrap replicates, the handling of ties, the reference level for a categorical variable, the penalty strength, the convergence criterion. Two analyses with the same citation and different defaults are different analyses.

**Reporting standards have versions.** A checklist is a document with a version and a date. TRIPOD+AI supersedes TRIPOD 2015 for prediction models, and the items changed. A methods section judged against a superseded checklist can pass and still be incomplete.

A useful sentence template forces the three links into one claim: "We estimated [quantity] in [population] using [estimator], under [assumption], with [software version], reported per [checklist version]." A draft that cannot fill the template without a placeholder is a draft with a missing link.

The template is also a diagnostic for review. When a reviewer asks "what did you actually do", the answer should be reconstructable from the sentence, not from a conversation with the analyst.

| Link | What it establishes | What it cannot establish | Typical artefact |
|---|---|---|---|
| Idea | The estimator, its assumptions, and its estimand | What the software did | Method citation with section |
| Implementation | The computation actually performed | Whether the method suits the question | Software version and settings |
| Reporting | Completeness against a standard | Whether the analysis is correct | Checklist with version |

The three links also explain why methods errors are hard to see in a finished paper. A reader sees a citation and a result. The implementation is often a sentence in the supplementary material, and the standard is often not named at all, so two of the three links are invisible unless the reader looks for them.

## 4. Definitions and vocabulary

### 4.1 Study designs and estimands

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Randomised trial | Allocation by chance to compare interventions. | Assumed to answer any question about the intervention. | Design label |
| Target trial | The hypothetical trial a observational analysis emulates. | Omitted from observational studies. | Protocol |
| Estimand | The precise quantity being estimated, including population, outcome, and handling of intercurrent events. | Used as a synonym for "question". | Structured definition |
| Intention-to-treat | Analysis by assigned group regardless of adherence. | Confused with per-protocol. | Analysis set |
| Per-protocol | Analysis of those who adhered to the assigned strategy. | Presented as the main result without labelling. | Analysis set |
| Confounding | A common cause of exposure and outcome. | Addressed by adjusting for everything measured. | Directed acyclic graph |
| Positivity | Every covariate pattern has a non-zero probability of each exposure. | Assumed without checking. | Overlap diagnostic |
| Immortal time bias | Bias from requiring survival to a later exposure definition. | Common in observational drug studies. | Design feature |

### 4.2 Effect measures

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Risk difference | Absolute difference in event proportions. | Replaced by relative measures in communication. | Percentage points |
| Risk ratio | Ratio of risks between groups. | Interpreted as a rate ratio. | Ratio |
| Odds ratio | Ratio of odds; not the same as a risk ratio. | Read as a risk ratio when the outcome is common. | Ratio |
| Hazard ratio | Ratio of hazard functions under proportional hazards. | Interpreted as a risk ratio over the whole follow-up. | Ratio |
| Number needed to treat | Reciprocals of the risk difference. | Quoted without the time horizon. | Count |
| Confidence interval | A range compatible with the data at a stated confidence level. | Read as a probability about the parameter. | Interval |

### 4.3 Prediction-model metrics

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Discrimination | Ability to rank patients by risk. | Treated as the whole of model quality. | AUC or c-statistic |
| Calibration | Agreement between predicted probabilities and observed frequencies. | Omitted when discrimination is good. | Slope, intercept, curve |
| Calibration slope | The slope of observed on predicted; 1 is ideal. | Reported without the intercept. | Unitless |
| Calibration intercept | Systematic over- or under-prediction. | Ignored. | Log-odds or probability |
| Net benefit | The clinical value of a model at a decision threshold, accounting for false positives and false negatives. | Replaced by AUC in comparisons. | Proportion, in units of true positives per patient |
| Decision curve | Net benefit plotted across thresholds. | Treated as a significance test. | Curve |
| Brier score | Mean squared error of predicted probabilities. | Used without a reference model. | Unitless |
| AUC | Probability that a random case is ranked above a random non-case. | Described as accuracy or as clinical utility. | 0.5 to 1 |
| Precision | The proportion of flagged cases that are true cases. | Reported without the threshold. | Proportion |
| Recall | The proportion of true cases that are flagged. | Called sensitivity only in some fields. | Proportion |
| F1 score | Harmonic mean of precision and recall. | Treated as a decision metric. | 0 to 1 |
| Net reclassification | Change in the proportion correctly reclassified at a threshold. | Reported without the threshold. | Proportion |
| Integrated discrimination | Average improvement in predicted probabilities. | Confused with net benefit. | Unitless |

### 4.4 Reporting and appraisal instruments

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| TRIPOD+AI | Reporting guideline for prediction models using regression or machine learning, 2024. | Confused with the 2015 TRIPOD checklist. | 27 items |
| PROBAST | Risk-of-bias tool for prediction model studies. | Used as a reporting checklist. | Domains and signalling questions |
| CONSORT-AI | Reporting guideline for trials of AI interventions. | Applied to observational studies. | Checklist |
| SPIRIT-AI | Reporting guideline for trial protocols of AI interventions. | Confused with CONSORT-AI. | Checklist |
| STARD-AI | Reporting guideline for diagnostic accuracy studies of AI. | Applied to prediction models. | Checklist |
| DECIDE-AI | Reporting guideline for early clinical evaluation of AI. | Applied to full trials. | Checklist |
| PRISMA 2020 | Reporting guideline for systematic reviews. | Assumed to assess review quality. | Checklist |
| STROBE | Reporting guideline for observational studies. | Applied to prediction models. | Checklist |
| ICH E9(R1) | Estimand framework for clinical trials. | Treated as a statistical method rather than a specification. | Addendum |

### 4.5 Software and reproducibility terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Implementation | The software that computes the estimator. | Assumed to match the paper. | Package and version |
| Default | A setting used when the analyst does not specify one. | Assumed to be neutral. | Setting and value |
| Seed | The value that makes a random process reproducible. | Omitted from the report. | Integer |
| Version pin | Recording the exact software version used. | Replaced by "latest". | Version string |
| Unit of resampling | The entity resampled in a bootstrap or cross-validation. | Rows used when patients are the unit. | Entity key |
| Reproducibility | Same data, same code, same result. | Confused with replicability. | Artefact |
| Replicability | Same question, new data, consistent result. | Confused with reproducibility. | Study |
| Numerical convergence | Whether an iterative algorithm reached a stable solution. | Assumed when a result appears. | Convergence flag |
| Separation | A predictor perfectly separates outcomes. | Ignored, producing huge coefficients. | Diagnostic |
| Penalisation | Shrinkage applied to coefficients. | Assumed to be neutral. | Method and strength |
| Tuning | Selection of hyperparameters. | Done on the test set. | Procedure |
| Leakage | Outcome or future information in features. | Invisible in performance. | Audit |
| Environment | The software stack used. | Recorded as "Python". | Lock file |

### Questions to ask of any methods paper

- Does it define the estimator, or only apply it?
- What assumptions does the estimator require?
- What estimand does it target?
- What simulation conditions were used to assess performance?
- Has it been corrected or criticised since publication?
- Is there an implementation the authors maintain?
- What reporting standard applied when it was published?

The questions take about twenty minutes for a methods paper and save the much larger cost of discovering, after review, that the paper does not define what the analysis used.

## 5. The mechanics, step by step

The procedure has 24 steps in five phases. It follows the three-link model: establish the idea, establish the implementation, establish the reporting standard, then check that the three are consistent with each other and with the question.

Methods evaluation is different from regulatory evaluation because the object is not a legal status but a chain of reasoning. The failure mode is not "the document is superseded" but "the citation does not support the sentence, because the sentence is about the software and the citation is about the idea".

### Phase A — Frame the methods claim

**Step 1. Write the methods claim as a sentence with an estimand.**
Input: the analysis plan, the draft methods section, or the code.
Output: a sentence naming the population, the quantity estimated, the estimator, and the unit of analysis.
Rationale: "we used a mixed-effects model" is not a methods claim. "We estimated the association between exposure and 30-day readmission in adults, using a mixed-effects logistic model with hospital random intercepts" is.
Branch: if the sentence has no estimand, the analysis is not yet specified, and the citation check is premature.
Edge case: a descriptive analysis has no causal estimand, and the claim should say so rather than borrowing causal language.
Check: can a reader write the model equation from the sentence.
Record: the estimand and the unit of analysis.

**Step 2. Classify the methods claim by type.**
Input: the framed sentence.
Output: one of five types: estimation, prediction, causal inference, description, or software behaviour.
Rationale: the type determines which assumptions and which reporting standard apply.
Branch: a prediction claim needs calibration and net benefit; an estimation claim needs a confidence interval and a model specification; a causal claim needs identification assumptions.
Edge case: a single paper can contain an estimation claim and a prediction claim, and the two have different requirements.
Check: does the type match the verb in the sentence.
Record: the type and the assumptions it implies.

**Step 3. Identify the claim's decision context.**
Input: the intended use of the result.
Output: a statement of what the result will be used for and at what threshold or capacity.
Rationale: the decision context determines whether discrimination, calibration, or net benefit is the primary metric.
Branch: if the decision has a fixed capacity, precision at capacity is the relevant metric, not AUC.
Edge case: a research claim with no immediate decision still has a reader who will interpret the effect size.
Check: is the metric named in the plan appropriate to the decision.
Record: the decision context and the primary metric.

| Claim type | Load-bearing assumptions | Primary metric | Reporting standard |
|---|---|---|---|
| Prediction | Transportability, calibration in the target population | Discrimination, calibration, net benefit | TRIPOD+AI |
| Causal | No unmeasured confounding, positivity, consistency | Effect estimate with interval | Design-specific, often CONSORT or STROBE |
| Estimation | Correct model specification, independent observations | Coefficient with interval | Journal or field standard |
| Description | Representative sampling, complete capture | Counts and proportions with denominators | Field-specific |
| Software behaviour | Implementation matches the specification | Test output and agreement | Package documentation |

The table is not exhaustive, and it is useful because it makes the mismatch visible. A prediction study that reports only a hazard ratio, or a causal study that reports only AUC, has a claim-type problem rather than a formatting problem.

### Phase B — Establish the idea

**Step 4. Identify the method paper or text.**
Input: the method name and the field.
Output: the primary methodological reference, with section and page.
Rationale: a method name is not a citation. Many methods have several papers, and the one that defines the estimator is not always the one that popularised it.
Branch: if the method is named after a person, check whether the named paper defines it or merely applies it.
Edge case: a method can have a foundational paper and a later correction that changes the recommendation.
Check: does the paper define the estimator, or only use it.
Record: the reference and the specific section.

**Step 5. State the assumptions the method requires.**
Input: the method paper's assumptions section.
Output: a list of assumptions with a note on whether each is plausible in the current data.
Rationale: the assumptions are what make the estimate interpretable, and they are usually not in the abstract.
Branch: if an assumption is implausible, the method may still be usable with a sensitivity analysis or a different estimator.
Edge case: some assumptions are untestable, such as no unmeasured confounding, and must be argued rather than checked.
Check: is each assumption named in the methods section of the draft.
Record: the assumption list and its plausibility.

**Step 6. Identify the estimand the method targets.**
Input: the method paper and the analysis plan.
Output: a statement of whether the method estimates a conditional or marginal effect, a risk or an odds, a hazard or a cumulative incidence.
Rationale: different estimands answer different questions, and a method can be correct and answer the wrong question.
Branch: if the estimand in the paper differs from the one in the plan, the citation supports the method and not the interpretation.
Edge case: a marginal estimand from a conditional model requires standardisation or a different estimator.
Check: does the reported quantity match the estimand.
Record: the estimand and its scale.

**Step 7. Check for later corrections or critiques.**
Input: the citation graph around the method paper.
Output: a list of corrections, critiques, or updated recommendations, with dates.
Rationale: methods recommendations change when simulation evidence accumulates, and a foundational paper can be partially superseded without being retracted.
Branch: if a later paper shows a failure mode, cite both and state how the analysis addresses it.
Edge case: a critique can be contested, in which case the methods section should present the disagreement rather than pick a side silently.
Check: has the method been recommended against in the last five years.
Record: the critique and the response.

**Step 8. Check the reporting standard for the claim type.**
Input: the claim type and the EQUATOR library.
Output: the applicable checklist and its version.
Rationale: the reporting standard defines what a complete methods section contains, and it changes over time.
Branch: if the study predates the current standard, state which version was followed.
Edge case: a study can fall under more than one guideline, such as TRIPOD+AI and DECIDE-AI.
Check: is the checklist version named in the draft.
Record: the checklist and version.

### Phase C — Establish the implementation

**Step 9. Identify the software and version.**
Input: the code, the environment file, or the methods section.
Output: package name and version, plus the language and platform.
Rationale: the implementation is a separate object, and a version change can change the result.
Branch: if the version is not recorded, the analysis is not reproducible, regardless of the method citation.
Edge case: a container or environment lock file is stronger evidence than a prose statement.
Check: can a reader install the exact version.
Record: package, version, and environment.

**Step 10. Trace the default settings.**
Input: the software documentation and the code.
Output: a list of defaults that affect the result, with the values used.
Rationale: defaults encode analyst choices, and the paper usually assumes different ones.
Branch: if a default is inappropriate, set it explicitly and say so.
Edge case: a default can change between versions without a change in the function signature.
Check: are the consequential defaults stated in the methods.
Record: the settings and their justification.

**Step 11. Check the unit of resampling and splitting.**
Input: the code for cross-validation or bootstrap.
Output: the entity resampled and whether it respects clustering.
Rationale: resampling rows when patients are the unit produces confidence intervals that are too narrow and performance estimates that are too optimistic.
Branch: if the unit is wrong, recompute rather than caveat.
Edge case: repeated measures and multi-site data need group-aware resampling.
Check: does the split keep all records from one patient on one side.
Record: the resampling unit.

**Step 12. Check convergence and numerical warnings.**
Input: the model output and logs.
Output: a statement of whether the algorithm converged and whether warnings were addressed.
Rationale: a non-converged model can produce plausible-looking coefficients that are not estimates of anything.
Branch: if convergence failed, change the optimiser or the specification rather than reporting the result.
Edge case: separation in logistic regression produces large coefficients and should be handled explicitly.
Check: are warnings visible in the saved output.
Record: the convergence diagnostics.

**Step 13. Check the software's own validation.**
Input: the package documentation and any associated paper.
Output: a statement of whether the implementation has been tested against a reference.
Rationale: a package can implement a method incorrectly, and the package citation is not evidence of correctness.
Branch: if no validation exists, compare against an independent implementation on a small example.
Edge case: a package can be correct for one estimator and wrong for a related one.
Check: is there a unit test or a reference comparison.
Record: the validation evidence.

**Step 14. Record the random seed and the computational environment.**
Input: the code and the run logs.
Output: a seed, a session information dump, and a hash of the input data.
Rationale: stochastic methods are only reproducible with the seed and the environment.
Branch: if the result cannot be reproduced from the record, the record is incomplete.
Edge case: parallel execution can change the random stream even with a fixed seed.
Check: does a fresh run reproduce the headline number.
Record: the seed and environment.

### Phase D — Establish the performance claim

**Step 15. Distinguish discrimination from calibration.**
Input: the reported metrics.
Output: a statement of whether both are reported, with values.
Rationale: a model can rank well and be badly calibrated, and a clinical decision needs probabilities that mean what they say.
Branch: if only AUC is reported, the calibration claim is unsupported.
Edge case: calibration depends on the population, so a model calibrated in one setting can be miscalibrated in another.
Check: is a calibration slope or curve reported.
Record: discrimination and calibration values.

**Step 16. Check the performance claim's population and setting.**
Input: the validation study.
Output: a statement of the population, setting, and era, and whether they match the deployment.
Rationale: performance is a property of the model in a population, not of the model alone.
Branch: if the validation is internal, the claim is about apparent performance, not transportability.
Edge case: a temporal split tests transportability over time but not across sites.
Check: is the validation external to the development data.
Record: the validation design.

**Step 17. Compute or check the decision-analytic metric.**
Input: the contingency table at the deployed threshold.
Output: net benefit at the relevant thresholds, compared with treat-all and treat-none.
Rationale: a model can improve ranking and still be worse than treat-all at the threshold a clinic uses.
Branch: if net benefit is negative at the deployed threshold, the model should not be deployed at that threshold.
Edge case: the relevant threshold range depends on the harm of a missed case relative to a false alert.
Check: is a decision curve or net benefit reported.
Record: the net benefit and the threshold.

**Step 18. Check the uncertainty around the performance estimate.**
Input: the reported confidence intervals.
Output: intervals for discrimination, calibration, and net benefit where available.
Rationale: a point estimate without an interval cannot support a comparison between models.
Branch: if the intervals overlap, the ranking is not established.
Edge case: bootstrap intervals must respect the resampling unit from step 11.
Check: are intervals reported for the primary metric.
Record: the intervals.

**Step 19. Check for optimism and overfitting.**
Input: the development and validation results.
Output: a statement of the optimism correction and the number of events per variable.
Rationale: apparent performance is optimistic, and the gap grows with model complexity and small samples.
Branch: if no optimism correction is reported, treat the development performance as an upper bound.
Edge case: machine learning models with many features need stronger correction than regression models.
Check: is there an internal validation with optimism correction.
Record: the optimism estimate.

### Phase E — Reporting and record

**Step 20. Map the draft to the checklist.**
Input: the draft methods section and the checklist.
Output: a completed checklist with a location for each item.
Rationale: the checklist is the definition of completeness for the claim type, and mapping it exposes omissions.
Branch: if an item does not apply, state why rather than leaving it blank.
Edge case: a checklist can have items that the journal does not require and the reviewer will still expect.
Check: is there a location for every applicable item.
Record: the completed checklist.

**Step 21. Check the citations for the three links.**
Input: the reference list.
Output: a citation for the idea, the implementation, and the reporting standard.
Rationale: a methods section with one citation for three objects is under-specified.
Branch: if the software has no citable paper, cite the package and version with a URL and access date.
Edge case: a reporting guideline can be cited by its statement paper and by its checklist URL.
Check: are all three present.
Record: the three citations.

**Step 22. Write the reproducibility statement.**
Input: the code, data, and environment.
Output: a statement of what is available and what is restricted, with a route to access.
Rationale: reproducibility is a claim about artefacts, and it should be specific.
Branch: if data cannot be shared, describe the access conditions and provide the code.
Edge case: a synthetic data generator can support reproducibility without sharing the original data.
Check: does the statement name a repository and a version.
Record: the statement and the identifiers.

**Step 23. State the residual uncertainty in the method choice.**
Input: the assumption checks and the sensitivity analyses.
Output: a sentence naming what would change the conclusion.
Rationale: method choice is a judgement, and a stated sensitivity is more useful than a defended default.
Branch: if the conclusion is sensitive to a plausible alternative, report both.
Edge case: a sensitivity analysis that is not pre-specified is exploratory and should be labelled.
Check: is there at least one sensitivity analysis for the main assumption.
Record: the sensitivity result.

**Step 24. Set the review trigger for the methods source set.**
Input: the method, software, and reporting standard.
Output: review dates or event triggers for each.
Rationale: software versions change frequently, reporting standards change rarely, and method recommendations change on the publication cycle.
Branch: if the software is pinned, the trigger is a security or bug fix rather than a new release.
Edge case: a major version change can alter defaults without changing the function name.
Check: is the software version pinned in the environment file.
Record: the triggers.

### The five tests, restated for methods sources

The procedure above is domain-specific, but every methods source admitted to the list must still pass the five tests that govern the whole module.

| Test | Question for a methods source | Fail signal | Response |
|---|---|---|---|
| Authority | Is this the method's originators or a later summary | A blog restates a paper | Chase the primary paper |
| Currency | Is this the current version of the method or standard | A 2015 checklist cited as current | Replace with the current version |
| Applicability | Does the method match the question and the data | A method for a different estimand | Reconsider the design |
| Accessibility | Can the team read the method and its assumptions | Paywalled without an accessible preprint | Use the preprint and label it |
| Provenance | Is the implementation traceable to a version | Package named without a version | Record the version and environment |

**Branch: the source is a retracted methods paper.** Remove it, check whether the implementation depended on it, and re-run the affected analysis if it did. The citation may remain in a change note, never in the methods.

**Branch: the source is a preprint.** It is admissible as a signal and as a pointer to an implementation, not as settled authority. Check for a journal version before relying on a numerical claim.

**Branch: the source is a superseded reporting standard.** TRIPOD 2015 is the clearest case; TRIPOD+AI and its 2024 update supersede it for prediction-model work. Cite the standard that applies at the date of the analysis and say which one was used.

### Defaults that change results

| Setting | Typical default | Why it matters | What to record |
|---|---|---|---|
| Bootstrap replicates | 1,000 | Interval stability | Number and seed |
| Cross-validation folds | 5 or 10 | Performance variance | Folds and grouping |
| Missing-data handling | Listwise deletion | Selection bias and lost power | Method and assumptions |
| Penalty strength | Data-driven | Effective model complexity | Value and tuning method |
| Reference category | First level alphabetically | Coefficient interpretation | Reference level |
| Tie handling | Package-specific | AUC in small samples | Method |
| Convergence tolerance | Package-specific | Whether the estimate exists | Tolerance and status |
| Class weighting | None | Threshold and calibration | Weights used |

A methods section that names the package but not these settings is a methods section that cannot be reproduced. The settings are not implementation trivia; they are part of the estimator.

### A branch worked in miniature: one citation, two implementations

A paper describes a regularised regression with a particular penalty and a particular cross-validation scheme. Two packages implement it. The first uses the paper's scheme; the second uses a different default for penalty selection and a different fold assignment.

Both analyses cite the same paper. The results differ, and neither is wrong relative to its own implementation. The citation identifies the idea, and the implementation determines the number.

The correct response is not to choose the citation that supports the preferred result. It is to record both implementations, state which was used, and check whether the conclusion is sensitive to the difference. If it is, the methods section should report both and explain the choice.

The procedure produces a methods citation set rather than a single citation, and that is the point. A reader who can see the idea, the implementation, and the standard can reproduce the reasoning; a reader who sees one citation cannot.

### What a methods reviewer checks first

- Whether the estimand is stated before the model.
- Whether the metric matches the decision.
- Whether calibration is reported alongside discrimination.
- Whether the validation is internal, temporal, or external.
- Whether the software version and settings are recorded.
- Whether the resampling unit respects the data structure.
- Whether the conclusion is sensitive to a plausible alternative.
- Whether the reporting checklist version is named.
- Whether the code and data statement is specific enough to act on.
- Whether any claim in the abstract is stronger than the results.

Most of these checks take minutes and catch the failures that survive to review. The last one is the most common: an abstract that reports an optimistic internal result as if it were a validated one.

| Where the citation fails | What the sentence claims | What the citation supports | Fix |
|---|---|---|---|
| Idea cited for software | "We used method X" | The method exists | Add the package and version |
| Software cited for method | "We used package Y" | The computation was run | Add the method paper and its assumptions |
| Assumption omitted | "We adjusted for confounding" | A model was fitted | Name the identification assumption |
| Metric mismatch | "The model performs well" | Discrimination only | Add calibration and net benefit |
| Standard superseded | "Reported per TRIPOD" | The 2015 checklist | Cite TRIPOD+AI and map the items |

- A citation that names a method but not an estimand.
- A citation that names a package but not a version.
- A citation that names a checklist but not a version.
- A citation that supports the analysis but not the interpretation.
- A citation that is correct and irrelevant to the sentence.

The last of these is the hardest to catch, because everything about the citation is accurate. The sentence has drifted from the analysis, and the citation still points at the method.

A useful test is to read the sentence with the citation removed. If the sentence still asserts a result, the citation is doing work; if the sentence becomes a description of a method, the citation was decoration.

- Remove the citation and reread the sentence.
- If the claim survives without support, find the support or cut the claim.

## 6. Worked example

A product team writes: "The model has an AUC of 0.82 in external validation, so it is ready for clinical deployment." The sentence will appear in a health-system pitch. The team has the validation paper, a vendor summary, and a methods textbook on the shelf.

### Framing the claim

The sentence contains two claims and one hidden assumption:

1. **Discrimination claim.** The model ranks a randomly chosen case above a randomly chosen non-case with probability 0.82 in the validation population.
2. **Readiness claim.** The model should be deployed in the intended clinical workflow.
3. **Hidden assumption.** The threshold and the decision context are such that the ranking produces clinical value.

The first claim is a statistical property of the validation sample. The second is a decision about a workflow, a population, and a threshold. The citation supports the first and cannot support the second.

| Sub-claim | Type | What could support it |
|---|---|---|
| AUC 0.82 in external validation | Prediction performance | The validation study, with its population and setting |
| Probabilities are accurate | Calibration | Calibration slope, intercept, or curve |
| Deployment improves decisions | Decision-analytic | Net benefit at the deployed threshold |
| The workflow can absorb the alerts | Operational | Alert volume and review capacity |

### Candidate sources

| # | Candidate | Type | Role |
|---|---|---|---|
| C1 | External validation paper reporting AUC 0.82 | Peer-reviewed study | The discrimination claim |
| C2 | Vendor summary repeating the AUC | Commercial summary | The likely origin of the sentence |
| C3 | Steyerberg, *Clinical Prediction Models* | Textbook | The distinction between discrimination and calibration |
| C4 | Van Calster et al., calibration paper, 2019 | Peer-reviewed methods paper | Why calibration matters for decisions |
| C5 | Vickers and Elkin, decision curve analysis, 2006 | Peer-reviewed methods paper | Net benefit as a decision metric |
| C6 | TRIPOD+AI statement, 2024 | Reporting guideline | What a complete performance report contains |

### Scoring the candidates

**C1, the validation paper.**
- Authority: high for the performance estimate in its own population.
- Currency: current, if the model version is the deployed one.
- Applicability: conditional, because external validation is still one setting.
- Accessibility: open access.
- Provenance: methods, cohort, and analysis are reported.
- Verdict: primary support for the discrimination sub-claim, with the setting stated.

**C2, the vendor summary.**
- Authority: commercial interest.
- Currency: unclear, because the model version is not stated.
- Applicability: unstated population.
- Accessibility: public.
- Provenance: no methods.
- Verdict: background only; it is the source of the sentence's confidence, not of its evidence.

**C3, the textbook.**
- Authority: high for the concepts.
- Currency: depends on the edition; the discrimination-calibration distinction is stable.
- Applicability: general.
- Accessibility: library copy.
- Provenance: authored synthesis.
- Verdict: definitional support for the distinction, not evidence about this model.

**C4 and C5, the methods papers.**
- Authority: high, peer-reviewed and widely cited.
- Currency: current; both remain standard references.
- Applicability: general methods, directly relevant to the claim.
- Accessibility: open access for C4; C5 is paywalled but the method is widely described.
- Provenance: methods and worked examples are reported.
- Verdict: primary support for the requirement to report calibration and net benefit.

**C6, TRIPOD+AI.**
- Authority: high as a reporting standard.
- Currency: current, and it supersedes TRIPOD 2015.
- Applicability: prediction models, which is the claim type.
- Accessibility: open access.
- Provenance: consensus process described.
- Verdict: primary support for the completeness claim, not for the performance claim.

### The arithmetic

The validation paper reports discrimination at AUC 0.82. At the threshold the vendor recommends, the confusion matrix implies sensitivity 33% and specificity 83% in a population with 7% prevalence. From those numbers:

- Cases: 0.07 × 38,455 = 2,692.
- True positives: 0.33 × 2,692 = 888.
- False negatives: 2,692 − 888 = 1,804.
- Controls: 38,455 − 2,692 = 35,763.
- False positives: 0.17 × 35,763 = 6,080.
- True negatives: 35,763 − 6,080 = 29,683.
- Positive predictive value: 888 / (888 + 6,080) = 12.7%.
- Negative predictive value: 29,683 / (29,683 + 1,804) = 94.3%.
- Number needed to evaluate: 1 / 0.127 = 7.9.

The PPV is the number a clinical team will feel. Fewer than one in eight alerts is a true case, and the clinic must review about eight alerts for each case it finds.

Net benefit at a threshold probability pt is:

```text
NB = TP/N - (FP/N) x (pt / (1 - pt))
```

At pt = 0.12, NB = 0.0231 − 0.1581 × 0.1364 = 0.0015, which is barely positive. At pt = 0.15, NB = 0.0231 − 0.1581 × 0.1765 = −0.0048, which is negative, and worse than treating nobody. At pt = 0.10, NB = 0.0231 − 0.1581 × 0.1111 = 0.0055, which is positive but smaller than treat-all at that threshold in this example only if the alert volume is manageable.

The arithmetic is the argument. An AUC of 0.82 describes ranking across all thresholds. The deployment decision happens at one threshold, with one prevalence, and with a review capacity. The ranking can be good and the decision can still be bad.

### What each metric answers

| Metric | Question it answers | Question it does not answer |
|---|---|---|
| AUC | How well does the model rank? | Are the probabilities right? |
| Calibration slope | Do predictions track outcomes across the range? | Is the threshold useful? |
| Calibration intercept | Is there systematic over- or under-prediction? | Does the ranking hold in a new site? |
| Net benefit | Is using the model better than the alternatives at this threshold? | Is the model fair across subgroups? |
| Brier score | How close are probabilities to outcomes overall? | Where the errors are concentrated |

A calibration slope of 0.5 means the model's predictions are too extreme: high risks are too high and low risks are too low, relative to observed outcomes. An intercept below zero means systematic over-prediction. Neither is visible in an AUC, and both change the decision.

### Questions to ask before accepting a readiness claim

- What population and setting was the validation in?
- Was the validation external to the development data?
- Is calibration reported, and with what slope and intercept?
- What threshold will the clinic use, and what is the net benefit there?
- What is the alert volume at that threshold, and who reviews it?
- Is the model version in the paper the version being deployed?

A team that can answer all six has a deployment case. A team that can answer only the first has a citation.

### Evidence needed by claim

| Claim | Minimum evidence | Still insufficient |
|---|---|---|
| The model ranks well | External discrimination with interval | Internal cross-validation alone |
| The probabilities are usable | Calibration in the target population | Discrimination alone |
| The model improves decisions | Net benefit at the deployed threshold | Calibration alone |
| The model improves outcomes | Prospective comparative evaluation | Any retrospective metric |
| The model is fair | Subgroup performance with intervals | Overall performance |

The table is a ladder, and each rung requires the one below it. A team cannot claim decision benefit from a calibration curve, and cannot claim outcome benefit from a decision curve.

- A model with good discrimination and poor calibration can be worse than a simple rule.
- A model with good calibration and bad net benefit can be worse than treat-all.
- A model with good net benefit in one setting can be worse in another.

Each sentence in that list is a claim type, and each needs its own evidence.

### Verdict

| Sub-claim | Verdict | Supportable sentence |
|---|---|---|
| AUC 0.82 | Primary support | "External validation in one health system reported an AUC of 0.82." |
| Ready for deployment | Reject | "Deployment readiness requires calibration and net benefit at the intended threshold, which were not reported." |
| Probabilities are accurate | Unsupported | No calibration was reported |
| Workflow can absorb alerts | Unsupported | No alert-volume analysis was reported |

The sentence should be replaced, not footnoted. A footnote that says "AUC only" preserves the impression that the model is ready while technically disclosing that the evidence is incomplete.

### Two diagnostics that would change the verdict

**Diagnostic 1: recalibration with a calibration slope near 1 and net benefit positive at the deployed threshold.** If the team recalibrates the model in the target population and reports a calibration slope of about 1 with positive net benefit at the threshold the clinic will use, the readiness claim becomes supportable for that population and threshold. The diagnostic is specific: it requires the calibration and the net benefit, not a second AUC.

**Diagnostic 2: a threshold change that makes net benefit positive with an acceptable alert volume.** The net benefit is negative at 0.15 and positive at 0.12 in the arithmetic above. If the clinic can operate at a lower threshold with a review capacity that matches the resulting alert volume, the deployment claim could be supportable at that threshold. The diagnostic requires both the net benefit and the operational capacity, because a positive net benefit with an unmanageable alert volume is not deployable.

A third diagnostic, weaker, is a prospective evaluation showing that the alert changes a clinical action. It would support an implementation claim without resolving the calibration question.

### What the team should record

The ledger records the claim, the three sub-claims, the citations for the idea, the implementation, and the standard, and the arithmetic at the deployed threshold. The review trigger is a new model version, a recalibration, or a change in the clinical threshold.

The case also produces a reusable sentence for the pitch: "The model ranks well in external validation; calibration and net benefit at our threshold are the next evidence we need." That sentence is weaker than the original and is the one a clinical audience will trust.
## 7. Code, query, or calculation

The script below turns a contingency table into operating characteristics and net benefit. It uses the published sensitivity, specificity, prevalence, and cohort size from the validation in the worked example, so the output can be checked by hand.

```python
"""Discrimination is not utility: operating characteristics and net benefit."""

N = 38_455
PREVALENCE = 0.07
SENS = 0.33
SPEC = 0.83
ALERTS = 6_971


def table(n, prevalence, sens, spec):
    cases = round(n * prevalence)
    controls = n - cases
    tp = round(cases * sens)
    fn = cases - tp
    fp = round(controls * (1 - spec))
    tn = controls - fp
    return tp, fp, fn, tn


def metrics(tp, fp, fn, tn):
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    lr_pos = sens / (1 - spec) if spec < 1 else float("inf")
    lr_neg = (1 - sens) / spec
    return sens, spec, ppv, npv, lr_pos, lr_neg


def net_benefit(tp, fp, n, pt):
    return tp / n - (fp / n) * (pt / (1 - pt))


tp, fp, fn, tn = table(N, PREVALENCE, SENS, SPEC)
sens, spec, ppv, npv, lr_pos, lr_neg = metrics(tp, fp, fn, tn)

print(f"cohort {N:,}   prevalence {PREVALENCE:.0%}   alerts {ALERTS:,}")
print(f"2x2 from sens/spec: TP {tp:,}  FP {fp:,}  FN {fn:,}  TN {tn:,}")
print(f"sensitivity {sens:.2%}  specificity {spec:.2%}  PPV {ppv:.2%}  NPV {npv:.2%}")
print(f"LR+ {lr_pos:.2f}  LR- {lr_neg:.2f}")
print()
print(f"{'threshold':>9s} {'net benefit':>12s} {'treat all':>10s} {'verdict':>10s}")
for pt in (0.05, 0.10, 0.12, 0.15, 0.20, 0.30):
    nb = net_benefit(tp, fp, N, pt)
    ta = PREVALENCE - (1 - PREVALENCE) * (pt / (1 - pt))
    verdict = "use" if nb > max(0.0, ta) else "worse"
    print(f"{pt:9.2f} {nb:12.5f} {ta:10.5f} {verdict:>10s}")

print()
print("alert rate implied by the table:", f"{ALERTS / N:.1%}")
print("number needed to evaluate:", round(1 / ppv, 1))
```

Expected output:

```text
cohort 38,455   prevalence 7%   alerts 6,971
2x2 from sens/spec: TP 888  FP 6,080  FN 1,804  TN 29,683
sensitivity 32.99%  specificity 83.00%  PPV 12.74%  NPV 94.27%
LR+ 1.94  LR- 0.81

threshold  net benefit  treat all    verdict
     0.05      0.01477    0.02105      worse
     0.10      0.00552   -0.03333        use
     0.12      0.00153   -0.05682        use
     0.15     -0.00481   -0.09412      worse
     0.20     -0.01643   -0.16250      worse
     0.30     -0.04467   -0.32857      worse

alert rate implied by the table: 18.1%
number needed to evaluate: 7.8
```

Three readings of the output matter. First, the net benefit is positive only in a narrow band of thresholds, and at 0.15 it is negative even though the model discriminates. Second, the model beats treat-all at 0.10 and 0.12, which is the argument for deploying at a lower threshold if the alert volume is manageable. Third, the number needed to evaluate is 7.8, so the operational burden is the binding constraint as much as the statistics.

The script recomputes the table from sensitivity and specificity rather than using the published counts, which is why the false positives are 6,080 rather than the 6,083 implied by subtracting alerts from true positives. The discrepancy is rounding, and a methods section should state which numbers were used.

To adapt the script, substitute your own prevalence and operating characteristics, and add the treat-none line at zero. The comparison that matters is between the model, treat-all, and treat-none at the threshold the clinic will actually use.

Two cautions about the arithmetic. Net benefit is expressed in units of true positives per patient, so it is comparable across models only when the outcome and the population are the same. And the treat-all comparator is not always the relevant alternative; a triage rule, a clinician's judgment, or an existing alert may be the thing to beat.

- Net benefit is not a hypothesis test.
- A negative net benefit at a threshold does not mean the model is useless at every threshold.
- A positive net benefit does not mean the model is fair or that the workflow can absorb it.

## 8. Data, standards, and regulatory context

Methods sources are governed by reporting standards rather than by law. The exception is the regulated product context, where a methods choice can be part of a submission and a guidance can describe the evidence the regulator expects.

### 8.1 Reporting standards

| Standard | Applies to | Version or date | Binding status |
|---|---|---|---|
| TRIPOD+AI | Prediction models using regression or machine learning | 2024, supersedes TRIPOD 2015 | Journal and editorial policy |
| PROBAST | Risk of bias in prediction model studies | 2019 | Methodological convention |
| CONSORT-AI | Randomised trials of AI interventions | 2020 | Journal policy |
| SPIRIT-AI | Protocols for trials of AI interventions | 2020 | Journal policy |
| STARD-AI | Diagnostic accuracy studies of AI | Current version per EQUATOR | Journal policy |
| DECIDE-AI | Early clinical evaluation of AI | 2022 | Journal policy |
| PRISMA 2020 | Systematic reviews | 2020 | Journal policy |
| ICH E9(R1) | Estimands in clinical trials | Addendum | Regulatory expectation in trials |

The distinction between a reporting standard and a risk-of-bias tool matters. A checklist asks whether the report contains an item; a risk-of-bias tool asks whether the design supports the claim. A study can be fully reported and still be at high risk of bias, and a study can be at low risk and poorly reported.

### 8.2 Regulatory expectations

In a device submission, the methods used to develop and validate a model are part of the evidence. The regulator's guidance on software and AI describes expectations for data provenance, model development, validation, and performance reporting. Those expectations are not the same as a journal checklist, and a team that satisfies TRIPOD+AI has not necessarily satisfied a regulator.

The reverse is also true. A regulatory submission can contain confidential evidence that is not published, so the public methods literature may be silent on exactly the questions a reviewer asks. In that case, the methods citation supports the approach, and the submission supplies the evidence.

### 8.3 The reproducibility infrastructure

| Element | What it establishes | Common gap |
|---|---|---|
| Code repository with a version tag | The code that produced the result | No tag, or a branch that moved |
| Environment lock file | The software versions | Only a requirements list without pins |
| Data dictionary | Variable definitions and units | Definitions in prose only |
| Synthetic or sample data | A runnable example | No data at all |
| Analysis log | The sequence of steps | Only the final script |
| Pre-registration | The analysis was specified in advance | Registration after seeing data |

None of these is legally required in most research settings. All of them are required for a reader to check the work, which is the standard this module applies.

### 8.4 What a methods citation does not establish

- That the method is appropriate for the question.
- That the implementation matches the paper.
- That the assumptions hold in the data.
- That the result is reproducible.
- That the reporting is complete.

Those are separate checks, and the procedure in section 5 runs them separately. A citation is a pointer, not a proof.

### 8.5 Reporting and appraisal are different instruments

| Instrument | Question | Output | Use at |
|---|---|---|---|
| TRIPOD+AI | Is the report complete? | Checklist with item locations | Writing and review |
| PROBAST | Does the design support the claim? | Risk-of-bias judgement by domain | Appraisal |
| CONSORT-AI | Is the trial reported completely? | Checklist | Trial reporting |
| DECIDE-AI | Is the early evaluation reported completely? | Checklist | Early clinical evaluation |
| PRISMA 2020 | Is the review reported completely? | Checklist | Systematic review |

A study can be reported completely and still be at high risk of bias. A study can be at low risk of bias and poorly reported. The two instruments answer different questions, and a methods section should not claim one by satisfying the other.

### 8.6 The regulatory layer for model evidence

In a device submission, the evidence about a model is assessed against the regulator's expectations for software and AI, not against a journal checklist. Those expectations typically cover data provenance, development and validation, performance reporting, human factors, and post-market monitoring. The methods literature supplies the vocabulary; the guidance supplies the requirement.

| Question a regulator may ask | Methods source that helps | What it does not supply |
|---|---|---|
| How was the training data selected? | Reporting standards and study design texts | The regulator's specific expectation |
| How was the model validated? | Prediction model literature | The required evidence level |
| How is performance reported? | TRIPOD+AI | The acceptance threshold |
| How are changes controlled? | Software lifecycle standards | The regulatory pathway |
| How is performance monitored? | Post-market surveillance guidance | The specific metric |

The table is a map, not a substitute. Each row requires the guidance as well as the methods source.

## 9. Industry practice and case evidence

**Propensity scores (documented).** Rosenbaum and Rubin set out the central role of the propensity score in 1983, including the balancing property that makes it useful: conditioning on the score balances the distribution of observed covariates between groups. The paper is a definitional source. It does not establish that a particular matching implementation is correct, and it does not address unmeasured confounding.

**Target trial emulation (documented).** Hernán and Robins described the target trial framework in 2016 as a way to structure observational analyses so that the design decisions are explicit. The framework has become a standard way to state an estimand before analysis. The paper is a methods source; it does not supply the data or the implementation.

**TRIPOD+AI (documented).** The 2024 statement updates the reporting guidance for prediction models and explicitly supersedes the 2015 checklist. It adds open-science items, patient and public involvement, and subgroup performance reporting. The practical effect is that a methods section written in 2020 may need substantial revision to satisfy the current standard, even if the analysis has not changed.

**Calibration and decision curves (documented).** The 2019 calibration paper and the 2006 decision curve paper are both widely cited and both argue that the standard reporting of prediction models was incomplete. Their adoption has been uneven across fields, which is why a reviewer in one specialty may expect calibration and a reviewer in another may not.

**The implementation gap (documented pattern).** Package documentation records defaults, and version histories record changes. The specific effect on any one analysis depends on the package and the settings, and the general pattern is that the paper does not determine the number.

### Documented facts and inferences

| Statement | Status |
|---|---|
| Propensity score balancing property | Documented in Rosenbaum and Rubin, 1983 |
| Target trial framework | Documented in Hernán and Robins, 2016 |
| TRIPOD+AI supersedes TRIPOD 2015 | Documented in the 2024 statement |
| Calibration is under-reported | Documented as the argument of the 2019 paper; extent varies |
| Implementations differ from papers | Documented in package documentation and version histories |
| Teams under-report software versions | Inference from the reproducibility literature |

### What a methods review looks like in practice

- The reviewer reads the estimand before the model.
- The reviewer checks the metric against the decision.
- The reviewer looks for calibration and net benefit, not only discrimination.
- The reviewer checks the validation design and the population.
- The reviewer checks the software version and the seed.
- The reviewer checks the reporting checklist version.
- The reviewer runs one number by hand.

The last step is the most effective. A reviewer who recomputes a single proportion or a single net benefit from the reported table finds most arithmetic and reporting errors in minutes.


**The TRIPOD transition (documented).** TRIPOD+AI was published in *BMJ* in April 2024 and explicitly supersedes the 2015 TRIPOD checklist for prediction models. It contains 27 main items, adds open-science and patient-involvement items, and recommends performance reporting in key subgroups. The lesson is that a methods section written to the 2015 checklist can be complete by its own standard and incomplete by the current one.

**Calibration reporting (documented).** Van Calster and colleagues argued in *BMC Medicine* in 2019 that calibration receives too little attention relative to discrimination, and that poorly calibrated models can mislead clinical decisions. The paper describes levels of calibration assessment and recommends against relying on the Hosmer-Lemeshow test alone. The lesson is that a metric can be standard, widely reported, and still insufficient for the decision.

**Decision curve analysis (documented).** Vickers and Elkin described decision curve analysis in *Medical Decision Making* in 2006 as a way to evaluate prediction models against decision thresholds. The method has become a standard way to connect a model's output to a clinical decision. The lesson is that a metric invented for a decision problem can become more informative than the metric the field inherited.

**The reproducibility problem (documented pattern).** Surveys of published analyses have repeatedly found that a substantial fraction cannot be reproduced from the available code and data, often because of missing versions, missing seeds, or unavailable data. The specific percentages vary by field and by definition of reproducibility. The lesson is that a methods citation is not a reproducibility guarantee, and the gap is usually in the implementation rather than the idea.

**Software defaults (documented pattern, variable specifics).** Package documentation records defaults that differ from the defaults assumed in the originating paper, and version changes can alter results without changing the function name. The lesson is that pinning versions is part of the method, not an engineering detail.

### Documented facts and inferences

| Statement | Status |
|---|---|
| TRIPOD+AI supersedes TRIPOD 2015 | Documented in the TRIPOD+AI statement |
| Calibration is under-reported relative to discrimination | Documented as the argument of the 2019 paper; the extent varies by field |
| Decision curve analysis is a standard method | Documented by its citation record and adoption |
| Many analyses are not reproducible | Documented as a pattern across surveys; the rate is field-specific |
| Defaults differ between implementations | Documented in package documentation |
| Teams cite the paper and ignore the software | Inference from practice |

The last row is an inference. It is consistent with the reproducibility literature and with the observation that methods sections rarely report seeds or package versions.

**What mature teams do (inference).** They treat the methods citation as a set: paper, package, version, and checklist. They pin versions in an environment file, record seeds, and run a fresh-environment reproduction before submission. The practice is more common in regulated environments than in academic ones, because the cost of a failed reproduction is higher.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| AUC treated as utility | Performance reported only as AUC | Discrimination conflated with decision value | Deployment at an unhelpful threshold | Report calibration and net benefit |
| Calibration omitted | No slope, intercept, or curve | Probabilities assumed accurate | Systematic over- or under-treatment | Calibrate in the target population |
| Assumption omitted | Method named, assumptions not | Citation supports the estimator, not the inference | Unsupported causal or predictive claim | Name and test the assumption |
| Software cited for method | Package name in the methods, no paper | Implementation conflated with idea | Reader cannot assess the estimator | Cite both |
| Method cited for software | Paper cited, no version | Idea conflated with implementation | Result not reproducible | Record package and version |
| Defaults unstated | "We used the default" | Defaults encode choices | Different results on re-run | State consequential defaults |
| Wrong resampling unit | Rows resampled in patient data | Dependence ignored | Intervals too narrow | Group-aware resampling |
| Seed omitted | Stochastic method, no seed | Randomness unreported | Result cannot be reproduced | Record the seed |
| Optimism uncorrected | Development performance reported | Overfitting ignored | Inflated expectations | Internal validation with correction |
| Superseded checklist | TRIPOD 2015 cited in 2026 | Reporting standard changed | Incomplete methods section | Use TRIPOD+AI |
| Estimand drift | Model and conclusion on different scales | Conditional and marginal effects conflated | Wrong interpretation | State the estimand |
| Threshold chosen post hoc | Threshold picked after seeing results | Selection on the test set | Optimistic net benefit | Pre-specify the threshold |
| Single-site validation | One hospital, national claim | Transportability assumed | Performance drop at deployment | External validation |
| Metric mismatch | Hazard ratio for a prediction claim | Different question answered | Claim not supported | Match metric to claim |
| Unstated missing-data method | Complete-case analysis unlabelled | Selection bias | Biased estimate | Report method and sensitivity |

Three red flags are specific to methods citations. A methods section with no software version is not reproducible. A performance section with no calibration is incomplete. And a sensitivity analysis that appears only in the discussion, after the conclusion, was probably run after the result was known.

### A methods audit in ten checks

- Is the estimand stated before the model?
- Does the primary metric match the decision?
- Is calibration reported with discrimination?
- Is the validation external, temporal, or internal, and is it labelled?
- Is the sample size adequate for the number of predictors?
- Is the resampling unit the patient, not the row?
- Is the software version recorded and pinned?
- Is the seed reported for stochastic steps?
- Is the reporting checklist version named?
- Does the abstract match the results?

A methods section that fails three or more of these checks is not ready for review, and a reader should treat its performance claims as provisional.

- A methods section with no estimand cannot be assessed for appropriateness.
- A methods section with no calibration cannot support a probability claim.
- A methods section with no version cannot be reproduced.
- A methods section with no seed cannot be reproduced for stochastic steps.
- A methods section with no checklist version cannot be judged for completeness.

Each of these is a missing link, and each is fixed by adding a sentence rather than by redoing the analysis.

The audit is also a useful template for a methods review meeting. Walking through the ten checks in order takes about fifteen minutes and produces a list of concrete fixes rather than a general impression.

A reviewer who runs the audit on their own work before submission will usually find two missing links. Both are cheap to fix at that stage and expensive to fix after review.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The method paper describes the estimator used | The citation supports the idea | Implementation diverges | Compare code against the paper |
| The software implements the method correctly | The result depends on it | Known bug or different default | Independent implementation or tests |
| The assumptions are plausible in the data | The estimate is interpretable | Positivity or confounding violations | Diagnostics and sensitivity analysis |
| The validation population resembles deployment | Transportability is claimed | Different setting or era | External validation |
| The metric matches the decision | The conclusion is actionable | Fixed capacity or cost asymmetry | Map metric to decision |
| Calibration is adequate | Probabilities are used | Miscalibration | Calibration curve |
| The reporting standard is current | Completeness is judged | Standard superseded | Check version |
| The seed and environment are recorded | Reproducibility is claimed | Missing version or seed | Fresh-environment run |
| The threshold was pre-specified | Net benefit is unbiased | Post hoc selection | Check the analysis plan |
| The code and data statement is accurate | The claim can be checked | Restricted or missing data | Request access or run a reproduction |
| The validation sample is independent of development | Performance is honest | Overlapping cohorts | Check the cohort definitions |
| The threshold is clinically meaningful | Net benefit is relevant | Threshold chosen for convenience | Justify from harm ratios |
| The outcome definition is consistent across sites | Pooling is valid | Different coding or ascertainment | Harmonise or stratify |
| The model version is the deployed version | The evidence applies | Retrained or updated model | Record the version identifier |
| The reported metric is pre-specified | The result is not selected | Metric chosen after seeing results | Check the analysis plan |
| The comparator is current practice | Net benefit is meaningful | Practice changed after the study | Re-run with a current comparator |
| The outcome definition is stable over time | Pooling is valid | Coding or guideline change | Check definition dates |
| The population is the deployment population | Transportability holds | Different age, setting, or case mix | Compare cohort definitions |
| The reporting standard is the current one | Completeness is judged correctly | Checklist superseded | Check the version |
| The outcome is measured the same way at all sites | Pooling is valid | Different assays or definitions | Harmonise or stratify |
| The follow-up is long enough for the outcome | The estimate is not truncated | Short follow-up | Check the risk window |
| The comparator reflects current care | The effect is relevant | Outdated control arm | Compare with current practice |
| The model is the version described in the paper | Evidence applies to the product | Silent retraining | Record the model identifier |
| Subgroup analyses are pre-specified | Subgroup claims are interpretable | Post hoc selection | Label as exploratory |

Two limits are structural. First, a methods citation cannot establish that the method suits the question; that is a judgement about the estimand and the decision. Second, a reporting checklist cannot establish that the analysis is correct; it establishes that the report is complete.

A third limit is practical. Many analyses cannot be fully reproduced because the data are restricted, and the correct response is to state the restriction and provide the code, not to claim reproducibility that the artefacts do not support.

A fourth limit is disciplinary. Methods conventions differ across fields, and a reviewer from one field may apply a standard that another field does not use. The citation set should name the standard being followed, so that the disagreement is about the standard rather than about the analysis.

## 12. Exercises

**Exercise 1 — Three-link citation (recall, 20 minutes).**
Take a methods section you have written and identify the citation for the idea, the implementation, and the reporting standard.
Deliverable: a three-row table with citation, version, and what each supports.
Acceptance criteria: all three links are present or explicitly missing; the software version is recorded; the checklist version is named.
Time: 20 minutes.

**Exercise 2 — Assumption audit (application, 30 minutes).**
For one model in your work, list the assumptions it requires and state how each was checked or argued.
Deliverable: an assumption table with assumption, plausibility, evidence, and sensitivity.
Acceptance criteria: at least one untestable assumption is argued rather than asserted; at least one sensitivity analysis is described.
Time: 30 minutes.

**Exercise 3 — Discrimination versus calibration (calculation, 30 minutes).**
Using a validation report, extract the AUC and any calibration statistics, and state what can and cannot be concluded.
Deliverable: a one-page assessment with the extracted numbers and two conclusions.
Acceptance criteria: the conclusions distinguish ranking from probability accuracy; missing calibration is stated as missing, not assumed.
Time: 30 minutes.

**Exercise 4 — Software default trace (application, 35 minutes).**
Choose a function you use, find its defaults in the documentation, and identify one default that would change your result if altered.
Deliverable: a short note with the function, version, default, and effect.
Acceptance criteria: the default is verified in the documentation for the installed version; the effect is described concretely; a recommendation is made.
Time: 35 minutes.

**Exercise 5 — Checklist mapping (application, 40 minutes).**
Map a methods section against the current reporting standard for its claim type and identify the missing items.
Deliverable: a completed checklist with locations and a list of gaps.
Acceptance criteria: the correct standard and version are used; every applicable item has a location or a stated reason for non-applicability.
Time: 40 minutes.

**Exercise 6 — Citation failure hunt (critique, 30 minutes).**
Find three methods citations in a document that fail in the ways described in section 10, and write the replacement citation set.
Deliverable: three findings with the failing link and the fix.
Acceptance criteria: each finding names the link (idea, implementation, or standard); the fix is specific; at least one finding concerns software.
Time: 30 minutes.

**Exercise 7 — Net benefit calculation (calculation, 40 minutes).**
Using a contingency table from your work, compute operating characteristics and net benefit at three thresholds, and compare with treat-all.
Deliverable: a table of results and a one-paragraph recommendation.
Acceptance criteria: the arithmetic is reproducible; the recommendation names a threshold; the comparison includes treat-all.
Time: 40 minutes.

**Exercise 8 — Methods source register (design, 35 minutes).**
Build a register for one project with the method, software, and reporting standard, each with a version and a review trigger.
Deliverable: a register with six to ten entries.
Acceptance criteria: every entry has a version; software entries are pinned; each entry has a trigger for review.
Time: 35 minutes.

### Marking guide

| Exercise | Evidence of a pass | Common failure |
|---|---|---|
| 1 Three links | All three present or missing | Only the method paper |
| 2 Assumptions | Untestable assumption argued | Assumptions listed without assessment |
| 3 Calibration | Missing stated as missing | Calibration assumed from AUC |
| 4 Defaults | Verified in installed version | Recalled from memory |
| 5 Checklist | Correct version used | TRIPOD 2015 in 2026 |
| 6 Failure hunt | Names the failing link | General criticism |
| 7 Net benefit | Arithmetic reproducible | AUC reported instead |
| 8 Register | Versions and triggers | A list of names |

Two of the exercises produce artefacts that can be reused directly. Exercise 5 produces a checklist mapping for a real manuscript, and Exercise 8 produces a register that can be maintained. The others build the judgement that makes those artefacts reliable.

A learner who has not run a model should still complete Exercise 7 using a published contingency table. The arithmetic is the lesson, not the modelling.

## 13. Self-check answers

**1. What are the three links in a methods citation?**
The idea (the paper or text that defines the estimator and its assumptions), the implementation (the software, version, and settings), and the reporting standard (the checklist and its version). A citation set that omits one of them is incomplete.

**2. Why is an AUC of 0.82 not evidence of clinical utility?**
Because AUC measures ranking across all thresholds and does not describe calibration or the net benefit at the threshold a clinic will use. A model can rank well and still produce probabilities that mislead decisions.

**3. What does a calibration slope of 0.5 mean?**
It means the model's predictions are too extreme: high predicted risks are too high and low predicted risks are too low relative to observed outcomes. The model needs recalibration before its probabilities are used.

**4. When is a hazard ratio the wrong metric?**
When the claim is about prediction at a fixed time horizon, about absolute risk, or about a decision at a threshold. A hazard ratio describes a relative rate under proportional hazards and does not directly give the probability a decision needs.

**5. Why does the unit of resampling matter?**
Because observations from the same patient are not independent. Resampling rows instead of patients produces intervals that are too narrow and performance estimates that are too optimistic, and the error is invisible in the point estimate.

**6. What does a software version change alter?**
It can alter defaults, numerical routines, tie handling, and convergence behaviour. The same function name can produce different results, which is why the version is part of the method.

**7. Why does TRIPOD+AI supersede TRIPOD 2015?**
Because it updates the items for prediction models using regression or machine learning, adds open-science and patient-involvement items, and harmonises terminology. A methods section written to the older checklist can be complete by that standard and incomplete by the current one.

**8. What is the difference between reproducibility and replicability?**
Reproducibility is obtaining the same result from the same data and code. Replicability is obtaining a consistent result from a new study or dataset. A reproducible analysis can fail to replicate, and that is a scientific finding rather than a computational error.

**Rubric for Exercises 5 and 7.** A strong checklist mapping uses the correct version, locates every applicable item, and states why an item does not apply. A weak mapping ticks items without locations. A strong net benefit calculation shows the arithmetic, compares with treat-all, and recommends a threshold with a reason. A weak calculation reports a number without the comparator.

**9. Why is a calibration curve more informative than a single slope?**
A slope summarises the average agreement across the risk range, while a curve shows where the model deviates. A model can have a slope near 1 and still be badly calibrated at the high-risk end, which is where decisions are often made.

**10. What does it mean if the net benefit is positive but the alert volume is unmanageable?**
It means the model has decision value in principle and an implementation problem in practice. The deployment decision requires both, and the methods section should report both rather than only the statistical result.

**11. Why is external validation not the same as transportability?**
External validation tests performance in one other setting. Transportability is the claim that performance holds across a range of settings, which requires validation in more than one and an argument about the mechanisms that would make it transfer.

**12. What should be reported for a missing-data method?**
The mechanism assumed, the method used, the proportion missing by variable, and a sensitivity analysis under a plausible alternative. Complete-case analysis without these is an unstated assumption.

**13. Why can a large AUC be consistent with a low PPV?**
Because PPV depends on prevalence. At a low prevalence, even a well-discriminating model produces many false positives for each true positive, which is why the same model can have a different PPV in a screening population and a referral population.

**14. What is the risk of tuning on the test set?**
It produces an optimistic estimate that cannot be reproduced on new data, because the test set has become part of model selection. The reported performance is then a property of the selection procedure, not of the model.

**15. Why is a decision curve threshold not a property of the model alone?**
Because the threshold encodes the relative harm of a false positive and a false negative, which is a clinical and organisational judgement. The model supplies the curve; the clinic supplies the threshold.

**16. What should a methods register contain for a software dependency?**
The package name, the exact version, the source of the version (lock file or container), the licence, and a review trigger for security or major updates.

**17. Why is a simulation study not evidence that a method works in your data?**
Because the simulation's data-generating process defines the conditions under which the method was assessed. If those conditions differ from yours, the performance claim does not transfer, and the simulation is a description of a scenario rather than a guarantee.

**18. What is the difference between a confidence interval and a credible interval?**
A confidence interval is a frequentist procedure with a stated long-run coverage property. A credible interval is a posterior probability statement under a prior. They can be numerically similar and mean different things.

## 14. Interview questions

**1. How do you decide whether a model is ready for deployment?**
Strong answer: external discrimination, calibration in the target population, net benefit at the intended threshold, alert volume, and a plan for monitoring. Weak answer: cites AUC alone.

**2. What is the difference between discrimination and calibration?**
Strong answer: discrimination is ranking, calibration is agreement between predicted and observed risk, and both are needed for a probability-based decision. Weak answer: treats them as synonyms.

**3. Why do you record the software version?**
Strong answer: defaults and numerical routines change between versions, and the same function can produce different results; the version is part of the method. Weak answer: says it is for auditing.

**4. What assumptions does a propensity score analysis require?**
Strong answer: no unmeasured confounding, positivity, correct specification of the propensity model, and a well-defined intervention. Weak answer: says matching balances the groups.

**5. How do you handle a method that has been criticised in the literature?**
Strong answer: cites the critique, states how the analysis addresses it, and reports a sensitivity analysis. Weak answer: ignores the critique or abandons the method without analysis.

**6. What is an estimand and why does it matter?**
Strong answer: the precise quantity being estimated, including population, outcome, and intercurrent events; different estimands answer different questions and require different methods. Weak answer: calls it the research question.

**7. How do you check that a package implements the method correctly?**
Strong answer: compares against an independent implementation on a known example, checks the package tests, and reads the documentation for defaults. Weak answer: assumes a widely used package is correct.

**8. What would you report for a prediction model besides AUC?**
Strong answer: calibration slope and intercept or a calibration curve, net benefit at relevant thresholds, subgroup performance, and the validation design. Weak answer: adds sensitivity and specificity only.

**9. Why is a p-value not a measure of effect?**
Strong answer: it describes compatibility with a null hypothesis and depends on sample size; the effect size and interval carry the magnitude and uncertainty. Weak answer: treats significance as importance.

**10. How do you make an analysis reproducible for a reviewer?**
Strong answer: version-pinned environment, seed, data dictionary, runnable example, and a tagged repository release. Weak answer: shares a script without versions or data description.

**11. What is the difference between a reporting checklist and a risk-of-bias tool?**
Strong answer: a checklist asks whether the report contains an item; a risk-of-bias tool asks whether the design supports the claim. Weak answer: treats them as interchangeable.

**12. How do you handle a method whose assumptions are clearly violated?**
Strong answer: names the violation, explains the direction of the likely bias, runs a sensitivity analysis or an alternative estimator, and weakens the conclusion accordingly. Weak answer: proceeds and mentions the assumption in a footnote.

**13. Why is the number of events per variable important?**
Because models with many parameters relative to events overfit, producing optimistic apparent performance and unstable coefficients. The threshold is not universal, and the principle is that the effective complexity must be supported by the data.

**14. What does a decision curve add to a calibration plot?**
It translates calibration and discrimination into the consequences of using the model at a threshold, compared with treating all or none. It answers a different question, and a good model on one can be unimpressive on the other.

**15. How would you review a methods section in fifteen minutes?**
Strong answer: reads the estimand, the metric, the validation design, the calibration, the software version, and the checklist; recomputes one number. Weak answer: reads the abstract and the limitations.

**16. What is the risk of citing a method paper you have not read?**
Strong answer: the paper may not define the estimator you used, may state assumptions you have not met, or may have been superseded by a correction. Weak answer: says it is a well-known paper.

**17. How do you report a model that is well calibrated but poorly discriminating?**
Strong answer: reports both, explains that the model may still be useful for a narrow decision or as a component, and does not claim ranking ability it lacks. Weak answer: reports only the calibration.

**18. What does it mean when two models have overlapping confidence intervals for AUC?**
It means the data do not establish a ranking between them, regardless of the point estimates. The comparison needs a paired test or a decision-analytic comparison.

**19. Why is the unit of analysis a methods question and not a data question?**
Because it determines what the observations are independent of, which determines the variance, the model, and the resampling. A row is not always the unit, and treating it as one is a modelling error.

**20. How would you explain net benefit to a clinical audience?**
Strong answer: it compares the model to treating everyone and to treating no one at the threshold the clinic would use, in units of net true positives per patient. Weak answer: explains AUC.

**21. What is the difference between a pre-specified and an exploratory analysis?**
Strong answer: a pre-specified analysis was fixed before seeing outcome data, so its error rates are interpretable; an exploratory analysis generates hypotheses and should be labelled. Weak answer: treats them as the same analysis with different wording.

**22. Why might a simpler model be preferred to a more accurate one?**
Strong answer: transportability, calibration stability, interpretability for clinicians, fewer failure modes, and easier maintenance, with the accuracy difference quantified. Weak answer: says simpler is always better.

## 15. Key terms

- **Estimand** — the precise quantity being estimated.
- **Discrimination** — the ability to rank by risk.
- **Calibration** — agreement between predicted and observed risk.
- **Calibration slope** — steepness of observed on predicted risk.
- **Calibration intercept** — systematic over- or under-prediction.
- **Net benefit** — decision value at a threshold.
- **Decision curve** — net benefit across thresholds.
- **AUC** — probability of correct ranking of a random case and non-case.
- **Brier score** — mean squared error of predicted probabilities.
- **Optimism** — inflated apparent performance from overfitting.
- **Internal validation** — validation within the development data.
- **External validation** — validation in a separate population or setting.
- **Temporal validation** — validation in a later period.
- **Positivity** — every covariate pattern has non-zero exposure probability.
- **Consistency** — the observed outcome under the received treatment equals the potential outcome.
- **Target trial** — the hypothetical trial an observational analysis emulates.
- **Implementation** — the software that computes the estimator.
- **Default** — a setting used when none is specified.
- **Seed** — the value that makes a random process reproducible.
- **Version pin** — recording the exact software version.
- **Resampling unit** — the entity resampled in bootstrap or cross-validation.
- **Reproducibility** — same data and code, same result.
- **Replicability** — new data, consistent result.
- **Reporting standard** — a checklist defining a complete report.
- **Risk-of-bias tool** — an instrument assessing whether a design supports a claim.
- **PROBAST** — risk-of-bias tool for prediction models.
- **TRIPOD+AI** — reporting guideline for prediction models, 2024.
- **CONSORT-AI** — reporting guideline for AI trials.
- **DECIDE-AI** — reporting guideline for early clinical evaluation.
- **Estimand drift** — a mismatch between the modelled quantity and the interpreted quantity.
- **Overfitting** — fitting noise, producing optimistic apparent performance.
- **Events per variable** — the ratio of outcome events to model parameters.
- **Shrinkage** — penalisation that reduces overfitting.
- **Bootstrapping** — resampling to estimate uncertainty or optimism.
- **Cross-validation** — held-out folds for performance estimation.
- **Grouped split** — a split that keeps related records together.
- **Data leakage** — information from the outcome or future entering the features.
- **Sensitivity analysis** — a re-analysis under a different plausible assumption.
- **Pre-specification** — stating the analysis before seeing the results.
- **Analysis plan** — the document that records the intended analyses.
- **Null imputation** — a sensitivity approach for missing data.
- **Lock file** — a record of exact dependency versions.
- **Container** — a packaged environment that reproduces the software stack.
- **Deterministic** — a method with no random component.
- **Stochastic** — a method whose result depends on random draws.
- **Hyperparameter** — a setting chosen outside the fitting procedure.
- **Regularisation** — penalisation of model complexity.
- **Convergence** — whether an iterative algorithm reached a stable solution.
- **Separation** — a predictor that perfectly separates outcome classes.
- **Leakage** — use of information not available at prediction time.
- **Transportability** — whether performance holds in a new setting.
- **Case mix** — the distribution of patient characteristics in a population.
- **Spectrum bias** — performance varying with the case mix of the validation sample.
- **Verification bias** — differential outcome ascertainment by test result.
- **Attrition** — loss of participants from a cohort.
- **Intercurrent event** — an event after treatment start that affects interpretation.
- **Proportional hazards** — a constant hazard ratio over time.
- **Non-inferiority** — a design testing that an effect is not worse by more than a margin.
- **Equivalence margin** — the largest difference considered clinically acceptable.
- **Intention-to-treat** — analysis by assigned group.
- **Per-protocol** — analysis of adherent participants.
- **Estimand framework** — the ICH E9(R1) approach to specifying what is estimated.
- **Missing at random** — a missingness mechanism conditional on observed data.

## 16. Sources

All URLs were checked on 26 October 2026. DOIs resolve to the cited works.

**Collins, G. S., Moons, K. G. M., Dhiman, P., Riley, R. D., Beam, A. L., Van Calster, B., et al.** "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods." *BMJ* 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
Good for: the current reporting standard for prediction models, including calibration, fairness, and open science. Not for: risk-of-bias assessment, which is PROBAST. It supersedes TRIPOD 2015.

**Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., Steyerberg, E. W.** "Calibration: the Achilles heel of predictive analytics." *BMC Medicine* 2019;17:230. https://doi.org/10.1186/s12916-019-1466-7
Good for: why calibration matters, how to assess it, and why the Hosmer-Lemeshow test is insufficient on its own. Not for: a decision-analytic metric, which is decision curve analysis.

**Vickers, A. J., and Elkin, E. B.** "Decision curve analysis: a novel method for evaluating prediction models." *Medical Decision Making* 2006;26(6):565-574. https://doi.org/10.1177/0272989X06295361
Good for: net benefit and the threshold-based evaluation of prediction models. Not for: causal inference or model development.

**Hernán, M. A., and Robins, J. M.** "Using big data to emulate a target trial when a randomized trial is not available." *American Journal of Epidemiology* 2016;183(8):758-764. https://doi.org/10.1093/aje/kwv254
Good for: the target trial framework and the estimand-first approach to observational analysis. Not for: a software implementation.

**Rosenbaum, P. R., and Rubin, D. B.** "The central role of the propensity score in observational studies for causal effects." *Biometrika* 1983;70(1):41-55. https://doi.org/10.1093/biomet/70.1.41
Good for: the definition and balancing property of the propensity score. Not for: practical guidance on modern matching estimators or their failure modes.

**Steyerberg, E. W.** *Clinical Prediction Models*. Springer.
Good for: the development, validation, and updating of prediction models, including calibration and optimism correction. Not for: the current reporting checklist, which is TRIPOD+AI.

**Harrell, F. E.** *Regression Modeling Strategies*. Springer.
Good for: model specification, validation, and the limits of automated variable selection. Not for: causal identification.

**PROBAST.** Risk-of-bias tool for prediction model studies. https://www.probast.org/
Good for: assessing whether a prediction study's design supports its conclusions. Not for: reporting completeness.

**EQUATOR Network.** Reporting guideline library. https://www.equator-network.org/
Good for: identifying the correct checklist and its current version for a study design. Not for: appraising the quality of a completed study.

**DECIDE-AI.** Reporting guideline for early clinical evaluation of AI. https://doi.org/10.1136/bmj-2022-070904
Good for: the reporting of early live clinical evaluations of AI systems. Not for: full randomised trials, which use CONSORT-AI.

**van Smeden, M., et al.** Work on sample size and events per variable in prediction models. A starting point is the PROBAST and TRIPOD documentation.
Good for: understanding why small development samples produce optimistic models. Not for: a single universal threshold; the guidance depends on the model and the intended use.

> **Note:** The last entry is a pointer rather than a single verified citation. The specific sample-size guidance is spread across several methods papers, and a reader should consult the TRIPOD+AI explanation and elaboration document for the current recommendation rather than relying on a rule of thumb.

**Little, R. J. A., and Rubin, D. B.** *Statistical Analysis with Missing Data*. Wiley.
Good for: missing-data mechanisms and the assumptions behind multiple imputation. Not for: a software tutorial.

**Efron, B., and Tibshirani, R. J.** *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
Good for: the bootstrap and its resampling logic. Not for: the clustered or hierarchical extensions, which need additional sources.

**LeCun, Y., Bengio, Y., and Hinton, G.** "Deep learning." *Nature* 2015;521:436-444. https://doi.org/10.1038/nature14539
Good for: the conceptual foundations of representation learning. Not for: clinical validation or reporting standards.

**Vaswani, A., et al.** "Attention is all you need." *Advances in Neural Information Processing Systems* 2017. https://arxiv.org/abs/1706.03762
Good for: the transformer architecture. Not for: clinical applications or evaluation.

**Collins, G. S., et al.** TRIPOD+AI explanation and elaboration document. Available through the TRIPOD statement site. https://www.tripod-statement.org/
Good for: item-by-item guidance on what the checklist requires. Not for: a risk-of-bias assessment.

**Reproducible research resources.** Project Jupyter, R Markdown, and container tooling documentation.
Good for: practical implementation of the reproducibility practices in this topic. Not for: the methodological argument, which is in the references above.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_regulation-and-policy.md`](03_regulation-and-policy.md)
- Next topic: [`05_clinical-background.md`](05_clinical-background.md)
- Datasets and benchmarks: [`06_datasets-and-benchmarks.md`](06_datasets-and-benchmarks.md)
- Survey file: [`../../course/22_sources_and_further_reading.md`](../../course/22_sources_and_further_reading.md)
- Labs: [`../../labs/`](../../labs/)
- Module 04, statistics: [`../04_statistics/`](../04_statistics/)
- Module 06, machine learning: [`../06_machine_learning/`](../06_machine_learning/)
- Module 05, study design: [`../05_study_design/`](../05_study_design/)
- Module 15, MLOps: [`../15_mlops/`](../15_mlops/)
- Module 18, quality and appraisal: [`../18_quality/`](../18_quality/)
- Module 21, dataset selection: [`../21_datasets/01_five-gate-decision.md`](../21_datasets/01_five-gate-decision.md)
- Module 07, deep learning: [`../07_deep_learning/`](../07_deep_learning/)
- Module 19, capstone: [`../19_capstone/`](../19_capstone/)
- Module 10, imaging methods: [`../10_imaging/`](../10_imaging/)
- Module 11, signal methods: [`../11_signals/`](../11_signals/)
- Module 12, omics methods: [`../12_omics/`](../12_omics/)
- Survey checklist: [`../../course/22_sources_and_further_reading.md`](../../course/22_sources_and_further_reading.md) section 10

The next topic moves from methods to clinical background, where the currency problem is definitional: a term can stay the same while its definition changes.

That distinction is the subject of the next file, and it is harder than it sounds because the vocabulary is stable and only the meaning moves.

