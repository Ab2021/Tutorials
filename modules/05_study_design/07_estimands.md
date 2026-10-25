# Estimands

> Module 05 — Study Design and Evidence Generation
> Topic 07 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 02, and 06
> Version 2026.3

## 1. Why this topic exists

Every effect estimate is an estimate of something. The estimand is the precise definition of that something. It states who is being studied, what treatments are being compared, what outcome is measured, how events that interrupt treatment are handled, and what summary measure is used. Without it, the analysis is a calculation in search of a question.

The problem is not academic. Two trials can use the same drug, the same endpoint, and the same population, and produce different results because they handle discontinuation differently. One trial counts a patient who stopped treatment as a failure; another counts the patient's outcome regardless of stopping; a third estimates the effect only in patients who tolerated treatment. The three results answer three different questions. If the label, the payer, and the clinician assume they are the same, the evidence is misused.

The International Council for Harmonisation (ICH) addressed this in the E9(R1) addendum on estimands and sensitivity analysis, adopted at Step 4 on 20 November 2019. The European Medicines Agency (EMA) adopted it at Step 5 on 30 January 2020, with effect from 30 July 2020. The US Food and Drug Administration (FDA) issued it as final guidance on 12 May 2021. The addendum requires the estimand to be defined before the design and analysis, and it requires the handling of intercurrent events to be specified.

The commercial consequence is direct. A label claim is a statement about an estimand. If the trial estimates a hypothetical effect ("if everyone adhered") but the label describes a treatment-policy effect ("as used in practice"), the label overstates or understates the benefit. A payer who reads the label as a treatment-policy effect will expect a different result from a clinician who reads it as a hypothetical effect. The mismatch is a commercial and clinical risk.

This topic teaches you to write an estimand, choose the intercurrent-event strategies, align the estimator, and report the result so that the claim and the number match.

### The estimand is not the endpoint

The endpoint is the variable: all-cause mortality at 12 months. The estimand adds the population, the treatment conditions, the handling of intercurrent events, and the summary measure. The same endpoint can support several estimands:

- the effect of assignment on mortality, regardless of discontinuation;
- the effect of treatment if everyone adhered;
- the effect in patients who would tolerate treatment;
- the effect on a composite of mortality and discontinuation.

Each is a legitimate question. Each requires a different analysis. The protocol must say which one it is answering.

### Why the topic is hard

Three pressures make estimands difficult.

1. **Clinical language is ambiguous.** "Does the drug work?" can mean "does it work when taken?" or "does offering it help?" The two questions have different answers and different policy implications.
2. **The analysis defaults to a method, not a question.** Analysts are trained to run intention-to-treat, per-protocol, or a composite, and the choice is often made by habit rather than by the question.
3. **Intercurrent events are common.** Discontinuation, rescue medication, switching, and death occur in most trials. Ignoring them does not make them disappear; it makes the estimand implicit and the interpretation ambiguous.

The E9(R1) framework is a response to these pressures. It does not prescribe a single correct estimand. It requires the team to choose one, justify it, and align the analysis with it.

### The consequences of an undefined estimand

- The label may describe a different effect from the one the trial estimated.
- The payer may model a different effect from the one the trial reported.
- The clinician may expect a different effect from the one the patient experiences.
- The sensitivity analyses may test the wrong assumptions.
- The protocol may be unreviewable because the question is not stated.
- The trial may be uninterpretable even if it is statistically significant.

The consequences are not hypothetical. They are the reason the addendum was written.

### What an estimand buys

- A protocol that can be reviewed against a stated question.
- A label that describes the effect the trial estimated.
- A payer dossier that uses the right effect.
- A clinician who knows whether the effect is the effect of offering or of taking.
- A sensitivity analysis that tests the right assumptions.
- A result that can be compared with other trials.

The list is the practical case for the framework. The statistical case is that the estimand is the only way to make the analysis falsifiable.

## 2. Learning outcomes

By the end of this topic you can:

1. **Write** an estimand with the five attributes for a given clinical question. (Exercise 1, self-check 1)
2. **Identify** intercurrent events in a protocol and distinguish them from missing data. (Exercise 2, self-check 2)
3. **Choose** among the five strategies for each event and justify the choice from the claim. (Exercise 3, interview 3)
4. **Align** the estimator and the analysis population with the estimand. (Exercise 4, self-check 4)
5. **Distinguish** a treatment-policy effect from a hypothetical effect, and explain how the same data produce different numbers. (Exercise 5, self-check 5)
6. **Design** sensitivity analyses that test the robustness of the estimand assumptions. (Exercise 6, interview 6)
7. **Critique** a published trial's estimand and propose a more appropriate one for the decision. (Exercises 7 and 8, interview 8)

### How the outcomes connect

- Outcomes 1 and 2 establish the definition and the event list.
- Outcomes 3 and 4 connect the definition to the analysis.
- Outcomes 5 and 6 test the definition under alternative assumptions.
- Outcomes 7 and 8 apply the framework to real evidence.
- A learner who can perform outcomes 1 to 4 can write an estimand.
- A learner who can also perform outcomes 5 to 8 can defend it in review.

## 3. The core mental model

The estimand is a bridge between the clinical question and the analysis. It has five attributes, and each attribute constrains a design or analysis choice.

```text
   CLINICAL QUESTION
   "Does the drug help patients like these?"
              |
              v
   ESTIMAND (five attributes)
   1. Population
   2. Treatment condition
   3. Variable (endpoint)
   4. Intercurrent events and strategy
   5. Population-level summary
              |
              v
   ESTIMATOR
   ITT | per-protocol | IPW | composite | principal stratum
              |
              v
   ANALYSIS POPULATION
   all randomised | adherent | tolerators | at risk
              |
              v
   ESTIMATE AND SENSITIVITY ANALYSES
              |
              v
   CLAIM AND LABEL
```

The bridge is bidirectional in review. If the estimate does not match the claim, either the estimand or the claim must change. The analysis cannot repair a mismatch between the question and the number.

### The five attributes

| Attribute | Question | Example |
|---|---|---|
| Population | Who is the target? | Adults with heart failure and reduced ejection fraction |
| Treatment condition | What treatments, doses, durations, and concomitant care? | Drug X 10 mg daily versus placebo, added to standard care |
| Variable | What outcome, measured how, at what time? | All-cause mortality or heart failure hospitalisation at 12 months |
| Intercurrent events | What events change the interpretation, and how are they handled? | Discontinuation due to adverse events; rescue therapy; transplant |
| Population-level summary | What contrast is estimated? | Risk ratio at 12 months |

A common error is to treat the variable as the whole estimand. The variable is one attribute. The other four determine what the variable means.

### The five strategies for intercurrent events

ICH E9(R1) defines five strategies. Each answers a different question.

| Strategy | Question it answers | Example | Analysis implication |
|---|---|---|---|
| Treatment policy | What is the effect of assigning the strategy, regardless of the event? | Effect of offering drug X, whether or not patients stop | Include all outcomes regardless of discontinuation |
| Hypothetical | What would the effect be if the event did not occur? | Effect if every patient tolerated and adhered | Model the counterfactual; requires assumptions |
| Composite | Does the event itself count as a failure? | Death, hospitalisation, or discontinuation | Include the event in the endpoint |
| While on treatment | What is the effect while the patient is on treatment? | Effect on blood pressure while taking the drug | Restrict observation to the on-treatment period |
| Principal stratum | What is the effect in patients who would (or would not) experience the event? | Effect in patients who would tolerate the drug | Estimate in a latent subgroup; requires assumptions |

The choice is not a statistical preference. It is a statement about what the decision-maker needs to know. A regulator may want a treatment-policy effect because it reflects usual use. A clinician may want a hypothetical effect because it reflects biological efficacy. A payer may want a composite because it reflects the full burden.

### Estimand, estimator, and estimate

Three terms are often confused.

- The **estimand** is the quantity to be estimated. It is defined by the five attributes.
- The **estimator** is the method used to estimate it. It is a statistical procedure.
- The **estimate** is the numerical result.

A treatment-policy estimand is often estimated with an intention-to-treat analysis, but the two are not identical. Intention-to-treat is a principle; the estimand is adefinition. A hypothetical estimand cannot be estimated by a standard intention-to-treat analysis without additional assumptions.

The protocol should state all three: the estimand, the estimator, and the planned sensitivity analyses. A protocol that states only the estimator has not defined the question.

### Estimand and design interactions

| Design feature | Estimand consequence | Example |
|---|---|---|
| Follow-up length | Determines the variable's horizon | 12-month versus 5-year mortality |
| Visit schedule | Determines ICE ascertainment | Discontinuation detected at visits |
| Blinding | Affects ICE reporting | Adverse events reported differently |
| Comparator | Defines the treatment condition | Placebo versus active comparator |
| Allocation ratio | Affects precision, not the estimand | 2:1 versus 1:1 |
| Cluster allocation | Changes the population and the ICE | Cluster-level discontinuation |
| Adaptive design | Must preserve the estimand | Dose switching at interim |
| Decentralised elements | Affects measurement and missingness | Home versus site assessment |

The table shows that the estimand is not independent of the design. A design choice that changes the ICE profile changes the estimand's interpretation, even if the sentence is unchanged.

### Intercurrent events versus missing data

An intercurrent event is an event after treatment initiation that affects the interpretation or existence of the outcome. Missing data is the absence of an outcome measurement. The distinction matters because they are handled differently.

- Discontinuation of treatment is an intercurrent event. It is handled by the estimand strategy.
- Withdrawal of consent is missing data. It is handled by the missing-data model.
- Death is an intercurrent event for some outcomes and a competing event for others. It is handled by the estimand strategy or a competing-risk analysis.
- A missed visit is missing data. It is handled by imputation or a likelihood-based method.

A protocol that treats discontinuation as missing data confuses the question with the analysis. A protocol that treats withdrawal as an intercurrent event may overcomplicate the estimand.

### The estimand on one page

A one-page estimand summary for a protocol contains:

- the question in one sentence;
- the five attributes in a table;
- the ICE list and the strategy for each;
- the estimator and analysis population;
- the missing-data plan;
- the sensitivity analyses;
- the label claim, compared with the estimand;
- the untestable assumptions.

The page is a communication device. It can be reviewed by a clinician, a statistician, and a regulatory lead in a single meeting.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Estimand | The precise quantity to be estimated, defined by five attributes | Confused with the endpoint or the estimator | Structured definition |
| Estimator | The statistical method used to estimate the estimand | Confused with the estimand | Method name |
| Estimate | The numerical result | Treated as the estimand | Effect measure |
| Intercurrent event (ICE) | An event after treatment initiation that affects the interpretation or existence of the outcome | Confused with missing data | Event category |
| Treatment policy | A strategy that ignores the ICE and uses the outcome regardless | Confused with intention-to-treat | Strategy |
| Hypothetical | A strategy that estimates the effect in a scenario where the ICE did not occur | Confused with per-protocol | Strategy |
| Composite | A strategy that includes the ICE in the endpoint | Confused with a co-primary endpoint | Strategy |
| While on treatment | A strategy that restricts observation to the on-treatment period | Confused with per-protocol | Strategy |
| Principal stratum | A strategy that targets a latent subgroup defined by the potential occurrence of the ICE | Confused with a subgroup defined by observed events | Strategy |
| Population-level summary | The contrast used to summarise the effect | Confused with the endpoint | Effect measure |
| Treatment condition | The intervention, comparator, dose, duration, and concomitant care | Confused with the treatment arm label | Description |
| Variable | The outcome, its measurement, and its timing | Confused with the estimand | Endpoint definition |
| Analysis population | The set of participants included in an analysis | Confused with the estimand population | Population definition |
| Intention-to-treat (ITT) | Analysis by assigned group regardless of treatment received | Confused with treatment policy | Principle |
| Per-protocol | Analysis restricted to adherent participants | Confused with hypothetical | Population |
| As-treated | Analysis by treatment received | Confused with per-protocol | Population |
| Missing data | Absence of an outcome measurement | Confused with an ICE | Data problem |
| Withdrawal of consent | A participant's request to stop participating | Confused with discontinuation | Data problem |
| Discontinuation | Stopping the assigned treatment | Confused with withdrawal | ICE |
| Rescue medication | Additional treatment given for lack of efficacy | Ignored or treated as protocol deviation | ICE |
| Switching | Changing to another treatment | Ignored or treated as discontinuation | ICE |
| Death | A terminal event | Treated as missing data | ICE or competing event |
| Competing risk | An event that prevents the outcome of interest | Confused with an ICE | Analysis concept |
| Principal stratum | A subgroup defined by potential outcomes under both treatments | Confused with an observed subgroup | Latent population |
| Survivor average causal effect | The effect in patients who would survive to a time point | Confused with the overall effect | Estimand |
| While-on-treatment effect | The effect during the period of treatment | Confused with the effect of assignment | Estimand |
| Treatment-policy effect | The effect of assigning the strategy, regardless of subsequent events | Confused with biological efficacy | Estimand |
| Hypothetical effect | The effect under a counterfactual scenario | Confused with the observed effect | Estimand |
| Sensitivity analysis | An analysis that tests the robustness of the result to an assumption | Confused with a subgroup analysis | Analysis |
| Principal ignorability | The assumption that potential outcomes are independent of principal stratum given covariates | Assumed without discussion | Assumption |
| Intercurrent-event strategy | The rule for handling an ICE in the estimand | Confused with the analysis method | Strategy |
| Tipping-point analysis | An analysis that finds the departure from an assumption that changes the conclusion | Confused with a sensitivity analysis | Analysis |
| Reference standard | The best available method for determining the outcome | Confused with the test under study | Diagnostic concept |
| Diagnostic accuracy estimand | The sensitivity and specificity of a test against a reference standard | Confused with a treatment effect | Estimand |
| Attributable effect | The effect of treatment in those who receive it | Confused with the average effect | Estimand |
| Survivor bias | Bias from conditioning on survival | Confused with an ICE strategy | Bias |
| Treatment-policy effect | The effect of assigning a strategy | Confused with the effect of receiving treatment | Estimand |

Three distinctions deserve emphasis.

**Treatment policy versus intention-to-treat.** Treatment policy is an estimand strategy: the outcome is used regardless of the ICE. Intention-to-treat is an analysis principle: participants are analysed by assigned group. They usually align, but a treatment-policy estimand can require additional assumptions if the outcome is missing after discontinuation.

**Hypothetical versus per-protocol.** A hypothetical estimand asks what would happen if the ICE did not occur. A per-protocol analysis restricts to participants who did not experience the ICE. The per-protocol analysis is an estimator, not an estimand, and it can be biased if the ICE is related to prognosis.

**Principal stratum versus subgroup.** A subgroup is defined by an observed characteristic. A principal stratum is defined by a potential outcome: the patients who would tolerate treatment, whether or not they were assigned to it. The principal stratum cannot be identified from observed data without assumptions.

## 5. The mechanics, step by step

The procedure has fourteen steps. Steps 1 to 4 define the question and the variable. Steps 5 to 8 define the intercurrent events and the estimand. Steps 9 to 12 align the analysis. Steps 13 and 14 check and document.

### Procedure at a glance

| Step | Input | Output | Failure if skipped |
|---:|---|---|---|
| 1 | Claim and decision | Clinical question | Ambiguous target |
| 2 | Question and eligibility | Population attribute | Estimand applies to wrong patients |
| 3 | Intervention and care | Treatment condition | Undefined contrast |
| 4 | Outcome and horizon | Variable attribute | Wrong endpoint |
| 5 | Protocol and pathway | ICE list | Incomplete estimand |
| 6 | ICE list and claim | Strategy per event | Ambiguous effect |
| 7 | Variable and decision | Summary measure | Uninterpretable contrast |
| 8 | Five attributes | Estimand sentence | No testable definition |
| 9 | Estimand | Estimator | Mismatch |
| 10 | Estimand and estimator | Analysis population | Population mismatch |
| 11 | ICEs and missingness | Separate plans | Confused handling |
| 12 | Assumptions | Sensitivity analyses | Unknown robustness |
| 13 | Estimand and design | Alignment statement | Design cannot identify the estimand |
| 14 | All of the above | Protocol and report | Unauditable claim |

### A library of estimand templates

The following templates show the sentence structure for common questions. Replace the bracketed terms.

- **Treatment policy:** "In [population], the [summary] of [variable] at [time], comparing [intervention] with [comparator], regardless of [ICE list]."
- **Hypothetical:** "In [population], the [summary] of [variable] at [time], comparing [intervention] with [comparator], in the hypothetical scenario where [ICE] did not occur."
- **Composite:** "In [population], the [summary] of the composite of [variable] and [ICE] at [time], comparing [intervention] with [comparator]."
- **While on treatment:** "In [population], the [summary] of [variable] during the on-treatment period, comparing [intervention] with [comparator], with observation censored at [ICE]."
- **Principal stratum:** "In [population] patients who would [tolerate/respond to] [intervention], the [summary] of [variable] at [time], comparing [intervention] with [comparator]."

The template does not replace the five-attribute table. It is a check that the sentence contains all five attributes.

### Common estimand patterns

| Setting | Typical ICE | Common strategy | Reason |
|---|---|---|---|
| Confirmatory drug trial | Discontinuation due to AE | Treatment policy | Reflects labelled use |
| Efficacy trial in a responsive subgroup | Non-adherence | Hypothetical | Reflects biological effect |
| Oncology | Subsequent therapy | While on treatment or composite | PFS conventions |
| Rare disease | Death | Composite or competing risk | Small samples |
| Device trial | Crossover to the device | Treatment policy | Reflects assignment |
| Pragmatic trial | Co-interventions | Treatment policy | Reflects routine care |
| Non-inferiority | Discontinuation | Both ITT and per-protocol | Assay sensitivity |
| Diagnostic study | None | Descriptive concordance | No treatment |

The table is a starting point. The strategy should be justified by the question, not copied from convention.

### Step 1: Write the clinical question in decision terms

**Input:** the claim and the decision.

**Output:** a question that names the population, the decision, and the consequence.

**Rationale:** The estimand is a translation of the question. If the question is ambiguous, the estimand will be arbitrary.

A useful format is: "In [population], should we offer [treatment] rather than [comparator] to achieve [outcome] over [horizon], and what would that mean for [decision]?" The decision determines whether the relevant effect is the effect of offering, the effect of taking, or the effect in a subgroup.

### Step 2: Define the population

**Input:** the clinical question and the eligibility criteria.

**Output:** a population definition that matches the intended use and the decision.

**Rationale:** The population attribute determines who contributes to the estimate. A broad population supports a broad claim; a narrow population supports a narrow claim.

The population should be defined by clinical characteristics and by the setting. It should also state whether the estimand applies to a subgroup, and if so, whether the subgroup is defined by a baseline characteristic or by a post-randomisation event.

### Step 3: Define the treatment condition

**Input:** the intervention, the comparator, and the care pathway.

**Output:** a treatment condition that specifies dose, duration, titration, concomitant care, and the comparator.

**Rationale:** The treatment condition determines what is being compared. "Drug X versus placebo" is incomplete without the dose, the duration, and the background therapy.

The treatment condition should also state the start and stop rules. A strategy that allows dose reduction is different from a strategy that requires discontinuation after the first adverse event.

### Step 4: Define the variable

**Input:** the outcome and the decision.

**Output:** a variable definition with the measurement, the timing, and the handling of death.

**Rationale:** The variable is the outcome attribute. It determines what is measured and when. The timing must match the decision horizon.

The variable should state whether it is a single event, a composite, a repeated measure, or a time-to-event outcome. For a composite, the components and their relative importance should be stated.

### Step 5: Identify intercurrent events

**Input:** the protocol, the care pathway, and the clinical context.

**Output:** a list of intercurrent events with their expected frequency and their relationship to the outcome.

**Rationale:** Intercurrent events are the events that make the estimand ambiguous. They must be identified before the strategy is chosen.

The common events are:

- discontinuation of the assigned treatment,
- dose reduction or interruption,
- rescue medication,
- switching to another treatment,
- initiation of a contraindicated concomitant treatment,
- death,
- transplant or surgery,
- pregnancy,
- loss to follow-up (which is missing data, not an ICE).

The list should be specific to the trial. A heart failure trial has different events from an oncology trial.

For each event, record:

- the definition and the ascertainment method;
- the expected frequency by arm;
- whether the event is terminal or non-terminal;
- whether the event is related to efficacy, tolerability, or the care pathway;
- whether the event is expected to differ by arm;
- whether the event is already part of the endpoint;
- whether the event can be measured after it occurs.

The record is the basis for the strategy choice. An event that is expected to differ by arm and is related to tolerability is the one most likely to change the estimand.

### Step 6: Choose a strategy for each event

**Input:** the event list and the clinical question.

**Output:** a strategy for each event, with a justification.

**Rationale:** The strategy determines what the estimand means. Different strategies answer different questions.

| Event | Possible strategies | Question implied |
|---|---|---|
| Discontinuation due to adverse event | Treatment policy | Effect of offering the drug, including tolerability |
| Discontinuation due to adverse event | Hypothetical | Effect if everyone tolerated the drug |
| Discontinuation due to adverse event | Principal stratum | Effect in patients who would tolerate the drug |
| Discontinuation due to adverse event | Composite | Effect on a composite of efficacy and tolerability |
| Rescue medication | Treatment policy | Effect of the strategy, including rescue use |
| Rescue medication | Hypothetical | Effect if rescue were unavailable |
| Rescue medication | While on treatment | Effect before rescue |
| Death | Composite | Effect on death or the non-fatal outcome |
| Death | While on treatment | Effect while alive |
| Death | Principal stratum | Not applicable; death cannot be avoided |
| Transplant | Treatment policy | Effect regardless of transplant |
| Transplant | Composite | Effect on transplant or death |

The choice should be consistent across events. A trial that uses treatment policy for discontinuation and hypothetical for rescue medication has a mixed estimand that is difficult to interpret.

A practical decision rule for the strategy is:

- If the claim is about the effect of offering the strategy, use treatment policy.
- If the claim is about the effect of receiving the treatment, use hypothetical.
- If the claim is about the full burden of the disease and its treatment, use a composite.
- If the claim is about the effect during treatment, use while on treatment.
- If the claim is about patients who can tolerate treatment, use principal stratum.

The rule is not mechanical. The strategy must also be estimable with the available data and defensible to the decision-maker.

### Step 7: Define the population-level summary

**Input:** the variable and the decision.

**Output:** a summary measure: risk difference, risk ratio, odds ratio, hazard ratio, mean difference, or another contrast.

**Rationale:** The summary measure determines how the effect is expressed. A risk difference is decision-relevant for public health; a risk ratio is decision-relevant for etiology; a hazard ratio is decision-relevant for time-to-event decisions.

The summary should be appropriate for the variable and the decision. A hazard ratio assumes proportional hazards; if the hazards are not proportional, a restricted mean survival time or a milestone risk may be more appropriate.

### Step 8: Write the estimand sentence

**Input:** the five attributes.

**Output:** a single sentence that states the population, treatment condition, variable, intercurrent-event strategy, and summary.

**Rationale:** The sentence is the test of clarity. If it cannot be written in one sentence, the estimand is not yet defined.

An example:

> In adults with heart failure and reduced ejection fraction, the risk ratio at 12 months for the composite of all-cause mortality or heart failure hospitalisation, comparing a strategy of drug X 10 mg daily plus standard care with placebo plus standard care, regardless of discontinuation of study treatment or use of rescue therapy.

The phrase "regardless of discontinuation or rescue" is the treatment-policy strategy. Removing it changes the estimand.

### Step 9: Align the estimator

**Input:** the estimand.

**Output:** an estimator that targets the estimand.

**Rationale:** The estimator must estimate the quantity the estimand defines. A treatment-policy estimand is often estimated with an intention-to-treat analysis; a hypothetical estimand requires a model for the counterfactual; a principal-stratum estimand requires a principal-stratum method.

| Estimand strategy | Candidate estimator | Key assumption |
|---|---|---|
| Treatment policy | ITT; complete follow-up | Outcomes observed regardless of ICE |
| Hypothetical | IPW; g-computation; multiple imputation | Model for the counterfactual |
| Composite | Analysis of the composite endpoint | Components are clinically meaningful |
| While on treatment | On-treatment window analysis | Window is prespecified |
| Principal stratum | Principal-stratum regression; instrumental variables | Principal ignorability or exclusion restriction |

The estimator should be prespecified in the SAP. A change of estimator after unblinding is an analysis change, not an estimand change.

### Step 10: Align the analysis population

**Input:** the estimand and the estimator.

**Output:** an analysis population that matches the estimand.

**Rationale:** The analysis population is the set of participants whose data are used. A treatment-policy estimand usually uses all randomised participants. A hypothetical estimand may use all randomised participants with a model for the counterfactual, or a subset with a different weighting.

The population should be defined before the analysis. A population defined after seeing the data is vulnerable to selection bias.

### Step 11: Separate intercurrent events from missing data

**Input:** the event list and the data.

**Output:** a plan for the ICEs and a separate plan for missing data.

**Rationale:** ICEs are handled by the estimand strategy; missing data is handled by the analysis. Confusing the two leads to an estimand that cannot be estimated.

For each ICE, state the strategy. For each missing outcome, state the assumed mechanism and the handling method. The two plans should be consistent: a treatment-policy estimand with 30 percent missing outcomes requires a missing-data model, not a claim that the outcomes are unnecessary.

### Step 12: Prespecify sensitivity analyses

**Input:** the estimand and the assumptions.

**Output:** sensitivity analyses that test the robustness of the result.

**Rationale:** Every estimand relies on assumptions. The sensitivity analyses show whether the conclusion survives plausible departures.

The sensitivity analyses should include:

- an alternative ICE strategy,
- an alternative missing-data assumption,
- an alternative summary measure,
- a tipping-point analysis for the ICE or missing data,
- a per-protocol analysis for comparison.

The results should be reported alongside the primary estimate, not hidden in a supplement.

### Step 13: Check alignment with the design and claim

**Input:** the estimand, the design, and the claim.

**Output:** an alignment statement.

**Rationale:** The design must be able to identify the estimand, and the claim must match the estimand. A design that cannot identify the estimand is not a valid design for the claim.

The alignment check asks:

- Does the design collect the outcome after the ICE?
- Does the design have a comparator that supports the contrast?
- Does the follow-up cover the variable's horizon?
- Does the claim use the same population and summary?
- Does the label or the commercial message describe the same strategy?

### Step 14: Document and report

**Input:** all of the above.

**Output:** an estimand section in the protocol and the SAP, and a reporting section in the results.

**Rationale:** The estimand must be auditable. A reader should be able to reconstruct the question and the analysis from the protocol.

The protocol should include the estimand sentence, the five attributes, the ICE strategies, the estimator, the analysis population, and the sensitivity analyses. The results should report the primary estimate, the sensitivity analyses, and the interpretation in terms of the estimand.

### Edge cases

**Death as an intercurrent event.** Death is terminal and cannot be handled by a hypothetical strategy. For a non-fatal outcome, death may be a competing event or a component of a composite. The choice changes the estimand and the analysis. A while-on-treatment strategy can estimate the effect while alive, but it does not answer the question of whether the treatment extends life.

**Rescue medication.** Rescue medication is an ICE when it is given for lack of efficacy. A treatment-policy strategy includes the effect of rescue; a hypothetical strategy estimates the effect if rescue were unavailable. The choice depends on whether the decision is about the strategy or the drug's intrinsic effect.

**Non-inferiority.** A non-inferiority claim requires an estimand that is sensitive to differences and robust to biases toward the null. Both ITT and per-protocol analyses are usually reported, because ITT can favor the new treatment when adherence is poor and per-protocol can favor it when adherence is related to prognosis.

The estimand for a non-inferiority trial should state the margin and the strategy. A treatment-policy estimand with a margin is a different claim from a per-protocol estimand with the same margin. The protocol should state both the estimand and the analysis populations, and the report should show that the conclusion is robust to the choice.

A useful check is whether the per-protocol and ITT estimates fall on the same side of the margin. If they do not, the claim is fragile and should be weakened.

**Composite endpoints.** A composite estimand counts the ICE as part of the outcome. The components should be clinically meaningful and prespecified. A composite that is driven by a mild component can produce a claim that patients do not value.

**Principal stratum.** A principal-stratum estimand targets patients who would tolerate treatment. It requires assumptions that cannot be fully tested. It is useful when the decision is about whether to offer treatment to patients likely to tolerate it, not about the effect of offering it to everyone.

**Multi-arm trials.** Each arm can have a different ICE profile. The estimand should be defined for each comparison, and the strategies should be consistent across arms.

**Pandemic or external shock.** An external event can change the care pathway and the ICE profile. The estimand should be robust to the change, or the analysis should be stratified by period. A treatment-policy estimand is often more robust than a hypothetical one in a changing environment.

### Attribute checks

| Attribute | Check | Common defect |
|---|---|---|
| Population | Is it defined by baseline characteristics? | Defined by post-randomisation events |
| Treatment condition | Are dose, duration, and care specified? | Comparator undefined |
| Variable | Is the measurement and timing specified? | Composite components unclear |
| ICEs | Is every event listed with a strategy? | Events omitted |
| Summary | Is the contrast appropriate for the decision? | Hazard ratio with non-proportional hazards |

A defect in any attribute makes the estimand ambiguous, even if the sentence reads well.

### Estimand quality checklist

- Is the population defined by baseline characteristics?
- Is the treatment condition complete, including dose, duration, and care?
- Is the variable measured at a decision-relevant time?
- Is every intercurrent event listed?
- Is the strategy for each event stated and justified?
- Is the summary measure appropriate for the variable?
- Is the estimator aligned with the estimand?
- Is the analysis population aligned with the estimand?
- Are missing data handled separately?
- Are sensitivity analyses prespecified?
- Does the label claim match the estimand?
- Is the estimand reproducible from the protocol alone?

A negative answer to any question is a defect to fix before the protocol is locked.

### Decision table: strategy by decision

| Decision | Relevant estimand | Strategy | Why |
|---|---|---|---|
| Regulatory approval of a drug | Effect of the drug under its labelled use | Treatment policy | Reflects real use |
| Clinician deciding whether to prescribe | Effect if the patient adheres | Hypothetical | Reflects biological efficacy |
| Payer deciding coverage | Effect on the full burden | Composite | Includes tolerability and rescue |
| Patient deciding whether to start | Effect in patients like me who tolerate | Principal stratum | Personalised |
| Label describing the effect | Must match the trial estimand | Same as trial | Consistency |
| Post-marketing safety | Effect in routine use | Treatment policy | Reflects real-world use |

The table is a starting point. The choice should be justified in the protocol, not derived mechanically from the decision type.

## 6. Worked example

### The setting

A biotech is developing a drug for heart failure with reduced ejection fraction. The Phase 3 trial randomises 4,000 patients to drug X 10 mg daily or placebo, both added to standard care. The primary outcome is the composite of all-cause mortality or heart failure hospitalisation at 12 months. About 20 percent of patients in the drug arm discontinue due to adverse events, compared with 8 percent in the placebo arm. About 15 percent of patients in both arms receive rescue therapy.

The commercial team wants a label that says the drug "reduces the risk of death or hospitalisation." The clinical team wants to know whether the drug works in patients who tolerate it. The payer wants to know the effect on the full burden, including discontinuation.

### Step 1: Clinical question

> In adults with heart failure and reduced ejection fraction, should we offer drug X rather than placebo to reduce death or hospitalisation over 12 months, and does the answer differ for patients who tolerate the drug?

The question contains two sub-questions: the effect of offering the drug (treatment policy) and the effect in tolerators (principal stratum). The primary estimand must answer the first; the second is a supportive analysis.

### Step 2: Population

Adults aged 18 or older with symptomatic heart failure and ejection fraction of 40 percent or less, on stable standard care for at least four weeks, with no contraindication to the drug. The population is the labelled population.

### Step 3: Treatment condition

Drug X 10 mg once daily versus matching placebo, added to standard care, with a titration period of two weeks. Dose reduction to 5 mg is allowed for tolerability. Discontinuation is allowed at the investigator's discretion. Rescue therapy is allowed for worsening heart failure.

### Step 4: Variable

The variable is the composite of all-cause mortality or heart failure hospitalisation at 12 months. The components are adjudicated by a blinded committee. Death is a component, not a competing event.

### Step 5: Intercurrent events

The events are:

- discontinuation of study treatment due to an adverse event,
- dose reduction,
- rescue therapy for worsening heart failure,
- heart transplant,
- death.

Death is a component of the composite, so it is not an ICE for the primary estimand. Transplant is an ICE because it removes the patient from the risk of the composite. Rescue therapy is an ICE because it changes the treatment condition. Discontinuation is an ICE because it changes the treatment received.

### Step 6: Strategies

| Event | Strategy | Justification |
|---|---|---|
| Discontinuation | Treatment policy | The label describes the effect of offering the drug, including tolerability |
| Dose reduction | Treatment policy | Dose reduction is part of the labelled strategy |
| Rescue therapy | Treatment policy | The label describes the strategy, including rescue use |
| Transplant | Treatment policy | Transplant is a clinical event, not a reason to exclude |
| Death | Composite | Death is a component of the primary composite |

The estimand is a treatment-policy estimand for all non-terminal events. This is consistent with a label that describes the effect of the strategy.

### Step 7: Summary

The summary is the risk ratio at 12 months for the composite, with the risk difference as a secondary summary. The risk ratio is chosen for consistency with the label; the risk difference is chosen for the payer.

### Step 8: Estimand sentence

> In adults with heart failure and reduced ejection fraction, the risk ratio at 12 months for the composite of all-cause mortality or heart failure hospitalisation, comparing a strategy of drug X 10 mg daily plus standard care with placebo plus standard care, regardless of discontinuation, dose reduction, rescue therapy, or transplant.

The sentence states all five attributes. The phrase "regardless of" is the treatment-policy strategy.

### Step 9: Estimator

The primary estimator is an intention-to-treat analysis of the composite, with a Cox proportional hazards model or a log-binomial model for the 12-month risk, adjusted for stratification factors. Participants who discontinue remain in their assigned group. Participants who receive rescue therapy remain in their assigned group. Participants who receive a transplant are followed for the composite until the end of the study.

The analysis population is all randomised participants, analysed as randomised.

### Step 10: Analysis population

The ITT population is all randomised participants with at least one dose of study treatment? No: the estimand is the effect of assignment, so the population is all randomised participants, regardless of whether they received treatment. The protocol defines the ITT population as all randomised participants.

A per-protocol population is defined for a sensitivity analysis: participants who took at least 80 percent of the assigned dose and did not take prohibited medication. The per-protocol analysis is not the primary analysis.

### Step 11: ICEs versus missing data

The ICEs are handled by the treatment-policy strategy. Missing composite outcomes are handled by a separate missing-data plan: multiple imputation under a missing-at-random assumption, with a tipping-point analysis. A participant who withdraws consent is missing data, not an ICE.

### Step 12: Sensitivity analyses

| Analysis | Purpose | Expected direction |
|---|---|---|
| Per-protocol | Estimate the effect in adherent participants | Larger effect if non-adherence dilutes |
| Hypothetical (if all adhered) | Estimate the effect of the drug if taken | Larger effect |
| Composite including discontinuation | Test the effect of tolerability | Smaller effect or harm |
| Principal stratum (tolerators) | Estimate the effect in patients who tolerate | Similar to hypothetical |
| Tipping point for missing data | Test the missing-data assumption | Range |

### Step 13: Alignment

The design collects the composite after discontinuation, rescue, and transplant, so the treatment-policy estimand is identifiable. The follow-up covers 12 months. The label claim uses the same population and the same composite. The claim "reduces the risk of death or hospitalisation" matches the treatment-policy estimand.

The clinical team's question about tolerators is addressed by a principal-stratum sensitivity analysis, not by the primary estimand. The payer's question about the full burden is addressed by the composite-including-discontinuation analysis.

### Step 14: Reporting

The primary result is reported as a risk ratio with a confidence interval, with the risk difference and the number needed to treat. The sensitivity analyses are reported in a table. The interpretation states that the estimate is the effect of offering the strategy, not the effect of taking the drug in patients who tolerate it.

### The arithmetic of the different estimands

The trial data can be analysed under each strategy. The numbers below are from the simulation in Section 7, calibrated to the example's discontinuation rates.

| Estimand | Strategy | Risk ratio | Interpretation |
|---|---|---:|---|
| Hypothetical | If all adhered | 0.71 | The drug's effect if taken |
| Treatment policy | Regardless of discontinuation | 0.77 | The effect of offering the drug |
| Composite including discontinuation | Discontinuation counts as failure | 1.25 | The effect on the full burden |
| Principal stratum | Tolerators only | 0.70 | The effect in patients who tolerate |

The table is the central lesson. The same data produce a 29 percent relative reduction under the hypothetical estimand, a 23 percent reduction under the treatment-policy estimand, a 25 percent increase under the composite estimand, and a 30 percent reduction in the principal stratum. The choice of estimand changes the direction of the conclusion for the composite.

The composite result is not a paradox. It reflects the fact that discontinuation is more common in the drug arm, and a composite that counts discontinuation as failure measures the burden of tolerability as well as efficacy. A payer who cares about the full burden may find the composite relevant; a clinician who wants to know whether the drug works may find it misleading.

### What each stakeholder should hear

| Stakeholder | Question | Estimand | What to say |
|---|---|---|---|
| Regulator | Does the labelled use provide benefit? | Treatment policy | The effect of offering the strategy |
| Clinician | Does the drug work if taken? | Hypothetical | The effect under adherence |
| Patient | Will it work for someone like me who tolerates it? | Principal stratum | The effect in tolerators |
| Payer | What is the full burden? | Composite | The effect including discontinuation |
| Commercial team | What can we claim? | Must match the label | Do not exceed the estimand |

The table is a communication tool. It prevents the same number from being presented as if it answered every stakeholder's question.

### Why the composite reverses

The composite reverses because discontinuation is an event that the drug causes more often than the comparator. In the example, 20 percent of the drug arm discontinue versus 8 percent of the placebo arm. If the composite counts discontinuation as failure, the drug arm has an extra 12 percentage points of "failure" that is unrelated to the efficacy of the drug.

The reversal is a feature, not a bug, if the decision is about the full burden. It is a bug if the composite is presented as an efficacy result. The protocol should state which interpretation is intended, and the report should present the components separately.

### Extended example: a diagnostic AI study

The estimand framework also applies to diagnostic and predictive studies, even though the intercurrent-event attribute is often absent.

A company has developed an AI tool that flags suspected pulmonary embolism on computed tomography pulmonary angiography (CTPA). The claim is that the tool "improves detection of pulmonary embolism." The five attributes are:

| Attribute | Specification |
|---|---|
| Population | Adults who undergo CTPA in the emergency department for suspected pulmonary embolism |
| Treatment condition | AI-assisted reading versus unaided reading, with the same reference standard |
| Variable | Sensitivity and specificity for pulmonary embolism at the patient level, against a radiologist adjudication |
| Intercurrent events | None for the diagnostic accuracy estimand; a subsequent clinical action is a separate question |
| Summary | Difference in sensitivity and specificity, with confidence intervals |

The estimand is a diagnostic accuracy estimand, not a treatment effect. The claim "improves detection" is ambiguous: it could mean higher sensitivity, higher specificity, or better clinical outcomes. The protocol must state which.

If the claim is about clinical outcomes, the estimand changes: the population is the same, but the variable becomes a patient outcome such as pulmonary embolism diagnosed within 24 hours or an adverse event, and the intercurrent events include the clinician's decision to act on the AI output. The design then requires a randomised comparison, not an accuracy study.

The example shows that the estimand is a general tool for aligning a claim with a study. The attributes change with the claim, but the discipline is the same.

### Two diagnostics that reveal a flawed design

**Diagnostic 1: the discontinuation rate by arm.** If discontinuation is 20 percent in the drug arm and 8 percent in the placebo arm, the treatment-policy estimand will be diluted relative to the hypothetical estimand. The diagnostic is to report the discontinuation rate by arm and the reason. A trial that does not report the rate cannot interpret the difference between the estimands.

**Diagnostic 2: the estimand-estimator alignment.** If the protocol states a hypothetical estimand but the analysis is intention-to-treat, the estimate does not answer the question. The diagnostic is to compare the estimand sentence with the analysis population and the estimator. A mismatch is a design flaw, not a reporting issue.

### Vary one assumption: the discontinuation rate

If discontinuation is 30 percent in the drug arm rather than 20 percent, the treatment-policy risk ratio rises from 0.77 to 0.79, and the composite risk ratio rises from 1.25 to 1.61. The hypothetical and principal-stratum estimates are unchanged, because they condition on adherence or tolerance. The difference between the estimands grows with the discontinuation rate.

### Vary one assumption: the effect size

If the true effect is a risk ratio of 0.85 rather than 0.70, the hypothetical estimate is 0.84, the treatment-policy estimate is 0.88, and the composite estimate is 1.33. The treatment-policy estimate is close to the null, and the composite suggests harm. The choice of estimand is now decisive for the claim.

### Vary one assumption: the summary measure

If the summary is a risk difference rather than a risk ratio, the treatment-policy estimate is a 4.1 percentage point reduction, and the composite estimate is a 3.2 percentage point increase. The direction is the same as the risk ratio, but the magnitude is easier to interpret for a payer. The choice of summary should match the decision.

### What the label should say

A label that matches the treatment-policy estimand should say:

- "In adults with heart failure and reduced ejection fraction..."
- "...drug X reduced the risk of death or hospitalisation..."
- "...when added to standard care..."
- "...regardless of whether patients continued treatment."

A label that says "drug X reduces death or hospitalisation" without the population and the strategy is broader than the evidence. A label that says "drug X reduces death or hospitalisation in patients who tolerate treatment" describes the principal-stratum analysis, not the primary estimand. The wording is part of the evidence, not a marketing afterthought.

### The decision

The company submits the treatment-policy estimand as the primary analysis, with the hypothetical and principal-stratum analyses as supportive. The label describes the effect of offering the drug, including discontinuation. The payer receives the composite-including-discontinuation analysis as a secondary analysis, with a clear explanation of what it measures.

The clinical team's question is answered by the principal-stratum analysis, which shows a similar effect to the hypothetical analysis. The team does not claim that the drug works in all patients; it claims that it works in patients who tolerate it, and that offering it to all patients produces a smaller average benefit.

The example shows the central lesson: the estimand is a choice, and the choice must match the claim. A trial that reports one number without stating the estimand invites the reader to assume the wrong question.

## 7. Code, query, or calculation

The code below simulates a two-arm trial with a binary outcome and an intercurrent event (discontinuation). It computes the estimate under four estimands: hypothetical, treatment policy, composite, and principal stratum. It uses only the Python standard library. The data are simulated for teaching.

```python
import random


def simulate_estimands(n=20000, seed=7, true_rr=0.70,
                       discontinue_treated=0.20, discontinue_control=0.08):
    """Simulate potential outcomes and compute four estimand estimates."""
    random.seed(seed)
    hyp = [0, 0, 0]       # n, events treated, events control (if all adhered)
    policy = [0, 0, 0]    # treatment policy
    comp = [0, 0, 0]      # composite including discontinuation
    princ = [0, 0, 0]     # principal stratum of tolerators

    for _ in range(n):
        baseline_risk = random.uniform(0.05, 0.30)
        y0 = 1 if random.random() < baseline_risk else 0
        y1 = 1 if random.random() < baseline_risk * true_rr else 0
        d1 = random.random() < discontinue_treated
        d0 = random.random() < discontinue_control

        hyp[0] += 1
        hyp[1] += y1
        hyp[2] += y0

        y_policy = y1 if not d1 else y0
        policy[0] += 1
        policy[1] += y_policy
        policy[2] += y0

        comp[0] += 1
        comp[1] += 1 if (d1 or y1) else 0
        comp[2] += 1 if (d0 or y0) else 0

        if not d1 and not d0:
            princ[0] += 1
            princ[1] += y1
            princ[2] += y0

    def risk_ratio(events_t, n_t, events_c, n_c):
        return (events_t / n_t) / (events_c / n_c)

    return {
        "hypothetical": round(risk_ratio(hyp[1], hyp[0], hyp[2], hyp[0]), 3),
        "treatment_policy": round(risk_ratio(policy[1], policy[0],
                                             policy[2], policy[0]), 3),
        "composite": round(risk_ratio(comp[1], comp[0], comp[2], comp[0]), 3),
        "principal_stratum": round(risk_ratio(princ[1], princ[0],
                                              princ[2], princ[0]), 3),
        "n_principal": princ[0],
        "n_total": n,
    }


if __name__ == "__main__":
    print("base case:", simulate_estimands())
    print("higher discontinuation:", simulate_estimands(discontinue_treated=0.30))
    print("smaller true effect:", simulate_estimands(true_rr=0.85))
```

Expected output:

```text
base case: {'hypothetical': 0.706, 'treatment_policy': 0.767, 'composite': 1.245, 'principal_stratum': 0.703, 'n_principal': 14781, 'n_total': 20000}
higher discontinuation: {'hypothetical': 0.706, 'treatment_policy': 0.79, 'composite': 1.607, 'principal_stratum': 0.697, 'n_principal': 12920, 'n_total': 20000}
smaller true effect: {'hypothetical': 0.843, 'treatment_policy': 0.877, 'composite': 1.326, 'principal_stratum': 0.842, 'n_principal': 14781, 'n_total': 20000}
```

The output shows four things.

- The hypothetical estimate (0.706) is close to the true effect (0.70), because it conditions on adherence.
- The treatment-policy estimate (0.767) is diluted by discontinuation, because patients who stop treatment do not receive the benefit.
- The composite estimate (1.245) is above 1, because discontinuation is more common in the treated arm and is counted as a failure. The composite measures a different quantity: the burden of efficacy plus tolerability.
- The principal-stratum estimate (0.703) is close to the hypothetical, but it applies to 14,781 of 20,000 patients (74 percent). It answers a narrower question.

The higher-discontinuation scenario shows the treatment-policy estimate moving toward the null (0.790) and the composite moving further from 1 (1.607). The smaller-effect scenario shows the treatment-policy estimate (0.877) close to the null and the composite at 1.326. The choice of estimand is decisive when the effect is modest and discontinuation is common.

The simulation assumes that discontinuation removes the benefit entirely. A real analysis would model carryover, partial adherence, and the reason for discontinuation. The teaching point is the direction and the magnitude of the divergence, not the specific numbers.

### A sensitivity analysis for carryover

The treatment-policy estimate depends on what happens after discontinuation. If the drug's benefit persists after stopping, the treatment-policy effect is closer to the hypothetical effect. The code below varies the carryover assumption.

```python
import random


def policy_estimate(carryover=0.0, n=20000, seed=7, true_rr=0.70,
                    discontinue_treated=0.20):
    """Treatment-policy risk ratio under a carryover assumption."""
    random.seed(seed)
    events_t = events_c = 0
    for _ in range(n):
        baseline = random.uniform(0.05, 0.30)
        y0 = 1 if random.random() < baseline else 0
        y1 = 1 if random.random() < baseline * true_rr else 0
        d1 = random.random() < discontinue_treated
        if d1:
            y = y1 if random.random() < carryover else y0
        else:
            y = y1
        events_t += y
        events_c += y0
    return round((events_t / n) / (events_c / n), 3)


if __name__ == "__main__":
    for carry in (0.0, 0.5, 1.0):
        print(f"carryover={carry}: treatment-policy RR="
              f"{policy_estimate(carry)}")
```

Expected output:

```text
carryover=0.0: treatment-policy RR=0.763
carryover=0.5: treatment-policy RR=0.732
carryover=1.0: treatment-policy RR=0.693
```

The output shows that the treatment-policy estimate ranges from 0.693 to 0.763 depending on the carryover assumption. The range is wider than the difference between the hypothetical and treatment-policy estimates under the no-carryover assumption. The carryover assumption is therefore a material part of the estimand, not a technical detail.

A protocol that uses a treatment-policy estimand should state the assumed carryover or, better, report a sensitivity analysis over plausible values. The assumption should be based on the drug's pharmacokinetics and mechanism, not on convenience.

### Reporting checklist

- The estimand sentence, with all five attributes.
- The ICE list and the strategy for each.
- The estimator and the analysis population.
- The missing-data plan, separate from the ICE plan.
- The primary estimate with a confidence interval.
- The sensitivity analyses, including an alternative strategy.
- The interpretation in terms of the estimand.
- The label claim, compared with the estimand.
- The limitations, including untestable assumptions.
- The components of any composite endpoint.

### A calculation of the composite risk

The composite risk in the treated arm is the probability of discontinuation or the outcome, whichever comes first. If the discontinuation rate is 20 percent and the outcome risk among those who do not discontinue is 10 percent, the composite risk is:

```text
P(composite) = P(discontinue) + P(no discontinue) * P(outcome | no discontinue)
             = 0.20 + 0.80 * 0.10
             = 0.28
```

In the control arm, with an 8 percent discontinuation rate and a 14 percent outcome risk:

```text
P(composite) = 0.08 + 0.92 * 0.14
             = 0.209
```

The composite risk ratio is 0.28 / 0.209 = 1.34, which is above 1 even though the drug reduces the outcome risk. The arithmetic shows why a composite that includes discontinuation can reverse the direction of the conclusion.

## 8. Data, standards, and regulatory context

### Binding requirements

**ICH E9(R1).** The addendum is the primary standard. It was adopted at Step 4 on 20 November 2019. The EMA Step 5 version took effect on 30 July 2020, and the FDA issued it as final guidance on 12 May 2021. It requires the estimand to be defined before the design and analysis, and it requires the intercurrent-event strategy to be specified.

The addendum is a guideline, not a statute. In practice, it is treated as the standard of review in the three major jurisdictions. A confirmatory trial that does not state its estimand is difficult to defend.

**ICH E9.** The parent guideline (1998) establishes the statistical principles, including the intention-to-treat principle and the analysis population. The addendum supplements it; it does not replace it.

**EU Clinical Trials Regulation.** Regulation (EU) No 536/2014 requires the protocol to include the statistical analysis plan, including the estimand. It is binding.

**US regulations.** The FDA's regulations on adequate and well-controlled studies (21 CFR 314.126) require a clear statement of the objectives and a valid comparison. The estimand is the operational form of the objective. The regulations are binding.

### Formal guidance

**FDA estimands guidance.** The FDA issued the ICH E9(R1) addendum as final guidance in May 2021. The guidance states that the estimand should be defined before the design, and that the intercurrent-event strategy should be justified. It does not prescribe a particular strategy.

**FDA non-inferiority guidance.** The November 2016 guidance describes the analysis populations for non-inferiority trials, including the use of both ITT and per-protocol analyses. The choice of estimand interacts with the choice of population.

**EMA guideline on missing data.** The EMA guideline on missing data in confirmatory trials (2010) addresses the handling of missing data separately from intercurrent events. It is guidance, not regulation.

**ICH M11.** The Clinical electronic Structured Harmonised Protocol (CeSHarP) reached Step 4 on 19 November 2025. Its template includes a section on objectives and estimands, which makes the estimand a required protocol element in the structured format.

### What the standards constrain and what they leave open

| Instrument | Constrains | Leaves open |
|---|---|---|
| ICH E9(R1) | The five attributes and the ICE strategies | The choice of strategy |
| ICH E9 | The statistical principles and populations | The estimand |
| FDA estimands guidance | The definition before analysis | The strategy and the summary |
| EU CTR | The protocol content | The estimand |
| ICH M11 | The structured protocol format | The clinical question |

The standards require the estimand to be explicit and justified. They do not require a particular strategy. The choice depends on the question, the decision, and the claim.

### Questions a regulatory reviewer will ask

- What is the estimand, stated in one sentence?
- What are the five attributes?
- Which intercurrent events were identified, and why?
- What strategy was chosen for each event, and why?
- Is the strategy consistent with the proposed label?
- Does the analysis population match the estimand?
- Does the estimator target the estimand?
- Are the sensitivity analyses adequate?
- How are missing data handled separately from ICEs?
- Does the label describe the same population, treatment, outcome, and summary?

A protocol that answers these questions is reviewable. A protocol that answers only the first is not.

## 9. Industry practice and case evidence

### The E9(R1) addendum in practice

The addendum was developed because regulators and sponsors disagreed about how to handle discontinuation and rescue medication. Before the addendum, a protocol might say "the primary analysis is intention-to-treat" without defining what the intention-to-treat estimate meant when patients discontinued. The addendum requires the sponsor to state the strategy and to justify it.

The documented fact is the adoption of the addendum and its effective dates. The inference is that the addendum has shifted the review conversation from "which analysis" to "which question." Sponsors now need to explain the estimand in the protocol, not only in the statistical analysis plan.

**Lesson.** The estimand is a regulatory document, not only a statistical concept. It must be written for the reviewer as well as for the analyst.

### Oncology trials and the while-on-treatment strategy

Oncology trials often use progression-free survival (PFS) as a primary endpoint. PFS is a composite of progression and death, and it is sensitive to the timing of assessment. The estimand for PFS typically uses a composite strategy for death and a while-on-treatment strategy for subsequent therapy.

The documented fact is the widespread use of PFS and the methodological debate about its interpretation. The inference is that the estimand for PFS must state how subsequent therapy and assessment timing are handled, because these choices can change the estimated effect.

**Lesson.** A composite endpoint is not self-defining. The estimand must state the components and the handling of subsequent events.

### COVID-19 trials and treatment policy

The RECOVERY trial used all-cause mortality as the primary outcome and a treatment-policy approach to assignment. The trial randomised 2,104 participants to dexamethasone and 4,321 to usual care. Overall 28-day mortality was 22.9 percent versus 25.7 percent (age-adjusted rate ratio 0.83, 95 percent confidence interval 0.75 to 0.93). Among participants receiving invasive mechanical ventilation, mortality was 29.3 percent versus 41.4 percent (rate ratio 0.64, 95 percent confidence interval 0.51 to 0.81).

The documented fact is the effect and the population. The inference is that the treatment-policy estimand was appropriate for a pandemic in which the decision was whether to offer the drug, not whether it worked in patients who adhered perfectly. The subgroup analysis was prespecified and biologically plausible.

**Lesson.** The estimand should match the decision. In a public-health emergency, the effect of offering the strategy is often more relevant than the effect under perfect adherence.

### The Apple Heart Study and the detection estimand

The Apple Heart Study enrolled 419,297 participants and estimated the concordance of an irregular pulse notification with atrial fibrillation on a simultaneous ECG patch. The estimand was a diagnostic concordance, not a treatment effect. The study did not define an intercurrent-event strategy because there was no treatment assignment.

The documented fact is the design and the confirmation rate. The inference is that the estimand framework applies to diagnostic studies as well: the population, the condition, the variable, and the summary must be defined. The intercurrent-event attribute is not relevant when there is no treatment.

**Lesson.** The five attributes are a general framework. Not every attribute is relevant to every study, but the framework forces the team to state which are.

### What companies actually do

The following practices are inferred from public protocols, guidance, and published trials.

1. **Estimands are written in the protocol.** Mature teams include the estimand sentence and the five attributes in the protocol, not only in the SAP.
2. **ICE strategies are justified.** The protocol explains why treatment policy, hypothetical, or composite was chosen.
3. **Sensitivity analyses are prespecified.** The primary estimand is accompanied by analyses under alternative strategies.
4. **Labels match estimands.** Regulatory affairs reviews the label claim against the estimand to avoid overstatement.
5. **Payer dossiers use the estimand.** Health economics teams map the estimand to the model inputs.

### Where practice falls short

- **Estimand omitted.** The protocol describes the endpoint and the analysis but not the quantity being estimated.
- **Strategy inconsistent.** Treatment policy for discontinuation and hypothetical for rescue, with no justification.
- **Composite used to hide tolerability.** A composite that counts discontinuation as failure is presented as an efficacy result.
- **Principal stratum overclaimed.** A principal-stratum analysis is presented as the overall effect.
- **Label mismatch.** The label describes a hypothetical effect while the trial estimated a treatment-policy effect.

### Case comparison

| Case | Estimand issue | Consequence | Lesson |
|---|---|---|---|
| Heart failure drug | Discontinuation and rescue | Treatment policy dilutes the effect | Match the strategy to the label |
| Oncology PFS | Subsequent therapy and assessment | Composite and while-on-treatment choices | Report the components |
| RECOVERY | Treatment policy for mortality | Effect of offering the strategy | Match the estimand to the decision |
| Diagnostic AI | No treatment, accuracy estimand | Claim ambiguity | The framework applies to diagnostics |
| Apple Heart | Detection concordance | No clinical outcome claim | State what the study estimates |

The table is a summary of the cases discussed in this section. It is not a systematic review, and the inferences are labelled as such in the text.

### Common disagreements about estimands

- "The trial is negative" may mean "the treatment-policy effect is null," not "the drug does not work."
- "The trial is positive" may mean "the hypothetical effect is large," not "offering the drug helps everyone."
- "The composite is significant" may mean "the composite is driven by a mild component."
- "The subgroup is positive" may mean "the principal stratum is different from the overall population."
- "The label says it reduces hospitalisation" may mean "the trial estimated a different quantity."
- "The payer will not cover it" may mean "the estimand does not answer the payer's question."

The disagreements are not about statistics. They are about which question the evidence answers.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Estimand omitted | Protocol describes only the endpoint | No explicit question | Ambiguous result | Write the five attributes |
| Strategy inconsistent | Different strategies for different ICEs | No unifying question | Uninterpretable estimate | One strategy per ICE, justified |
| Treatment policy confused with ITT | Protocol says "ITT" but does not define the estimand | Analysis principle treated as estimand | Misinterpretation | State both |
| Hypothetical estimated by ITT | Protocol claims a hypothetical effect | Estimator does not match estimand | Biased estimate | Use a counterfactual method |
| Composite hides harm | Discontinuation counted as failure | Composite dominated by tolerability | Reversed conclusion | Report components |
| Principal stratum overclaimed | Subgroup presented as overall | Latent population ignored | Overstatement | Label as supportive |
| Death treated as missing | Death excluded from the analysis | Terminal event mishandled | Biased estimate | Composite or competing risk |
| Rescue ignored | Rescue therapy not recorded | ICE not identified | Ambiguous treatment condition | Record and specify |
| Label mismatch | Label uses a different strategy | Regulatory and statistical misalignment | False claim | Estimand-label review |
| Sensitivity absent | Only one analysis reported | No test of assumptions | Unknown robustness | Prespecify alternatives |
| Analysis population changed | Population defined after unblinding | Selection | Bias | Prespecify |
| Estimand not reproducible | Reader cannot reconstruct the question | Ambiguous sentence | Unreviewable protocol | One-page summary |
| Untestable assumptions hidden | No sensitivity analysis | Assumptions not stated | Overconfident claim | State and test |
| Stakeholder mismatch | Different audiences use different estimands | No communication plan | Misuse of the result | Stakeholder table |

### Red flags in an estimand review

- The protocol does not contain the word "estimand."
- The estimand is described only as an endpoint.
- The intercurrent events are not listed.
- The strategy is not stated for each event.
- The analysis population does not match the strategy.
- The sensitivity analyses do not include an alternative strategy.
- The label claim is broader than the estimand.
- The composite components are not reported separately.
- The principal-stratum analysis is presented as the main result.
- Death is treated as missing data.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The clinical question is well defined | The estimand is a translation | Ambiguous question | Decision review |
| The population matches the claim | The estimate applies to the claim | Population drift | Eligibility review |
| The treatment condition is stable | The contrast is defined | Dose changes; rescue | Protocol adherence |
| The variable measures the outcome | The estimate is meaningful | Surrogate; measurement error | Validation |
| The ICE list is complete | The estimand is complete | Unlisted events | Clinical review |
| The strategy matches the decision | The estimate is decision-relevant | Wrong strategy | Stakeholder review |
| The estimator targets the estimand | The estimate is valid | Estimator mismatch | Simulation; methods review |
| The analysis population matches | The estimate applies to the population | Population change | SAP review |
| Missing data are handled separately | The estimate is not biased | Confusion with ICE | Missing-data plan |
| The sensitivity analyses are adequate | Robustness is tested | Omitted alternatives | Prespecification |
| The label matches the estimand | The claim is accurate | Regulatory misalignment | Label review |
| The report states the strategy | Readers can interpret the estimate | Implicit strategy | Reporting checklist |
| The components are reported | Composite interpretation is possible | Hidden driver | Component table |
| The stakeholder questions are mapped | The evidence is used correctly | Miscommunication | Stakeholder table |

### Limits of this topic

This topic covers the estimand framework and its application to common trial designs. It does not cover the full theory of causal inference, the details of principal-stratum methods, or the analysis of complex longitudinal data. Those are specialised topics.

The worked example and code use simulated data. The numbers illustrate how the choice of estimand changes the estimate; they are not results from a real trial. The regulatory statements are educational summaries, not legal advice.

## 12. Exercises

### Exercise 1 — Write an estimand (application, 30 minutes)

**Deliverable:** a five-attribute estimand for a trial of a new insulin in type 2 diabetes, with a primary outcome of HbA1c change at 6 months.

**Acceptance criteria:** all five attributes present; the ICE strategy stated; the summary measure appropriate; the sentence clear.

**Time:** 30 minutes.

### Exercise 2 — Identify intercurrent events (application, 25 minutes)

**Deliverable:** a list of intercurrent events for a trial of a biologic in rheumatoid arthritis, distinguishing them from missing data.

**Acceptance criteria:** at least five events; each classified correctly; the missing-data items separated.

**Time:** 25 minutes.

### Exercise 3 — Choose a strategy (design, 35 minutes)

**Deliverable:** a strategy for each event in Exercise 2, with a justification from the decision context.

**Acceptance criteria:** strategy chosen from the five; justification linked to the claim; consistency across events.

**Time:** 35 minutes.

### Exercise 4 — Align estimator and population (application, 30 minutes)

**Deliverable:** an estimator and analysis population for the estimand in Exercise 1, with the assumptions stated.

**Acceptance criteria:** estimator targets the estimand; population matches; assumptions explicit.

**Time:** 30 minutes.

### Exercise 5 — Compare estimands (calculation, 40 minutes)

**Deliverable:** using the simulation in Section 7, compute the four estimates for a trial with a 25 percent discontinuation rate and a true risk ratio of 0.75, and explain the differences.

**Acceptance criteria:** code run; results reported; differences explained; the composite result interpreted.

**Time:** 40 minutes.

### Exercise 6 — Sensitivity analyses (design, 30 minutes)

**Deliverable:** a sensitivity analysis plan for the estimand in Exercise 1, including an alternative strategy and a tipping-point analysis.

**Acceptance criteria:** at least three analyses; each tied to an assumption; reporting plan stated.

**Time:** 30 minutes.

### Exercise 7 — Critique a published estimand (critique, 40 minutes)

**Deliverable:** a critique of the estimand in a published trial, identifying the strategy, the analysis population, and any mismatch.

**Acceptance criteria:** specific to the trial; mismatch identified; alternative proposed.

**Time:** 40 minutes.

### Exercise 8 — Label alignment (design, 30 minutes)

**Deliverable:** a one-page review of a label claim against the trial estimand, with recommended wording if they mismatch.

**Acceptance criteria:** claim and estimand compared; mismatch identified; wording specific and defensible.

**Time:** 30 minutes.

## 13. Self-check answers

**Answer 1 (estimand).** "In adults with type 2 diabetes inadequately controlled on metformin, the mean difference in HbA1c change at 6 months between a strategy of adding insulin X and a strategy of adding placebo, regardless of discontinuation or rescue therapy." The population is adults on metformin; the treatment conditions are insulin X versus placebo; the variable is HbA1c change at 6 months; the ICE strategy is treatment policy for discontinuation and rescue; the summary is the mean difference.

**Answer 2 (intercurrent events).** Events: discontinuation due to adverse events, dose reduction, rescue with a conventional synthetic disease-modifying antirheumatic drug, switching to another biologic, surgery, and death. Missing data: missed visits, withdrawal of consent, and loss to follow-up. Death may be an ICE or a competing event depending on the outcome.

**Answer 3 (strategy).** For a label describing the effect of offering the biologic, treatment policy is appropriate for discontinuation, dose reduction, rescue, and switching. Death is a composite or a competing event. For a clinical question about efficacy in patients who tolerate the drug, a hypothetical or principal-stratum strategy is appropriate. The choice should be consistent with the claim.

**Answer 4 (estimator and population).** A treatment-policy estimand is estimated with an ITT analysis of all randomised participants, using a mixed model for repeated measures or a multiple-imputation approach for missing HbA1c. The assumptions are that missing data are missing at random and that the model is correctly specified.

**Answer 5 (compare estimands).** With a 25 percent discontinuation rate and a true risk ratio of 0.75, the hypothetical estimate is about 0.75, the treatment-policy estimate is about 0.81, the composite estimate is above 1, and the principal-stratum estimate is about 0.75 in the tolerator subgroup. The differences arise because discontinuation removes the benefit and because the composite counts discontinuation as failure.

**Answer 6 (sensitivity).** The plan should include a per-protocol analysis, a hypothetical analysis with a counterfactual model, a composite analysis, and a tipping-point analysis for missing data. Each addresses a different assumption.

**Answer 7 (critique rubric).** A strong critique identifies the estimand sentence, the ICE strategy, the analysis population, and the estimator, and shows whether they align. A weak critique summarises the results without identifying the estimand.

**Answer 8 (label rubric).** A strong review compares the label claim's population, treatment condition, outcome, and strategy with the trial estimand, and proposes wording that matches. A weak review says the label is "consistent" without the comparison.

## 14. Interview questions

**Question 1. What is an estimand?**

A strong answer: the precise quantity to be estimated, defined by population, treatment condition, variable, intercurrent-event strategy, and summary. A weak answer says it is "the endpoint" or "what you want to measure."

**Question 2. Why does the estimand come before the design?**

A strong answer: because the design must identify the estimand, and different estimands require different designs and analyses. A weak answer says it is "good practice."

**Question 3. What are the five intercurrent-event strategies?**

A strong answer: treatment policy, hypothetical, composite, while on treatment, and principal stratum. A weak answer names fewer than five.

**Question 4. What is the difference between treatment policy and intention-to-treat?**

A strong answer: treatment policy is an estimand strategy; ITT is an analysis principle. They usually align but are not identical. A weak answer treats them as synonyms.

**Question 5. How do you handle discontinuation in a trial?**

A strong answer: identify it as an ICE, choose a strategy that matches the claim, prespecify the estimator and population, and report sensitivity analyses under alternative strategies. A weak answer says "use ITT."

**Question 6. When would you use a principal-stratum estimand?**

A strong answer: when the decision is about patients who would tolerate or respond to treatment, and the effect in that latent subgroup is the relevant question; it requires untestable assumptions. A weak answer says "for subgroups."

**Question 7. What is a composite estimand, and what is its risk?**

A strong answer: an estimand that counts the ICE as part of the outcome; the risk is that a mild component can drive the result and reverse the interpretation. A weak answer says it is "an endpoint with multiple parts."

**Question 8. How do you align the label with the estimand?**

A strong answer: compare the population, treatment condition, outcome, strategy, and summary in the label with the trial estimand; revise the label or the analysis if they differ. A weak answer says "regulatory affairs handles it."

**Question 9. What is the difference between an ICE and missing data?**

A strong answer: an ICE changes the interpretation of the outcome; missing data is the absence of a measurement. They are handled by different plans. A weak answer treats them as the same.

**Question 10. Why can the same data produce different conclusions?**

A strong answer: because different estimands answer different questions; a composite including discontinuation can reverse the direction of a treatment-policy effect. A weak answer says "because of bias."

## 15. Key terms

- **Analysis population:** the set of participants included in an analysis.
- **As-treated:** analysis by treatment received.
- **Composite:** an ICE strategy that includes the event in the endpoint.
- **Estimand:** the precise quantity to be estimated, defined by five attributes.
- **Estimate:** the numerical result.
- **Estimator:** the statistical method used to estimate the estimand.
- **Hypothetical:** an ICE strategy that estimates the effect if the event did not occur.
- **Intercurrent event (ICE):** an event after treatment initiation that affects interpretation.
- **Intention-to-treat (ITT):** analysis by assigned group regardless of treatment received.
- **Missing data:** absence of an outcome measurement.
- **Per-protocol:** analysis restricted to adherent participants.
- **Population-level summary:** the contrast used to summarise the effect.
- **Principal ignorability:** the assumption that potential outcomes are independent of stratum given covariates.
- **Principal stratum:** a latent subgroup defined by potential occurrence of an ICE.
- **Rescue medication:** additional treatment given for lack of efficacy.
- **Sensitivity analysis:** an analysis that tests robustness to an assumption.
- **Survivor average causal effect:** the effect in patients who would survive to a time point.
- **Switching:** changing to another treatment.
- **Treatment condition:** the intervention, comparator, dose, duration, and care.
- **Treatment policy:** an ICE strategy that uses the outcome regardless of the event.
- **Variable:** the outcome, its measurement, and its timing.
- **While on treatment:** an ICE strategy that restricts observation to the on-treatment period.
- **Withdrawal of consent:** a participant's request to stop participating.

## 16. Sources

All URLs accessed 25 October 2026.

### Guidelines

- ICH, E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials, Step 4, 20 November 2019. https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf
- ICH, E9 Statistical Principles for Clinical Trials, Step 4, 1998. https://database.ich.org/sites/default/files/E9_Guideline.pdf
- ICH, M11 Clinical electronic Structured Harmonised Protocol (CeSHarP), Step 4, 19 November 2025. https://database.ich.org/sites/default/files/ICH_Step4_M11_Final_Guideline_2025_1119.pdf
- EMA, ICH E9(R1) Step 5, effective 30 July 2020. https://www.ema.europa.eu/en/documents/scientific-guideline/ich-e9-r1-addendum-estimands-and-sensitivity-analysis-clinical-trials-guideline-statistical-principles-clinical-trials-step-5_en.pdf
- FDA, E9(R1) Estimands and Sensitivity Analysis in Clinical Trials, final guidance, May 2021. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/e9r1-estimands-and-sensitivity-analysis-clinical-trials
- European Union, Regulation (EU) No 536/2014 on clinical trials. https://eur-lex.europa.eu/eli/reg/2014/536/oj

### Methods

- Hernán MA, Robins JM. Causal Inference: What If. Chapman and Hall/CRC, 2020.
- Frangakis CE, Rubin DB. Principal stratification in causal inference. Biometrics 2002;58:21-29. https://doi.org/10.1111/j.0006-341X.2002.00021.x
- Athey S, Imbens GW. The state of applied econometrics: causality and policy evaluation. J Econ Perspect 2017;31:3-32. https://doi.org/10.1257/jep.31.2.3

### Case evidence

- RECOVERY Collaborative Group. Dexamethasone in Hospitalized Patients with Covid-19. N Engl J Med 2021;384:693-704. https://doi.org/10.1056/NEJMoa2021436
- Perez MV, Mahaffey KW, Hedlin H, et al. Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation. N Engl J Med 2019;381:1909-1917. https://doi.org/10.1056/NEJMoa1901183

### Further reading

- Akacha M, Bretz F, Ruberg S. Estimands in clinical trials: broadening the regulatory perspective. Stat Med 2017;36:1-16. https://doi.org/10.1002/sim.7208
- Leuchs AK, Zinserling J, Brandt A, et al. Choosing appropriate estimands in clinical trials. Ther Innov Regul Sci 2015;49:584-592.
- Rufibach K. Treatment effect quantification for clinical trials with time-to-event endpoints: estimands and sensitivity analysis. Patient Prefer Adherence 2019;13:1833-1846.
- Mallinckrodt CH, Bell J, Liu G, et al. Aligning items in the estimand framework with the ICH E9(R1) addendum. Pharm Stat 2020;19:1-14. https://doi.org/10.1002/pst.1977
- Bornkamp B, Rufibach K, Lin J, et al. Principal stratum strategy: potential role in drug development. Pharm Stat 2021;20:737-751. https://doi.org/10.1002/pst.2104
- Darken PF, Hoaglin DC. Evaluation of the estimand framework. Stat Biopharm Res 2021;13:1-10.
- Mehrotra DV, Hemmings RJ, Russek-Cohen E. Seeking harmony: estimands and sensitivity analyses for confirmatory clinical trials. Clin Trials 2016;13:456-458.
- Ratitch B, Bell J, Mallinckrodt C, et al. Choosing estimands in clinical trials with missing data. Pharm Stat 2020;19:1-15.
- Holzhauer B, Akacha M, Bermann G. Choice of estimand and analysis methods in diabetes trials with rescue medication. Pharm Stat 2015;14:433-447. https://doi.org/10.1002/pst.1701
- Qu Y, Lipkovich I. Propensity score estimation with missing data. Pharm Stat 2020;19:1-12.
- Wang Y, Tu W, Kim Y, et al. Statistical considerations for estimands in vaccine trials. Pharm Stat 2022;21:1-14.
- Lipkovich I, Ratitch B, O'Kelly M. Sensitivity analysis for estimands in clinical trials. Pharm Stat 2022;21:1-18.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`06_sample-size-and-power.md`](06_sample-size-and-power.md)
- Next topic: [`08_external-validation.md`](08_external-validation.md)
- Related topic: [`01_from-claim-to-evidence.md`](01_from-claim-to-evidence.md)
- Module survey file: [`../../05_study_design_evidence.md`](../../05_study_design_evidence.md)
- Labs: [`../../labs/`](../../labs/)
