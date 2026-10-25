# From Claim to Evidence

> Module 05 — Study Design and Evidence Generation
> Topic 01 of 10
> Estimated study time: 3 hours
> Prerequisites: Module 04 (statistics and causal inference foundations)
> Version 2026.3

## 1. Why this topic exists

Every biomedical product is sold, approved, reimbursed, or rejected on the strength of a sentence. The sentence is the claim: "this device detects diabetic retinopathy," "this drug reduces hospitalisation," "this model flags deterioration twelve hours earlier," "this pathway lowers cost per episode." The claim is the commercial object. The study is only a machine for producing evidence about the claim.

Most failed evidence programmes do not fail at the analysis stage. They fail earlier, when a claim is translated into a study that answers a slightly different question. The failure is quiet because the study still produces a number. The number is simply not the number the decision needs. A trial can be well powered, well conducted, and statistically significant while remaining commercially useless because it measured a surrogate the payer does not accept, in a population the label excludes, against a comparator nobody uses, over a horizon shorter than the decision cycle.

Three groups of people pay for this translation error:

- **Regulators** receive a submission whose primary endpoint does not support the proposed indication. The review cycle lengthens, a post-marketing requirement is imposed, or the application is refused. The cost is measured in quarters of revenue and in repeated studies.
- **Payers and health technology assessment (HTA) bodies** receive an economic dossier built on a claim that was never tested against the relevant comparator. Coverage is denied or restricted, and the product must fund a new study from its own margin.
- **Clinical adopters** buy a product whose benefit was demonstrated in a setting unlike theirs. The product underperforms, the champion loses credibility, and procurement adds a validation requirement that the vendor did not budget for.

Data scientists inherit studies after design decisions are frozen. That inheritance is the specific problem this topic addresses. By the time you are asked to "analyse the data," the claim, population, comparator, endpoint, and analysis population have usually been fixed in a protocol or a contract. If those choices are wrong, no amount of modelling repairs them. The professional skill is to intervene one step earlier: to read a claim, expose its hidden assumptions, and convert it into an evidence plan whose results can actually support the decision.

The industry question for this module is: "What is the smallest, most credible study that can support this claim?" This topic gives the first half of the answer: what "support" means, and what counts as the claim.

### The claim is not the hypothesis

A marketing claim and a statistical hypothesis are different objects. Marketing language compresses several propositions into one sentence and omits the conditions that make the sentence true. "Reduces sepsis mortality by 30%" omits the population, the comparator, the time window, the baseline risk, the mechanism of action, and the difference between relative and absolute risk. A statistical hypothesis is precise enough to be wrong.

The work of this topic is decompression. You take a compressed commercial sentence and expand it into a set of testable propositions, each with a population, an intervention, a comparator, an outcome, a time horizon, and an effect measure. Only then can you ask whether a design exists that can test those propositions at acceptable cost.

### Why this is hard in practice

Four structural pressures push teams toward under-specified claims.

1. **Commercial urgency.** Sales needs a number before the evidence exists. The claim is written by marketing, then handed to clinical affairs, then to statistics. Each hand-off loses precision.
2. **Regulatory and commercial claims diverge.** A regulator may accept a surrogate endpoint for approval; a payer may refuse to pay without a patient-relevant outcome. One evidence package rarely satisfies both without deliberate design.
3. **The claim drifts.** During development, the population narrows, the endpoint changes, and the comparator shifts as standard of care evolves. The protocol may be amended; the commercial claim often is not.
4. **AI products invite performance claims.** A model's area under the receiver operating characteristic curve (AUC) is easy to report and easy to market. It says nothing about clinical benefit, workflow, or harm. The gap between discrimination and decision value is where most AI evidence programmes fail.

### What "evidence" means here

Evidence is not data. Evidence is data plus a design plus an argument that connects the data to the claim. A dataset of one million patients is raw material. Evidence is the combination of:

- a defined question,
- a design that isolates the effect of interest from plausible alternatives,
- a prespecified analysis,
- a result with quantified uncertainty,
- and a transparent statement of what the result does and does not support.

This definition matters because it makes evidence quality a property of the whole chain, not of the sample size. A large biased study is weak evidence. A small well-designed study can be strong evidence for a narrow claim. The rest of this module elaborates that chain; this topic builds its first link.

### What happens when the claim is skipped

Three recurring patterns appear in industry post-mortems.

**The endpoint substitution.** A team sets out to show clinical benefit, discovers that a mortality trial is too large, and substitutes a biomarker. The biomarker moves. The product launches with a claim the payer treats as unproven. The commercial team then commissions the mortality study anyway, two years later, at a higher cost.

**The population drift.** A model is developed on adults aged 18 to 65 with complete laboratory data, then deployed in an emergency department where half the patients are over 75 and a third have missing labs. The published validation was correct for the development population. The claim was silently universal.

**The comparator lag.** A trial compares a new drug with the standard of care at protocol design. By the time the trial reports, the standard of care has changed. The result is statistically valid and clinically obsolete. The claim survives; the decision context does not.

Each pattern is preventable at the claim-translation step. None is repairable at the analysis step.

### What a good evidence plan looks like

An evidence plan is a short document, usually two to five pages, that sits between the commercial strategy and the protocol. It is not the protocol and it is not a statistical analysis plan. It answers five questions:

1. What exactly is claimed, and for whom?
2. Who must believe it, and what decision follows?
3. What evidence would be sufficient for that decision?
4. What is the smallest study that can produce that evidence?
5. What result would cause the programme to stop or change the claim?

The plan should be readable by a commercial lead, a clinician, and a statistician without translation. That is a demanding constraint. It forces the writer to define terms and to avoid statistical shorthand. A plan that only a statistician can read will not be used by the people who decide whether to fund the study.

A good plan has four properties.

**It is specific.** "Improves outcomes" is replaced by a named outcome, a named population, a named comparator, and a named horizon.

**It is decision-linked.** Each study in the plan is attached to a decision and a decision date. Studies that support no decision are either exploratory or unnecessary.

**It is sized.** The plan states the sample size and the assumptions behind it, even when the number is approximate. A plan without a number cannot be budgeted.

**It is falsifiable.** The plan states what result would contradict the claim. A plan that cannot fail is not an evidence plan; it is a communication plan.

### The three questions this topic answers

The rest of the module assumes that the claim has been translated. This topic supplies three questions that every later topic depends on.

**Question 1: What is the estimand?** Every effect estimate is an estimate of something. The estimand defines that something. Without it, the choice between a t-test, a mixed model, and a propensity score is arbitrary.

**Question 2: What design can identify it?** The design is the structure that makes the estimand estimable. Randomisation, blinding, concurrent controls, and prespecified analyses are tools for identification, not rituals.

**Question 3: What decision does it inform?** The decision determines the acceptable uncertainty. A regulator may require a 95 percent confidence interval excluding a null effect; a procurement committee may require a local validation; a payer may require a cost-effectiveness estimate. The same result can be sufficient for one and insufficient for another.

If you can answer these three questions for a claim, you can design a study. If you cannot, no amount of data will help.

## 2. Learning outcomes

By the end of this topic you can:

1. **Decompose** a commercial or clinical claim into atomic propositions covering population, intervention, comparator, outcome, time horizon, and effect magnitude, and identify which propositions are unspecified. (Exercise 1, self-check 1)
2. **Classify** a claim as descriptive, diagnostic, predictive, comparative, or impact, and select the minimum design family that can support each class. (Exercise 2, self-check 2)
3. **Identify** the decision and decision-maker behind a claim, and distinguish regulatory, payer, procurement, and clinical evidence requirements. (Exercise 3, interview 2)
4. **Write** an estimand with the five ICH E9(R1) attributes before selecting a design, and explain how intercurrent events change the claim. (Exercise 4, self-check 4)
5. **Distinguish** a binding regulatory requirement from a methodological convention, with the source and date of the requirement. (Exercise 5, interview 4)
6. **Size** a simple comparative study from the claim's minimum clinically important difference, including dropout, and state what the resulting study can and cannot conclude. (Exercise 6, self-check 6)
7. **Audit** an existing evidence package for claim-evidence gaps and write a one-page remediation plan with go/no-go criteria. (Exercises 7 and 8, interview 8)

Each outcome is observable. "Decompose" means producing a table, not describing the idea of decomposition. "Size" means producing a number with units and assumptions. The exercises in Section 12 are the assessment instrument for these outcomes.

## 3. The core mental model

The model has four translations and one loop. Start with the sentence the organisation wants to be true. End with a decision rule that says what result would change someone's behaviour.

```text
   COMMERCIAL OR CLINICAL SENTENCE
   "X reduces Y by Z% in population P"
                |
                |  1. DECOMPRESS
                v
   ATOMIC PROPOSITIONS
   population | intervention | comparator |
   outcome | horizon | effect measure | magnitude
                |
                |  2. LOCATE THE DECISION
                v
   DECISION + DECISION-MAKER + CONSEQUENCE
   approve | cover | adopt | prescribe | buy | label
                |
                |  3. STATE THE ESTIMAND
                v
   ESTIMAND (five attributes)
   population | treatment condition | variable |
   intercurrent events | population-level summary
                |
                |  4. SELECT AND SIZE THE DESIGN
                v
   DESIGN -> PROTOCOL -> DATA -> ANALYSIS -> RESULT
                |
                |  5. COMPARE RESULT WITH DECISION RULE
                v
   DECISION  ---- feedback: claim revised, new study, or stop
```

The loop is the important part. Evidence generation is not a linear pipeline from claim to approval. It is a sequence of decisions, each of which can send the programme back to revise the claim. A good evidence plan makes the loop explicit and cheap: it defines in advance which result triggers which decision.

### Translation 1: sentence to propositions

A usable claim has seven slots. If a slot is empty, the claim is not yet testable.

| Slot | Question it answers | Example for a sepsis alert |
|---|---|---|
| Population | Who is eligible, and in what setting? | Adults admitted to hospital with suspected infection and at least two systemic inflammatory response syndrome criteria |
| Intervention | What exactly is delivered, by whom, with what adherence? | A real-time alert to the attending team, with a fixed response protocol |
| Comparator | What happens instead? | Usual care without the alert |
| Outcome | Which variable, measured how? | All-cause mortality at 28 days |
| Horizon | Over what period? | 28 days from randomisation |
| Effect measure | Ratio, difference, or time-to-event? | Risk ratio |
| Magnitude | What counts as success? | Relative reduction of at least 20 percent |

The marketing sentence usually supplies two or three slots and implies the rest. The implied slots are where disputes later occur. Writing them down is the first act of design.

### Translation 2: propositions to decision

A claim can be true and still not matter. The decision determines which truth is needed.

| Decision | Decision-maker | What convinces them | Typical evidence |
|---|---|---|---|
| Market authorisation | Regulator | Substantial evidence of efficacy and a favourable benefit-risk profile | Adequate and well-controlled trials |
| Coverage and payment | Payer or HTA body | Cost-effectiveness, budget impact, patient-relevant benefit | Comparative effectiveness, economic model |
| Hospital adoption | Procurement committee, clinical leadership | Local performance, workflow fit, liability, integration cost | External validation, workflow study, reference sites |
| Clinical behaviour | Individual clinician or guideline panel | Net benefit in the patient in front of them | Guideline-grade evidence, decision aids |
| Investor confidence | Board or investor | A credible path to a defensible claim | Milestone data, regulatory agreement |

A single study rarely satisfies all five. The evidence plan should name the decision it is designed for and state which other decisions it leaves unanswered. This is not a weakness; it is honesty about scope.

### Translation 3: decision to estimand

The estimand is the precise definition of the quantity to be estimated. ICH E9(R1) defines five attributes: population, treatment condition, variable, handling of intercurrent events, and population-level summary. The estimand is not the same as the estimator, and it is not the same as the endpoint. It is the answer to "what number, in what people, under what treatment conditions, summarised how?"

The order matters. If you choose the analysis first, the estimand becomes whatever the analysis happens to estimate. That is how trials end up with a "treatment policy" result being presented as if it measured biological efficacy, or a per-protocol result being presented as if it measured the effect of assignment.

For an AI alert, a treatment-policy estimand asks: what is the effect of offering the alert strategy, regardless of whether clinicians acted on it? A hypothetical estimand asks: what would the effect be if every clinician acted on every alert? These answer different questions and can produce different numbers in the same trial. The claim usually implies one of them; the protocol must say which.

### Translation 4: estimand to design

The design is the structure that makes the estimand identifiable. Randomisation addresses confounding. Blinding addresses ascertainment bias. A concurrent control addresses secular trends. A run-in addresses population selection. Follow-up duration addresses the horizon. Allocation ratio and sample size address precision.

Design selection is constrained by the estimand, not the reverse. If the estimand requires a counterfactual "if no rescue medication were available," a trial in which rescue medication is routinely used cannot estimate it without strong assumptions. If the estimand is the effect of assignment, discontinuation must be handled by treatment policy, and the analysis must retain discontinued participants.

### The evidence ladder

Not all claims require the same strength of design. A useful heuristic is a five-rung ladder. The rungs are not a hierarchy of prestige; they are a hierarchy of what the design can support.

| Rung | Claim type | Minimum credible design | What it cannot support |
|---|---|---|---|
| 1 | Descriptive: prevalence, usage, burden | Cross-sectional or registry | Any causal or comparative statement |
| 2 | Diagnostic or predictive performance | Cross-sectional accuracy study with reference standard | Clinical benefit, impact on outcomes |
| 3 | Comparative performance | Head-to-head comparison, ideally randomised | Real-world effectiveness, cost |
| 4 | Comparative effectiveness | Pragmatic randomised trial or strong target trial emulation | Mechanism, subgroup biology |
| 5 | Impact: mortality, cost, system performance | Randomised impact trial or stepped-wedge with adjudicated outcomes | Fine-grained mechanism, long-term rare harms |

Most commercial claims are rung 4 or 5, while most AI evidence is rung 2. The gap between rung 2 and rung 5 is the most common claim-evidence gap in digital health.

### Two failure shapes

The model produces two diagnostic shapes that recur in review.

**The cone.** The claim is broad at the top ("improves outcomes for all patients") and the evidence is narrow at the bottom (one site, one age band, one endpoint). The cone is unstable: any reviewer can push on the narrow base and the claim falls.

**The inverted cone.** The evidence is broad (many datasets, many endpoints, many subgroup analyses) and the claim is narrow (one indication). This is wasteful rather than invalid. It usually indicates that the team ran exploratory work before deciding what it needed to prove.

The evidence plan should make the shape visible. A one-page table with the claim in one column and the supporting study in the next exposes both shapes immediately.

## 4. Definitions and vocabulary

Terms are listed in the order they appear in the workflow. The "common misuse" column records how the term is used incorrectly in industry documents.

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Claim | A statement the organisation wants a decision-maker to accept | Used interchangeably with "hypothesis" or "message" | None |
| Intended use | The specific clinical purpose, population, user, and setting for which a product is authorised | Confused with "indication" | Free text, regulated |
| Indication | The disease or condition for which a product is approved or used | Used as a synonym for intended use, omitting user and setting | Controlled vocabulary in labelling |
| Label | The regulator-approved product information, including indications, contraindications, and warnings | Treated as marketing copy | Structured document |
| Substantial evidence | The US statutory standard for effectiveness: adequate and well-controlled investigations, or one such investigation plus confirmatory evidence | Treated as "any statistically significant result" | Legal standard, FD&C Act section 505(d) |
| Adequate and well-controlled study | A study with a clear statement of objectives, a valid comparison, a defined population, and a prespecified analysis | Applied loosely to any randomised study | Regulatory term |
| Confirmatory evidence | Independent evidence that strengthens a single trial's conclusion | Confused with a subgroup analysis of the same trial | Regulatory term, context-dependent |
| Primary endpoint | The outcome on which the primary claim and sample size are based | Chosen after seeing the data | Prespecified variable |
| Surrogate endpoint | A marker substituted for a patient-relevant outcome, justified by evidence that changes in the marker predict changes in the outcome | Assumed to be valid because it is correlated with the outcome | Variable, with validation evidence |
| Estimand | The precise quantity to be estimated, defined by five attributes | Confused with the estimator or the analysis method | Structured definition |
| Estimator | The statistical method used to estimate the estimand | Used as a synonym for estimand | Method name |
| Intercurrent event | An event after treatment initiation that affects the interpretation or existence of outcome measurements | Ignored or treated as missing data | Event category |
| Effect measure | The scale on which a contrast is expressed: risk difference, risk ratio, odds ratio, hazard ratio, mean difference | Chosen for convenience rather than decision relevance | Unit depends on scale |
| Minimum clinically important difference (MCID) | The smallest change in an outcome that a patient or clinician would consider important | Replaced by "statistically detectable difference" | Outcome-specific units |
| Non-inferiority margin | The largest loss of effect relative to an active comparator that is clinically acceptable | Set by convention without historical evidence | Effect-measure units |
| Assay sensitivity | The ability of a trial to distinguish an effective treatment from an ineffective one | Assumed rather than demonstrated | Trial property |
| Decision-maker | The person or body whose behaviour the evidence is meant to change | Left unnamed | Role |
| Evidence gap | The difference between what the claim requires and what the planned study can support | Discovered after the study | Gap description |
| Regulatory pathway | The procedural route to market authorisation: 510(k), De Novo, premarket approval (PMA), new drug application (NDA), biologics licence application (BLA) | Confused with the evidence standard | Regulatory category |
| 510(k) | US premarket notification showing substantial equivalence to a legally marketed predicate device | Treated as an approval | Device pathway |
| De Novo | US classification pathway for a novel low- to moderate-risk device with no predicate | Confused with PMA | Device pathway |
| PMA | US premarket approval for high-risk devices, requiring valid scientific evidence | Treated as the only rigorous device pathway | Device pathway |
| Coverage with evidence development (CED) | Payer coverage conditioned on participation in a study that generates further evidence | Treated as unconditional coverage | Payer mechanism |
| Health technology assessment (HTA) | Systematic evaluation of clinical, economic, and social effects of a technology | Treated as a synonym for regulatory review | Process |
| Statistical analysis plan (SAP) | The prespecified document describing analyses, populations, and methods | Written after unblinding | Document |
| Analysis population | The set of participants included in a given analysis, defined before analysis | Chosen to produce significance | Definition |
| Intention-to-treat (ITT) | Analysis by assigned group, regardless of received treatment | Applied only to complete cases | Principle |
| Per-protocol | Analysis restricted to participants who adhered to the protocol | Presented as the main result when ITT is null | Population |
| Power | The probability of rejecting the null hypothesis when a specified alternative is true | Reported without the assumed effect size | Probability, 0 to 1 |
| Alpha | The prespecified type I error rate | Adjusted informally after multiple looks | Probability |
| Go/no-go criterion | A prespecified result threshold that determines whether the programme proceeds | Invented after results are known | Decision rule |
| Evidence package | The set of studies, analyses, and arguments submitted to a decision-maker | Treated as a single study | Collection |

Three vocabulary distinctions deserve emphasis because they cause most of the confusion in review meetings.

**Claim versus message.** A message is what the sales force says. A claim is what the evidence can defend. The two should be identical, but they are governed by different review processes. When they diverge, the evidence team is usually asked to "support the message," which is backwards.

**Endpoint versus estimand.** The endpoint is the variable. The estimand adds the population, the treatment condition, the handling of intercurrent events, and the summary measure. Two trials can share an endpoint and have different estimands, which is why endpoint-level comparisons across trials are unreliable.

**Requirement versus convention.** A requirement is binding: a statute, regulation, or formal guidance. A convention is a shared methodological preference: a reporting checklist, a journal policy, a company standard. Both matter, but only the first can cause a submission to fail on compliance grounds. Section 8 separates them.

## 5. The mechanics, step by step

This section is the procedure. Each step states its input, its output, and the reason it exists. Steps 1 to 5 translate the claim. Steps 6 to 10 translate the decision into a design. Steps 11 to 14 make the design executable and auditable. The branch points are marked, and the edge cases are collected at the end.

The procedure is written for a design review that lasts ninety minutes and ends with a go, a revise, or a stop. It is not a template for a protocol. A protocol is written after the procedure has produced a defensible claim and design; Topic 10 covers that document. Keeping the two artefacts separate prevents the common failure in which the protocol is written first and the claim is reverse-engineered from it.

### Procedure at a glance

| Step | Input | Output | Decision it forces |
|---:|---|---|---|
| 1 | Sentence from deck, label, or grant | Verbatim claim with source and date | Is there a single sentence to test? |
| 2 | Verbatim claim | Seven-slot table with gaps marked | Which slots are assumptions? |
| 3 | Business context | Named decision, decision-maker, date | Who acts on the result? |
| 4 | Seven-slot table | Claim class | What is the minimum design? |
| 5 | Claim class and context | Intended-use paragraph | Who uses it, on whom, to do what? |
| 6 | Intended use and practice | Comparator and care pathway | What is the alternative? |
| 7 | Decision and comparator | Primary outcome and horizon | What would change behaviour? |
| 8 | Population, treatment, outcome | Five-attribute estimand | What exactly is the effect? |
| 9 | Estimand and decision rule | Effect measure and threshold | How much is enough? |
| 10 | All of the above | Design family and rejected alternatives | What structure identifies the effect? |
| 11 | Effect, variance, alpha, power | Sample size with arithmetic trail | Is the study feasible and decision-relevant? |
| 12 | Estimand and design | SAP and analysis population | What will be estimated, in whom? |
| 13 | Design and data sources | Bias register and sensitivity analyses | What could overturn the result? |
| 14 | Everything above | Go/no-go rule and evidence package | What result triggers which action? |

The table is also a review checklist. If a step has no output, the review cannot proceed to the next step, because the next step depends on it. In practice, steps 3 and 8 are the ones most often skipped, and they are the two that most often determine whether the evidence is usable.

### Step 1: Write the claim verbatim

**Input:** the sentence as it appears in the pitch deck, label draft, grant application, or sales script.

**Output:** the exact sentence, with its source and date, plus a list of every qualifier that was present.

**Rationale:** Paraphrasing early hides the ambiguity. The verbatim sentence is the object you will later test against the evidence. Recording the source and date matters because claims drift; a claim from an investor deck and a claim from a draft label are different commitments.

Do not improve the sentence at this stage. If it says "up to 30 percent," record "up to." If it says "in some patients," record that. If it cites a study, record the study identifier.

### Step 2: Decompress into atomic propositions

**Input:** the verbatim claim.

**Output:** a seven-slot table (population, intervention, comparator, outcome, horizon, effect measure, magnitude), with each slot marked as explicit, implied, or absent.

**Rationale:** Each empty slot is an assumption that someone will later dispute. Marking it as empty converts a hidden assumption into a design question.

A practical test: read the claim to a clinician who has not seen the product and ask what study would change their mind. The gaps in their answer are the gaps in the claim.

### Step 3: Identify the decision and the decision-maker

**Input:** the seven-slot table and the business context.

**Output:** a named decision, a named decision-maker, a decision date, and the consequence of a negative decision.

**Rationale:** Evidence requirements are decision-specific. A claim intended for a hospital procurement committee needs local validation and workflow evidence; the same claim intended for a regulator needs a controlled premarket study. Without a named decision, the evidence plan has no stopping rule.

If more than one decision matters, rank them. The primary decision drives the design; secondary decisions are addressed by additional analyses or separate studies.

### Step 4: Classify the claim

**Input:** the seven-slot table.

**Output:** one claim class from the table below.

**Rationale:** The claim class determines the minimum design family and the reporting guideline. Misclassification is the most common cause of a rung-2 study being used to support a rung-5 claim.

| Claim class | Question form | Minimum design | Typical reporting guideline |
|---|---|---|---|
| Descriptive | How common is X? | Cross-sectional, registry, or cohort | STROBE |
| Diagnostic accuracy | How well does X detect Y? | Cross-sectional with reference standard | STARD |
| Predictive performance | How well does X predict Y? | Cohort with prespecified model and calibration | TRIPOD+AI |
| Comparative performance | Is X better than Z on measure Y? | Randomised or matched comparison | CONSORT or STROBE |
| Comparative effectiveness | Does X improve Y in routine care? | Pragmatic trial or target trial emulation | CONSORT extension or STROBE |
| Impact | Does X change mortality, cost, or system outcomes? | Randomised impact trial, stepped-wedge, or strong quasi-experiment | CONSORT or TRIPOD+AI plus economic reporting |

Note that "diagnostic accuracy" and "predictive performance" are different classes even when both use an AUC. A diagnostic test has a reference standard and a clinical action; a prognostic model estimates future risk. The evidence requirements differ, particularly around calibration and decision thresholds.

### Step 5: Define the intended use precisely

**Input:** claim class and clinical context.

**Output:** a paragraph specifying the condition, population, user, setting, input data, output, and clinical action.

**Rationale:** Intended use is the anchor for regulatory and procurement review. A model that predicts sepsis in intensive care is not the same product as a model that predicts sepsis on admission; a tool used by a specialist is not the same tool used by a triage nurse.

The paragraph should answer six questions: who uses it, on whom, with what data, at what moment, to make what decision, and with what expected action if the output is positive. If any answer is "it depends," the intended use is not yet defined.

### Step 6: Specify the comparator and the care pathway

**Input:** intended use and current practice.

**Output:** a description of usual care, including the standard of care at the time of the study and the expected trajectory without the intervention.

**Rationale:** A comparative claim is only meaningful against a named alternative. "Usual care" is not a comparator until it is described. In fast-moving fields, usual care at the end of a three-year trial may differ from usual care at design.

Record the version of the care pathway, the guideline date, and any co-interventions. If the comparator is expected to change, plan a contemporaneous control or a design that is robust to drift.

### Step 7: Choose the outcome and the horizon

**Input:** intended use, comparator, and the decision the study must inform.

**Output:** a primary outcome with a definition, a measurement instrument, a timing rule, and a justification for its patient relevance or surrogate status.

**Rationale:** The outcome is the claim's operational definition. A claim about "improving care" cannot be tested; a claim about "28-day all-cause mortality" can. The horizon must match the decision: a 30-day outcome cannot support a claim about chronic disease management.

The outcome should pass four tests:

1. **Relevance:** would a patient, clinician, or payer act differently if it changed?
2. **Sensitivity:** can the intervention plausibly move it within the study period?
3. **Measurability:** can it be ascertained reliably and completely?
4. **Specificity:** is it plausibly affected by the intervention rather than by concurrent changes?

If the primary outcome is a surrogate, record the evidence that links it to the patient-relevant outcome, and state the residual uncertainty. A surrogate with a strong meta-analytic link is defensible; a surrogate chosen because it is cheap is not.

### Step 8: State the estimand before the design

**Input:** population, treatment condition, outcome, and the clinical question.

**Output:** a five-attribute estimand statement, plus a list of intercurrent events and the strategy chosen for each.

**Rationale:** The estimand forces the team to decide what "the effect" means before choosing a model. It also exposes disagreements that would otherwise surface after unblinding.

The five attributes, following ICH E9(R1):

1. **Population:** the patients of interest, including any subgroup restriction.
2. **Treatment condition:** the intervention and comparator, including dose, duration, and concomitant care.
3. **Variable:** the outcome, including the measurement and timing.
4. **Intercurrent events:** events that affect interpretation, and the strategy for each: treatment policy, hypothetical, composite, while on treatment, or principal stratum.
5. **Population-level summary:** the contrast, such as a risk difference, risk ratio, or hazard ratio.

Write the estimand as a single sentence. For example: "In adults hospitalised with suspected sepsis, the risk difference in all-cause mortality at 28 days between a strategy of alert-assisted care and usual care, regardless of whether clinicians acted on the alert." The phrase "regardless of whether clinicians acted" is the intercurrent-event strategy. Removing it changes the estimand.

### Step 9: Choose the effect measure and the success threshold

**Input:** the estimand and the decision rule.

**Output:** an effect measure, a target magnitude, and a justification for the magnitude.

**Rationale:** Statistical significance is not a decision threshold. The decision needs a magnitude that is clinically or commercially meaningful. That magnitude may come from a minimum clinically important difference, a payer threshold, a competitor's performance, or a regulatory precedent.

For a superiority claim, the threshold is the smallest effect that would change practice. For a non-inferiority claim, the threshold is the largest acceptable loss, and it must be justified from historical evidence of the comparator's effect. For an impact claim, the threshold may be a cost per quality-adjusted life year (QALY) or a budget-impact limit.

If the team cannot state a threshold, the claim is not yet a decision claim. It is a research question. That is acceptable, but it should be labelled as such.

### Step 10: Select the design family

**Input:** claim class, estimand, comparator, outcome, and constraints.

**Output:** a design family, a justification, and a list of rejected alternatives with reasons.

**Rationale:** The design must make the estimand identifiable. Randomisation is the most credible way to handle confounding, but it is not always feasible or ethical. When it is not, the design must substitute explicit assumptions and sensitivity analyses.

Use the following branch logic.

| Condition | Design branch | Required justification |
|---|---|---|
| Comparative claim, feasible randomisation, equipoise | Parallel-group randomised trial | Equipoise and acceptable burden |
| Comparative claim, strong patient preference | Randomised trial with preference option, or comprehensive cohort | Preference effects quantified |
| Comparative claim, rare outcome | Case-control or registry-based cohort | Selection and recall bias addressed |
| Comparative claim, randomisation infeasible | Target trial emulation or propensity-matched cohort | Positivity, exchangeability, and time-zero alignment |
| Diagnostic claim | Cross-sectional accuracy study with consecutive sampling | Reference standard and spectrum |
| Predictive claim | Prospective cohort with prespecified model | Calibration and missing data |
| Impact claim, system-level intervention | Cluster randomised or stepped-wedge trial | Intracluster correlation and contamination |
| Impact claim, single-arm device | Single-arm trial with external control | Historical effect and drift |
| Exploratory signal | Single-arm or case series | Explicitly labelled exploratory |

The branch is not a menu. Each row carries an assumption that must be defended. A single-arm device study with an external control is a legitimate design only when the natural history is well characterised, the endpoint is objective, and the external data are contemporaneous and comparable. Those conditions are rarely all true.

### Step 11: Size the study for the decision, not for the endpoint

**Input:** effect measure, target magnitude, variance or event rate, alpha, power, allocation ratio, expected dropout, and clustering.

**Output:** a sample size with an explicit arithmetic trail and a sensitivity table.

**Rationale:** The sample size is where the claim's ambition meets its budget. A study powered for a 30 percent relative reduction will be much smaller than one powered for 15 percent. The team must decide which effect it is actually trying to detect, and whether a smaller, less precise study is still decision-relevant.

Three checks are mandatory:

- **Event count:** for binary and time-to-event outcomes, the number of events, not the number of participants, drives power.
- **Dropout and non-adherence:** inflate the sample size for expected missing data, and state the assumption.
- **Multiplicity:** if there are multiple primary endpoints, subgroups, or interim looks, the alpha must be controlled, and the sample size must reflect the adjusted alpha.

The arithmetic is covered in Topic 06. For this topic, the requirement is that the claim's magnitude, not a convenient round number, drives the calculation.

### Step 12: Prespecify the analysis and the analysis population

**Input:** estimand, design, and sample size.

**Output:** a statistical analysis plan with the primary estimand, the primary estimator, the analysis population, the handling of missing data, and the sensitivity analyses.

**Rationale:** Prespecification is what separates confirmatory evidence from exploratory analysis. The analysis population must match the estimand. A treatment-policy estimand requires an ITT population; a hypothetical estimand may require a per-protocol or principal-stratum analysis with explicit assumptions.

The plan should also state what will not be done. A list of excluded analyses is as useful as a list of planned ones, because it prevents post hoc rescue attempts from being presented as prespecified.

### Step 13: Map the biases and the sensitivity analyses

**Input:** design and data sources.

**Output:** a bias register with the expected direction and magnitude of each bias, and a sensitivity analysis for each material bias.

**Rationale:** Every design has residual biases. The question is whether they are small enough to leave the decision unchanged. A bias register makes that judgement explicit and gives reviewers a way to test it.

For each bias, record:

- the mechanism,
- the likely direction,
- the plausible magnitude,
- the design feature that mitigates it,
- the sensitivity analysis that quantifies the residual.

If the conclusion survives the sensitivity analyses, the evidence is robust. If it does not, the claim must be weakened.

### Step 14: Define the go/no-go rule and the evidence package

**Input:** all of the above.

**Output:** a one-page decision rule and a list of studies and analyses that constitute the package.

**Rationale:** A programme without a stopping rule drifts. The rule should specify the result that triggers continuation, the result that triggers a revised claim, and the result that triggers termination.

The evidence package should name the primary study, the supporting studies, the analyses that address each claim element, and the gaps that remain. It should also state which decision each element supports.

### Branch points and how to record them

A branch point is a place where the procedure can legitimately take more than one path. The purpose of recording it is not bureaucracy. It is to make the choice auditable when the result is known and someone asks why the study was designed that way. Each branch record has four fields: the condition that triggers the branch, the options considered, the option chosen, and the evidence or constraint that decided it.

| Branch point | Condition | Options | Record required |
|---|---|---|---|
| Claim class | The claim mixes performance and benefit language | Performance claim, benefit claim, or split into two claims | Which claim the primary study will support, and which is deferred |
| Regulatory route | Product is software used for a clinical decision | Unregulated wellness, clinical decision support, device | Legal analysis of intended use and claims |
| Comparator | An effective standard of care exists | Placebo, active comparator, usual care, external control | Ethical and regulatory justification for the choice |
| Endpoint | A patient-relevant outcome is infeasible | Surrogate, composite, or shorter horizon | Evidence linking the surrogate to the outcome |
| Randomisation | Individual randomisation is infeasible or unethical | Cluster, stepped-wedge, target trial emulation, single-arm | Assumption ledger and sensitivity analyses |
| Estimand strategy | Participants may discontinue or switch | Treatment policy, hypothetical, composite, while on treatment, principal stratum | Which strategy the claim implies, and why |
| Margin | Claim is non-inferiority or equivalence | Fixed margin, synthesis method, or superiority instead | Historical effect estimate and assay sensitivity |
| Alpha allocation | More than one primary endpoint or population | Hierarchy, split alpha, or single primary endpoint | The family-wise error rate and the testing sequence |
| Interim analysis | Early stopping or adaptation is contemplated | No interim, futility only, efficacy and futility, adaptive sample size | Operating characteristics and data access plan |
| Evidence package | One study is insufficient | One pivotal plus confirmatory evidence, two pivotal trials, or a conditional pathway | What each element contributes and what remains uncertain |

The branch record should be written in the protocol, not in a separate decision log that nobody reads. A protocol that says "a pragmatic design was chosen" without recording the alternatives invites the reviewer to assume the choice was arbitrary. A protocol that says "a cluster design was rejected because the ICC was expected to be 0.05, which would have required 84 clusters and exceeded the number of eligible wards" demonstrates that the design was reasoned.

Not every branch needs a quantitative justification. Some are constrained by ethics, law, or feasibility. The record should state the constraint plainly. "Individual randomisation was rejected because the intervention is delivered at the ward level and could not be withheld from control patients" is a complete justification. It does not need a simulation.

A worked branch record for the comparator decision shows the level of detail that survives review:

```text
Branch: comparator
Condition: an effective standard of care exists; placebo would deny it.
Options considered:
  A. Placebo control. Rejected: withholding the standard of care would be unethical
     in a condition with proven effective therapy.
  B. Active comparator, current guideline regimen. Selected.
  C. Usual care as practised. Rejected: too heterogeneous to define a contrast.
  D. External control. Rejected: no contemporaneous registry with the required
     endpoint definitions and severity variables.
Chosen: B, with the guideline version and date recorded in the protocol.
Residual risk: the guideline may change during the trial. Mitigation: a
  prespecified sensitivity analysis excluding participants enrolled after a
  guideline change, and a contemporaneous control arm.
```

The record is short, but it answers the three questions a reviewer will ask: why this comparator, why not the alternatives, and what happens if the comparator moves. A protocol that contains this record is easier to defend than one that contains only the selected design.

### Edge cases

**Non-inferiority claims.** The claim is not "better" but "not materially worse." The design requires an active comparator, a justified margin, assay sensitivity, and a prespecified analysis population that is often both ITT and per-protocol. The margin must be smaller than the comparator's historical effect, and the constancy assumption must be argued. A non-inferiority claim with a margin chosen for convenience is not a claim; it is a negotiation.

The practical test is whether the margin can be derived from a historical trial or meta-analysis showing that the comparator was superior to placebo. If no such evidence exists, the non-inferiority claim has no anchor, and the trial cannot distinguish a genuinely equivalent product from an ineffective one. The FDA guidance of November 2016 calls this the assay-sensitivity problem, and it is the most common reason a non-inferiority submission fails.

**Multi-arm and multi-stage claims.** When a study tests several doses, several subgroups, or several endpoints, the family-wise error rate rises. The design must prespecify the hierarchy, the alpha allocation, and the decision rules. A claim that emerges from an unadjusted subgroup comparison is exploratory, regardless of the p-value.

Multi-arm trials also create a shared-control problem. If several experimental arms are compared with one control, the control participants are used more than once, and the correlation between comparisons must be accounted for in the analysis. The design should state whether the comparisons are independent, whether a gatekeeping procedure is used, and whether the trial is powered for the most important comparison or for all of them.

**Cluster designs.** When the intervention is delivered at the level of a clinic or ward, the unit of allocation is the cluster. The sample size must account for the intracluster correlation coefficient (ICC), and the analysis must use methods that respect the clustering. A cluster trial analysed as if individuals were independent will overstate precision.

Cluster size variation is a separate problem. If clusters vary widely in size, the effective sample size is not the total number of individuals divided by the design effect; the variation itself reduces precision. A common rule is that a coefficient of variation of cluster size above 0.23 requires an additional inflation. The design should state the expected distribution of cluster sizes, not only the mean.

**Adaptive designs.** Interim looks at the data can stop a trial early for efficacy or futility, or change the sample size. The adaptation rule must be prespecified, the operating characteristics must be simulated, and the type I error must be controlled. An adaptation invented at the interim is not an adaptive design; it is an unplanned analysis.

Timing matters. An interim analysis that occurs after most of the events have accrued has little ability to stop the trial early, while an interim that occurs too early has unstable estimates and a high chance of stopping on noise. The design should state the information fraction at each interim and the reason for that timing.

**External controls.** When the comparator comes from outside the trial, the claim depends on the comparability of populations, eras, and endpoint definitions. The design must include a prespecified confounding-adjustment strategy, sensitivity analyses for unmeasured confounding, and, where possible, a concurrent control arm. Drift in standard of care is the central threat.

External-control designs also require a positivity check: every combination of confounders observed in the trial population must have a counterpart in the external population. If the trial enrols patients with a severity profile that the external data do not contain, adjustment cannot repair the gap. The design should report the overlap before the analysis, not after.

**Missing endpoint data.** Missingness can bias the estimate if it is related to outcome. The analysis plan must state the assumed missingness mechanism, the primary handling method, and the sensitivity analyses under plausible departures. Complete-case analysis is rarely adequate for a confirmatory claim.

The most useful sensitivity analysis is a tipping-point analysis: how much worse would the missing outcomes have to be in one arm before the conclusion reverses? If the answer is implausibly extreme, the result is robust. If the answer is a small departure from the assumed mechanism, the result is fragile and the claim should be softened.

**Subgroup claims.** A subgroup claim is a claim about a population subset. It requires either a prespecified subgroup analysis with an interaction test and adequate power, or a separate study in the subgroup. A forest plot that shows one subgroup with a significant effect and others without is not evidence of a subgroup effect; the interaction test is the evidence. The design should state which subgroups are prespecified, how many, and whether the trial is powered for any of them.

**Composite endpoints.** A composite endpoint combines several outcomes into one. It can increase the event rate and reduce sample size, but it changes the claim: a benefit driven by a mild component may not be a benefit patients value. The design should prespecify the components, their relative importance, and an analysis of each component. A composite that is not decomposed is a claim that cannot be interpreted.

### Decision table: claim type to evidence package

| Claim type | Primary design | Supporting evidence | Typical failure if omitted |
|---|---|---|---|
| Diagnostic accuracy | Cross-sectional accuracy study | External validation, subgroup accuracy | Spectrum bias, unusable thresholds |
| Predictive performance | Prospective cohort with calibration | Temporal and external validation | Overfitting, calibration drift |
| Comparative efficacy | Randomised controlled trial | Dose-response, long-term extension | Unrepresentative population |
| Comparative effectiveness | Pragmatic trial or target trial emulation | Adherence and subgroup analyses | Confounding, low uptake |
| Impact on mortality or cost | Randomised impact trial or stepped-wedge | Economic model, implementation study | Underpowered for rare outcomes |
| Non-inferiority | Active-comparator trial with justified margin | Historical effect estimate, assay sensitivity | Margin not defensible |
| Safety | Large simple trial or registry | Spontaneous reporting, observational signal | Rare events undetected |
| Label expansion | Pivotal trial in new population | Bridging study, PK/PD | Population not representative |

The table is a starting point, not a substitute for judgement. The design must still be justified in the protocol, and the justification must address the specific claim.

## 6. Worked example

### The claim

A hospital software vendor is preparing a procurement bid for "SepsisGuard," a machine-learning alert that runs on the electronic health record (EHR) and notifies the attending team when a hospitalised adult meets a sepsis deterioration pattern. The sales deck says:

> "SepsisGuard reduces 28-day mortality in hospitalised patients with sepsis by 25 percent."

The procurement committee has asked for the evidence behind the number. The vendor has a retrospective development study (AUC 0.81 in a held-out sample) and a 400-patient pilot at two sites with a 2.1 percentage point absolute mortality difference favouring the alert.

### Step 1: Verbatim claim and source

The sentence is recorded from the sales deck, version 4.2, dated 12 March 2026. The qualifier "hospitalised patients with sepsis" is present. The comparator is absent. The time horizon is present (28 days). The effect measure is relative (25 percent), but the baseline risk is absent.

### Step 2: Seven-slot decompression

| Slot | Value in the claim | Status |
|---|---|---|
| Population | Hospitalised adults with sepsis | Explicit but undefined (which definition of sepsis?) |
| Intervention | SepsisGuard alert plus whatever response the team makes | Implied |
| Comparator | Not stated | Absent |
| Outcome | All-cause mortality | Explicit |
| Horizon | 28 days | Explicit |
| Effect measure | Relative reduction | Explicit but ambiguous (risk ratio or odds ratio?) |
| Magnitude | 25 percent | Explicit |
| Setting | Not stated | Absent |
| User | "The team" | Absent |

The absent slots are the negotiation surface. The committee will ask: compared with what, in which hospitals, with what response protocol, and measured as a risk ratio or an odds ratio? Each unanswered question is a reason to defer the purchase.

### Step 3: Decision and decision-maker

The decision is a hospital procurement decision, expected in the fourth quarter of 2026. The decision-maker is a committee comprising the chief medical information officer, the chief nursing officer, an infectious diseases physician, and a finance lead. A negative decision means no contract; a positive decision means a three-year deployment.

The primary evidence requirement is local credibility: the committee wants to know whether the effect observed elsewhere is plausible in their hospitals. The regulator is not the decision-maker here, because the product is marketed as clinical decision support and the vendor has not sought a device authorisation. That does not remove regulatory risk, but it changes the evidence plan.

### Step 4: Claim classification

The claim is comparative effectiveness (rung 4), not diagnostic accuracy (rung 2). The vendor's AUC evidence is necessary but insufficient. The procurement decision requires a comparison of outcomes under two care strategies.

### Step 5: Intended use

The vendor writes the following intended-use paragraph:

> SepsisGuard is intended for use by attending physicians and charge nurses in adult inpatient wards of acute-care hospitals. It ingests vital signs, laboratory results, and nursing assessments from the EHR every 15 minutes and displays a sepsis deterioration alert when the modelled risk exceeds a fixed threshold. The expected action is a bedside review within 60 minutes and, if sepsis is suspected, initiation of the local sepsis bundle.

The paragraph defines the user, the population, the data, the timing, the output, and the action. It also reveals a design dependency: the effect depends on the response protocol, not only on the algorithm.

### Step 6: Comparator and care pathway

The comparator is usual care without the alert, defined as the local sepsis screening protocol, which uses a modified early warning score and a nurse-initiated review. The standard of care includes a sepsis bundle with antibiotics within one hour, fluid resuscitation, and lactate measurement. The guideline version is recorded (local protocol v6, January 2026).

This is important: the alert's incremental effect is measured against an existing screening process, not against no screening. A trial that compares the alert with no screening would overstate the benefit.

### Step 7: Outcome and horizon

The primary outcome is all-cause mortality at 28 days from randomisation. Secondary outcomes are time to antibiotics, intensive care unit (ICU) transfer, length of stay, and alert burden (alerts per 100 patient-days). The horizon matches the procurement decision, which is about short-term safety and throughput.

The outcome passes the four tests: it is patient-relevant, plausibly affected within 28 days, ascertainable from the EHR and death records, and specific enough that a concurrent control can isolate the alert's contribution.

### Step 8: Estimand

The estimand is:

> In adults admitted to an acute-care hospital ward with suspected sepsis, the risk difference in all-cause mortality at 28 days between a strategy of SepsisGuard alert-assisted care and usual care, regardless of whether clinicians viewed or acted on the alert.

The intercurrent-event strategy is treatment policy. Clinicians who ignore the alert remain in their assigned strategy, because the claim is about offering the alert, not about obeying it. A hypothetical estimand ("if every clinician acted on every alert") would answer a different question and would require an adherence-adjusted analysis.

### Step 9: Effect measure and success threshold

The vendor's 25 percent relative reduction corresponds, at a 20 percent baseline mortality, to an absolute reduction of 5 percentage points (from 20 percent to 15 percent). The procurement committee's threshold is a relative reduction of at least 20 percent, with the upper bound of the 95 percent confidence interval below 1.0 for the risk ratio. The committee also wants the absolute difference reported, because the budget case depends on deaths avoided per 1,000 admissions.

### Step 10: Design selection

A parallel-group randomised trial is feasible: the alert can be randomised at the patient level within wards, and there is equipoise because the alert's benefit is unproven. The alternative, a stepped-wedge cluster design, was rejected because the vendor expects contamination between patients and because the committee wants an individual-level effect estimate. A target trial emulation was rejected because the EHR does not contain a credible instrument for alert exposure.

The design is a pragmatic, open-label, two-arm, individually randomised trial with 1:1 allocation, stratified by site and by baseline sepsis severity (qSOFA score 0 to 1 versus 2 or more). The analysis is by intention to treat.

### Step 11: Sample size arithmetic

Assumptions:

- Baseline 28-day mortality in the usual-care arm: 20 percent.
- Target effect: 25 percent relative reduction, giving 15 percent in the alert arm.
- Alpha: 0.05, two-sided.
- Power: 90 percent.
- Allocation: 1:1.
- Expected dropout or loss to follow-up: 10 percent.

The standard normal quantiles are 1.959964 for alpha 0.05 two-sided and 1.281552 for 90 percent power. The per-arm sample size for a two-proportion comparison is:

```text
n = (z_alpha/2 + z_beta)^2 * [p0(1-p0) + p1(1-p1)] / (p1 - p0)^2
  = (1.959964 + 1.281552)^2 * [0.20*0.80 + 0.15*0.85] / (0.15 - 0.20)^2
  = 10.507 * [0.16 + 0.1275] / 0.0025
  = 10.507 * 0.2875 / 0.0025
  = 1208.3
```

Rounding up gives 1,209 participants per arm, or 2,418 in total. Inflating for 10 percent dropout gives 1,209 / 0.90 = 1,343.3, so 1,344 per arm and 2,688 in total.

The expected number of events is 2,688 * 0.20 = 537.6, or about 538 events. The event count is adequate for the normal approximation.

### Step 12: Analysis plan

- Primary analysis: risk difference and risk ratio with 95 percent confidence intervals, adjusted for stratification factors.
- Analysis population: ITT, defined as all randomised participants with a valid outcome or censoring time.
- Missing outcome: multiple imputation under a missing-at-random assumption, with a tipping-point sensitivity analysis.
- Sensitivity analyses: per-protocol (alert viewed within 60 minutes), as-treated, and a worst-case imputation for missing outcomes.
- Subgroups: prespecified for baseline severity, age band, and site; reported as interaction tests, not as separate claims.
- Multiplicity: one primary endpoint, so no alpha adjustment for the primary analysis. Secondary endpoints are reported with confidence intervals and labelled supportive.

### Step 13: Bias register

| Bias | Mechanism | Direction | Mitigation |
|---|---|---|---|
| Confounding | Randomisation | None expected | Stratified allocation |
| Performance bias | Clinicians know the assignment | Could increase effect if teams respond | Standardised response protocol; as-treated analysis |
| Detection bias | Mortality is objective | Minimal | Death registry ascertainment |
| Attrition | Transfer or discharge before 28 days | Could dilute effect | Complete follow-up via registry |
| Contamination | Alert information shared between arms | Toward the null | Ward-level separation, monitoring |
| Alert fatigue | High false-positive rate reduces response | Toward the null | Report alert burden; prespecified threshold |
| Secular trend | Sepsis bundle changes during trial | Either direction | Concurrent control; short enrolment window |

### Step 14: Go/no-go rule

The programme proceeds if the adjusted risk ratio is below 0.85 and the upper bound of the 95 percent confidence interval is below 1.0, with no safety signal in alert-related adverse events. If the point estimate is between 0.85 and 1.0, the claim is revised to "may reduce mortality" and a larger confirmatory study is required. If the point estimate is above 1.0, the programme stops and the alert is redeveloped.

### Two diagnostics that reveal a flawed design

**Diagnostic 1: the event-rate check.** The sample size assumes 20 percent baseline mortality. If the actual control-arm mortality is 12 percent, the expected number of events falls to 2,688 * 0.12 = 322.6, and the power for a 25 percent relative reduction falls well below 90 percent. A blinded sample-size review at 50 percent enrolment would reveal the discrepancy. If the review shows 12 percent, the options are to increase the sample size, extend enrolment, or redefine the claim around a larger relative effect. A design that does not plan this review is fragile to a single wrong assumption.

**Diagnostic 2: the uptake check.** The estimand is treatment policy, so the trial's effect includes clinicians ignoring the alert. If the pilot showed that only 40 percent of alerts were viewed within 60 minutes, the intention-to-treat effect will be diluted. The design should prespecify a measure of uptake and a per-protocol sensitivity analysis. If uptake is below 50 percent, the trial may be testing the alert's implementation rather than its algorithm, and the claim should be reframed as an implementation claim.

### Vary one assumption: the effect size

If the true effect is a 15 percent relative reduction rather than 25 percent, the alert-arm mortality is 17 percent. The per-arm sample size becomes:

```text
n = 10.507 * [0.20*0.80 + 0.17*0.83] / (0.17 - 0.20)^2
  = 10.507 * [0.16 + 0.1411] / 0.0009
  = 10.507 * 0.3011 / 0.0009
  = 3515.6
```

Rounding up gives 3,516 per arm, or 7,032 before dropout and 7,814 after 10 percent dropout. The trial is 2.9 times larger. The procurement committee's 20 percent threshold is therefore the difference between a feasible study and a programme that may not be funded.

### Vary one assumption: dropout

If dropout is 25 percent rather than 10 percent, the sample size becomes 1,209 / 0.75 = 1,612 per arm, or 3,224 in total. The event count remains adequate, but the cost rises by about 20 percent. If dropout is informative (sicker patients leave), the ITT estimate may be biased toward the null, and the tipping-point analysis becomes central to the interpretation.

### Vary one assumption: the margin

If the claim were non-inferiority against an active comparator with a margin of 3 percentage points, the sample size would be driven by the margin, not by the expected effect. With a one-sided alpha of 0.025, power 90 percent, and assumed event rates of 20 percent in both arms, the per-arm sample size would be approximately:

```text
n = (1.959964 + 1.281552)^2 * [0.20*0.80 + 0.20*0.80] / (0.03)^2
  = 10.507 * 0.32 / 0.0009
  = 3735.8
```

That is 3,736 per arm, larger than the superiority design. The margin, not the point estimate, is the expensive parameter. This is why non-inferiority claims are often more costly than they appear.

### The decision

The vendor presents the design and the arithmetic to the procurement committee. The committee accepts the 25 percent target as the claim it will evaluate, requires the trial to be registered and the SAP published, and sets the go/no-go rule as above. The vendor's original 400-patient pilot is reclassified as a feasibility study, not as evidence for the claim. The procurement decision is deferred until the trial reports, with a conditional letter of intent.

### Why the pilot was not evidence

The pilot compared 200 alert patients with 200 concurrent controls and observed a 2.1 percentage point absolute difference in mortality. At a baseline of 20 percent, that corresponds to 18.0 percent versus 20.1 percent, a relative reduction of about 10 percent. The 95 percent confidence interval for the difference is wide: with 200 per arm and an event proportion near 19 percent, the standard error of the difference is approximately the square root of 0.19 times 0.81 times (1/200 + 1/200), or 0.039, giving a half-width of about 7.7 percentage points. The interval therefore runs from roughly -5.6 to +9.8 percentage points, and it includes both no effect and a clinically important benefit.

The pilot is not useless. It establishes recruitment feasibility, alert delivery, and the plausibility of the event rate. Those are design inputs, not evidence for the claim. The distinction is the difference between a feasibility study and a confirmatory study, and it is the distinction the procurement committee was implicitly asking about when it asked for "the evidence behind the number."

### The arithmetic in one table

| Scenario | Control risk | Alert risk | Relative reduction | n per arm | Total with 10% dropout |
|---|---:|---:|---:|---:|---:|
| Base case | 20% | 15% | 25% | 1,209 | 2,688 |
| Smaller effect | 20% | 17% | 15% | 3,516 | 7,814 |
| Higher dropout | 20% | 15% | 25% | 1,209 | 3,224 |
| Lower baseline risk | 12% | 9% | 25% | 2,190 | 4,868 |
| Non-inferiority, 3-point margin | 20% | 20% | 0% | 3,736 | 8,304 |

The table is the argument. Each row is a different world, and the sample size changes by a factor of three across the rows. A design review that considers only the base case is not a review; it is a forecast.

A second lesson is hidden in the table. The non-inferiority row has a zero relative reduction by construction, because the two arms are assumed to have the same event rate. The sample size is driven entirely by the margin. A team that thinks of non-inferiority as "the cheaper superiority design" has the arithmetic backwards: a tight margin can require more participants than a superiority design with a realistic effect. The choice between superiority and non-inferiority is a claim decision, not a budget decision.

### What the example shows

The example shows the central lesson: the claim did not become credible because the vendor believed it. It became credible when the sentence was converted into a population, a comparator, an outcome, a horizon, an estimand, a design, and a decision rule.

## 7. Code, query, or calculation

The code below implements the sample-size arithmetic used in the worked example. It uses only the Python standard library, including an inverse normal quantile function so that no external package is required. Run it as a script; the expected output is shown after the block.

```python
import math


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def n_per_arm(p0, p1, alpha=0.05, power=0.80):
    """Sample size per arm for a two-sided two-proportion comparison."""
    za = norm_ppf(1 - alpha / 2)
    zb = norm_ppf(power)
    return (za + zb) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p1 - p0) ** 2


def minimum_detectable_effect(n, p0, alpha=0.05, power=0.80):
    """Approximate absolute risk difference detectable with n per arm."""
    za = norm_ppf(1 - alpha / 2)
    zb = norm_ppf(power)
    return (za + zb) * math.sqrt(2 * p0 * (1 - p0) / n)


def inflate(n, dropout):
    """Inflate a per-arm sample size for dropout."""
    return math.ceil(n / (1 - dropout))


if __name__ == "__main__":
    p0 = 0.20
    print("Assumption: control mortality 20%, alpha 0.05 two-sided, power 90%")
    for rel in (0.25, 0.20, 0.15):
        p1 = p0 * (1 - rel)
        n = n_per_arm(p0, p1, power=0.90)
        print(f"relative reduction {rel:.0%}: p1={p1:.4f}, "
              f"n/arm={math.ceil(n)}, total={2*math.ceil(n)}")
    n90 = math.ceil(n_per_arm(p0, 0.15, power=0.90))
    print(f"with 10% dropout: n/arm={inflate(n90, 0.10)}, "
          f"total={2*inflate(n90, 0.10)}")
    print(f"with 25% dropout: n/arm={inflate(n90, 0.25)}, "
          f"total={2*inflate(n90, 0.25)}")
    print(f"expected events at 20%: {2*inflate(n90, 0.10)*0.20:.1f}")
    print(f"MDE with 200/arm: {minimum_detectable_effect(200, p0, power=0.80):.4f} "
          f"absolute ({minimum_detectable_effect(200, p0, power=0.80)/p0:.1%} relative)")
    print(f"MDE with 1000/arm: {minimum_detectable_effect(1000, p0, power=0.80):.4f} "
          f"absolute ({minimum_detectable_effect(1000, p0, power=0.80)/p0:.1%} relative)")
```

Expected output:

```text
Assumption: control mortality 20%, alpha 0.05 two-sided, power 90%
relative reduction 25%: p1=0.1500, n/arm=1209, total=2418
relative reduction 20%: p1=0.1600, n/arm=1934, total=3868
relative reduction 15%: p1=0.1700, n/arm=3516, total=7032
with 10% dropout: n/arm=1344, total=2688
with 25% dropout: n/arm=1612, total=3224
expected events at 20%: 537.6
MDE with 200/arm: 0.1121 absolute (56.0% relative)
MDE with 1000/arm: 0.0501 absolute (25.1% relative)
```

The numbers should be read as follows.

- The first line reproduces the worked example: 1,209 per arm at 90 percent power, not the 903 per arm that 80 percent power would give. The power target is a design choice, and it changes the budget by about one third.
- The second and third lines show the cost of a smaller effect. Halving the assumed effect roughly triples the sample size.
- The fourth and fifth lines show the dropout inflation. The arithmetic is simple, but the assumption is not: a 25 percent dropout rate must be justified from pilot data, not chosen to make the budget work.
- The last two lines show why a 400-patient pilot cannot support the claim. With 200 per arm, the smallest detectable relative reduction is about 56 percent. A 25 percent effect is invisible.

The code deliberately avoids a continuity correction. For large samples the difference is negligible; for small samples, an exact or continuity-corrected method should be used. The choice should be stated in the SAP.

### A small claim-audit function

The second block turns the seven-slot table into a machine-readable completeness check. It is not a statistical method; it is a discipline tool for design reviews.

```python
CLAIM_SLOTS = ("population", "intervention", "comparator", "outcome",
               "horizon", "effect_measure", "magnitude")


def audit_claim(claim):
    """Return missing slots and a crude sufficiency flag."""
    missing = [s for s in CLAIM_SLOTS if not claim.get(s)]
    explicit = len(CLAIM_SLOTS) - len(missing)
    return {
        "missing": missing,
        "explicit_slots": explicit,
        "testable": len(missing) == 0,
    }


if __name__ == "__main__":
    example = {
        "population": "adults hospitalised with sepsis",
        "intervention": "SepsisGuard alert plus local response protocol",
        "comparator": None,
        "outcome": "all-cause mortality",
        "horizon": "28 days",
        "effect_measure": "risk ratio",
        "magnitude": 0.25,
    }
    result = audit_claim(example)
    print(result)
```

Expected output:

```text
{'missing': ['comparator'], 'explicit_slots': 6, 'testable': False}
```

The function is intentionally simple. Its value is procedural: a claim with a missing slot cannot pass a design review, regardless of how promising the pilot data look.

## 8. Data, standards, and regulatory context

This section separates three kinds of authority: binding law and regulation, formal guidance that is not itself binding but is treated as the standard of review, and methodological convention. Confusing them leads teams to over-comply with checklists and under-comply with statutes.

### Binding requirements

**United States.** The effectiveness standard for drugs is in section 505(d) of the Federal Food, Drug, and Cosmetic Act (FD&C Act): substantial evidence consists of adequate and well-controlled investigations, including clinical investigations, by experts qualified by scientific training and experience. The statute permits one adequate and well-controlled investigation plus confirmatory evidence. For devices, section 513(a)(3)(A) defines valid scientific evidence, and the premarket approval standard requires it. These are legal requirements, not conventions.

**European Union.** Regulation (EU) 2017/745 governs medical devices, and Regulation (EU) 2017/746 governs in vitro diagnostic devices. The Medical Device Regulation (MDR) requires clinical evaluation and, for certain high-risk devices, a clinical investigation. The In Vitro Diagnostic Regulation (IVDR) imposes performance evaluation requirements. The Clinical Trials Regulation (EU) No 536/2014 governs clinical trials of medicinal products. These are binding instruments.

**Good clinical practice.** ICH E6(R3), adopted at Step 4 on 6 January 2025, is the current international good clinical practice (GCP) guideline. It replaced the E6(R2) addendum and restructures GCP around principles, an Annex 1 covering investigator, sponsor, and data governance, and appendices including the protocol. In the EU, the European Medicines Agency (EMA) published the Step 5 version with effect from 23 July 2025. In the US, the Food and Drug Administration (FDA) issued the guideline in September 2025. E6(R3) is binding to the extent a jurisdiction adopts it; the protocol content requirements in its appendix are the closest thing to a binding protocol skeleton.

### Formal guidance that defines the standard of review

**ICH E9(R1).** The addendum on estimands and sensitivity analysis reached Step 4 on 20 November 2019. The EMA adopted it at Step 5 on 30 January 2020, with effect from 30 July 2020. The FDA issued it as final guidance on 12 May 2021. It is not a statute, but a confirmatory trial that cannot state its estimand is now difficult to defend in any of the three major jurisdictions.

**FDA guidance on substantial evidence.** The FDA issued a revised draft guidance, "Demonstrating Substantial Evidence of Effectiveness for Human Drug and Biological Products," on 24 June 2026 (docket FDA-2019-D-4964). It is a draft, not for implementation, with comments due 22 September 2026. It clarifies the one-trial-plus-confirmatory-evidence route and removes the language that treated one trial as the legal equivalent of two. Because it is a draft, it should be cited as a statement of current thinking, not as a binding requirement. The 1998 guidance it is intended to replace remains the formal document until finalisation.

**FDA guidance on non-inferiority.** "Non-Inferiority Clinical Trials to Establish Effectiveness" (November 2016) remains the current final guidance. It describes the M1 and M2 margins, assay sensitivity, and the constancy assumption. It is guidance, not regulation, but a non-inferiority claim that ignores it will not survive review.

**FDA guidance on adaptive designs.** "Adaptive Designs for Clinical Trials of Drugs and Biologics" (December 2019) remains current. It requires prespecification of adaptation rules, a data access plan, simulation reports, and appropriate interaction with the FDA. No 2025 revision was identified in the sources checked for this topic; teams should verify the current version on the FDA guidance page before relying on the date.

**FDA guidance on externally controlled trials.** "Considerations for the Design and Conduct of Externally Controlled Trials for Drug and Biological Products" was issued as a draft in February 2023 and remains a draft. It applies to drugs and biologics, not devices. Device practice relies on the 2013 pivotal-design guidance and the real-world evidence guidance described below.

**FDA real-world evidence guidance.** For devices, the final guidance "Use of Real-World Evidence to Support Regulatory Decision-Making for Medical Devices" was issued in December 2025 (docket FDA-2023-D-4395), superseding the August 2017 version. The Federal Register notice of availability was published on 18 December 2025. For drugs and biologics, the current framework guidance is "Considerations for the Use of Real-World Data and Real-World Evidence to Support Regulatory Decision-Making for Drug and Biological Products" (August 2023). The FDA's December 2025 announcement that it will accept certain device real-world evidence without identifiable patient-level data is a policy statement; the guidance is the operative document.

**Reporting guidelines.** CONSORT 2025 (BMJ 2025;389:e081123) supersedes CONSORT 2010 for randomised trials. SPIRIT 2025 (BMJ 2025;389:e081477) supersedes SPIRIT 2013 for protocols. TRIPOD+AI (BMJ 2024;385:e078378) supersedes TRIPOD 2015 for prediction models. PROBAST+AI (BMJ 2025;388:e082505) supersedes PROBAST 2019 for risk-of-bias assessment. STROBE (2007) remains the reporting guideline for observational studies. These are conventions enforced by journals and increasingly by regulators, not statutes.

### What the standards constrain and what they do not

| Instrument | Constrains | Does not constrain |
|---|---|---|
| FD&C Act 505(d) | The evidence standard for drug effectiveness | The choice of endpoint, comparator, or design |
| MDR/IVDR | Clinical evaluation and investigation obligations | The statistical method or sample size |
| ICH E6(R3) | GCP, data governance, protocol content | The estimand or analysis strategy |
| ICH E9(R1) | Estimand definition and sensitivity analysis | The choice of estimator |
| CONSORT 2025 | Reporting content | Study conduct |
| SPIRIT 2025 | Protocol content | Study conduct |
| TRIPOD+AI | Reporting of prediction models | Model development choices |
| PROBAST+AI | Risk-of-bias assessment | The conduct of the study |
| FDA RWE guidance (2025) | Relevance and reliability of real-world data | The causal claim itself |

The table is a warning against treating a reporting checklist as a design method. A study can comply with CONSORT and still answer the wrong question. The checklist improves transparency; it does not improve the claim.

### Data standards relevant to the evidence package

The evidence package for a regulated product usually includes data in a standard format. The Clinical Data Interchange Standards Consortium (CDISC) Study Data Tabulation Model (SDTM) and Analysis Data Model (ADaM) are the conventional formats for FDA submissions. The ICH M11 Clinical electronic Structured Harmonised Protocol (CeSHarP) reached Step 4 on 19 November 2025 and provides a structured protocol template with controlled terminology. For observational data, the Observational Medical Outcomes Partnership (OMOP) Common Data Model is widely used, and the Sentinel system uses a distributed data model. These standards constrain how evidence is represented, not what evidence is needed.

### The regulatory-versus-payer gap

A regulator asks whether the product is safe and effective for its intended use. A payer asks whether it is worth paying for relative to the alternatives. The evidence for the first is usually a placebo-controlled or active-comparator trial; the evidence for the second is usually a comparative effectiveness study with economic endpoints. The same trial can sometimes serve both, but only if the comparator, population, and endpoint are chosen with both decisions in mind.

The coverage with evidence development (CED) mechanism is an explicit acknowledgement of the gap. The Centers for Medicare and Medicaid Services (CMS) may cover a technology conditionally while requiring further evidence. For the product team, CED is both an opportunity and a warning: it means the evidence was insufficient for unconditional coverage, and the study that follows is part of the commercial plan, not an academic afterthought.

## 9. Industry practice and case evidence

This section separates documented facts from inference. Case facts are drawn from the published record; the lessons are inferred and labelled.

### RECOVERY: a claim matched to a pragmatic design

The RECOVERY trial randomised hospitalised patients with COVID-19 to dexamethasone or usual care. In the final analysis, 2,104 participants received dexamethasone and 4,321 received usual care. Overall 28-day mortality was 22.9 percent versus 25.7 percent (age-adjusted rate ratio 0.83, 95 percent confidence interval 0.75 to 0.93). Among participants receiving invasive mechanical ventilation, mortality was 29.3 percent versus 41.4 percent (rate ratio 0.64, 95 percent confidence interval 0.51 to 0.81). Among those receiving no respiratory support, there was no benefit (17.8 percent versus 14.0 percent; rate ratio 1.19, 95 percent confidence interval 0.92 to 1.55).

The documented fact is the effect estimate and its heterogeneity by respiratory support. The inference is that the claim should have been population-specific from the start: "reduces mortality in patients requiring respiratory support" is supported; "reduces mortality in all hospitalised patients" is not. RECOVERY's design made the heterogeneity visible because it prespecified the subgroup analysis and enrolled a broad population. A narrower trial in ventilated patients only would have produced a cleaner claim but a slower answer.

**Lesson for claim translation.** A broad population can support a narrow claim if the analysis is prespecified and the subgroup is biologically plausible. A narrow population cannot support a broad claim. The direction of inference matters.

### Apple Heart Study: detection is not impact

The Apple Heart Study enrolled 419,297 participants in a siteless, app-based study of an irregular pulse notification algorithm. Over a median 117 days, 2,161 participants (0.52 percent) received a notification. Among 450 notified participants who returned analyzable electrocardiogram patches, atrial fibrillation was present in 34 percent. The positive predictive value for notification concordance with simultaneous patch-detected atrial fibrillation was 0.84.

The documented fact is the feasibility of large-scale screening and the modest confirmation rate among those who returned patches. The inference is that the study supports a detection claim, not an outcome claim. It did not randomise participants to notification versus no notification, did not measure stroke or mortality, and could not estimate the downstream clinical workload per true case. The 0.52 percent notification rate, applied to a health system's population, is a workload estimate, but the study did not measure the care pathway.

**Lesson for claim translation.** A diagnostic or screening claim needs a reference standard and a care-pathway analysis. The Apple Heart Study supplied the first and not the second. A commercial claim about "improving outcomes" would require a different design.

### Epic Sepsis Model: the vendor claim versus external validation

Wong and colleagues externally validated a widely implemented proprietary sepsis prediction model at Michigan Medicine. The cohort included 27,697 patients with 38,455 hospitalisations; sepsis was present in 2,552 hospitalisations (7 percent). The hospitalisation-level AUC was 0.63 (95 percent confidence interval 0.62 to 0.64), compared with the vendor's reported 0.76 to 0.83. At the alert threshold of 6 or more, 6,971 of 38,455 hospitalisations (18 percent) would trigger an alert; sensitivity was 33 percent, positive predictive value 12 percent, and the model missed 1,709 of 2,552 sepsis cases (67 percent).

The documented fact is the performance gap and the alert burden. The inference is that the vendor's claim was developed and validated in a different population and data environment, and that discrimination measured at a fixed threshold does not translate into clinical utility. The external validation did not test whether the model reduced mortality; it tested whether the model performed as claimed. The answer was no.

**Lesson for claim translation.** A performance claim must specify the population, the data environment, and the decision threshold. An AUC without a threshold and a calibration assessment is not a claim about clinical use.

### IDx-DR: a narrow claim with a prospective package

The FDA authorised IDx-DR through the De Novo pathway in 2018 (DEN180001) for autonomous detection of diabetic retinopathy in adults with diabetes. The authorisation was based on a prospective clinical study with a clear intended use, a defined image-acquisition protocol, and a reference standard. The product's claim was narrow: detect more than mild diabetic retinopathy and refer; do not diagnose other conditions.

The documented fact is the De Novo authorisation and the intended-use restriction. The inference is that autonomy was supportable because the claim was bounded: a specific population, a specific capture protocol, a specific decision, and a specific action. The evidence package did not need to show that the device improved long-term visual outcomes; it needed to show that the device performed its detection task reliably in the intended population.

**Lesson for claim translation.** Narrow claims are cheaper to support and easier to defend. The commercial temptation is to broaden the claim after authorisation; the evidence rarely broadens with it.

### What companies actually do

The following practices are inferred from public regulatory documents, guidance, and published study reports, not from a systematic survey.

1. **Evidence gap analysis before protocol design.** Mature teams maintain a claim register that maps each commercial claim to a study, an endpoint, and a decision-maker, and flags claims with no study. The register is reviewed at phase gates.
2. **Early regulator interaction.** The FDA's Q-Submission programme and the EMA's scientific advice process are used to test endpoint and design choices before committing to a trial. The interaction is not a guarantee, but it reduces the risk of a design that the regulator will reject.
3. **Payer research in parallel.** Teams that sell into coverage systems often run payer advisory boards before the pivotal trial to test whether the endpoint and comparator will support a coverage decision.
4. **Pre-registration and SAP publication.** Registration and SAP publication are increasingly expected by journals and payers, and they reduce the temptation to revise the claim after results.
5. **External validation as a launch requirement.** For AI products, hospital procurement often requires a local or independent validation, even when the regulator did not. The evidence plan should budget for it.

### Where practice falls short

The same public record shows recurring shortfalls.

- **AUC-led evidence.** Many AI publications report discrimination without calibration, decision-curve analysis, or workflow impact. The claim then outruns the evidence.
- **Single-site validation.** Models validated at the development site are presented as generalisable. The Epic Sepsis Model case is the clearest public example of the gap.
- **Surrogate claims.** Biomarker or imaging endpoints are presented as clinical benefit without a validated link.
- **Unregistered analyses.** Subgroup claims appear without a prespecified analysis plan, and the reader cannot distinguish confirmation from exploration.

### The claim register as an operating artefact

Mature organisations do not manage claims in slide decks. They maintain a register with one row per claim and the following columns: claim identifier, verbatim claim, intended use, decision, decision-maker, claim class, primary study, primary endpoint, estimand summary, current evidence status, gap, owner, and next review date. The register is reviewed at every phase gate.

The register is useful for three reasons. First, it prevents claim drift: a change to the sentence requires a change to the row and a review of the evidence. Second, it exposes orphan claims: sentences that appear in marketing material but have no study behind them. Third, it makes the evidence investment legible: each study can be traced to the claims it supports, and each claim to the decision it serves.

A register is not a substitute for a protocol. It is a portfolio view. The protocol is the detailed plan for one study; the register is the map of which studies are needed and why. Teams that maintain only protocols tend to over-invest in the study in front of them and under-invest in the claim that has no study.

### Inference and its limits

The case evidence in this section supports three inferences, each with a boundary.

**Inference 1: narrow claims are easier to support.** IDx-DR and the RECOVERY subgroup result both support this. The boundary is that a narrow claim may be commercially insufficient. A product that only works in ventilated patients may not justify a broad label, and a device that only detects more than mild retinopathy may not address the full screening need. The design should support the claim the business needs, not merely the claim that is easiest to prove.

**Inference 2: performance evidence is not impact evidence.** The Epic Sepsis Model and the Apple Heart Study both illustrate the gap. The boundary is that performance evidence is still necessary. A product that cannot detect the condition cannot improve outcomes. The error is treating the performance study as sufficient, not conducting it.

**Inference 3: prespecification protects the claim.** RECOVERY's subgroup analysis was credible because it was prespecified and biologically motivated. The boundary is that prespecification cannot rescue a study with no plausible effect. A prespecified analysis of an underpowered subgroup is still underpowered.

These inferences are not laws. They are patterns from a small number of public cases, selected because they are well documented. A team should test them against its own portfolio rather than treating them as rules.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Claim has no comparator | The sentence says "improves" without "versus" | Comparator omitted to avoid an unfavorable comparison | Study cannot support a comparative claim | Force the seven-slot table at design review |
| Population is undefined | "Patients with sepsis" with no definition | Sepsis definitions vary by setting and era | Result not transferable to the intended population | Specify case definition, setting, and eligibility |
| Endpoint is a surrogate | Primary outcome is a biomarker with no linked outcome evidence | Surrogate chosen for cost or speed | Payer rejects the claim; regulator may require outcome study | Justify the surrogate with a validated link |
| Horizon is too short | 30-day outcome for a chronic disease claim | Follow-up cost | Effect missed or reversed over time | Match horizon to decision and disease course |
| Effect magnitude is invented | 30 percent reduction with no source | Marketing number carried into the protocol | Underpowered or overpowered study; claim not credible | Derive magnitude from pilot, literature, or MCID |
| Estimand is unstated | Protocol describes endpoints but not the effect of interest | Analysis-first culture | Result answers a different question | Write the five-attribute estimand before design |
| Intercurrent events ignored | Discontinuation treated as missing data | No prespecified strategy | Biased estimate; ambiguous claim | Prespecify strategy for each event |
| Analysis population mismatched | ITT used for a hypothetical estimand | Default analysis | Estimate does not match the question | Align population with estimand |
| Multiplicity uncontrolled | Multiple primary endpoints or subgroups | No hierarchy or alpha allocation | Inflated false-positive rate | Prespecify hierarchy, alpha, or false discovery control |
| External control drift | Historical control from a different era | Standard of care changed | Biased effect estimate | Contemporaneous control, sensitivity analyses |
| Alert burden ignored | High false-positive rate in pilot | Threshold optimised for sensitivity only | Clinicians ignore alerts; effect diluted | Report alerts per patient-day; prespecify threshold |
| Claim drift | Sales deck and protocol use different populations | Independent document owners | Evidence does not support the marketed claim | Single claim register with version control |
| No go/no-go rule | Programme continues regardless of results | Sunk-cost pressure | Resources wasted on an unsupported claim | Prespecify decision thresholds |

### Red flags in a protocol review

- The primary endpoint appears only in the sample-size section and not in the estimand.
- The sample size assumes an effect larger than any effect observed in the pilot.
- The comparator is described as "standard of care" with no guideline date.
- The analysis population is "all patients with complete data."
- The subgroup analyses are listed without a multiplicity statement.
- The protocol says "exploratory" for the analysis that the commercial team is already quoting.
- The claim uses "up to" or "may" in the protocol and a definite verb in the sales deck.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The claim can be stated as a testable proposition | Without it, no design can be evaluated | Claim is inherently vague or political | Seven-slot audit; clinician elicitation |
| The decision-maker is identifiable | Evidence requirements are decision-specific | Multiple decision-makers with conflicting criteria | Stakeholder map; pre-submission meetings |
| The comparator is stable over the study period | A moving comparator invalidates the contrast | Guideline change, new product launch | Guideline surveillance; contemporaneous control |
| The outcome measures the patient-relevant benefit | Surrogates may not translate | Surrogate-outcome disconnect | Validation evidence; decision-curve analysis |
| The effect magnitude is plausible | Sample size depends on it | Marketing inflation or optimistic pilot | Sensitivity table; blinded sample-size review |
| The intended population is accessible | Recruitment feasibility | Narrow eligibility, competing trials | Feasibility study; site activation data |
| The intervention can be delivered consistently | Effect depends on implementation | Workflow variation, training gaps | Fidelity monitoring; per-protocol analysis |
| Participants will adhere sufficiently | ITT effect is diluted by non-adherence | Low uptake, side effects | Pilot uptake data; adherence monitoring |
| Missing data are limited and non-informative | Complete-case analysis is biased otherwise | Differential dropout | Missing-data sensitivity analyses |
| The result will be interpreted against the prespecified rule | Post hoc interpretation invalidates the decision | Pressure to rescind the rule | Independent review; SAP lock |

### Limits of this topic

This topic teaches claim translation, not study execution. It does not cover the full sample-size mathematics (Topic 06), the details of estimand strategies (Topic 07), or the design of validation studies (Topics 04 and 08). It also does not provide legal advice. The regulatory statements are educational summaries; a submission team should verify the current text and date of every instrument it relies on.

The case evidence is drawn from public sources and is not a systematic review. The inferences about company practice are labelled as inferences. No claim in this topic should be read as an endorsement of a product or a statement about a company's internal processes.

### What this topic deliberately does not decide

Three questions are left to later topics, and the boundary is deliberate.

**How large the study must be.** This topic shows the arithmetic for one simple case and demonstrates the sensitivity to assumptions. The general treatment, including survival, cluster, and non-inferiority designs, is in Topic 06. A reader who needs a defensible sample size for a real protocol should not stop here.

**Which estimand strategy is correct.** This topic introduces the five attributes and the treatment-policy example. The choice among treatment policy, hypothetical, composite, while-on-treatment, and principal stratum depends on the clinical question and the decision, and it is developed in Topic 07. There is no universally correct strategy; there is only a strategy that matches the claim.

**How to validate a product in a new setting.** This topic names external validation as an evidence requirement. The design of validation studies, including sample size, recalibration, and transportability, is in Topics 04 and 08. A procurement committee that asks for "local validation" is asking a question this topic equips you to frame, not to answer.

The boundary is not a gap in the argument. It is a division of labour. A claim-translation review that tries to settle every methodological question becomes a protocol, and a protocol written before the claim is settled is the failure this topic exists to prevent.

### A note on evidence hierarchies

The evidence hierarchy in the module survey is a heuristic, not a ranking. This topic uses a claim-class ladder instead, because the hierarchy ranks designs while the ladder ranks what a design can support. The two can disagree. A large observational study with a credible target trial emulation may support a comparative effectiveness claim better than a small randomised trial with a surrogate endpoint. The ladder asks what the design can identify, not what label the design carries.

The practical consequence is that a team should not choose a design because it sits higher on a hierarchy. It should choose the design that identifies the estimand with the least bias and the most feasible cost. Sometimes that is a randomised trial; sometimes it is a well-designed emulation. The justification belongs in the protocol, and it should name the assumptions that make the choice credible.

## 12. Exercises

### Exercise 1 — Decompress a claim (recall and application, 30 minutes)

**Deliverable:** a seven-slot table for the sentence "Our AI triage tool reduces emergency department waiting time by 40 percent," with each slot marked explicit, implied, or absent, and a one-paragraph note on the three most consequential gaps.

**Acceptance criteria:** all seven slots present; at least three gaps identified; each gap linked to a design consequence; no slot filled with a guess presented as fact.

**Time:** 30 minutes.

### Exercise 2 — Classify and select a design (application, 25 minutes)

**Deliverable:** a table classifying five claims (prevalence of a condition, diagnostic accuracy of a test, prediction of readmission, comparative effectiveness of two drugs, mortality impact of a care pathway) by claim class, minimum design, and reporting guideline.

**Acceptance criteria:** each classification justified in one sentence; the distinction between diagnostic accuracy and predictive performance is correct; the reporting guideline matches the design.

**Time:** 25 minutes.

### Exercise 3 — Map the decision-makers (application, 30 minutes)

**Deliverable:** a stakeholder map for a new continuous glucose monitor, listing regulator, payer, hospital procurement, clinician, and patient, with the evidence each needs and the decision each makes.

**Acceptance criteria:** at least five stakeholders; each with a distinct evidence requirement; at least one conflict between requirements identified; a primary decision named.

**Time:** 30 minutes.

### Exercise 4 — Write an estimand (application, 35 minutes)

**Deliverable:** a five-attribute estimand for a trial of a smoking-cessation app, including two intercurrent events (app discontinuation and use of pharmacological aid) and the strategy for each.

**Acceptance criteria:** all five attributes present; strategies chosen from the ICH E9(R1) set; the estimand written as one sentence; the analysis population implied by the estimand stated.

**Time:** 35 minutes.

### Exercise 5 — Separate requirement from convention (critique, 25 minutes)

**Deliverable:** a two-column table classifying ten evidence-related instruments (FD&C Act 505(d), MDR, ICH E6(R3), ICH E9(R1), CONSORT 2025, SPIRIT 2025, TRIPOD+AI, STROBE, a company SOP, a journal checklist) as binding requirement, formal guidance, or convention, with the source and date.

**Acceptance criteria:** each classification correct; dates accurate; at least one instrument whose status is contested or jurisdiction-dependent flagged.

**Time:** 25 minutes.

### Exercise 6 — Size a study from a claim (calculation, 40 minutes)

**Deliverable:** a sample-size calculation for a claim of a 20 percent relative reduction in 30-day readmission from a baseline of 15 percent, alpha 0.05 two-sided, power 80 percent, 15 percent dropout, with the arithmetic shown and a sensitivity table for 10, 15, and 25 percent relative reduction.

**Acceptance criteria:** formula correct; per-arm and total numbers correct; dropout inflation applied; sensitivity table shows the direction and magnitude of the change; assumptions listed.

**Time:** 40 minutes.

### Exercise 7 — Audit an evidence package (critique, 45 minutes)

**Deliverable:** a one-page audit of a published AI validation study of your choice, identifying the claim, the evidence, the gap, and three questions a procurement committee should ask.

**Acceptance criteria:** claim stated verbatim; evidence described with design and population; gap linked to a specific missing element; questions answerable from the paper or its supplement; no generic criticism.

**Time:** 45 minutes.

### Exercise 8 — Write a remediation plan (design, 60 minutes)

**Deliverable:** a one-page plan to close the gap identified in Exercise 7, including the design, the primary endpoint, the sample size approach, the decision rule, and the estimated timeline.

**Acceptance criteria:** design matches the claim class; endpoint is patient-relevant or the surrogate is justified; sample size approach stated with assumptions; go/no-go rule present; timeline realistic.

**Time:** 60 minutes.

## 13. Self-check answers

**Answer 1 (claim decomposition).** The sentence "Our AI triage tool reduces emergency department waiting time by 40 percent" has an explicit population (emergency department patients, undefined), an implied intervention (the triage tool plus the workflow that uses it), an absent comparator (waiting time under what alternative?), an explicit outcome (waiting time, undefined as door-to-provider or door-to-disposition), an absent horizon (per visit, per shift, annual average), an explicit effect measure (relative reduction), and an explicit magnitude (40 percent). The three most consequential gaps are the comparator, the outcome definition, and the horizon. A 40 percent reduction relative to a nurse-led triage process is a different claim from a 40 percent reduction relative to no triage, and a door-to-provider reduction is different from a door-to-disposition reduction.

**Answer 2 (classification).** Prevalence is descriptive, minimum design cross-sectional, STROBE. Diagnostic accuracy is diagnostic, minimum design cross-sectional with reference standard, STARD. Readmission prediction is predictive, minimum design prospective cohort with calibration, TRIPOD+AI. Comparative effectiveness of two drugs is comparative effectiveness, minimum design pragmatic trial or target trial emulation, CONSORT or STROBE. Mortality impact of a care pathway is impact, minimum design randomised impact trial or stepped-wedge, CONSORT.

**Answer 3 (decision-makers).** The regulator needs evidence of safety and effectiveness for the intended use, usually a controlled premarket study. The payer needs comparative effectiveness and cost-effectiveness against the relevant alternative. Hospital procurement needs local validation, workflow fit, integration cost, and liability terms. The clinician needs evidence that the device changes a decision in a way that benefits the patient. The patient needs evidence of benefit, burden, and alternatives. The conflicts are predictable: the regulator may accept a surrogate that the payer rejects; procurement may require local data that the regulator does not; the clinician may value workflow benefits that the payer does not reimburse.

**Answer 4 (estimand).** "In adults who want to stop smoking, the difference in 12-month biochemically verified abstinence between a strategy of offering the app plus standard advice and standard advice alone, regardless of whether participants used the app or pharmacological aids." The intercurrent events are app discontinuation and use of pharmacological aid; both are handled by treatment policy, because the claim is about offering the intervention. The implied analysis population is ITT.

**Answer 5 (requirement versus convention).** FD&C Act 505(d), MDR, and ICH E6(R3) (where adopted) are binding requirements. ICH E9(R1), FDA non-inferiority guidance, and FDA RWE guidance are formal guidance. CONSORT 2025, SPIRIT 2025, TRIPOD+AI, STROBE, a company SOP, and a journal checklist are conventions. ICH E9(R1) is a hybrid: it is guidance, not law, but it is treated as the standard of review in the three major jurisdictions, so its practical force is close to binding.

**Answer 6 (sample size).** Baseline 15 percent, target 12 percent (20 percent relative reduction). Alpha 0.05 two-sided, power 80 percent. The quantiles are 1.959964 and 0.841621. The per-arm sample size is (1.959964 + 0.841621)^2 * [0.15*0.85 + 0.12*0.88] / (0.03)^2 = 7.849 * [0.1275 + 0.1056] / 0.0009 = 7.849 * 0.2331 / 0.0009 = 2032.9, so 2,033 per arm. With 15 percent dropout, 2,033 / 0.85 = 2,391.8, so 2,392 per arm and 4,784 total. At a 10 percent relative reduction (13.5 percent), the per-arm sample size rises to about 8,300; at 25 percent (11.25 percent), it falls to about 1,300. The exact values depend on the rounding and the quantile approximation, but the direction is the point: a smaller effect is disproportionately expensive.

**Answer 7 (audit rubric).** A strong audit states the claim verbatim, identifies the study design and population, reports the primary performance metric with a confidence interval, checks calibration and subgroup performance, and names the missing element that prevents the commercial claim. A weak audit summarises the paper, praises the AUC, and lists generic limitations such as "single centre" without linking them to the claim.

**Answer 8 (remediation rubric).** A strong plan names the claim, selects a design that can support it, specifies a patient-relevant primary endpoint or justifies a surrogate, states the sample-size approach and assumptions, defines a go/no-go rule, and gives a timeline with dependencies. A weak plan proposes "a larger study" without a design, an endpoint, or a decision rule.

## 14. Interview questions

**Question 1. What is the difference between a claim and a hypothesis?**

A strong answer: a claim is a statement the organisation wants a decision-maker to accept, usually compressed and incomplete; a hypothesis is a precise, falsifiable statement about a parameter. The job of design is to translate the claim into hypotheses and estimands that can be tested. A weak answer treats the two as synonyms or says the hypothesis is "the scientific version" without explaining the translation.

**Question 2. Who is the decision-maker for a hospital AI product, and how does that change the evidence?**

A strong answer names procurement, clinical leadership, and possibly the regulator, and explains that procurement needs local validation, workflow evidence, and cost, while the regulator needs premarket safety and effectiveness. A weak answer says "the hospital" and stops.

**Question 3. Why does the estimand come before the design?**

A strong answer: because the design must make the estimand identifiable, and different estimands imply different populations, intercurrent-event strategies, and analyses. Choosing the design first makes the estimand an accident of the method. A weak answer says the estimand is "the endpoint" or "what you want to measure" without the five attributes.

**Question 4. Which parts of ICH E9(R1) are binding?**

A strong answer: none directly, unless a jurisdiction incorporates them; E9(R1) is an ICH guideline, adopted at Step 4 in November 2019, effective in the EU from July 2020 and issued as FDA guidance in May 2021. It is treated as the standard of review, so its practical force is high, but it is not a statute. A weak answer says "it is the law" or "it is optional."

**Question 5. How would you decide between a randomised trial and a target trial emulation?**

A strong answer: randomisation if feasible and ethical, because it addresses confounding by design; emulation when randomisation is infeasible, with explicit assumptions about exchangeability, positivity, and time-zero alignment, and sensitivity analyses for unmeasured confounding. A weak answer prefers one method categorically.

**Question 6. What makes a surrogate endpoint acceptable?**

A strong answer: evidence that the surrogate is on the causal pathway and that changes in it reliably predict changes in the patient-relevant outcome, ideally from meta-analytic evidence across trials, plus a statement of residual uncertainty. A weak answer says surrogates are acceptable if they are correlated with the outcome.

**Question 7. How do you handle multiple endpoints in a confirmatory trial?**

A strong answer: prespecify a hierarchy or a family, control the family-wise error rate or false discovery rate, and label everything else exploratory. A weak answer says "use Bonferroni" without discussing the hierarchy or the claim structure.

**Question 8. What is the first thing you would ask for when inheriting a study?**

A strong answer: the protocol, the SAP, the estimand, and the decision the study is meant to support. A weak answer asks for the dataset.

**Question 9. Why can a statistically significant trial fail commercially?**

A strong answer: the effect may be too small to matter, the population may not match the market, the comparator may be obsolete, the endpoint may not be accepted by the payer, or the price may exceed the value. A weak answer says "the p-value was not significant enough" or "the market was not ready."

**Question 10. What is the smallest credible study for a claim?**

A strong answer: the smallest study that can estimate the claim's estimand with enough precision to support the decision rule, given the expected effect, the variance, the dropout, and the design constraints. It is not the smallest study that can produce a p-value. A weak answer gives a number without assumptions.

## 15. Key terms

- **Adequate and well-controlled study:** a regulatory category for a study with clear objectives, a valid comparison, a defined population, and a prespecified analysis.
- **Alpha:** the prespecified probability of a type I error.
- **Assay sensitivity:** the ability of a trial to distinguish an effective treatment from an ineffective one.
- **Claim:** a statement the organisation wants a decision-maker to accept.
- **Comparator:** the alternative against which an intervention is assessed.
- **Confirmatory evidence:** independent evidence that strengthens a single trial's conclusion.
- **Coverage with evidence development (CED):** payer coverage conditioned on further evidence generation.
- **Decision rule:** a prespecified result threshold that determines an action.
- **De Novo:** a US device pathway for novel low- to moderate-risk devices without a predicate.
- **Effect measure:** the scale on which a contrast is expressed.
- **Estimand:** the precise quantity to be estimated, defined by five attributes.
- **Estimator:** the statistical method used to estimate the estimand.
- **Evidence gap:** the difference between what a claim requires and what the evidence supports.
- **Evidence package:** the set of studies, analyses, and arguments submitted to a decision-maker.
- **Health technology assessment (HTA):** systematic evaluation of clinical, economic, and social effects.
- **Intended use:** the specific clinical purpose, population, user, and setting for a product.
- **Intercurrent event:** an event after treatment initiation that affects interpretation of the outcome.
- **Intention-to-treat (ITT):** analysis by assigned group regardless of treatment received.
- **Minimum clinically important difference (MCID):** the smallest change that matters to patients or clinicians.
- **Non-inferiority margin:** the largest acceptable loss of effect relative to an active comparator.
- **Payer:** an insurer or public programme that decides coverage.
- **Per-protocol:** analysis restricted to participants who adhered to the protocol.
- **Power:** the probability of rejecting the null hypothesis when a specified alternative is true.
- **Premarket approval (PMA):** the US pathway for high-risk devices.
- **Primary endpoint:** the outcome on which the primary claim and sample size are based.
- **Regulatory pathway:** the procedural route to market authorisation.
- **Statistical analysis plan (SAP):** the prespecified document describing analyses.
- **Substantial evidence:** the US statutory standard for drug effectiveness.
- **Surrogate endpoint:** a marker substituted for a patient-relevant outcome.
- **510(k):** US premarket notification showing substantial equivalence to a predicate device.

## 16. Sources

All URLs accessed 25 October 2026. Where a document has a version or effective date, that date is stated.

### Statutes and regulations

- United States, Federal Food, Drug, and Cosmetic Act, section 505(d). https://www.fda.gov/regulatory-information/laws-enforced-fda/federal-food-drug-and-cosmetic-act-fdc-act
- European Union, Regulation (EU) 2017/745 on medical devices. https://eur-lex.europa.eu/eli/reg/2017/745/oj
- European Union, Regulation (EU) 2017/746 on in vitro diagnostic medical devices. https://eur-lex.europa.eu/eli/reg/2017/746/oj
- European Union, Regulation (EU) No 536/2014 on clinical trials. https://eur-lex.europa.eu/eli/reg/2014/536/oj

### ICH guidelines

- ICH, E6(R3) Guideline for Good Clinical Practice, Step 4, 6 January 2025. https://database.ich.org/sites/default/files/ICH_E6%28R3%29_Step4_FinalGuideline_2025_0106.pdf
- ICH, E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials, Step 4, 20 November 2019. https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf
- ICH, E10 Choice of Control Group and Related Issues in Clinical Trials, Step 4, 20 July 2000. https://database.ich.org/sites/default/files/E10_Guideline.pdf
- ICH, M11 Clinical electronic Structured Harmonised Protocol (CeSHarP), Step 4, 19 November 2025. https://database.ich.org/sites/default/files/ICH_Step4_M11_Final_Guideline_2025_1119.pdf

### FDA guidance

- FDA, "Demonstrating Substantial Evidence of Effectiveness for Human Drug and Biological Products," revised draft guidance, 24 June 2026 (docket FDA-2019-D-4964). https://www.fda.gov/regulatory-information/search-fda-guidance-documents/demonstrating-substantial-evidence-effectiveness-human-drug-and-biological-products
- FDA, "Non-Inferiority Clinical Trials to Establish Effectiveness," final guidance, November 2016. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/non-inferiority-clinical-trials
- FDA, "Adaptive Designs for Clinical Trials of Drugs and Biologics," final guidance, December 2019. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/adaptive-design-clinical-trials-drugs-and-biologics-guidance-industry
- FDA, "Considerations for the Design and Conduct of Externally Controlled Trials for Drug and Biological Products," draft guidance, February 2023. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/considerations-design-and-conduct-externally-controlled-trials-drug-and-biological-products
- FDA, "Use of Real-World Evidence to Support Regulatory Decision-Making for Medical Devices," final guidance, December 2025 (docket FDA-2023-D-4395). https://www.fda.gov/regulatory-information/search-fda-guidance-documents/use-real-world-evidence-support-regulatory-decision-making-medical-devices
- FDA, "Considerations for the Use of Real-World Data and Real-World Evidence to Support Regulatory Decision-Making for Drug and Biological Products," final guidance, August 2023. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/considerations-use-real-world-data-and-real-world-evidence-support-regulatory-decision-making-drug
- FDA, "Conducting Clinical Trials With Decentralized Elements," final guidance, September 2024. https://www.fda.gov/media/167696/download

### Reporting guidelines

- Hopewell S, Chan A-W, Collins GS, et al. CONSORT 2025 statement: updated guideline for reporting randomised trials. BMJ 2025;389:e081123. https://doi.org/10.1136/bmj-2024-081123
- Chan A-W, Boutron I, Hopewell S, et al. SPIRIT 2025 statement: updated guideline for protocols of randomised trials. BMJ 2025;389:e081477. https://doi.org/10.1136/bmj-2024-081477
- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. BMJ 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
- Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI. BMJ 2025;388:e082505. https://doi.org/10.1136/bmj-2024-082505
- von Elm E, Altman DG, Egger M, et al. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement. Lancet 2007;370:1453-1457. https://doi.org/10.1016/S0140-6736(07)61602-X

### Case evidence

- RECOVERY Collaborative Group. Dexamethasone in Hospitalized Patients with Covid-19. N Engl J Med 2021;384:693-704. https://doi.org/10.1056/NEJMoa2021436
- Perez MV, Mahaffey KW, Hedlin H, et al. Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation. N Engl J Med 2019;381:1909-1917. https://doi.org/10.1056/NEJMoa1901183
- Wong A, Otles E, Donnelly JP, et al. External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients. JAMA Intern Med 2021;181:1065-1070. https://doi.org/10.1001/jamainternmed.2021.2626
- FDA, De Novo classification decision for IDx-DR, DEN180001, 2018. https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf

### Methods

- Hernán MA, Robins JM. Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available. Am J Epidemiol 2016;183:758-764. https://doi.org/10.1093/aje/kwv254
- Loudon K, Treweek S, Sullivan F, et al. The PRECIS-2 tool: designing trials that are fit for purpose. BMJ 2015;350:h2147. https://doi.org/10.1136/bmj.h2147

## 17. Links

- Module map: [`_index.md`](_index.md)
- Next topic: [`02_trial-designs.md`](02_trial-designs.md)
- Related topic: [`07_estimands.md`](07_estimands.md)
- Related topic: [`10_protocol-writing.md`](10_protocol-writing.md)
- Module survey file: [`../../05_study_design_evidence.md`](../../05_study_design_evidence.md)
- Labs: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)
