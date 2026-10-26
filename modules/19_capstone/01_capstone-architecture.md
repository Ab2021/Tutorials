# Capstone Architecture

> **Module 19 — Capstone, Hackathons, and Competitions**
> Topic 01 of 11
> Estimated study time: 4 hours
> Prerequisites: all core modules; Module 05 for study design; Module 14 for regulatory triage
> Version 2026.3

## 1. Why this topic exists

A capstone is the first piece of work in the programme where no single technique is the point.
The point is integration under constraints that resemble employment: a real decision, a real
owner, imperfect data, a fixed date, and several people who each hold part of the truth.
Teams that treat the capstone as an extended lab assignment produce an artefact that is
technically correct and organisationally weightless.

The characteristic failure is not a weak model. It is an inverted architecture: eleven weeks
spent improving a metric on layer two, then one week retrofitting the decision, the system,
and the governance around it.
The model cannot be rescued at that point, because the validation cohort was never chosen to
match the deployment population, the intended use was never written down, and no clinician was
ever asked what would change their behaviour.

The cost of that failure is concrete.
A hospital analytics team that cannot name the owner of a readmission score will not get the
score into a discharge workflow, so the intervention it was meant to target is never delivered.
A vendor that cannot state intended use cannot classify its product, so the regulatory pathway
is guessed late and the clinical evidence is designed for the wrong claim.
A research group that validates on a random split reports a number that will not survive the
first prospective contact with a new site.

Industry has a name for the discipline that prevents this: architecture.
Architecture here does not mean software architecture, although that is part of it.
It means the deliberate arrangement of four layers, in the order in which evidence accumulates:
the decision, the evidence, the system, and the governance.
Each layer constrains the one above it.
A decision with no capacity for action caps the value of the best evidence.
Evidence collected on the wrong population caps the credibility of the best system.
Governance that arrives last caps every claim the team is allowed to make.

Regulators have arrived at a compatible position from a different direction.
The United States Food and Drug Administration (FDA) guidance on clinical decision support (CDS)
software, finalised in September 2022, turns on intended use rather than on algorithm class.
The TRIPOD+AI statement (Transparent Reporting of a multivariable prediction model for Individual
Prognosis Or Diagnosis, AI extension, 2024) requires intended use, target population, and
prediction horizon to be reported before results.
DECIDE-AI (Developmental and Exploratory Clinical Investigations of DEcision support systems
driven by Artificial Intelligence, 2022) adds the human-in-the-loop workflow as a reporting item.
None of these is a statute applied to coursework, but each encodes the same operational lesson:
a model is a component of a decision system, and describing it without the system is incomplete.

Three audiences read the capstone, and each reads a different layer.
The clinician reads layer 1 and asks whether the action is one they would actually take.
The engineer reads layer 3 and asks whether the pipeline can run at the moment of decision.
The compliance officer reads layer 4 and asks what claim is being made and on what authority.
A team that prepares only for the clinician's question will fail the other two, and a team that
prepares only for the compliance question will produce a defensible artefact that nobody wants.
Architecture is the discipline of making all three readings consistent with one another.

Consistency is harder than it sounds because the layers use different vocabularies for the same
objects.
The clinician says "the patient is deteriorating".
The modeler says "the label is sepsis onset within six hours".
The regulatory memo says "the intended use is triage support for adult inpatients".
The monitoring plan says "the alert fires when the score exceeds 0.18".
If those four sentences describe the same event, the architecture is coherent. If any one drifts,
the dossier contains a contradiction that a panel will find.

This topic gives the capstone its skeleton.
The output is a one-page architecture statement that names the four layers, the twelve-week
shape, and the three scope tiers.
Later topics populate it: the milestone plan in Topic 02, the roles in Topic 03, the archetype
in Topic 04, and the brief in Topic 05.
A team that skips this topic will still produce files, but the files will not cohere.

## 2. Learning outcomes

The outcomes are written as observable actions because the assessment is a defense, not an exam.
Each one maps to an exercise in Section 12 and to a question a panel is likely to ask.
If you cannot perform the action in front of another person, you have not met the outcome.

By the end of this topic you can:

1. **Draw** the four-layer capstone architecture from memory and explain which layer constrains
   which, with one example of a constraint propagating upward.
2. **Write** a decision statement of the form "on [event], [role] decides [action] for
   [population]", and test it against three criteria: one action, one owner, one frequency.
3. **Compute** the operating characteristics of a proposed alert or score, including alert
   volume per day, positive predictive value at deployment prevalence, and the precision the
   available cohort can actually support.
4. **Select** a validation level from the L0 to L4 ladder and justify the choice against the
   claim being made rather than against the data that happen to be available.
5. **Allocate** the twelve weeks across layers and gates, and identify which two weeks may never
   be compressed.
6. **Classify** each planned deliverable into core, stretch, or out of scope, and defend one
   exclusion to a panel.
7. **Diagnose** a doomed capstone from two cheap signals before any modelling begins.
8. **Produce** an architecture statement that a clinical owner, a data steward, and a regulatory
   reviewer can each sign without qualifying their signature.

Outcomes 1 to 3 are assessed in Exercises 1 to 3 and revisited in the midpoint review.
Outcomes 4 and 5 are assessed in Exercises 4 and 6 and tested directly at G2.
Outcomes 6 and 7 are assessed in Exercises 7 and 8 and appear in the defense as the
"what would make you stop" question.
Outcome 8 is the charter itself, which is signed at G0 and re-signed if the scope changes.

The mapping matters because a capstone grade is not a single number. A team can be strong on
outcomes 1 to 3 and weak on 4 and 5, which produces a well-designed model with an indefensible
validation claim. The defense is structured to detect that combination.

## 3. The core mental model

A capstone has four layers and a clock.
The layers are ordered by dependency, not by importance.
The clock is twelve weeks, with five gates that convert work into decisions.

```text
        WEEK  1    2    3    4    5    6    7    8    9   10   11   12
             |    |    |    |    |    |    |    |    |    |    |    |
  L4 GOVERN  [G1 governance]..............................[G4]  |
  L3 SYSTEM            [pipeline]............[serving].........| freeze
  L2 EVIDENCE                 [baseline][plan][candidate][L2/L3]|
  L1 DECISION[G0]................................................|
             |         G2: split+metric frozen     G3: go/re-scope/stop
             v
   decision -> evidence -> system -> governance
   (each layer constrains the credibility of the layer above)
```

Read the diagram from the bottom.
Layer 1 is the decision: an action, an owner, a frequency, a comparator, and the cost of each
kind of error.
Layer 2 is the evidence: a cohort, a split that mimics deployment, a baseline, a candidate, and
metrics that include calibration and subgroup performance.
Layer 3 is the system: the pipeline that turns raw data into an output at the moment the decision
is made, with a latency budget and a monitoring plan.
Layer 4 is governance: data rights, regulatory classification, risk management, and equity.

The arrows run upward because each layer supplies the constraints for the next.
The owner named at layer 1 determines the capacity number that sets the operating threshold at
layer 2.
The deployment setting named at layer 1 determines which split is honest at layer 2 and which
latency budget applies at layer 3.
The claim made at layer 2 determines which regulatory regime applies at layer 4.

The clock matters because evidence has a cost curve that is not linear.
Weeks 1 and 2 are cheap and determine most of the outcome: a decision statement, an owner, a
data-access request, a fallback.
Weeks 3 to 6 are where the baseline and the analysis plan are fixed; after gate G2, changes are
allowed but must be logged, and the log is read at the defense.
Weeks 7 to 10 are where robustness and governance are built, and where most teams discover that
they needed a clinician in week 1.
Weeks 11 and 12 are freeze and defense; nothing substantive can be repaired there.

Two properties of the model are worth memorising.

**Coupling.** A change in one layer propagates.
If the owner says the review capacity is four patients per shift, the threshold must produce
roughly four alerts per shift; that fixes sensitivity and specificity, which fixes the metric,
which fixes the validation design.
A team that changes the threshold in week 10 without revisiting capacity has changed the decision
without telling the owner.

**Irreversibility.** Some choices are cheap to reverse and some are not.
The model family is cheap to change in week 6.
The split is expensive to change after G2 because every reported number becomes suspect.
The data-access route is expensive to change after week 4 because credentialing takes weeks.
The intended use is expensive to change after week 9 because the regulatory memo and the model
card both depend on it.
Architecture is mostly the art of deciding which choices to make early because they are
irreversible, and which to defer because they are not.

A third property is easy to miss: the layers are not equally visible to a panel.
A panel sees the demo and the slides, which are layer 3 and layer 2 artefacts.
It infers layer 1 from the first two minutes of the pitch and layer 4 from one or two questions.
That asymmetry is why weak teams over-invest in the demo.

### Reading the diagram in the other direction

The upward arrows are constraints; the downward arrows are dependencies.
Once the evidence exists, it also constrains what the team is allowed to claim at layer 1.
If site-held-out performance is materially worse than temporal performance, the decision
statement cannot include the new site, even though the sponsor wants it to.
The architecture is therefore bidirectional in practice: it is designed top-down and revised
bottom-up after each gate.

A concrete propagation example makes the coupling visible.
The nurse manager says capacity is four assessments per shift, with three shifts per day, so
twelve per day.
The team computes that a threshold producing twelve alerts per day at 3.1 percent prevalence
corresponds to an alert rate of about 0.29 percent per admission, which is far below what the
model can achieve at useful sensitivity.
The arithmetic shows the capacity constraint and the detection ambition are incompatible: at
twelve alerts per day, the model can capture only a small fraction of the 130 annual cases.
The team must either raise capacity, accept lower sensitivity, or change the decision from
"assess first" to "assess if the score is high and the nurse has capacity", which is a different
workflow with a different comparator.

That is what architecture looks like in practice.
It is not a diagram on a slide; it is the moment a number at one layer invalidates a plan at
another, and the team notices before building anything.

### The three scope tiers as an architectural control

Core, stretch, and out are not a prioritisation exercise; they are a contract.
Core is what the defense requires. Stretch is what the team may attempt after freeze, which in
practice means it is not attempted. Out is what the team promises not to do, and naming it
protects the core.
A team that lists twelve core items has no core. A team that lists three core items, two stretch
items, and five exclusions has a plan that can survive a bad week.

## 4. Definitions and vocabulary

Vocabulary drift is the most common cause of a dossier that contradicts itself.
The same word means different things to a clinician, a statistician, and a regulator, and each
believes the meaning is obvious.
The table fixes the meanings used in this module.
Where a term has a legal definition, the legal definition wins and the table says so.

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Capstone architecture | The ordered arrangement of decision, evidence, system, and governance layers with their gates and scope tiers | Used to mean software architecture only | Four layers, five gates |
| Decision statement | One sentence naming an event, an action, an owner, and a population | A topic area ("sepsis prediction") stated as if it were a decision | One action per statement |
| Decision owner | The named role or committee that acts on the output and is accountable for the consequence | "The clinical team" or "stakeholders" | Named role, not a group |
| Comparator | Current practice measured on the same cohort and time window as the candidate | A strawman such as "no model at all" when practice already has a score | Same cohort, same outcome definition |
| Intended use | The population, setting, inputs, output, user, and contraindications for which the artefact is claimed to be valid | Written after results, to match what the model happened to do | One paragraph, versioned |
| Minimum viable evidence package | The twelve artefacts without which the capstone does not pass, regardless of metric | Treated as a stretch goal | Fixed list, see Section 5 |
| Validation level | L0 random split, L1 patient-level holdout, L2 temporal, L3 site-held-out, L4 prospective or shadow | Claiming L2 from a random split with a date column present | L0 to L4 |
| Gate | A scheduled review at which work becomes a decision, with a named reviewer and a pass condition | A status meeting with no authority to stop work | G0 to G5 |
| Kill criterion | A pre-specified result that ends the project or the claim | A vague intention to "reassess" | Threshold on a named metric |
| Fallback A | A pre-approved public or demo dataset with the same schema as the primary request | Discovered in week 5 after access fails | Named dataset and licence |
| Fallback B | A synthetic generator calibrated to published summary statistics, labelled synthetic everywhere | Presented as if it were real cohort evidence | Generator version and calibration source |
| Scope tier | Core (cannot slip), stretch (only after freeze), out (named exclusion) | Everything declared core, so nothing is | Tier label per deliverable |
| Alert burden | Alerts generated per unit time at the deployment prevalence and threshold | Reported as a rate on a balanced test set | Alerts per day or per shift |
| Positive predictive value (PPV) | Proportion of flagged cases that are true positives at deployment prevalence | Confused with precision on a case-control sample | Proportion, 0 to 1 |
| Operating point | The threshold and the sensitivity/specificity pair it produces | Chosen to maximise a metric with no capacity constraint | Threshold plus (sens, spec) |
| Capacity | The number of actions the owner can take per unit time | Assumed infinite, or omitted | Actions per shift or per day |
| Time zero | The moment at which prediction inputs are fixed and the outcome clock starts | Left implicit, allowing post-outcome features | Timestamp and definition |
| Data tier | Open, credentialed, restricted, or synthetic, each with different access obligations | "Public" applied to data that require registration | Tier plus licence identifier |
| De-identification | The documented process that removes or transforms identifiers under a named standard | Assumed because the file has no name column | Standard and method |
| Software as a Medical Device (SaMD) | Software intended for a medical purpose that is not part of a hardware device | Applied to any health-adjacent calculator | Regulatory classification |
| Predetermined change control plan (PCCP) | A pre-authorised description of how a model may be updated, with a protocol and acceptance criteria | Written after the first retrain | FDA guidance, December 2024 |
| Estimand | The precise quantity a study estimates, including population, variable, and handling of intercurrent events | Treated as a synonym for endpoint | ICH E9(R1) structure |
| Evidence sufficiency | The point at which the confidence interval is narrow enough for the owner's decision | "We have a significant result" | Minimum detectable difference |
| Freeze | The moment after which artefacts change only for factual correction, announced on stage | A hope that work stops | Date and version tag |
| Dossier | The versioned folder of thirteen artefacts submitted at G5 | A slide deck | Repository plus PDF index |
| Institutional review board (IRB) | The body that reviews and approves research involving human subjects, or issues a determination that the work is not human-subjects research | Assumed unnecessary because data are retrospective | Approval number or determination letter |
| Protected health information (PHI) | Individually identifiable health information held by a covered entity or business associate | Treated as a synonym for "clinical data" | 18 identifier categories under HIPAA |
| Minimum necessary | The privacy principle that only the data needed for the stated purpose are accessed | Used to justify pulling every field "in case" | Field list mapped to purpose |
| Calibration | Agreement between predicted probabilities and observed frequencies | Confused with discrimination | Slope, intercept, reliability plot |
| Discrimination | Ability to rank cases above non-cases | Reported as if it implied calibration | AUROC, AUPRC |
| Net benefit | Decision-analytic value of using a model at a threshold, in units of true positives per patient | Reported without stating the threshold range | Proportion, dimensionless |
| Subgroup | A pre-specified population defined by a characteristic with a plausible harm or access mechanism | Any post-hoc split that produces an interesting difference | Group definition and event count |
| Model card | Short structured document describing intended use, training data, metrics, and limits | A paragraph in the appendix | Standard sections |
| Datasheet | Structured description of a dataset's motivation, composition, collection, and recommended uses | Confused with a data dictionary | Standard sections |
| Data provenance | Source, version, access date, licence, extraction query, and transformations | "From the EHR" | Version identifier and date |
| Shadow mode | Running a model on live data without showing the output to users | A demo on historical data | Duration and exposure count |
| Lockfile | A pinned record of exact dependency versions | A requirements file with loose ranges | Hash and version list |
| Contribution log | Weekly record of who produced which artefact | A team grade | Hours and artefact links |

Two conventions are worth stating because they cause repeated confusion.
First, "validation" in this module always means an evaluation on data not used for development,
and the level (L0 to L4) qualifies it. A team that says "we validated the model" without a level
has said nothing.
Second, "deployment" means the output reaches a user who acts on it. A notebook that scores a
held-out file is not deployment, and a shadow-mode run is deployment without action.

## 5. The mechanics, step by step

Architecture is a procedure with fourteen steps, run across the first two weeks and revisited at
each gate.
Each step names its input, its output, and the reason it cannot be skipped.
Steps 1 to 4 are completed before any data are requested.
Steps 5 to 9 fix the evidence design.
Steps 10 to 12 fix the delivery and governance obligations.
Steps 13 and 14 set the clock and the freeze.

### Step 1. Choose the archetype before the dataset

**Input:** a problem area, a sponsor, and whatever data are visible.

**Output:** one archetype label from the eight in Topic 04, with a one-sentence justification.

**Rationale:** The archetype determines the evidence bar, the characteristic failure, and the
regulatory angle. A team with imaging data and no clinical owner should select the evaluation or
benchmark archetype, not the detection archetype, because the detection archetype requires a
reader-workflow claim that nobody in the team can own. Choosing by data availability is the most
common architectural error, and it is cheap to avoid at this step.

**Branch point:** if two archetypes fit, choose the one with the stricter evidence bar only if
the team can reach it in twelve weeks. Otherwise choose the narrower claim and say what would
have been needed for the stronger one.

### Step 2. Write the decision statement

**Input:** the archetype, the sponsor's complaint, and one conversation with a user.

**Output:** one sentence: "On [event], [role] decides whether to [action] for [population]."

**Rationale:** The sentence forces three commitments that later steps depend on: a trigger, an
owner, and an action. It also exposes the difference between a prediction and a decision. A model
that outputs a risk score is not a decision; the decision is what the owner does with the score.

**Branch point:** if the sentence needs "and" to describe the action, there are two decisions.
Split them, pick one for the capstone, and list the other as out of scope.

### Step 3. Name the owner and obtain a capacity number

**Input:** the decision statement.

**Output:** a named role, a channel by which they receive the output, and a number: how many
actions per shift, day, or week they can take.

**Rationale:** Capacity converts a ranking into an operating point. Without it, the team cannot
choose a threshold, cannot compute alert burden, and cannot say whether the model is usable. The
capacity number is also the cheapest early diagnostic of a doomed project: if nobody can supply
it, nobody owns the decision.

**Branch point:** if the owner is a committee, the capacity is the committee's agenda slots per
month, which is usually far smaller than a shift-level number. Record the smaller number.

### Step 4. Specify comparator and error economics

**Input:** the decision statement and the capacity number.

**Output:** current practice described in one paragraph, its measured performance on a comparable
cohort if available, and a sentence for each error type: what a false positive costs, what a
false negative costs, and who bears each cost.

**Rationale:** The comparator sets the bar the candidate must beat, and the error economics set
the threshold region worth considering. A false positive that costs twelve minutes of review and
a false negative that costs a missed deterioration are not symmetric, and no single metric
encodes that asymmetry without a decision analysis.

**Branch point:** if current practice has never been measured, the capstone includes measuring
it. That measurement is often the most valuable output, and it is legitimate evidence even if
the candidate model underperforms.

### Step 5. Fix the population and time zero

**Input:** the clinical pathway and the data dictionary.

**Output:** inclusion and exclusion criteria with counts, a flow diagram from all candidate
records to the eligible cohort, and a timestamp definition for time zero.

**Rationale:** Population and time zero determine prevalence, case mix, and which features are
legally available at prediction time. Most leakage enters here, not in the model. A feature
recorded after the decision time, such as a lactate drawn because the alert fired, will make the
model look excellent and useless.

**Branch point:** if exclusions remove more than about 20 percent of records, test whether they
are informative. Excluding transfers may remove the sickest patients and change the decision.

### Step 6. Choose the data tier and file access in week 1

**Input:** the population definition and a list of candidate sources.

**Output:** a primary request with a submission date, fallback A with its licence checked, and
fallback B with a generator version.

**Rationale:** Credentialing, data-use agreements (DUAs), and institutional review board (IRB)
determinations take weeks and are outside the team's control. Filing in week 1 is the only
reliable way to have data by week 4. The fallback must preserve the scientific question; if it
does not, the brief is wrong and should be changed now.

**Branch point:** if the primary source is restricted and the institution has no executed
agreement, assume the primary will not arrive and plan the twelve weeks on fallback A. Announce
the assumption at G1.

### Step 7. Fix the validation level against the claim

**Input:** the intended deployment setting and the claim language.

**Output:** one level from the ladder, with a sentence explaining what the level does and does
not prove.

**Rationale:** The level is a property of the claim, not of the data. A claim about a new site
requires site-held-out validation; a claim about next year requires temporal validation. A
random split proves only that the pipeline runs, which is why L0 is labelled debugging only.

**Branch point:** if the claim requires L3 and only one site is available, narrow the claim to
L2 and write the limitation into the model card. Do not describe an L2 result as external
validation.

### Step 8. Specify the primary metric and the decision rule

**Input:** the capacity number, the error economics, and the validation level.

**Output:** one primary metric with a pre-specified threshold, one secondary metric for
calibration or utility, and the decision rule that maps output to action.

**Rationale:** The metric must be computable at the operating point the owner can staff.
For a fixed-capacity queue, recall at k or precision at k is appropriate; for risk-based
outreach, calibration and net benefit; for a rule-out test, sensitivity at a fixed specificity.
Area under the receiver operating characteristic curve (AUROC) alone answers a question nobody
asked: how well the model ranks across all thresholds.

**Branch point:** if the owner cannot state a threshold, use decision curve analysis to show net
benefit across a range and ask the owner to choose a region. Record the choice.

### Step 9. Pre-specify the analysis plan before candidate results

**Input:** steps 5 to 8.

**Output:** a two-page plan with split definition and dates, metric and threshold, at least three
subgroups, missing-data handling, confidence interval method, multiplicity plan, sensitivity
analyses, and the stopping rule.

**Rationale:** The plan is the defence against choosing the analysis after seeing the result.
It is also the artefact a biostatistics mentor can review in twenty minutes, which is the
highest-leverage review in the project.

**Branch point:** if a planned subgroup has fewer than about 30 events, pre-specify that the
result will be reported with an explicit warning rather than dropped. Dropping it silently is
the failure mode the equity reviewer is looking for.

### Step 10. Design the system layer to the decision latency

**Input:** the workflow map and the decision frequency.

**Output:** a pipeline diagram from source to output, the latency budget, the interface by which
the owner sees the output, and the fallback behaviour when the pipeline fails.

**Rationale:** A model that takes four hours to score cannot support a decision made in fifteen
minutes. The latency budget also constrains model size and infrastructure, which constrains the
compute request. Fail-safe behaviour matters more than accuracy in clinical settings: a silent
failure is worse than a visible abstention.

**Branch point:** if the decision is asynchronous, such as a daily list, batch scoring is
acceptable and cheaper. If it is synchronous, the system needs a service, a queue, and an
on-call owner.

### Step 11. Triage the regulatory path in week 2, not week 9

**Input:** the intended use statement, the output type, and the user.

**Output:** a one-page memo naming the regime, the classification, the binding artefact, and
three open questions for a regulatory mentor.

**Rationale:** The regime determines what evidence is admissible and what claims are permitted.
Under the FDA CDS guidance, software that analyses a medical image or signal, or that provides a
specific treatment directive or a time-critical alert, remains a device. That classification
changes the evidence package, so it cannot be deferred.

**Branch point:** if the product is intended for clinicians in a jurisdiction with a device
regime, the memo must say so even when the capstone will never be submitted. Claiming
"research only" while pitching a clinical workflow is the failure the panel punishes.

### Step 12. Write the governance and equity plan

**Input:** the data tier, the population, and the intended use.

**Output:** a risk register with owners and triggers, a de-identification record, an access list,
and at least three pre-specified subgroups with the direction of expected bias.

**Rationale:** Governance is not a compliance formality; it is the set of constraints under
which the evidence is interpretable. A subgroup analysis planned after the results is a fishing
expedition; one planned in week 2 is a commitment.

**Branch point:** if a subgroup cannot be estimated, report the count and the interval that
would be needed, and state the likely direction of harm. Absence of evidence is a finding.

### Step 13. Set the twelve-week clock and the gates

**Input:** the scope tiers and the team's availability in hours per week.

**Output:** a week-by-week table with one deliverable per gate, a named reviewer, and a pass
condition; plus the effort arithmetic showing the plan fits the available hours.

**Rationale:** A plan that assumes forty hours per person per week across twelve weeks is not a
plan; it is a wish. The arithmetic in Section 6 shows how quickly a plausible team exceeds its
capacity once review, writing, and rework are counted.

**Branch point:** if the arithmetic does not fit, cut a stretch goal, not a gate. Gates 0, 1, 2,
and 5 are not compressible.

### Step 14. Define freeze and the definition of done

**Input:** the dossier requirements and the defense date.

**Output:** a freeze date, a version tag, and a one-sentence definition of done that a stranger
could test.

**Rationale:** Freeze converts a project into a defensible artefact. Without it, the numbers in
the slides will not match the numbers in the repository, and the first panel question will
expose it.

**Branch point:** if a factual error is found after freeze, correct it and announce the
correction on stage. A disclosed correction costs less than a discovered inconsistency.

### Starting late: the three-week compression

Some teams begin in week 3 because a preceding module ran long, a placement changed, or a team
member joined late. The compression rule is not "do the same work faster". It is "drop a layer of
ambition, not a gate".

The compressed sequence is: week 3 becomes two days of decision and governance, with G0 and G1
held together; week 4 becomes the data audit and baseline; week 5 becomes the analysis plan and
the first candidate; weeks 6 to 9 run as normal; weeks 10 to 12 are unchanged.
What is lost is the second candidate, the external site, and the value-case sensitivity analysis.
Those are stretch items, which is exactly why the tiers exist.

| Week lost | What can be compressed | What cannot | Consequence to accept |
|---|---|---|---|
| Week 1 | Charter drafted in a single two-hour session | Owner interview | Higher risk of a wrong decision statement |
| Week 2 | Governance memo drafted from a template | DUA filing and IRB submission | Data may arrive late; fallback may be needed |
| Week 3 | Data audit scoped to the fields the plan needs | Provenance and missingness review | Leakage risk higher; screen twice |
| Week 4 | Baseline reduced to one model | Locked split and intervals | Fewer comparisons; claim narrower |
| Week 6 | Candidate tuning capped at a fixed trial budget | Error analysis | Weaker error taxonomy |

The compression table is a negotiation aid, not a permission slip.
A team that loses weeks 1 and 2 should expect to defend a narrower claim, and should say so at
the midpoint rather than at the defense.

### What each gate actually tests

G0 tests whether the project is a decision.
The panel asks one question: "Who acts, and what do they do differently?" A vague answer fails
the gate regardless of the data.

G1 tests whether the team can legally and practically obtain data.
The panel asks for the agreement number, the de-identification method, and the fallback.
An approval "in progress" is not a pass unless a fallback is approved and the claim survives it.

G2 tests whether the evidence design was fixed before results.
The panel reads the analysis plan timestamp against the first results commit.
A plan written after the first result is a description, not a pre-specification.

G3 tests whether the evidence supports continuing.
The panel makes an explicit recommendation: go, re-scope, or stop. "Continue and see" is not a
recommendation and is recorded as a fail.

G4 tests whether the claim matches the governance.
The panel reads the intended use against the model card and the regulatory memo, and looks for
any sentence in the slides that is stronger than the evidence.

G5 tests whether the artefacts are consistent and reproducible.
The panel runs the one-command reproduction and compares the output with the dossier. A mismatch
is a factual error and must be announced.

### Decision table: what to do when a gate is not met

| Gate | Condition observed | Options | Recommended action | Who decides |
|---|---|---|---|---|
| G0 | No owner can be named | Reframe as methods project; change brief; stop | Change brief now, before data work | Program lead with clinical mentor |
| G1 | Primary data not accessible | Fallback A; fallback B; narrow claim to public data | Activate fallback A and narrow the claim in writing | Data steward with team lead |
| G2 | Split or metric changed after seeing results | Log and continue; revert; re-run on untouched holdout | Log the change, revert if a holdout remains | Biostatistics mentor |
| G3 | Site-held-out performance below threshold | Narrow claim to L2; re-scope to subgroup; stop | Narrow the claim and update intended use | Full panel |
| G4 | Regulatory path unclear | Name the regime with open questions; restrict claim; stop | Restrict claim to research use and say so | Regulatory mentor |
| G5 | Artefacts inconsistent at freeze | Correct and announce; delay defense; submit as draft | Correct, announce, and version-tag | Program lead |

### Edge cases and forced decisions

**Data access denied late.** If the primary source is refused in week 6, the project has four
weeks of wasted modelling and six weeks left. The correct response is not to renegotiate but to
switch to fallback A, re-run the pipeline on the fallback schema, and rewrite the intended use to
the fallback population. The claim narrows; the evidence remains honest.

**A team member drops out.** Reallocate by artefact, not by sentiment. The evidence lead's work
cannot be absorbed by the modeler in week 8, so the lead takes the regulatory memo and the
modeler keeps validation. Reduce the stretch tier to zero the same day.

**Scope creep.** A new question after week 6 is parked in a backlog with a date and a name. It is
not evaluated, because evaluation is the cost. If the owner insists, the team asks which core
artefact will be dropped to pay for it.

**Negative result.** A candidate that fails to beat the comparator is a legitimate capstone
outcome if the comparator was measured properly and the analysis plan was followed. The defense
then argues that the decision should not change, quantifies the evidence against deployment, and
shows what would have been needed. Panels reward this more than an overfit win.

**Cannot recruit a clinician.** Without a clinical liaison, the decision statement is a guess.
Options are a documented need assessment from published workflow studies, a retired clinician
advisor, or a change of archetype to evaluation science. Do not simulate a clinician's judgement
and call it validation.

**Compute exhausted.** If the training budget runs out before the candidate is tuned, the
baseline is the result. Report it, report the compute used, and describe the candidate that was
planned. A reproducible baseline with honest limits outranks an unreproducible partial candidate.

**Submission deadline.** The deadline is a gate, not a negotiation. The freeze date exists so
that the last 72 hours produce rehearsal, not modelling. Teams that model until the deadline
submit slides that do not match their repository.

## 6. Worked example

A four-person team at a 420-bed community hospital proposes a sepsis early-warning capstone.
The sponsor is the chief medical officer; the clinical liaison is an intensivist with four hours
per week; the data steward has offered a de-identified extract from the electronic health record
(EHR) under an institutional DUA.

### Week 1, gate G0: the charter

The team's first draft says: "Build a machine-learning model to predict sepsis and improve
outcomes." It fails all three tests: no action, no owner, no frequency. The rewrite is:

> On each inpatient shift, the rapid-response nurse reviews the ranked list and decides which
> four patients to assess first, among non-intensive-care inpatients with suspected infection.

The action is a review; the owner is the rapid-response nurse; the frequency is per shift; the
capacity is four assessments per shift.

The eligible cohort is medical and surgical inpatients with a suspected-infection order and no
prior intensive-care stay in the current admission. At the site this yields 4,200 admissions per
year. Sepsis prevalence in the eligible cohort is 3.1 percent, so the site sees about 130 cases
per year.

The comparator is the existing modified early-warning score, which the team measures on the same
cohort in weeks 3 and 4. The team expects, from a local audit, sensitivity of about 0.45 at a
threshold producing six alerts per day.

### The arithmetic that changes the design

The team's candidate, at a plausible operating point, has sensitivity 0.72 and specificity 0.93.
At 3.1 percent prevalence:

- true-positive alert rate = 0.72 × 0.031 = 0.0223;
- false-positive alert rate = 0.07 × 0.969 = 0.0678;
- total alert rate = 0.0901, or 9.0 percent of admissions;
- alerts per year = 0.0901 × 4,200 = 378.6, or 1.04 per day;
- positive predictive value = 0.0223 / 0.0901 = 0.248;
- cases detected per year = 93.7 of 130.2.

Two conclusions follow immediately.
First, the alert volume is manageable: about one per day, twelve minutes of review each, so
12.4 minutes per day of added work. That is well inside the four-per-shift capacity.
Second, three of four alerts are false, which is acceptable only if the review is cheap and
safe. If the review triggered a cascade of cultures and antibiotics, the harm calculation would
change the threshold.

The team then tests whether its validation cohort can support the claim. With 130 cases per
year, a held-out year gives about 130 positive cases; observed sensitivity of 94/130 = 0.723 has
a 95 percent Wilson interval of 0.641 to 0.793, a width of 0.152. That is too wide to distinguish
the candidate from a comparator at 0.65.
For a half-width of 0.07 around 0.72, the team needs 158 positive cases, which at 3.1 percent
prevalence is 5,098 admissions, or 14.6 months of site data. A half-width of 0.05 needs 310
positives, 9,993 admissions, and 28.6 months.

This is the first diagnostic that the project as written is doomed: a single-site, twelve-week
capstone cannot produce a sensitivity estimate precise enough to support a shift-level claim.
The design must change, not the ambition.

### Diagnostic one: no capacity number, or a capacity number nobody will own

At G0 the team asks the rapid-response nurse manager: "How many extra assessments per shift can
you absorb?" The answer is "it depends on staffing." Pressed, the manager says two on nights,
four on days.
A team that accepts "it depends" and proceeds will later choose a threshold that produces six
alerts per shift, and the alert will be ignored within a month.
The diagnostic is cheap: if the owner cannot give a number, or gives a range wider than a factor
of three, the decision is not operational yet.

### Diagnostic two: the cohort cannot support the precision the claim needs

The second diagnostic is arithmetic, not judgement. Divide the available positive cases by the
number needed for the target half-width.
At 130 positives against a requirement of 158, the site is 18 percent short for a half-width of
0.07, and 58 percent short for 0.05.
A team that proceeds will report an interval that spans the comparator and will be unable to say
whether the model is better.

### The revision: vary one assumption

The team cannot create more sepsis cases, but it can pool sites. Assume a second hospital of
similar size joins, doubling admissions to 8,400 per year and positives to 260.
At the same prevalence, 158 positives now require 0.73 years, or 8.8 months, which fits inside a
twelve-week project only if historical data are available immediately.
The team negotiates a two-year retrospective extract across both sites, giving 260 positives and
a half-width of about 0.055, and reserves the most recent six months as a temporal holdout.

The arithmetic also changes the claim. With two sites, the team can attempt site-held-out
validation at L3 for one site and temporal validation at L2 for the other. The intended use
narrows to "adult inpatients in community hospitals with a suspected-infection order", and the
model card records that critical-access hospitals and pediatric patients are out of scope.

Now vary prevalence instead. If the eligible cohort is broadened to all admissions, prevalence
falls to 1.55 percent. Then:

- alert rate = 0.72 × 0.0155 + 0.07 × 0.9845 = 0.0112 + 0.0689 = 0.0801, or 8.0 percent;
- alerts per year = 336.3, or 0.92 per day;
- PPV = 0.0112 / 0.0801 = 0.139;
- cases detected = 46.9 of 65.1;
- 158 positives require 10,194 admissions, or 29.1 months.

Broadening the population lowers alert volume slightly but halves PPV and doubles the data
requirement. The narrowed cohort was the better architecture, and the team can now show why with
three numbers.

### Effort and budget arithmetic

The team has five roles at the following weekly hours: lead 8, data engineer 12, modeler 14,
evidence lead 8, clinical liaison 3. That is 45 hours per week, or 540 hours across twelve weeks.
At blended rates of 65, 55, 60, 60, and 120 dollars per hour respectively, labour is 34,320
dollars. Compute for the candidate and the sensitivity analyses is budgeted at 900 dollars.
The total is 35,220 dollars, or 2,935 dollars per week.

The effort arithmetic exposes a second problem: 540 hours is roughly 13.5 full working weeks
for one person, spread across five people who have other obligations. Review cycles, writing,
and rework typically consume 30 to 40 percent of a project's hours, so the plan has to reserve
about 160 hours for them. That leaves 380 hours for analysis and engineering, which is enough for
a baseline, one candidate, and a disciplined validation, and not enough for a foundation-model
comparison as well.

### Gate G1: governance

At G1 the team presents the DUA, the de-identification method, the access list, and the risk
register. The data steward notes that the extract contains admission timestamps, which are
permitted under the agreement but must not appear in any figure. The team adds a rule: all
figures use relative days from admission.
The regulatory mentor notes that a shift-level ranked list with a time-critical alert is likely
a device function under the FDA CDS guidance, because it provides a time-critical alert and does
not require the user to review the basis of the recommendation. The memo records this and lists
the open question: whether the intended user is the rapid-response nurse or the attending.

The panel passes G1 with two conditions: the claim is narrowed to decision support with a named
user, and the team must report alert volume per shift rather than per day, because staffing is
per shift.

### Budget sensitivity

The budget is small enough that one line dominates: labour.
A 20 percent error in hours is 6,864 dollars, while the entire compute budget is 900 dollars.
That asymmetry should drive management attention: negotiating an extra GPU is not the lever;
protecting team hours from meetings and rework is.

| Line | Base case | If a member halves availability | If compute doubles | If a clinician joins at 3 h/week |
|---|---:|---:|---:|---:|
| Team hours | 540 | 486 | 540 | 576 |
| Labour (USD) | 34,320 | 29,640 | 34,320 | 36,480 |
| Compute (USD) | 900 | 900 | 1,800 | 900 |
| Total (USD) | 35,220 | 30,540 | 36,120 | 37,380 |
| Cost per week (USD) | 2,935 | 2,545 | 3,010 | 3,115 |

The second column is the one teams forget. When a member halves availability, the project does
not cost less; it delivers less for nearly the same weekly burn, because the fixed costs of
coordination remain. The correct response is to cut a stretch item the same week, not to
absorb the loss silently.

### What the panel asked at G1

The panel asked four questions, and the answers are worth recording because they are typical.

"Who is the intended user?" The team had written "clinicians". The panel rejected it and asked
for the role that receives the list. The team named the rapid-response nurse, which changed the
latency requirement from hours to minutes and forced a system decision.

"What happens when the model is wrong?" The team had no answer. After the discussion, they added
a rule: a ranked list, not an interruptive alert, so that a false positive costs a review slot
rather than a cascade. That decision belongs to layer 1 and it changed the interface at layer 3.

"Why not use the existing early-warning score?" The team had not measured it. The panel required
the measurement, which became the comparator and the most defensible part of the eventual
dossier.

"Is this a device?" The team said no. The regulatory mentor disagreed, on the grounds of the
time-critical alert. The memo was rewritten to describe a device function with a named user and
a discussion of the 510(k) pathway, which is a stronger position than an incorrect denial.

### What the worked example demonstrates

The architecture was decided in week 1, and two cheap diagnostics killed the original design
before any model was trained: an owner who could not state capacity, and a cohort that could not
support the precision the claim needed.
The revision did not change the technology. It changed the population, the number of sites, and
the claim, which are architecture decisions.
The arithmetic in this section is reproducible with the code in Section 7.

## 7. Code, query, or calculation

The three blocks below use only the Python standard library and run in a few milliseconds.
They reproduce the alert arithmetic, the precision calculation, and the risk-weighted schedule
from the worked example.

### Block 1: alert burden and precision at deployment prevalence

```python
from math import sqrt

N = 4200            # eligible admissions per year
prev = 0.031        # sepsis prevalence in the eligible cohort
sens, spec = 0.72, 0.93
work_min = 12       # minutes per alert review

tp_rate = sens * prev
fp_rate = (1 - spec) * (1 - prev)
alert_rate = tp_rate + fp_rate
alerts_yr = alert_rate * N
ppv = tp_rate / alert_rate

print(f"alert rate        {alert_rate:6.2%} of admissions")
print(f"alerts per year   {alerts_yr:7.1f}")
print(f"alerts per day    {alerts_yr / 365:8.2f}")
print(f"review minutes/d  {alerts_yr / 365 * work_min:7.1f}")
print(f"PPV               {ppv:6.3f}")
print(f"cases detected/yr {sens * prev * N:7.1f} of {prev * N:.1f}")

def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d

lo, hi = wilson(94, 130)
print(f"observed sens     {94/130:.3f}  95% CI {lo:.3f}-{hi:.3f}  width {hi-lo:.3f}")
```

Expected output:

```text
alert rate         9.01% of admissions
alerts per year     378.6
alerts per day        1.04
review minutes/d     12.4
PPV                0.248
cases detected/yr    93.7 of 130.2
observed sens     0.723  95% CI 0.641-0.793  width 0.152
```

The block makes one assumption explicit: alerts are counted per admission, not per patient-day.
If a patient stays seven days and is scored daily, alert volume multiplies by the average length
of stay, which would make the burden 7.3 per day and change the threshold. State the unit.

### Block 2: positive cases needed for a target interval width

```python
from statistics import NormalDist

prev = 0.031
N = 4200
z = NormalDist().inv_cdf(0.975)
p = 0.72

for half_width in (0.10, 0.07, 0.05):
    n_pos = z * z * p * (1 - p) / (half_width ** 2)
    admissions = n_pos / prev
    months = admissions / N * 12
    print(f"half-width {half_width:.2f} -> {n_pos:5.0f} positives "
          f"-> {admissions:6.0f} admissions -> {months:4.1f} months")
```

Expected output:

```text
half-width 0.10 ->    77 positives ->   2498 admissions ->  7.1 months
half-width 0.07 ->   158 positives ->   5098 admissions -> 14.6 months
half-width 0.05 ->   310 positives ->   9993 admissions -> 28.6 months
```

This is a normal-approximation calculation for a single proportion. For small numbers of events
the Wilson interval and a simulation are more honest; the approximation is adequate above about
50 events and is used here to show the order of magnitude, not to certify a sample size.

### Block 3: risk-weighted schedule

```python
risks = [
    ("R-01 credentialed access delayed", 4, 5, 3.0),
    ("R-02 external site declines",      3, 3, 2.0),
    ("R-03 clinician unavailable",       3, 4, 1.5),
    ("R-04 labels unusable",             2, 5, 4.0),
    ("R-05 compute budget exhausted",    3, 3, 1.0),
    ("R-06 member leaves",               2, 4, 2.5),
]
STEP = 0.5

def add(dist, weeks, p):
    out = {k: v * (1 - p) for k, v in dist.items()}
    for k, v in dist.items():
        out[k + weeks] = out.get(k + weeks, 0.0) + v * p
    return out

dist = {0.0: 1.0}
for _, likelihood, impact, weeks in risks:
    dist = add(dist, weeks, likelihood * impact / 25)

mean = sum(k * v for k, v in dist.items())
print(f"expected slip {mean:.2f} weeks; P(no slip) = {dist.get(0.0):.4f}")
for b in (0.0, 1.0, 2.0, 3.0, 4.0):
    p = sum(v for k, v in dist.items() if k <= b + 1e-9)
    print(f"P(slip <= {b:3.1f} weeks) = {p:.3f}")
```

Expected output:

```text
expected slip 6.60 weeks; P(no slip) = 0.0174
P(slip <= 0.0 weeks) = 0.017
P(slip <= 1.0 weeks) = 0.027
P(slip <= 2.0 weeks) = 0.053
P(slip <= 3.0 weeks) = 0.145
P(slip <= 4.0 weeks) = 0.217
```

The model treats risks as independent, which they are not: a delayed agreement often precedes a
site withdrawing. It also treats the weeks as additive. Both assumptions make the estimate
optimistic in the tail and are stated in Section 11. The point of the block is not the exact
number but the comparison with the plan's slack. A twelve-week plan with no slack and a 14.5
percent chance of a slip of three weeks or less is a plan that will slip.

## 8. Data, standards, and regulatory context

The architecture is constrained by four bodies of rules: privacy law, device regulation, data-use
agreements, and reporting standards. They bind differently, and confusing a convention with a
requirement is a common source of wasted work.

### Privacy and data protection

In the United States, the Health Insurance Portability and Accountability Act (HIPAA) Privacy Rule
at 45 CFR Parts 160 and 164 governs protected health information (PHI) held by covered entities
and their business associates. Research use requires either an IRB-approved protocol with a waiver
of authorisation or a determination that the data are de-identified under the Safe Harbor or
Expert Determination method. This is binding law, not guidance.
In the European Union, the General Data Protection Regulation (Regulation (EU) 2016/679) treats
health data as a special category under Article 9, requiring an explicit legal basis and
additional safeguards. National implementations add constraints; a capstone that assumes GDPR
alone is sufficient may still breach member-state law.

What the rules do not do is make a dataset suitable for a claim. A de-identified extract can still
be unrepresentative, and an IRB approval does not validate a model. Privacy compliance is
necessary and not sufficient.

### Device regulation and clinical decision support

The FDA guidance "Clinical Decision Support Software", issued in September 2022, implements
section 520(o)(1)(E) of the Federal Food, Drug, and Cosmetic Act. A CDS function is excluded from
the device definition only if it meets all four criteria: it does not acquire, process, or analyse
a medical image, in vitro diagnostic signal, or signal from a signal acquisition system; it
displays or analyses medical information; it supports a health care professional's
recommendation; and it enables the professional to independently review the basis of the
recommendation.
Software that provides a specific treatment directive, a time-critical alert, or a risk
probability typically fails the third criterion and remains a device. The guidance is binding in
the sense that it states FDA's current interpretation; it is not a statute and does not by itself
determine a classification.

In the European Union, Regulation (EU) 2017/745 on medical devices (MDR) governs device
classification, and Regulation (EU) 2024/1689, the Artificial Intelligence Act, adds obligations
for high-risk AI systems. Medical device software that is high-risk under the AI Act is subject to
conformity assessment requirements that overlap with MDR. The AI Act entered into force in August
2024 and its high-risk obligations apply on a staged timeline; a capstone should name the regime
and the stage rather than assert a date without checking.

The FDA's predetermined change control plan (PCCP) guidance, finalised in December 2024 under
section 515C of the FD&C Act, describes how a manufacturer can pre-authorise a class of model
updates with a modification protocol, acceptance criteria, and an impact assessment. It is
optional but increasingly expected for adaptive models. A capstone that plans retraining should
say whether a PCCP would be needed, even if it never submits one.

### Data-use agreements and credentialed access

PhysioNet hosts both open and credentialed datasets. Credentialed access requires an individual
account, completion of the CITI "Data or Specimens Only Research" course, and a signed data-use
agreement per dataset. The agreement prohibits redistribution and requires that access not be
shared, which means a course cannot circulate MIMIC-IV files to students. A capstone using such
data must run on approved infrastructure and must not place extracts in a public repository.

This constrains architecture directly: the reproducibility artefact becomes code plus a data
access statement, not code plus data. The definition of done has to accommodate that.

### Reporting standards

TRIPOD+AI (2024) is a reporting guideline with 27 main items, including intended use, data
sources, sample size justification, calibration, subgroup performance, and open science. It is a
convention, not a legal requirement, but a capstone dossier that omits its items looks
incomplete to a clinical reviewer.
DECIDE-AI (2022) covers early live clinical evaluation and adds human factors, human-computer
agreement, and safety reporting. It applies only if the capstone runs in a live workflow; most
course capstones do not, and should say so rather than claim DECIDE-AI compliance.
The ICH E9(R1) estimand framework is binding for regulated clinical trials, not for coursework,
but its discipline of specifying the population, variable, and intercurrent-event handling is
useful for any capstone with a time-to-event outcome.
ISO 14971 for risk management and IEC 62304 for medical device software life-cycle processes are
binding only within a regulatory submission; a capstone uses their structure as a checklist.

### Interoperability standards constrain the system layer

If the capstone reads from a hospital system rather than a static extract, the data model matters
at layer 3. Fast Healthcare Interoperability Resources (FHIR) is the current HL7 standard for
exchanging clinical data; the US Core Implementation Guide profiles the resources required for
United States certification, and the Data Exchange for Quality Measures (DEQM) guide covers
quality-measure reporting. Health Level Seven (HL7) Version 2 messages remain common in
interface engines, and a pipeline that assumes FHIR will fail at a site that still sends
HL7 v2.
Imaging data arrive as Digital Imaging and Communications in Medicine (DICOM) objects with
metadata that often encode scanner, site, and acquisition parameters. That metadata is both a
feature source and a leakage risk: a model that learns scanner identity instead of pathology
will fail at the next site.
Observational data warehouses frequently use the Observational Medical Outcomes Partnership
(OMOP) Common Data Model, which normalises vocabularies through SNOMED CT, LOINC, and RxNorm.
A capstone using OMOP inherits its conventions, including the distinction between source values
and standard concept identifiers, and must say which it used.

None of these standards is a regulatory requirement for coursework.
They are constraints on the system layer, and a dossier that does not name the data model is
hard to reproduce.

### Binding requirement versus convention

| Instrument | Status | What it constrains | What it does not |
|---|---|---|---|
| HIPAA Privacy Rule | Binding law in the US | Use and disclosure of PHI | Model validity or representativeness |
| GDPR | Binding law in the EU | Lawful basis and safeguards for health data | Clinical evidence requirements |
| FDA CDS guidance | Binding interpretation of statute | Whether a function is a device | Study design quality |
| EU MDR and AI Act | Binding regulation in the EU | Classification and conformity assessment | Local adoption or reimbursement |
| PhysioNet DUA | Contractual | Access, redistribution, security | Scientific adequacy |
| TRIPOD+AI | Reporting convention | What must be reported | Whether the study was done well |
| DECIDE-AI | Reporting convention | Early live evaluation items | Applicability to retrospective work |
| ICH E9(R1) | Binding for regulated trials | Estimand specification | Coursework claims |
| ISO 14971 | Binding within a submission | Risk management process | Clinical benefit |
| FHIR US Core | Convention with regulatory use | Data representation | Data quality |

The distinction matters at the defense. Saying "TRIPOD+AI requires us to report subgroup
performance" is imprecise; the guideline recommends it, and the panel may still judge the work
incomplete if it is absent. Saying "HIPAA requires de-identification" is precise, and the
follow-up question is which method and with what residual risk.

### 21 CFR Part 11 and audit trails

Where a capstone writes to a system of record, electronic records rules may apply. 21 CFR Part 11
sets requirements for electronic signatures, audit trails, and record retention for records
submitted to the FDA. A course project does not submit, but a prototype that writes a
reconciled medication list into a chart is exercising the same function. The architecture should
record who wrote what and when, even if the answer is "a research prototype, no write-back".
Auditability is cheap to design in and expensive to retrofit.

### Clinical laboratory rules

If the output is a laboratory result rather than a decision aid, the Clinical Laboratory
Improvement Amendments (CLIA) framework applies to the laboratory, not to the model. This is a
frequent category error in capstones that propose "a diagnostic test": a model that estimates
risk is not a laboratory test, and a laboratory test that uses a model still requires a
certified laboratory and a validated assay. Naming the category prevents an inflated claim.

### What none of this does

No standard validates a model. No licence makes a dataset representative. No regulatory pathway
makes a claim true. The architecture exists to keep those distinctions visible in the dossier,
and the defense will test them.

## 9. Industry practice and case evidence

### The evidence gap between approval and benefit

The United Kingdom Royal College of Radiologists, in a 2025 public consultation document on AI
deployment fundamentals, states that device marking does not establish clinical or
cost-effectiveness, and that independent evaluation of imaging AI remains limited. That is a
professional body's assessment, not a trial result, but it matches the pattern in the published
literature: regulatory clearance and clinical benefit are separate claims.

The National Institute for Health and Care Excellence (NICE) early value assessment programme
makes the same point operationally. Its evidence generation plans, such as the 2025 plan for AI
in opportunistic detection of vertebral fragility fractures, contain explicit evidence-gap tables
marking health-related quality of life, resource use, and diagnostic accuracy against standard
care as absent or limited, and propose real-world designs to fill them over multi-year periods.
A twelve-week capstone cannot fill such a gap, and should not claim to.

### External validation failures are the norm, not the exception

Wong and colleagues published an external validation of a proprietary sepsis prediction model in
JAMA Internal Medicine in 2021. Across 27,697 patients at one academic medical centre, the model
achieved an AUROC of 0.63, against vendor-reported values of 0.76 to 0.83, and failed to identify
67 percent of sepsis patients. The study is a single-site retrospective validation and has its own
limitations, but it is the canonical example of why the comparator and the validation population
must be fixed before deployment claims.

Obermeyer and colleagues, in Science in 2019, showed that a commercial risk-stratification
algorithm used health care costs as a proxy for need, and that at the same predicted risk Black
patients were substantially sicker than White patients. Correcting the label would have increased
the proportion of Black patients identified for extra care from 17.7 percent to 46.5 percent. The
lesson for architecture is that the outcome definition at layer 2 is a governance decision at
layer 4.

### What large public challenges demonstrate about scale

The RSNA 2025 Intracranial Aneurysm Detection challenge, run on Kaggle from July to October 2025,
attracted 1,147 teams and used a reference dataset from 18 sites on five continents annotated by
more than 60 radiologists. The first-place solution reported an area under the curve of 0.915 for
aneurysm presence and 0.916 across 13 locations, using a vessel-segmentation then
region-of-interest classification pipeline. The number is not the lesson; the lesson is the
architecture of the data: multi-site, expert-annotated, with a held-out test set that no
competitor could see.

The George B. Moody PhysioNet Challenge 2025 asked teams to identify Chagas disease from 12-lead
electrocardiograms (ECGs) to prioritise scarce serological testing. The official score is the
fraction of Chagas patients captured in the top 5 percent of the cohort, a capacity-constrained
metric rather than AUROC. The organisers report 111 teams, 630 participants, and 41 ranked teams;
the top test score was 0.323, and performance fell to 0.100 on an external cohort from a largely
asymptomatic population. The fall is the finding: a metric that matches capacity does not
guarantee transportability.

The 2026 PhysioNet Challenge moves to cognitive impairment screening from polysomnography, with
an age-conditioned AUROC that removes the age shortcut. The task and data were released; results
are not asserted here. Age conditioning is an architectural choice at layer 2 that anticipates a
confounder at layer 1, and it is worth studying for that reason alone.

### Hackathons produce prototypes, not products

Olson and colleagues surveyed 356 projects from twelve healthcare hackathons in India, Uganda, and
the United States between 2012 and 2015, published in BMJ Innovations in 2017. Of 196 responding
projects, 30.3 percent had made progress after a mean of 12.2 months, and 24.7 percent had begun
pilot testing. The study is a survey with a 55 percent response rate and conservative coding of
non-responders, so the true continuation rate is uncertain, but the order of magnitude is clear:
most hackathon output does not continue.

That is not a criticism of hackathons; it is a boundary on what they evidence. A hackathon
artefact demonstrates framing speed and demo capability. A validated product requires the four
layers, and most of the work is in layers 1 and 4.

### Benchmarks that reward the baseline

The Open Problems in Single-Cell Analysis initiative, reported in Nature Biotechnology in 2025,
runs a set of benchmark tasks with live leaderboards. In label projection, a standard logistic
regression on log-normalised principal components was the best-scoring method in five of eight
datasets, ahead of several neural methods designed for the task. The finding is not that deep
learning is useless; it is that a well-implemented simple baseline is hard to beat when the
benchmark controls the split properly.

The lesson for a capstone is architectural. A baseline is not a formality to be disposed of in
week 4; it is the evidence that complexity buys something. If the candidate does not beat a
logistic model with the same features on the same split, the correct conclusion is to report the
baseline and justify the simpler system, which is often the better product.

### Open-weight model challenges

The MedGemma Impact Challenge, run with Kaggle and announced in January 2026, asked teams to
build prototypes on open-weight health models for low-resource and on-device settings. Winners
were reported in March 2026, with projects spanning offline outbreak surveillance, on-device
tuberculosis screening, and medication-error reconciliation. The winning entries are prototypes;
none is a validated product, and the competition did not test clinical outcomes. The relevant
architectural point is that the constraints were stated up front, such as offline operation and
device memory, and teams that designed to those constraints beat teams that designed for a
desktop benchmark.

### How to read a competition result

Three rules apply to every number in this section.
First, a leaderboard position is evidence about a metric on a specific hidden split, not about
clinical value.
Second, organiser-reported team counts measure marketing and prize appeal as much as difficulty.
Third, a reported trend among winning solutions is a description of what won, not a controlled
comparison of methods.

The PhysioNet 2025 Challenge illustrates the second and third rules together. The top test score
of 0.323 and the external-cohort score of 0.100 come from the same models; the difference is the
population, not the method. A team that cites 0.323 without 0.100 has cited half a result.

### Inference, labelled

It is an inference, not a documented finding, that teams which fix the decision and the owner in
week 1 produce more defensible capstones than teams which fix the model first. The published case
evidence supports the components: intended use drives regulatory classification, label choice
drives equity outcomes, and external validation frequently degrades performance. The causal claim
about course teams is plausible and untested.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Solution seeking a decision | Model family discussed before any user is named | Team optimises the artefact it can build | No adoption story; weak G0 | Rewrite the decision statement; interview an owner |
| Inverted architecture | Weeks 3 to 9 contain only modelling tasks | Metric work is comfortable and measurable | Governance and system missing at defense | Put layer 1 in week 1 and layer 4 in week 2 |
| Owner without capacity | "It depends on staffing" | No operating point can be chosen | Threshold unusable; alerts ignored | Obtain a number from a shift-level manager |
| Cohort too small for the claim | Interval spans the comparator | Precision depends on events, not records | Claim cannot be supported | Pool sites, extend time, or narrow the claim |
| Random split presented as external validation | No date or site variable in the split | Convenience and habit | Evidence invalid for deployment | Fix validation level against claim at G2 |
| Leakage through time zero | Suspiciously high validation performance | Post-decision features included | Invalid evidence; silent in production | Leakage screen; rebuild features as of prediction time |
| Metric mismatch | AUROC reported for a fixed-capacity queue | Metric chosen by familiarity | Model ranks well where nobody acts | Derive metric from capacity and error costs |
| Data access filed late | No DUA or IRB number in week 2 | Access is outside team control | Three-to-six week delay | File in week 1; pre-approve two fallbacks |
| Governance deferred | Regulatory memo started in week 11 | Uncomfortable, unfamiliar work | Claim exceeds evidence; deduction | Draft intended use in week 2 |
| Scope creep | New questions after week 6 | Curiosity and sponsor requests | Core artefacts unfinished | Three-tier scope; backlog with dates |
| Demo dependency | Demo needs live network and real data | Demo built once, late | Public failure at defense | Recorded backup; synthetic inputs |
| Free riding | Two members commit most code | Team grade hides contribution | Individual defense fails | Contribution log; reassign at midpoint |
| Unfunded compute | Training queue times exceed plan | Candidate tuning unbounded | No candidate at freeze | Cap trials; baseline is the fallback result |
| Inconsistent artefacts | Slide numbers differ from repository | Manual transcription | Credibility loss at first question | Metrics computed by script; freeze and tag |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The decision owner is identifiable and reachable | Every later step depends on capacity and workflow | Owner is a committee or an absent vendor | Ask for a 30-minute meeting in week 1; if refused, change brief |
| Capacity is stable over the project | Threshold and alert burden are computed once | Seasonal staffing or a reorganisation | Ask for night and day numbers; use the smaller |
| Prevalence in the deployment cohort is known | PPV and alert volume depend on it | Coding changes or a new case definition | Measure on a recent extract; report a range |
| The available cohort represents deployment | Validation is only as good as the sampling | Referral centre data used for community claim | Compare case mix on age, acuity, comorbidity |
| Time zero can be reconstructed from timestamps | Leakage control depends on it | Order times missing or overwritten | Audit 50 records manually |
| The comparator can be measured | Otherwise the claim has no bar | No historical score is recorded | Recompute the score from raw variables |
| Subgroup variables are recorded | Equity analysis depends on them | Race or language fields incomplete | Report missingness by group |
| Credentialing takes two to four weeks | Schedule depends on it | Institutional legal review adds weeks | File in week 1; track status weekly |
| Risks are independent and additive | Risk-weighted schedule is tractable | Correlated risks compound | Re-run with correlated scenarios |
| Freeze can be enforced | Otherwise artefacts diverge | Team continues modelling past the date | Version tag and a named enforcer |

The largest limit is that architecture is a necessary and not a sufficient condition. A team can
follow every step and still produce a weak capstone because the data are poor or the question is
unimportant. The architecture prevents a specific class of failure, not all failure.

## 12. Exercises

### Exercise 1. Reconstruct the architecture (recall, 20 minutes)

Draw the four-layer diagram from memory, including the gate positions on the twelve-week axis.
For each layer, write one constraint it imposes on the layer above.
**Deliverable:** one page, hand-drawn or text.
**Criteria:** all four layers correct; at least three constraints that are causal rather than
descriptive; gates placed at weeks 1, 2, 5, 7, 9, 11.
**Time:** 20 minutes.

### Exercise 2. Rewrite three decision statements (application, 30 minutes)

Take these three statements and rewrite each as a decision with an owner, an action, and a
frequency: "predict readmission", "detect pneumonia on chest X-ray", "summarise discharge notes".
**Deliverable:** three sentences plus one sentence each on what is now out of scope.
**Criteria:** exactly one action per sentence; a named role; a frequency; the excluded action
named.
**Time:** 30 minutes.

### Exercise 3. Compute alert burden at three thresholds (calculation, 45 minutes)

Using the cohort in Section 6, compute alert rate, alerts per day, PPV, and cases detected for
sensitivity/specificity pairs of (0.85, 0.80), (0.72, 0.93), and (0.55, 0.98).
**Deliverable:** a table and a one-paragraph recommendation to the nurse manager.
**Criteria:** correct arithmetic; recommendation tied to the four-per-shift capacity; the
trade-off between detection and false alerts stated in numbers.
**Time:** 45 minutes.

### Exercise 4. Choose a validation level (critique, 40 minutes)

A team proposes to claim "works in community hospitals" using a random split of data from one
academic centre. Critique the claim and specify the minimum design that would support it.
**Deliverable:** a 300-word memo with a revised claim and a design.
**Criteria:** identifies the mismatch between claim and split; names L3 as the requirement;
proposes a feasible alternative or an explicit narrowing.
**Time:** 40 minutes.

### Exercise 5. Design two fallbacks (design, 40 minutes)

For a capstone that needs MIMIC-IV, specify fallback A and fallback B, including licence,
schema compatibility, and what part of the scientific question survives each.
**Deliverable:** a one-page table plus a sentence on when the project should change brief.
**Criteria:** fallback A is genuinely accessible; fallback B is labelled synthetic everywhere;
the survival condition is explicit.
**Time:** 40 minutes.

### Exercise 6. Build the risk-weighted schedule (calculation, 60 minutes)

Take your own capstone's six largest risks, assign likelihood, impact, and slip in weeks, and run
the convolution from Section 7. State the probability of finishing within your slack.
**Deliverable:** the script output and a 200-word decision about whether to cut scope.
**Criteria:** inputs justified; independence assumption challenged; a concrete cut proposed if
the probability is below 50 percent.
**Time:** 60 minutes.

### Exercise 7. Write the regulatory triage paragraph (application, 45 minutes)

For a shift-level ranked list shown to a nurse, write the intended use, the likely FDA
classification under the 2022 CDS guidance, and the binding artefact.
**Deliverable:** one paragraph plus three open questions for a regulatory mentor.
**Criteria:** all four CDS criteria addressed; the time-critical alert issue named; no assertion
of a classification as settled fact.
**Time:** 45 minutes.

### Exercise 8. Defend one exclusion (design, 30 minutes)

Choose one item from your out-of-scope tier and prepare a 90-second defence of why excluding it
signals maturity rather than laziness.
**Deliverable:** a spoken script and one slide.
**Criteria:** names the cost of including it; names the risk of excluding it; states what would
have to be true to include it later.
**Time:** 30 minutes.

## 13. Self-check answers

**1. Which layer constrains which, and in what direction?**
The decision constrains the evidence, which constrains the system, which is constrained by
governance. Constraints propagate upward: capacity fixes the operating point, the operating point
fixes the metric and the validation design, the validation design fixes the evidence claim, and
the claim fixes the regulatory path. A change at layer 1 invalidates work at layers 2 and 4.

**2. Why is a random split labelled L0?**
Because it proves only that the pipeline runs on held-out rows from the same distribution and
era. It says nothing about a new patient population, a new time period, or a new site. For a
deployment claim it is debugging evidence, not validation evidence.

**3. Compute the alert rate for sensitivity 0.85 and specificity 0.80 at 3.1 percent prevalence.**
True-positive rate = 0.85 × 0.031 = 0.02635. False-positive rate = 0.20 × 0.969 = 0.1938. Total =
0.22015, or 22.0 percent. At 4,200 admissions that is 924.6 alerts per year, or 2.53 per day, and
PPV = 0.02635 / 0.22015 = 0.120. The higher sensitivity costs 546 more alerts per year for 17 more
cases detected, and each alert is more likely to be false.

**4. Why does broadening the population reduce PPV?**
Because PPV depends on prevalence. Halving prevalence from 3.1 to 1.55 percent roughly halves the
true-positive rate while the false-positive rate changes little, so the ratio falls from 0.248 to
0.139. The model has not changed; the population has.

**5. How many positive cases are needed for a half-width of 0.07 around 0.72?**
Using the normal approximation, n = 1.96² × 0.72 × 0.28 / 0.07² = 158. At 3.1 percent prevalence
that is about 5,098 admissions, or 14.6 months at 4,200 per year.

**6. What are the two cheapest early diagnostics of a doomed capstone?**
First, whether the owner can state a capacity number with a range narrower than a factor of three.
Second, whether the available positive cases reach the number required for the target interval
half-width. Both are answerable in week 1 and neither requires a model.

**7. Why is the regulatory memo a week-2 artefact?**
Because the regime determines what evidence is admissible and what claims are permitted. If the
output is a time-critical alert, it is likely a device function, and the validation and
documentation requirements change. Discovering that in week 9 leaves no time to redesign.

**8. What does freeze protect?**
It protects the correspondence between the repository, the dossier, and the slides. After freeze,
only factual corrections are allowed, and they are announced. This makes the defense a
description of work rather than a negotiation about which number is current.

**9. Why is a negative result acceptable?**
Because the capstone's claim is about a decision, not about a model's superiority. A well-measured
comparator plus a candidate that fails to beat it supports the recommendation "do not change
practice", which is a decision-ready finding. The failure would be an unmeasured comparator or a
post-hoc metric.

**10. Rubric for Exercise 5.** Fallback A scores high when it is accessible without a new
agreement, has compatible schema, and preserves the decision. Fallback B scores high when the
generator is versioned, calibrated to a cited source, labelled synthetic in every artefact, and
never used for a clinical claim. A fallback that changes the decision is not a fallback; it is a
new brief.

## 14. Interview questions

**Q1. "Walk me through how you scoped your capstone."**
Strong answer names the decision, owner, capacity, comparator, and the point at which scope was
cut, with a number for each. Weak answer narrates the model and the dataset.

**Q2. "What would you do differently?"**
Strong answer identifies an irreversible choice made too late, such as the split or the data
request, and says what it cost. Weak answer says "start earlier" or "use a bigger model".

**Q3. "How did you decide the validation design?"**
Strong answer derives the level from the claim: temporal for forward use, site-held-out for a new
environment, and explains why a random split would not support the claim. Weak answer says "we
used cross-validation".

**Q4. "Your model has an AUROC of 0.82. Is that good?"**
Strong answer asks: at what prevalence, at what operating point, against what comparator, with
what calibration, and for which subgroup. Weak answer says yes, or compares to a rule of thumb.

**Q5. "Who owns the decision your model supports?"**
Strong answer names a role and describes how that person receives the output and what they do
differently. Weak answer says "clinicians" or "the hospital".

**Q6. "How did you handle alert fatigue?"**
Strong answer computes alerts per shift at deployment prevalence and ties the threshold to
capacity. Weak answer says the model has high specificity.

**Q7. "What is the regulatory status?"**
Strong answer names the regime, the criteria applied, the likely classification, and the binding
artefact, with uncertainty stated. Weak answer says "it is research only" while describing a
clinical workflow.

**Q8. "What did you exclude, and why?"**
Strong answer names an exclusion with its cost and the condition under which it would be
reversed. Weak answer claims nothing was excluded.

**Q9. "How do you know your data are representative?"**
Strong answer compares case mix and reports the direction of expected bias. Weak answer says the
dataset is large.

**Q10. "What would make you stop the project?"**
Strong answer gives a numeric kill criterion on a named metric or a safety event. Weak answer
says "if it does not work".

## 15. Key terms

- **Architecture** — the ordered arrangement of decision, evidence, system, and governance layers.
- **Alert burden** — alerts per unit time at deployment prevalence and threshold.
- **Capacity** — actions the owner can take per shift, day, or week.
- **Comparator** — current practice measured on the same cohort.
- **Core tier** — deliverables that cannot slip without failing the capstone.
- **Decision owner** — the named role accountable for acting on the output.
- **Decision statement** — one sentence naming event, action, owner, and population.
- **De-identification** — documented removal or transformation of identifiers under a named standard.
- **Dossier** — the versioned thirteen-artefact submission at G5.
- **DUA** — data-use agreement governing access and redistribution.
- **Estimand** — the precise quantity a study estimates, per ICH E9(R1).
- **Evidence sufficiency** — the precision at which the owner's decision can be made.
- **Fallback A** — a pre-approved accessible dataset with the same schema.
- **Fallback B** — a versioned synthetic generator, labelled synthetic everywhere.
- **Freeze** — the date after which only factual corrections are permitted.
- **Gate** — a scheduled review with a named reviewer and a pass condition.
- **Intended use** — population, setting, inputs, output, user, and contraindications.
- **Kill criterion** — pre-specified result that ends the project or the claim.
- **L0 to L4** — the validation ladder from random split to prospective evaluation.
- **Minimum viable evidence package** — the twelve artefacts required regardless of metric.
- **Operating point** — threshold plus the sensitivity and specificity it produces.
- **PCCP** — predetermined change control plan for pre-authorised model updates.
- **PPV** — proportion of flagged cases that are true positives at deployment prevalence.
- **SaMD** — software as a medical device.
- **Scope tier** — core, stretch, or out.
- **Stretch tier** — optional work attempted only after freeze.
- **Time zero** — the moment inputs are fixed and the outcome clock starts.
- **TRIPOD+AI** — 2024 reporting guideline for prediction model studies.
- **Validation level** — the design that supports a specific generalisation claim.
- **Wilson interval** — a binomial confidence interval that behaves better than the normal approximation near 0 and 1.

## 16. Sources

All URLs accessed 26 October 2026.

- FDA, *Clinical Decision Support Software*, guidance, September 2022.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- FDA, *Marketing Submission Recommendations for a Predetermined Change Control Plan for
  Artificial Intelligence-Enabled Device Software Functions*, final guidance, December 2024.
  https://www.fda.gov/media/185050/download
- FDA, *Software as a Medical Device (SaMD)*.
  https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd
- Regulation (EU) 2017/745 on medical devices (MDR). https://eur-lex.europa.eu/eli/reg/2017/745/oj
- Regulation (EU) 2024/1689, Artificial Intelligence Act.
  https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- Regulation (EU) 2016/679, General Data Protection Regulation.
  https://eur-lex.europa.eu/eli/reg/2016/679/oj
- HIPAA Privacy Rule, 45 CFR Parts 160 and 164. https://www.hhs.gov/hipaa/
- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
  https://doi.org/10.1136/bmj-2023-078378
- Vasey B, Nagendran M, Campbell B, et al. DECIDE-AI reporting guideline. *Nature Medicine*
  2022;28:924-933. https://doi.org/10.1038/s41591-022-01772-9
- ICH E9(R1), *Addendum on Estimands and Sensitivity Analysis in Clinical Trials*.
  https://www.ich.org/
- ISO 14971, *Medical devices — Application of risk management to medical devices*.
  https://www.iso.org/standard/72704.html
- IEC 62304, *Medical device software — Software life cycle processes*. https://www.iec.ch/
- PhysioNet, required training and data-use agreement for MIMIC-IV.
  https://physionet.org/content/mimiciv/view-required-training/3.0/
- PhysioNet, CITI course instructions. https://www.physionet.org/about/citi-course/
- Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
  sepsis prediction model. *JAMA Internal Medicine* 2021;181(8):1065-1070.
  https://doi.org/10.1001/jamainternmed.2021.2626
- Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to
  manage the health of populations. *Science* 2019;366(6464):447-453.
  https://doi.org/10.1126/science.aax2342
- Olson KR, Walsh M, Garg P, et al. Health hackathons: theatre or substance? *BMJ Innovations*
  2017;3:37-44. https://doi.org/10.1136/bmjinnov-2016-000147
- RSNA, Intracranial Aneurysm Detection AI Challenge, 2025.
  https://www.rsna.org/artificial-intelligence/ai-image-challenge/intracranial-aneurysm-detection-ai-challenge
- Reyna MA, Koscova Z, Pavlus J, et al. Detection of Chagas disease from the ECG: the George B.
  Moody PhysioNet Challenge 2025. arXiv:2510.02202. https://arxiv.org/abs/2510.02202
- Royal College of Radiologists, *AI deployment fundamentals*, public consultation draft, 2025.
  https://www.rcr.ac.uk/media/r0th2x0k/ai-deployment-fundamentals_public-consultation.pdf
- NHS England, *Planning and implementing real-world AI evaluations: lessons from the AI in
  Health and Care Award*. https://www.england.nhs.uk/long-read/planning-and-implementing-real-world-ai-evaluations-lessons-from-the-ai-in-health-and-care-award/

## 17. Links

- Module map: [`_index.md`](_index.md)
- Next topic: [`02_milestone-plan.md`](02_milestone-plan.md)
- Related topic: [`04_capstone-archetypes.md`](04_capstone-archetypes.md)
- Related topic: [`08_scoping-playbook.md`](08_scoping-playbook.md)
- Survey file: [`../../course/19_capstone_hackathons.md`](../../course/19_capstone_hackathons.md)
- Labs: [`../../labs/`](../../labs/)
