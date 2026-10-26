# The Evidence Chain

> Module 18 — Quality, Evaluation, and Reporting
> Topic 01 of 09
> Estimated study time: 3 hours
> Prerequisites: none
> Version 2026.3

## 1. Why this topic exists

A biomedical AI product is sold on a sentence. "This model detects sepsis earlier." "This
algorithm reads mammograms at the level of a specialist." "This tool reduces radiologist
reading time by 30 percent." Each sentence is a claim, and each claim rests on a chain of
artifacts: an endpoint definition, a dataset, an analysis, a table, and a paragraph. When the
chain holds, a regulator, a payer, a hospital procurement committee, or a clinician can follow
the sentence back to the raw measurement and check it. When the chain breaks, nobody can.

The break is rarely dramatic. It is usually a small substitution made under deadline. The
endpoint in the abstract is "sepsis," but the endpoint in the methods is "ICD-10 code for
sepsis plus two SIRS criteria within a six-hour window." The dataset in the methods is "38,455
hospitalizations," but the analysis unit in the results is "27,697 patients." The claim is
"reduces mortality," but the measured outcome is "time to antibiotic order." Each substitution
is defensible in isolation. Together they mean the sentence at the top of the press release
cannot be reconstructed from the numbers at the bottom of the paper.

Three groups pay for broken chains.

The first is the buyer. A hospital system evaluating a sepsis or imaging product receives a
vendor dossier, a published paper, and a conference poster. The dossier cites the paper; the
paper cites an internal validation; the internal validation is not published. Procurement has
six weeks and no mechanism to detect that the citation chain terminates in an unpublished
sponsor document. The purchase proceeds on the strength of a figure that was generated on a
different population, at a different prevalence, with a different label definition.

The second is the regulator. A 510(k) submission for an AI-enabled device must demonstrate
substantial equivalence, and the clinical evidence in the submission is summarised in a
public summary that frequently omits the design of the performance study. A 2025 review of
903 FDA-authorised AI-enabled devices found that clinical performance studies were reported
for 55.9 percent of devices, that 38.2 percent of the reported studies were retrospective,
and that only 16.2 percent reported an area under the receiver operating characteristic curve
(AUROC). The reviewer is not necessarily hiding anything. The chain simply was never written
down in a form that survives the submission.

The third is the patient. A model deployed on the strength of a chain that broke at the
dataset link will fail on the population that was excluded from that dataset. The failure is
silent because no one is measuring the link that broke.

This topic teaches the chain as an object you can draw, inspect, and score. It is the
foundation for the rest of the module. Reporting guidelines tell you which links to document.
Risk-of-bias tools tell you which links are weak. Model cards and datasheets document two
specific links. Reproducibility tests the analysis link. Quality management, audit, and
documentation are the industrial processes that keep the chain intact across staff turnover.

The industry question for the module is: would an independent reviewer reach the same
conclusion from our evidence, and can they reproduce it? The evidence chain is the object that
question is asked about.

A note on scope. This topic does not teach statistics. It teaches traceability: the discipline
of knowing, for every sentence you publish, which number supports it, where that number came
from, and what would have to be true for the sentence to be false. That discipline is cheap to
apply at design time and expensive to retrofit.

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** the evidence chain for a given claim, from sentence to raw measurement, in a form
   another reviewer can follow without asking you a question.
2. **Classify** each link as documented, inferable, or missing, and distinguish a reporting
   deficiency from a methodological flaw.
3. **Write** a claim decomposition that separates the population, the intervention or model,
   the comparator, the endpoint, the time horizon, and the direction of effect.
4. **Trace** a published performance number back to the dataset and analysis that produced it,
   or state precisely where the trace fails.
5. **Score** chain strength with an explicit rubric, and defend the score with quoted evidence
   rather than impression.
6. **Detect** the seven common chain breaks: incomplete reporting, selective outcome reporting,
   unsupported subgroup claims, protocol deviations, post-hoc changes, conflicting reference
   standards, and retracted or corrected studies.
7. **Write** a one-page traceability memo that a procurement committee or a regulator can act
   on.
8. **Decide** when a broken chain is disqualifying and when it is a documentation request that
   can be satisfied before a decision.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

The chain has seven nodes and six links. Nodes are artifacts. Links are the claims of
correspondence between artifacts. Almost every quality failure is a broken link, not a missing
node: the dataset exists, the model exists, the number exists, but the number does not
correspond to the node it is attached to.

```text
   [1] Claim
       |  link A: claim <-> endpoint
   [2] Endpoint / estimand
       |  link B: endpoint <-> dataset variable
   [3] Dataset and cohort
       |  link C: dataset <-> analysis population
   [4] Analysis plan and code
       |  link D: analysis <-> reported number
   [5] Reported result
       |  link E: result <-> interpretation
   [6] Interpretation / recommendation
       |  link F: recommendation <-> deployment context
   [7] Decision and monitoring
```

Read the diagram as a set of obligations, not a pipeline. Each link imposes an obligation on
the author and creates a verification question for the reviewer.

**Link A, claim to endpoint.** The claim must name the same construct as the endpoint. A claim
of "earlier detection" requires an endpoint of detection time, not an endpoint of alert
volume. A claim of "reduces mortality" requires a mortality endpoint measured in a design that
can attribute the difference to the model.

**Link B, endpoint to dataset variable.** The endpoint must be operationalised in the dataset.
If the endpoint is "sepsis onset," the dataset must contain a variable that marks onset, with
a definition, a source, and a time stamp. Many datasets contain a diagnosis code that is
assigned at discharge. That code is not onset. The link is broken by a substitution of
convenience.

**Link C, dataset to analysis population.** The analysed population must be the dataset
population, or the difference must be described. Exclusions, missing data handling, and
patient-level versus encounter-level units change the population. A model evaluated on
hospitalisations and deployed on patients is a different product.

**Link D, analysis to number.** The reported number must be produced by the analysis described.
This is where reproducibility lives. If the code does not reproduce the table, the link is
broken regardless of whether the number is correct.

**Link E, number to interpretation.** The interpretation must be licensed by the number. An
AUROC of 0.63 licenses "poor discrimination at this site." It does not license "the model does
not work anywhere," and it does not license "the model works but needs recalibration" unless
calibration was measured.

**Link F, recommendation to context.** The recommendation must match the deployment context:
prevalence, case mix, workflow, available actions, and the consequences of both error types.
A model validated in an academic centre with a 30 percent disease prevalence and deployed in a
community setting with 5 percent prevalence will have a different positive predictive value at
the same sensitivity and specificity. The number did not change. The context did.

Two properties of the chain matter more than the others.

**The weakest link governs.** A chain is not stronger than its weakest link. A model with
excellent discrimination and a label derived from billing codes is weak evidence, because link
B fails. A perfect dataset analysed with a random split that leaks patients across folds is
weak evidence, because link C fails. Reviewers who average across links ("mostly good")
misjudge the artifact.

**Load-bearing links are not equally important.** Link A and link B carry the most weight for
claims about clinical effect. Link C and link D carry the most weight for claims about
performance. Link F carries the most weight for claims about local utility. A defect in a
non-load-bearing link is a documentation request. A defect in a load-bearing link is a
finding.

The model also distinguishes three states for each link, and the distinction is the single most
useful habit in this module.

| State | Meaning | Reviewer action |
|---|---|---|
| Documented | Evidence in the artifact permits verification | Accept, record the citation |
| Inferable | Evidence permits a reasonable inference, with stated assumptions | Accept with caveat, request confirmation |
| Missing | No evidence, or evidence contradicts the claim | Do not accept the claim; request data or downgrade |

A missing link is not automatically a methodological flaw. It may be a reporting deficiency:
the work was done correctly and not described. The distinction matters because the remedies
differ. A reporting deficiency is fixed by disclosure, which may be possible in a revision, a
supplement, or a data-access request. A methodological flaw is fixed only by new work. Section
5 builds the test that separates them.

**Failure propagates downward, never upward.** A defect in link B does not merely weaken link
B; it contaminates every estimate downstream, because the estimate was computed against the
wrong label. A defect in link F does not contaminate the estimate at all; the number remains
valid for the population studied, and only the transfer to a new context fails. This asymmetry
determines the order of investigation: check the label before checking the metric, and check
the metric before checking the recommendation. Reviewers who start at the discussion section
and work upward routinely spend their time on the least consequential defect.

| Defect | Links affected | Number still valid for the study population? | Number valid for your population? |
|---|---|---|---|
| Wrong label construct | B, D, E, F | No | No |
| Unrepresentative cohort | C, F | Yes, for that cohort | No |
| Leaky split | C, D | No | No |
| Post-hoc threshold | D, E | No, as reported | No |
| Overreaching conclusion | E | Yes | Conditional |
| Context mismatch | F | Yes | No |

Read the table as a guide to what you can salvage. A context mismatch leaves a usable number
and an unusable recommendation. A label defect leaves nothing usable, which is why it is worth
checking first even though it is the least visible from the abstract.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or system |
|---|---|---|---|
| Claim | A sentence asserting an effect, a capability, or a level of performance | Treated as the study finding rather than as a summary of it | None |
| Endpoint | The quantity measured to evaluate the claim | Used interchangeably with "outcome" even when one is a surrogate | Defined per study |
| Estimand | The precise quantity to be estimated, including population, variable, intercurrent events, and summary measure | Omitted; analysts report "the effect" without saying which | ICH E9(R1) framework |
| Outcome | A measured variable in a dataset | Conflated with endpoint; a dataset can have many outcomes and one endpoint | Variable name |
| Intended use | The purpose, population, user, and setting for which a device is intended | Written so broadly that no validation population can fail it | FDA predicate; EU MDR intended purpose |
| Indication | The disease or condition the product addresses | Used as a synonym for intended use | Clinical terminology |
| Data provenance | The origin and processing history of a data element | Reduced to "we used the EHR" | Metadata, lineage records |
| Data lineage | The traceable path from a source field to an analysis variable | Confused with provenance, which is origin, not transformation | Transformation graph |
| Cohort | The set of units selected for analysis | Used for the full dataset when exclusions were applied | Count with inclusion criteria |
| Analysis population | The units actually contributing to a reported estimate | Not reported; denominator ambiguity is common | Count; per-analysis |
| Unit of analysis | The entity a row represents (patient, encounter, image, lesion) | Silently switched between dataset and results | Patient, encounter, image |
| Traceability matrix | A table mapping each claim to its supporting artifact | Maintained informally or not at all | Document |
| Reporting deficiency | A link is undocumented but the underlying work may be sound | Used as a euphemism for a flaw | Assessment category |
| Methodological flaw | A link is substantively broken by design or execution | Used loosely for any weakness | Assessment category |
| Selective outcome reporting | Reporting a subset of measured outcomes based on results | Confused with choosing a primary endpoint in advance | Bias category |
| Post-hoc analysis | An analysis specified after seeing data | Assumed to be invalid; it is hypothesis-generating, not confirmatory | Analysis label |
| Protocol deviation | A departure from the registered or approved plan | Conflated with a protocol amendment | Trial record |
| Reference standard | The test or adjudication used to establish the true condition | Called "ground truth," which overstates certainty | Diagnostic accuracy |
| Surrogate endpoint | A measure substituted for the outcome of interest | Treated as equivalent without validation | Biomarker, proxy |
| Construct validity | The degree to which a measure represents the concept it claims to measure | Assumed from the variable name | Psychometrics |
| External validity | The degree to which results hold in a different population or setting | Confused with applicability, which is a judgement about a specific context | Design property |
| Applicability | Whether the evidence matches the review question or intended use | Merged with risk of bias into one score | Separate judgement |
| Risk of bias | The likelihood that a result deviates systematically from the truth | Merged with applicability | Separate judgement |
| Evidence gap | A link with no supporting artifact | Treated as a negative finding | Assessment category |
| Retraction | Withdrawal of a published work by the publisher | Treated as rare and irrelevant to device evidence | Publication status |
| Correction / erratum | A published fix to a specific error | Assumed to be minor; some corrections change a conclusion | Publication status |
| Audit trail | A chronological record of creation, modification, and deletion of records | Treated as a log file rather than a controlled record | 21 CFR 11.10(e) |
| Substantial equivalence | The 510(k) standard for a new device compared with a predicate | Treated as proof of clinical benefit | FDA regulatory |
| Clinical utility | Evidence that using the model changes decisions or outcomes for the better | Confused with accuracy | Decision-analytic |

Two vocabulary distinctions do more work than the rest and deserve separate treatment.

**Endpoint versus outcome.** An outcome is any variable you measured. An endpoint is the
variable you committed to before analysis, with a definition, a time window, and a summary
measure. A study can measure forty outcomes and have one endpoint. When a paper reports a
positive result on an outcome that was not the endpoint, the correct description is "positive
secondary outcome," not "the study met its endpoint." Reviewers who do not make this
distinction accept post-hoc findings as confirmatory.

**Reporting deficiency versus methodological flaw.** A reporting deficiency means the reader
cannot verify the link. A methodological flaw means the link does not hold even if fully
reported. The test is counterfactual: if the authors disclosed everything, would the link
still fail? If the answer is no, it is a reporting deficiency and the remedy is disclosure. If
the answer is yes, it is a flaw and the remedy is new evidence. Apply this test before writing
any criticism. It prevents the two most common appraisal errors: treating a missing detail as
a fatal flaw, and treating a disclosed flaw as a documentation issue.

A third distinction is subtler but appears in every regulatory review. **Applicability and risk
of bias are orthogonal.** A study can have low risk of bias and poor applicability: a
well-conducted randomised trial in a population that does not resemble yours. It can have high
risk of bias and good applicability: a flawed retrospective study in exactly your setting.
Collapsing them into a single quality score destroys the information a decision-maker needs.

## 5. The mechanics, step by step

This procedure takes a claim and produces a scored traceability record. It is designed to be
run by one reviewer in two to four hours for a single study, and by a team across a dossier in
two to five days. Steps 1 through 9 are sequential. Step 10 is the decision.

### Step 1. Capture the claim verbatim

**Input:** the sentence or sentences under evaluation, copied exactly, with a source locator
(abstract, press release, label, proposal, slide number).

**Output:** a claim record containing the verbatim text, the source type, the date, and the
claimed direction and magnitude.

**Rationale:** Paraphrasing at the start loses the modifiers that carry the claim. "May improve"
and "improves" are different claims. "In a retrospective single-centre cohort" and "in clinical
practice" are different claims. Copy first, interpret later.

**Branch point:** if the claim appears in more than one artifact with different wording (label,
paper, website), record each version as a separate claim and note the discrepancy. Divergent
wording across artifacts is itself a finding.

### Step 2. Decompose the claim into a structured statement

**Input:** the verbatim claim.

**Output:** a six-part decomposition: population, intervention or model, comparator, endpoint,
time horizon, and effect direction with magnitude.

**Rationale:** Decomposition exposes missing parts. A claim that does not name a comparator
cannot support a relative claim. A claim that does not name a time horizon cannot support a
temporal claim. The decomposition is the skeleton that later steps attach evidence to.

**Branch point:** if the claim cannot be decomposed because it is genuinely vague ("improves
care"), record it as an unverifiable claim. Do not invent the missing parts on the author's
behalf. An unverifiable claim is a finding, not a gap in your analysis.

### Step 3. Specify the endpoint and estimand

**Input:** the decomposed claim and the study methods.

**Output:** an endpoint definition with variable, threshold, time window, and summary measure;
an estimand statement with the target population and the handling of intercurrent events.

**Rationale:** The endpoint is where most chains break silently. "Sepsis" as a word and "sepsis
as defined by the Centers for Disease Control and Prevention surveillance criteria within a
six-hour window" as a variable are different endpoints. The estimand forces the author to say
what population the effect applies to and what happens to patients who discontinue, die, or
receive a rescue intervention.

**Branch point:** three cases arise, and they require different handling.

| Case | Evidence | Action |
|---|---|---|
| Endpoint pre-specified and matches claim | Registered protocol or statistical analysis plan with the same definition | Record as documented; proceed |
| Endpoint defined in methods but not pre-specified | Methods define the endpoint; no protocol available | Record as inferable; flag the absence of pre-specification |
| Endpoint changed or absent | Registered endpoint differs; or only a narrative definition exists | Record as missing or as a post-hoc change; go to Step 8 |

**Edge case, selective outcome reporting:** compare the outcomes listed in the protocol or
registry against the outcomes reported. If a registered primary outcome is absent from the
report and a secondary outcome is promoted to the headline, record a selective reporting
finding. The test is not whether the reported outcome is valid; it is whether the selection was
influenced by the result. Absence of a registry entry is not proof of selection, but it removes
the ability to rule it out.

### Step 4. Map the endpoint to a dataset variable

**Input:** the endpoint definition and the data dictionary or methods description.

**Output:** the exact field or fields, their source system, their collection timing, and their
missingness.

**Rationale:** This is the link where convenience substitutes for construct validity. A
discharge diagnosis code, a laboratory value, a radiology report label, and a prospective
adjudication are four different measurements of the same nominal condition, with different
error structures. The mapping must name which one was used and when it became available
relative to the prediction time.

**Branch point:** if the endpoint variable is only available after the prediction window (for
example, a discharge code used to label a prediction made on admission), record the leakage
risk. If the variable is a proxy (billing code for disease), record the proxy and the direction
of likely misclassification.

**Edge case, conflicting reference standards:** when two reference standards are available (for
example, expert adjudication and a registry), record both, the agreement between them, and
which one the analysis used. If the reported performance differs by reference standard, the
choice is a degree of freedom that must be disclosed. A study that reports only the more
favourable reference standard without saying so has a reporting problem at minimum.

### Step 5. Verify dataset provenance and cohort construction

**Input:** the dataset description, inclusion and exclusion criteria, and the reported counts.

**Output:** a cohort accounting that reconciles the source population to the analysis
population, with every exclusion quantified.

**Rationale:** Link C fails when the analysis population is not the described dataset. The
common forms are silent exclusions (patients with missing labels dropped), unit switches
(encounters analysed as patients), and temporal leakage (patients in both training and test).
A reconciling flow is the only way to detect them.

**Branch point:** if the counts do not reconcile, do not assume error. Record the discrepancy
and its size. A 2 percent discrepancy may be a reporting convention; a 30 percent discrepancy
is a different study.

**Edge case, protocol deviations:** in a trial, deviations from the registered protocol
(eligibility violations, delayed consent, unblinding) must be listed with counts and handling.
The analysis population (intention-to-treat, per-protocol, as-treated) determines what the
result means. A per-protocol result in a trial with substantial deviations estimates the effect
in a population that may not exist in practice.

### Step 5a. Build the reconciliation record

**Input:** every count in the participant flow, the data dictionary, and the analysis output.

**Output:** a table that accounts for every unit between the source population and the
denominator of each reported estimate.

**Rationale:** Reviewers accept flow diagrams at face value. Reconstructing the arithmetic
turns the diagram into a test. The table below is the record to fill in; the difference column
must be zero or explained.

| Stage | Count | Source | Difference from previous | Explanation |
|---|---:|---|---:|---|
| Source population (all eligible records) | | | | |
| After exclusions for eligibility | | | | |
| After exclusions for data availability | | | | |
| After exclusions for label availability | | | | |
| Analysis population for estimate 1 | | | | |
| Analysis population for estimate 2 | | | | |

Two rules apply. First, every reported estimate has its own denominator; if two estimates in
the same paper use different denominators, the paper has two analysis populations and the
reviewer must record both. Second, a difference that is not explained is a link C defect until
proven otherwise, even when the direction of the difference would improve performance.

A worked reconciliation for a hypothetical screening study: 1,200 invited, 900 enrolled, 8 did
not complete procedures, 73 lacked a gradable reference image, 819 analysed. The chain from
1,200 to 819 has four steps and three reasons. The 73 without a gradable reference image are
the interesting group, because their exclusion may be related to disease severity. The reviewer
should ask whether the excluded group differed, and if the study does not say, record the
question as open rather than assuming the exclusion was harmless.

### Step 6. Map the dataset to the analysis

**Input:** the statistical analysis plan or methods, the code if available, and the reported
tables.

**Output:** a mapping from each reported estimate to the analysis that produced it: model
specification, validation design, metric definition, and confidence-interval method.

**Rationale:** A number without its analysis is uninterpretable. An AUROC computed on a random
split and an AUROC computed on a temporally held-out cohort are both AUROCs and mean different
things. The mapping forces the reviewer to name the design.

**Branch point:** if code is available and runs, the link is testable. If code is unavailable,
the link is inferable at best. Record which.

**Edge case, post-hoc changes:** any analysis not in the plan is post-hoc. Post-hoc analyses
are legitimate and often informative, but they are hypothesis-generating. Record them as such
and do not let them carry the confirmatory claim. The common pattern is a pre-specified
primary analysis that is null, followed by a subgroup or threshold analysis that is positive
and is reported as the finding.

### Step 7. Verify the reported number against the analysis output

**Input:** the reported table or figure and the analysis output, if reproducible.

**Output:** a pass, a discrepancy with magnitude, or an untestable status.

**Rationale:** This is the reproducibility link. It is the only link that can be tested
mechanically. A discrepancy of rounding is not a finding. A discrepancy that changes a
conclusion is.

**Branch point:** if the code reproduces the number within a stated tolerance, record the
tolerance and the environment. If it does not, record the discrepancy and whether the reported
number is more or less favourable than the reproduced one. Direction matters: a reported number
more favourable than the reproducible number is a stronger signal than the reverse.

**Edge case, retracted studies:** before relying on any source, check its current publication
status. A retraction or a correction that changes a result invalidates the link regardless of
the quality of the original work. Record the check date. This is a two-minute step that is
skipped more often than any other.

### Step 8. Verify the interpretation against the number

**Input:** the discussion, abstract conclusion, and any derived recommendation.

**Output:** a list of interpretive statements, each marked as licensed, overreaching, or
unsupported.

**Rationale:** Interpretation is where a local result becomes a general claim. The common
overreaches are generalisation across populations, substitution of discrimination for clinical
utility, and causal language applied to observational associations.

**Branch point:** three outcomes, handled differently.

| Interpretation | Test | Action |
|---|---|---|
| Licensed | The statement is within the scope of the design and the estimate | Record as supported |
| Overreaching | The statement extends beyond the population, design, or metric | Record as a finding; state the boundary that was crossed |
| Unsupported | No estimate in the study bears on the statement | Record as a finding; treat the claim as unverified |

**Edge case, subgroup claims:** a subgroup claim requires three things: a pre-specified
subgroup, an interaction test or a stated rationale for within-subgroup comparison, and
adequate precision. A forest plot with a significant result in one subgroup and non-significant
results in others is not evidence of a subgroup effect unless the interaction is tested.
Record the subgroup size, the event count, and the confidence interval. Small subgroups produce
wide intervals and unstable estimates, and the temptation to report the point estimate without
the interval is the mechanism of the error.

### Step 9. Verify the recommendation against the deployment context

**Input:** the intended use statement and the local context (prevalence, case mix, workflow,
available actions).

**Output:** an applicability judgement: low, moderate, or high concern, with the specific
mismatch named.

**Rationale:** Applicability is a separate judgement from risk of bias. A study can be
methodologically sound and inapplicable. The judgement requires local numbers, not just study
numbers: the same sensitivity and specificity produce a different positive predictive value at
a different prevalence.

**Branch point:** if prevalence is not reported, it can sometimes be derived from the cohort
counts. If it cannot, record applicability as unclear rather than assuming the study prevalence
transfers.

### Step 10. Score the chain and decide

**Input:** the link states from Steps 1 through 9.

**Output:** a chain score, a decision, and the conditions under which the decision would change.

**Rationale:** The score forces consistency across reviewers and makes disagreement visible.
The conditions make the decision auditable: a reviewer who disagrees can point to the
condition that failed.

Score each link on a three-point scale and take the minimum as the chain score, not the mean.
The minimum is the correct aggregation because the chain is only as strong as its weakest
load-bearing link.

| Link | Documented (2) | Inferable (1) | Missing (0) |
|---|---|---|---|
| A claim to endpoint | Endpoint named and matches claim | Endpoint implied by methods | No endpoint or mismatch |
| B endpoint to variable | Field, source, timing specified | Field named, timing unclear | No mapping |
| C dataset to analysis | Cohort flow reconciles | Minor unreconciled difference | Major discrepancy or leakage |
| D analysis to number | Code reproduces | Code unavailable, method clear | Method unclear or number unreproducible |
| E number to interpretation | Conclusions within scope | Minor overreach | Material overreach |
| F recommendation to context | Applicability addressed with local numbers | Applicability discussed qualitatively | No applicability discussion |

### Decision table

| Chain score | Load-bearing links | Decision | Typical next action |
|---:|---|---|---|
| 2 | All documented | Accept | Cite in dossier with the source locator |
| 1 | A, B, C documented; D or E inferable | Accept with conditions | Request code or clarification before contracting |
| 1 | A or B inferable | Conditional | Independent validation required before use |
| 0 | Any of A, B, C missing | Reject the claim | Do not use; seek alternative evidence |
| 0 | D missing but A-C documented | Provisional accept | Recompute from supplied data; verify before decision |

### The two-claim split

A single study often supports two different claims with different chain structures, and the
reviewer should split the record rather than score once.

The first claim is usually about performance: "the model discriminates disease at this
level." Its load-bearing links are B (label validity) and C (population). The second claim is
usually about utility: "using the model improves decisions." Its load-bearing links are A
(endpoint is a decision or outcome, not accuracy) and F (workflow and context). A study can
score 2 on the performance claim and 0 on the utility claim. Reporting a single score hides
that difference, and the single score is almost always the higher of the two because the
performance claim is the one the study was designed to support.

The practical rule: one claim record per sentence that a buyer could act on. If the dossier
contains a performance sentence and a utility sentence, build two records. The records can
share links B and C but not A, D, E, or F.

### Edge case summary

The seven recurring edge cases and their handling:

1. **Incomplete reporting.** Mark the link inferable, not missing, when the underlying work is
   plausibly done. Request the specific missing item by name.
2. **Selective outcome reporting.** Compare registry to report. Record the discrepancy in
   primary outcomes as a finding independent of the result's direction.
3. **Subgroup claims.** Require pre-specification plus interaction test plus precision. Without
   all three, label as hypothesis-generating.
4. **Protocol deviations.** Record counts and the analysis population. A per-protocol result
   with large deviations is not the trial's randomised effect.
5. **Post-hoc changes.** Label as post-hoc and do not let it carry the confirmatory claim. It
   can still be the basis for a new study.
6. **Conflicting reference standards.** Record both and the agreement between them. Performance
   that depends on the reference standard is conditional evidence.
7. **Retracted or corrected studies.** Check status at the date of use. A correction that
   changes a number invalidates downstream links built on that number.

The edge cases co-occur, and the order of handling matters. Check retraction and correction
first, because a retracted source removes every link downstream of it and the rest of the
procedure is wasted. Check selective outcome reporting second, because it changes which
endpoint is the endpoint and therefore what every later step is about. Check protocol
deviations and post-hoc changes third, because they determine the analysis population and the
status of the analysis. Check reference standards fourth, because they determine the label.
Check incomplete reporting and subgroup claims last, because both are usually about how a
result is described rather than what was done.

Two edge cases look similar and are not. A post-hoc change to the analysis is a change after
seeing data. A protocol deviation is a departure from a plan that was fixed in advance. The
first affects what the result means; the second affects who is in the result. A study can have
both, and the reviewer should record them as separate findings with separate remedies.

A third confusion is worth naming. Selective outcome reporting is a property of the
publication, not of the analysis. All of the measured outcomes may be valid and the analysis
sound; the problem is that the reader cannot see the ones that were not reported. This is why
the remedy is access to the protocol, not reanalysis.

### What this procedure does not do

It does not estimate an effect, pool studies, or judge clinical benefit. It produces a
traceability record. Pooling and certainty assessment are covered in topic 02 (GRADE) and
topic 03 (appraisal). The record from this procedure is an input to those.

## 6. Worked example

The case is the IDx-DR system, an autonomous artificial intelligence (AI) system for detecting
diabetic retinopathy, evaluated in a prospective study in primary care. The primary report is
Abràmoff and colleagues, "Pivotal trial of an autonomous AI-based diagnostic system for
detection of diabetic retinopathy in primary care offices," *npj Digital Medicine* 2018. The
claim under evaluation comes from the paper's conclusion and from how such a system is
typically described in procurement: the system can autonomously screen for more than mild
diabetic retinopathy and decide whether to refer, without a clinician reading the image.

### Claim record

Verbatim claim, condensed: the system's diagnostic accuracy for detecting more than mild
diabetic retinopathy exceeded pre-specified endpoints in a prospective primary care study, and
the system is the first autonomous AI diagnostic system authorised for this use.

Decomposition:

| Element | Content |
|---|---|
| Population | Adults with diabetes attending primary care sites, no known retinopathy history required |
| Intervention | Autonomous AI system acquiring and grading retinal images |
| Comparator | Reading centre grading of the same images (reference standard) |
| Endpoint | Sensitivity and specificity for more than mild diabetic retinopathy, against the reference standard |
| Time horizon | Single visit |
| Direction and magnitude | Sensitivity and specificity above pre-specified thresholds; exact values below |

### Link-by-link trace

**Link A, claim to endpoint.** The claim is about diagnostic accuracy for a defined condition.
The endpoint is sensitivity and specificity for more than mild diabetic retinopathy. The
construct matches: detection accuracy, not workflow effect, not treatment outcome. Score 2,
documented. Note what the claim does not include: it does not claim improved visual outcomes,
and readers who infer that are overreaching on their own.

**Link B, endpoint to variable.** The endpoint is operationalised as the grade assigned by a
reading centre using the Early Treatment Diabetic Retinopathy Study (ETDRS) severity scale,
with "more than mild" as the threshold. The field, the grading protocol, and the timing are
specified. The system's output is a binary referral decision mapped to that threshold. Score 2.
The residual issue is that the reference standard is human grading, which has its own
inter-grader variability; the study reports adjudication, which mitigates but does not remove
it. This is a limit, not a break.

**Link C, dataset to analysis.** The study was prospective, multicentre, in primary care
settings. The analysis population required images of sufficient quality; the imageability rate
is reported, and patients whose images could not be graded were accounted for. Score 2. Two
questions a reviewer should still ask. First, were the sites representative of primary care
generally, or were they selected for camera installation and staff training? Second, what
happened to participants who did not attend or whose images failed capture? Both are
applicability questions, not bias questions. Applicability concern: moderate, because the
deployment context includes sites without the same image-acquisition support.

**Link D, analysis to number.** The report gives sensitivity and specificity with confidence
intervals against the reference standard. The analysis is a straightforward cross-tabulation.
Score 2 for transparency; score 1 for reproducibility, because the code and image data are not
public. Under the rubric, a documented method with unavailable code is inferable. The chain
score is not reduced because the method is simple enough to recompute from the reported counts,
which is a judgement the reviewer must state.

**Link E, number to interpretation.** The conclusion states that the system exceeded
pre-specified accuracy endpoints. That is licensed by the numbers. The interpretation does not
claim clinical outcome benefit. Score 2. A derived procurement sentence, "this replaces the
need for a retinal specialist," would be overreach: the study measured accuracy against a
reading centre, not the downstream effect of removing specialist review, and it did not measure
whether the system misses disease that a clinician would have caught through clinical context.
Score 1 for that derived sentence, recorded separately.

**Link F, recommendation to context.** The intended use is primary care, with image acquisition
by trained staff. Applicability to a different setting depends on camera, staff training, and
the local prevalence of disease, which affects positive predictive value. The paper reports the
study prevalence. A local deployment should recompute positive predictive value at local
prevalence. Score 2 for the study's own scope; moderate concern for transfer to settings that
differ on those three factors.

### Scoring

| Link | Score | Justification |
|---|---:|---|
| A | 2 | Endpoint matches construct |
| B | 2 | ETDRS reference standard, adjudicated |
| C | 2 | Prospective, multicentre, accounted imageability |
| D | 1 | Method clear, code not available |
| E | 2 | Conclusions within scope |
| F | 2 | Intended use stated; local transfer conditional |

Minimum score: 1. No load-bearing link is missing. Decision under the table: accept with
conditions, where the condition is that local deployment recomputes predictive value at local
prevalence and verifies image-acquisition equivalence. That is a reasonable outcome for a
well-reported prospective study with a simple analysis.

### Quantifying link F

The study reported sensitivity 87.2 percent and specificity 90.7 percent for more than mild
diabetic retinopathy, with a prevalence of 23.8 percent in the analysed cohort (198 of 819).
Those three numbers determine the positive predictive value, and the positive predictive value
is what the clinic experiences.

The arithmetic, using the reported point estimates as if they were exact:

```text
true positive rate  = 0.872 x 0.238 = 0.2075
false positive rate = (1 - 0.907) x (1 - 0.238) = 0.093 x 0.762 = 0.0709
PPV = 0.2075 / (0.2075 + 0.0709) = 0.2075 / 0.2784 = 0.745

At 5 percent local prevalence:
TP = 0.872 x 0.05 = 0.0436
FP = 0.093 x 0.95 = 0.0884
PPV = 0.0436 / 0.1320 = 0.330
NPV = (0.907 x 0.95) / ((0.907 x 0.95) + (0.128 x 0.05)) = 0.8617 / 0.8681 = 0.993
```

| Local prevalence | PPV | Patients referred to find one case |
|---:|---:|---:|
| 5% | 33.0% | 3.0 |
| 10% | 51.0% | 2.0 |
| 23.8% (study) | 74.5% | 1.3 |
| 40% | 86.1% | 1.2 |

The numbers are approximations: the published estimates use a Firth-adjusted logistic model and
the confidence intervals are not symmetric. For a decision memo, quote the published interval
and present the table as a sensitivity analysis, not as a replacement estimate.

The table is the concrete meaning of link F. The model did not change between rows. The
population did. A procurement memo that cites 87.2 percent sensitivity without the prevalence
context has documented link D and broken link F, and the break is invisible unless someone
does this arithmetic.

### Two diagnostics that would change the verdict

**Diagnostic 1: the cohort was assembled retrospectively from stored images.** Suppose the
study had taken images collected during routine care and graded them later, rather than
acquiring them prospectively under the protocol. Link C would drop to 0 or 1: the population
would be whatever was photographed, image quality would reflect routine practice rather than
protocol, and the spectrum of disease would reflect who was referred. The chain score would
become 0, the decision would move to "independent validation required," and the appraisal
would need to ask whether the system's imageability rate in routine practice matches the
reported rate. The verdict changes from accept-with-conditions to reject-for-now.

**Diagnostic 2: the reference standard was a single grader without adjudication.** Suppose the
reading centre grading had been done once, by one grader, without adjudication or masking to
the system output. Link B would drop to 1. The performance estimate would then be conditional
on grader reliability, and a reviewer could not distinguish system error from grader error.
The chain score would become 1 and the decision would remain conditional, but the condition
would change: the request would be for a reader study with adjudication, not for local
prevalence recalculation. The verdict does not flip, but the remedy does. This is the practical
value of scoring links separately rather than giving one quality score.

### Vary one assumption: sample size

Take the same study and reduce the sample to a single site with 100 participants instead of a
multicentre study with several hundred. Link A and B are unchanged. Link C stays documented but
the precision collapses: sensitivity confidence intervals widen to a range that spans clinically
important differences. Link D becomes easier to reproduce and Link E becomes more fragile,
because a conclusion stated as "exceeded the endpoint" may depend on a point estimate whose
interval includes the threshold. Under the rubric, Link E drops to 1 because the interpretation
is not robust to the interval. The chain score is 1, the decision is conditional, and the
condition is a larger prospective study. The lesson: chain scoring is not independent of
precision. A documented link with an imprecise estimate is still documented, but the
interpretation link inherits the imprecision.

## 7. Code, query, or calculation

The script below builds a traceability record for a set of claims, scores each link, applies
the decision table, and prints a memo. It uses only the standard library and is deterministic.
It also demonstrates the minimum-score aggregation and the reporting-deficiency test.

```python
"""Evidence chain scorer.

Takes a list of claims, each with per-link states, and produces a decision memo.
States: 2 = documented, 1 = inferable, 0 = missing.
Load-bearing links are A (claim/endpoint), B (endpoint/variable), C (dataset/analysis).
"""

from dataclasses import dataclass, field
from typing import Dict, List

LINKS = ["A", "B", "C", "D", "E", "F"]
LOAD_BEARING = {"A", "B", "C"}
LINK_NAMES = {
    "A": "claim to endpoint",
    "B": "endpoint to dataset variable",
    "C": "dataset to analysis population",
    "D": "analysis to reported number",
    "E": "number to interpretation",
    "F": "recommendation to context",
}


@dataclass
class Claim:
    cid: str
    text: str
    scores: Dict[str, int]
    notes: Dict[str, str] = field(default_factory=dict)

    def chain_score(self) -> int:
        return min(self.scores[l] for l in LINKS)

    def missing_load_bearing(self) -> List[str]:
        return [l for l in sorted(LOAD_BEARING) if self.scores[l] == 0]


def decide(claim: Claim) -> str:
    s = claim.chain_score()
    missing = [l for l in LINKS if claim.scores[l] == 0]
    if any(l in LOAD_BEARING for l in missing):
        return "REJECT: load-bearing link missing (" + ", ".join(missing) + ")"
    if missing:
        return "REJECT: non-load-bearing link missing (" + ", ".join(missing) + ")"
    if s == 2:
        return "ACCEPT"
    if claim.scores["D"] == 1 and all(claim.scores[l] == 2 for l in "ABCEF"):
        return "PROVISIONAL ACCEPT: recompute before decision"
    return "ACCEPT WITH CONDITIONS"


def deficiency_or_flaw(claim: Claim, link: str, would_hold_if_disclosed: bool) -> str:
    """Counterfactual test from section 4."""
    if claim.scores[link] >= 2:
        return "not a finding"
    if would_hold_if_disclosed:
        return "reporting deficiency: request disclosure"
    return "methodological flaw: requires new evidence"


claims = [
    Claim(
        cid="C1",
        text="Autonomous screening accuracy exceeds pre-specified endpoints in primary care",
        scores={"A": 2, "B": 2, "C": 2, "D": 1, "E": 2, "F": 2},
        notes={"D": "code unavailable; method reproducible from reported counts"},
    ),
    Claim(
        cid="C2",
        text="System replaces retinal specialist review",
        scores={"A": 1, "B": 2, "C": 2, "D": 1, "E": 0, "F": 1},
        notes={"E": "no workflow or outcome estimate supports replacement"},
    ),
    Claim(
        cid="C3",
        text="Model reduces sepsis mortality by 20 percent",
        scores={"A": 0, "B": 1, "C": 1, "D": 0, "E": 0, "F": 0},
        notes={"A": "endpoint is alert timing, not mortality"},
    ),
]

print(f"{'ID':<4}{'score':<7}{'decision'}")
print("-" * 78)
for c in claims:
    print(f"{c.cid:<4}{c.chain_score():<7}{decide(c)}")

print()
print("Link detail for C3:")
c3 = claims[2]
for l in LINKS:
    print(f"  {l} {LINK_NAMES[l]:<38} {c3.scores[l]}  {c3.notes.get(l, '')}")

print()
print("Deficiency test, C2 link E (disclosure would not fix):")
print("  " + deficiency_or_flaw(claims[1], "E", would_hold_if_disclosed=False))
print("Deficiency test, C1 link D (disclosure would fix):")
print("  " + deficiency_or_flaw(claims[0], "D", would_hold_if_disclosed=True))

# Aggregate across a dossier: count decisions and compute the share of claims accepted.
from collections import Counter
decisions = Counter(decide(c).split(":")[0] for c in claims)
total = sum(decisions.values())
print()
print("Dossier summary:", dict(decisions), f"of {total} claims")
print(f"Unconditional accept share: {decisions.get('ACCEPT', 0)/total:.1%}")
```

Expected output:

```text
ID  score  decision
------------------------------------------------------------------------------
C1  1      PROVISIONAL ACCEPT: recompute before decision
C2  0      REJECT: non-load-bearing link missing (E)
C3  0      REJECT: load-bearing link missing (A, D, E, F)

Link detail for C3:
  A claim to endpoint                      0  endpoint is alert timing, not mortality
  B endpoint to dataset variable           1
  C dataset to analysis population         1
  D analysis to reported number            0
  E number to interpretation               0
  F recommendation to context              0

Deficiency test, C2 link E (disclosure would not fix):
  methodological flaw: requires new evidence
Deficiency test, C1 link D (disclosure would fix):
  reporting deficiency: request disclosure

Dossier summary: {'PROVISIONAL ACCEPT': 1, 'REJECT': 2} of 3 claims
Unconditional accept share: 0.0%
```

Notes on the script.

The decision function checks for missing links before it checks the score. That order matters.
A claim with a missing non-load-bearing link and a minimum score of zero is rejected, but the
message distinguishes it from a claim that fails on a load-bearing link, because the remedies
differ: one is usually fixed by disclosure or argument, the other requires new evidence.

`chain_score` returns the minimum across all six links, including the non-load-bearing ones.
This is deliberately conservative. A reviewer who wants to aggregate only the load-bearing
links can change the function, but should then state the rule in the memo.

The `deficiency_or_flaw` function takes the counterfactual as an argument rather than
computing it. That is honest: the counterfactual is a judgement, not a calculation. The
function only encodes the consequence of the judgement so that the memo is consistent.

The script does not check publication status, reconcile cohort counts, or compute
confidence intervals. Those are manual steps in section 5, and automating them would create a
false impression that the appraisal is mechanical.

## 8. Data, standards, and regulatory context

Traceability is not an invented virtue. It is codified in several frameworks, and the
obligations differ by regime.

**Reporting guidelines.** The TRIPOD+AI statement (Collins et al., *BMJ* 2024;385:e078378,
published 16 April 2024) requires, among other items, a clear description of the intended use,
the source of data, the outcome definition, the participant flow, and the model's performance
with measures of uncertainty. It supersedes the 2015 TRIPOD checklist, which the authors state
should no longer be used. The 27-item checklist (52 subitems) is the closest thing to a
standard schema for links A through E in a prediction-model study. It is a reporting
guideline, not a regulatory requirement, unless a journal or regulator adopts it.

**Risk-of-bias tools.** PROBAST+AI (Moons et al., *BMJ* 2025;388:e082505, published 24 March
2025) assesses model development quality and evaluation risk of bias across four domains:
participants and data sources, predictors, outcomes, and analyses. It has 16 signalling
questions for development and 18 for evaluation, plus six applicability items covering the
first three domains. The tool is the operational form of the question "is this link strong
enough to carry the claim?" Its separation of development quality from evaluation risk of bias
mirrors the chain's separation of link D from links A through C.

**Device regulation, United States.** The Food and Drug Administration (FDA) regulates AI-enabled
device software functions through the 510(k), De Novo, and premarket approval pathways. The
Quality Management System Regulation (QMSR), published 2 February 2024 and effective 2 February
2026, incorporates ISO 13485:2016 by reference and requires design controls, traceability, and
document control. Design controls are the regulated ancestor of the evidence chain: they require
that each design input be linked to a design output and verified. The FDA's draft guidance
"Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing
Submission Recommendations" (issued 7 January 2025, docket FDA-2024-D-4488, comment period
closed 7 April 2025) describes documentation expectations across the total product life cycle
and explicitly discusses model cards as a labelling format. It was issued as a draft and should
be cited as such.

**Device regulation, European Union.** The Medical Device Regulation (EU) 2017/745 and the In
Vitro Diagnostic Regulation (EU) 2017/746 require clinical evaluation and post-market clinical
follow-up. The clinical evaluation report is, structurally, an evidence chain document: it
links claims to clinical data to a benefit-risk conclusion.

**Clinical trial records.** The International Council for Harmonisation (ICH) E6 Good Clinical
Practice guideline and ICH E9(R1) on estimands constrain how endpoints and analysis populations
are defined. A trial that changes its primary endpoint after unblinding must document the
change, and the change is visible to a reviewer.

**Data integrity.** The ALCOA principles (attributable, legible, contemporaneous, original,
accurate) and the extended ALCOA+ set (complete, consistent, enduring, available) constrain how
the raw records behind the chain are kept. The FDA's 1999 guidance on computerized systems in
clinical trials and its October 2024 question-and-answer guidance on electronic records and
signatures describe audit trail expectations: who changed what, when, and why, with old and new
values retained.

**What the standards do not do.** None of these frameworks guarantees that a claim is true.
They guarantee that the claim is documented well enough to be checked. A study can be fully
TRIPOD+AI compliant and still estimate the wrong thing, because the checklist asks whether the
endpoint was defined, not whether it was the right endpoint. Compliance is a floor for
traceability, not a certificate of validity.

**A note on the 510(k) pathway.** Substantial equivalence does not require clinical performance
data in all cases. A device can be cleared on bench testing and comparison with a predicate.
The public summary may therefore contain no clinical study at all. Absence of a clinical study
in a 510(k) summary is not a violation; it is a property of the pathway. The 2025 review of 903
devices found that 55.9 percent reported clinical performance studies, 24.1 percent explicitly
stated none were conducted, and 19.9 percent did not specify. Those categories mean different
things and should not be pooled into "44 percent lack evidence."

## 9. Industry practice and case evidence

**What mature teams do.** In companies with a regulatory affairs function, the evidence chain
exists as a document called a clinical evaluation report, a design history file, or a
traceability matrix. It is maintained by a named owner, versioned, and reviewed at each design
review. In analytics teams without that function, the chain usually exists only in the head of
the person who built the model. The difference is not diligence; it is whether the organisation
has a place to put the artifact.

**Documented fact: the Epic Sepsis Model external validation.** Wong and colleagues evaluated a
proprietary sepsis prediction model at Michigan Medicine in 27,697 patients across 38,455
hospitalisations, using data from December 2018 to October 2019 (*JAMA Internal Medicine*
2021;181(8):1065-1070). The hospitalisation-level AUROC was 0.63 (95 percent confidence interval
0.62 to 0.64), against vendor-reported values of 0.76 to 0.83. At the operational threshold,
sensitivity was 33 percent and positive predictive value 12 percent, and the score triggered in
18 percent of hospitalisations, failing to identify 1,709 of 2,552 sepsis cases. This is a
chain break at link C and link F: the validation population and the deployment population were
not the same, and the operating threshold was not matched to the local workflow.

**Documented fact: FDA authorisation evidence.** The review by Windecker and colleagues in
*JAMA Network Open* 2025;8(4):e258052 examined 903 AI-enabled devices authorised as of 31 August
2024. Clinical performance studies were reported for 505 devices (55.9 percent); of the reported
studies, 193 (38.2 percent) were retrospective, 41 (8.1 percent) prospective non-randomised, and
12 (2.4 percent) randomised, with 259 (51.3 percent) of unspecified design. Only 82 devices
(16.2 percent) reported an AUROC. Sex-specific subgroup data were reported for 145 devices (28.7
percent) and age-specific for 117 (23.2 percent). The inference, labelled as inference: for a
large share of authorised devices, a buyer cannot reconstruct links C through E from public
documents alone.

**Documented fact: early recalls.** Lee and colleagues, *JAMA Health Forum* 2025 (doi
10.1001/jamahealthforum.2025.3172), linked 60 of 950 cleared AI-enabled devices (6.3 percent) to
182 recall events. Seventy-nine recalls (43.4 percent) occurred within 12 months of clearance,
and diagnostic or measurement errors caused 109 recalls. Lack of clinical validation was
associated with recall (odds ratio 2.8, 95 percent confidence interval 1.6 to 4.7). The
association is observational; it is consistent with a chain break at link C, but it does not
prove causation for any individual device.

**Inference: why chains break in industry.** Three mechanisms recur. First, the claim is
written by commercial or clinical leadership and the evidence is produced by a data team, with
no shared template, so the decomposition in Step 2 never happens. Second, the validation
dataset is chosen for availability rather than representativeness, which breaks link C while
leaving links A and B intact, so the study looks rigorous. Third, the operating threshold is
tuned on the validation set and reported as if pre-specified, which breaks link E and is nearly
invisible without a protocol.

**What procurement can ask for.** A one-page traceability record per claim, with the source
locator for each link. A vendor that cannot produce it is not necessarily dishonest, but the
buyer cannot verify the claim and should price that risk. This is the practical use of the
topic.

**What a good record looks like.** Teams that do this well produce a document with five
properties, and each property is visible in the artifact rather than asserted in a cover note.

- One row per claim, with the claim quoted verbatim and a locator to the artifact where it
  appears.
- One column per link, with either a locator or the word "gap" and a named owner for closing
  it.
- A version and date that match the model version under review, so that a reader can tell
  whether the record describes the model being bought.
- A separate applicability section with local numbers, not a sentence saying the population
  was similar.
- A status field per link, using the documented, inferable, missing vocabulary, so that a
  reader can see disagreement between reviewers rather than an averaged score.

Teams that do this badly produce a table with one row per study and one column called
"evidence," containing a citation. The format is not the point; the granularity is. A citation
answers "where is the study," not "which link does the study support."

**Inference: the cost curve.** The cost of building the record at design time is a few days of
a clinical affairs analyst. The cost of reconstructing it after a regulator, payer, or
procurement committee asks is weeks, because the people who built the model have moved on and
the raw artifacts may no longer be reproducible. The asymmetry is the business case for the
topic, and it is the reason quality management (topic 06) treats traceability as a design
control rather than a documentation task.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Claim drift | Abstract says "detects," methods say "flags" | Commercial wording outruns the endpoint | Buyer expects a diagnosis, gets a triage signal | Compare verbatim claim to endpoint definition |
| Endpoint substitution | "Sepsis" without a definition | Convenience variable used as proxy | Performance applies to a different condition | Demand the operational definition and time window |
| Discharge-code labelling | Label defined by code assigned at discharge | Label unavailable at prediction time | Temporal leakage inflates performance | Check label timing against prediction time |
| Unit-of-analysis switch | Dataset counts encounters, results say patients | Silent denominator change | Effect size and precision misstated | Reconcile every count in the flow |
| Random split with patient overlap | "Randomly divided into training and test" | Same patient in both folds | Optimistic performance | Demand patient-level or temporal split |
| Threshold tuned post-hoc | Operating point reported without pre-specification | Threshold selected on test data | Reported sensitivity/specificity not reproducible | Ask for the protocol or threshold rationale |
| Subgroup claim without interaction | Forest plot with one significant subgroup | Multiple comparisons | Spurious subgroup effect | Require interaction test and interval |
| Selective outcome reporting | Registered primary outcome absent | Result-dependent reporting | Publication bias in the study itself | Compare registry to report |
| Reference standard conflict | Two standards, one reported | Favourable standard chosen | Performance conditional on an undisclosed choice | Ask for agreement between standards |
| Unreconciled cohort | Counts do not add up | Silent exclusions | Population differs from description | Request a flow diagram with counts |
| Retracted or corrected source | Citation predates a correction | Downstream links built on a changed number | Claim rests on invalid evidence | Check status at date of use |
| Applicability collapse | "Validated in a large health system" | One site treated as general | Local prevalence and workflow differ | Recompute PPV at local prevalence |
| Composite endpoint inflation | Composite reported as single outcome | Components with different clinical weight pooled | Claim overstates benefit | Decompose the composite |
| Surrogate treated as outcome | Biomarker change reported as benefit | Surrogate not validated for the claim | Clinical benefit unproven | Name the surrogate and its validation status |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The claim can be decomposed | Steps 2 onward need parts to attach evidence to | Claim is genuinely vague or metaphorical | Attempt the six-part decomposition; if it fails, record unverifiable |
| A single endpoint carries the claim | Scoring assumes one primary construct | Claim is multi-dimensional (accuracy and workflow) | Split into multiple claim records |
| Link states can be judged from documents | Reviewer has only the paper and dossier | Underlying documents unavailable or confidential | Mark inferable, not documented; request access |
| Minimum aggregation is correct | Chain is only as strong as weakest load-bearing link | Links are partially compensatory (e.g., strong C offsets weak D) | Test sensitivity of the decision to aggregation rule |
| Reported numbers correspond to the described analysis | Otherwise every number is untestable | Code unavailable or data private | Recompute from reported counts where possible |
| Publication status is stable at date of use | Retraction check is a snapshot | Later retraction or correction | Record check date; re-check before decision |
| Applicability can be judged with local prevalence | Link F needs local numbers | Local prevalence unknown | Treat applicability as unclear, not low |
| Reviewers apply the rubric consistently | Scores are compared across reviewers | Undefined anchors for 0/1/2 | Pilot on two studies; compute agreement |
| Documented implies correct | Score 2 means verifiable, not true | Well-documented but wrong analysis | Separate verification (Step 7) from documentation |
| The chain is linear | Diagram is a path | Iterative model development with feedback loops | Record version and date for each link |

## 12. Exercises

**Exercise 1. Decompose three claims (recall and application, 25 minutes).**
Deliverable: a table with six columns (population, intervention, comparator, endpoint, horizon,
direction) for three claims taken from one vendor website, one abstract, and one regulatory
summary. Acceptance criteria: every cell either filled with a quoted phrase and locator, or
marked "not stated"; no invented content. Time: 25 minutes.

**Exercise 2. Trace one number (application, 45 minutes).**
Deliverable: a one-page trace from a reported AUROC in a published paper back to the dataset
description and the validation design, with a source locator for each hop. Acceptance criteria:
the trace names the dataset, the split, the unit of analysis, and the metric definition; any hop
that cannot be completed is named as missing rather than glossed. Time: 45 minutes.

**Exercise 3. Reporting deficiency or methodological flaw (critique, 30 minutes).**
Deliverable: five criticisms of a published study, each classified with the counterfactual test
and a one-sentence justification. Acceptance criteria: at least two of each category; the
classification follows from the test, not from severity. Time: 30 minutes.

**Exercise 4. Detect selective outcome reporting (critique, 40 minutes).**
Deliverable: a comparison table of registered outcomes versus reported outcomes for one trial
with a public registration, with the primary outcome change flagged. Acceptance criteria: uses
the registry entry date and the publication date; distinguishes a change from an unreported
outcome. Time: 40 minutes.

**Exercise 5. Build a traceability matrix (design, 60 minutes).**
Deliverable: a matrix for a model you have built or studied, with one row per claim and one
column per link, populated with locators. Acceptance criteria: at least five claims; every
load-bearing link has a locator or an explicit gap; the matrix fits on two pages. Time: 60
minutes.

**Exercise 6. Score two studies and compare (application, 50 minutes).**
Deliverable: two completed scoring tables and a paragraph explaining any difference in
decision. Acceptance criteria: scores are anchored to quoted evidence; the minimum rule is
applied; disagreement between the two studies is attributed to specific links. Time: 50 minutes.

**Exercise 7. Write the procurement memo (design, 45 minutes).**
Deliverable: a 400-word memo recommending accept, conditional accept, or reject, with the three
conditions that would change the recommendation. Acceptance criteria: no claim without a
locator; conditions are specific and testable; applicability is addressed separately from risk
of bias. Time: 45 minutes.

**Exercise 8. Design the check (design, 30 minutes).**
Deliverable: a two-minute pre-use checklist that would have caught one of the failures in
section 10, written as yes/no questions. Acceptance criteria: each question maps to a link;
the checklist can be run without the original analyst present. Time: 30 minutes.

## 13. Self-check answers

**A1. What is the difference between an endpoint and an outcome?** An outcome is any measured
variable. An endpoint is the pre-specified variable, with definition, time window, and summary
measure, that the study commits to for the primary evaluation. A study can have many outcomes
and one endpoint. Reporting a positive non-endpoint outcome as the finding is a secondary or
post-hoc result, not confirmation.

**A2. State the counterfactual test for a reporting deficiency.** If the authors disclosed
everything, would the link still fail? If no, it is a reporting deficiency. If yes, it is a
methodological flaw.

**A3. Why is the minimum the right aggregation for link scores?** Because the chain's ability to
support a claim is limited by its weakest load-bearing link. A strong dataset does not repair a
label that measures the wrong construct. Averaging allows strong links to mask the one that
matters.

**A4. A study reports AUROC 0.91 on a random split and 0.78 on a temporal split. Which number
supports the claim, and what does the other number tell you?** The temporal number supports a
claim about future patients at the same site; the random-split number is optimistic because
patients may appear in both folds. The gap between them is a measure of how much the random
split inflated performance.

**A5. A trial's registered primary outcome is 30-day readmission; the paper reports a
significant reduction in length of stay and calls it the main finding. What is the finding?**
Selective outcome reporting, unless the change was documented before unblinding. The length of
stay result is a secondary or post-hoc finding and is hypothesis-generating.

**A6. What are the three requirements for a subgroup claim?** Pre-specification, an interaction
test (or a stated rationale for within-subgroup comparison), and adequate precision with the
interval reported.

**A7. Why is applicability separate from risk of bias?** Because a study can be internally valid
and externally irrelevant, or internally flawed and locally relevant. Combining them loses the
information needed to decide whether to request more evidence or to reject the study.

**A8. What does it mean when a link is "inferable"?** The available evidence permits a
reasonable inference under stated assumptions, but the artifact does not let a reviewer verify
it directly. It scores 1.

**A9. A correction changes a table's confidence interval so that it now includes the null.
What happens to the chain?** Link D is re-opened because the reported number changed, and link
E may drop because the interpretation may no longer be licensed. Any downstream document citing
the original interval must be updated.

**A10. Compute the chain score.** A claim has A=2, B=2, C=1, D=1, E=2, F=1. Minimum is 1. No
load-bearing link is missing. Decision: accept with conditions, because D is not 2 and F is
not 2.

**Rubric for Exercise 5 (traceability matrix).** Full credit requires: five or more claims; a
locator for every documented link; explicit gaps for missing links; load-bearing links
identified; the matrix readable by someone who has not seen the study. Partial credit for a
matrix that covers claims but not links, or links but not locators.

**Rubric for Exercise 7 (procurement memo).** Full credit requires: a clear recommendation; the
recommendation follows from the scoring table; three conditions stated as tests; applicability
treated separately; no claim without a source. Deductions for generic criticism, unsupported
numbers, and a recommendation that does not match the scoring.

## 14. Interview questions

**Q1. Walk me through how you would verify a vendor's claim that their model reduces sepsis
mortality by 20 percent.**
Strong answer: decompose the claim; locate the endpoint and its definition; check whether the
study design could attribute mortality to the model; check the comparator; check the validation
population against the deployment population; check whether the 20 percent is relative or
absolute; check the confidence interval; state what evidence would change the conclusion. Weak
answer: asks only for the AUROC.

**Q2. What is the difference between a reporting deficiency and a methodological flaw, and why
does it matter commercially?**
Strong answer: the counterfactual test; the remedies differ (disclosure versus new study); the
commercial consequence is a documentation request versus a delay or rejection. Weak answer:
treats every gap as a flaw, which leads to over-rejection, or every flaw as a gap, which leads
to over-acceptance.

**Q3. How would you detect patient leakage in a random split?**
Strong answer: check the unit of splitting against the unit of analysis; look for duplicate
patient identifiers across folds; check whether the reported performance is implausibly high
relative to external validation; ask for the split code. Weak answer: says "random splits are
fine."

**Q4. A paper reports a significant subgroup effect. What do you check?**
Strong answer: pre-specification, interaction test, subgroup size and event count, confidence
interval width, and the number of subgroups tested. Weak answer: accepts the p-value.

**Q5. Why does a 510(k) summary often lack clinical performance data?**
Strong answer: substantial equivalence can be established without clinical data; the pathway
does not require it in all cases; absence is a property of the pathway, not necessarily a
violation. Weak answer: assumes all cleared devices were clinically validated.

**Q6. What is the weakest link in most clinical AI evidence, in your experience?**
Strong answer: names a specific link and gives a mechanism, typically label validity (link B) or
population representativeness (link C), and describes how it would test the claim. Weak answer:
gives a generic answer such as "data quality" without a link or a test.

**Q7. How do you handle a study that has been corrected but not retracted?**
Strong answer: check what changed; if a reported number changed, re-open the affected links;
update downstream documents; record the check date. Weak answer: ignores corrections.

**Q8. How would you present a conditional acceptance to a procurement committee?**
Strong answer: states the decision, the two or three conditions, the evidence that would
satisfy each, and the residual risk if the conditions are not met. Weak answer: presents a
single quality score.

**Q9. What does applicability mean when the local prevalence is unknown?**
Strong answer: applicability is unclear, not low; the reviewer should either obtain local
prevalence or present the performance as a function of prevalence. Weak answer: assumes the
study prevalence applies.

**Q10. How do you keep a traceability matrix current after deployment?**
Strong answer: version it with the model; link it to change control; update it at each model
change and each post-market review; assign an owner. Weak answer: treats it as a one-time
document.

**Q11. A claim is fully supported by the study but the study population is nothing like ours.
What do you recommend?**
Strong answer: separates risk of bias from applicability; accepts the estimate for the studied
population; states that local performance is unknown; proposes a local silent-mode evaluation
with predefined metrics before clinical use; does not describe the study as invalid. Weak
answer: rejects the study as biased, which confuses the two judgements and discards usable
information.

**Q12. What would make you distrust a reported improvement in a secondary outcome?**
Strong answer: absence from the registry, a primary outcome that was null, a threshold or
subgroup chosen after the fact, and an effect size much larger than the primary analysis would
predict. Weak answer: only checks the p-value.

**Q13. How would you explain a traceability matrix to a commercial team without sounding like
a compliance officer?**
Strong answer: frames it as the fastest way to answer the question a customer will ask, and as
the artifact that shortens diligence; shows one page; ties each link to a deal risk. Weak
answer: presents it as a regulatory obligation and loses the audience.

## 15. Key terms

- **Claim** — a sentence asserting effect, capability, or performance.
- **Endpoint** — the pre-specified measured quantity that evaluates the claim.
- **Estimand** — the precise quantity to be estimated, including population and intercurrent
  events.
- **Link** — a claimed correspondence between two chain artifacts.
- **Load-bearing link** — a link whose failure invalidates the claim.
- **Traceability matrix** — a table mapping claims to supporting artifacts.
- **Reporting deficiency** — an undocumented link that may still hold.
- **Methodological flaw** — a link that fails even if fully disclosed.
- **Selective outcome reporting** — result-dependent choice of which outcomes to report.
- **Post-hoc analysis** — an analysis specified after seeing data.
- **Protocol deviation** — a departure from the registered or approved plan.
- **Reference standard** — the method used to establish the true condition.
- **Applicability** — whether evidence matches a specific context of use.
- **Risk of bias** — whether a result deviates systematically from the truth.
- **Construct validity** — whether a measure represents the concept it claims.
- **Data provenance** — the origin and history of a data element.
- **Data lineage** — the transformation path from source to analysis variable.
- **Unit of analysis** — the entity a row represents.
- **Substantial equivalence** — the 510(k) comparison standard.
- **Clinical utility** — evidence that use changes decisions or outcomes for the better.

## 16. Sources

- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting
  clinical prediction models that use regression or machine learning methods. *BMJ*
  2024;385:e078378. doi:10.1136/bmj-2023-078378. Accessed 26 October 2026.
- Moons KGM, Damen J, Kaul T, et al. PROBAST+AI: an updated quality, risk of bias, and
  applicability assessment tool for prediction models using regression or artificial
  intelligence methods. *BMJ* 2025;388:e082505. doi:10.1136/bmj-2024-082505. Accessed 26
  October 2026.
- Abràmoff MD, Lavin PT, Birch M, Shah N, Folk JC. Pivotal trial of an autonomous AI-based
  diagnostic system for detection of diabetic retinopathy in primary care offices. *npj Digital
  Medicine* 2018;1:39. doi:10.1038/s41746-018-0040-6. Accessed 26 October 2026.
- Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
  sepsis prediction model in hospitalized patients. *JAMA Internal Medicine*
  2021;181(8):1065-1070. doi:10.1001/jamainternmed.2021.2626. Accessed 26 October 2026.
- Windecker D, Baj G, Shiri I, et al. Generalizability of FDA-approved AI-enabled medical devices
  for clinical use. *JAMA Network Open* 2025;8(4):e258052.
  doi:10.1001/jamanetworkopen.2025.8052. Accessed 26 October 2026.
- Lee B, Kramer D, Sandri F, et al. Early recalls and clinical validation gaps in artificial
  intelligence-enabled medical devices. *JAMA Health Forum* 2025.
  doi:10.1001/jamahealthforum.2025.3172. Accessed 26 October 2026.
- US Food and Drug Administration. Quality Management System Regulation (QMSR), 21 CFR Part 820,
  final rule published 2 February 2024, effective 2 February 2026.
  https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr.
  Accessed 26 October 2026.
- US Food and Drug Administration. Artificial Intelligence-Enabled Device Software Functions:
  Lifecycle Management and Marketing Submission Recommendations. Draft guidance, 7 January
  2025, docket FDA-2024-D-4488. https://www.fda.gov/media/184856/download. Accessed 26 October
  2026.
- International Council for Harmonisation. E9(R1) Addendum on Estimands and Sensitivity Analysis
  in Clinical Trials. 2019. Accessed 26 October 2026.
- EQUATOR Network. Reporting guideline library. https://www.equator-network.org/reporting-guidelines/.
  Accessed 26 October 2026.

> **Note:** Regulatory content in this topic is educational and is not legal advice. Verify
> current guidance status before relying on it in a submission.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Next topic: [`02_reporting-guidelines.md`](02_reporting-guidelines.md)
- Appraisal method: [`03_appraising-clinical-ai-evidence.md`](03_appraising-clinical-ai-evidence.md)
- Model documentation: [`04_model-cards-and-datasheets.md`](04_model-cards-and-datasheets.md)
- Module survey: [`../../course/18_quality_evaluation_reporting.md`](../../course/18_quality_evaluation_reporting.md)
- Labs: [`../../labs/`](../../labs/)
