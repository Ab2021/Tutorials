# Summarization and Grounding

> **Module 09 — Clinical NLP**
> Topic 09 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 03, 05, 07, and 08
> Version 2026.3

## 1. Why this topic exists

Clinicians spend a large share of their time reading and writing summaries. A discharge summary
compresses an admission into a document that a primary care clinician reads in minutes. A handoff
compresses a shift into a few sentences. A prior authorization letter compresses a chart into an
argument. Each is a summary, and each is produced under time pressure.

A generated summary is useful when it is faithful, complete enough for the decision, and grounded in
the source. It is dangerous when it is fluent and wrong. The danger is different from an extraction
error. An extraction error produces a missing or mislabelled fact that a downstream rule may catch.
A summary error produces a sentence that a clinician reads as the record. If the sentence says the
patient was started on a beta blocker and the patient was not, the clinician may continue a
medication that does not exist, or fail to start one that does.

The measurement problem is severe. Summaries have no single correct form. Two clinicians writing the
same discharge summary produce different selections, different wording, and different emphasis, and
both can be correct. Metrics that compare a generated summary with a reference summary, such as
ROUGE or BLEU, reward lexical overlap and penalize legitimate paraphrase. They correlate poorly with
clinician judgment, which is why current evaluations use claim-level review rather than similarity.

The reported hallucination rates illustrate the definition problem. A 2024 blinded study of
discharge summaries reported hallucination rates of 0.23 per summary for a model and 0.31 for
physicians. A separate 2024 framework reported a 1.47 percent hallucination rate with 44 percent of
those rated major, and a 3.45 percent omission rate with 17 percent major. An evaluation of emergency
department discharge summaries reported hallucinations in 42 percent of cases. These numbers are not
comparable: they use different units (per summary, per sentence, per case), different definitions
(any unsupported statement versus a clinically meaningful fabrication), and different source
conditions. A reader who compares them without the definitions will draw the wrong conclusion.

Grounding is the design response. A grounded summary cites the source span for each claim, and the
citation can be checked. An extractive-first approach selects sentences from the source before
rewriting. An abstention rule declines to state a claim when the evidence is insufficient. A human
review step catches what the automatic checks miss. None of these eliminates error; together they
make the error visible and bounded.

This topic covers the summarization pipeline, the claim unit, grounding and citation, omission and
hallucination measurement, abstention, human review, and the regulatory context. It treats a
summary as a set of claims with evidence, not as a document with a similarity score.

The commercial stakes are high. Documentation burden is a named driver of clinician burnout, and
summarization is one of the few automation opportunities that addresses it directly. The value
proposition is time. The risk is that the time saved in drafting is spent in correction and that
the residual error is invisible. A deployment that measures only drafting time will overstate the
benefit; one that measures the full ledger will be honest, even when the answer is unfavorable.

The technical stakes are also high because summarization is the task where a language model's
failure is most persuasive. An extraction error produces a missing field. A summarization error
produces a sentence that reads like a clinician wrote it. The fluency is the hazard: it recruits
the reader's trust before the reader has checked the evidence. Grounding is the countermeasure
because it puts the evidence next to the claim, where a reviewer can see it.

The evaluation stakes follow. A field-level error has a natural metric: was the field correct. A
summary has no single correct form, so the evaluation must construct one. The construction is the
method: the claim unit, the support definition, the omission list, and the reader task. A report
that does not describe the construction has not described the measurement, and its number cannot be
compared with any other.

> **Note:** A summary that omits a critical finding can be more harmful than one that states a
> minor falsehood. Omission is a first-class error, and it is harder to measure than fabrication
> because it requires a list of what should have been included.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** extractive, abstractive, and hybrid summarization and **choose** an approach for
   a stated clinical use. The test is whether the choice is justified by the purpose and the risk
   profile, not by model preference.
2. **Define** a claim unit and a decomposition rule, and **explain** how granularity changes the
   measured hallucination rate. The test is whether two analysts would decompose the same summary
   the same way.
3. **Design** a grounded summarization workflow with source spans, citations, and abstention. The
   test is whether every claim can be traced to a span and every insufficient case has an output.
4. **Audit** a summary at the claim level: supported, unsupported, uncertain, omitted. The test is
   whether the audit distinguishes commission from omission.
5. **Measure** citation coverage and citation precision separately, and **explain** why coverage
   alone is not grounding. The test is whether both numbers appear in the report.
6. **Measure** omission against a pre-specified critical-finding list. The test is whether the list
   was defined before the summaries were read.
7. **Estimate** the net utility of a summary workflow, including review and correction time. The
   test is whether the baseline was measured in the same workflow.
8. **Critique** a summarization evaluation that reports only a similarity metric or an aggregate
   hallucination rate without a definition. The test is whether the critique names the unit, the
   support definition, and the omission measurement.

Outcomes 2 through 7 are analytical and computational; 1 and 8 are design and critique.

## 3. The core mental model

A clinical summary is a set of claims, each of which should be traceable to evidence.

```text
  source documents
        |
        v
  evidence selection  ---------> which sentences matter for the purpose
        |
        v
  claim generation    ---------> sentence or clause stating a fact
        |
        v
  grounding           ---------> citation span per claim
        |
        v
  verification        ---------> supported | unsupported | uncertain
        |
        v
  assembly            ---------> summary with citations
        |
        v
  human review        ---------> accept | correct | reject
```

The diagram makes the unit explicit. The claim is the atom of evaluation. A document-level score
averages over claims and hides the one that matters.

A second model is the **three-error typology**.

```text
  fabrication   : a claim with no support in the source
  distortion    : a claim that misstates the source (polarity, attribution, magnitude)
  omission      : a critical fact present in the source but absent from the summary
```

Fabrication and distortion are commission errors; omission is an error of selection. Most
evaluations measure commission because it is easier: the claim is in the output and can be checked.
Omission requires knowing what should have been included, which requires a purpose-specific list.
The two error types have different consequences and different mitigations.

Omission also has a subtle form: the summary states a fact but not the fact the reader needed. A
discharge summary that lists the discharge medications but not which one changed is not missing a
category; it is missing the distinction that matters. This is why the omission list should be
defined in terms of decisions, not categories. "New or changed medication" is a decision-relevant
item; "medication" is a category that can be nominally present while the decision-relevant content
is absent.

A third model is the **grounding contract**.

```text
  every claim -> at least one citation -> the cited span supports the claim
```

The contract has two failure points: a claim with no citation (coverage), and a citation that does
not support the claim (precision). A summary can have 100 percent coverage and 60 percent citation
precision, which means every claim looks grounded and many are not. Reporting coverage alone is a
form of metric substitution.

A fourth model is the **utility ledger**.

```text
  time saved by drafting            +6 min per summary
  time spent reviewing              -3 min per summary
  time spent correcting errors      -? min per summary
  harm from undetected error        -? (clinical)
  net utility                       = the sum
```

The ledger is the only way to answer whether a summarization workflow is worth deploying. A system
that saves six minutes of drafting and costs four minutes of correction plus a small probability of
harm may be neutral or negative. The ledger should be computed with observed values, not assumed
ones.

A fifth model is the **claim-evidence graph**, which makes multi-source grounding visible.

```text
  claim A: "admitted with heart failure"  ---> source: discharge summary, sentence 1
  claim B: "started furosemide 40 mg"     ---> source: medication list, row 3
  claim C: "potassium repleted"           ---> source: progress note day 1, sentence 4
  claim D: "follow up in two weeks"       ---> source: discharge plan, sentence 2
```

Each claim can have one source, several sources, or none. A claim with two independent sources is
stronger than a claim with one, and the graph can record that. A claim with a citation that points
to a document but not a span is weaker than it appears. The graph is the data structure behind the
citation audit, and it is also the structure a reviewer needs to check a summary efficiently.

A sixth model is the **abstention boundary**.

```text
  evidence sufficient  -> state the claim with citation
  evidence partial     -> state with hedge and citation
  evidence absent      -> abstain, or state "not documented"
```

Abstention is not a failure. A system that says "no documented evidence of X" is more useful and
safer than one that invents X. The rule for abstention must be defined and tested, because a system
that abstains too often is useless and one that never abstains is unsafe.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Summarization | Producing a shorter representation of source text | Confused with extraction | Document |
| Extractive | Selecting source sentences | Called abstractive if lightly edited | Sentence |
| Abstractive | Generating new wording | Assumed to be less faithful | Claim |
| Hybrid | Extractive selection plus rewriting | Called extractive | Sentence and claim |
| Claim | A statement that can be verified against source | Confused with a sentence | Claim |
| Claim decomposition | Splitting output into verifiable units | Left implicit | Rule |
| Grounding | Linking a claim to source evidence | Called citation without a span | Claim-span pair |
| Citation | A reference to a source span | Assumed to be correct | Span |
| Citation coverage | Share of claims with a citation | Called grounding quality | Proportion |
| Citation precision | Share of citations that support the claim | Omitted | Proportion |
| Faithfulness | Agreement between claim and source | Confused with fluency | Claim |
| Hallucination | An unsupported or fabricated claim | Used for any error | Claim |
| Fabrication | A claim with no source support | Confused with distortion | Claim |
| Distortion | A claim that misstates the source | Confused with fabrication | Claim |
| Omission | A critical fact absent from the summary | Ignored in many evaluations | Fact |
| Critical finding | A fact the summary must include | Defined ad hoc | Fact |
| Abstention | Declining to state a claim | Called a failure | Claim |
| Hedging | Stating a claim with uncertainty | Confused with abstention | Claim |
| Attribution | The subject a claim applies to | Confused with grounding | Claim |
| Temporal accuracy | Correct timing of a claim | Confused with faithfulness | Claim |
| Fluency | Readability of the summary | Confused with quality | Text |
| Coherence | Logical structure of the summary | Confused with faithfulness | Text |
| ROUGE | N-gram overlap with a reference | Called a faithfulness metric | Score |
| BLEU | N-gram precision against references | Used for clinical summaries | Score |
| Claim-level audit | Human review of each claim | Confused with document review | Review |
| Major error | An error that could change management | Defined by scale | Category |
| Minor error | An error with no management impact | Defined by scale | Category |
| Review burden | Time to verify and correct | Omitted from utility | Minutes |
| Claim-evidence graph | Claims linked to source spans | Confused with a citation list | Graph |
| Decision-relevant omission | Missing the distinction the reader needs | Confused with a category omission | Fact |
| Reader study | Clinician evaluation of quality and usability | Confused with a survey | Study |
| Source quality | Accuracy of the underlying document | Assumed correct | Document |
| Audit sample | Subset reviewed for monitoring | Confused with the gold sample | Sample |

Three definitions need emphasis.

**Faithfulness versus fluency.** A fluent summary reads well. A faithful summary states only what
the source supports. A model can be fluent and unfaithful, and human readers are biased toward
fluent text. The evaluation must separate the two, which means reading for support rather than for
style.

**Hallucination versus omission.** A hallucination is a claim that is not supported. An omission is
a fact that is not stated. A summary can have zero hallucinations and still be dangerous because it
left out the critical finding. The two require different reference data: hallucinations need the
source, omissions need a list of what should be included.

**Claim versus sentence.** A sentence can contain several claims. "She was admitted with heart
failure and started on furosemide" contains an admission claim and a treatment claim. If the
treatment is wrong, a sentence-level judgment may mark the whole sentence as unsupported, which
overstates the error, or as supported because the admission is correct, which understates it. The
claim unit must be defined before the audit.

## 5. The mechanics, step by step

This section is the procedure for building and evaluating a grounded summarization workflow.

### Step 1. Define the purpose and audience

**Input:** The clinical or business use.

**Output:** A purpose statement naming the reader, the decision, and the required content.

**Rationale:** A discharge summary for a primary care clinician, a handoff for the night team, and a
prior authorization letter have different content requirements. The purpose determines what counts
as a critical finding and therefore what counts as an omission.

**Branch point:** If the purpose is not stated, the omission list cannot be built, and the
evaluation will measure only commission errors.

### Step 2. Define the source scope and time window

**Input:** The purpose and the available documents.

**Output:** A statement of which documents, sections, and time window the summary covers.

**Rationale:** A summary of the admission is not a summary of the patient's history. Including
outpatient notes changes the content and the risk. The scope determines what the system can know.

**Branch point:** If the scope includes documents with different note types, the system needs a
selection policy. A discharge summary should prioritize the hospital course and the discharge plan;
a handoff should prioritize the last 24 hours.

### Step 3. Choose extractive, abstractive, or hybrid

**Input:** The purpose, the source, and the risk tolerance.

**Output:** An approach with a rationale.

**Rationale:** Extractive methods are more faithful by construction and less fluent. Abstractive
methods are more fluent and less faithful. Hybrid methods select evidence and rewrite it, which can
preserve both if the rewrite is constrained.

**Branch point:** The choice has different risk profiles.

| Approach | Faithfulness | Fluency | Omission risk | Verification |
|---|---|---|---|---|
| Extractive | High | Low | Selection bias | Citation is the sentence |
| Abstractive | Low to medium | High | High | Claim-level audit required |
| Hybrid | Medium to high | Medium | Medium | Citation plus claim audit |

### Step 4. Define the claim unit and decomposition rule

**Input:** The approach and the output format.

**Output:** A written rule for splitting output into claims.

**Rationale:** The claim is the evaluation unit. A rule that splits conjunctions produces more
claims and a higher measured error rate. A rule that keeps them together produces fewer claims and
hides partial errors.

**Branch point:** Decomposition rules differ in granularity.

| Rule | Example | Claims | Effect on error rate |
|---|---|---|---|
| Sentence | "Admitted with HF and started on furosemide." | 1 | Lower; hides partial errors |
| Clause | Same | 2 | Higher; exposes partial errors |
| Atomic fact | Same, with dose and timing | 3 to 4 | Highest; most sensitive |
| Section | Whole plan paragraph | 1 | Lowest; hides most errors |

The rule must be stated in the report, because the hallucination rate is not comparable across
rules.

### Step 5. Build the input with evidence

**Input:** Source documents and the purpose.

**Output:** A prompt or input containing the selected evidence, with identifiers for each span.

**Rationale:** The model can only ground claims in what it receives. If the input omits the
medication list, the summary cannot state the discharge medications. The selection is therefore part
of the faithfulness problem.

**Branch point:** Input construction has three common forms.

| Form | Content | Risk |
|---|---|---|
| Full documents | All text in scope | Long context; dilution |
| Section-selected | Relevant sections | Selection bias |
| Retrieval-selected | Sentences ranked by relevance | Missed critical facts |

### Step 6. Generate with constraints

**Input:** The input and the instructions.

**Output:** A summary with citations, or an abstention.

**Rationale:** Constraints shape the output. Requiring a citation per claim, a fixed schema, and an
explicit "not documented" option reduces fabrication and makes verification possible.

**Branch point:** Constraints can conflict. A strict citation requirement can reduce fluency; a
strict schema can force a claim where evidence is absent. The instruction should state which
constraint takes precedence.

### Step 7. Decompose the output into claims

**Input:** The generated summary.

**Output:** A claim list with spans and citations.

**Rationale:** The claim list is the audit unit. Automated decomposition is imperfect; a
rule-based splitter plus human review of a sample is the practical approach.

**Branch point:** If decomposition is automated, report its agreement with human decomposition on a
sample. If it is manual, report the annotator time.

### Step 8. Verify each claim against the source

**Input:** Claims and cited spans.

**Output:** A label per claim: supported, unsupported, uncertain.

**Rationale:** Verification is the core measurement. It can be automated with entailment models or
lexical checks, and it must be validated against human judgment.

**Branch point:** Verification methods differ in recall and precision.

| Method | Strength | Weakness | Use |
|---|---|---|---|
| Lexical overlap | Fast, transparent | Fails on paraphrase | Triage |
| Entailment model | Handles paraphrase | Domain shift; opaque | Screening |
| Human review | Gold standard | Expensive | Validation sample |
| Rule plus human | Balanced | Complex | Production audit |

### Step 9. Measure omission against a critical-finding list

**Input:** The source and a pre-specified list of critical findings.

**Output:** Omission rate per summary and per finding type.

**Rationale:** Omission is invisible without a list. The list should be derived from the purpose and
reviewed by a clinician, not inferred from the summary.

**Branch point:** The list can be generic, such as "critical lab, critical medication change,
follow-up plan," or purpose-specific, such as "anticoagulation status" for a surgical handoff. The
more specific the list, the more useful the omission measurement.

### Step 10. Measure citation coverage and precision separately

**Input:** Claims and citations.

**Output:** Coverage and precision, with per-section breakdowns.

**Rationale:** Coverage says every claim has a citation. Precision says the citation supports the
claim. A system can game coverage by citing the whole document.

**Branch point:** If coverage is high and precision is low, the grounding is decorative. If coverage
is low and precision is high, the system is honest but incomplete.

### Step 11. Measure utility and harm

**Input:** Workflow timings, correction rates, and error severity.

**Output:** Net time and a harm estimate.

**Rationale:** The decision to deploy depends on the ledger, not on the hallucination rate alone.

**Branch point:** If net utility is negative, the workflow needs redesign, not more model
fine-tuning. The redesign may be a narrower task, a different reader, or a human-first workflow with
model assistance.

### Step 12. Calibrate abstention

**Input:** Claims labeled uncertain and the system's confidence.

**Output:** An abstention threshold and its coverage.

**Rationale:** Abstention trades coverage for safety. The threshold should follow from the cost of a
wrong claim relative to the cost of a missing claim.

**Branch point:** If abstention is too frequent, users will ignore the system. If it is too rare,
users will over-trust it. The rate should be monitored after deployment.

### Step 13. Design the human review

**Input:** The error profile and the workflow.

**Output:** A review protocol with roles, time, and escalation.

**Rationale:** Human review is the safety net. It works only if the reviewer has the source, the
citations, and the authority to correct.

**Branch point:** Review designs differ in cost and detection.

| Design | Reviewer sees | Detection | Cost |
|---|---|---|---|
| Full read | Summary and source | High | High |
| Citation check | Claim and cited span | Medium | Medium |
| Exception review | Flagged claims only | Low to medium | Low |
| Spot audit | Random sample | Low for any single summary | Low |

### Step 14. Monitor after deployment

**Input:** Production summaries and a review sample.

**Output:** A monitoring report with error rates, abstention rate, and drift signals.

**Rationale:** Models, templates, and case mix change. A summary system validated on one service may
fail on another.

**Branch point:** Monitor commission and omission separately. A drop in citation coverage is a
different signal from a rise in unsupported claims.

### Step 15. Run a reader study

**Input:** Generated and comparison summaries.

**Output:** Reader judgments of quality, safety, and preference, with the reader's role recorded.

**Rationale:** Claim-level metrics do not capture whether the summary is useful. A reader study
measures comprehension, confidence, and time. The reader's role matters: a primary care clinician
reads differently from a hospitalist.

**Branch point:** Reader-study designs differ in what they can detect.

| Design | Readers see | Detects | Limitation |
|---|---|---|---|
| Blinded comparison | Generated and reference, unlabeled | Preference and quality | Small samples |
| Error detection | Summary with planted errors | Sensitivity to errors | Artificial |
| Comprehension | Summary only, then questions | Understanding | Costly |
| Workflow timing | Summary in a simulated task | Time and usability | Ecological validity |

A reader study with 10 clinicians and 100 summaries is a common scale. It can detect a large
difference and not a small one. The report should state the smallest difference the design can
detect, or the readers' judgments will be over-interpreted.

### Decision table: what to do when evidence is partial

The hardest generation decision is not whether a fact is true but how much of it the source supports.
The table below is a policy for the boundary cases.

| Evidence state | Example | Output | Citation | Risk |
|---|---|---|---|---|
| Direct statement | "Started furosemide 40 mg" | State as fact | Yes | Low |
| Two consistent statements | Note and medication list agree | State as fact | Yes, both | Low |
| One statement, no corroboration | Note says "likely diuresed" | State with hedge | Yes | Medium |
| Conflicting statements | Note says started; list says held | State conflict | Yes, both | Medium |
| Inference from data | Potassium 3.1 and repleted | State the value, not "normalized" | Yes, value | Medium |
| Absent | No mention of anticoagulation | "Not documented" | No | Low if explicit |
| Absent but clinically expected | Surgical patient, no VTE plan | Abstain and flag | No | High |

The last row is the dangerous one. A system that silently omits an expected item looks complete and
is not. The flag tells the reviewer that the absence is notable, which is different from an absence
that carries no information.

### Step 16. Version and monitor the whole stack

**Input:** Model, prompt, source selection, claim decomposer, and audit sample.

**Output:** A versioned artifact and a monitoring plan.

**Rationale:** A summarization system is a stack, not a model. A prompt change, a source-selection
change, or a model update can change the output. Versioning only the model misses most of the
variation.

**Branch point:** Monitoring has three signals, and they fail differently.

| Signal | What it detects | Response |
|---|---|---|
| Unsupported rate by section | Generation drift | Constrain the section; update prompt |
| Omission rate | Selection drift or source change | Re-check source scope |
| Citation precision | Retrieval or grounding drift | Re-index; re-audit |

A rise in unsupported plan claims after a source change is not necessarily a model regression. It
may be that the new source documents have less plan detail, so the model fills more gaps. The
monitoring report should compare the input distribution as well as the output.

### Edge cases that decide summary quality

**Negation.** "No evidence of bleeding" can become "bleeding was ruled out" or, worse, "bleeding."
A negation flip is a distortion that changes management. The audit must include negation-heavy
claims.

**Hypotheticals.** "If the patient develops fever, start antibiotics" is a plan, not an event. A
summary that states "patient developed fever" has converted a conditional into a fact. Plan
sections are the highest-risk source of this error.

**Family history.** "Mother had breast cancer" can become "patient has breast cancer" through
attribution loss. The summary must preserve the experiencer.

**Copy-forward duplicates.** A source with duplicated text can lead the model to weight a stale fact
as if it were repeated evidence. Deduplicate or flag before summarization.

**Template boilerplate.** A template's "denies chest pain" may be summarized as a negative finding
even when the clinician did not ask. The summary inherits the template's assumption.

**Abbreviation ambiguity.** "MS" expanded wrongly changes the meaning. The summary should preserve
the abbreviation or use the local expansion, and the audit should flag expansions.

**Section boundaries.** A finding in the assessment is not the same as a finding in the review of
systems. The summary should preserve the section context, and the audit should check it.

**Unicode and OCR noise.** A corrupted character can change a lab value or a medication name. The
pipeline should normalize and flag low-quality source text.

## 6. Worked example

A hospital is piloting a discharge summary assistant for 120 admissions. The summary is generated
from the discharge summary draft, the medication list, and the last 48 hours of progress notes. The
reader is the primary care clinician. The purpose is to support the first post-discharge visit.

### Annotation design

Two clinicians audit every generated summary. They decompose each summary into claims using a
clause-level rule: each clause with a subject and a predicate is a claim. They label each claim as
supported, unsupported, or uncertain, and they record whether an unsupported claim is major (could
change management) or minor.

They also audit omission against a pre-specified list of six critical findings: discharge diagnosis,
new or changed medication, anticoagulation status, pending test results, follow-up appointment, and
red-flag symptoms.

### Claim-level results

The 120 summaries contain 2,400 claims, an average of 20 per summary.

| Claim label | Count | Share |
|---|---:|---:|
| Supported | 2,040 | 85.0% |
| Unsupported | 156 | 6.5% |
| Uncertain | 204 | 8.5% |
| Total | 2,400 | 100% |

Of the 156 unsupported claims, 44 are major and 112 are minor.

Major hallucination rate = 44 / 2,400 = 1.83%.
Any-hallucination rate = 156 / 2,400 = 6.5%.

The two rates answer different questions. The any-hallucination rate describes how often the system
is wrong; the major rate describes how often it is dangerously wrong. A report that gives only one
number invites the reader to assume the other.

### Omission results

The critical-finding audit finds 84 omissions across the 120 summaries.

| Critical finding | Omissions | Summaries affected |
|---|---:|---:|
| Discharge diagnosis | 6 | 6 |
| New or changed medication | 31 | 21 |
| Anticoagulation status | 18 | 14 |
| Pending test results | 17 | 13 |
| Follow-up appointment | 9 | 8 |
| Red-flag symptoms | 3 | 3 |
| Total | 84 | 65 |

A summary can omit more than one finding, so the summaries-affected column sums to more than the
number of unique summaries. The union of the affected sets is 23 summaries with at least one
omission, or 19.2 percent of 120.

Omission rate per summary = 84 / 120 = 0.70.
Share of summaries with at least one omission = 23 / 120 = 0.192.

The most frequent omission is the new or changed medication, which is also the item most likely to
matter at the first post-discharge visit. The system's medication section is generated from the
medication list, but the list does not distinguish new from continued medications, so the summary
omits the distinction. The error is a source-data limitation, not a generation error. Separating the
two is essential for the fix.

### Citation results

Of the 2,400 claims, 2,244 have a citation.

Citation coverage = 2,244 / 2,400 = 0.935.

Of the 2,244 citations, 1,980 point to a span that supports the claim.

Citation precision = 1,980 / 2,244 = 0.882.

The 264 unsupported citations are not the same as the 156 unsupported claims. A claim can be true
and cited to the wrong span; a claim can be false and cited to a span that does not support it. The
two errors have different fixes: the first is a citation assignment problem, the second is a
generation problem.

### Two diagnostics that reveal a broken pipeline

**Diagnostic 1: unsupported claims by section.** The audit records the section of the summary where
each unsupported claim appears.

| Summary section | Unsupported claims | Share of unsupported |
|---|---:|---:|
| Hospital course | 22 | 14.1% |
| Assessment | 38 | 24.4% |
| Plan | 96 | 61.5% |
| Follow-up | 0 | 0.0% |
| Total | 156 | 100% |

Sixty-one percent of unsupported claims are in the plan. The plan is where the model is asked to
predict future actions from sparse notes, and it fills gaps with plausible language. A pipeline with
this profile should not generate the plan section without explicit evidence, or should mark plan
claims as suggestions rather than facts. The diagnostic is the section table; an aggregate rate
would have hidden the concentration.

**Diagnostic 2: citation precision by section versus coverage.** Citation coverage is high
everywhere, but precision varies.

| Section | Coverage | Citation precision |
|---|---:|---:|
| Hospital course | 0.971 | 0.941 |
| Assessment | 0.958 | 0.902 |
| Plan | 0.902 | 0.790 |
| Follow-up | 0.988 | 0.972 |

Coverage is 90 percent or higher in every section, which looks like strong grounding. Precision in
the plan is 0.790, which means one in five plan citations does not support its claim. A report that
showed only coverage would have described the system as well grounded. The diagnostic is the
two-column table; coverage without precision is decoration.

### Utility ledger

The pilot measures clinician time in the same workflow.

| Activity | Time per summary |
|---|---:|
| Manual drafting (measured baseline) | 18 minutes |
| Reviewing a generated summary | 3 minutes |
| Correcting errors when present | 15 minutes |
| Share of summaries needing correction | 19.2% |

The gross drafting time saved is 18 − 3 = 15 minutes per summary. The correction cost is incurred
on 19.2 percent of summaries, so the expected correction cost is 0.192 × 15 = 2.88 minutes.

Net time per summary = 15 − 2.88 = 12.12 minutes.
For 120 summaries: 120 × 12.12 = 1,454 minutes, or 24.2 hours.

If the baseline were 9 minutes instead of 18, net time would be 9 − 3 − 2.88 = 3.12 minutes per
summary, or 6.2 hours for 120 summaries. The conclusion changes by a factor of 3.9 depending on an
assumption that is rarely reported. The correct practice is to measure the baseline in the same
workflow, with the same clinicians, and to report it. A vendor's time-saving claim is not a
substitute.

The ledger also omits harm. A major hallucination that reaches the patient can cause an adverse
event whose cost is not measured in minutes. The ledger should carry the harm separately, as a
probability and a severity, rather than folding it into time.

### Vary one assumption: atomic claim decomposition

Suppose the auditors split each clause into atomic facts, separating the subject, action, dose, and
timing. The claim count rises from 2,400 to 3,100.

| Metric | Clause-level | Atomic-level |
|---|---:|---:|
| Claims | 2,400 | 3,100 |
| Supported | 2,040 | 2,540 |
| Unsupported | 156 | 340 |
| Uncertain | 204 | 220 |
| Any-hallucination rate | 6.5% | 11.0% |
| Major unsupported | 44 | 52 |
| Major hallucination rate | 1.83% | 1.68% |

The any-hallucination rate nearly doubles, because compound claims that were counted once are now
counted two or three times, and partial errors become visible. The major rate barely changes,
because the number of dangerous errors is a property of the content, not the decomposition.

The conclusion changes depending on which number is reported. A vendor reporting the clause-level
rate would appear twice as faithful as one reporting the atomic rate, on the same output. The
decomposition rule must be stated, and comparisons across studies must use the same rule.

The worked example also shows why the omission and citation numbers belong next to the
hallucination number. A summary with a 6.5 percent any-hallucination rate and a 19.2 percent
summary-level omission rate is not safer than one with an 11.0 percent hallucination rate and a 5
percent omission rate; the two profiles have different clinical consequences. The report should
present the profile, not a single score, and the deployment decision should be made against the
profile the workflow can tolerate.

The audit itself has a cost. Two clinicians reviewing 2,400 claims at roughly 45 seconds per claim
spend 30 hours, plus adjudication. That cost is comparable to the drafting time the system saves in
the pilot. A production audit cannot repeat it at that intensity; it needs a sample, an automated
triage, and a smaller expert review. The sampling design should be reported so that the audit's
precision is visible, and the extrapolation from sample to corpus should be explicit.

## 7. Code, query, or calculation

The following module implements clause-level claim decomposition, a lexical support check with a
polarity guard, and citation coverage and precision. It uses only the standard library and synthetic
text.

```python
import re

SOURCE = [
    "The patient was admitted with acute decompensated heart failure.",
    "She received intravenous furosemide 40 mg twice daily.",
    "She denies chest pain throughout the admission.",
    "Potassium was 3.1 mmol/L on admission and repleted.",
    "Discharge plan includes daily weights and a cardiology follow-up in two weeks.",
]

SUMMARY = [
    ("Admitted for acute decompensated heart failure.", 0),
    ("Treated with intravenous furosemide.", 1),
    ("No chest pain reported.", 2),
    ("Potassium normalized after repletion.", 3),
    ("Follow up with cardiology in two weeks.", 4),
    ("She has chest pain.", 2),
    ("Started on a new beta blocker.", None),
]

NEG = {"no", "not", "denies", "without", "never"}


def content_tokens(text):
    stop = {"the", "a", "an", "was", "were", "is", "are", "with", "for", "and",
            "on", "in", "of", "to", "she", "he", "patient", "after", "reported"}
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in stop}


def polarity(text):
    return bool(content_tokens(text) & NEG)


def support(claim, source):
    ct, st = content_tokens(claim), content_tokens(source)
    if not ct:
        return 0.0, False
    overlap = len(ct & st) / len(ct)
    return overlap, polarity(claim) == polarity(source)


for claim, idx in SUMMARY:
    if idx is None:
        print(f"NO CITATION: {claim!r}")
        continue
    ov, pol = support(claim, SOURCE[idx])
    print(f"cite={idx} overlap={ov:.2f} polarity_ok={pol} "
          f"supported={ov >= 0.5 and pol} :: {claim}")

cited = sum(1 for _, i in SUMMARY if i is not None)
covered = cited / len(SUMMARY)
correct = sum(1 for c, i in SUMMARY if i is not None
              and (lambda o, p: o >= 0.5 and p)(*support(c, SOURCE[i])))
print("citation coverage:", round(covered, 3))
print("citation precision (proxy):", round(correct / cited, 3))
```

Expected output:

```text
cite=0 overlap=1.00 polarity_ok=True supported=True :: Admitted for acute decompensated heart failure.
cite=1 overlap=0.67 polarity_ok=True supported=True :: Treated with intravenous furosemide.
cite=2 overlap=0.67 polarity_ok=True supported=True :: No chest pain reported.
cite=3 overlap=0.33 polarity_ok=True supported=False :: Potassium normalized after repletion.
cite=4 overlap=1.00 polarity_ok=True supported=True :: Follow up with cardiology in two weeks.
cite=2 overlap=0.67 polarity_ok=False supported=False :: She has chest pain.
NO CITATION: 'Started on a new beta blocker.'
citation coverage: 0.857
citation precision (proxy): 0.667
```

The output exposes the limits of the lexical proxy. "Potassium normalized after repletion" is
faithful to the source, but the overlap is only 0.33 because the summary uses "normalized" and the
source gives a value. The proxy marks it unsupported. Conversely, "She has chest pain" has an
overlap of 0.67 and is caught only by the polarity guard. A production audit needs an entailment
model or human review; the lexical check is a triage tool, not a verdict.

A second block computes the utility ledger under two baselines and the decomposition sensitivity:

```python
def ledger(baseline_min, review_min, correction_min, p_error, n):
    net = baseline_min - review_min - p_error * correction_min
    return net, net * n / 60


for baseline in (9, 18):
    net, hours = ledger(baseline, 3, 15, 23 / 120, 120)
    print(f"baseline {baseline} min: net {net:.2f} min/summary, {hours:.1f} hours saved")

for claims, unsupported, major in [(2400, 156, 44), (3100, 340, 52)]:
    print(f"claims {claims}: any={unsupported/claims:.3f} major={major/claims:.3f}")
```

Expected output:

```text
baseline 9 min: net 3.12 min/summary, 6.2 hours saved
baseline 18 min: net 12.12 min/summary, 24.2 hours saved
claims 2400: any=0.065 major=0.018
claims 3100: any=0.110 major=0.017
```

The ledger shows a factor-of-four difference in time saved from a baseline assumption. The
decomposition block shows the any-error rate rising from 6.5 to 11.0 percent while the major rate
falls slightly, because the atomic split redistributes rather than creates danger.

## 8. Data, standards, and regulatory context

**HIPAA and generated text.** A summary generated from PHI is PHI. If the generation happens on a
hosted endpoint, the disclosure rules apply, including a business associate agreement. The summary
must be stored and access-controlled like any other note. A generated summary that is inserted into
the record becomes part of the designated record set and is subject to amendment and audit.

**Record retention and authorship.** A generated summary that a clinician signs becomes the
clinician's note. The clinician is responsible for its content, and the system should record that
the text was machine-generated and which model version produced it. Several institutions require a
disclosure statement in the note. The requirement is institutional policy, not a HIPAA rule, and it
varies.

**Patient access.** Under the 21st Century Cures Act information blocking rules, patients can access
their notes. A generated summary is part of that record. The language should be written for a
patient reader as well as a clinician, and the hallucination risk extends to the patient's
understanding of their own care.

**FDA device regulation.** A summarization system that informs diagnosis or treatment for a specific
patient may be a medical device. The FDA's January 2025 draft guidance on AI-enabled device software
functions recommends documentation of intended use, development data, reference standards,
validation, subgroup performance, and postmarket monitoring. A summary that is presented as a draft
for clinician review may have a different regulatory position from one that is automatically filed;
the intended use statement determines which. The guidance is a draft, not a final rule.

**ONC HTI-1.** Predictive decision support interventions supplied as part of certified health IT are
subject to source attribute transparency. A summarization system that predicts or recommends may
fall within the predictive DSI definition; a pure drafting tool may not. The distinction depends on
function, and the source attributes should be documented regardless.

**Clinical documentation integrity.** Hospitals have documentation integrity programs that review
notes for accuracy, completeness, and compliance. A generated summary enters that program's scope.
The program may require a review process, a query mechanism for ambiguous documentation, and a
record of corrections. The summarization workflow should integrate with it rather than bypass it.

**Research reporting.** There is no binding standard for reporting summarization evaluations. The
claim-level designs used in recent studies are becoming a de facto standard, and they should report
the claim unit, the decomposition rule, the annotator qualifications, the support definition, the
omission list, and the severity scale. A report that gives only ROUGE or only an aggregate
hallucination rate is not comparable.

**Liability.** A missed finding in a summary can cause harm, and the liability question is who is
responsible: the clinician who signed, the institution that deployed, or the vendor that supplied.
The answer depends on the jurisdiction and the contract, and it is outside this module's scope. The
design implication is that the review protocol and the audit trail should be documented before
deployment, not after an incident.

**Consent and patient communication.** A generated summary may be shared with the patient as a visit
summary or an after-visit instruction. The language must be understandable, and the omission of a
critical finding is a communication failure with the patient, not only a documentation failure. A
patient-facing summary needs its own review standard, which may differ from the clinician-facing
one.

**Language and translation.** A summary generated in one language and translated into another
introduces a second error layer. The claim-level audit should be performed in the language the
reader will use, because support can change in translation. A translated summary that is faithful
in English may be ambiguous in the target language.

**Quality measurement.** A summary can be used as an input to a quality measure, such as a
documentation completeness score. The measure then inherits the summary's errors. A team that uses
summaries for measurement should validate the measure against the source, not only against the
summary.

**Audit trail.** A generated summary should record the model, version, prompt, source documents, and
generation time. The record is needed for reproducibility, for incident review, and for any
regulatory submission. A summary without provenance is a note with no author history, which is a
record-keeping problem independent of its accuracy.

> **Note:** A citation is not proof of support. A system can cite a source span that is related to
> the claim without entailing it. Citation precision must be measured, not assumed from coverage.

## 9. Industry practice and case evidence

**Blinded comparison of discharge summaries.** Williams et al. (2024, medRxiv) compared
physician-written and model-generated discharge summaries for 100 inpatient encounters, reviewed by
22 clinicians. They reported no significant difference in overall quality (3.67 versus 3.77) or
reviewer preference. Model summaries were more concise and coherent but less comprehensive, with
more errors per summary (2.91 versus 1.82), driven by omissions and inaccuracies. Hallucination
rates were similar (0.23 versus 0.31 per summary, p = 0.230). These are the study's reported results
on its corpus and rating scale.

**Hallucination and omission framework.** Asgari et al. (2024, medRxiv) reported a manual evaluation
in which 50 physicians annotated 18 experiments, covering 49,590 transcript sentences and 12,999
note sentences. They reported a 1.47 percent hallucination rate, with 44 percent of hallucinations
rated major, and a 3.45 percent omission rate, with 17 percent rated major. Hallucinations clustered
in the plan and assessment sections; the most common types were fabrications (43 percent) and
negations (30 percent). These are the study's reported figures and definitions.

**Adapted models versus experts.** Van Veen et al. (2024, Nature Medicine) conducted a
10-physician reader study across four summarization tasks. The best model, GPT-4 with in-context
learning, was judged equivalent to expert summaries in 45 percent of cases and superior in 36
percent, with fewer errors and less fabricated information; hallucinations occurred on 5 percent of
samples versus 12 percent for experts. The result is task-specific and reader-specific, and it does
not imply that generated summaries are safe without review.

**Emergency department summaries.** A 2024 evaluation of GPT-4 for drafting emergency department
discharge summaries reported inaccuracies in 10 percent, hallucinations in 42 percent, and omissions
in 47 percent of cases, with hallucinations concentrated in the plan and often involving redacted
information or invented follow-up instructions. The contrast with the discharge summary study is
instructive: the task, the source, and the definition differ, and the rates differ by an order of
magnitude.

**The definition problem.** The four studies above report hallucination rates from 0.23 per summary
to 42 percent of cases. The numbers are not comparable. A reader who compares them without the
units and definitions will conclude either that the technology is safe or that it is unusable. The
correct conclusion is that the measurement is not standardized, and a local evaluation is required.

**Grounding practice.** Retrieval-augmented generation and citation-required prompting are common
design patterns. The published evaluations of grounding usually measure whether a citation exists,
and less often whether it supports the claim. Inference: citation precision is the weaker link in
most production systems, and it is the one an audit should target.

**Human review practice.** Operational deployments commonly use a draft-and-review model, where the
clinician edits the generated summary before signing. The review is the safety net, and its
effectiveness depends on the reviewer's workload and the salience of the errors. Inference: a review
that is rushed or that trusts the draft is not a control.

**Vendor and product landscape.** Summarization features appear in EHR vendors, ambient scribes,
and standalone products. Their evaluations range from a benchmark similarity score to a full
clinician reader study. The buyer should ask for the claim unit, the support definition, the
omission measurement, the citation precision, and the review protocol. A product that cannot supply
them has not been evaluated in the terms that matter for a clinical deployment. Inference from
procurement practice.

**Ambient documentation and summarization.** Ambient scribes generate a note from a conversation,
which is a summarization task with a different source: speech rather than text. The error profile
includes misheard medication names, speaker attribution errors, and omissions of non-verbal content.
The evaluation design in this topic applies, with the transcript as the source and the note as the
summary. Inference: the same claim-level audit is the right instrument, but the source quality is
lower and the error rate is higher.

**Published benchmarks.** Summarization benchmarks in the general domain use reference summaries and
similarity metrics. Clinical benchmarks increasingly use claim-level or error-level annotation,
because similarity metrics do not capture clinical fidelity. The shift is not universal, and a
paper that reports only ROUGE should be read as a general-domain result, not a clinical validation.

**The gap between demonstration and deployment.** A demonstration summarizes one note for one
reader. A deployment summarizes thousands of notes for a workflow, with liability, audit, and
training implications. The gap is not the model; it is the review protocol, the record-keeping, and
the governance. Inference: most published evaluations stop before the deployment questions, and the
operational evidence is thin.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Plan fabrication | Unsupported claims concentrated in plan | Model fills gaps | Wrong follow-up | Require evidence; mark suggestions |
| Negation flip | "No bleeding" becomes "bleeding" | Polarity loss | Wrong management | Polarity check; audit |
| Attribution loss | Family history becomes patient history | Experiencer dropped | Wrong diagnosis | Preserve subject; audit |
| Conditional becomes fact | "If fever, start antibiotics" becomes "started" | Modality loss | Wrong treatment | Modality check |
| Omission of critical finding | No omission list | Selection bias | Missed care | Pre-specified list |
| Citation coverage without precision | Coverage near 1, precision unknown | Decorative citation | False assurance | Measure precision |
| Citation to wrong span | Claim true, citation unrelated | Retrieval error | Unverifiable | Citation audit |
| Similarity metric as quality | ROUGE reported as faithfulness | Lexical overlap | Misleading | Claim-level audit |
| Aggregate hallucination rate | One number, no definition | Unit ambiguity | Incomparable | State unit and rule |
| Abstention too rare | Every claim stated | No uncertainty path | Confident errors | Calibrate threshold |
| Abstention too frequent | Most claims hedged | Over-conservative | Unusable | Recalibrate |
| Review too rushed | Reviewer time under 1 min | Automation bias | Errors missed | Time target; spot audit |
| Source omission | Summary misses a document | Input selection | Incomplete | Scope statement |
| Template contamination | Template negatives summarized as findings | Boilerplate | False negatives | Flag template text |
| Source error propagated | Summary faithful to a wrong note | Faithfulness is not truth | Wrong care | Source quality check |
| Reader mismatch | Summary written for the wrong audience | Purpose not specified | Unusable | Purpose statement; reader study |
| Translation drift | Claim changes in translation | Second error layer | Miscommunication | Audit in target language |
| Provenance missing | No model or source record | Audit trail absent | Unreviewable | Log model, prompt, sources |
| Baseline unmeasured | Time saving from a vendor claim | No local baseline | Wrong ROI | Local timing study |
| Decision-relevant omission | Category present, distinction missing | Omission list defined by category | Unusable summary | Define list by decision |
| Audit cost ignored | Review hours exceed drafting savings | Full audit at production scale | Unsustainable | Sampling design |

The red flags cluster around measurement. A report with a single hallucination rate, no claim unit,
no omission list, and no citation precision has not measured the system in a way that supports a
deployment decision. The second cluster is around the plan section: if unsupported claims cluster
there, the system is generating expectations rather than reporting evidence, and the fix is a
constraint, not a better model.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The source contains the facts the summary needs | Grounding depends on it | Missing document | Scope audit |
| The claim decomposition is stable | Comparability | Rule ambiguity | Agreement on decomposition |
| The support definition is clear | Measurement | Borderline paraphrase | Annotator agreement |
| The omission list is complete | Omission measurement | Missing critical fact | Clinician review |
| Citations point to the supporting span | Verification | Retrieval error | Citation precision |
| The reviewer has time and source | Safety net | Workload | Time audit |
| The baseline is measured in the same workflow | Utility estimate | Vendor benchmark | Local time study |
| The model version is pinned | Reproducibility | Hosted update | Version log |
| The summary is labelled as generated | Trust and audit | No label | Record check |
| The reader's purpose matches the summary's purpose | Relevance | Different audience | Reader study |

The strongest assumption is that the source contains the facts. In the worked example, the
medication omission was a source-data limitation: the list did not distinguish new from continued
medications. No generation improvement could fix it. A team that tuned the model without checking
the source would have spent effort on the wrong layer.

The second strongest is that the support definition is clear. Paraphrase sits on a boundary between
supported and unsupported, and different annotators place it differently. The report should state
the definition and the agreement on borderline cases, not only the aggregate rate.

A limit of claim-level audit is cost. A 2,400-claim audit is feasible for a pilot and not for
continuous monitoring. Production monitoring needs sampling, automated triage, and a smaller
human-reviewed sample. The sampling design should be reported, and the extrapolation from the
sample to the population should be explicit.

A further limit is that the source itself may be wrong. A summary can be faithful to a note that
contains an error, and the claim-level audit will mark it supported. Faithfulness is not truth. The
evaluation should state that it measures the relationship to the source, not to the patient's
actual state, and should not claim more.

A final limit is that summary quality is reader-dependent. A hospitalist and a primary care
clinician need different content from the same admission. A summary optimized for one reader may
be incomplete for another. The purpose statement and the reader study are the mechanisms for
handling this, and a single aggregate quality score cannot.

A further limit is that the evaluation cannot see what the system did not know. If the source
selection omitted a document, the summary may be faithful to the selected source and still miss the
fact. The scope statement and a source-coverage audit are the only defences, and both are outside
the claim-level check.

## 12. Exercises

**Exercise 9.1 — Claim decomposition (application, 40 minutes).**
Deliverable: take five generated summary sentences and decompose them under sentence, clause, and
atomic rules. Count the claims under each.
Acceptance criteria: three counts per sentence; a statement of which rule you would use and why.

**Exercise 9.2 — Support audit (application, 45 minutes).**
Deliverable: audit ten claims against a synthetic source, labeling supported, unsupported, or
uncertain, with a one-line justification.
Acceptance criteria: ten labels; at least one paraphrase and one negation case; justifications.

**Exercise 9.3 — Citation precision (calculation, 35 minutes).**
Deliverable: using the Section 6 numbers, compute coverage and precision, and explain why the two
differ.
Acceptance criteria: 0.935 and 0.882; the difference explained by claim-level versus citation-level
errors.

**Exercise 9.4 — Omission list (design, 40 minutes).**
Deliverable: a six-item critical-finding list for a surgical handoff, with a definition for each.
Acceptance criteria: six items; definitions; a note on how each would be detected in the source.

**Exercise 9.5 — Abstention rule (design, 40 minutes).**
Deliverable: an abstention policy with three evidence states and the output for each.
Acceptance criteria: three states; output wording; a threshold and its expected coverage.

**Exercise 9.6 — Utility ledger (calculation, 40 minutes).**
Deliverable: compute net time saved under two baselines and two error rates, and state the
assumption that changes the conclusion most.
Acceptance criteria: four values; the sensitive assumption identified.

**Exercise 9.7 — Review design (design, 45 minutes).**
Deliverable: a human review protocol with reviewer role, time target, source access, and escalation.
Acceptance criteria: four elements; a detection estimate; an audit sample.

**Exercise 9.8 — Critique a summary evaluation (critique, 30 minutes).**
Deliverable: critique a report that gives ROUGE 0.42 and "low hallucination" with no definition.
Name five missing elements.
Acceptance criteria: five elements; one question per element.

**Exercise 9.9 — Reader study (design, 45 minutes).**
Deliverable: a reader-study protocol with reader role, sample size, blinding, task, and the
smallest difference the design can detect.
Acceptance criteria: five elements; a power statement; a blinding check.

**Exercise 9.10 — Audit sampling (design, 35 minutes).**
Deliverable: a production audit plan that reviews 10 percent of summaries, with a triage rule for
which claims get human review and an escalation rule for a major error.
Acceptance criteria: sampling rule; triage rule; escalation owner; the extrapolation stated.

## 13. Self-check answers

**9.3 (citation).** Coverage = 2,244 / 2,400 = 0.935. Precision = 1,980 / 2,244 = 0.882. The
difference is that coverage counts claims with any citation, while precision counts citations that
support the claim. A claim can be true and cited to an unrelated span, which counts in coverage and
fails precision. A claim can be false and cited to a supporting-looking span, which also counts in
coverage. The two metrics answer different questions and must be reported together.

**9.6 (ledger).** Baseline 9: net = 9 − 3 − (23/120) × 15 = 9 − 3 − 2.875 = 3.125 minutes per
summary; 120 × 3.125 / 60 = 6.25 hours. Baseline 18: net = 18 − 3 − 2.875 = 12.125; 120 × 12.125 /
60 = 24.25 hours. The baseline assumption changes the result by a factor of 3.9. The error rate also
matters: at 50 percent error prevalence, the correction cost is 7.5 minutes and the net at baseline
9 is negative (−1.5 minutes). The conclusion is sensitive to both the baseline and the error rate,
and the report should state both.

**9.1 (decomposition) rubric.** Full credit: three counts per sentence, a rationale for the chosen
rule, and a note on how the rule changes the error rate. Half credit: counts without rationale.

**9.2 (support audit) rubric.** Full credit: ten labels, a paraphrase and a negation case, and
justifications tied to the source. Half credit: labels without justifications.

**9.4 (omission list) rubric.** Full credit: six items with definitions and a detection method for
each, tied to the surgical handoff purpose. Half credit: list without detection methods.

**9.5 (abstention) rubric.** Full credit: three evidence states, output wording for each, a
threshold, and expected coverage. Half credit: states without coverage estimate.

**9.7 (review) rubric.** Full credit: reviewer role, time target, source access, escalation, a
detection estimate, and an audit sample. Half credit: three of six elements.

**9.8 (critique).** Missing: the claim unit and decomposition rule; the support definition; the
omission measurement; the citation precision; the annotator qualifications and agreement; the
reader and purpose. Questions: "what counts as a claim?", "what counts as supported?", "what was
omitted?", "who reviewed and how long did they spend?", "was the citation checked?"

**9.9 (reader study) rubric.** Full credit: reader role, sample size with a detectable difference,
blinding procedure, task description, and a check that readers did not see the condition label.
Half credit: protocol without a power statement. Zero: a survey of satisfaction with no task.

**9.10 (audit sampling) rubric.** Full credit: a sampling rule, a triage rule that sends unsupported
or plan claims to review, an escalation owner, and a statement that the sample rate limits
precision. Half credit: sampling without triage.

## 14. Interview questions

**1. "What is the difference between faithfulness and fluency?"**
Strong answer: faithfulness is support by the source, fluency is readability; a fluent summary can
be unfaithful; evaluate separately. Weak answer: "good summaries are both."

**2. "How would you evaluate a clinical summary?"**
Strong answer: claim-level audit, support definition, omission list, citation precision, severity,
and a reader study. Weak answer: "ROUGE score."

**3. "What is a claim unit and why does it matter?"**
Strong answer: the unit of verification; granularity changes the measured error rate; must be
stated. Weak answer: "a sentence."

**4. "What is grounding?"**
Strong answer: linking each claim to a source span, with coverage and precision; citation alone is
not grounding. Weak answer: "adding references."

**5. "Why is omission harder to measure than hallucination?"**
Strong answer: omission requires a pre-specified list of what should be included; hallucination can
be checked against the source. Weak answer: "omissions are invisible."

**6. "How do you handle a plan section with sparse evidence?"**
Strong answer: constrain generation, mark suggestions, or abstain; do not fill gaps. Weak answer:
"let the model infer."

**7. "What is abstention and when should the system use it?"**
Strong answer: declining to claim when evidence is insufficient; threshold tied to error costs;
monitor the rate. Weak answer: "when unsure."

**8. "How would you measure the utility of a summary workflow?"**
Strong answer: time ledger with baseline, review, correction, and harm; measured locally. Weak
answer: "time saved."

**9. "What does citation coverage tell you?"**
Strong answer: the share of claims with a citation; says nothing about support; pair with precision.
Weak answer: "that the summary is grounded."

**10. "Why do published hallucination rates vary so much?"**
Strong answer: different units, definitions, tasks, and source conditions; not comparable without
the protocol. Weak answer: "some models are better."

**11. "How do you prevent negation flips?"**
Strong answer: polarity check, negation-aware audit, claim-level review of negation-heavy sections.
Weak answer: "the model should learn it."

**12. "What would you monitor after deployment?"**
Strong answer: unsupported rate by section, omission rate, citation precision, abstention rate, and
a rolling audit sample. Weak answer: "user satisfaction."

**13. "How does the source's quality affect the summary's quality?"**
Strong answer: a faithful summary of a wrong note is still wrong; source quality is a separate
layer; check it before blaming the model. Weak answer: "the model should handle it."

**14. "What is the smallest defensible evaluation for a summary pilot?"**
Strong answer: a claim-level audit of a sample with a stated support definition, an omission list,
citation precision, and a reader study, with sample sizes for each. Weak answer: "a clinician looks
at a few summaries."

**15. "Why not just use ROUGE?"**
Strong answer: ROUGE rewards lexical overlap, penalizes paraphrase, and does not measure support or
omission; it is a general-domain proxy. Weak answer: "ROUGE is standard."

## 15. Key terms

- **Abstention** — declining to state a claim.
- **Abstractive** — generating new wording.
- **Atomic fact** — the finest claim unit.
- **Attribution** — the subject a claim applies to.
- **BLEU** — n-gram precision against references.
- **Citation** — a reference to a source span.
- **Citation coverage** — share of claims with a citation.
- **Citation precision** — share of citations that support the claim.
- **Claim** — a statement verifiable against source.
- **Claim decomposition** — splitting output into verifiable units.
- **Coherence** — logical structure of the summary.
- **Critical finding** — a fact the summary must include.
- **Distortion** — a claim that misstates the source.
- **Extractive** — selecting source sentences.
- **Fabrication** — a claim with no source support.
- **Faithfulness** — agreement between claim and source.
- **Fluency** — readability of the summary.
- **Grounding** — linking a claim to evidence.
- **Hallucination** — an unsupported or fabricated claim.
- **Hedging** — stating a claim with uncertainty.
- **Hybrid** — extractive selection plus rewriting.
- **Major error** — an error that could change management.
- **Omission** — a critical fact absent from the summary.
- **Review burden** — time to verify and correct.
- **ROUGE** — n-gram overlap with a reference.
- **Summarization** — producing a shorter representation.
- **Temporal accuracy** — correct timing of a claim.
- **Claim-evidence graph** — claims linked to their source spans.
- **Decision-relevant omission** — missing the distinction a reader needs, not just the category.
- **Reader study** — clinician evaluation of summary quality and usability.
- **Source quality** — accuracy of the document the summary is based on.
- **Audit sample** — the subset of summaries reviewed for monitoring.
- **Decomposition rule** — the written rule for splitting a summary into claims.
- **Support definition** — the criterion for labeling a claim supported.
- **Omission list** — the pre-specified critical findings.
- **Net utility** — time saved minus review and correction time.
- **Grounding contract** — every claim has a citation that supports it.
- **Plan section risk** — the concentration of unsupported claims in forward-looking text.
- **Citation audit** — checking whether each citation supports its claim.
- **Extractive-first** — selecting source sentences before rewriting.
- **Severity scale** — the major/minor classification of an error.
- **Time ledger** — the accounting of drafting, review, and correction time.

## 16. Sources

All URLs accessed 25 October 2026.

1. Williams CYK, Gasparini A, Caskey J, et al. "Physician- and large language model-generated
   hospital discharge summaries: a blinded, comparative quality and safety study." medRxiv. 2024.
   https://doi.org/10.1101/2024.09.29.24314562
2. Asgari A, et al. "A framework to assess clinical safety and hallucination rates of LLMs for
   medical text summarisation." medRxiv. 2024. https://doi.org/10.1101/2024.09.12.24313556
3. Van Veen D, Van Uden C, Blankemeier L, et al. "Adapted large language models can outperform
   medical experts in clinical text summarization." Nat Med. 2024;30:1134-1142.
   https://doi.org/10.1038/s41591-024-02855-5
4. Hegselmann S, et al. "A data-centric approach to generate faithful and high quality patient
   summaries with large language models." Proc Mach Learn Res. 2024;248.
   https://proceedings.mlr.press/v248/hegselmann24a.html
5. Williams CYK, et al. "Evaluating large language models for drafting emergency department
   discharge summaries." medRxiv. 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11023681/
6. U.S. Food and Drug Administration. "Artificial Intelligence-Enabled Device Software Functions:
   Lifecycle Management and Marketing Submission Recommendations." Draft guidance, January 2025.
   https://www.fda.gov/media/184856/download
7. ONC. "HTI-1 Decision Support Interventions (DSI) Fact Sheet."
   https://healthit.gov/wp-content/uploads/2023/12/HTI-1_DSI_fact-sheet_508.pdf
8. ONC. "Information Blocking." https://www.healthit.gov/topic/information-blocking
9. U.S. Department of Health and Human Services. "Business Associates."
   https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/business-associates/index.html
10. Velupillai S, Suominen H, Liakata M, et al. "Using clinical Natural Language Processing for
    health outcomes research." J Biomed Inform. 2018;88:11-19.
    https://doi.org/10.1016/j.jbi.2018.10.005

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`08_deidentification.md`](08_deidentification.md)
- Next topic: [`10_llm-safety-in-clinical-text.md`](10_llm-safety-in-clinical-text.md)
- Claim-level evaluation design: [`07_evaluation.md`](07_evaluation.md)
- Annotation of claims: [`06_annotation.md`](06_annotation.md)
- Module survey: [`../../09_clinical_nlp.md`](../../09_clinical_nlp.md)
- Labs: [`../../labs/`](../../labs/)
