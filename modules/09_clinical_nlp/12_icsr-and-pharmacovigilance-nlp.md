# NLP for Pharmacovigilance and ICSR Automation

> **Module 09 — Clinical Natural Language Processing**
> Topic 12 of 13
> Estimated study time: 4 hours
> Prerequisites: Topics 03, 04, 05, and 06
> Version 2026.3

## 1. Why this topic exists

A patient takes a new medicine, develops jaundice three weeks later, and is admitted to hospital.
The prescriber, a nurse, or the patient reports it. From that moment, a legal clock is running. In
the United States, a marketing authorisation holder that receives a serious and unexpected
suspected adverse reaction must report it to the Food and Drug Administration (FDA) within 15
calendar days (21 CFR 314.80(c)(1)(i)). In the European Union, the same case must reach EudraVigilance
within 15 days, or within 7 days if it was fatal or life-threatening in a clinical trial (Regulation
(EU) 536/2014, Article 42). The clock starts when any employee or contractor of the company first
receives the information, not when the safety department learns of it.

Pharmacovigilance (PV) is the function that runs this process. It is unusual among data functions
because its output is a regulatory submission with a statutory deadline, and because the volume is
large and the input is unstructured. A global sponsor receives reports by email, telephone,
web form, literature alert, social media post, clinical trial serious adverse event form, and
regulator exchange. Most arrive as free text in a language the safety database does not read.

The economics and the law point in opposite directions. Economics favours automation: intake,
triage, and coding are labour-intensive, and the per-case cost is dominated by human reading.
Law favours conservatism: a missed serious case is a regulatory finding, a labelling change, or in
the worst case a patient harmed by a drug that should have carried a warning. The design problem is
therefore not "can a model extract adverse events" but "where in the process can a model's output
be trusted without a human, and where must a human see every case".

This topic covers the pipeline from an unstructured report to a submitted Individual Case Safety
Report (ICSR): the four minimum criteria for a valid case, the ICH E2B(R3) data model, Medical
Dictionary for Regulatory Activities (MedDRA) coding, seriousness, causality, expectedness, duplicate
detection, and the timeliness clocks. It also covers the metrics that matter for case intake, which
are not the metrics a general NLP team would choose. A model with 0.90 F1 on event extraction can
still miss one serious case in twenty, and that one case is the regulatory event.

> **Note:** The regulatory content here is educational and describes requirements as published in
> 2026. It is not legal advice, and the applicable requirements depend on the product, the
> jurisdiction, and the marketing authorisation.

## 2. Learning outcomes

By the end of this topic you can:

1. **Determine** the day-zero date for a case from source documents and **compute** the resulting
   reporting deadline in calendar days.
2. **Assess** the four minimum criteria for a valid ICSR and **classify** a document as valid,
   incomplete, or invalid with a stated reason.
3. **Map** a verbatim adverse event term to a MedDRA Lowest Level Term (LLT) and Preferred Term (PT),
   and **explain** why the PT rather than the LLT drives regulatory counting.
4. **Apply** the six seriousness criteria and the expectedness rule against reference safety
   information, and **derive** whether a case is a Suspected Unexpected Serious Adverse Reaction
   (SUSAR).
5. **Compute** intake sensitivity, positive predictive value (PPV), duplicate-link precision, and
   clock latency, each with its unit and denominator.
6. **Design** a duplicate-detection strategy with blocking, scoring, and a review band, and
   **justify** the thresholds.
7. **Build** a human-in-the-loop triage that prioritises by regulatory deadline rather than by model
   confidence alone.
8. **Defend** an AI-assisted PV pipeline in an inspection by naming the validation evidence, the
   audit trail, and the residual manual controls.

## 3. The core mental model

The useful mental model is that an ICSR is not a document. It is a **legal object assembled from
fields, with a clock attached**.

Three consequences follow, and each is a design constraint.

**The case is assembled, not written.** An ICSR is a set of structured fields: who reported, who
was the patient, what drug, what event, how serious, how related, what was the outcome. The
narrative is one field among many. A pipeline that produces a good narrative and a wrong
seriousness flag has failed, because the seriousness flag drives the submission decision.

**The clock starts before the case exists.** Day zero is the date the information was first received
by any company personnel. A sales representative who hears about a reaction at a conference starts
the clock. The safety department may not see the document for a week. Every day between receipt and
intake is a day removed from the reporting window, and it is invisible in a case-level metric that
starts at intake.

**The classifications are simultaneous and independent.** Validity, seriousness, causality, and
expectedness are four different questions with four different evidence sources and four different
error costs. A model tuned to predict "will this be a 15-day case" collapses them and makes the
error unattributable.

```text
  SOURCES                    INTAKE                    CASE ASSEMBLY
  email / call log           +----------------+       +------------------+
  literature alert   ----->  | 1 classify     | ----> | validity gate    |
  trial SAE form             |   source type  |       | 4 minimum        |
  web form                   +----------------+       | criteria         |
  social media                      |                 +--------+---------+
  regulator exchange                v                          |
                            +----------------+         valid  |  invalid
                            | 2 day zero     |                v
                            |   metadata     |         follow-up queue
                            +----------------+                |
                                    |                          |
                                    v                          v
                            +----------------+       +------------------+
                            | 3 language     |       | duplicate search |
                            |   translate    | ----> | block, score,    |
                            +----------------+       | link or review   |
                                                     +--------+---------+
                                                              |
                            +----------------+                v
                            | 7 event        |       +------------------+
                            |   extraction   | ----> | 8-12 classify    |
                            | 9 MedDRA code  |       | seriousness      |
                            +----------------+       | causality        |
                                                     | expectedness     |
                                                     +--------+---------+
                                                              |
                                              +---------------+---------------+
                                              |                               |
                                     SUSAR: 15-day                 non-SUSAR: periodic
                                     (7-day if fatal or            report
                                     life-threatening)
                                              |                               |
                                              v                               v
                                     priority review lane          standard lane
```

Read the diagram with the clock in mind. The left column consumes calendar days. The right column
consumes clinical judgement. The only stage that is purely mechanical is the day-zero determination,
and it is the stage most often got wrong, because it requires reading the metadata of the source
document rather than the case record.

### The four questions

| Question | Evidence source | Output | Error that matters |
|---|---|---|---|
| Is this a valid case? | The report itself | Four-criteria status | A valid case discarded |
| Is it serious? | The event description and outcome | Six-criterion flags | A serious case treated as non-serious |
| Is it caused by the drug? | Temporality, dechallenge, rechallenge, alternatives | Causality category | A causal case attributed to background |
| Is it expected? | Reference safety information | Expected or unexpected | An unexpected case filed as expected |

The four answers combine. A case that is serious, unexpected, and at least possibly related is a
SUSAR and enters the expedited lane. A case that is serious and expected enters periodic reporting.
A case that is non-serious enters periodic reporting regardless of expectedness. A case that fails
validity enters follow-up, and if follow-up fails, it is not submitted but must still be retained
and reconciled.

The combination is the product of the four classifications, which is why a single end-to-end
classifier is a poor design. When it makes an error, the reviewer cannot tell whether the event was
missed, the seriousness was missed, or the expectedness lookup used an outdated document.

### The clock budget

Think of the 15-day window as a budget with lanes.

```text
 day 0   1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
   |----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
   [intake][triage ][code][medical review][QC][submit]
   <-------------------- planned lane: 8 days ------------------->[buffer 7 d]
```

A planned lane of 8 days leaves 7 days of buffer. The buffer absorbs weekends, queries to the
reporter, translation, and duplicate resolution. It is consumed by intake latency first, because
intake latency is the only delay that happens before anyone is accountable for the case.

The arithmetic is unforgiving. If intake takes 4 days instead of 1, the buffer falls from 7 to 4. If
a duplicate query then takes 3 days, the case is at the deadline with no margin. This is why mature
organisations measure intake latency as a first-class metric, not as a back-office detail.

### Why sensitivity dominates

In most NLP tasks, precision and recall trade off against each other and the choice depends on the
cost of each error. In ICSR intake the costs are asymmetric by orders of magnitude. A false positive
costs a human review, perhaps 10 to 20 minutes of a safety specialist's time. A false negative
costs a missed regulatory deadline, a potential labelling change, and in the worst case a patient
exposed to a risk that was not characterised.

The practical consequence is that the intake model should operate at a high-sensitivity operating
point and push the precision problem to the review queue. A model with sensitivity 0.99 and PPV
0.40 is often preferable to one with sensitivity 0.90 and PPV 0.90, because the first misses one
case in a hundred and the second misses ten.

The same logic does not apply to coding. A wrong MedDRA PT is corrected by a human coder at low
cost, and the coding step is downstream of the decision to process the case. So the coding model can
operate at a balanced point, while the intake model cannot.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Pharmacovigilance (PV) | Science and activities relating to detection, assessment, understanding, and prevention of adverse effects | Used to mean only case processing; it includes signal detection and risk management | None |
| Adverse event (AE) | Any untoward medical occurrence in a patient administered a medicinal product, not necessarily caused by it | Used interchangeably with adverse drug reaction | None |
| Adverse drug reaction (ADR) | An AE for which a causal relationship with the drug is at least a reasonable possibility | Called an AE when causality has been assessed | None |
| Serious adverse event (SAE) | An AE meeting at least one of six seriousness criteria | Confused with "severe", which is an intensity grading | Six criteria |
| Severity | Intensity of an event (mild, moderate, severe) | Confused with seriousness; a severe headache is not serious | Ordinal scale |
| Seriousness criterion | One of: death, life-threatening, hospitalisation or prolongation, disability, congenital anomaly, medically important | Applied to the drug rather than the event | E.i.3.2 a-f |
| Life-threatening | The patient was at immediate risk of death at the time of the event | Applied to an event that could have been fatal if untreated | Boolean |
| Expectedness | Whether the reaction is listed in the reference safety information | Judged against the investigator brochure instead of the company core data sheet | Expected or unexpected |
| Unexpected | Not listed in the reference safety information, or listed at greater severity or specificity | Treated as "surprising" | Boolean |
| Causality | The assessment of whether the drug caused the event | Treated as a probability; it is a structured judgement | WHO-UMC categories |
| Suspected unexpected serious adverse reaction (SUSAR) | A serious, unexpected reaction with at least a reasonable possibility of causation | Applied to any serious case | Boolean |
| Individual Case Safety Report (ICSR) | The structured report of one case, transmitted in the E2B format | Used to mean the narrative only | E2B(R3) |
| E2B(R3) | ICH standard for electronic transmission of ICSRs, with defined data elements and XML message structure | Assumed to be a database schema; it is a message specification | Data element identifiers |
| Day zero | Date the information was first received by any personnel of the company or its contractors | Set to the date of intake | Calendar date |
| Minimum criteria | Identifiable patient, identifiable reporter, suspect product, adverse event | Treated as a quality score; it is a gate | Four booleans |
| Follow-up | Attempt to obtain missing information for an incomplete case | Confused with case amendment | None |
| Duplicate | Two or more reports describing the same case | Defined by identical sender identifiers only | Count, and similarity score |
| Case version | An iteration of a submitted case with new information | Confused with a new case | Integer |
| Nullification | Withdrawal of a submitted case, with a reason | Used for corrections that should be amendments | Code |
| MedDRA | Medical Dictionary for Regulatory Activities, the ICH terminology for regulatory coding | Treated as a clinical ontology; it is a regulatory dictionary | Version, for example 29.1 |
| Lowest Level Term (LLT) | The most granular MedDRA term, closest to the verbatim | Used for counting; PT is the counting level | Code |
| Preferred Term (PT) | The standard term for a medical concept, used for reporting and signal detection | Confused with the verbatim term | Code |
| System Organ Class (SOC) | The top-level MedDRA hierarchy, 27 categories | Used for statistical analysis without noting multi-axial assignment | Code |
| Standardised MedDRA Query (SMQ) | A validated grouping of terms for a medical condition or area of interest | Treated as a single PT | Query identifier |
| Reference Safety Information (RSI) | The document that defines expectedness for a product | Confused with the full prescribing information | Document version |
| Company Core Data Sheet (CCDS) | The company's reference document for a product's safety information | Used when the local label is the applicable reference | Document version |
| Periodic Benefit-Risk Evaluation Report (PBRER) | Periodic report of the benefit-risk profile, replacing the PSUR in the ICH regions | Confused with the Development Safety Update Report | Report cycle |
| Development Safety Update Report (DSUR) | Annual report for an investigational drug | Confused with the PBRER | Annual |
| Marketing authorisation holder (MAH) | The company responsible for the product and its reporting obligations | Confused with the local affiliate | Legal entity |
| Qualified Person for Pharmacovigilance (QPPV) | The individual responsible for the PV system in the EU | Treated as a ceremonial role | Named person |
| FAERS | FDA Adverse Event Reporting System | Treated as a complete record of all US cases | Database |
| EudraVigilance | The EU database for adverse reaction reports | Confused with the national competent authority systems | Database |
| VigiBase | The World Health Organization global individual case safety report database | Treated as public case-level data | Database |
| Literature case | A case derived from a published article, with the citation as the source | Treated as lower quality than a spontaneous report | Citation |
| Solicited report | A report from an organised data collection system, such as a patient support programme | Treated as spontaneous | Source type code |
| Consumer report | A report from a patient or non-health professional | Assumed invalid; validity depends on the four criteria, not the reporter type | Reporter type code |

Two definitions carry most of the operational weight.

**Serious versus severe.** Severity is the intensity of the event. Seriousness is the regulatory
consequence. A severe migraine that resolves in an hour at home is not serious. A mild rash that
leads to a three-day admission is serious. Systems that learn from text often conflate the two,
because "severe" is a frequent token and because the training label is usually the seriousness flag
rather than the reasoning behind it.

**Expected versus known.** Expectedness is not a judgement about whether the event is biologically
plausible. It is a lookup against a specific document version. A reaction that every clinician knows
about can still be unexpected if it is not in the reference safety information, and a reaction that
is listed at a lower severity can be unexpected when it occurs at a higher severity. The lookup
must be versioned, because the reference document changes and a case assessed against last year's
label is assessed incorrectly.

## 5. The mechanics, step by step

This is the procedure for taking one incoming document to one submitted case. It assumes a
marketing authorisation holder with an established safety database and a validated intake system.
The steps are ordered by the clock, not by the data model.

### Step 1. Classify the source and the report type

**Input:** A raw document or message, its metadata, and the channel it arrived through.

**Output:** A source type, a report type, and a routing decision.

**Rationale:** Source type determines the regulatory treatment. A spontaneous report from a
consumer, a solicited report from a patient support programme, a clinical trial serious adverse
event, and a literature case have different follow-up obligations and different expectedness
references. A report from a clinical trial is assessed against the investigator brochure; a
post-marketing report is assessed against the company core data sheet. Routing the wrong source
type to the wrong reference document is a silent error that produces a wrong expectedness answer.

Source classification is also where the clock is set, because the source metadata carries the
receipt date. A pipeline that classifies the text but ignores the metadata has thrown away the only
field that determines the deadline.

**Branch point:** Source and report type.

| Source | Report type code | Reference document | Follow-up obligation |
|---|---|---|---|
| Consumer email or web form | Spontaneous | CCDS or local label | Attempt if invalid or serious |
| Healthcare professional call | Spontaneous | CCDS or local label | Attempt for missing criteria |
| Patient support programme | Solicited | CCDS or local label | Protocol-defined |
| Clinical trial SAE form | Study | Investigator brochure | Mandatory within trial timelines |
| Literature article | Spontaneous, literature | CCDS or local label | Author contact where feasible |
| Regulatory authority | Spontaneous or study | As designated | Respond within the authority's timeline |
| Social media post | Spontaneous, unsolicited | CCDS or local label | Limited; document the attempt |

### Step 2. Determine day zero

**Input:** The source document, its headers, the mailbox or call-log metadata, and any forwarding
chain.

**Output:** A day-zero date with its evidence, and the derived reporting deadline.

**Rationale:** Day zero is the date the information was first received by any personnel of the
company or its contractors. It is not the date of intake, the date of triage, or the date the case
is entered into the database. In a forwarded email chain, the earliest timestamp in the chain is
usually the relevant one. In a call log, the call start time is. In a literature alert, the date the
article was received by the company, not the publication date.

The difference is not academic. In the worked example in Section 6, a three-day discrepancy between
the mailbox date and the date in the document body consumes the entire buffer.

**Branch point:** If the receipt date cannot be established, use the earliest documented date and
record the uncertainty. Do not use the intake date as a default. A default that flatters the metric
is worse than a documented uncertainty.

### Step 3. Identify the language and translate

**Input:** The source text and any attachments.

**Output:** A language tag, a translation, and a preserved original.

**Rationale:** Cases arrive in the reporter's language and are processed in the company's working
language. Translation is a lossy step and a regulated one: the original verbatim must be preserved
in the case, because the E2B standard carries both the reported term and its translation.

Two failure modes dominate. The first is a false cognate or a domain term translated by a
general-purpose engine. The second is a translation that silently normalises an ambiguous statement
into a definite one, turning "could be" into "is". Both are addressed by keeping the original, by
using a medically adapted translation path, and by having a speaker of the language review cases
that drive a serious decision.

**Branch point:** If the language is not supported by the translation path, the case goes to a human
translator before triage, not after. A machine translation of an unsupported language is a guess
with a fluent surface.

### Step 4. Assess the four minimum criteria

**Input:** The translated case text and the extracted entities.

**Output:** Four booleans, a validity status, and a list of missing elements.

**Rationale:** A valid ICSR requires an identifiable patient, an identifiable reporter, a suspect
product, and an adverse event. The criteria are gates, not scores. A case missing any one is not
submitted as an expedited report; it enters follow-up, and if follow-up fails it is retained as an
invalid report with the reason recorded.

The judgement calls are in "identifiable". A patient described as "a 54-year-old woman in the
waiting room" is identifiable enough for a spontaneous report in most jurisdictions. "A patient"
with no age, sex, initials, or record number is not. A reporter described as "a physician" with no
name or contact point is borderline; most companies accept a named institution with a verifiable
role.

**Branch point:** Validity outcomes.

| Status | Condition | Action | Clock |
|---|---|---|---|
| Valid | All four present | Proceed to duplicate search | Running |
| Valid with query | All four present, one is low confidence | Proceed, flag for verification | Running |
| Incomplete | One or more missing, reporter contactable | Follow-up request within 24 hours | Running |
| Invalid, retained | Missing and not obtainable | Retain with reason, no submission | Stopped for expedited |
| Not a case | No adverse event, or product not identifiable as ours | Route out of PV | Not applicable |
| Duplicate intake | Same case already in the system | Link to the existing case | Inherited from the parent |

### Step 5. Search for duplicates

**Input:** The case's identifying fields: product, event terms, patient age and sex, initials,
reporter, receipt date, and narrative.

**Output:** A ranked candidate list with similarity scores, and a link, review, or distinct decision.

**Rationale:** Duplicates are common because the same event reaches a company through a
representative, a call centre, a regulator, and a literature alert. Counting the same case twice
inflates the safety database and distorts signal detection. Missing a duplicate fragments the case,
so that no single record contains the full outcome.

There is no regulatory standard algorithm. The established approach is probabilistic record
linkage: block on one or two high-specificity fields, score candidates on the rest, and route the
middle band to a human. Blocking on product plus PT keeps the comparison set small. The score should
weight rare agreements more than common ones, which is the Fellegi-Sunter logic: agreement on a
rare event is stronger evidence than agreement on a common one.

**Branch point:** Duplicate score bands.

| Score | Interpretation | Action | Typical error |
|---|---|---|---|
| 0.85 to 1.00 | Strong match | Auto-link, merge fields, keep the earliest receipt date | Over-merging two patients on the same drug |
| 0.65 to 0.85 | Possible match | Manual review within one business day | Queue delay consumes the clock |
| Below 0.65 | Distinct | Create a new case | Missed duplicate fragments the case |

### Step 6. Merge or link the case and reconcile follow-up

**Input:** The new document, the existing case if linked, and prior follow-up correspondence.

**Output:** A merged case version with field-level provenance and a follow-up plan.

**Rationale:** Merging is not concatenation. Conflicting values (two ages, two onset dates, two
outcomes) must be resolved with a rule and a record of the rule. The usual rule is that the most
recent information from the most reliable source wins, but the earlier value is retained in the
audit trail. A merge that silently overwrites the first receipt date is a compliance defect,
because it resets the clock.

Follow-up should be targeted. A generic request for "more information" produces a generic response.
A request that names the missing element ("was the patient hospitalised, and on what date") produces
a usable answer.

### Step 7. Segment the narrative and extract events

**Input:** The case narrative, which may mix history, concomitant medications, event description,
and outcome in one paragraph.

**Output:** Candidate event mentions with spans, negation status, subject, and temporality.

**Rationale:** The narrative is where the event is described, and it is the least structured part of
the case. Extraction must distinguish the event from the indication ("treated with atorvastatin for
hyperlipidaemia" is not an event), from a history item ("previous MI in 2019"), and from a negated
statement ("no rash").

The three highest-value distinctions are:

- **Indication versus event.** The reason for prescription is not an adverse reaction. This is the
  most common extraction error in real narratives, because indication and event often appear in the
  same sentence.
- **Event versus history.** A prior event is not the current event, but it may be relevant to
  causality.
- **Event versus outcome.** "Hospitalised with hepatic failure and died" contains two events and
  one outcome. Collapsing them loses the seriousness criterion.

**Branch point:** If the narrative contains more event mentions than distinct events, deduplicate
by PT before counting. A case with five mentions of "jaundice" is one event, not five.

### Step 8. Classify the drug role

**Input:** All products named in the case: suspect, concomitant, interacting, vaccine, comparator,
and products mentioned as prior therapy.

**Output:** A role label per product, with supporting text.

**Rationale:** Only the suspect product drives the causality assessment and the expectedness
lookup. A case that lists eight medications has one or two suspects and six concomitant products.
Misclassifying a concomitant as suspect creates a false signal for a drug that was not implicated;
misclassifying a suspect as concomitant hides a real one.

The signals for "suspect" are explicit ("the patient took X and developed Y"), temporal proximity,
and dechallenge. The signals for "concomitant" are lists, "also taking", and long-standing therapy.
Neither is reliable alone, which is why the role should be a field a human can correct, not a hidden
intermediate.

### Step 9. Code the events to MedDRA

**Input:** Verbatim event terms and the current MedDRA version.

**Output:** An LLT, a PT, the SOC path, and a coding confidence.

**Rationale:** MedDRA is a five-level hierarchy: LLT, PT, High Level Term (HLT), High Level Group
Term (HLGT), and SOC. Coding to the LLT preserves the verbatim's specificity; reporting and signal
detection use the PT. A term can belong to more than one SOC, which is why SOC counts do not sum to
the number of events.

Autocoding is a retrieval problem with a constraint layer. Dense retrieval over LLT synonyms
produces candidates; the constraint layer rejects candidates that conflict with the event's
anatomy, with the drug's indication, or with the negation status. The output should be a ranked
list with a confidence, not a single code, because the coder needs to see alternatives when the
top choice is wrong.

MedDRA changes twice a year, in March and September. A coding that was correct in version 28.1 may
be non-current in 29.1 because a term was demoted or split. Version drift is a real inspection
finding, not a theoretical one.

**Branch point:** Coding outcomes.

| Confidence | Condition | Action |
|---|---|---|
| High | Single candidate, exact synonym, unambiguous context | Accept with audit record |
| Medium | Top candidate plausible, alternatives differ in specificity | Coder confirms |
| Low | Candidates span different SOCs, or the verbatim is ambiguous | Coder codes manually |
| No candidate | Verbatim is a phrase or a sentence | Coder splits and codes each event |
| Version conflict | Term non-current in the current version | Remap and record the mapping |

### Step 10. Classify seriousness

**Input:** The event descriptions, outcome, hospitalisation dates, and any narrative statements
about severity.

**Output:** Six booleans, a seriousness status, and the supporting span for each true criterion.

**Rationale:** Seriousness is a legal classification with six criteria, and each must be supported
by a quote. "Hospitalised" in the narrative is evidence for criterion three. "Went to the emergency
room and was sent home" is not. "Severe" is not a criterion. "Life-threatening" requires that the
patient was at immediate risk of death at the time of the event, not that the event could have been
fatal if untreated.

The most common error is inferring hospitalisation from context. The second is treating an
emergency department visit as an admission. The third is missing a "medically important" event
because it does not fit the other five criteria.

**Branch point:** Seriousness decision.

| Evidence | Serious | Criterion | Note |
|---|---|---|---|
| Death recorded as outcome | Yes | Death | Verify the date and cause |
| Immediate risk of death at the time | Yes | Life-threatening | Not the same as "could have died" |
| Admitted, or stay prolonged | Yes | Hospitalisation | Elective admission for a pre-existing condition does not count |
| Persistent disability | Yes | Disability | Requires a statement about function |
| Congenital anomaly | Yes | Congenital anomaly | Applies to exposure in pregnancy |
| Requires intervention to prevent a serious outcome | Yes | Medically important | Judgement; document the reasoning |
| Emergency visit, discharged same day | Usually no | None | Unless another criterion applies |
| "Severe" pain, no other criterion | No | None | Severity is not seriousness |

### Step 11. Assess causality

**Input:** The event, the suspect drug, the temporal relationship, dechallenge and rechallenge
information, alternative causes, and prior knowledge of the reaction.

**Output:** A causality category with the reasoning recorded.

**Rationale:** Causality is a structured judgement, not a probability. The World Health Organization
Uppsala Monitoring Centre (WHO-UMC) scale is the most widely used: certain, probable, possible,
unlikely, conditional, and unassessable. The categories depend on four pieces of evidence:
a plausible time relationship, a positive dechallenge, a positive rechallenge, and the absence of
alternative explanations.

A language model can gather the evidence and flag which criteria are met. It should not assign the
final category without a reviewer, because the category is a medical judgement with regulatory
consequences and because the evidence is frequently incomplete in ways the text does not announce.

**Branch point:** The WHO-UMC categories, abbreviated.

| Category | Time relationship | Dechallenge | Rechallenge | Alternatives |
|---|---|---|---|---|
| Certain | Plausible | Positive | Positive | Not plausible |
| Probable | Plausible | Positive | Not done | Not plausible |
| Possible | Plausible | Not done or unclear | Not done | Plausible |
| Unlikely | Implausible | Negative | Not done | Plausible |
| Conditional | Plausible | Not done | Not done | Not assessable |
| Unassessable | Insufficient information to assess any criterion | | | |

### Step 12. Assess expectedness

**Input:** The coded PT, the applicable reference safety information, and its version.

**Output:** Expected or unexpected, with the reference document and version recorded.

**Rationale:** Expectedness is a lookup, and the lookup is only as good as the version control. The
applicable document differs by region and by product status: the company core data sheet for a
marketed product, the local label where the local regulator requires it, the investigator brochure
for a trial. A case assessed against the wrong document can be filed as expected when it is not, or
the reverse.

Unexpected also covers greater severity or specificity. A reaction listed as "hepatotoxicity" does
not automatically make "hepatic failure" expected, and a reaction listed as "rash" may not cover
"Stevens-Johnson syndrome". The judgement is clinical and should be made by a qualified reviewer,
with the model providing the candidate terms and the label text.

**Branch point:** If the reference document is not the current version, the assessment is void. The
case must be re-assessed. Version drift in the reference safety information is a common inspection
finding.

### Step 13. Decide the submission route and manage the clock

**Input:** Validity, seriousness, causality, expectedness, and the deadline.

**Output:** A submission route, a due date, and a priority.

**Rationale:** The combination determines the route. Serious, unexpected, and at least possibly
related is expedited. Everything else is periodic, with the exception of specific local rules.
The decision should be computed from the four fields, not predicted end to end, so that an error is
attributable to a field.

Priority should be by deadline, not by model confidence. A case with two days left and a low model
confidence is more urgent than a case with fourteen days left and a high one.

**Branch point:** Submission routes.

| Serious | Unexpected | Related | Route | Deadline |
|---|---|---|---|---|
| Yes | Yes | Yes or possible | Expedited, SUSAR | 15 calendar days (7 if fatal or life-threatening in a trial) |
| Yes | Yes | No or unlikely | Periodic | Per report cycle |
| Yes | No | Any | Periodic | Per report cycle |
| No | Any | Any | Periodic | Per report cycle |
| Invalid | Any | Any | None until follow-up | Clock running for follow-up |

### Step 14. Quality control, audit, and reconciliation

**Input:** The assembled case, the extraction and coding decisions, the reviewer edits, and the
submission record.

**Output:** A QC sign-off, an immutable audit trail, and a reconciliation report.

**Rationale:** The audit trail is the evidence that the process was controlled. It should record
the model version, the input text, the extracted candidates, the selected codes, the reviewer, the
timestamp, and the reason for each override. Under 21 CFR Part 11, electronic records must be
attributable, legible, contemporaneous, original, and accurate.

Reconciliation closes the loop. Every serious case in the source systems should appear in the safety
database, and every submitted case should have a matching acknowledgement. A reconciliation that
compares counts without matching identifiers will miss the cases that matter, because the failures
are individual.

The procedure produces a case file that can answer four questions after the fact: what did we
receive, when did we receive it, what did we decide, and who decided it. A pipeline that produces
only a submitted XML message cannot answer the second question, and the second question is the one
that determines whether the submission was on time.

## 6. Worked example

One case, from a forwarded email to a submitted ICSR. The case is synthetic. Dates, names, and
clinical details are invented, and the MedDRA-style codes are illustrative rather than licensed.

### The source document

A sales representative forwards an email from a physician to the medical information mailbox. The
email body reads:

> "I saw Mrs M.J. (63, female) on 27 February. She started atorvastatin 40 mg on 5 February for
> hyperlipidaemia. On 26 February she presented with jaundice and palpitations and was admitted to
> St Anne's for investigation. Liver enzymes were elevated. The statin has been stopped."

The email header shows the representative received it on 27 February and forwarded it on 2 March.
The safety mailbox received it on 2 March. Intake was completed on 3 March.

### Step 2: day zero

The regulation says the clock starts on first receipt by any personnel. The representative is
company personnel. Day zero is therefore 27 February 2026, not 2 March.

Deadline = 27 February + 15 calendar days = 14 March 2026.

March 14, 2026 is a Saturday. Calendar days do not pause for weekends, so a team that plans to
submit "on the deadline" has planned to submit on a day when regulatory gateways are quiet.

If the pipeline had used the mailbox date, it would have computed 2 March + 15 = 17 March and
believed it had five days more than it did. That is the first diagnostic in this topic, and it is
the most consequential single field in the case.

### Step 4: validity

| Criterion | Evidence | Status |
|---|---|---|
| Identifiable patient | Initials, age, sex | Met |
| Identifiable reporter | Named physician, hospital, email domain | Met |
| Suspect product | Atorvastatin 40 mg, start date | Met |
| Adverse event | Jaundice, palpitations, elevated liver enzymes | Met |

The case is valid. The criteria are about identifiability, not about proof: the reporter need not
be a healthcare professional, the event need not be confirmed, and the drug need not be the only one.

### Steps 7 to 9: events and coding

The narrative contains five event mentions and two distinct events.

| Mention | Role | Coded PT | SOC | Note |
|---|---|---|---|---|
| Jaundice | Event | Jaundice | Hepatobiliary disorders | Also a sign of hepatotoxicity |
| Palpitations | Event | Palpitations | Cardiac disorders | Separate event |
| Hyperlipidaemia | Indication | Not coded | Not applicable | Reason for atorvastatin |
| Elevated liver enzymes | Event, supporting | Hepatic enzyme increased | Investigations | Supports the hepatotoxicity hypothesis |
| "Statin stopped" | Action | Not coded | Not applicable | Dechallenge, not an event |

Coding the indication as an event is the most common error in this narrative. Hyperlipidaemia
appears in the same sentence as the drug and the start date, and a sentence-level extractor that
learns "drug near condition means reaction" will code it. The consequence is a false signal for a
reaction that is in fact the treated condition.

The two event PTs are Jaundice and Palpitations. Hepatic enzyme increased is a third PT if the
company codes supporting findings separately. The choice is a convention, and it must be consistent
across the database, because inconsistency changes the denominator in signal detection.

### Step 10: seriousness

| Criterion | Flag | Evidence |
|---|---|---|
| Death | False | No death recorded |
| Life-threatening | False | No statement of immediate risk of death |
| Hospitalisation | True | Admitted 26 February, still inpatient |
| Disability | False | Not stated |
| Congenital anomaly | False | Not applicable |
| Medically important | False | Not needed; hospitalisation already establishes seriousness |

Serious = true, on one criterion. The classification does not depend on the severity of the
jaundice, the elevation of the enzymes, or the clinician's opinion that the case was mild. A
one-night admission for investigation is a hospitalisation.

### Step 11: causality

| WHO-UMC criterion | Finding | Direction |
|---|---|---|
| Plausible time relationship | 21 days from start to onset | Supports |
| Positive dechallenge | Drug stopped 26 February, enzymes improving by 3 March | Supports |
| Positive rechallenge | Not done | Neutral |
| Alternative explanation | Gallstones on ultrasound, no duct dilatation | Weakens |

The alternative explanation is plausible but not sufficient: gallstones without duct dilatation do
not fully explain a hepatocellular pattern. The category is **possible**, not probable, because an
alternative cause is present.

The evidence table is a good model output; the category is a reviewer decision.

### Step 12: expectedness

The applicable reference document is the company core data sheet, version 7.2, effective 1 January
2026. It lists hepatotoxicity, myalgia, and elevated transaminases. It does not list jaundice or
palpitations.

| PT | Listed in CCDS 7.2 | Expectedness |
|---|---|---|
| Jaundice | No, though hepatotoxicity is listed | Unexpected |
| Palpitations | No | Unexpected |
| Hepatic enzyme increased | Yes, as elevated transaminases | Expected |

The jaundice judgement is the interesting one. A reviewer could argue that jaundice is a
manifestation of hepatotoxicity and therefore expected. The conservative and common practice is to
treat the more specific and more severe term as unexpected unless the reference document names it
or a severity qualifier covers it. The decision should be recorded with its reasoning, because it
determines whether the case is expedited.

### Step 13: the route

Serious: yes. Unexpected: yes, on jaundice. Related: possible.

The case is a SUSAR and enters the 15-day expedited route. The 7-day route does not apply because
the event was not fatal or life-threatening.

### The clock, stage by stage

| Stage | Date | Day | Days consumed | Cumulative |
|---|---|---:|---:|---:|
| Receipt by representative | 27 Feb | 0 | 0 | 0 |
| Forwarded to safety mailbox | 2 Mar | 3 | 3 | 3 |
| Intake complete | 3 Mar | 4 | 1 | 4 |
| Triage and validity | 4 Mar | 5 | 1 | 5 |
| Coding | 5 Mar | 6 | 1 | 6 |
| Medical review | 9 Mar | 10 | 4 | 10 |
| QC | 11 Mar | 12 | 2 | 12 |
| Submission | 12 Mar | 13 | 1 | 13 |

Days remaining at submission = 15 − 13 = 2 calendar days.

The largest consumption is four days of medical review; the second is three days between receipt
and forwarding. The modelling work usually targets coding, which took one day. That is a
misallocation, and it is typical: the automated step is fast and the handoffs are slow.

### The duplicate that changes the answer

The duplicate search returns a candidate: a case received from the hospital pharmacist on
24 February, same patient initials, same age and sex, same drug, same event, different reporter.
The similarity score is 0.95, in the auto-link band. The case is linked, and the earlier receipt
date governs.

New day zero = 24 February 2026.
New deadline = 24 February + 15 = 11 March 2026.
Submission on 12 March is 16 days after receipt.

Days late = 16 − 15 = 1 calendar day.

The case that appeared to have two days of margin was one day late. Only a link changed, which is
why duplicate detection is a timeliness control and not only a data-quality control.

### Diagnostic A: recompute day zero from source metadata

Take 50 consecutive cases. For each, extract the earliest timestamp in the source documents (email
headers, call logs, fax headers) and compare it with the recorded first-receipt date in the case.

| Result | Interpretation |
|---|---|
| Mean discrepancy 0 days, maximum 0 | Metadata extraction is working |
| Mean discrepancy greater than 0, clustered at the intake date | The pipeline is defaulting to intake |
| Maximum discrepancy greater than 3 days on any case | Cases are at risk of late submission |
| Discrepancy correlated with source channel | One channel, usually sales or literature, is not being parsed |

A pipeline that records the intake date as day zero will pass a spot check on email cases and fail
on every case that arrives through a representative, a call centre, or a literature alert. The
diagnostic is cheap and it should run monthly.

### Diagnostic B: indication-as-event rate

Sample 100 narratives that mention both a drug and a condition. Count the cases where the condition
is the indication and is also coded as an event.

| Rate | Interpretation |
|---|---|
| Below 2 percent | The role classifier is working |
| 2 to 10 percent | Review the drug-role and indication features |
| Above 10 percent | The extractor is coding co-occurrence as causation; signals are contaminated |

The rate is easy to compute because indications are usually a separate case field, so the audit
compares coded events with the recorded indication. A high rate means the extractor is not modelling
sentence structure, and every downstream count is inflated by treated conditions.

### The varied assumption

Suppose the admission on 26 February was a previously scheduled elective admission for a
cholecystectomy, and the jaundice was noted on admission but did not cause or prolong the stay.

Under the seriousness criteria, hospitalisation counts only if the event caused or prolonged an
inpatient admission. A planned admission for an unrelated procedure does not. If no other criterion
applies, the case is not serious, and a non-serious case is not a SUSAR regardless of expectedness
or causality.

The route changes from 15-day expedited to periodic reporting, and the deadline disappears. The
case still needs to be coded, assessed, and included in the periodic report, and the causality
assessment still matters for signal detection. What changes is the clock and the submission
priority.

This is the assumption most often buried in an intake system: "hospitalisation" is extracted as a
flag from the presence of the word, and the causal link between the event and the admission is not
checked. A model that reads "admitted for investigation of jaundice" correctly flags the criterion.
A model that reads "admitted for cholecystectomy, also noted to have jaundice" and flags
hospitalisation has made a legal error from a plausible text pattern.

### Summary

| Question | Answer | Basis |
|---|---|---|
| Valid case | Yes | Four criteria met |
| Serious | Yes | Hospitalisation |
| Unexpected | Yes | Jaundice not in CCDS 7.2 |
| Related | Possible | Alternative cause present |
| Suspect product | Atorvastatin | Temporal and dechallenge evidence |
| Route | SUSAR, 15-day | Serious, unexpected, related |
| Day zero | 24 February | Earliest receipt, after duplicate link |
| Deadline | 11 March | Day zero plus 15 calendar days |
| Submitted | 12 March | One day late |

## 7. Code, query, or calculation

The script below implements the clock arithmetic, the duplicate score, the MedDRA-style hierarchy
lookup, the seriousness rule, the expectedness lookup, and the event-extraction metrics used in this
topic. It uses only the standard library and produces deterministic output.

Assumptions: MedDRA codes are omitted because the dictionary is licensed and codes change with each
release; the hierarchy is keyed by term for illustration. The duplicate score is a fixed-weight
approximation of a probabilistic linkage model, with weights chosen so that agreement on drug and
event dominates. Calendar days are used throughout, including weekends.

```python
from datetime import date, timedelta

def clock(receipt, submit=None, limit=15):
    """Return day zero, deadline, and the days used and left."""
    deadline = receipt + timedelta(days=limit)
    out = {"day_zero": receipt.isoformat(), "deadline": deadline.isoformat()}
    if submit:
        out["submitted"] = submit.isoformat()
        out["days_used"] = (submit - receipt).days
        out["days_left"] = (deadline - submit).days
    return out

print("A", clock(date(2026, 3, 2)))
print("B", clock(date(2026, 3, 2), date(2026, 3, 14)))
print("C", clock(date(2026, 2, 27), date(2026, 3, 14)))
print("D", clock(date(2026, 2, 27), date(2026, 3, 14), limit=7))

# Duplicate scoring: weighted agreement, Fellegi-Sunter style.
def dup_score(a, b):
    s = 0.0
    if a["drug"].lower() == b["drug"].lower(): s += 0.30
    if a["event"].lower() == b["event"].lower(): s += 0.30
    if a["initials"].upper() == b["initials"].upper(): s += 0.20
    if abs(a["age"] - b["age"]) <= 1: s += 0.10
    if a["sex"] == b["sex"]: s += 0.05
    if a["reporter"] == b["reporter"]: s += 0.05
    return round(s, 2)

a = dict(drug="atorvastatin", event="jaundice", initials="MJ", age=63, sex="F", reporter="clinic-441")
cands = [
    dict(drug="atorvastatin", event="jaundice", initials="MJ", age=63, sex="F", reporter="clinic-441"),
    dict(drug="atorvastatin", event="jaundice", initials="MJ", age=64, sex="F", reporter="clinic-902"),
    dict(drug="atorvastatin", event="jaundice", initials="AR", age=71, sex="M", reporter="clinic-118"),
    dict(drug="simvastatin", event="jaundice", initials="MJ", age=63, sex="F", reporter="clinic-441"),
]
for i, c in enumerate(cands, 1):
    sc = dup_score(a, c)
    band = "link" if sc >= 0.85 else ("review" if sc >= 0.65 else "distinct")
    print(f"candidate {i}: score {sc:.2f} -> {band}")

# MedDRA-style hierarchy, keyed by PT name for illustration.
meddra = {
    "jaundice":            ("PT", "Jaundice", "Hepatobiliary signs and symptoms", "Hepatobiliary disorders"),
    "hepatotoxicity":      ("PT", "Hepatotoxicity", "Hepatic disorders NEC", "Hepatobiliary disorders"),
    "palpitations":        ("PT", "Palpitations", "Cardiac arrhythmias", "Cardiac disorders"),
    "atrial fibrillation": ("PT", "Atrial fibrillation", "Supraventricular arrhythmias", "Cardiac disorders"),
    "rash":                ("PT", "Rash", "Rashes, eruptions and exanthems NEC", "Skin and subcutaneous tissue disorders"),
}
for term in ("jaundice", "palpitations", "rash"):
    pt, name, hlt, soc = meddra[term]
    print(f"{term:20s} -> {pt} {name} | {hlt} | {soc}")

# Seriousness: any true criterion makes the case serious.
def seriousness(flags):
    labels = ["death", "life_threatening", "hospitalisation", "disability",
              "congenital_anomaly", "medically_important"]
    return [l for l in labels if flags.get(l)]

case = dict(death=False, life_threatening=False, hospitalisation=True,
            disability=False, congenital_anomaly=False, medically_important=False)
print("serious criteria:", seriousness(case), "-> serious:", bool(seriousness(case)))

# Expectedness is a lookup against a versioned document, not a judgement of plausibility.
ccds = {"hepatotoxicity", "abdominal pain", "nausea"}
reported_pt = "jaundice"
print("unexpected:", reported_pt not in ccds)

# Event extraction metrics on a small gold set.
gold = {"jaundice", "palpitations", "hospitalisation", "atorvastatin"}
pred = {"jaundice", "palpitations", "hospitalisation", "liver failure"}
tp = len(gold & pred); fp = len(pred - gold); fn = len(gold - pred)
sens = tp / (tp + fn); ppv = tp / (tp + fp)
print(f"event extraction: tp={tp} fp={fp} fn={fn} sensitivity={sens:.3f} ppv={ppv:.3f}")

# Clock budget: planned lane versus intake delay.
milestones = [("intake", 1), ("triage", 2), ("coding", 1), ("medical review", 2), ("QC", 1), ("submit", 1)]
total = sum(m[1] for m in milestones)
print("planned lane days:", total, "buffer:", 15 - total)
for delay in (0, 2, 4, 6):
    print(f"  intake delay {delay} d -> buffer {15 - total - delay} d")
```

Output:

```text
A {'day_zero': '2026-03-02', 'deadline': '2026-03-17'}
B {'day_zero': '2026-03-02', 'deadline': '2026-03-17', 'submitted': '2026-03-14', 'days_used': 12, 'days_left': 3}
C {'day_zero': '2026-02-27', 'deadline': '2026-03-14', 'submitted': '2026-03-14', 'days_used': 15, 'days_left': 0}
D {'day_zero': '2026-02-27', 'deadline': '2026-03-06', 'submitted': '2026-03-14', 'days_used': 15, 'days_left': -8}
candidate 1: score 1.00 -> link
candidate 2: score 0.95 -> link
candidate 3: score 0.60 -> distinct
candidate 4: score 0.70 -> review
jaundice             -> PT Jaundice | Hepatobiliary signs and symptoms | Hepatobiliary disorders
palpitations         -> PT Palpitations | Cardiac arrhythmias | Cardiac disorders
rash                 -> PT Rash | Rashes, eruptions and exanthems NEC | Skin and subcutaneous tissue disorders
serious criteria: ['hospitalisation'] -> serious: True
unexpected: True
event extraction: tp=3 fp=1 fn=1 sensitivity=0.750 ppv=0.750
planned lane days: 8 buffer: 7
  intake delay 0 d -> buffer 7 d
  intake delay 2 d -> buffer 5 d
  intake delay 4 d -> buffer 3 d
  intake delay 6 d -> buffer 1 d
```

Four readings of the output matter.

Case A and case B show the same receipt date with different submission dates: the deadline does not
move, and the days left fall from a full window to three. Case C shows the effect of correcting day
zero by three days: the same submission is now exactly on the deadline, with zero days left. Case D
shows the 7-day trial clock applied to the same dates: the submission is eight days late. These four
lines are the whole argument for treating day zero as a data field with provenance.

The duplicate scores show the value and the danger of a fixed-weight model. Candidate 1 is the same
case and links at 1.00. Candidate 2 differs by one year of age and a different reporter, and still
links at 0.95, which is correct for a duplicate but would be wrong for two patients with common
initials on a common drug. Candidate 4 shares the patient but names a different statin, scores 0.70,
and lands in review. Fixed weights cannot express that a different drug should block a link, which
is why the blocking step matters more than the score.

The extraction line shows sensitivity 0.750 and PPV 0.750 on four gold items. The numbers are
uninformative at this sample size, but the structure is the point: sensitivity and PPV are computed
against a gold set, not against the model's own output, and both are reported because a single
number would hide which error dominates.

The clock budget line shows the arithmetic that operations managers should see: a lane of 8 days
leaves 7, and each day of intake delay removes one. At 6 days of delay the buffer is one day, and a
single weekend makes the case late.

## 8. Data, standards, and regulatory context

| Instrument | Issuer | What it constrains | Status |
|---|---|---|---|
| ICH E2A | International Council for Harmonisation (ICH) | Definitions of adverse event, adverse reaction, and serious | Binding via regional adoption |
| ICH E2B(R3), Implementation Guide v5.03 | ICH | Structure and content of the electronic ICSR message | Binding in the ICH regions |
| ICH E2C(R2) | ICH | Periodic benefit-risk evaluation report | Binding in the ICH regions |
| ICH E2D | ICH | Post-approval safety data management, including day zero and minimum criteria | Binding via regional adoption |
| ICH E2E | ICH | Pharmacovigilance planning | Guidance |
| 21 CFR 314.80 | US Food and Drug Administration (FDA) | Postmarketing reporting for drugs: 15-day, periodic | Binding |
| 21 CFR 600.80 | FDA | Reporting for biologics, including a 7-day clock for fatal or life-threatening suspected reactions | Binding |
| 21 CFR 312.32 | FDA | Investigational new drug safety reports: 7-day and 15-day | Binding |
| Regulation (EU) 536/2014, Article 42 | European Union | SUSAR reporting in clinical trials: 7 days fatal or life-threatening, 15 days otherwise | Binding |
| GVP Module VI | European Medicines Agency (EMA) | Management and reporting of adverse reactions | Binding in the EU |
| 21 CFR Part 11 | FDA | Electronic records and signatures | Binding |
| MedDRA | ICH | Terminology for coding events and indications | Required in the ICH regions |
| WHO-UMC causality scale | Uppsala Monitoring Centre | Structure of the causality assessment | Convention |
| AI guidance for drug development, draft, January 2025 | FDA | Use of AI in regulatory decision-making | Draft guidance |
| Reflection paper on AI in the medicinal product lifecycle, 2024 | EMA | AI use across the lifecycle, including pharmacovigilance | Reflection paper |

### The E2B(R3) structure

E2B(R3) is a message specification, not a database schema. It defines data elements grouped into
sections, each with a conformance status (required, optional, or conditional) and a data type. The
sections a data scientist will touch most are:

| Section | Content | Elements that drive decisions |
|---|---|---|
| C.1 | Case identification | C.1.1 sender's case identifier, C.1.2 date of creation, C.1.3 type of report, C.1.4 date first received from source, C.1.5 date of most recent information |
| C.2 | Primary source | Reporter identification and qualification |
| D | Patient characteristics | D.1 patient identification, age, sex, weight |
| E.i | Reaction or event | E.i.1.1a reported term, E.i.1.1b language, E.i.1.2 translated term, E.i.2.1a MedDRA version, E.i.2.1b LLT code, E.i.3.1 reporter's seriousness highlight, E.i.3.2a-f criteria, E.i.7 outcome |
| F | Test results | F.r test name, result, unit, normal range |
| G.k | Drug information | G.k drug name, dose, indication, start and stop dates, G.k.9.i.2.r drug-reaction matrix |
| H | Narrative | H.1 case narrative, H.4 sender's comments |

Two details are easy to get wrong. First, **seriousness lives at the event level**, not the case
level. A case with three events may have one serious and two non-serious events, and the flags
attach to the event. Second, **C.1.4 is the date first received from the source**, which is the
regulatory day zero for that report. It is not the date of creation (C.1.2), and the two must not be
confused when a case is amended.

### The timeliness clocks

| Jurisdiction | Trigger | Clock | Basis |
|---|---|---|---|
| US, marketed drugs | Serious and unexpected | 15 calendar days from initial receipt | 21 CFR 314.80(c)(1)(i) |
| US, biologics | Fatal or life-threatening suspected reaction | 7 calendar days, with a 15-day written follow-up | 21 CFR 600.80(c)(1) |
| US, investigational drugs | Unexpected fatal or life-threatening suspected reaction | 7 calendar days | 21 CFR 312.32(c)(1)(i) |
| US, investigational drugs | Other serious unexpected suspected reactions | 15 calendar days | 21 CFR 312.32(c)(1)(ii) |
| EU, clinical trials | Fatal or life-threatening SUSAR | 7 calendar days | Regulation (EU) 536/2014, Article 42 |
| EU, clinical trials | Other SUSARs | 15 calendar days | Regulation (EU) 536/2014, Article 42 |
| EU, post-authorisation | Serious adverse reactions | 15 calendar days | GVP Module VI |

A common error in requirements documents is to state that US drug reporting has a 7-day clock for
fatal cases. It does not. The 7-day clock for drugs exists in the biologics regulation and in the
investigational drug regulation, not in 21 CFR 314.80. A pipeline built to a 7-day rule for
postmarketing drugs will submit early, which is harmless, but the requirements document is wrong and
the audit will find it.

### MedDRA versioning

MedDRA is released twice a year, in March and September. Version 29.1, released in September 2026,
contains 91,499 Lowest Level Terms in total, of which 82,124 are current, and 27,504 Preferred
Terms, organised into 27 System Organ Classes, 337 High Level Group Terms, and 1,739 High Level
Terms. The counts matter for capacity planning: a dictionary-based coder that indexes all LLTs
indexes a vocabulary roughly three times the size of the PT set.

Version changes are not cosmetic. A complex change release can add several hundred LLTs, promote
terms, and make others non-current. A coding that points to a non-current LLT is not wrong in
content but is wrong in status, and an inspection will ask why the case was coded to a term that
was withdrawn. The pipeline should record the MedDRA version on every coded event, which E2B
requires at E.i.2.1a, and should run a regression remapping when the version changes.

### What the standards do not cover

No standard defines the accuracy required of an NLP component. No standard defines how to validate
an autocoder, how to measure duplicate-detection performance, or how to set thresholds. The
regulations require the outcome (a complete, accurate, timely case) and leave the method to the
sponsor, which is why the validation evidence is the sponsor's responsibility and why an inspection
asks for it rather than for a certificate.

The closest thing to a method standard is the general expectation that computerised systems be
validated, that electronic records be attributable and contemporaneous, and that changes be
controlled. Those expectations are met by a validation plan, a traceability matrix, test evidence,
and a change-control procedure, not by a model card.

## 9. Industry practice and case evidence

### The systems

The safety database market is mature and consolidated. Oracle Argus, ArisGlobal LifeSphere, Veeva
Vault Safety, and IQVIA's safety platform are the systems most large sponsors use; regional
providers serve smaller companies. These systems hold the case, manage the workflow, and generate
the E2B message. The NLP layer sits in front of them: it reads the incoming document, proposes the
fields, and hands a draft to a human.

The AI vendors in this space sell intake and triage rather than autonomous submission. That is a
regulatory position as much as a technical one. The marketing authorisation holder remains
responsible for the case, and a submission generated without human review would require the sponsor
to accept that a model's output is a controlled record. As of 2026, the public position of the major
vendors and of the sponsors that use them is that a qualified person reviews every case that
reaches a regulatory decision, and that automation is applied to extraction, coding proposals,
duplicate candidates, and prioritisation.

### What is documented

The peer-reviewed literature on AI in pharmacovigilance is thinner than the vendor material. The
most cited paper is Schmider et al. (Clinical Pharmacology and Therapeutics, 2019), which describes
the use of AI in adverse event processing and frames the validation questions. Danysz et al.
(Drug Safety, 2023) is a practical guide to industrial applications that discusses data quality,
validation, and the limits of automation. Both are reviews and position papers rather than
controlled evaluations, and neither reports a prospective trial of a production system against a
human baseline.

The regulatory guidance is more specific than the literature. The EMA reflection paper on AI across
the medicinal product lifecycle (2024) sets out a risk-based expectation: the higher the impact of
the output on a regulatory decision, the stronger the evidence required. The FDA draft guidance on
AI in drug development (January 2025) makes a similar point for submissions. Neither sets an
accuracy threshold, and both assume human oversight for consequential outputs.

### The gap between vendor claims and evidence

Vendor material typically reports extraction accuracy on a curated test set, coding agreement with
human coders, and a percentage reduction in handling time. Each is a real measure and each is
measured under conditions that differ from production. Three gaps recur.

- **The test set is clean.** Real intake documents contain forwarded email chains, scanned faxes,
  mixed languages, and narratives that describe three cases in one message. A test set drawn from
  the safety database after intake has already removed most of that noise.
- **The baseline is a human coder at a bench.** The relevant comparison is the same team with and
  without the tool, including the time spent correcting the tool's output.
- **The endpoint is field accuracy, not case timeliness.** A tool that improves field accuracy by
  10 points and adds a day of queue latency has made the compliance position worse.

### How a mature organisation deploys it

The pattern that appears in public descriptions of large PV operations has four properties.

First, **automation is applied per field, not per case.** Extraction proposes values; a rules engine
computes seriousness and expectedness; a human confirms the medical judgement. Second, **the clock
is monitored independently of the pipeline**, with a daily report of cases by days remaining that
does not depend on the model. Third, **duplicate detection is a monitored process**, with a monthly
review of the link rate and a sample audit of merged cases. Fourth, **the validation file is
maintained as a living document**, with the model version, the test set, the acceptance criteria, and
the change history.

The inference, clearly labelled as inference, is that the organisations that do this well treat the
model as a component with a specification and a validation record, not as a product that is either
trusted or not. That is the same posture as any other computerised system in a regulated
environment, and it is the posture that survives an inspection.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong day zero | Recorded date equals intake date | Metadata not parsed; intake date used as default | Late or missed submission | Extract earliest timestamp from source; audit monthly |
| Missed valid case | Follow-up queue grows without resolution | Minimum criteria treated as a score | Under-reporting | Binary gate; 24-hour follow-up SLA |
| Indication coded as event | Signal counts rising across products | Co-occurrence treated as causation | Contaminated safety database | Role classifier; indication-as-event audit |
| Wrong MedDRA PT | Coder override rate rising | Retrieval without context constraints | Distorted signal detection | Ranked candidates; coder confirmation |
| Severity confused with seriousness | "Severe" in the narrative drives the flag | Token-level learning | Non-serious cases expedited, serious cases missed | Six explicit criteria with evidence spans |
| Hospitalisation inferred from context | ED visit flagged as admission | Pattern matching without semantics | Over-reporting, alert fatigue | Quote-based evidence for each criterion |
| Expectedness against the wrong document | CCDS version field empty | No version control on the reference | Misclassified SUSAR | Version the reference; block on missing version |
| Missed duplicate | Duplicate link rate near zero | Blocking fields too narrow | Fragmented cases, wrong clock | Monitor link rate; sample audit |
| Over-merging | Two patients linked on initials and drug | Score threshold too low | Data loss, wrong safety profile | Blocking on identifiers; review band |
| Translation normalises hedging | "Could be" rendered as "is" | General-purpose MT | Certainty overstated | Preserve original; bilingual review |
| Clock blind spot in the UI | No days-remaining field | Workflow built around case status | Late submissions discovered after the fact | Deadline as a first-class field and queue sort |
| Version drift in MedDRA | Non-current LLTs in new cases | No remapping after release | Inspection finding | Version stamp; regression remap |
| Audit trail gaps | Overrides without reason codes | UI does not require a reason | Unattributable decisions | Reason code required on override |
| Automation bias in review | High acceptance with low edit time | Reviewer trusts the draft | Errors pass through | Sample blind re-review; time-on-task monitoring |

The pattern in this table is that the failures with the largest regulatory consequence are not the
model failures. They are the metadata, versioning, and workflow failures around the model. A perfect
extractor with a wrong day zero is a late submission. A perfect coder with an unversioned reference
document is a misclassified SUSAR.

The red flags to watch in a dashboard are therefore operational: the duplicate link rate, the
override rate by field, the share of cases whose recorded day zero equals intake, the age
distribution of open cases, and the number of cases submitted in the last 24 hours of their window.
Each of those is a symptom of a process problem that a model will not fix.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The source document contains a reliable receipt timestamp | Day zero depends on it | Forwarded chains, scanned faxes, verbal reports | Compare recorded and extracted dates on a sample |
| The reporter's description is accurate | Extraction is grounded in it | Misremembered dates, second-hand reports | Follow-up; source comparison |
| The narrative distinguishes indication from event | Coding depends on the distinction | Terse narratives, polypharmacy | Indication-as-event audit |
| The reference safety document is current | Expectedness depends on it | Version drift, local label differences | Version field required; periodic reconciliation |
| MedDRA version is recorded and current | Coding status depends on it | Release cycles, legacy cases | Version stamp; remap regression |
| Duplicate scoring generalises across products | Threshold is fixed | Rare events, common initials, paediatric cases | Sample audit by product family |
| Human review catches model errors | The control depends on it | Volume, fatigue, automation bias | Blind re-review sample |
| The extraction test set resembles production | Performance claims depend on it | Curated test sets, language mix | Prospective shadow-mode evaluation |
| Calendar-day arithmetic is correct | Deadlines depend on it | Time zones, local holidays, submission gateway outages | Independent clock report |
| The submission gateway confirms receipt | A submission is not a delivery | Gateway rejection, format validation failure | Acknowledgement reconciliation |

The largest limit is the counterfactual for automation benefit. A vendor's time saving is measured
against a human doing the same task manually, usually in a study where the human knows the output
will be checked. In production the human still checks, so the saving is the difference between
doing the task and correcting a draft, which is smaller and can be negative for complex cases. The
only credible measurement is a prospective comparison in the sponsor's own workflow, with the
correction time included.

The second limit is that the safety database is not the patient population. Under-reporting is
structural, and a better intake pipeline increases the number of reports without changing the
underlying rate. A rise in case counts after deployment is expected and is not evidence of a new
safety signal, which is a common misreading in the first quarter after go-live.

## 12. Exercises

### Exercise 1. Clock arithmetic

A distributor receives a report on Friday 3 April 2026 and forwards it to the marketing
authorisation holder on Tuesday 7 April. The holder's safety mailbox timestamps it 7 April at
09:12. The case is serious and unexpected, and it is submitted on 20 April.

Compute day zero, the deadline under the 15-day rule, the days used, and the days remaining.
Then recompute under the assumption that the distributor is a contractor whose receipt starts the
clock.

**Deliverable:** A calculation sheet with dates, the rule applied, and the result.

**Acceptance criteria:** Day zero is justified by the regulation's definition; calendar days are
used; the result is reported as a signed number of days; the weekend is not excluded.

**Time:** 30 minutes.

### Exercise 2. Validity triage

Take ten short synthetic reports, five of which are missing at least one minimum criterion. For
each, decide valid, incomplete, or invalid, and state the missing element and the follow-up action.

**Deliverable:** A triage table with ten rows and a one-line justification per row.

**Acceptance criteria:** Every decision names the criterion at issue; no case is rejected for
reporter type alone; the follow-up action is specific enough to send.

**Time:** 45 minutes.

### Exercise 3. Design a duplicate-detection strategy

Design blocking, scoring, and threshold bands for a portfolio with three products, two of which
have common adverse events.

**Deliverable:** A specification with blocking keys, scoring features and weights, thresholds, and a
monitoring plan.

**Acceptance criteria:** The blocking step is justified by the expected candidate-set size; rare
agreements are weighted above common ones; the review band has a service-level target; the plan
includes a false-merge audit.

**Time:** 90 minutes.

### Exercise 4. Critique an autocoding output

You are given a verbatim term, five candidate LLTs with scores, and the surrounding sentence.
Identify the correct PT, explain why the top candidate may be wrong, and state what context the
coder should use.

**Deliverable:** A 300-word critique with the chosen PT, the reasoning, and the failure mode the
ranking exhibits.

**Acceptance criteria:** The critique distinguishes specificity from correctness; it names the
context feature that resolves the ambiguity; it does not invent a MedDRA code.

**Time:** 40 minutes.

### Exercise 5. Seriousness with evidence spans

For five synthetic narratives, assign the six seriousness criteria and quote the span that supports
each true flag.

**Deliverable:** A table with narrative, criterion, flag, and quoted evidence.

**Acceptance criteria:** No flag is assigned without a quote; "severe" is not used as a criterion;
a planned admission is correctly distinguished from an event-caused admission.

**Time:** 60 minutes.

### Exercise 6. Expectedness version control

A case is assessed against CCDS version 6.0 when version 7.2 is current. Identify what changes, and
design the control that prevents the error.

**Deliverable:** A short analysis plus a control description with the field, the check, and the
escalation path.

**Acceptance criteria:** The analysis distinguishes expected from unexpected with the specific
terms; the control blocks submission when the reference version is missing or stale; the escalation
path names a role, not a team.

**Time:** 45 minutes.

### Exercise 7. Operating point

A model flags 500 of 20,000 documents. Of the flags, 180 are true cases; the gold standard contains
200 cases. Compute sensitivity, PPV, and the number of documents reviewed per true case. Then
compute what happens if the threshold is raised so that 300 documents are flagged and 150 are true.

**Deliverable:** The calculations with units, and a recommendation with the reasoning.

**Acceptance criteria:** Sensitivity and PPV are computed against the correct denominators; the
review burden is expressed per true case; the recommendation accounts for the cost of a missed
serious case.

**Time:** 40 minutes.

### Exercise 8. Validation file

Assemble the outline of a validation file for an autocoding component, as an inspector would expect
to see it.

**Deliverable:** A table of contents with a one-line purpose for each section, and the acceptance
criteria for the component.

**Acceptance criteria:** The file covers intended use, specification, test data, acceptance
criteria, results, deviations, change control, and periodic review; the acceptance criteria are
numeric with units; the file names the responsible role for each section.

**Time:** 90 minutes.

## 13. Self-check answers

**Exercise 1.** Day zero is 3 April 2026, because the distributor is receiving the information on
behalf of the holder and the regulation starts the clock at first receipt by any personnel or
contractor. Deadline = 3 April + 15 = 18 April 2026, a Saturday. Submission on 20 April uses 17
days, so the case is 2 days late. If the mailbox date of 7 April were used, the deadline would be
22 April and the case would appear 2 days early. The four-day difference is entirely a metadata
question, which is why the first diagnostic in Section 6 exists.

**Exercise 2.** A correct table assigns valid only when all four criteria are present, incomplete
when a missing element is obtainable through a named reporter, and invalid when the element is
missing and the reporter is not contactable. The common errors are rejecting a consumer report
(reporter type is not a criterion) and accepting a report with a product but no patient
identification. A follow-up action should name the element and the channel, for example "call the
reporter to obtain the patient's date of birth and initials".

**Exercise 3.** A strong design blocks on product plus PT, or on product plus a normalised event
token, and scores on initials, age within one year, sex, reporter, and receipt date proximity.
Rare-event agreement receives a higher weight than agreement on a common event such as nausea. The
link band should be narrow, because a false merge loses a case; the review band should be wide,
because a review costs minutes. Monitoring includes the monthly link rate, the false-merge rate
from a sample audit, and the time from candidate generation to review decision.

**Exercise 4.** A strong critique notices that the top candidate is often the most frequent term
rather than the most specific one, that the surrounding sentence may contain the indication, and
that the correct PT may be a more specific term that ranks second. It names the context feature
that resolves the ambiguity, such as the anatomical site or the temporal relationship to the drug,
and it declines to invent a code.

**Exercise 5.** A strong table quotes the span for each flag. "Admitted to St Anne's for
investigation" supports hospitalisation; "was sent home from the emergency department" does not.
"Severe" alone supports no criterion. "Admitted for a planned cholecystectomy, also noted to have
jaundice" does not support hospitalisation caused by the event, and the reasoning should say so.

**Exercise 6.** A strong analysis identifies the specific terms that changed between versions and
the cases whose expectedness flips. The control requires the reference document identifier and
version on every case, blocks submission when the version is missing or older than the current
effective version, and escalates to the named safety physician. A weak answer relies on a periodic
manual review without a blocking check.

**Exercise 7.** At the lower threshold: sensitivity = 180/200 = 0.90; PPV = 180/500 = 0.36; review
burden = 500/180 = 2.78 documents per true case. At the higher threshold: sensitivity = 150/200 =
0.75; PPV = 150/300 = 0.50; review burden = 300/150 = 2.00. Raising the threshold halves the review
burden per true case and misses 25 more cases out of 200. In this domain the first configuration is
usually correct despite the lower PPV, because a missed serious case is a regulatory event and a
review is minutes.

**Exercise 8.** A strong outline covers intended use and scope, the component specification, the
data sources and sampling, the reference standard and its construction, the acceptance criteria,
the test results with confidence intervals, deviations and their resolution, the change-control
procedure, the periodic review cycle, and the roles responsible. The acceptance criteria are
numeric and carry units, for example "sensitivity at the PT level of at least 0.95 with a lower
95 percent confidence bound of 0.90 on a test set of 500 events".

## 14. Interview questions

**1. When does the regulatory clock start?**

A strong answer says first receipt by any personnel or contractor, not intake, and gives an example
of a sales representative receiving the information. A weak answer says when the case is entered
into the safety database.

**2. What are the four minimum criteria, and what happens if one is missing?**

A strong answer names patient, reporter, product, and event, and says the case enters follow-up and
is retained as invalid if follow-up fails. A weak answer treats the criteria as a quality score.

**3. Why is sensitivity more important than PPV for intake?**

A strong answer quantifies the asymmetry: a false negative is a missed regulatory deadline and a
possible patient safety event, while a false positive is minutes of review. A weak answer says
sensitivity is always more important, without the domain reason.

**4. How would you detect duplicate cases?**

A strong answer describes blocking, weighted scoring, threshold bands, and a monitoring plan, and
notes that duplicates affect the clock as well as the count. A weak answer proposes exact matching
on sender identifiers.

**5. What is the difference between serious and severe?**

A strong answer says seriousness is a regulatory classification with six criteria and severity is
intensity, and gives a counterexample in each direction. A weak answer conflates them.

**6. How does expectedness work when the label lists a broader term?**

A strong answer says expectedness is a versioned lookup, that greater severity or specificity can
make an event unexpected, and that the judgement is recorded with its reasoning. A weak answer says
if the organ system is listed, the event is expected.

**7. What would you monitor after go-live?**

A strong answer names duplicate link rate, override rate by field, day-zero discrepancy, open-case
age, and submissions in the final 24 hours. A weak answer names model accuracy only.

**8. How would you validate an autocoder for an inspection?**

A strong answer describes a validation plan with a defined reference standard, a test set, numeric
acceptance criteria, results with intervals, change control, and periodic review, and notes that
the sponsor owns the evidence. A weak answer says the vendor provides a model card.

**9. What is the failure mode you would worry about most in a new deployment?**

A strong answer identifies a metadata or versioning failure, such as a wrong day zero or a stale
reference document, and explains why it is worse than a model error. A weak answer names
hallucination without connecting it to a regulatory consequence.

## 15. Key terms

- **Adverse drug reaction (ADR)** — an adverse event with at least a reasonable possibility of causation.
- **Adverse event (AE)** — any untoward medical occurrence in a patient administered a product.
- **Causality** — structured judgement of the drug-event relationship; WHO-UMC categories.
- **Company Core Data Sheet (CCDS)** — the company reference document for expectedness.
- **Day zero** — date of first receipt by any company personnel or contractor.
- **Development Safety Update Report (DSUR)** — annual report for an investigational drug.
- **Duplicate** — two or more reports of the same case.
- **E2B(R3)** — ICH standard for electronic ICSR transmission.
- **EudraVigilance** — the EU adverse reaction database.
- **Expectedness** — whether the reaction is listed in the applicable reference safety information.
- **FAERS** — the FDA Adverse Event Reporting System.
- **Follow-up** — attempt to obtain missing case information.
- **ICSR** — Individual Case Safety Report.
- **Life-threatening** — immediate risk of death at the time of the event.
- **Lowest Level Term (LLT)** — the most granular MedDRA term.
- **Marketing authorisation holder (MAH)** — the legally responsible company.
- **MedDRA** — the ICH regulatory terminology.
- **Minimum criteria** — patient, reporter, product, event.
- **Nullification** — withdrawal of a submitted case.
- **Periodic Benefit-Risk Evaluation Report (PBRER)** — periodic benefit-risk report.
- **Pharmacovigilance (PV)** — detection, assessment, and prevention of adverse effects.
- **Preferred Term (PT)** — the MedDRA term used for reporting and counting.
- **Qualified Person for Pharmacovigilance (QPPV)** — the EU role responsible for the PV system.
- **Reference Safety Information (RSI)** — the document defining expectedness.
- **Serious adverse event (SAE)** — an event meeting at least one of six seriousness criteria.
- **Severity** — intensity of an event; not the same as seriousness.
- **Signal** — information suggesting a new or changed causal association.
- **Standardised MedDRA Query (SMQ)** — a validated term grouping for a condition.
- **SUSAR** — suspected unexpected serious adverse reaction.
- **System Organ Class (SOC)** — the top MedDRA level; 27 categories.
- **Unexpected** — not listed, or listed at greater severity or specificity.
- **VigiBase** — the WHO global ICSR database.

## 16. Sources

All URLs accessed 27 October 2026.

1. International Council for Harmonisation. "ICH E2B(R3) Electronic Transmission of Individual Case
   Safety Reports (ICSRs) Implementation Guide: Data Elements and Message Specification," version
   5.03, 18 July 2025. https://ich.org/page/e2br3-individual-case-safety-report-icsr-specification-and-related-files
2. International Council for Harmonisation. "ICH E2A: Clinical Safety Data Management: Definitions
   and Standards for Expedited Reporting." 1994.
3. International Council for Harmonisation. "ICH E2D: Post-Approval Safety Data Management:
   Definitions and Standards for Expedited Reporting." 2003.
4. International Council for Harmonisation. "ICH E2C(R2): Periodic Benefit-Risk Evaluation Report."
   2012.
5. US Food and Drug Administration. "21 CFR 314.80: Postmarketing reporting of adverse drug
   experiences." https://www.ecfr.gov/current/title-21/chapter-I/subchapter-D/part-314/subpart-B/section-314.80
6. US Food and Drug Administration. "21 CFR 600.80: Postmarketing reporting of adverse experiences."
   https://www.ecfr.gov/current/title-21/chapter-I/subchapter-F/part-600/subpart-B/section-600.80
7. US Food and Drug Administration. "21 CFR 312.32: IND safety reporting."
   https://www.ecfr.gov/current/title-21/chapter-I/subchapter-D/part-312/subpart-B/section-312.32
8. European Parliament and Council. "Regulation (EU) No 536/2014 on clinical trials on medicinal
   products for human use." Official Journal of the European Union, 27 May 2014.
9. European Medicines Agency. "Guideline on Good Pharmacovigilance Practices (GVP) Module VI:
   Management and Reporting of Adverse Reactions to Medicinal Products." Rev 2, 2017, as amended.
10. European Medicines Agency. "Reflection paper on the use of artificial intelligence in the
    medicinal product lifecycle." 2024.
11. US Food and Drug Administration. "Considerations for the Use of Artificial Intelligence to
    Support Regulatory Decision-Making for Drug and Biological Products." Draft guidance, January
    2025.
12. Medical Dictionary for Regulatory Activities Maintenance and Support Services Organization.
    "What's New: MedDRA Version 29.1." September 2026.
    https://alt.meddra.org/files_acrobat/whatsnew_29_1_English.pdf
13. Uppsala Monitoring Centre. "The use of the WHO-UMC system for standardised case causality
    assessment." https://who-umc.org/media/164200/who-umc-causality-assessment_new-logo.pdf
14. Schmider J, Kumar K, LaForest C, Swankoski B, Naim K, Caubel P. "Innovation in
    pharmacovigilance: use of artificial intelligence in adverse event processing." Clinical
    Pharmacology and Therapeutics. 2019;105(4):954-961.
15. Danysz K, Cicirello S, Mingle E, et al. "Artificial intelligence and machine learning in
    pharmacovigilance: a practical guide for industrial applications." Drug Safety.
    2023;46(6):519-532.
16. US Food and Drug Administration. "21 CFR Part 11: Electronic Records; Electronic Signatures."
    https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`11_ambient-ai-and-documentation.md`](11_ambient-ai-and-documentation.md)
- Next topic: [`13_linguistic-equity-in-clinical-nlp.md`](13_linguistic-equity-in-clinical-nlp.md)
- Task taxonomy for extraction and assertion: [`03_task-taxonomy.md`](03_task-taxonomy.md)
- Classical methods for rules and dictionaries: [`04_classical-methods.md`](04_classical-methods.md)
- Annotation and adjudication: [`06_annotation.md`](06_annotation.md)
- Evaluation design: [`07_evaluation.md`](07_evaluation.md)
- Module survey: [`../../course/09_clinical_nlp.md`](../../course/09_clinical_nlp.md)
- Labs: [`../../labs/`](../../labs/)
