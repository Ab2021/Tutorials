# Clinical Text Sources

> **Module 09 — Clinical NLP**
> Topic 01 of 10
> Estimated study time: 3 hours
> Prerequisites: none
> Version 2026.3

## 1. Why this topic exists

Every clinical NLP project begins with a corpus that someone assembled for a reason other than
your analysis. A discharge summary was written to communicate a hospital course to a downstream
clinician and to support billing. A radiology report was written to answer a specific clinical
question and to protect the radiologist legally. A patient portal message was written by a patient
who is worried, angry, or brief. None of these documents was written to be a data source, and the
structure, vocabulary, and omissions of each reflect its original purpose.

The consequence is that source selection is not a procurement detail. It determines what questions
can be answered, what error rates are achievable, what privacy exposure the project carries, and
how the results generalize. A team that builds a sepsis phenotype from discharge summaries and then
deploys it on emergency department notes has changed the task without changing the model. A team
that counts documents when the decision is made per patient will overstate its sample and
understate its uncertainty. A team that treats a radiology impression and a nursing note as
interchangeable text has discarded the information that makes each useful.

This topic exists because source confusion is the most common structural error in applied clinical
NLP, and it is usually invisible in the final report. The report says "we used clinical notes." It
does not say which notes, from which encounter types, written by which roles, under which template
regime, with which documents excluded, and with what duplication. A reviewer cannot reproduce the
cohort, and a downstream team cannot tell whether the model applies to their data.

The problem is not academic. Health systems routinely hold dozens of note types across inpatient,
ambulatory, emergency, perioperative, and messaging contexts. The same patient may have hundreds of
documents per year. The same clinical fact may appear in a discharge summary, a progress note, a
problem list entry, a medication order, and a billing code, each with different reliability and
different timing. Selecting among these is a design decision with measurable consequences.

Three decisions are blocked until source selection is settled.

The first is cohort definition. A phenotype that requires "documented diagnosis" needs a rule for
which documents count as documentation. A discharge summary authored by the attending after review
of the full admission is different evidence from a triage note authored in four minutes.

The second is evaluation design. Span-level metrics computed on radiology reports do not transfer
to progress notes, because sentence length, section structure, and abbreviation density differ.
Reporting a single F1 across a mixed corpus hides the stratum where the system fails.

The third is privacy and governance. Free-text notes are the highest-risk protected health
information (PHI) source in an enterprise data warehouse. The source inventory determines the
de-identification burden, the minimum necessary review, and the data use agreement scope.

A useful way to frame the topic is that a clinical document is a record of an act of communication,
not a record of facts. The facts are inferences a reader draws from the document. Clinical NLP
automates part of that inference, and the inference is only as good as the document's completeness,
its genre conventions, and the reader's shared context. The rest of this module is about making
that inference explicit, bounded, and auditable.

There is also an economic dimension. Corpus assembly often consumes a third of a project's
calendar time and a larger share of its governance effort. A team that spends three weeks
negotiating access to the wrong note type has spent three weeks not modeling. A short source
inventory at the start, with the clinical sponsor in the room, is the cheapest risk reduction
available. It converts an open-ended data request into a scoped one, and it gives the privacy
office a precise object to review instead of "all notes."

> **Note:** "Clinical notes" is not a data type. It is a family of genres. Treating the family as
> one corpus is the source of most cross-site generalization failures discussed later in this
> module.

## 2. Learning outcomes

By the end of this topic you can:

1. **Inventory** the clinical text sources in a health system and **classify** each by author role,
   encounter context, template regime, and expected clinical content.
2. **Explain** how the purpose of a document shapes its structure, vocabulary, and omissions, and
   **predict** which extraction tasks each source supports.
3. **Trace** a document's provenance envelope: identity, author role, creation and storage time,
   encounter linkage, version and addendum chain.
4. **Segment** a note into sections and **identify** where section boundaries change the meaning of
   a mention.
5. **Choose** a unit of analysis (span, document, encounter, patient) and **justify** the choice
   against the decision the project supports.
6. **Detect** source-mix confounding in an evaluation and **stratify** metrics by note type.
7. **Write** a corpus description that a reviewer could use to reproduce the cohort.
8. **Critique** a published clinical NLP study whose source selection does not support its claimed
   deployment context.

Outcomes 1 through 4 are descriptive and computational; 5 through 8 are design and critique.

## 3. The core mental model

A clinical document carries two layers: the clinical narrative and the provenance envelope. Most
NLP pipelines model the first and ignore the second. The second is what makes the first
interpretable.

```text
                    PROVENANCE ENVELOPE
  +-------------------------------------------------------------+
  | document id | patient id | encounter id | note type | author |
  | role        | site       | service line | charttime | storetime |
  | version     | addendum-of| template id  | signed    | amended |
  +-------------------------------------------------------------+
                              |
                              v
  +-------------------------------------------------------------+
  | DOCUMENT                                                    |
  |   header / metadata block                                   |
  |   -------------------------------------------------------   |
  |   SECTION: Chief complaint                                  |
  |   SECTION: History of present illness                       |
  |   SECTION: Past medical history                             |
  |   SECTION: Medications                                      |
  |   SECTION: Assessment and plan                              |
  |   SECTION: Labs / imaging                                   |
  +-------------------------------------------------------------+
                              |
                              v
  +-------------------------------------------------------------+
  | SENTENCE  ->  TOKEN  ->  MENTION  ->  ASSERTION             |
  |                              |                              |
  |                              +--> concept identifier        |
  |                              +--> temporality               |
  |                              +--> attribution               |
  +-------------------------------------------------------------+
                              |
                              v
              ENCOUNTER-LEVEL FACT -> PATIENT-LEVEL FACT
```

The vertical flow is the module's spine. A string in a document is a **mention**. A mention plus
its local context yields an **assertion** (present, negated, uncertain, hypothetical, historical,
family). An assertion attached to the right patient at the right time yields a **fact**. A fact
sustained across encounters, or confirmed by a diagnostic criterion, yields a **diagnosis**. Each
arrow can fail independently, and a failure at one arrow cannot be repaired downstream by a better
model at another.

The envelope matters because it changes the prior probability that a mention is a fact. A mention
of "pneumonia" in the "Assessment and plan" section of a discharge summary, written by the attending
physician, affirmed in the present tense, has a high probability of being a treated diagnosis. The
same string in a "Review of systems" section is a symptom report. The same string in a radiology
"Indication" field is the reason for the exam, not a finding. The same string in a problem list may
be stale for years.

Source selection therefore sets three things at once:

- **The base rate of the target.** Radiology reports contain more incidental findings; ED notes
  contain more undifferentiated symptoms; discharge summaries contain more resolved conditions.
- **The ceiling on recall.** If the fact is not documented in the selected source, no model can
  extract it. A sepsis phenotype built only on discharge summaries will miss patients who died in
  the ED before admission.
- **The error structure.** Abbreviation ambiguity, template boilerplate, and copy-forward are not
  uniformly distributed across note types. Each source has a characteristic failure profile.

A second model is useful for corpus assembly: the **document life cycle**.

```text
draft --> signed --> addendum --> corrected --> amended --> (copy-forward into next note)
```

Only the signed version is the legal record. The draft may be overwritten. The addendum may
contradict the original. The corrected version may replace the original in the EHR view but both
may exist in the warehouse extract. The next note may contain the previous note's text verbatim.
If the extract has no version field, the pipeline may read a draft, double-count an addendum, or
treat a correction as a second clinical event.

A third model is the **source-to-task fit matrix**. Not every source supports every task, and the
mismatch is often the real reason a model underperforms.

| Task | Best-fit sources | Weak-fit sources | Why |
|---|---|---|---|
| Symptom onset | HPI, ED note, patient message | Discharge summary | Summaries compress and reinterpret onset |
| Confirmed diagnosis | Discharge summary, problem list | Radiology report | Reports describe findings, not final diagnoses |
| Treatment response | Progress notes, oncology notes | Admission note | Admission note precedes treatment |
| Social determinants | Social history, care plan, messages | Operative note | Not recorded in procedural genres |
| Device complications | Operative note, progress note | Radiology report | Device status described incidentally |
| Medication adherence | Medication history, messages | Billing codes | Codes record prescribing, not taking |
| Severity | ICU notes, vitals flowsheet text | Outpatient note | Severity language is setting-specific |

The practical implication is that corpus design should be task-first. Write the phenotype or
extraction specification, list the evidence that would satisfy it, then ask which documents contain
that evidence. Assembling a corpus first and choosing a task second is how projects end up with a
well-curated dataset that cannot answer the question.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Clinical note | A narrative document authored during care, stored in the EHR | Used for any text field, including orders and structured forms | Document |
| Note type | The genre label assigned by the EHR, such as discharge summary or progress note | Treated as interchangeable across sites | Local code; often mapped to LOINC |
| Document type code | A standard code identifying the document genre | Assumed to be universally populated | LOINC document ontology (for example 18842-5 discharge summary) |
| Encounter | A single interaction or stay with a defined start and end | Confused with admission or visit | Encounter identifier |
| Admission | An inpatient stay, possibly spanning transfers | Used as synonym for encounter | `hadm_id` in MIMIC |
| Provenance | The origin and history of a document: author, time, system, version | Reduced to "source system" | Envelope fields |
| Charttime | The clinical time to which a note applies | Confused with storage time | Timestamp |
| Storetime | The time the note was saved or signed in the EHR | Used as a proxy for clinical time | Timestamp |
| Addendum | A later addition that supplements a signed note without replacing it | Merged silently into the original | Document version |
| Amendment | A correction that changes the signed record | Treated as a new document | Document version |
| Section | A labelled region of a note with a clinical function | Detected by font or layout only | Span within document |
| Section header | The label that names a section | Matched without synonym handling | String or ontology term |
| Template | The pre-populated structure a clinician fills in | Ignored as "not clinical text" | Template identifier or inferred |
| Boilerplate | Text repeated across notes by design, not by copying | Counted as clinical content | Span |
| Mention | A text span referring to a clinical concept | Called an entity or a finding | Character offsets |
| Assertion | The contextual status of a mention: present, negated, uncertain, hypothetical, historical, family | Collapsed into "positive" | Categorical label |
| Attribution | The person or entity the assertion applies to | Assumed always to be the patient | Categorical label |
| Temporality | The relationship of the event to the document creation time | Treated as binary past or present | Categorical or interval |
| Concept identifier | A code from a controlled vocabulary that a mention maps to | Assumed unique per mention | SNOMED CT, UMLS CUI, RxNorm, LOINC |
| Phenotype | A computable definition of a patient-level condition | Equated with a single code | Algorithm plus validation |
| Corpus | The set of documents selected for a task | Described only by size | Documents plus inclusion criteria |
| Sampling frame | The population from which documents were drawn | Left unstated | Encounters or patients |
| Note bloat | Note length inflation from templates and copying | Treated as a quality issue only | Tokens or characters per note |
| De-identification | Removal or transformation of identifiers | Assumed complete after redaction | Per-identifier recall |
| Data use agreement | A contract governing access to restricted data | Treated as a licence to redistribute | Legal instrument |
| Credentialed access | Access gated by identity verification and training | Confused with open access | PhysioNet credentialing |
| Token | The unit produced by a tokenizer | Assumed to equal a word | Tokenizer-specific |
| Sentence boundary | The end of a syntactic sentence | Assumed to be a period | Character offset |
| Abbreviation | A shortened form with local meaning | Expanded by a general dictionary | Local lexicon |
| OCR noise | Character errors from scanned documents | Ignored in evaluation | Character error rate |

Two definitions deserve emphasis because they recur in every later topic.

**Mention versus assertion.** A mention is a span. "No acute pneumonia" contains the mention
"acute pneumonia" and the assertion "absent." A pipeline that reports the mention without the
assertion has reported a string, not a finding. Chapter 04 covers the mechanics of assertion, but
the vocabulary distinction belongs here because source selection determines how much assertion
work is required. Discharge summaries contain long differentials and resolved problems; radiology
reports contain indication text and recommendation language; messages contain patient-reported
negation that may be lay rather than clinical ("I don't think it's pneumonia").

**Diagnosis versus documentation of diagnosis.** A diagnosis is a clinical determination, often
supported by criteria, imaging, labs, and response to treatment. Documentation of a diagnosis is a
string in a note. A problem list entry is a third thing: an assertion by someone, possibly years
ago, possibly never reconciled. The module uses "diagnosis" only for the clinical determination
and "documented diagnosis" for the text evidence. Every metric in this module states which of the
three it measures.

## 5. The mechanics, step by step

This section is the procedure for turning a source inventory into a defensible corpus. It is
written as a sequence a data engineer and a clinical lead can execute together. Each step states
its input, output, and rationale, and each step has at least one branch point.

### Step 1. Fix the decision and the unit of analysis

**Input:** A stated business or research decision, for example "identify admitted patients with
documented heart failure for a readmission model."

**Output:** A written unit of analysis: patient, encounter, document, or span. A written prediction
or measurement time.

**Rationale:** The unit determines which documents are eligible and how duplicates aggregate.
A patient-level phenotype needs all documents up to the index time. A document-level classifier
needs one row per document and a rule for repeated documents. Choosing the unit after assembling
the corpus almost guarantees the wrong denominator.

**Branch point:** If the decision is about a patient over time, the unit is the patient and the
documents are evidence. If the decision is about a single note, such as triage of a message, the
unit is the document and the patient is context, not the analysis row.

### Step 2. Inventory candidate sources

**Input:** EHR report catalogue, interface inventory, and interviews with clinical and health
information management staff.

**Output:** A table with one row per source: note type, author role, setting, typical length,
template family, retention, and known text quality issues.

**Rationale:** The catalogue is usually incomplete. Many sites have note types created by local
configuration that do not appear in vendor documentation. Interviews recover them.

**Branch point:** Sources fall into four availability classes.

| Class | Definition | Consequence |
|---|---|---|
| Available and complete | Extract includes all documents of the type | Eligible without caveat |
| Available but truncated | Extract caps characters or excludes addenda | Recall ceiling must be stated |
| Available only in the EHR UI | No bulk export path | Manual abstraction or screen scraping, with governance review |
| Not available | Outside the enterprise or under separate control | Out of scope; state as a limitation |

### Step 3. Establish document identity and version

**Input:** Raw document records with identifiers and timestamps.

**Output:** One canonical record per document version, with a stable document identifier and a
parent identifier where applicable.

**Rationale:** Warehouse extracts frequently contain multiple rows per document: drafts, signed
versions, addenda, and corrections. Without version resolution, term frequency counts, duplicate
detection, and temporal ordering are all wrong.

**Branch point:** Identity resolution has five outcomes.

| Outcome | Test | Action |
|---|---|---|
| New document | No parent identifier and no matching hash | Keep as independent |
| Duplicate extract | Same identifier and same content hash | Deduplicate, keep earliest extract time |
| Revised version | Same identifier, different hash, later storetime | Keep latest signed version; retain prior in audit table |
| Addendum | Parent identifier present and addendum flag set | Keep separately; concatenate only for document-level tasks with a flag |
| Unresolved | Missing identifier or conflicting metadata | Quarantine; report count |

### Step 4. Capture the provenance envelope

**Input:** Document metadata fields.

**Output:** A provenance table keyed by document identifier with author role, service, site,
charttime, storetime, signed status, and template identifier if present.

**Rationale:** Provenance is the cheapest source of priors. Author role predicts vocabulary and
abbreviation use. Service predicts section structure. Site predicts template and coding practice.
Charttime anchors temporality. Storetime detects late entries and batch signing.

**Branch point:** If author role is missing, infer it from note type only when the mapping is
one-to-many and stable at that site; otherwise mark unknown and stratify.

### Step 5. Resolve patient and encounter linkage

**Input:** Document identifiers and crosswalk tables.

**Output:** Documents linked to a patient and, where applicable, an encounter.

**Rationale:** Many documents are encounter-linked but some, such as messages and telephone notes,
are patient-linked only. A pipeline that inner-joins on encounter silently drops the patient-linked
sources, which are often the ones containing social context.

**Branch point:** Documents that cannot be linked are excluded and counted. The exclusion count is
a required line in the corpus description.

### Step 6. Segment into sections

**Input:** Document text and a section header lexicon.

**Output:** A list of sections with character offsets and a normalized section label.

**Rationale:** Section context changes assertion and attribution. "History of" in a past medical
history section is not the same cue as "history of" in an assessment. Section labels are also the
cheapest way to restrict extraction to the region where the evidence lives.

**Branch point:** Section detection has three regimes.

| Regime | Signal | Method | Failure |
|---|---|---|---|
| Explicit headers | Line-initial label followed by colon or newline | Lexicon plus regex | Local synonym not in lexicon |
| Implicit structure | Paragraph breaks, headings in bold or caps | Layout or casing heuristics | Flattened text loses formatting |
| Free narrative | No structure | Fixed windows or sentence clustering | Boundaries are arbitrary |

### Step 7. Detect template and boilerplate

**Input:** Sectioned text and a corpus of documents from the same template family.

**Output:** Spans flagged as template text, plus a template identifier where inferable.

**Rationale:** Template text is not clinical observation. It inflates term counts, creates spurious
negatives ("no chest pain" pre-populated in every review of systems), and teaches models the
template rather than the patient.

**Branch point:** Boilerplate is either removable or merely flaggable. Remove when the text carries
no patient-specific content. Flag when the template contains slots that may be filled, because
removal would delete the filled value too.

### Step 8. Segment sentences with clinical abbreviations protected

**Input:** Section text.

**Output:** Sentence spans.

**Rationale:** Periods appear in abbreviations ("Dr.", "b.i.d.", "5.4"), decimals, and list
markers. Naive splitting truncates sentences and breaks negation scope.

**Branch point:** If sentence splitting confidence is low, prefer a larger unit (paragraph) and let
the assertion module use clause-level scope. A wrong sentence boundary is worse than a coarse one
for negation.

### Step 9. Tokenize with clinical rules

**Input:** Sentence text.

**Output:** Tokens with character offsets.

**Rationale:** Clinical text contains hyphens, slashes, Greek letters, unit expressions, and
abbreviations that general tokenizers split badly. "BP 120/80" should not become three tokens if
the extraction targets blood pressure. Offsets are required to map predictions back to text.

**Branch point:** Tokenization scheme must match the model. A transformer's subword tokenizer and a
regex tokenizer produce different spans; evaluation must use character offsets, not token indices.

### Step 10. Normalize surface forms

**Input:** Tokens and candidate phrases.

**Output:** Normalized strings plus a record of the transformation.

**Rationale:** Case folding, abbreviation expansion, spelling correction, and Unicode normalization
reduce lexical variation. Each transformation also destroys information. Lowercasing loses
"RA" versus "ra" distinctions; expansion loses the original for audit.

**Branch point:** Normalize for matching, but always retain the original span and offsets.

### Step 11. Locate candidate mentions

**Input:** Normalized sentences.

**Output:** Candidate spans with a provisional type.

**Rationale:** Candidate generation is high recall and low precision by design. The assertion and
concept stages filter.

**Branch point:** Dictionary matching must handle overlap. "Congestive heart failure" and "heart
failure" match the same region; the longest match is usually correct, but not when the longer
string crosses a coordination boundary.

### Step 12. Assign assertion and attribution

**Input:** Candidate spans plus sentence and section context.

**Output:** Assertion label per span.

**Rationale:** This is where a mention becomes a finding or does not. The assertion stage consumes
section context, negation cues, uncertainty cues, hypothetical markers, and attribution cues.

**Branch point:** A span may carry multiple labels, for example historical and family. The output
schema must permit a set, not a single enum.

### Step 13. Assign temporality relative to document creation time

**Input:** Assertion-labelled spans and document charttime.

**Output:** Temporal interval or category per span.

**Rationale:** "History of MI" in a note from 2026 may refer to 2019. Copy-forward makes the
document time a poor proxy for the event time. Explicit temporal expressions should be parsed when
present; otherwise the assertion category is the fallback.

**Branch point:** If the document has no reliable charttime, temporality cannot be computed. Mark
unknown and exclude from temporal analyses rather than imputing.

### Step 14. Map to a concept, then to a patient-level fact

**Input:** Assertion-labelled spans.

**Output:** Concept identifiers and, separately, patient-level facts with evidence links.

**Rationale:** Concept mapping and patient-level inference are different operations. A mention may
map cleanly to a SNOMED CT concept and still not constitute a patient-level diagnosis. Keeping the
layers separate makes errors attributable.

**Branch point:** Patient-level fact generation has four policies.

| Policy | Rule | Use when | Cost |
|---|---|---|---|
| Any affirmative mention | One affirmed mention anywhere | High sensitivity screening | Low precision |
| Section-restricted | Affirmed mention in assessment or problem list | Diagnosis documentation | Misses narrative-only evidence |
| Repeated evidence | Two or more affirmed mentions across encounters | Chronic condition phenotype | Misses single-encounter patients |
| Criterion-based | Mention plus lab, medication, or procedure evidence | High-stakes cohort | Complex, requires structured data |

### Step 15. Measure coverage and representativeness

**Input:** Final corpus and the source population.

**Output:** Coverage table by note type, site, service, author role, and patient subgroup.

**Rationale:** A corpus can be large and unrepresentative. If one service writes long notes and
another writes short ones, a token-weighted sample over-represents the first. If one site
contributes most documents, site-specific templates dominate.

**Branch point:** If coverage is skewed, either reweight, stratify, or restrict the claim. Do not
report a single aggregate metric without the stratum table.

### Edge cases that decide the quality of the corpus

**Negation scope across section boundaries.** A negation cue in one section should not negate a
mention in the next. Sentence-level and section-level scoping must agree. When a note has no
sentence punctuation, section boundaries become the only reliable scope delimiter.

**Hypotheticals in plan text.** "If she develops shortness of breath, consider diuresis" contains
a mention of shortness of breath that has not occurred. Plan sections are dense with conditionals,
and a bag-of-words model will read them as findings.

**Family history.** "Mother with breast cancer" and "patient with breast cancer" differ by one
word and by the entire patient-level conclusion. Family history sections are often templated,
which makes the cue reliable when the section is detected and unreliable when it is not.

**Copy-forward duplicates.** The same sentence can appear in ten consecutive notes. Counting
mentions counts the sentence ten times. Counting documents counts ten documents. Neither equals
ten clinical events.

**Template boilerplate.** Pre-populated normal findings are the most common source of false
negatives in symptom extraction. The template says "denies chest pain" before the clinician has
asked.

**Abbreviation ambiguity.** "MS" can be mitral stenosis, multiple sclerosis, morphine sulfate, or
mental status. "PT" can be prothrombin time, physical therapy, or patient. Local abbreviation
lexicons are necessary and insufficient; context disambiguation is required.

**Section boundary errors.** A header misdetected as body text merges the medication list into the
assessment. A header detected inside a sentence splits a finding. Both change assertion.

**Unicode and OCR noise.** Smart quotes, non-breaking spaces, ligatures, and OCR substitutions
("1" for "l", "0" for "O") break exact matching. Normalize Unicode to a known form, retain the raw
text, and never evaluate only on normalized text if the deployed pipeline sees raw text.

**Late entries and back-dating.** A note charted at 23:50 but stored at 09:15 the next day may
describe the previous day's events. Using storetime as the prediction cutoff leaks information
from after the cutoff.

### Step 16. Write the corpus description before modeling

**Input:** The resolved corpus, the exclusion log, and the coverage table.

**Output:** A corpus description of 300 to 600 words with a companion table.

**Rationale:** The description is the contract between the analysis and its reviewers. Writing it
before modeling exposes gaps while they are still cheap to fix. A team that cannot state the
document-to-patient ratio, the addendum policy, and the exclusion counts does not yet know what it
is modeling.

**Branch point:** If the description cannot be written without guessing, return to Step 2. The
missing field identifies which inventory question was never answered.

The description has seven required elements:

| Element | Content | Why a reviewer needs it |
|---|---|---|
| Source systems | EHR, scanned records, messages, external feeds | Transportability |
| Note types | Code, label, count, mean length | Genre balance |
| Date range | First and last charttime | Temporal drift |
| Inclusion | Rules with counts at each filter | Reproducibility |
| Exclusion | Rules with counts and reasons | Hidden selection |
| Version policy | Draft, signed, addendum, amendment handling | Legal record alignment |
| Privacy | De-identification method and audit result | Governance |

### Decision table: from mention to patient-level fact

The last three steps of the procedure converge on one decision that teams often make implicitly.
Making it explicit prevents the most common category error in the module: reporting a mention count
as a patient count.

| Evidence pattern | Assertion required | Patient-level conclusion | Confidence | Typical use |
|---|---|---|---|---|
| Affirmed mention in assessment, one encounter | Present | Probable fact | Moderate | Screening cohort |
| Affirmed mention in assessment, two or more encounters | Present | Fact | High | Chronic disease phenotype |
| Affirmed mention plus matching medication or lab | Present | Fact | High | Treatment-aware phenotype |
| Mention only in past medical history | Historical | Possible fact, date unknown | Low to moderate | Sensitivity-first cohort |
| Mention only in family history | Family | Not a patient fact | None | Exclude from patient phenotype |
| Mention only under negation | Absent | Not a fact | High | Exclude |
| Mention only under uncertainty | Uncertain | Unresolved | Low | Flag for review |
| Mention only in an indication field | Not a finding | Not a fact | None | Exclude |
| Mention only in a copied sentence | Present but duplicated | One fact, not many | Moderate | Deduplicate before counting |
| Mention only in a scanned external record | Present, OCR-noisy | Possible fact | Low | Flag for manual review |

The table is a policy, not a truth. A project may choose the sensitivity-first row for a screening
use case and the two-encounter row for a quality measure. What matters is that the choice is
written down, applied consistently, and reflected in the reported unit.

### Worked decision: a single sentence, five readings

Consider the sentence "No evidence of HF, though mother had HF and she was treated for HF in 2019."
It contains three mentions of the same string. A naive matcher reports three findings. The correct
reading depends on scope and attribution:

| Mention | Cue | Assertion | Attribution | Temporality | Patient fact |
|---|---|---|---|---|---|
| HF after "no evidence of" | Negation | Absent | Patient | Current | No |
| HF after "mother had" | Experiencer | Present | Family | Historical | No |
| HF after "treated for" | None | Present | Patient | 2019 | Yes, historical |

The example is synthetic, but the pattern is routine. A discharge summary medication section, a
family history section, and an assessment paragraph can each contribute a mention of the same
concept with three different statuses. The corpus design must preserve enough context for the
assertion stage to separate them; a pipeline that concatenates all sections into one string has
made that separation harder.

### What "source" means for the rest of the module

Every later topic assumes a corpus with the following properties: documents are version-resolved,
note types are known, sections are recoverable, and the unit of analysis is declared. When a later
topic says "span-level F1," it means spans within these documents. When it says "patient-level
sensitivity," it means aggregation over these documents under a stated policy. If the corpus lacks
these properties, the metrics are not comparable across projects, and the comparison tables in
Topics 07 and 09 should be read as within-corpus only.

## 6. Worked example

A health system wants to identify admitted patients with documented heart failure (HF) for a
30-day readmission model. The prediction time is discharge. The unit of analysis is the admission.
The evidence is text documentation in the admission's documents.

### Corpus assembly

The warehouse extract contains 12,480 documents across 4,000 admissions for 3,410 patients.

| Note type | Documents | Admissions with at least one | Mean tokens |
|---|---:|---:|---:|
| Discharge summary (DS) | 4,000 | 4,000 | 2,980 |
| Progress note (PN) | 6,200 | 3,880 | 610 |
| Radiology report (RR) | 2,280 | 1,940 | 240 |
| Total | 12,480 | 4,000 | n/a |

Every admission has exactly one DS. This is a design feature of the site, not a universal truth.
The team confirms it by counting DS per admission and finding a maximum of one and a minimum of
one.

Two hundred and forty documents are addenda: 180 radiology addenda and 60 discharge summary
addenda. The initial extract included them as separate documents with no parent link, which
inflated the document count by 240 before resolution. After identity resolution the count is
12,240 base documents plus 240 addenda retained separately.

### Reference standard

The team samples 300 admissions by simple random sampling from the 4,000. Two clinicians review
all documents for each sampled admission and assign the admission-level label "documented HF"
under a written guideline. Disagreements are adjudicated by a third clinician.

| Reference standard | Admissions |
|---|---:|
| Positive | 128 |
| Negative | 172 |
| Total | 300 |

Observed agreement between the first two reviewers before adjudication is 0.903 (271 of 300).
Cohen's kappa is 0.798. Details of the agreement computation appear in Topic 06; here the point is
that the reference standard itself carries uncertainty, and the kappa is reported.

### System output at document level

The rule-based system flags a document if it contains an affirmed HF mention in the assessment
section or the problem list. At the admission level, any flagged document makes the admission
positive. On the 300 sampled admissions:

| | Reference positive | Reference negative |
|---|---:|---:|
| System positive | 96 | 24 |
| System negative | 32 | 148 |

Precision = 96 / (96 + 24) = 96 / 120 = 0.800.
Recall = 96 / (96 + 32) = 96 / 128 = 0.750.
F1 = 2 × 0.800 × 0.750 / (0.800 + 0.750) = 1.200 / 1.550 = 0.774.
Specificity = 148 / 172 = 0.860.
Accuracy = (96 + 148) / 300 = 244 / 300 = 0.813.

Accuracy is the least informative of these because prevalence in the sample is 42.7 percent, which
is not the cohort prevalence.

### Span-level view

Within the same 300 admissions, the reference standard identifies 1,540 HF mentions across all
documents. The system predicts 1,410 spans.

| Span match (exact offsets) | Count |
|---|---:|
| True positive | 1,180 |
| False positive | 230 |
| False negative | 360 |

Span precision = 1,180 / 1,410 = 0.837.
Span recall = 1,180 / 1,540 = 0.766.
Span F1 = 2 × 0.837 × 0.766 / (0.837 + 0.766) = 1.282 / 1.603 = 0.800.

The span F1 (0.800) is higher than the admission-level F1 (0.774). This is expected: many span
errors occur in documents that do not change the admission label. Reporting only the span number
would overstate the cohort quality.

### Where the admission-level errors come from

The 24 false positives and 32 false negatives are reviewed and categorized.

| Error category | FP | FN | Total |
|---|---:|---:|---:|
| Negated or historical mention treated as present | 15 | 0 | 15 |
| Radiology indication text | 6 | 0 | 6 |
| Abbreviation ambiguity (HF read as heart failure in a non-HF context) | 3 | 0 | 3 |
| HF documented only in a progress note not selected by the section rule | 0 | 18 | 18 |
| HF documented only in an addendum | 0 | 9 | 9 |
| HF documented only in a scanned external record with OCR noise | 0 | 5 | 5 |
| Total | 24 | 32 | 56 |

Two facts stand out. First, 15 of 24 false positives are assertion errors, not extraction errors.
The string was found; the status was wrong. Second, 14 of 32 false negatives are source-coverage
errors: the evidence existed but the pipeline did not read it.

### The assertion-corrected result

If the assertion module is applied and the 15 negated or historical false positives are removed,
with no change to true positives or false negatives:

| | Reference positive | Reference negative |
|---|---:|---:|
| System positive | 96 | 9 |
| System negative | 32 | 163 |

Precision = 96 / 105 = 0.914.
Recall = 0.750.
F1 = 2 × 0.914 × 0.750 / 1.664 = 1.371 / 1.664 = 0.824.

The improvement is 5.0 F1 points from a single assertion fix. That is larger than most modeling
gains reported in the literature for this task, which is a useful reminder that data-layer fixes
often dominate model-layer fixes.

### Two diagnostics that reveal a broken pipeline

**Diagnostic 1: note-type stratification of false positives.** Among the 24 false positives, 6
come from radiology reports, which is 25 percent of FPs from 18.3 percent of documents. The
radiology FPs are all "indication" mentions: "Evaluate for HF exacerbation." A pipeline that reads
the whole report cannot distinguish indication from finding. The diagnostic is to compute the
false-positive rate by note type and by section. A pipeline that is broken on source boundaries
shows a spike in a section that should not contain the target evidence.

**Diagnostic 2: document-count versus patient-count inflation.** The extract has 12,480 documents
but 4,000 admissions and 3,410 patients. If the team had reported "12,480 clinical notes" as the
corpus size, a reader would infer a much larger evidence base than exists. The diagnostic is to
report the ratio of documents to encounters to patients. A ratio above roughly 3:1:1 for inpatient
text usually signals repeated progress notes and copy-forward, and it should trigger a duplicate
check (Topic 02).

### Vary one assumption: include addenda

The base analysis excluded addenda from the extraction. Suppose the team includes addenda as
document text. Nine of the 32 false negatives were HF documented only in a discharge summary
addendum. Including addenda recovers those nine:

| Metric | Base | With addenda |
|---|---:|---:|
| True positive | 96 | 105 |
| False positive | 24 | 24 |
| False negative | 32 | 23 |
| Precision | 0.800 | 0.814 |
| Recall | 0.750 | 0.820 |
| F1 | 0.774 | 0.817 |

Precision = 105 / 129 = 0.814. Recall = 105 / 128 = 0.820. F1 = 2 × 0.814 × 0.820 / 1.634 =
1.335 / 1.634 = 0.817.

The conclusion changes: addenda are not a marginal formatting detail. They carried 7.0 percent of
the positive evidence in this cohort. A pipeline that drops them has a structural recall ceiling
that no model improvement can overcome.

### Downstream propagation

Assume the cohort is used to train a readmission model and the sampled estimates hold in the full
4,000 admissions at an HF prevalence of 32 percent, or 1,280 true positives.

With sensitivity 0.750 and specificity 0.860, the confusion matrix at cohort scale is:

| | HF present | HF absent |
|---|---:|---:|
| Predicted HF | 960 | 381 |
| Predicted not HF | 320 | 2,339 |

The negative group is 4,000 − 1,280 = 2,720 patients. Applying specificity 0.860 gives
0.860 × 2,720 = 2,339 true negatives and 381 false positives. Predicted HF total = 960 + 381 =
1,341. PPV = 960 / 1,341 = 0.716. The model's "HF cohort" would contain 381 patients without HF,
or 28.4 percent of the cohort. Those patients are not a random sample: they are patients whose
notes mention HF in an ambiguous context, which correlates with cardiac workup and comorbidity.
The readmission model would learn cardiac care patterns as if they were HF biology. This is error
propagation, and it is the reason patient-level PPV must be estimated at the deployment
prevalence, not the sample prevalence. At the sample prevalence of 42.7 percent, the same
sensitivity and specificity give PPV = (0.750 × 0.427) / ((0.750 × 0.427) + (0.140 × 0.573)) =
0.3203 / 0.4005 = 0.800, which is 8.4 points higher than the deployment value.

## 7. Code, query, or calculation

The following module implements the three operations this topic depends on: section segmentation,
clinical sentence splitting, and offset-preserving tokenization. It uses only the Python standard
library and runs on synthetic text.

```python
import re
import unicodedata

NOTE = """\
Chief Complaint: worsening shortness of breath.
History of Present Illness: Ms. A is a 71 y/o woman with a hx of HFpEF
who presents with 3 days of dyspnea. She denies chest pain. No fever.
Past Medical History: HTN, HFpEF, CKD stage 3. Mother with breast CA.
Medications: furosemide 40 mg b.i.d., metformin 500 mg.
Assessment and Plan: Acute on chronic HF exacerbation, likely volume
overload. If hypotension develops, hold metoprolol. Consider echo.
"""

HEADERS = [
    "chief complaint", "history of present illness", "past medical history",
    "medications", "assessment and plan", "social history", "review of systems",
    "physical exam", "labs", "imaging",
]

ABBREV = {"y/o", "hx", "b.i.d.", "mg.", "dr.", "ms.", "vs.", "approx.", "no."}


def normalize_unicode(text):
    text = unicodedata.normalize("NFKC", text)
    return text.replace(" ", " ").replace("’", "'").replace("–", "-")


def segment_sections(text):
    """Return list of (label, start, end) using line-initial headers."""
    lines = text.splitlines(keepends=True)
    offset = 0
    marks = []
    for line in lines:
        stripped = line.strip()
        m = re.match(r"^([A-Za-z][A-Za-z /&]{2,40}):", stripped)
        if m and m.group(1).strip().lower() in HEADERS:
            marks.append((m.group(1).strip().lower(), offset, offset + len(line)))
        offset += len(line)
    sections = []
    for i, (label, start, line_end) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        sections.append((label, start, end))
    return sections


def split_sentences(text, start, end):
    """Split on sentence punctuation, protecting known abbreviations."""
    chunk = text[start:end]
    protected = chunk
    for i, abbr in enumerate(sorted(ABBREV, key=len, reverse=True)):
        protected = protected.replace(abbr, f"@@{i}@@")
    parts = re.split(r"(?<=[.!?])\s+", protected)
    out, cursor = [], 0
    for part in parts:
        for i, abbr in enumerate(sorted(ABBREV, key=len, reverse=True)):
            part = part.replace(f"@@{i}@@", abbr)
        if part.strip():
            out.append((start + cursor, start + cursor + len(part), part.strip()))
        cursor += len(part) + 1
    return out


def tokenize(sentence):
    """Offset-preserving clinical tokenizer."""
    pattern = re.compile(r"\d+/\d+|[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?|[^\sA-Za-z\d]")
    return [(m.group(0), m.start(), m.end()) for m in pattern.finditer(sentence)]


text = normalize_unicode(NOTE)
sections = segment_sections(text)
print("sections:", [(s[0], s[1], s[2]) for s in sections])

for label, s, e in sections:
    for sent_start, sent_end, sent in split_sentences(text, s, e):
        if label == "history of present illness" and "denies" in sent:
            print("negation sentence:", repr(sent))
            print("tokens:", tokenize(sent)[:12])

covered = sum(e - s for _, s, e in sections)
print("chars covered by sections:", covered, "of", len(text))
```

Expected output:

```text
sections: [('chief complaint', 0, 48), ('history of present illness', 48, 189), ('past medical history', 189, 259), ('medications', 259, 315), ('assessment and plan', 315, 451)]
negation sentence: 'She denies chest pain.'
tokens: [('She', 0, 3), ('denies', 4, 10), ('chest', 11, 16), ('pain', 17, 21), ('.', 21, 22)]
chars covered by sections: 451 of 451
```

The output demonstrates three properties. Section offsets tile the document exactly, so no text is
silently dropped. The sentence splitter keeps "y/o" and "b.i.d." intact. The tokenizer preserves
"71" and "y/o" as separate tokens while keeping character offsets, which is what a span-level
evaluation needs.

A second calculation, the corpus ratio diagnostic:

```python
documents = 12480
admissions = 4000
patients = 3410
print("docs per admission:", round(documents / admissions, 2))
print("docs per patient:", round(documents / patients, 2))
print("admissions per patient:", round(admissions / patients, 2))
```

Expected output:

```text
docs per admission: 3.12
docs per patient: 3.66
admissions per patient: 1.17
```

A ratio of 3.12 documents per admission is plausible for an inpatient stay with daily progress
notes. A ratio of 12 would indicate either a long stay or duplicated extraction, and would justify
the duplicate check before any modeling.

## 8. Data, standards, and regulatory context

Source description is governed partly by standards and partly by contract. The distinction matters:
a LOINC code is a convention for interoperability, while a data use agreement is a binding legal
instrument.

**Document type coding.** Health Level Seven (HL7) Fast Healthcare Interoperability Resources
(FHIR) R4 represents a clinical document as a `DocumentReference` with a `type` coded element, or
as a `Composition` with sections. The `type` is commonly drawn from the LOINC document ontology.
USCDI v3 defines a Clinical Notes data class with five elements: Consultation Note (LOINC 11488-4),
Discharge Summary Note (18842-5), History & Physical (34117-2), Procedure Note (28570-0), and
Progress Note (11506-3). The class definition is "narrative patient data relevant to the context
identified by note types," and ONC explicitly notes that the elements should not be conflated with
structured document templates of the same name. That caveat is the regulatory version of this
topic's central point.

What the standard constrains: the code used for exchange, and the expectation that a discharge
summary contains admission and discharge dates, discharge instructions, and reasons for
hospitalization.

What it does not constrain: whether the note is complete, whether the local template matches the
LOINC definition, whether addenda are included, or whether the text is suitable for a given NLP
task. Certification of a health IT module against a USCDI element does not certify the clinical
content of the documents exchanged.

**FHIR provenance.** FHIR `Provenance` and `AuditEvent` resources can carry author, timestamp, and
agent information. In practice, warehouse extracts often flatten these into a few columns and lose
the agent role. A pipeline that needs author role should request it explicitly rather than assume
it survives the extract.

**Restricted corpora.** MIMIC-IV-Note is distributed by PhysioNet under the PhysioNet Credentialed
Health Data License 1.5.0. Access requires completion of CITI "Data or Specimens Only Research"
training and a signed data use agreement. Version 2.2, published 6 January 2023, contains 331,794
discharge summaries for 145,915 patients and 2,321,355 radiology reports for 237,427 patients. PHI
was replaced with three underscores. The dataset documentation reports 99.9 percent sensitivity
for radiology de-identification and states that manual review of discharge summaries found no PHI.
Those are the dataset owners' reported figures, not an independent audit of every note.

The binding constraint is the agreement, not the documentation. A credential lets a named
individual access the data in an approved environment; it does not permit redistribution,
re-identification, or use outside the approved protocol.

**HIPAA.** Free-text notes are designated record set content and are presumed to contain PHI. The
HIPAA Privacy Rule de-identification standard at 45 CFR 164.514(b) offers two paths: expert
determination, or Safe Harbor removal of 18 identifier categories. Safe Harbor also requires that
the covered entity have no actual knowledge that the remaining information could identify an
individual. A corpus built from notes is not de-identified because a tool removed names; it is
de-identified when the standard's conditions are met and documented. Topic 08 covers the
mechanics.

**Retention and minimum necessary.** Source inventory work frequently surfaces documents outside
the expected retention schedule, such as draft notes or messages retained under a different policy.
Including them may exceed the minimum necessary for the stated purpose. The governance question is
not only "can we read this" but "is this the least data that answers the question."

**Institutional review.** Research use of notes typically requires institutional review board
approval or a determination of non-human-subjects research, plus the data use agreement. The
corpus description in the protocol should match the corpus description in the paper. Divergence
between the two is a common audit finding.

**OMOP and the analytic gap.** The Observational Medical Outcomes Partnership (OMOP) Common Data
Model includes a `NOTE` table for free-text notes and a `NOTE_NLP` table for extracted terms with
fields for the extracted concept, the source span, the temporal relationship, and the assertion.
OMOP standardizes the storage of the extraction output, not the extraction itself. Two sites can
both populate `NOTE_NLP` and still have incomparable content because the note selection, the
section handling, and the assertion vocabulary are local. The common data model reduces schema
heterogeneity; it does not reduce corpus heterogeneity.

**What no standard covers.** There is no binding standard for note completeness, template
stability, copy-forward prevalence, or minimum corpus description for an NLP study. ISO 13606 and
openEHR define architectures for clinical information, and HL7 FHIR defines exchange resources,
but none of them specifies what a research corpus description must contain. The closest thing to
a reporting standard is the proposal in Velupillai et al. (2018), which recommends a minimal
reporting protocol covering data, approach, development, and evaluation. It is a recommendation,
not a requirement.

**Medical device regulation threshold.** If the text pipeline's output informs diagnosis or
treatment for a specific patient, the software may meet the definition of a medical device and be
subject to FDA or EU regulation. Source selection enters the regulatory argument because the
intended use statement names the documents the device reads. A device cleared for discharge
summaries is not cleared for patient messages. Topic 10 covers the regulatory framing in detail.

> **Note:** The distinction between binding requirement and convention is not cosmetic. A LOINC
> code in a `DocumentReference` is a convention that supports exchange; a signed data use
> agreement is a contractual obligation; a HIPAA de-identification standard is a legal condition.
> A corpus description that mixes the three invites a reviewer to assume protections that are not
> present.

> **Note:** LOINC codes identify document genres for exchange. They do not certify that a note of
> that genre contains the facts your phenotype needs. A "Discharge Summary Note" can be a
> two-line auto-generated stub at one site and a four-page synthesis at another.

## 9. Industry practice and case evidence

**MIMIC-IV-Note as a de facto source reference.** The MIMIC-IV-Note release separates discharge
summaries from radiology reports and provides `charttime` and `storetime` for each note. The
separation is instructive: the two genres have different structures and different labeling
properties, and the dataset does not pretend they are one corpus. The reported counts (331,794
discharge summaries; 2,321,355 radiology reports) also illustrate the genre imbalance that any
multi-source pipeline faces. Inference: a model trained on the pooled corpus would be dominated by
radiology language unless the training sample is stratified.

**n2c2 and i2b2 corpora as genre-specific benchmarks.** The 2010 i2b2/VA challenge used discharge
summaries and progress reports from three health systems, with 394 training reports and 477 test
reports across concept, assertion, and relation tasks. The 2014 i2b2/UTHealth de-identification
corpus contained 1,304 longitudinal records for 296 patients, split 790 training and 514 test. The
2018 n2c2 cohort selection track used 288 patients and 781,006 tokens from the 2014 corpus,
annotated against 13 selection criteria, with a 202/86 patient train/test split. These are
documented corpus facts from the challenge overview papers. Inference: the repeated reuse of a
small number of source corpora is a generalization risk, because the genres and sites are few and
the annotators are shared.

**MedCAT and local vocabulary.** MedCAT performs concept extraction and linking to SNOMED CT and
UMLS, and is designed for local tuning of the vocabulary and the concept database. The design
choice reflects an operational reality: a concept dictionary that works at one site does not
automatically cover another site's abbreviations and shorthand. Inference: source inventory and
vocabulary curation are the same activity performed at different granularity.

**CheXpert and report-derived labels.** CheXpert extracted labels from radiology reports,
including uncertainty labels, to create a large supervision set. The resulting labels inherit the
reporting conventions of the source institution, including how often "cannot exclude" appears and
what it means there. Inference: report-derived labels are a form of source-specific annotation,
and their transportability is limited by reporting practice.

**Operational corpus descriptions in industry.** A recurring pattern in vendor and health system
documentation is to describe a corpus by note count and site count, without note-type distribution
or duplication rate. That description is insufficient to judge transportability. A stronger
description states: source systems, note types with counts, date range, inclusion and exclusion
rules, addendum handling, de-identification method, and the document-to-patient ratio. Teams that
adopt this template find that reviewers ask fewer unanswerable questions.

**cTAKES and the pipeline-as-source-model.** The Mayo clinical Text Analysis and Knowledge
Extraction System (cTAKES) was built around the clinical document architecture: it expects
sections, and its components use section context for assertion and concept mapping. The
architectural choice is evidence that source structure was treated as part of the method, not as
preprocessing. Inference: pipelines that discard section structure lose a signal that the
reference systems relied on.

**What is not publicly documented.** Most health systems do not publish note-type inventories or
template change histories. Claims about "typical" note-type mixes should therefore be treated as
inference from the author's experience, not as established fact. The corpora that are documented,
such as MIMIC and the n2c2 sets, are not representative of all sites.

**A pattern from operational reviews.** In the author's experience of corpus audits, the three
most frequent findings are: an inner join on encounter that silently dropped patient-linked
messages; addenda excluded because the extract treated them as separate documents; and a note-type
label that did not mean what its name suggested, because the local configuration reused a
vendor label for a different workflow. These are inferences from practice, not published findings,
and they should be tested rather than assumed at any new site.

**Vendor documentation versus site reality.** EHR vendors publish note-type taxonomies, and sites
customize them. The customization is where the variation lives: a "progress note" at one site may
include a structured handoff template, while at another it is free narrative. A vendor taxonomy is
a useful starting inventory and a poor substitute for a site visit or a data profiling pass.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Treating all notes as one genre | Corpus description says "clinical notes" with no type table | Genre-specific vocabulary and structure averaged away | Metrics hide the stratum that fails | Stratify by note type; report per-type metrics |
| Ignoring addenda | Document count equals base note count exactly | Addenda stored separately and excluded | Recall ceiling from missing evidence | Include addenda with parent link; test sensitivity |
| Double-counting documents | Document-to-admission ratio far above clinical expectation | Drafts, addenda, or repeated extracts | Inflated sample size and confidence | Identity resolution with content hash |
| Using storetime as clinical time | Note times cluster at end of shift | Batch signing | Temporal leakage in prediction | Use charttime; flag late entries |
| Reading indication fields as findings | Radiology FPs concentrated in one section | Report structure not modeled | False positives in cohort | Section-aware extraction; exclude indication |
| Losing patient-linked documents | Inner join on encounter drops messages | Schema requires encounter id | Missing social context | Left join; count unlinked |
| Section header missed | Medication list merged into assessment | Local synonym absent from lexicon | Wrong assertion and attribution | Expand lexicon; validate on random sample |
| Template text counted as observation | "Denies chest pain" in 100 percent of notes | Pre-populated review of systems | False negatives and spurious features | Template detection; flag or remove |
| Abbreviation expanded globally | "MS" always mapped to multiple sclerosis | No context disambiguation | Systematic concept errors | Local lexicon plus context model |
| OCR noise unevaluated | Character error rate unknown for scanned docs | Scanned external records included raw | Span mismatch and missed mentions | Measure CER; normalize; exclude or flag |
| Unrepresentative site mix | One site contributes most documents | Convenience sampling | Claims do not generalize | Coverage table; reweight or restrict |
| Undocumented exclusion | Final N smaller than extract with no reason | Silent filtering in code | Irreproducible cohort | Exclusion log with counts |
| Draft text in extract | Duplicate identifiers with different content | Version field absent | Legal record differs from analyzed text | Version resolution; retain audit trail |
| PHI in unexpected fields | Identifiers found in free-text body | De-identification focused on headers | Privacy breach | Full-text de-identification; residual audit |

Three of these failures are structural rather than technical, and they deserve separate treatment
because no model change fixes them.

The first is genre averaging. When a corpus mixes discharge summaries, progress notes, and
radiology reports, the aggregate metric is a weighted average in which the most frequent genre
dominates. If radiology reports are 60 percent of documents and the system performs 12 points worse
on them, the aggregate hides a failure that will be the dominant failure in deployment if the
deployment mix differs. The red flag is a corpus table with one row. The mitigation is a stratified
table and a decision about which stratum the deployment resembles.

The second is hidden exclusion. Every pipeline filters: empty documents, very short documents,
documents in an unexpected language, documents that fail decoding. Each filter has a clinical
meaning. Dropping documents under 20 tokens removes the terse "no acute events overnight" notes
that are common in stable patients, which biases the corpus toward sicker patients. The red flag is
a final N that cannot be reconciled with the extract N. The mitigation is an exclusion log with a
reason and a count for every filter.

The third is version drift. Templates change, note types are renamed, and services merge. A corpus
spanning five years may contain three template regimes, and a model trained on the pooled corpus
learns an average of incompatible structures. The red flag is a header frequency that changes
sharply by quarter. The mitigation is to stratify by era, test for drift, and report performance
by period.

A practical red-flag checklist for a corpus review has six questions. Does the corpus description
state the note types and their counts? Does it state the document-to-patient ratio? Does it state
the addendum and version policy? Does it state the exclusion rules with counts? Does it report
metrics by note type? Does it state which stratum the deployment resembles? A corpus that cannot
answer all six is not ready for a generalization claim, regardless of its size.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The selected note type contains the target evidence | Justifies corpus restriction | Evidence documented only in another genre | Chart-review a sample of negatives for missed evidence |
| Document metadata are accurate | Provenance prior depends on it | Author or note type mislabelled at source | Compare metadata with note content on a sample |
| One document equals one clinical event | Simplifies counting | Copy-forward and addenda | Duplicate detection; event-date extraction |
| Charttime reflects clinical time | Temporality anchor | Late entries and back-dating | Distribution of storetime minus charttime |
| Section headers are stable within a site | Enables segmentation | Template changes mid-study | Header frequency by quarter |
| The corpus is representative of deployment | Supports generalization claim | Convenience sampling by site or service | Coverage table against source population |
| Reference standard captures the phenotype | Defines truth | Reviewer disagreement or guideline ambiguity | Inter-annotator agreement; adjudication rate |
| De-identification is complete | Permits research use | Residual identifiers in free text | Per-identifier recall audit (Topic 08) |
| Note text is the only evidence needed | Scopes the project | Structured data carries the diagnosis | Compare text-only and structured-only phenotypes |
| Token counts are comparable across notes | Supports length analyses | Tokenizer differences, templates | Fixed tokenizer; report characters too |
| Addenda are supplementary, not contradictory | Justifies concatenation | Addendum changes the diagnosis | Review addendum content sample |
| Patient linkage is correct | Prevents record mix-up | Overlay or merge errors | Check duplicate content across patients |
| Note headers are stable enough to detect | Enables section-aware extraction | Free-text notes with no headers | Measure header detection rate on a sample |
| The deployment mix resembles the training mix | Supports transportability | Case-mix or service change | Compare note-type distribution over time |
| Reference-standard reviewers saw the same documents as the pipeline | Makes comparison fair | Reviewers had chart access the pipeline lacked | Record reviewer information set |

The strongest assumption in the ledger is the first: that the selected note type contains the
target evidence. It is also the hardest to test, because testing it requires reviewing the
documents the pipeline never saw. A practical test is to sample admissions the pipeline labeled
negative, review the full record, and count how many had the target documented somewhere outside
the selected sources. That count is the recall ceiling imposed by source selection, and it is worth
reporting separately from model recall. In the worked example, the addendum-only cases were exactly
this kind of ceiling: 9 of 128 positives, or 7.0 percent, were unreachable before any model was
applied.

A second assumption worth isolating is that charttime is clinical time. In most EHRs it is a
human-entered field and can be wrong. A small fraction of notes are charted at the wrong hour or
the wrong day, and a smaller fraction are charted with a future date because of a template default.
The test is distributional: plot storetime minus charttime and look for a spike at the end of a
shift, a cluster at midnight, or negative values. Negative values mean storetime precedes
charttime, which is usually a timezone or interface error rather than a clinical event.

## 12. Exercises

**Exercise 1.1 — Source inventory (application, 60 minutes).**
Deliverable: a table of at least eight clinical text sources available in a health system you know
or can research, with columns for note type, author role, setting, typical length, template family,
and one extraction task the source supports well and one it supports poorly.
Acceptance criteria: eight rows; every row names a task and a failure mode; no source described
only as "notes."

**Exercise 1.2 — Section segmentation by hand (application, 45 minutes).**
Deliverable: take the synthetic note in Section 7 and add three sections with realistic local
headers not in the lexicon. Hand-label the character offsets of every section.
Acceptance criteria: offsets tile the document; at least one header is a synonym; you state which
header your lexicon would miss and why.

**Exercise 1.3 — Corpus description (application, 45 minutes).**
Deliverable: a 400-word corpus description for a phenotype of your choice, following the template
in Section 9 (sources, types with counts, dates, inclusion and exclusion, addendum handling,
de-identification, document-to-patient ratio).
Acceptance criteria: every element present; document-to-patient ratio computed; no claim that the
corpus is representative without a coverage argument.

**Exercise 1.4 — Unit of analysis (critique, 30 minutes).**
Deliverable: for three decisions (readmission risk, message triage, quality measure), state the
unit of analysis, the prediction time, and the documents eligible. Identify one decision where a
document-level design would produce a misleading result.
Acceptance criteria: three decisions; units justified; the misleading design explained with a
concrete mechanism.

**Exercise 1.5 — Addendum sensitivity (calculation, 40 minutes).**
Deliverable: using the numbers in Section 6, recompute precision, recall, and F1 if addenda are
included and, separately, if the nine addendum-only positives are excluded but the 24 false
positives remain.
Acceptance criteria: all arithmetic shown; the two results compared; one sentence on which result
you would report and why.

**Exercise 1.6 — Source-mix confounding (critique, 45 minutes).**
Deliverable: a one-page critique of a hypothetical study that reports a single span F1 of 0.86 on
a corpus of 50,000 notes from one site, with no note-type table and no duplication check. Name
three specific threats and the table you would request.
Acceptance criteria: three threats tied to mechanisms in this topic; the requested table specified
by column.

**Exercise 1.7 — Provenance audit (design, 60 minutes).**
Deliverable: a provenance schema for a multi-site corpus, with fields, types, and a rule for
missing values. Include how you would detect late entries and addenda.
Acceptance criteria: at least 12 fields; missing-value policy per field; detection rule for each of
late entry and addendum.

**Exercise 1.8 — Reproduction test (design, 45 minutes).**
Deliverable: a checklist that a second analyst could follow to rebuild your corpus from the raw
extract, including identity resolution, exclusions, and version handling.
Acceptance criteria: every step has a rule and an expected count; the checklist produces the same
document count as your analysis.

## 13. Self-check answers

**1.4 (unit of analysis).** Readmission risk: unit is the admission, prediction time is discharge,
eligible documents are those charted before discharge. Message triage: unit is the message,
prediction time is message receipt, eligible document is the message plus prior context. Quality
measure: unit is the eligible encounter defined by the measure specification, prediction time is
the measurement period end. A document-level design misleads for readmission because one admission
generates many documents and a single document flag would make the admission positive without
representing the admission's evidence base; the probability that any one document mentions the
target rises with note count, so long stays are preferentially flagged.

**1.5 (addendum sensitivity).** Base: TP 96, FP 24, FN 32. Precision 0.800, recall 0.750, F1
0.774. Including addenda: TP 105, FP 24, FN 23; precision 105/129 = 0.814; recall 105/128 = 0.820;
F1 = 2(0.814)(0.820)/(1.634) = 0.817. If the nine addendum-only positives are excluded but the 24
FPs remain: TP 96, FP 24, FN 32, which is the base case. The comparison shows that addenda affect
recall, not precision, in this cohort. Report the included-addenda result if addenda are part of
the deployed pipeline; otherwise report the base result with the recall ceiling stated.

**1.1 (source inventory) rubric.** Full credit requires eight sources, each with a task and a
failure mode, and at least two sources that are not inpatient notes. Half credit for a list of note
types without tasks. Zero for a list that treats "clinical notes" as one source.

**1.3 (corpus description) rubric.** Full credit: sources, counts by type, date range, inclusion
and exclusion with counts, addendum policy, de-identification method, ratio, and a coverage
statement. Half credit: all elements except exclusion counts or ratio. Zero if the description
cannot be used to rebuild the cohort.

**1.6 (source-mix critique).** Expected threats: (a) single-site template and abbreviation
conventions inflate apparent generalization; (b) no note-type table hides genre imbalance, so the
aggregate F1 is dominated by the most frequent genre; (c) no duplication check means the effective
sample is smaller than reported and copy-forward inflates agreement between train and test. The
requested table: note type, documents, admissions, patients, mean tokens, duplicate rate, and
per-type precision, recall, and F1.

**1.7 (provenance schema) rubric.** Full credit: document id, parent id, version, content hash,
patient id, encounter id, note type, author id, author role, service, site, charttime, storetime,
signed flag, template id, extract time; missing-value policy per field; late entry defined as
storetime minus charttime above a stated threshold; addendum detected by parent id or type code.
Half credit: 12 fields without policies.

**1.8 (reproduction checklist) rubric.** Full credit: ordered steps, each with a command or rule,
an expected count, and a reconciliation point where the count can diverge. Half credit: steps
without expected counts.

**1.2 (segmentation).** A correct answer tiles the document and identifies at least one synonym
the lexicon misses, for example "HPI" for history of present illness or "A/P" for assessment and
plan. The offsets must match the raw text, not the normalized text.

## 14. Interview questions

**1. "What do you mean by clinical notes?"**
Strong answer: names specific genres, states that each has a different structure and evidence
profile, and gives an example of a task that one genre supports and another does not. Weak answer:
"any text in the EHR," which reveals no source awareness.

**2. "How would you decide which notes to include in a phenotype?"**
Strong answer: starts from the evidence the phenotype requires, maps evidence to note types and
sections, then sets inclusion rules with counts and an exclusion log. Weak answer: "all notes for
the patient," which ignores duplication and genre.

**3. "What is the difference between charttime and storetime, and why does it matter?"**
Strong answer: charttime is clinical time, storetime is system time; late entries and batch signing
make storetime unsafe as a prediction cutoff. Weak answer: treats them as synonyms.

**4. "How do you handle addenda?"**
Strong answer: version-resolve, keep the parent link, decide whether to concatenate based on the
task, and measure sensitivity to the choice. Weak answer: "merge them," without a rule.

**5. "What is your unit of analysis?"**
Strong answer: states patient, encounter, document, or span, and ties it to the decision and the
denominator. Weak answer: "the note," for every problem.

**6. "How would you describe a corpus so someone could reproduce it?"**
Strong answer: sources, note types with counts, dates, inclusion and exclusion, addendum policy,
de-identification, document-to-patient ratio, and coverage. Weak answer: "50,000 notes from our
EHR."

**7. "Why is a single F1 across a mixed corpus misleading?"**
Strong answer: genre imbalance means the aggregate is dominated by the most frequent type, and
per-type performance can differ by more than 20 points. Weak answer: "F1 is fine."

**8. "What does LOINC 18842-5 tell you?"**
Strong answer: it identifies a discharge summary in the LOINC document ontology, and it constrains
exchange semantics, not content completeness or NLP suitability. Weak answer: assumes it
guarantees a complete summary.

**9. "How would you detect that your corpus is unrepresentative?"**
Strong answer: coverage table against the source population by site, service, author role, and
patient subgroup, plus note-length distributions. Weak answer: compares raw counts only.

**10. "What is the first thing you check when a note count looks too high?"**
Strong answer: identity resolution and duplication: content hash, parent identifiers, documents per
admission and per patient. Weak answer: assumes the extract is correct.

**11. "Where can source selection create privacy risk?"**
Strong answer: patient-linked messages, scanned external records, and free-text identifiers outside
structured fields, each expanding the PHI surface. Weak answer: only names and dates.

## 15. Key terms

- **Addendum** — a later addition to a signed note, retained as a separate version. See Section 4.
- **Assertion** — the contextual status of a mention; distinct from the mention itself.
- **Attribution** — the subject a statement applies to.
- **Boilerplate** — text repeated by template design rather than clinical observation.
- **Charttime** — the clinical time a note applies to.
- **Clinical note** — a narrative document authored during care.
- **Concept identifier** — a code from a controlled vocabulary.
- **Corpus** — the selected documents plus their inclusion criteria.
- **Coverage table** — counts of the corpus against the source population by stratum.
- **Credentialed access** — access gated by identity verification and training.
- **Data use agreement** — a binding contract governing data access.
- **De-identification** — removal or transformation of identifiers under a defined standard.
- **Diagnosis** — a clinical determination, not a string.
- **Document type code** — a standard code for a document genre, often LOINC.
- **Encounter** — a single interaction or stay with defined boundaries.
- **FHIR** — Fast Healthcare Interoperability Resources, an HL7 standard.
- **LOINC** — Logical Observation Identifiers Names and Codes.
- **Mention** — a text span referring to a concept.
- **Note bloat** — length inflation from templates and copying.
- **Note type** — the local genre label assigned by the EHR.
- **Phenotype** — a computable patient-level condition definition.
- **PHI** — protected health information under HIPAA.
- **Provenance** — the origin and history of a document.
- **Sampling frame** — the population from which documents were drawn.
- **Section** — a labelled region of a note with a clinical function.
- **SNOMED CT** — a clinical terminology used for concept normalization.
- **Storetime** — the system time a note was saved or signed.
- **Temporality** — the relation of an event to document creation time.
- **Template** — the pre-populated structure a clinician fills in.
- **Token** — the unit produced by a tokenizer.
- **UMLS** — Unified Medical Language System; source of concept unique identifiers.
- **USCDI** — United States Core Data for Interoperability.

## 16. Sources

All URLs accessed 25 October 2026.

1. U.S. Department of Health and Human Services. "Summary of the HIPAA Privacy Rule:
   De-identification." https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
2. 45 CFR 164.514(b), de-identification standard and Safe Harbor identifier list. Cornell Legal
   Information Institute. https://www.law.cornell.edu/cfr/text/45/164.514
3. ONC. United States Core Data for Interoperability, Version 3 (July 2022, October 2022 errata).
   https://isp.healthit.gov/sites/default/files/2022-07/USCDI-Version-3-July-2022-Final.pdf
4. ONC Interoperability Standards Platform. "Clinical Notes" data class.
   https://isp.healthit.gov/uscdi-data-class/clinical-notes
5. PhysioNet. MIMIC-IV-Note: Deidentified free-text clinical notes, v2.2 (6 January 2023).
   https://physionet.org/content/mimic-iv-note/2.2/
6. MIT Laboratory for Computational Physiology. MIMIC-IV Note module documentation.
   https://mimic.mit.edu/docs/iv/modules/note/
7. Uzuner O, South BR, Shen S, DuVall SL. "2010 i2b2/VA challenge on concepts, assertions, and
   relations in clinical text." J Am Med Inform Assoc. 2011;18(5):552-556.
   https://doi.org/10.1136/amiajnl-2011-000203
8. Stubbs A, Filannino M, Soysal E, Henry S, Uzuner O. "Cohort selection for clinical trials:
   n2c2 2018 shared task track 1." J Am Med Inform Assoc. 2019;26(11):1163-1171.
   https://doi.org/10.1093/jamia/ocz163
9. Stubbs A, Uzuner O. "Annotating longitudinal clinical narratives for de-identification: The
   2014 i2b2/UTHealth corpus." J Biomed Inform. 2015;58 Suppl:S20-S29.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC4978170/
10. Kraljevic Z, Searle T, Shek A, et al. "Multi-domain clinical natural language processing with
    MedCAT: The Medical Concept Annotation Toolkit." Artif Intell Med. 2021;117:102083.
    https://doi.org/10.1016/j.artmed.2021.102083
11. Irvin J, Rajpurkar P, Ko M, et al. "CheXpert: A large chest radiograph dataset with uncertainty
    labels and expert comparison." AAAI 2019. https://doi.org/10.1609/aaai.v33i01.3301590
12. HL7. FHIR R4 DocumentReference and Composition resources.
    https://hl7.org/fhir/R4/documentreference.html
13. Velupillai S, Suominen H, Liakata M, et al. "Using clinical Natural Language Processing for
    health outcomes research." J Biomed Inform. 2018;88:11-19.
    https://doi.org/10.1016/j.jbi.2018.10.005

## 17. Links

- Module map: [`_index.md`](_index.md)
- Next topic: [`02_documentation-behaviour.md`](02_documentation-behaviour.md)
- Source and annotation dependency: [`06_annotation.md`](06_annotation.md)
- Evaluation of source-stratified results: [`07_evaluation.md`](07_evaluation.md)
- De-identification of free-text corpora: [`08_deidentification.md`](08_deidentification.md)
- Module survey: [`../../09_clinical_nlp.md`](../../course/09_clinical_nlp.md)
- Labs: [`../../labs/`](../../labs/)
