# Documentation Behaviour

> **Module 09 — Clinical NLP**
> Topic 02 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Topic 01
> Version 2026.3

## 1. Why this topic exists

Clinical text is not a neutral recording of patient state. It is the residue of a workflow. The
clinician writes under time pressure, inside a template designed by a committee, with a note from
yesterday open in another window. The result is a document that mixes fresh observation, copied
text, pre-populated normal findings, and language written for billing or legal protection. A model
trained on this text learns the workflow as well as the patient.

The most consequential behaviour is copy-forward: the reuse of text from a prior note about the
same patient. A 2022 study of 104.5 million notes at one health system found that 50.1 percent of
all clinical text was duplicated from prior notes, rising from 33.0 percent in 2015 to 54.2 percent
in 2020 (Steinkamp, Kantrowitz, and Airan-Javia, JAMA Network Open). An earlier character-level
provenance study at another institution found that only 18 percent of note text was manually typed,
46 percent was copied, and 36 percent was imported from another source (Wang, Khanna, and Najafi,
JAMA Internal Medicine, 2017). These are different measures on different corpora, and they agree on
the direction: a large fraction of what a note contains was not written for this encounter.

Copy-forward is not simply a data quality nuisance. It changes the meaning of time. A diagnosis
copied into today's assessment may have been resolved three years ago. A plan copied forward may
describe an intention that was never carried out. A medication list copied forward may include a
drug discontinued last week. When an NLP pipeline reads the note as a statement about today, it
inherits the error. The model is not wrong about the text; it is wrong about what the text refers
to.

Templates create a second family of problems. A pre-populated review of systems that says "denies
chest pain" before the patient is asked is a statement about the template, not the patient. A
structured assessment template with a default "stable" may be left unchanged. A smart phrase may
insert a paragraph of generic counseling. These artefacts inflate note length, create false
negatives, and make certain phrases so frequent that they carry no information. A model can learn
to use template presence as a proxy for encounter type, which then becomes a proxy for acuity, and
the resulting feature is a workflow signature rather than a clinical one.

Late entries and addenda create a third family. A note charted hours after the event, an addendum
that reverses the impression, a correction that replaces a signed note: each is a versioning event
that changes what the record says and when it said it. A pipeline that reads only the final text
and the charttime will attribute the corrected impression to the original time, which can leak
outcome information into a prediction made before the correction existed.

This topic exists because these behaviours are measurable, and because measuring them changes the
modeling plan. A team that knows its corpus is 45 percent duplicated will not build a note-level
count feature. A team that knows its template contributes 60 percent of tokens will not train a
topic model on raw text. A team that knows a template changed in the middle of its study period
will stratify by era rather than discover the drift after deployment.

The topic also has a compliance dimension. Documentation is regulated: CMS sets evaluation and
management documentation requirements, payers audit notes, and the 21st Century Cures Act's
information blocking provisions give patients access to their notes. Clinicians write knowing that
the note may be read by a patient, an auditor, or a plaintiff. That audience shapes what is written
and what is omitted, and an NLP team that ignores it will misread euphemism as absence.

There is a measurement argument too. Documentation behaviour is one of the few properties of a
clinical corpus that can be estimated without clinical expertise. Duplication is a textual property;
template frequency is a textual property; late entry is a timestamp property. A data engineer can
produce the first pass of the audit, and a clinician can review the sample. That division of labour
makes the audit cheap enough to run before every project rather than after a model fails.

> **Note:** The distinction the module keeps returning to applies here with force. A copied mention
> is still a mention. Its assertion status may be stale, and its temporality is almost certainly
> not the note date. Documentation behaviour is the main reason a mention cannot be trusted as a
> statement about the present.

## 2. Learning outcomes

By the end of this topic you can:

1. **Identify** the documentation behaviours that alter the meaning of clinical text: copy-forward,
   templating, auto-population, late entry, addendum, and amendment.
2. **Measure** duplication prevalence in a corpus using shingle similarity, with a stated threshold,
   a sampling design, and a confidence interval.
3. **Distinguish** copy-forward from template boilerplate using a negative control and a
   time-gap analysis.
4. **Quantify** the effect of duplication on a text-derived feature at the document, encounter, and
   patient level.
5. **Design** a de-duplication policy that preserves clinical evidence while removing repeated
   text, and **state** what the policy loses.
6. **Detect** template drift across a study period and **choose** a stratification or
   normalization response.
7. **Explain** how documentation incentives and patient access shape what notes contain.
8. **Write** a documentation-behaviour section for a validation report that a reviewer can
   reproduce.

Outcomes 2 through 4 are computational; 5 through 8 are design and critique.

## 3. The core mental model

The note is produced by a workflow, and the workflow leaves a signature in the text. The signature
has four components: what was observed, what was copied, what was templated, and what was written
for an audience other than the next clinician.

```text
   patient state
        |
        v
  +-------------------+     +------------------+     +------------------+
  | clinical          |     | prior note text  |     | template +       |
  | observation       |     | (copy-forward)   |     | auto-population  |
  +---------+---------+     +--------+---------+     +--------+---------+
            |                        |                        |
            +------------+-----------+------------+-----------+
                         |
                         v
                 +---------------+
                 |  THIS NOTE    |
                 |  - fresh text |
                 |  - copied text|
                 |  - template   |
                 |  - attestation|
                 +-------+-------+
                         |
          +--------------+---------------+
          |              |               |
          v              v               v
    today's fact   stale fact      non-fact (template)
          |              |               |
          +--------------+---------------+
                         |
                         v
              what an NLP model sees
```

The diagram makes the central problem visible: three inputs produce one string, and the string does
not label its own provenance. The model sees "HF" and cannot tell whether it came from today's
assessment, last year's discharge summary, or a template's problem list.

A second model is the **duplication triangle**: similarity, time, and authorship.

```text
                 high similarity
                       /\
                      /  \
                     /    \
        short time gap      same author
                   \        /
                    \      /
                     \    /
                  copy-forward
```

Copy-forward typically shows high similarity, a short or moderate time gap, and the same author or
care team. Template boilerplate shows high similarity, any time gap, and often different authors,
because everyone uses the same template. Quoted text, such as a copied consultant recommendation,
shows high similarity, a short gap, and a different author, with an attribution phrase. The three
patterns are distinguishable, but only if the analysis includes a negative control: similarity
between notes from different patients of the same note type.

A third model is **information decay**. A copied statement does not become false at a fixed rate.
It decays at a rate that depends on the stability of the underlying fact.

| Fact type | Decay | Example | Consequence of copying |
|---|---|---|---|
| Permanent | None | Type 1 diabetes, mastectomy | Safe to carry forward |
| Slowly changing | Low | Hypertension, CKD stage | Usually safe, stage may be stale |
| Episodic | High | Pneumonia, DKA | Unsafe; resolved event reads as current |
| Plan | Very high | "Start antibiotics" | Unsafe; intent may be obsolete |
| Negative | High | "No chest pain" | Unsafe; symptom may have developed |

The table is a design tool. A phenotype should specify which fact types it extracts and how it
handles copied text for each. A permanent condition can tolerate aggressive copy-forward. An
episodic condition cannot.

A fourth model is the **note as a legal and communicative act**. Documentation requirements create
language that is not clinical observation: attestation statements, medical necessity language,
counseling templates, time statements. This language is highly predictable and highly frequent. It
is a major source of false positives for concepts that appear in it, such as "risk," "complication,"
or "discussed." The model that learns it learns billing.

Duplication is not uniform, and the uneven distribution is what makes it dangerous. It concentrates
in the sections that change least: past medical history, medications, allergies, social history, and
the problem list. Those are exactly the sections a phenotype uses for chronic conditions. The
sections that change most, the assessment and plan, are also copied often, but with edits, so they
produce near-duplicates rather than exact copies. A method tuned to exact duplication will find the
stable sections and miss the edited plan, which is where the stale clinical reasoning lives.

The interaction with note type is equally uneven. Discharge summaries synthesize an entire
admission and are written once, so their internal duplication is low but their overlap with
progress notes is high. Progress notes are written daily and copy the previous day, so their
consecutive duplication is high. Radiology reports are short and templated, so their similarity is
driven by template rather than copying. A single corpus-level prevalence averages these regimes and
describes none of them.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Copy-forward | Reuse of text from a prior note about the same patient | Called copy-paste, which is only one mechanism | Character or token span |
| Copy-paste | Manual duplication via clipboard within or across notes | Used as synonym for all duplication | Span |
| Carry-forward | Automatic population of prior values by the EHR | Confused with clinician copying | Field or span |
| Clone | A note that duplicates a prior note with minimal change | Used loosely for any similar note | Document pair |
| Template | Predefined structure with fixed and variable regions | Called boilerplate in full | Template identifier |
| Boilerplate | Fixed text repeated by design | Used for any frequent phrase | Span |
| Smart phrase | A user-triggered expansion macro | Confused with template | Span |
| Macro | A stored text expansion, often personal | Confused with organizational template | Span |
| Auto-population | System insertion of structured data into narrative | Invisible to the reader | Field |
| Note bloat | Length inflation from duplication and templating | Treated as a style issue | Tokens per note |
| Late entry | A note charted after the clinical event | Confused with back-dating | Time difference |
| Back-dating | Assigning charttime earlier than the actual writing time | Assumed rare | Timestamp |
| Addendum | A later addition to a signed note | Merged without a flag | Document version |
| Amendment | A correction that changes the signed record | Treated as a new note | Document version |
| Attestation | A statement that the author reviewed and accepts the content | Read as clinical content | Span |
| Problem list carryover | Unreconciled problem list entries persisting across encounters | Treated as current diagnosis | Problem list entry |
| Provenance | Origin of a text span: author, time, source | Reduced to document metadata | Span-level attribute |
| Shingle | A contiguous sequence of n tokens used for similarity | Called an n-gram without stating n | Token sequence |
| Jaccard similarity | Intersection over union of two shingle sets | Used without a threshold | Ratio, 0 to 1 |
| Containment | Intersection over the size of one set | Confused with Jaccard | Ratio, 0 to 1 |
| Near-duplicate | A pair above a similarity threshold but not identical | Treated as identical | Pair |
| First-occurrence policy | Keep only the earliest instance of repeated text | Assumed lossless | Span selection |
| Information decay | Rate at which a copied fact stops being true | Ignored | Fact-type dependent |
| Template drift | Change in template over time | Detected after deployment | Era or version |
| Documentation burden | Time and cognitive cost of note writing | Used only as a burnout claim | Minutes or clicks |
| Open notes | Patient access to clinical notes | Assumed to have no effect on content | Policy status |
| Evaluation and management (E/M) | CMS billing category with documentation requirements | Confused with clinical severity | CPT code family |
| Attending attestation | Statement of supervisory review | Read as independent confirmation | Span |
| Forward copy | Text in a later note taken from an earlier one | Assumed to be the only direction | Pair direction |
| Backward copy | Text in an earlier note that matches a later one | Mistaken for clinical foresight | Pair direction |
| Template signature | The set of phrases that identifies a template family | Confused with content | Phrase set |
| Duplicate-driven feature | A feature whose value changes when duplicates are removed | Reported as a clinical finding | Feature |
| Note interval | Time between consecutive notes for a patient | Ignored in similarity analysis | Hours or days |

Two terminological distinctions matter for measurement.

**Duplication versus redundancy.** Duplication is repeated text. Redundancy is repeated meaning in
different words. A clinician who rewrites yesterday's assessment in new language has produced
redundancy without duplication. Shingle methods miss it. Semantic methods catch some of it at much
higher cost. Most published prevalence estimates measure duplication, and they therefore
underestimate redundancy.

**Copy-forward versus template.** Both produce repeated text. The difference is the source: a prior
patient-specific note versus a shared template. The distinction matters because the remedies differ.
Copy-forward is addressed with de-duplication and temporality. Template is addressed with
boilerplate detection and section-aware extraction. A pipeline that treats both as "duplicates"
will remove patient-specific evidence along with the template.

## 5. The mechanics, step by step

This section describes a documentation-behaviour audit: a reproducible procedure for measuring
duplication, templating, and versioning in a corpus, and for estimating their effect on a planned
feature.

### Step 1. State the question the audit serves

**Input:** The planned NLP task and the feature it will produce.

**Output:** A written audit question, for example "what fraction of note text in this corpus is
copied from a prior note, and does copying inflate a diagnosis-count feature?"

**Rationale:** Documentation audits can measure many things. The useful ones are tied to a
decision. A prevalence estimate with no downstream use becomes a number in a slide.

**Branch point:** If the task is document classification, the audit focuses on label leakage from
copied outcome text. If the task is phenotyping, it focuses on stale facts and count inflation. If
the task is summarization, it focuses on omissions and duplicated content in the source.

### Step 2. Build longitudinal sequences

**Input:** Notes with patient, encounter, charttime, and author identifiers.

**Output:** For each patient, an ordered sequence of notes with intervals.

**Rationale:** Duplication is defined relative to a prior note. Without a correct sequence, the
comparison set is wrong. Notes must be ordered by charttime, with storetime retained for
late-entry detection.

**Branch point:** If a patient has notes from multiple sites, sequence them together but flag the
site. Cross-site copying is rare; cross-site template similarity is common and must not be read as
copying.

### Step 3. Establish authorship and role

**Input:** Author identifiers and role metadata.

**Output:** Author role per note, plus a same-author flag for each pair.

**Rationale:** Same-author duplication is the classic copy-forward pattern. Different-author
duplication within a care team is also common and often reflects handoff. Different-author
duplication across services is more likely to be template or quoted text.

**Branch point:** If author identifiers are missing or pseudonymized inconsistently, the
same-author analysis is not possible. Report the limitation and rely on time and template signals.

### Step 4. Normalize text for comparison only

**Input:** Raw note text.

**Output:** A comparison string per note, with the raw text retained.

**Rationale:** Case, whitespace, and Unicode differences inflate dissimilarity. Normalizing for
comparison makes duplication visible. The raw text must remain for extraction because normalization
can destroy clinically meaningful distinctions.

**Branch point:** Decide whether to remove section headers before comparison. Removing them reduces
false similarity from shared structure; keeping them detects copied sections including their
headers.

### Step 5. Choose the unit of comparison

**Input:** Comparison strings.

**Output:** A decision: document-level, section-level, paragraph-level, or sentence-level.

**Rationale:** Document-level similarity detects cloned notes but misses partial copying.
Section-level similarity is usually the best compromise because copy-forward often operates on
paragraphs or sections. Sentence-level detects the finest copying but produces many pairs and high
variance.

**Branch point:** If the corpus has reliable section segmentation, use section-level. If not, use
fixed windows of 100 to 200 tokens and accept boundary noise.

### Step 6. Build shingles and compute similarity

**Input:** Text units.

**Output:** A similarity matrix or, for large corpora, a candidate pair list.

**Rationale:** Shingle Jaccard is cheap, transparent, and robust to small edits. A five-token
shingle is a common default: long enough to be specific, short enough to survive punctuation
changes.

**Branch point:** The threshold determines the operating point.

| Threshold | Behaviour | Use |
|---:|---|---|
| 0.20 | High recall, many false positives | Exploratory scan |
| 0.30 | Balanced for paragraph-level copying | Default audit |
| 0.50 | High precision, misses paraphrased copying | Leakage detection |
| 0.80 | Near-identical only | Clone detection |

### Step 7. Include a negative control

**Input:** The pair list.

**Output:** A baseline similarity distribution from pairs that cannot be copy-forward: different
patients, same note type, same site.

**Rationale:** Template text creates similarity between unrelated notes. Without the control, the
audit attributes template similarity to copying. The control estimates the template floor.

**Branch point:** If the control median is above 0.30, the threshold is measuring template, not
copying. Raise the threshold or remove template regions before comparison.

### Step 8. Classify duplicated spans by mechanism

**Input:** Sampled duplicated pairs.

**Output:** A label per pair or span: copy-forward, template, quoted, auto-populated, coincidental.

**Rationale:** The mechanisms have different remedies and different implications for temporality.
Aggregating them produces a prevalence number that cannot guide action.

**Branch point:** Classification requires human review of a sample. There is no reliable automatic
classifier for mechanism, and claiming one is a red flag.

### Step 9. Estimate prevalence with a sampling design

**Input:** Flagged and unflagged pairs, plus human labels on a sample of each.

**Output:** Estimated duplication prevalence with a confidence interval.

**Rationale:** A threshold-based count is not a prevalence estimate. Both false positives (flagged
but not duplicated) and false negatives (duplicated but below threshold) must be estimated.

**Branch point:** Stratify the sample by note type and time gap. Duplication rates differ sharply
by genre, and an unstratified sample gives a prevalence that applies to no genre.

The estimator is:

```text
N_dup = N_flagged × P(dup | flagged) + N_unflagged × P(dup | unflagged)
prevalence = N_dup / N_pairs
```

### Step 10. Attribute duplication and direction

**Input:** Labeled pairs with author and time.

**Output:** Counts by same-author versus different-author, and by forward versus backward copying.

**Rationale:** Same-author forward copying is the dominant pattern and the one most likely to carry
stale facts. Backward copying, where an earlier note contains text from a later one, indicates a
timestamp or versioning error and must be investigated separately.

**Branch point:** If backward copying exceeds a small fraction, stop and audit the extract. It is
usually a storetime/charttime inversion, not a clinical behaviour.

### Step 11. Measure the effect on the target feature

**Input:** The feature definition and the labeled duplication.

**Output:** Feature performance before and after de-duplication, at the stated unit.

**Rationale:** This is the step that turns the audit into a decision. Duplication matters because
it changes a number the project will report.

**Branch point:** There are three comparison designs.

| Design | Method | Strength | Weakness |
|---|---|---|---|
| Ablation | Remove duplicated spans, recompute feature | Direct | Requires span-level duplication |
| Note-count versus evidence-count | Count distinct evidence instances, not documents | Simple | Needs an evidence definition |
| First-occurrence only | Keep earliest instance, drop later copies | Conservative | Loses legitimate re-documentation |

### Step 12. Design the mitigation

**Input:** The audit findings.

**Output:** A written policy with three parts: what is removed, what is flagged, and what is kept.

**Rationale:** De-duplication can destroy evidence. A copied consultant recommendation may be the
only place a diagnosis appears. Removing all duplicates removes that evidence. The policy must
distinguish repeated text that adds no information from repeated text that is the sole record.

**Branch point:** Mitigation options are not exclusive.

| Option | Action | Cost | Risk |
|---|---|---|---|
| Span de-duplication | Keep first occurrence of each duplicated span | Medium | Removes sole-source evidence |
| Document de-duplication | Drop near-identical notes | Low | Loses addenda and updates |
| Feature-level dedup | Count distinct evidence, not documents | Low | Requires feature-specific logic |
| Provenance features | Add "copied" flag as a feature | Low | Model may learn workflow |
| Template removal | Strip boilerplate before extraction | Medium | Removes filled slots |

The choice among these options should follow from the mechanism. If duplication is mostly
same-author copying of a stable problem list, feature-level dedup is sufficient and preserves the
note as a record. If duplication is cloned notes across a service, document de-duplication is
appropriate, but it must retain the earliest note and any addendum that changes the impression. If
the duplicated text is a template, span removal is the only option that leaves the patient-specific
slots intact.

A useful test before committing to a policy is the "lost positive" audit. Apply the policy to a
sample of 50 admissions known to be positive, and count how many lose their only affirmative
evidence. If the count is above 5 percent, the policy is too aggressive for a phenotype where recall
matters. If it is zero, the policy is probably safe but may be doing nothing.

There is also a reporting obligation. A report that describes de-duplication must state the policy,
the number of spans or documents removed, and the recall cost. "We removed duplicates" is not a
method; it is an assertion that a reviewer cannot evaluate.

### Step 12a. Decide whether to keep the duplicate as a feature

**Input:** The labeled duplication and the candidate feature set.

**Output:** A decision to include, exclude, or transform duplication-derived features.

**Rationale:** Duplication can carry signal. A note that copies a problem forward may indicate that
the problem is still active in the clinician's mind. A note that copies a plan without updating it
may indicate disengagement. Those are workflow signals, and they are legitimate features for some
tasks, such as predicting documentation burden or detecting copy-paste misuse. They are not
clinical facts, and a model that uses them for a clinical prediction is learning the workflow.

**Branch point:** Three policies are defensible, and they must be stated.

| Policy | Rule | Appropriate when | Not appropriate when |
|---|---|---|---|
| Remove | Drop copied spans from the feature space | The task is clinical phenotyping | The task is documentation quality |
| Flag | Keep the span, add a copied indicator | The workflow signal is plausibly causal | The indicator could proxy for the outcome |
| Ignore | Treat copied text as ordinary text | Duplication is low and unrelated | Duplication is high or correlated with the label |

The dangerous case is the flag policy. A "copied" indicator can correlate with length of stay,
service, and discharge disposition, and therefore with readmission. The model may then use copying
as a proxy for the outcome. Before including such a feature, test whether it predicts the outcome
after conditioning on the clinical variables. If it does, inspect the mechanism.

### Step 13. Re-evaluate and report the delta

**Input:** Mitigated corpus.

**Output:** Post-mitigation metrics with the same unit and the same reference standard.

**Rationale:** A mitigation that is not re-evaluated is a hope. The delta is the evidence that the
audit mattered.

**Branch point:** If the delta is small, report that. A documentation behaviour that does not
affect the feature is still worth monitoring, but it should not consume the project's quality
budget.

### Step 14. Detect template drift over time

**Input:** Section header frequencies and template signatures by month or quarter.

**Output:** A drift chart and a decision about stratification.

**Rationale:** Templates change with vendor upgrades, regulatory changes, and local projects. A
model trained across a template change learns a mixture.

**Branch point:** If drift is detected, choose one of three responses: stratify by era, restrict to
a stable era, or normalize the affected sections. Each changes the claim.

### Step 15. Set monitoring

**Input:** The baseline audit metrics.

**Output:** A monitoring plan with thresholds and owners.

**Rationale:** Documentation behaviour changes. Copy-forward rates rise under time pressure, and
templates change without notice. A model validated on a 30 percent duplication corpus may run on a
50 percent duplication corpus within a year.

**Branch point:** Monitor at three levels: corpus (duplication rate, note length), feature (positive
rate, evidence count per patient), and model (precision on a fixed audit sample). A change at any
level triggers review.

### Edge cases the audit must handle

**Negation scope.** Copied text can duplicate a negation into a new context. "No chest pain" copied
into a note where the patient now has chest pain is worse than duplication: it is a false negative
that looks authoritative. The audit should sample duplicated sentences containing negation cues and
check whether the copied negation contradicts fresh text in the same note.

**Hypotheticals.** Plan text is copied more often than any other section because plans change
slowly. "If fever develops, start ceftriaxone" copied for five days reads as a standing plan even
after the fever resolved. Count plan duplication separately.

**Family history.** Family history sections are highly templated and rarely change. Their
duplication rate is near 100 percent and carries almost no information about the current
encounter. Exclude them from duplication prevalence estimates or report them as a separate stratum.

**Copy-forward duplicates.** The audit is measuring the thing it is named for, but the edge case is
partial copying with edits. A sentence with one word changed is a near-duplicate that may reverse
meaning ("no" inserted, "not" removed). Shingle similarity above 0.8 can hide a meaning-changing
edit. Sample high-similarity pairs and diff them.

**Template boilerplate.** A template can be 90 percent of a short note. If the audit computes
document-level similarity, two unrelated patients on the same template look like duplicates. The
negative control is the only defence.

**Abbreviation ambiguity.** Copying preserves local abbreviations. A copied "MS" may have referred
to mental status in the original and be read as multiple sclerosis in the new context. The audit
should track abbreviation density in copied spans.

**Section boundaries.** Copy-forward often starts at a section header and ends at the next one. If
section detection fails, the copied span appears to start mid-sentence, and the boundary is lost.
Section-aware comparison improves both detection and interpretation.

**Unicode and OCR noise.** Scanned documents pasted into notes carry OCR errors. Their similarity
to the source is lower, so they evade threshold-based detection. Compute character error rate for
scanned content and treat it as a separate stratum.

## 6. Worked example

A quality team is building an admission-level feature for documented diabetes to support a
readmission model. The corpus is 400 admissions with 2,000 notes, five notes per admission on
average, from a single site. The team suspects copy-forward is inflating the feature and runs the
audit.

### Corpus and pair construction

Notes per admission range from 2 to 9. Total notes 2,000. Consecutive pairs within an admission:

```text
pairs = sum over admissions of (notes_in_admission - 1)
      = 2,000 - 400
      = 1,600
```

The team compares each note with the immediately preceding note, at section level, using five-token
shingles and Jaccard similarity. Sections are detected with a header lexicon; notes with no
detectable sections are compared as a single unit and flagged.

### Similarity distribution and threshold

| Statistic | Value |
|---|---:|
| Pairs compared | 1,600 |
| Median Jaccard | 0.18 |
| 75th percentile | 0.34 |
| 90th percentile | 0.52 |
| Pairs at or above 0.30 | 512 |

At a threshold of 0.30, 512 pairs are flagged, or 32.0 percent of pairs.

The negative control uses 400 randomly selected pairs of notes from different patients, same note
type, same site, same quarter. Median Jaccard is 0.11 and the 90th percentile is 0.24. The control
confirms that 0.30 is above the template floor for this corpus. In a more heavily templated
specialty, the control median could exceed 0.30, and the threshold would need to rise.

### Human review and prevalence estimate

The team samples 200 flagged pairs at random and 200 unflagged pairs at random. Two reviewers label
each pair with a mechanism, with adjudication for disagreements.

Flagged sample (n = 200):

| Label | Count |
|---|---:|
| Copy-forward | 160 |
| Template only | 24 |
| Coincidental similarity | 16 |
| Total | 200 |

Unflagged sample (n = 200):

| Label | Count |
|---|---:|
| Copy-forward | 4 |
| Template only | 3 |
| Coincidental | 193 |
| Total | 200 |

Estimated duplicates among flagged pairs = 512 × (160 / 200) = 512 × 0.800 = 409.6.
Estimated duplicates among unflagged pairs = 1,088 × (4 / 200) = 1,088 × 0.020 = 21.8.
Estimated total duplicated pairs = 409.6 + 21.76 = 431.36.
Estimated prevalence = 431.36 / 1,600 = 0.270, or 27.0 percent of consecutive pairs.

Detection recall at the 0.30 threshold = 409.6 / 431.36 = 0.950. Precision = 0.800.

The 95 percent confidence interval for the flagged-sample proportion, using the Wilson interval with
n = 200 and p = 0.80, is approximately 0.738 to 0.851. The interval is wide because the sample is
small; a validation report should state it rather than report 0.800 as a point estimate.

### Mechanism and authorship

| Mechanism | Estimated pairs | Share |
|---|---:|---:|
| Copy-forward, same author | 246 | 57.0% |
| Copy-forward, different author | 164 | 38.0% |
| Copy-forward, direction unclear | 21 | 5.0% |
| Total copy-forward | 431 | 100% |

Same-author copying dominates, consistent with the published pattern that most duplicated text
comes from the same author (54.1 percent in Steinkamp et al.). The different-author share is higher
here than in that study, which the team attributes to a shared handoff template. That attribution
is an inference and is labeled as such in the report.

### Effect on the diabetes feature

The feature rule is "diabetes mentioned in at least two documents in the admission," intended to
reduce false positives from a single ambiguous mention. Reference standard: chart review of all 400
admissions by two clinicians, with adjudication.

| Reference standard | Admissions |
|---|---:|
| Diabetes present | 120 |
| Diabetes absent | 280 |
| Total | 400 |

Before de-duplication:

| | Reference positive | Reference negative |
|---|---:|---:|
| Feature positive | 114 | 96 |
| Feature negative | 6 | 184 |

Sensitivity = 114 / 120 = 0.950.
Specificity = 184 / 280 = 0.657.
PPV = 114 / 210 = 0.543.
Accuracy = (114 + 184) / 400 = 0.745.

After collapsing duplicated text so that a mention counts once per distinct evidence instance:

| | Reference positive | Reference negative |
|---|---:|---:|
| Feature positive | 108 | 32 |
| Feature negative | 12 | 248 |

Sensitivity = 108 / 120 = 0.900.
Specificity = 248 / 280 = 0.886.
PPV = 108 / 140 = 0.771.
Accuracy = (108 + 248) / 400 = 0.890.

The feature loses 6 true positives and 64 false positives. PPV rises by 22.8 points. The 6 lost
true positives are admissions where the only two mentions were copies of the same sentence; the
reference standard says diabetes is present, so the loss is a recall cost of the de-duplication
policy, not a modeling error.

The arithmetic of the false-positive reduction is worth stating: 64 of the original 96 false
positives, or 66.7 percent, depended on duplicated text. Those admissions had one genuine mention,
copied into two or more notes, which satisfied the "at least two documents" rule. The rule was
measuring note-writing behaviour, not evidence.

### Two diagnostics that reveal a broken pipeline

**Diagnostic 1: similarity versus time gap.** Plot median Jaccard against the interval between
notes. In a healthy corpus, similarity falls as the gap grows: adjacent notes share context, notes
weeks apart do not. If similarity stays flat or rises with the gap, the pipeline is measuring
template, not copying.

In this corpus:

| Gap between notes | Pairs | Median Jaccard |
|---|---:|---:|
| 0 to 1 day | 900 | 0.34 |
| 2 to 3 days | 420 | 0.27 |
| 4 to 7 days | 200 | 0.19 |
| More than 7 days | 80 | 0.14 |

The decline is the expected shape. A flat curve would have sent the team back to Step 7.

**Diagnostic 2: negative control versus corpus.** The control median is 0.11 against a corpus
median of 0.18. The gap is 0.07, which is small enough that a 0.30 threshold is doing real work but
large enough that the team should check the control by note type.

| Note type | Control median | Corpus median | Gap |
|---|---:|---:|---:|
| Progress note | 0.13 | 0.21 | 0.08 |
| Discharge summary | 0.09 | 0.16 | 0.07 |
| Consultation note | 0.19 | 0.24 | 0.05 |

Consultation notes have the highest template floor. A threshold applied uniformly would flag more
consultation pairs as duplicates than the mechanism justifies. The team stratifies the threshold by
note type in the final report.

### Vary one assumption: raise the threshold to 0.50

Suppose the team chooses 0.50 to prioritize precision. The flagged set falls to 288 pairs.

Review of 200 sampled flagged pairs finds 184 copy-forward, or 0.920. Review of 200 sampled
unflagged pairs finds 10 copy-forward, or 0.050.

Estimated duplicates = 288 × 0.920 + 1,312 × 0.050 = 264.96 + 65.6 = 330.6.
Prevalence = 330.6 / 1,600 = 0.207, or 20.7 percent.
Recall = 264.96 / 330.6 = 0.801. Precision = 0.920.

The prevalence estimate falls from 27.0 to 20.7 percent, and recall falls from 0.950 to 0.801. The
higher threshold does not measure a different corpus; it measures the same corpus less completely.

Now propagate to the feature. The de-duplication step misses 19.9 percent of duplicates. If missed
duplicates distribute proportionally across the admissions that generated false positives, the 64
duplicate-driven false positives fall to 64 × 0.801 ≈ 51 removed, leaving 13. The feature result
becomes:

| | Reference positive | Reference negative |
|---|---:|---:|
| Feature positive | 110 | 45 |
| Feature negative | 10 | 235 |

Sensitivity = 110 / 120 = 0.917.
Specificity = 235 / 280 = 0.839.
PPV = 110 / 155 = 0.710.
Accuracy = (110 + 235) / 400 = 0.863.

PPV improvement over the naive feature is 0.710 − 0.543 = 16.7 points, not 22.8. The conclusion
changes in magnitude but not in direction: de-duplication helps, and a conservative threshold
captures most of the benefit. The report should present both operating points rather than a single
number, because the choice of threshold is a policy decision with a measurable cost.

The most useful summary for a clinical audience is not the prevalence. It is the sentence: "the
original rule counted notes, and 64 of its 96 false positives were one mention copied into two
notes; after counting evidence instead of documents, PPV rose from 0.54 to 0.77 while sensitivity
fell from 0.95 to 0.90." That sentence states the mechanism, the unit, the cost, and the benefit.
A prevalence estimate without it is a number that does not change a decision.

## 7. Code, query, or calculation

The following code computes shingle Jaccard similarity, applies a threshold, and reports a
duplication prevalence estimate from a labeled sample. It uses synthetic notes and only the
standard library.

```python
import re
from itertools import islice

NOTES = {
    ("P001", "n1"): "Patient with diabetes and hypertension. A1c 8.2. Continue metformin. "
                     "No chest pain. Assessment: diabetes uncontrolled.",
    ("P001", "n2"): "Patient with diabetes and hypertension. A1c 8.2. Continue metformin. "
                     "No chest pain. Assessment: diabetes uncontrolled. Plan: titrate metformin.",
    ("P002", "n1"): "Patient with diabetes and hypertension. A1c 8.2. Continue metformin. "
                     "No chest pain. Assessment: diabetes uncontrolled.",
    ("P002", "n2"): "Patient with COPD. Increased wheeze. Start prednisone. "
                     "Assessment: COPD exacerbation.",
}

N = 5


def normalize(text):
    text = text.lower()
    return re.sub(r"[^a-z0-9 ]", " ", text)


def shingles(text, n=N):
    tokens = normalize(text).split()
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def containment(a, b):
    """Share of the shorter note's shingles found in the longer note."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


pairs = [
    ("P001", "n1", "P001", "n2"),
    ("P002", "n1", "P002", "n2"),
    ("P001", "n1", "P002", "n1"),   # negative control: same text, different patients
]

for p1, n1, p2, n2 in pairs:
    s1, s2 = shingles(NOTES[(p1, n1)]), shingles(NOTES[(p2, n2)])
    print(f"{p1}/{n1} vs {p2}/{n2}: jaccard={jaccard(s1, s2):.3f} "
          f"containment={containment(s1, s2):.3f}")

threshold = 0.30
flagged = []
for p1, n1, p2, n2 in pairs[:2]:
    s1, s2 = shingles(NOTES[(p1, n1)]), shingles(NOTES[(p2, n2)])
    if jaccard(s1, s2) >= threshold:
        flagged.append((p1, n1, n2, round(jaccard(s1, s2), 3)))
print("flagged:", flagged)

n_flagged, p_dup_flagged = 512, 160 / 200
n_unflagged, p_dup_unflagged = 1088, 4 / 200
est = n_flagged * p_dup_flagged + n_unflagged * p_dup_unflagged
print("estimated duplicate pairs:", round(est, 1))
print("prevalence:", round(est / (n_flagged + n_unflagged), 4))
print("recall at threshold:", round(n_flagged * p_dup_flagged / est, 3))
```

Expected output:

```text
P001/n1 vs P001/n2: jaccard=0.800 containment=1.000
P002/n1 vs P002/n2: jaccard=0.000 containment=0.000
P001/n1 vs P002/n1: jaccard=1.000 containment=1.000
flagged: [('P001', 'n1', 'n2', 0.8)]
estimated duplicate pairs: 431.4
prevalence: 0.2696
recall at threshold: 0.95
```

The third line is the negative control: identical text under two patient identifiers yields
Jaccard 1.000. Any audit that does not separate that pair from a within-patient pair will count
template or duplicated documentation as clinical copying. The flag list contains only the
within-patient pair above threshold.

A second block estimates the feature effect with and without de-duplication:

```python
def metrics(tp, fp, fn, tn):
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    acc = (tp + tn) / (tp + fp + fn + tn)
    return sens, spec, ppv, acc


naive = metrics(114, 96, 6, 184)
dedup = metrics(108, 32, 12, 248)
thresh50 = metrics(110, 45, 10, 235)

for name, m in [("naive", naive), ("dedup", dedup), ("threshold 0.50", thresh50)]:
    print(f"{name}: sens={m[0]:.3f} spec={m[1]:.3f} ppv={m[2]:.3f} acc={m[3]:.3f}")
print("ppv gain dedup:", round(dedup[2] - naive[2], 3))
print("ppv gain threshold 0.50:", round(thresh50[2] - naive[2], 3))
```

Expected output:

```text
naive: sens=0.950 spec=0.657 ppv=0.543 acc=0.745
dedup: sens=0.900 spec=0.886 ppv=0.771 acc=0.890
threshold 0.50: sens=0.917 spec=0.839 ppv=0.710 acc=0.863
ppv gain dedup: 0.229
ppv gain threshold 0.50: 0.167
```

The code makes the policy trade-off explicit. Raising the threshold improves precision of the
duplicate detector but lowers recall, and the feature-level gain falls from 22.8 to 16.7 points.

## 8. Data, standards, and regulatory context

Documentation behaviour sits at the intersection of billing regulation, privacy law, and health IT
policy. The NLP team does not need to practice law, but it needs to know which constraints shape
the text.

**CMS evaluation and management documentation.** The Centers for Medicare and Medicaid Services
(CMS) set documentation requirements for evaluation and management (E/M) services. Revisions
effective January 2021 for office and outpatient E/M visits allowed clinicians to document based on
medical decision making or total time, and reduced some history and examination documentation
requirements. The effect on text is indirect but real: when a documentation element is no longer
required for payment, its presence becomes a weaker signal of clinical attention, and its absence
becomes less meaningful. A phenotype that relies on a comprehensive review of systems should check
whether the corpus spans a documentation regime change.

What the regulation constrains: what must be documented to support a billed service. What it does
not constrain: the clinical completeness of the note, the truth of the copied text, or the
suitability of the note for NLP.

**The note as a legal record.** In most jurisdictions the signed note is part of the designated
record set and may be produced in litigation or audit. Addenda and amendments are retained. A
pipeline that reads only the final version may read a corrected impression and attribute it to the
original time. A pipeline that reads all versions may double-count the same clinical event. The
resolution is a version table with explicit rules, as described in Topic 01.

**Information blocking and patient access.** The 21st Century Cures Act and ONC's information
blocking regulations require that patients have access to their electronic health information,
including clinical notes, without unreasonable barriers. Open notes change the audience for the
note. Clinicians may soften language, avoid certain terms, or move sensitive detail to a
non-note channel. The NLP consequence is that some concepts become less explicit over time, which
can look like a decline in prevalence rather than a change in writing style.

**HIPAA.** Notes contain PHI by default. Documentation behaviour interacts with de-identification
in two ways. First, copy-forward spreads identifiers across notes, so a missed identifier in one
note propagates to every note that copied it. Second, template text can contain default values or
example identifiers that are not patient data but look like it. Topic 08 covers the mechanics.

**21st Century Cures and certification.** Health IT certification criteria include audit log and
provenance expectations, and ONC's HTI-1 rule adds transparency requirements for predictive
decision support. Those requirements apply to the software, not to a research corpus. A team that
cites certification as evidence that its corpus is well-described is making an unsupported
inference.

**Institutional policy.** Many organizations have copy-paste policies, note-blinding rules for
sensitive specialties, and documentation templates maintained by a clinical informatics committee.
These policies are the closest thing to a ground truth about intended behaviour. An audit that
ignores them may interpret a policy-driven template as clinician copying.

**The Joint Commission and record standards.** Accreditation standards require that the medical
record be accurate, complete, and timely, and that entries be authenticated. They do not prescribe
a duplication rate. The gap is important: an organization can be fully accredited and still have
half its text duplicated, because the standards address authenticity of the entry, not novelty of
the content. An NLP team should not assume that accreditation implies a clean corpus.

**State and payer audit rules.** Payers audit documentation to support billed claims, and recovery
audit contractor reviews can target cloned notes. A note that is identical to a prior note may be
challenged as not supporting the service billed. This creates an institutional incentive to reduce
obvious cloning, which in turn changes the text distribution. The effect is a form of
observation bias: the notes most likely to be audited are the ones most likely to have been
cleaned, so an audit sample drawn from them understates duplication.

**Research data use agreements.** A data use agreement may prohibit re-identification and
redistribution but say nothing about duplication. That silence means the NLP team owns the
documentation-behaviour analysis. It cannot rely on the data provider to have cleaned or flagged
duplicated text. The MIMIC-IV-Note documentation, for example, describes de-identification and
provenance fields but does not publish a duplication rate for the corpus.

**International variation.** Documentation requirements differ across health systems and countries.
In systems where notes are shorter and more structured, duplication may be lower and template
similarity higher. In systems where notes are the primary communication channel, duplication may be
higher. A prevalence estimate from a US academic center should not be applied to a national health
service corpus without measurement. This is an inference from structural differences, not a
published comparison.

> **Note:** Documentation requirements are a moving target. A corpus that spans a CMS rule change, a
> template redesign, or an open-notes rollout contains at least two documentation regimes. Report
> the regimes and the dates, not just the corpus size.

## 9. Industry practice and case evidence

**Duplication prevalence at scale.** Steinkamp, Kantrowitz, and Airan-Javia (JAMA Network Open,
2022) analyzed 104.5 million notes from 1.96 million patients at Penn Medicine, 2015 to 2020, using
a 10-gram sliding-window method. They reported that 50.1 percent of clinical text was duplicated
from prior notes, that the share rose from 33.0 percent in 2015 to 54.2 percent in 2020, that 54.1
percent of duplicated text came from the same author, and that physician notes contained 30 to 70
percent duplicate content. These are the study's reported figures on its corpus and method; they
are not a universal prevalence.

**Provenance tracking at the character level.** Wang, Khanna, and Najafi (JAMA Internal Medicine,
2017) tracked text provenance in 23,630 notes from 460 clinicians at UCSF. They reported that 18
percent of text was manually typed, 46 percent was copied, and 36 percent was imported, with
residents copying most (51.4 percent). The categories differ from Steinkamp's, so the two studies
should not be averaged.

**Copy-paste practice surveys.** Tsou et al. (Applied Clinical Informatics, 2017) systematically
reviewed 51 studies and reported that 66 to 90 percent of clinicians routinely copy and paste, with
retrospective chart studies ranging from 7 to 82 percent depending on definitions and context. The
wide range is itself the finding: prevalence depends on what counts as copying.

**Safety signal.** Singh et al. (2013) found that copy-paste contributed to 2.6 percent of
diagnostic errors in primary care that led to unplanned care. The figure is small in absolute terms
and large in clinical consequence, and it is the strongest published link between duplication and
patient harm found in this review.

**Template and note bloat.** The same literature documents "note bloat": note length grows while
informational content per note falls. The mechanism is templates plus copying plus defensive
documentation. The NLP consequence is that token counts and note counts both become poor proxies
for information. Inference: a corpus described by token count overstates its evidence base, and a
model trained on token-weighted samples overweights the most templated specialties.

**Vendor and health system practice.** Many EHR vendors provide copy-forward controls, including
copy-previous-note buttons, smart phrases, and note comparison views. Some organizations monitor
copy-paste through note-similarity dashboards and feed metrics into provider performance review.
Published evaluations of these controls are sparse. Inference: the controls change behaviour at the
margin but do not eliminate duplication, because duplication is an adaptive response to
documentation burden.

**What the evidence does not establish.** There is no published estimate of duplication prevalence
that generalizes across sites, specialties, and note types. The two large studies come from academic
medical centers with mature EHRs and may not represent community hospitals or non-US settings. Any
project should measure its own corpus rather than import a prevalence estimate.

**Ambient documentation as a counter-trend.** Ambient scribes and automated draft notes change the
production process. If a model drafts the note and the clinician edits it, the final text may
contain fewer copied passages and more model-generated prose. The duplication rate may fall while a
new provenance problem appears: text generated from a conversation, with its own error profile, and
possibly copied between notes by the drafting system. Published evaluations of ambient scribes focus
on clinician time and note quality; duplication in the resulting corpus is not commonly reported.
Inference: the duplication audit will remain necessary, but the mechanism may shift from
clinician copy-forward to system reuse.

**A note on vendor marketing claims.** Vendors of documentation tools sometimes report
"notes per hour" or "time saved" rather than corpus-level duplication or error rates. Those metrics
do not answer the NLP question. A team evaluating a tool for a text pipeline should ask for the
output corpus characteristics: note length distribution, template usage, duplication rate, and
whether generated text is labelled as generated. Absent those, the marketing claim is not evidence
about the text.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Counting documents as evidence | Feature fires on repeated copies | Copy-forward duplicates a single mention | Inflated prevalence and PPV loss | Evidence-level counting; de-duplication |
| Treating template as clinical text | Phrase appears in nearly every note | Pre-populated normal findings | False negatives, spurious features | Boilerplate detection; template flag |
| Ignoring stale temporality | Diagnosis date equals note date for years | Text copied without updating | Wrong event time and cohort | First-occurrence dating; explicit temporal parsing |
| Threshold without control | Similarity threshold below template floor | Template inflates pair similarity | Prevalence overestimated | Negative control; stratified threshold |
| De-duplication removes sole evidence | Recall drops sharply after mitigation | Only record of a fact was a copy | Missed patients | Keep first occurrence; audit lost positives |
| Merging addenda silently | Final impression differs from original | Addendum contradicts base note | Attribution error, leakage | Version table with parent link |
| Ignoring template drift | Header frequency changes by quarter | Vendor or local template change | Model trained on mixed regimes | Drift chart; era stratification |
| Assuming copying is always bad | Removing legitimate re-documentation | Handoff and consultant notes | Loss of context | Mechanism-specific policy |
| Same-author assumption | Different-author duplication unmeasured | Team templates and handoffs | Misattributed mechanism | Author-stratified analysis |
| Note length as information proxy | Long notes with low unique content | Bloat | Sample weighting bias | Unique-content measures |
| Backward copying ignored | Earlier note contains later text | Timestamp or version error | Temporal leakage | Direction audit; fix extract |
| Open-notes effect unmeasured | Concept prevalence falls after rollout | Clinicians soften language | Apparent decline is writing change | Era stratification; clinician review |
| Policy treated as ground truth | Policy says no copy-paste, data says otherwise | Policy vs practice gap | False reassurance | Measure behaviour, not policy |
| Duplication metric reported without CI | Single prevalence number | Sampling variance ignored | Overconfident claim | Wilson interval; state n |
| Scribe or generated text unlabelled | Sudden style change, no author signal | Ambient documentation output mixed into notes | Provenance unknown | Request generation flags; stratify by era |
| Duplication measured only within note type | Cross-note-type copying unmeasured | Handoff text pasted between genres | Underestimated prevalence | Compare across genres within encounter |
| Template removal deletes filled slots | Extraction recall falls after cleaning | Slot and template not separated | Lost patient evidence | Slot-aware removal; validate |
| Copy rate used as quality metric | Clinicians penalized for necessary re-documentation | Duplication conflated with poor quality | Gaming and workarounds | Use mechanism-specific metrics |

A cluster of red flags appears together when a team has not audited documentation behaviour at all.
The corpus description says "notes" without a duplication rate. The feature counts documents rather
than evidence. The validation sample is drawn from the same duplicated corpus used for development.
The model's top feature is note length. Any one of these is a prompt to run the audit; three
together mean the reported performance is not interpretable.

A second cluster appears when the audit is done but the mitigation is not evaluated. The report
states a duplication prevalence and then presents metrics on the raw corpus. The prevalence number
becomes decoration. The test is simple: does the report contain a before-and-after table at the
same unit with the same reference standard? If not, the audit did not change the analysis.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Prior notes are the correct comparison set | Defines copy-forward | Text imported from outside the corpus | Check for quoted external sources; review sample |
| Five-token shingles capture copying | Cheap and stable | Heavy paraphrase or reordering | Compare with sentence embeddings on a sample |
| Jaccard threshold generalizes across note types | Simplifies reporting | Template density differs by type | Stratify threshold; report per type |
| Human labels of mechanism are reliable | Prevalence estimator depends on them | Ambiguous cases | Double-annotate; report agreement |
| Same author means same clinician | Supports attribution | Shared accounts or scribes | Check author identifier uniqueness |
| Charttime ordering is correct | Direction analysis depends on it | Back-dating or interface errors | Detect backward copying; audit extract |
| Duplicated text is redundant | Justifies removal | Copy is the sole record of a fact | Audit lost positives after mitigation |
| Template regions are identifiable | Enables removal | Free-text notes with no template | Measure detection rate; sample |
| Feature effect generalizes beyond the sample | Supports the conclusion | Sample selection bias | Stratify by note type and service |
| Documentation behaviour is stable over the study period | Pooling is valid | Policy or template change | Drift chart by quarter |
| De-identification is unaffected by copying | Privacy assumption | Copied identifier propagates | Residual PHI audit on duplicated spans |

The most fragile assumption is that duplicated text is redundant. In the worked example, 6 of 120
true positives were lost after de-duplication because the only two mentions were copies of one
sentence. The reference standard still called those admissions diabetic. A policy that removes
duplicates without preserving first occurrences would have produced a larger loss, and a policy that
removes all but the first occurrence in the entire record, rather than per encounter, would lose
legitimate re-documentation across years.

The second fragile assumption is threshold transportability. A threshold calibrated on progress
notes may be too low for consultation notes and too high for discharge summaries. The negative
control by note type in the worked example showed a template floor of 0.19 for consultation notes,
which is close to the 0.30 threshold. A single threshold in that setting would misclassify template
similarity as copying. The fix is stratification, and the cost is a more complex report.

A third limit is that duplication measurement says nothing about truth. A copied statement may be
correct and a freshly written statement may be wrong. The audit measures provenance, not accuracy.
A team that treats low duplication as a quality signal is making an inference the method does not
support.

## 12. Exercises

**Exercise 2.1 — Mechanism classification (application, 45 minutes).**
Deliverable: take ten note pairs you construct or obtain from a synthetic corpus. Label each as
copy-forward, template, quoted, auto-populated, or coincidental, and give the evidence for the
label.
Acceptance criteria: ten pairs; each label justified by a textual or metadata feature; at least one
pair where the label is ambiguous and you state why.

**Exercise 2.2 — Threshold sensitivity (calculation, 45 minutes).**
Deliverable: using the worked example, compute prevalence, recall, and feature PPV at thresholds
0.30, 0.50, and 0.70. Assume the labeled-sample proportions scale linearly and state that
assumption.
Acceptance criteria: all arithmetic shown; a table of three operating points; one paragraph on
which threshold you would deploy and why.

**Exercise 2.3 — Negative control design (design, 40 minutes).**
Deliverable: a sampling plan for a negative control in a corpus with three note types and four
sites. Specify strata, sample sizes, and the statistic you will compare.
Acceptance criteria: at least six strata; sample size per stratum justified; the comparison
statistic named.

**Exercise 2.4 — Feature impact (application, 60 minutes).**
Deliverable: define a text-derived feature of your choice, state its unit, and compute its
performance on a small synthetic set before and after de-duplication. Report sensitivity,
specificity, and PPV.
Acceptance criteria: unit stated; confusion matrices shown; the de-duplication policy described in
one sentence.

**Exercise 2.5 — Template drift (critique, 45 minutes).**
Deliverable: a one-page analysis of how a mid-study template change would affect a phenotyping
model. Name three mechanisms and the stratification you would apply.
Acceptance criteria: three mechanisms; stratification specified by variable and era; a statement of
what the stratified analysis cannot recover.

**Exercise 2.6 — Policy critique (critique, 30 minutes).**
Deliverable: critique a hypothetical organizational policy that bans copy-paste outright. Give two
reasons the policy may fail and one alternative.
Acceptance criteria: two mechanisms tied to documentation burden; one alternative with a measurement
plan.

**Exercise 2.7 — Addendum handling (design, 45 minutes).**
Deliverable: write the version-handling rule for a corpus with base notes, addenda, and amendments.
State how each is used for a document-level task and a patient-level task.
Acceptance criteria: rule covers all three version types; different rules for the two task levels;
an example of a case where the rules diverge.

**Exercise 2.8 — Report section (application, 60 minutes).**
Deliverable: write the documentation-behaviour section of a validation report, 500 words, with a
prevalence estimate, a confidence interval, a feature-impact table, and a mitigation policy.
Acceptance criteria: all four elements; numbers traceable to a method; limitations stated.

## 13. Self-check answers

**2.2 (threshold sensitivity).** At 0.30: flagged 512, precision 0.800, unflagged rate 0.020,
estimated duplicates 431.4, prevalence 0.270, recall 0.950. At 0.50: flagged 288, precision 0.920,
unflagged rate 0.050, estimated 330.6, prevalence 0.207, recall 0.801. At 0.70: assume flagged 140,
precision 0.975, unflagged rate 0.085. Estimated = 136.5 + 1,460 × 0.085 = 136.5 + 124.1 = 260.6;
prevalence 0.163; recall 0.524. The pattern is monotonic: higher thresholds buy precision and lose
recall, and the prevalence estimate falls because more duplicates sit below the threshold. The
feature PPV gain falls correspondingly, from 22.8 points at 0.30 to roughly 12 points at 0.70 under
proportional scaling. Deploy 0.30 for a phenotype where missing evidence is costly, and 0.50 for a
leakage check where false alarms are costly.

**2.1 (mechanism classification) rubric.** Full credit: labels use textual evidence (identical
sentences, template headers, quotation marks, author and time), and the ambiguous case is analyzed
rather than forced. Half credit: correct labels without evidence. Zero: all pairs labeled
"duplicate."

**2.3 (negative control) rubric.** Full credit: cross-patient pairs within note type and site, at
least 30 pairs per stratum, median and 90th percentile reported, and a decision rule for adjusting
the threshold. Half credit: strata named without sample sizes.

**2.4 (feature impact) rubric.** Full credit: unit stated, confusion matrices for both conditions,
PPV computed, and the de-duplication policy specified. Half credit: metrics without the policy.

**2.5 (template drift).** Mechanisms: (a) vocabulary shift makes previously learned phrases
uninformative; (b) section structure change breaks segmentation and assertion scope; (c) new
pre-populated defaults create false negatives. Stratify by era and note type, and test for
interaction. The stratified analysis cannot recover the counterfactual performance in the old
regime for patients recorded only in the new one, nor can it distinguish a template effect from a
concurrent change in case mix.

**2.6 (policy critique).** Reasons the ban may fail: duplication is an adaptive response to time
pressure, so a ban shifts copying to paraphrase or to unreviewed import; and the note's legal and
communication functions reward completeness, which copying provides cheaply. Alternative: measure
duplication, require review of copied content at sign-off, and redesign templates so that carrying
forward a problem requires an explicit status update.

**2.7 (addendum handling) rubric.** Full credit: base note used as primary text; addendum used for
document-level tasks only when it changes the impression; amendment replaces the base with the prior
version retained; patient-level tasks use the latest signed state plus any addendum that adds a new
fact, with a flag. Half credit: rules for two of three version types.

**2.8 (report section) rubric.** Full credit: prevalence with CI, mechanism table, before-and-after
feature table at one unit, mitigation policy, and a limitation. Half credit: three of five elements.

## 14. Interview questions

**1. "What is copy-forward and why does it matter for NLP?"**
Strong answer: defines reuse of prior text, explains that it decouples mention from time, and gives
a concrete consequence such as count inflation or stale diagnosis. Weak answer: "duplicated text,"
with no consequence.

**2. "How would you measure duplication in a corpus?"**
Strong answer: shingles, Jaccard, threshold, negative control, stratified sample, prevalence with a
confidence interval. Weak answer: "compare notes with a similarity score."

**3. "How do you distinguish copying from template boilerplate?"**
Strong answer: negative control across patients, time-gap analysis, and human review of a sample.
Weak answer: "template is shorter."

**4. "A feature fires on three notes per patient. Is that stronger evidence than one note?"**
Strong answer: only if the mentions are independent; copy-forward makes them one observation.
Weak answer: assumes more notes means more evidence.

**5. "What is the risk of removing all duplicated text?"**
Strong answer: the copy may be the only record of a fact; recall falls; first-occurrence or
evidence-level counting is safer. Weak answer: "none, duplicates are noise."

**6. "How would you detect a template change mid-study?"**
Strong answer: section header frequencies, phrase frequencies, note length distributions by month,
and a stratified comparison. Weak answer: reads the release notes.

**7. "What does the 50 percent duplication figure from the literature mean for your project?"**
Strong answer: it is a site-specific estimate from one health system and one method; measure your
own corpus. Weak answer: applies it as a prior without measurement.

**8. "How does open notes change documentation behaviour?"**
Strong answer: the patient becomes an audience; language softens; sensitive detail may move; some
concept prevalence falls without a clinical change. Weak answer: "it doesn't affect the text."

**9. "What is note bloat and how does it affect sampling?"**
Strong answer: length grows faster than information; token-weighted sampling overweights templated
specialties; use unique content. Weak answer: "long notes are bad."

**10. "How would you report duplication in a validation report?"**
Strong answer: prevalence with CI, method and threshold, negative control, mechanism breakdown,
feature impact before and after, and a limitation. Weak answer: a single percentage.

**11. "Why is a backward copy a red flag?"**
Strong answer: it implies a timestamp or version error, which threatens temporal validity. Weak
answer: treats it as ordinary copying.

## 15. Key terms

- **Addendum** — a later addition to a signed note.
- **Amendment** — a correction that changes the signed record.
- **Attestation** — a statement of review and acceptance by the author.
- **Auto-population** — system insertion of structured data into narrative text.
- **Back-dating** — assigning a charttime earlier than the writing time.
- **Boilerplate** — fixed text repeated by design.
- **Carry-forward** — automatic population of prior values.
- **Clone** — a note duplicating a prior note with minimal change.
- **Containment** — intersection over the size of one shingle set.
- **Copy-forward** — reuse of text from a prior note about the same patient.
- **Copy-paste** — manual duplication via clipboard.
- **Documentation burden** — time and cognitive cost of note writing.
- **E/M** — evaluation and management, a CMS billing category.
- **First-occurrence policy** — keep only the earliest instance of repeated text.
- **Information decay** — rate at which a copied fact stops being true.
- **Jaccard similarity** — intersection over union of two sets.
- **Late entry** — a note charted after the clinical event.
- **Macro** — a stored text expansion, often personal.
- **Near-duplicate** — a pair above a similarity threshold but not identical.
- **Note bloat** — length inflation from duplication and templating.
- **Open notes** — patient access to clinical notes.
- **Problem list carryover** — unreconciled problem entries persisting across encounters.
- **Provenance** — origin of a text span.
- **Redundancy** — repeated meaning in different words.
- **Shingle** — a contiguous sequence of n tokens.
- **Smart phrase** — a user-triggered expansion macro.
- **Template** — predefined structure with fixed and variable regions.
- **Template drift** — change in template over time.
- **Wilson interval** — a binomial confidence interval robust at extreme proportions.

## 16. Sources

All URLs accessed 25 October 2026.

1. Steinkamp J, Kantrowitz JJ, Airan-Javia S. "Prevalence and sources of duplicate information in
   the electronic medical record." JAMA Netw Open. 2022;5(9):e2233348.
   https://doi.org/10.1001/jamanetworkopen.2022.33348
2. Wang MD, Khanna R, Najafi N. "Characterizing the source of text in electronic health record
   progress notes." JAMA Intern Med. 2017;177(8):1212-1213.
   https://doi.org/10.1001/jamainternmed.2017.1548
3. Tsou AY, Lehmann CU, Michel J, Solomon R, Possanza L, Gandhi T. "Safe practices for copy and
   paste in the EHR: systematic review, recommendations, and novel model for health IT
   collaboration." Appl Clin Inform. 2017;8(1):12-34.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC5373750/
4. Singh H, Giardina TD, Meyer AND, Forjuoh SN, Reis MD, Thomas EJ. "Types and origins of
   diagnostic errors in primary care settings." JAMA Intern Med. 2013;173(6):418-424.
5. Vawdrey DK, et al. "A practical approach for monitoring the use of copy-paste in clinical
   notes." AMIA Annu Symp Proc. 2022.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC8861699/
6. Centers for Medicare and Medicaid Services. "Evaluation and Management Services Guide."
   https://www.cms.gov/outreach-and-education/medicare-learning-network-mln/mlnproducts/downloads/eval-mgmt-serv-guide-icn006764.pdf
7. Office of the National Coordinator for Health Information Technology. "Information Blocking."
   https://www.healthit.gov/topic/information-blocking
8. 45 CFR 164.514, de-identification standard.
   https://www.law.cornell.edu/cfr/text/45/164.514
9. Health Level Seven. FHIR R4 Provenance resource. https://hl7.org/fhir/R4/provenance.html
10. Uzuner O, South BR, Shen S, DuVall SL. "2010 i2b2/VA challenge on concepts, assertions, and
    relations in clinical text." J Am Med Inform Assoc. 2011;18(5):552-556.
    https://doi.org/10.1136/amiajnl-2011-000203
11. Velupillai S, Suominen H, Liakata M, et al. "Using clinical Natural Language Processing for
    health outcomes research." J Biomed Inform. 2018;88:11-19.
    https://doi.org/10.1016/j.jbi.2018.10.005

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`01_clinical-text-sources.md`](01_clinical-text-sources.md)
- Next topic: [`03_task-taxonomy.md`](03_task-taxonomy.md)
- Annotation and agreement: [`06_annotation.md`](06_annotation.md)
- Leakage from copied outcomes: [`07_evaluation.md`](07_evaluation.md)
- De-identification and identifier propagation: [`08_deidentification.md`](08_deidentification.md)
- Module survey: [`../../09_clinical_nlp.md`](../../course/09_clinical_nlp.md)
- Labs: [`../../labs/`](../../labs/)
