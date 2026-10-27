# Ambient Clinical AI and Documentation

> **Module 09 — Clinical Natural Language Processing**
> Topic 11 of 13
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 02, 05, and 09
> Version 2026.3

## 1. Why this topic exists

A clinician talks with a patient for twelve minutes. A microphone captures the conversation. A
system returns a draft note with a history of present illness, a review of systems, a physical
exam, and a plan. The clinician reads it, changes six sentences, and signs. That is the product
category this topic covers: **ambient clinical documentation**, also called ambient scribing or
ambient clinical intelligence.

The category matters because it moves generative language models from a chat window into the legal
medical record. A summary shown to a patient can be wrong and corrected in conversation. A note that
is signed, billed, and stored is a different class of object. It supports a level-of-service code,
it is discoverable in litigation, it is read by the next clinician who may never meet the patient,
and it is the input to every downstream analytics pipeline in Topic 01. When the draft contains a
sentence nobody said, the error is not a bad answer. It is a false statement in a record with a
clinician's attestation attached.

The scale is no longer experimental. The Permanente Medical Group described a regional rollout to
3,442 physicians covering 303,266 encounters in the first ten weeks (Tierney et al., NEJM Catalyst,
2024). Randomised and quasi-experimental evaluations followed in 2025. The interesting question for
a data scientist is no longer whether these systems are used. It is what they get wrong, how the
error is measured, and who is accountable when the measurement is flattering.

Three properties make ambient documentation harder than ordinary clinical NLP.

**The input is a conversation, not a document.** Clinical text in the previous topics arrives as
characters with stable offsets. Audio arrives as a stream with overlapping speakers, disfluency,
background noise, and no sentence boundaries. Every downstream artefact inherits a transcription
error, and transcription errors are not uniformly distributed: drug names, numbers, laterality,
and negations are the tokens most likely to be corrupted and the ones that change clinical meaning.

**The output is a legal attestation.** A signed note asserts that the clinician reviewed and agrees
with its content. Automation bias turns that assertion into a formality. If the interface makes a
fabricated sentence visually indistinguishable from a verified one, the signature certifies text the
signer did not read.

**The evaluation is confounded by the user.** The system's measured benefit depends on how much the
clinician edits, how fast they read, and what they would otherwise have typed. A product that
produces fluent, well-organised, partly wrong notes can reduce editing time while increasing risk.
A product that produces clumsy but faithful notes can increase editing time while reducing risk.
Time saved and safety are separate axes, and vendor material usually reports only the first.

This topic treats ambient documentation as a pipeline with a measurable chain of custody. Every
sentence in the note should be traceable to a span of audio or to an explicit template decision.
The data scientist's job is to make that chain auditable, to measure the errors that matter, and to
say plainly which claims about the product are supported by peer-reviewed evidence and which are
marketing.

> **Note:** The clinical, legal, and reimbursement content here is educational. Consent law varies
> by state and changes; device regulation depends on intended-use claims made by a specific
> manufacturer. Neither is legal advice.

## 2. Learning outcomes

By the end of this topic you can:

1. **Diagram** an ambient documentation pipeline from audio capture to EHR write-back, naming the
   input, output, and dominant error type at each stage.
2. **Compute** word error rate (WER), concept error rate, character edit distance ratio, and
   time-in-note from raw transcripts, drafts, and signed notes, and **state the unit** of each.
3. **Design** a grounding check that links each generated sentence to a transcript span, and
   **calibrate** its threshold against clinician judgements of support.
4. **Distinguish** omission, fabrication, and misattribution as error classes with different
   detection costs and different clinical consequences.
5. **Audit** a consent and data-handling design against state recording law, the Health Insurance
   Portability and Accountability Act (HIPAA) Business Associate Agreement requirements, and
   vendor retention terms.
6. **Classify** a product's intended use against the Food and Drug Administration (FDA) device
   framework and the Office of the National Coordinator for Health Information Technology (ONC)
   decision support intervention (DSI) transparency criterion.
7. **Critique** a vendor claim by locating the peer-reviewed study behind it, or by showing that
   none exists.
8. **Write** an evaluation plan for an ambient pilot with a pre-specified primary outcome, a
   safety endpoint, and a stopping rule.

## 3. The core mental model

The useful mental model is not "speech to text plus a summariser". It is **two records and a
translation between them**.

The first record is the encounter: what was said, by whom, when, in what tone, with what gestures
and what was pointed at on the body. It exists only as audio and as human memory. It is rich,
redundant, and ambiguous.

The second record is the note: a structured, codified, legally attested summary written in a
conventional genre. It is sparse, decontextualised, and authoritative.

Ambient documentation is a lossy translation from the first record to the second. Every stage of the
pipeline loses something and may add something. The design question at each stage is whether the
loss is acceptable and whether the addition is traceable.

```text
  RECORD 1: THE ENCOUNTER                       RECORD 2: THE NOTE
  audio, speakers, gesture, tone               structured text, codes, signature
          |                                             ^
          |                                             |
          v                                             |
  +-------------------+     +-------------------+       |
  | 1 Caption         |     | 5 Draft           |       |
  |   audio -> text   |---->|   facts -> prose  |-------+
  +-------------------+     +-------------------+
          |                         ^
          v                         |
  +-------------------+     +-------------------+
  | 2 Structure       |     | 4 Ground          |
  |   turns -> roles  |---->|   sentence -> span|
  |   turns -> phases |     |   verify / flag   |
  +-------------------+     +-------------------+
          |                         ^
          v                         |
  +-------------------+             |
  | 3 Extract         |-------------+
  |   text -> facts   |
  |   fact -> status  |
  +-------------------+

  Each arrow is a lossy transformation. The audit question is always:
  which span of Record 1 supports this sentence of Record 2?
```

Read the diagram from the bottom of the funnel upward. Four gaps open between the records, and
each gap has a different owner and a different measurement.

**Acoustic gap.** Speech recognition turns pressure waves into words. The gap is measured by WER
and, more usefully, by concept error rate: the share of clinically relevant concepts that survive.
A transcript can have a low WER and still lose the word "no".

**Referential gap.** Diarization and role assignment decide who said what. In a two-party visit the
problem is small. In a paediatric visit with a parent, a child, and a grandparent, attributing
"diabetes runs in the family" to the patient rather than the grandmother changes the problem list.

**Temporal gap.** A conversation has order and tense. "I used to take metoprolol" and "I take
metoprolol" differ by one verb and by one active medication. A note compresses a 12-minute
conversation into a few hundred words, and compression is where tense, hedging, and conditionality
are lost.

**Normative gap.** The note genre has conventions the conversation does not: a review of systems is
expected to be complete, a physical exam is expected to have findings for each system examined, and
a plan is expected to justify orders. A language model trained on notes has learned these
conventions. When the conversation omits a convention, the model may fill it. This is the single
most important failure mode in the category, because the inserted content is fluent,
genre-appropriate, and unsaid.

The four gaps compose. An acoustic error in a drug name becomes a wrong medication in the plan; a
referential error puts a family history in the patient's chart; a normative fill inserts a negative
review of systems nobody asked about; and the clinician, reading a well-formed note at the end of a
clinic day, signs it.

### The chain-of-custody principle

The design principle that addresses all four gaps is the same one used in laboratory medicine:
every claim carries provenance. For ambient documentation, provenance means a pointer from each
generated sentence to one of three sources:

| Provenance class | Meaning | Interface treatment |
|---|---|---|
| Spoken and supported | A transcript span entails the sentence | Show the audio offset; allow one-click playback |
| Structured and imported | A lab value, medication list entry, or prior note | Show the source record and timestamp |
| Inferred or template | Generated by convention, not by the conversation | Require explicit confirmation before it can remain |

A sentence with no provenance is a defect. The strongest version of this principle is that the
note's internal representation is a set of claims, each with a source pointer, and the prose is a
rendering of that set. Systems that generate prose first and look for support afterwards cannot
guarantee this property, because the support search is a rationalisation rather than a check.

### What the model is and is not doing

It helps to separate three functions that vendors bundle into one product.

- **Captioning** is speech recognition. It is a mature technology with measurable error rates.
- **Structuring** is extraction and classification: who spoke, what is a finding, what is a
  medication, what is negated, what is historical. This is the subject of Topics 03 and 04.
- **Drafting** is generation. It is the newest function and the one with the weakest evaluation
  base.

Most reported failures in 2025 come from drafting and from the interaction between captioning and
drafting. A system that captions well and drafts conservatively is safer than one that captions
well and drafts fluently. The distinction is invisible in a product demo, which is why the
evaluation design in this topic separates the three.

### The economics that drive adoption

The business case rests on documented time. Sinsky et al. (Annals of Internal Medicine, 2016)
measured physicians spending 49.2 percent of office hours on electronic health record (EHR) and
desk work and 27 percent on direct clinical face time, with one to two hours of after-hours
clerical work per hour of direct face time. That after-hours work is what the industry calls
"pajama time". If an ambient system removes 30 minutes per clinician per day, the arithmetic at a
300-clinician group is 150 clinician-hours per day, which is real money. The rest of this topic
treats that arithmetic as a claim to be measured rather than a fact to be quoted.

## 4. Definitions and vocabulary

The vocabulary of this topic mixes speech engineering, clinical documentation, and health law. The
table gives the term as used in this topic, the misuse that causes confusion in procurement
documents, and the unit or coding system where one exists.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Ambient clinical documentation | Generation of a clinical note from passively captured encounter audio, with clinician review and signature | Used interchangeably with "AI scribe", which also covers dictation post-processing | None |
| Ambient clinical intelligence | Vendor term for the same category, often implying extraction and coding beyond the note | Treated as a regulated category; it is a marketing category | None |
| Automatic speech recognition (ASR) | Conversion of audio to text | Called "transcription" when a human is in the loop; the two have different error profiles | Word error rate, percent |
| Word error rate (WER) | (Substitutions + insertions + deletions) divided by reference word count | Reported without the reference transcript; compared across corpora with different conventions | Proportion, 0 to 1 or percent |
| Concept error rate (CER) | Share of clinically relevant concepts in the reference that are wrong or missing after recognition | Confused with WER; a low WER can coexist with a high CER | Proportion, percent |
| Diarization | Partitioning audio into speaker-homogeneous segments ("who spoke when") | Assumed to equal role assignment; diarization labels speaker 1 and speaker 2, not doctor and patient | Diarization error rate, percent |
| Speaker role assignment | Mapping diarized speakers to clinical roles (clinician, patient, family, interpreter) | Assumed to be solved by diarization | Accuracy by role, percent |
| Transcript | Timestamped, speaker-attributed text of the encounter | Treated as ground truth; it is a measurement with its own error | Character offsets and timestamps |
| Draft note | Machine-generated note before clinician editing | Treated as the product; the signed note is the product | None |
| Signed note | Note attested by the clinician and filed in the EHR | Assumed to equal the draft plus corrections; corrections are data | None |
| Character edit distance ratio | Levenshtein distance between draft and signed note, divided by draft length | Computed on the whole note, which hides section-level rewrites | Unitless, 0 to 1 |
| Clinician edit burden | Time or edits required to bring a draft to signature | Reported as "acceptance rate", which hides large edits on a small share of notes | Minutes per encounter, or edits per note |
| Time-in-note | Total time a clinician spends with the note interface for an encounter, including review and editing | Confused with "documentation time", which may include inbox and orders | Minutes per encounter |
| Pajama time | EHR work performed outside scheduled hours | Used without a time window; the window must be stated | Minutes per day, with window |
| Review of systems (ROS) | Structured inventory of symptoms, usually with positive and negative findings | Treated as narrative; it is a checklist with legal weight | None |
| History of present illness (HPI) | Narrative account of the current problem | Confused with the full history | None |
| SOAP note | Subjective, Objective, Assessment, Plan structure | Treated as the only structure; many specialties use others | None |
| Hallucination | Generated content not supported by the encounter or any source record | Used for all errors, which hides omissions | Count per note, or rate per 100 sentences |
| Omission | A clinically relevant fact present in the encounter but absent from the note | Underweighted in evaluation because it is invisible in the draft | Count per note, or recall, percent |
| Misattribution | A true fact assigned to the wrong person, time, or body site | Lumped with hallucination; the fix is different | Count per note |
| Grounding | Linking a generated sentence to a supporting source span | Confused with factuality; grounding is a necessary condition, not a sufficient one | Proportion of sentences grounded, percent |
| Entailment | A relation where a source span logically supports a claim | Treated as a probability of truth; it is a relation judged by a model or annotator | Label, or model score 0 to 1 |
| Attestation | The clinician's signature asserting review and agreement | Treated as evidence of reading | None |
| Automation bias | Tendency to accept automated output without adequate verification | Described as "trust"; the failure is insufficient verification, not excess trust | None |
| Business Associate Agreement (BAA) | Contract required under HIPAA when a vendor handles protected health information on behalf of a covered entity | Assumed to cover all uses, including model training | Contract |
| Protected health information (PHI) | Individually identifiable health information held by a covered entity or business associate | Treated as synonymous with "audio"; audio of a clinical encounter is PHI in context | 18 identifier categories under 45 CFR 164.514 |
| Two-party consent | State rule requiring all parties to consent to recording a conversation | Applied nationwide as if federal; federal law is one-party | State statute |
| Software as a Medical Device (SaMD) | Software intended for a medical purpose without being part of a hardware device | Applied to any clinical software | Regulatory classification |
| Clinical decision support (CDS) | Software that supports clinical decisions, with a regulatory safe harbour for certain categories | Confused with documentation support | FDA guidance category |
| Decision support intervention (DSI) | ONC's certification category for AI and rule-based decision support in certified health IT | Assumed to cover note drafting | Certification criterion |
| Intended use | The manufacturer's stated purpose, which drives regulation | Confused with actual use in a hospital | None |
| Enrolled population | Patients whose encounters are captured by the system | Assumed to equal the panel; consent and rooming practices create selection | Count of patients |
| Encounter capture rate | Share of eligible encounters with a usable recording | Reported as adoption; adoption is a user metric, capture is a coverage metric | Percent |
| Edit provenance | Record of which clinician changed which span, and when | Not captured by most systems, which discards the most valuable training signal | Event log |
| Note quality | Fitness of a note for its clinical, legal, and billing purposes | Reduced to fluency or similarity to a reference | PDQI-9 or PDQI-10 score, 1 to 5 per attribute |
| Physician Documentation Quality Instrument (PDQI-9) | Nine-attribute rating of note quality (accurate, thorough, useful, organised, comprehensible, succinct, synthesised, up to date, internally consistent) | Applied to drafts by the generating team | Ordinal score, 9 to 45 |
| After-visit summary (AVS) | Patient-facing summary generated from the encounter | Treated as the same product as the note; the audience and reading level differ | None |
| Interpreter-mediated encounter | Encounter where a third party interprets between clinician and patient | Captured as a three-speaker conversation with no role label for the interpreter | Language code, interpreter mode |
| Retention policy | How long audio and transcripts are kept, and by whom | Assumed to be "zero" because the vendor says so; the BAA governs | Days, contractual |

Two definitions deserve expansion because they are routinely misused in procurement.

**Hallucination versus omission.** A hallucination is visible. A reviewer reading the draft can
notice a diagnosis that was never discussed, especially if the interface flags it. An omission is
invisible in the draft by construction: the reader does not know what was said and left out. This
asymmetry means that evaluations built on reading drafts over-report hallucinations and
under-report omissions, and it means that products optimised against a draft-reading reviewer will
drift toward more omission and less fabrication. The 2025 evidence suggests this is exactly what
happened: omissions dominate error counts by a wide margin (Draper et al., 2025; Anderson et al.,
2025), while hallucinations are rarer and more severe per event.

**Grounding versus factuality.** A sentence is grounded if a source span supports it. A sentence is
factual if it is true of the patient. Grounding can be achieved by quoting the transcript and still
be clinically wrong, because the patient can be wrong, the clinician can mishear, and the
transcript can misattribute. Grounding is therefore a floor: it eliminates the class of error where
the note contains content with no origin at all. It does not eliminate the class where the content
has an origin but the origin is a misunderstanding.

## 5. The mechanics, step by step

This section is the procedure for building and operating an ambient documentation pipeline. It is
written as the sequence a system executes for one encounter, with the design decision that matters
at each step. Steps 1 to 4 happen before any language model is involved, and most of the serious
failures originate there.

The procedure assumes a cloud-hosted system with a mobile or room microphone, a vendor-hosted
speech and language stack, and an integration into an EHR. Variants (on-premise inference, a
dedicated device, a telehealth-native system) change the implementation but not the sequence.

### Step 1. Establish the lawful basis and capture consent

**Input:** The patient's identity, the jurisdiction of the encounter, the organisation's consent
policy, and the room composition.

**Output:** A consent decision, a consent record, and a session configuration.

**Rationale:** Recording law and HIPAA are different regimes and both apply. HIPAA permits
treatment-related recording but requires the organisation to have a lawful basis and to disclose it
in its notice of privacy practices. State wiretap law governs whether the patient must affirmatively
agree. In the nine states that are strictly all-party for in-person recording (California,
Florida, Illinois, Maryland, Massachusetts, Montana, New Hampshire, Pennsylvania, and Washington),
plus the hybrid states whose rules depend on medium, the safe operating assumption is that every
person in the room must be informed and must be able to decline. A consent screen shown once at
check-in does not cover the friend who joins the room halfway through.

The consent record is also a data asset. It determines whether the audio may be retained, whether
it may be used for model improvement, and what must be deleted on request.

**Branch point:** Consent has more than two outcomes.

| Situation | Decision | Session configuration | Record kept |
|---|---|---|---|
| Patient affirms, no restrictions | Record | Full capture, standard retention | Consent event, audio per policy |
| Patient affirms but declines model training | Record, restricted use | Full capture, training flag off | Consent event with restriction |
| Patient declines recording, accepts live scribing | Do not retain audio | Streaming only, no persistence | Consent refusal only |
| Patient declines entirely | Do not record | Manual documentation | Refusal, offered alternative |
| Patient lacks capacity, surrogate present | Surrogate consents where law allows | Full capture with surrogate record | Surrogate identity and authority |
| Third party objects mid-encounter | Stop or mute | Capture paused from that point | Pause event with timestamp |

### Step 2. Capture audio and define the channel layout

**Input:** Microphone array or device stream, room acoustics, and the consent decision.

**Output:** One or more audio channels with sample rate, timestamps, and device metadata.

**Rationale:** Channel layout determines what diarization can achieve. A single mono channel forces
the system to separate speakers from voice characteristics alone. A device with a known clinician
channel and a far-field patient channel gives the diarization an anchor. A stereo room array gives
direction of arrival, which is more reliable than voice embedding when two speakers have similar
vocal characteristics.

The choice also determines what happens with more than two speakers. Paediatric and
interpreter-mediated encounters routinely have three or four. A two-channel assumption silently
merges the parent and the child, which is a referential error created at the hardware layer.

**Branch point:** If the encounter is interpreter-mediated, the channel layout should treat the
interpreter as a distinct role, not as a second clinician. If the encounter is group therapy or a
family meeting, the two-party model does not apply and the note structure must be chosen
accordingly.

### Step 3. Detect speech and segment the encounter into phases

**Input:** Audio channels.

**Output:** Speech segments with start and end times, non-speech intervals, and phase labels.

**Rationale:** Voice activity detection removes silence and noise before recognition, which reduces
cost and improves accuracy. Phase segmentation (greeting, history, examination, discussion, plan,
closing) is not required for transcription but is required for drafting, because the same sentence
means different things in different phases. "I will order a chest X-ray" said in the plan phase is
a plan; said in the history phase as a report of a previous visit, it is history.

Phase boundaries are usually inferred from a combination of lexical cues, speaker change patterns,
and duration. Errors here propagate into the draft as misplaced content, which is a class of error
that reviewers often describe as "the note is disorganised" without recognising it as a
segmentation failure.

**Branch point:** If the encounter has no clear phases, as in a telehealth message thread or an
async video, the pipeline should not force a SOAP structure. Forcing structure is the mechanism by
which content gets moved into the wrong section.

### Step 4. Diarize and assign clinical roles

**Input:** Speech segments and channel metadata.

**Output:** Speaker-labelled turns with a role label and a confidence score per turn.

**Rationale:** Diarization answers "who spoke when" in an anonymous sense. Role assignment answers
"which of these is the clinician". The second is a classification problem with a small label set
and a large prior: in an ambulatory encounter, the person who asks most of the questions and uses
clinical vocabulary is usually the clinician. The prior fails in teaching encounters, in
interpreter-mediated encounters, and when a patient arrives with a family member who is a nurse.

Role errors are the most consequential referential failures, because they change whose fact it is.
A grandmother's diabetes becomes the child's family history in the wrong column; a patient's "my
father had a heart attack at 52" becomes the patient's own history; an interpreter's "she says the
pain started Tuesday" becomes the clinician's finding.

**Branch point:** Role assignment outcomes and their handling.

| Assignment | Evidence | Action | Typical failure if wrong |
|---|---|---|---|
| Clinician, high confidence | Known device channel, question rate, vocabulary | Draft with clinician as author | None |
| Patient, high confidence | Channel, answer pattern, first-person reference | Attribute findings to patient | None |
| Family member, high confidence | Third voice, kinship terms, collateral phrasing | Attribute to family, not patient | Family history becomes patient history |
| Interpreter, detected | Alternating triad, first-person conversion | Attribute content to the patient, note the interpreter | Interpreter's paraphrase becomes patient's words |
| Unknown speaker | Overlap, low confidence | Flag the turn; exclude from assertive drafting | Content attributed by guess |

### Step 5. Recognise speech with a medical vocabulary

**Input:** Segmented, role-labelled audio.

**Output:** A timestamped transcript with per-word confidence and alternative hypotheses for
low-confidence tokens.

**Rationale:** General-purpose recognisers are trained on broadcast and conversational speech.
Clinical speech contains drug names, abbreviations, laterality, numbers, and units that are rare in
those corpora. A recogniser that hears "fifty" as "fifteen" changes a dose. One that hears "no
acute distress" as "no a cute distress" produces a phrase a language model may repair into
something plausible and wrong.

Medical adaptation takes three forms: a custom vocabulary or biasing list, a domain-trained acoustic
model, and a post-recognition correction model. Each has a cost. A biasing list improves recall of
listed terms and can increase false acceptance of similar-sounding terms. A domain acoustic model
improves in-domain accuracy and may degrade on accented speech if the training data were narrow.

Per-word confidence is the most useful output of this step and the one most often discarded.
Confidence is what allows the later stages to distinguish "the transcript says this" from "the
transcript is unsure about this". A pipeline that emits only a single transcript string has thrown
away its own uncertainty estimate.

**Branch point:** If the audio quality is below a usable threshold (signal-to-noise ratio, overlap
rate, or mean confidence), the correct action is to fail the session and tell the clinician, not to
generate a low-quality draft. A silent degradation to a fluent wrong note is worse than a visible
failure.

### Step 6. Normalise the transcript without destroying evidence

**Input:** Raw transcript with timestamps and confidence.

**Output:** A readable transcript plus a mapping from every normalised token to its raw span.

**Rationale:** Downstream models work better on text with disfluencies removed, numbers expanded,
and abbreviations resolved. The danger is that normalisation is itself an interpretation. Expanding
"q.d." to "once daily" is safe. Expanding "MS" to "multiple sclerosis" when the context is
"MS changes" (mental status) is not. Correcting "pt has no CP" to "patient has no chest pain" is
safe. Correcting "no CP" to "no chest pain" when the clinician said "no C. diff" is not.

The rule is that normalisation must be reversible. Every normalised token keeps a pointer to the
raw tokens it replaced, so that a reviewer can see what was actually said. Systems that normalise
in place and discard the raw text cannot support grounding at the word level, and their grounding
claims become claims about the normaliser rather than about the encounter.

**Branch point:** If an abbreviation is ambiguous in context, keep the raw form and mark it
ambiguous. Do not resolve it silently. Ambiguous abbreviations are a known source of clinical
error and a known source of fabricated findings when a language model resolves them by prior.

### Step 7. Extract candidate facts and assertion status

**Input:** Normalised transcript with roles and phases.

**Output:** A list of candidate facts, each with a type, a span, a subject, and an assertion status.

**Rationale:** This is the step that determines whether the note is faithful. The extraction schema
should be explicit and small. A workable schema has six entity types (problem, symptom, medication,
procedure, test, social factor) and six assertion statuses (present, absent, historical,
hypothetical, familial, uncertain). The subject is separate from the status: "mother has diabetes"
is a present finding about the mother, not a familial risk about the patient.

Negation is the hardest part. Clinical language negates in many ways: "no", "denies", "without",
"negative for", "never", "not", and constructions that invert twice. It also uses scoped negation,
where "no fever, chills, or night sweats" negates three findings with one marker, and pseudo-negation,
where "no increase in pain" asserts stable pain rather than absence of pain. A pipeline that
handles only the first pattern will systematically convert absent findings into present ones,
which is the error class with the largest clinical consequence in the review studies.

**Branch point:** Assertion status assignment.

| Status | Trigger examples | Note placement | Downstream risk |
|---|---|---|---|
| Present | "I have", "it hurts", unmarked declarative | Assessment or HPI | None |
| Absent | "no", "denies", "negative for" | ROS or pertinent negatives | Absent read as present |
| Historical | "used to", "history of", "back in 2019" | Past history | Historical read as active |
| Hypothetical | "if it gets worse", "we could try" | Plan, conditional | Plan read as ordered |
| Familial | "my mother", "runs in the family" | Family history | Family history read as patient history |
| Uncertain | "maybe", "I'm not sure", "possibly" | HPI with hedge | Hedge dropped in drafting |

### Step 8. Reconcile extracted facts against the record

**Input:** Candidate facts, the patient's medication list, problem list, allergies, and recent
notes.

**Output:** Facts labelled as new, changed, contradicted, or already known, with a reconciliation
note.

**Rationale:** A conversation is a sample of the patient's state, not a complete inventory. The note
must combine what was said with what is already known. Reconciliation is where the system decides
whether "I stopped the metoprolol" is a medication change to record, whether a reported allergy
conflicts with the allergy list, and whether a problem discussed today is new or a follow-up.

Reconciliation is also where a subtle fabrication enters. If the model is asked to produce a
complete medication list and the conversation mentions two of the patient's seven medications, a
helpful model may fill the other five from the chart. That is not a hallucination in the strict
sense, because the content is true. It is a provenance error: the note implies the list was
reviewed in the encounter when it was imported. The distinction matters for billing and for the
next clinician's interpretation.

**Branch point:** Reconciliation outcomes.

| Outcome | Example | Draft treatment | Review treatment |
|---|---|---|---|
| New | Patient reports a symptom not in the chart | Include as reported | No special flag |
| Changed | Dose changed, medication stopped | Include as change with date | Highlight the change |
| Contradicted | Patient reports penicillin allergy; chart lists none | Include both | Require explicit resolution |
| Already known | Chronic condition mentioned in passing | Include briefly or omit | No flag |
| Imported | Lab value or medication from the chart | Include with source label | Show source and timestamp |
| Unresolvable | Conflicting reports within the encounter | Include the conflict | Require clinician decision |

### Step 9. Plan the note and draft it section by section

**Input:** Reconciled facts, the note template for the specialty and visit type, and the clinician's
prior note style if available.

**Output:** A sectioned draft with a claim list behind each section.

**Rationale:** Drafting should be constrained by the extraction output, not free-running over the
transcript. A model asked to "write an HPI from this transcript" will produce a good HPI, and a good
HPI includes conventions the transcript may not support. A model asked to "render these twelve
claims as an HPI in this template" has less room to invent.

Section-by-section drafting also makes the failure local. If the plan section is wrong, the review
can focus there. A single free-form generation makes every error a whole-note error.

The template is a clinical and billing artefact, not a neutral container. A SOAP template expects a
complete ROS and a full physical exam; a specialty template may expect a functional history and a
device interrogation. The draft should follow the template the clinician will sign, including its
required fields, and should mark fields the encounter did not cover rather than filling them.

**Branch point:** If a required template field has no supporting claim, the correct output is an
empty field with a prompt, not a generated value. This single rule prevents most fabricated
negatives.

### Step 10. Ground every sentence and score support

**Input:** Draft sentences and the transcript with spans.

**Output:** Per-sentence support labels, a source span, and a grounding score.

**Rationale:** Grounding is the verification layer. For each generated sentence, the system searches
for the transcript span that best supports it and assigns one of four labels. The label set matters
more than the model. A binary "grounded / not grounded" collapses the two cases that need different
handling: a sentence that is contradicted by the transcript and a sentence for which there is no
evidence either way.

The score for a sentence is the maximum entailment probability over the candidate spans:

```text
 Grounding(S) = max over windows T_j of P(entailment | T_j, S)
```

A threshold on this score turns it into a flag, and the threshold is a policy choice rather than a
property of the model.

The scoring method can be lexical overlap, an entailment model, or a language model prompted to
judge support. Each has known biases. Lexical overlap rewards sentences that quote the transcript
and penalises paraphrase, so it under-flags fluent fabrication. Entailment models are better at
paraphrase and can be over-confident on long premises. Language-model judges are flexible and
expensive, and they share failure modes with the generator, which is a conflict of interest when
the same model family does both.

Whichever method is used, the threshold must be calibrated against clinician judgements on a
sample, and the calibration must be reported. A threshold of 0.8 on an uncalibrated score is a
number, not a guarantee.

**Branch point:** Grounding outcomes and their interface treatment.

| Label | Definition | Interface | Can it be signed without action |
|---|---|---|---|
| Supported | A span entails the sentence | Show offset, allow playback | Yes |
| Partially supported | A span supports part of the sentence | Highlight the unsupported clause | No, until edited |
| Contradicted | A span contradicts the sentence | Block and show the conflicting span | No |
| No source | No span in the encounter supports or contradicts | Mark as inferred or imported | No, unless confirmed |
| Imported | Supported by a structured record | Show source and timestamp | Yes, with source visible |

### Step 11. Present the draft for review

**Input:** Draft, support labels, source spans, and reconciliation flags.

**Output:** A review interface state and a set of clinician actions.

**Rationale:** The interface is part of the safety system. If every sentence looks the same, the
clinician's reading is governed by the note's fluency, and fluent text is read quickly and
leniently. If unsupported sentences are visually distinct, the clinician's attention is directed to
the places where it is needed.

Two interface properties have outsized effects. First, the ability to play the audio for a sentence
in one click converts a plausibility judgement into a verification. Second, showing what was
omitted, not only what was generated, addresses the error class that draft-reading cannot detect.
A "not mentioned" panel that lists topics the template expected but the encounter did not cover is
more valuable than a confidence score on the sentences that were generated.

Edit capture is the other function of this step. Every keystroke, deletion, and acceptance is a
label. A system that discards edits has discarded its best training data and its best monitoring
signal.

**Branch point:** If the clinician signs without opening any flagged sentence, that is a
measurable event, not a personal failing. It should be logged as a review-depth metric and used to
tune the interface.

### Step 12. Write back, attest, and retain

**Input:** The signed note, the edit log, the audio reference, and the consent record.

**Output:** A filed note with provenance metadata, an audit record, and a retention state.

**Rationale:** Write-back is where the pipeline meets the EHR's data model. The note is filed as a
document, the structured elements may be filed as observations or as text, and the audio is stored
separately under the retention policy. The linkage between them is the audit trail that makes the
system defensible.

Attestation must mean something. The signature should record that the clinician reviewed the note,
not merely that the note exists. Systems that pre-check an attestation box, or that allow signing
from a list view without opening the note, have converted a legal assertion into a click.

Retention has three separate questions: how long the audio is kept, whether the transcript is kept,
and whether either is used for model training. The BAA and the consent record answer them
separately. "Zero retention" in marketing material usually refers to one of the three.

**Branch point:** Write-back outcomes.

| Outcome | Condition | Action |
|---|---|---|
| Filed | Signed, all blocks resolved | Standard filing |
| Filed with flags | Signed, unresolved flags acknowledged | Filing plus quality event |
| Returned to draft | Clinician rejected the draft | Log rejection with reason |
| Failed | Integration error or timeout | Preserve draft, alert clinician, no silent loss |
| Deleted | Patient withdrew consent | Delete per policy, log deletion |
| Amended | Correction after signing | New version, prior version retained |

The procedure has a property worth stating explicitly: every step produces an artefact that can be
audited after the fact. A system that produces only a note and an audio file, with no transcript
provenance, no extraction output, and no edit log, cannot answer the question a regulator, a
plaintiff, or an internal safety committee will ask. The artefacts are the safety case.

## 6. Worked example

This example follows one ambulatory encounter through the pipeline and computes the four numbers a
pilot report would need: word error rate, concept error rate, edit burden, and time saved. Every
number is synthetic. No patient data are used.

### The encounter

A 67-year-old woman attends a follow-up for hypertension and knee pain. Her daughter is in the
room. The visit lasts 12 minutes 40 seconds, or 760 seconds. The audio contains 1,580 words: 610
from the clinician, 842 from the patient, and 128 from the daughter.

Speech rate = 1,580 / 760 = 2.08 words per second, or 125 words per minute. That is a normal
consultation rate and it matters later, because a recogniser's accuracy depends on how fast and how
overlapped the speech is.

The clinically relevant content is:

- Right knee pain, worse on stairs, swelling by the end of the day.
- The patient stopped lisinopril two weeks ago because of a cough.
- The daughter mentions that the patient's grandmother had a stroke at 70.
- The clinician plans a right knee radiograph, a trial of topical therapy, and a physical therapy
  referral, and asks the patient to monitor blood pressure at home.

The encounter never discusses chest pain, shortness of breath, or fever. Remember that. The draft
will contain all three.

### Stage 1: transcription

The reference transcript is the word sequence a human transcriber would produce. The system's
hypothesis differs by 28 substitutions, 17 deletions, and 11 insertions.

Word error rate = (28 + 17 + 11) / 1,580 = 56 / 1,580 = 0.0354, or 3.5 percent.

That is a good number for conversational clinical audio, and it is misleading on its own. The
errors are not uniformly distributed. Of the 56 errors:

| Error | Count | Example | Clinical weight |
|---|---:|---|---|
| Drug name | 3 | "lisinopril" heard as "Zestril" | Low, same drug |
| Laterality | 2 | "right" heard as "left" | High |
| Number | 4 | "two weeks" heard as "two months" | Medium |
| Negation marker | 1 | "no" dropped before "swelling" | High |
| Filler and disfluency | 41 | "um", repeated words | None |
| Other content | 5 | "stairs" heard as "stares" | Low |

Forty-one of 56 errors are disfluency. The WER is dominated by content that does not matter. The
errors that matter are 10, and three of them change clinical meaning.

### Stage 2: concepts

The reference transcript contains 42 clinical concepts, defined as a problem, symptom, medication,
test, procedure, or social factor that a clinician would consider part of the encounter.

The pipeline's extracted concept set contains 40 of them. Of those 40, one is wrong: the knee is
recorded as left. Two are missing: the home blood pressure monitoring instruction and the
grandmother's stroke.

Concept error rate = (1 wrong + 2 missing) / 42 = 3 / 42 = 0.0714, or 7.1 percent.

The comparison is the point of the example:

| Metric | Value | Unit |
|---|---:|---|
| Word error rate | 3.5 | percent of reference words |
| Concept error rate | 7.1 | percent of reference concepts |
| Ratio | 2.0 | unitless |

A WER of 3.5 percent and a concept error rate of 7.1 percent describe the same transcript. If a
vendor reports only the first, the buyer learns that the recogniser is good at words. The buyer does
not learn that one in fourteen clinical concepts is wrong.

### Stage 3: the draft and the edits

The draft note is 38 sentences, 612 words, and 3,480 characters. The clinician makes six edits:

| # | Edit | Type | Words changed |
|---:|---|---|---:|
| 1 | Delete "Patient denies chest pain, shortness of breath, or fever." | Fabrication removal | 22 |
| 2 | "left knee" to "right knee" in the HPI | Factual correction | 2 |
| 3 | "left knee" to "right knee" in the plan | Factual correction | 2 |
| 4 | "mother had a stroke" to "grandmother had a stroke" | Attribution correction | 9 |
| 5 | Add "discussed NSAID risk" to the plan | Omission repair | 14 |
| 6 | "X-ray" to "radiograph, two views" | Terminology | 4 |

Total words changed = 22 + 2 + 2 + 9 + 14 + 4 = 53.

Sentence edit rate = 6 / 38 = 0.158, or 15.8 percent of sentences.

Word edit rate = 53 / 612 = 0.0866, or 8.7 percent of draft words.

Character edit distance ratio: the Levenshtein distance between draft and signed note is 412
characters, against a draft of 3,480 characters. Ratio = 412 / 3,480 = 0.118, or 11.8 percent.

The three measures disagree, which is normal. The sentence rate counts a two-word laterality fix
the same as a 22-word deletion. The word rate weights by length. The character rate weights by
characters and therefore penalises the medical terminology edit disproportionately, because
"radiograph, two views" is longer than "X-ray".

The character ratio is the number most often quoted in vendor dashboards because it is easy to
compute automatically. It is also the number most easily gamed: a system that produces short
sentences and leaves out content has a low edit distance and a high omission rate.

One more calculation is worth doing. Edits 1 to 5 contain 49 words: 36 are the safety-relevant
correction, and the other 13 are the phrasing of the added instruction. Edit 6 contributes 4
stylistic words. The share of edited words that represent safety-relevant corrections is therefore
36 / 53 = 0.679, or 67.9 percent. That share, not the total edit distance, is the number a safety
committee should track.

### Stage 4: time arithmetic

The time model has four components.

| Component | Calculation | Minutes |
|---|---|---:|
| Read the draft | 612 words at 240 words per minute | 2.55 |
| Make the edits | 53 words at 25 words per minute | 2.12 |
| Sign and file | Fixed | 0.40 |
| Total with ambient | Sum | 5.07 |

The baseline is the time the same clinician would spend typing the note without the system. In this
clinic the measured baseline is 11.4 minutes per encounter.

Saving per encounter = 11.4 − 5.07 = 6.33 minutes.

That figure is optimistic because it ignores exceptions. One note in eight requires a call to the
patient or a chart check to resolve a contradiction, at 4.0 minutes each.

Expected exception cost = 0.125 × 4.0 = 0.50 minutes per encounter.

Net saving = 6.33 − 0.50 = 5.83 minutes per encounter.

At 14 encounters per clinic day:

Daily saving = 5.83 × 14 = 81.6 minutes per day.

At 200 clinic days per year:

Annual saving = 81.6 × 200 = 16,320 minutes = 272 hours per clinician-year.

At a loaded cost of 120 US dollars per clinician-hour, the time value is 272 × 120 = 32,640 dollars
per clinician-year. Against a licence of 600 dollars per clinician per month, or 7,200 dollars per
year, the time case is strongly positive.

### What the numbers say

| Claim | Value | Supported by this example |
|---|---:|---|
| The transcript is accurate | WER 3.5 percent | Yes, at the word level |
| The note is accurate | Concept error 7.1 percent | Partly: 1 in 14 concepts wrong |
| The clinician barely edits | Character edit ratio 11.8 percent | Not really: 68 percent of edits were safety-relevant |
| The system saves time | 5.8 minutes per encounter | Yes, under the stated baseline |
| The system is safe | 1 fabricated sentence in 38 | Depends on the review, not the draft |

The two claims that survive scrutiny are the time saving and the word-level accuracy. The claim
that the clinician "accepts" the note does not survive, because acceptance rate counts a note with
a deleted fabrication as accepted.

### Diagnostic A: the role permutation test

Take the transcript and swap the speaker labels for the clinician and the patient. Run the drafting
step again. Compare the two drafts by token Jaccard similarity.

A pipeline that uses speaker roles produces a very different note: the patient's instructions
become a patient's report, and the clinician's questions become the patient's account. Jaccard
similarity should fall well below 0.75.

A pipeline that ignores roles produces nearly the same note, with similarity above 0.95. That
result means the drafting step is reading the transcript as an undifferentiated bag of sentences,
and every role-dependent fact in the note is an accident of phrasing rather than a modelled
attribution.

In this encounter, the permutation test is decisive because the daughter's utterance is the only
source of the family history. A role-blind drafter will keep writing "family history of stroke"
whichever speaker is labelled as the patient.

### Diagnostic B: the control encounter

Run the pipeline on 20 audio clips that contain no clinical content: a scheduling conversation, a
room with background noise, and 60 seconds of silence. A correct pipeline returns an empty note, a
"no clinical content" status, or a note with all fields marked as not discussed.

A broken pipeline returns a plausible note. The fabricated-sentence rate on control audio is the
cleanest single measure of template filling, because there is nothing in the input that could
legitimately support a clinical sentence. In published evaluations of commercial products, this
test is rarely reported, and the omission-dominated error profiles suggest it would be informative.

### The varied assumption

The time case depends entirely on the baseline. Suppose the clinician is a fast templater who
completes the same note in 6.5 minutes, not 11.4.

Saving per encounter = 6.5 − 5.07 = 1.43 minutes.
Net of exceptions = 1.43 − 0.50 = 0.93 minutes.
Annual = 0.93 × 14 × 200 = 2,604 minutes = 43.4 hours.
Value at 120 dollars per hour = 5,208 dollars.

Against a licence of 7,200 dollars per year, the time case is now negative by about 2,000 dollars
per clinician. The break-even licence is 5,208 / 12 = 434 dollars per month.

The conclusion changes from "clear positive return" to "negative on time alone, positive only if
the product also reduces turnover, improves note quality, or recovers visits". That is the correct
form of the argument. An ambient programme justified on time savings for slow documenters will
underdeliver for fast ones, and a vendor reference site is likely to be a site with slow baselines.

The same arithmetic applies to the safety case. A clinician with a 6.5-minute baseline has less
slack to review a draft carefully, and the review time in the model is fixed at 2.55 minutes for
reading. The ratio of review time to baseline time rises from 0.44 to 0.78, which means the fast
documenter is spending a larger share of the saved time on verification. If verification is
skipped to preserve the saving, the safety margin is what gets spent.

## 7. Code, query, or calculation

The module below reproduces every number in Section 6 and adds a threshold sweep for a grounding
check. It uses only the standard library, takes no arguments, and is deterministic.

Assumptions: the encounter counts (1,580 reference words; 28 substitutions, 17 deletions, 11
insertions) are supplied rather than computed from audio; the grounding scores are synthetic and
stand in for whatever scorer the pipeline uses; the time model uses the reading and editing rates
stated in Section 6.

```python
"""Audit an ambient documentation pilot from transcripts, drafts, and signed notes.
Standard library only. All inputs synthetic."""

def edit_ops(ref, hyp):
    """Return (substitutions, deletions, insertions) via word-level Levenshtein."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    i, j = n, m
    s = d = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, d, ins


# 1. Validate the aligner on a short pair, then apply to the encounter counts.
ref = "no chest pain no fever and no chills".split()
hyp = "no chest pain fever and chills".split()
s, d, i = edit_ops(ref, hyp)
print("alignment check:", (s, d, i), "WER", round((s + d + i) / len(ref), 4))

N = 1580
S, D, I = 28, 17, 11
wer = (S + D + I) / N
print(f"encounter WER: {S}+{D}+{I} over {N} words = {wer:.4f}")

# 2. Concept-level error.
concepts_total, wrong, missing = 42, 1, 2
cer = (wrong + missing) / concepts_total
print(f"concept error rate: {wrong + missing}/{concepts_total} = {cer:.4f}")
print(f"WER to CER ratio: {cer / wer:.2f}")

# 3. Edit burden.
sentences, edited_sentences = 38, 6
draft_words, changed_words = 612, 53
safety_words = 36
draft_chars, char_distance = 3480, 412
print(f"sentence edit rate: {edited_sentences}/{sentences} = {edited_sentences/sentences:.3f}")
print(f"word edit rate:     {changed_words}/{draft_words} = {changed_words/draft_words:.3f}")
print(f"char edit ratio:    {char_distance}/{draft_chars} = {char_distance/draft_chars:.3f}")
print(f"safety-relevant share of edits: {safety_words}/{changed_words} = {safety_words/changed_words:.3f}")

# 4. Time model.
def minutes_saved(baseline_min, read_wpm=240, edit_wpm=25, sign_min=0.40,
                  exception_rate=0.125, exception_min=4.0):
    read = draft_words / read_wpm
    edit = changed_words / edit_wpm
    with_system = read + edit + sign_min + exception_rate * exception_min
    return baseline_min - with_system, with_system

for baseline in (11.4, 6.5):
    saved, total = minutes_saved(baseline)
    annual_h = saved * 14 * 200 / 60
    print(f"baseline {baseline:>4} min -> review {total:.2f}, saved {saved:+.2f} min, "
          f"{annual_h:.1f} h/yr, ${annual_h*120:,.0f}/yr")
    print(f"    break-even licence: ${annual_h*120/12:,.0f}/month")

# 5. Grounding threshold sweep.
supported = [0.88, 0.91, 0.76, 0.83, 0.67, 0.95, 0.72, 0.61, 0.58, 0.81, 0.90, 0.86,
             0.79, 0.64, 0.55, 0.93, 0.87, 0.70, 0.66, 0.59, 0.74, 0.82, 0.89, 0.77,
             0.68, 0.62, 0.85, 0.91, 0.73, 0.57, 0.44, 0.38]
unsupported = [0.18, 0.22, 0.30, 0.41, 0.52, 0.74]
print("\nthreshold  flagged  TP  FP  FN  precision  recall  auto-signable")
for t in (0.40, 0.50, 0.60, 0.70, 0.80):
    tp = sum(x < t for x in unsupported)
    fn = len(unsupported) - tp
    fp = sum(x < t for x in supported)
    flagged = tp + fp
    prec = tp / flagged if flagged else 0.0
    rec = tp / (tp + fn)
    auto = sum(x >= t for x in supported + unsupported) / len(supported + unsupported)
    print(f"   {t:.2f}      {flagged:>2}   {tp:>2}  {fp:>2}  {fn:>2}      {prec:.3f}     {rec:.3f}      {auto:.3f}")
```

Output:

```text
alignment check: (0, 2, 0) WER 0.25
encounter WER: 28+17+11 over 1580 words = 0.0354
concept error rate: 3/42 = 0.0714
WER to CER ratio: 2.02
sentence edit rate: 6/38 = 0.158
word edit rate:     53/612 = 0.087
char edit ratio:    412/3480 = 0.118
safety-relevant share of edits: 36/53 = 0.679
baseline 11.4 min -> review 5.57, saved +5.83 min, 272.1 h/yr, $32,648/yr
    break-even licence: $2,721/month
baseline  6.5 min -> review 5.57, saved +0.93 min, 43.4 h/yr, $5,208/yr
    break-even licence: $434/month

threshold  flagged  TP  FP  FN  precision  recall  auto-signable
   0.40       4    3   1   3      0.750     0.500      0.895
   0.50       6    4   2   2      0.667     0.667      0.842
   0.60      11    5   6   1      0.455     0.833      0.711
   0.70      17    5  12   1      0.294     0.833      0.553
   0.80      25    6  19   0      0.240     1.000      0.342
```

Three things in the output deserve comment.

The aligner validates cleanly: two deletions, no substitutions, WER 0.25 on an eight-word
reference. That confirms the traceback, but it also shows why WER is fragile on short texts. A
single deleted word moves the rate by 12.5 percentage points.

The threshold sweep shows the trade-off the interface designer faces. At 0.50 the system flags six
sentences, catches four of six fabrications, and leaves 84.2 percent of sentences auto-signable. At
0.80 it catches all six but flags twenty-five sentences, and only 34.2 percent of the note can be
signed without attention. The precision at 0.80 is 0.240, which means three of every four flagged
sentences are supported and the reviewer learns to ignore the flags. That is the mechanism by which
a well-intentioned grounding gate produces alert fatigue.

The one fabrication scored at 0.74 is the important case. It is the fluent negative review of
systems, which shares most of its vocabulary with the transcript because the transcript contains
the words "chest", "pain", and "no" in other contexts. Lexical overlap cannot separate it. Only a
scorer that models negation and scope can, which is why the choice of grounding method is a safety
decision rather than an implementation detail.

## 8. Data, standards, and regulatory context

Ambient documentation sits at the intersection of four regulatory regimes that are often discussed
as if they were one. They are not, and a compliance argument that satisfies one can fail another.

| Instrument | Issuer | What it constrains | Status | Applies to note drafting |
|---|---|---|---|---|
| HIPAA Privacy Rule, 45 CFR 164.500-534 | US Department of Health and Human Services (HHS) | Uses and disclosures of protected health information (PHI) | Binding | Yes, audio and transcripts are PHI |
| HIPAA Security Rule, 45 CFR 164.302-318 | HHS | Administrative, physical, technical safeguards | Binding | Yes |
| Business Associate Agreement, 45 CFR 164.504(e) | HHS | Contractual obligations of a vendor handling PHI | Binding when applicable | Yes |
| State wiretap and eavesdropping statutes | State legislatures | Whether recording requires all-party consent | Binding, varies by state | Yes, for capture |
| FTC Health Breach Notification Rule, 16 CFR 318 | Federal Trade Commission | Breach notification by non-HIPAA health apps | Binding for direct-to-consumer products | Only outside the covered-entity relationship |
| Device definition, FD&C Act §201(h); 21st Century Cures §3060; CDS guidance (2022) | US Food and Drug Administration (FDA) | Whether software is a regulated device | Binding if intended use is medical | Depends on claims |
| HTI-1 DSI criterion, 45 CFR 170.315(b)(11) | Office of the National Coordinator for Health IT (ONC) | Transparency for decision support in certified health IT | Binding on certified developers | Generally not for drafting alone |
| Medicare E/M documentation rules | Centers for Medicare and Medicaid Services (CMS) | What documentation supports a level of service | Binding for payment | Yes |
| Conditions of Participation and Joint Commission standards | CMS; Joint Commission | Hospital record content and authentication | Binding for accreditation | Yes |
| EU Artificial Intelligence Act, Regulation (EU) 2024/1689 | European Union | Risk classification and obligations for AI systems | Binding, phased | Via the medical device route if applicable |
| 42 CFR Part 2 | Substance Abuse and Mental Health Services Administration | Extra consent for substance use disorder records | Binding | Yes, where applicable |

### Where the device line falls

The FDA regulates software as a device when it is intended for a medical purpose. The 21st Century
Cures Act removed certain clinical decision support functions from the device definition, and the
FDA's 2022 final guidance on clinical decision support software describes the boundary. The
practical test for an ambient product is whether it does any of the following:

- It provides a diagnosis or a treatment recommendation that the clinician is expected to rely on.
- It analyses a medical image, signal, or in vitro specimen.
- It drives a device or a closed-loop process.
- It makes claims about clinical outcomes that go beyond documentation.

A product that captions a conversation and formats a draft for clinician review, with the clinician
as the author and decision-maker, is generally positioned as administrative documentation. A
product that suggests "consider pulmonary embolism" from the transcript is making a clinical
claim, and the intended-use analysis changes. As of October 2026 the marketing position of the major
products is documentation support; the classification is a function of the manufacturer's claims,
not of the technology.

> **Note:** The FDA does not clear products by category. A statement that "ambient scribes are not
> devices" is a generalisation about current marketing, not a legal determination about a specific
> product.

### Where the ONC line falls

HTI-1 replaced the clinical decision support criterion with the decision support intervention
criterion. A predictive DSI must disclose source attributes including training data
representativeness and fairness performance in test, external, and local data, and must maintain
risk management covering fairness, validity, robustness, and safety. An ambient product that only
drafts a note from audio is not a predictive DSI. The boundary is crossed when the product produces
a prediction, classification, or recommendation that the clinician acts on, such as a risk score or
a suggested diagnosis.

The distinction has a practical consequence for a health system. A product outside the DSI
criterion escapes the disclosure requirements, which means the buyer has no standardised fairness
or validation data to compare products. The absence of a requirement is not evidence of safety.

### Consent and recording

Federal law permits one-party consent under 18 USC §2511(2)(d). State law is stricter in the nine
strict all-party states and in several hybrid states where the rule depends on the medium. The
operational advice that follows is not legal advice but risk management: announce the recording at
the start of every encounter, capture the acknowledgement in the record, allow any participant to
decline, and stop capture if someone objects mid-encounter.

Two details are commonly missed. The first is that a consent captured at check-in does not cover a
family member who joins later. The second is that consent to record is not consent to retain. A
patient may agree to a recording for documentation and decline its use for model training, and the
BAA must be able to honour that distinction at the level of an individual encounter.

### What the standards do not cover

There is no standard for ambient note quality. FHIR can carry a `DocumentReference` with the note
and a `Provenance` resource with the generating system, but FHIR does not define what counts as
adequate support for a sentence. HL7 does not define a grounding pointer. No vocabulary defines
"fabricated negative review of systems". The evaluation design in this topic exists because the
standards stop at transport.

## 9. Industry practice and case evidence

The commercial landscape in 2026 has three layers: platform vendors that bundle speech and
language services into an EHR, specialist ambient vendors that sell a dedicated product, and health
systems that build on a cloud speech service. The named products include Microsoft Dragon Copilot
(formerly Nuance DAX Copilot), Abridge, Nabla, Suki, Ambience, and platform services such as Amazon
Web Services HealthScribe. Product names are useful for orientation, not for evaluation.

### What is documented

| Study | Design | Scale | Main finding | Caveat |
|---|---|---|---|---|
| Tierney et al., NEJM Catalyst, 2024 | Pre-post with difference-in-differences | 3,442 physicians, 303,266 encounters, 10 weeks | Relative reductions in time-in-note per appointment and EHR time outside 07:00-19:00; dose-response by usage | No randomisation; self-selected users; no direct EHR integration; English only |
| Lukac et al., NEJM AI, 2025 | Three-arm pragmatic randomised trial | 238 physicians, 14 specialties | Time-in-note change −9.5 percent (95 percent CI −17.2 to −1.8) for one product; −1.7 percent (not significant) for another | Short follow-up; vendor-independent but single institution |
| Afshar et al., NEJM AI, 2025 | Stepped-wedge randomised trial | 66 practitioners, 71,487 notes | Documentation time −0.36 hours per day; work exhaustion −0.44 points | Small practitioner count; one product |
| Draper et al., BMJ Digital Health and AI, 2025 | Standardised consultations across 7 products | 8 consultations | Omissions 83.8 percent of all errors; hallucinations 7.8 percent but disproportionately severe | Simulated encounters, not live patients |
| Anderson et al., Mayo Clinic Proceedings: Digital Health, 2025 | 5 platforms, 14 simulated encounters | 14 encounters | 26.3 percent of key clinical elements omitted or wrong; about 3 errors per case with potential for moderate to severe harm | Simulated encounters; small sample |
| Biro et al., JMIR, 2025 | 2 products, real encounter scripts | 44 draft notes | 70 percent of notes contained at least one error; mean 2.9 errors per note | Scripted, not live |

The pattern across the 2025 studies is consistent. Time savings are real but smaller and more
product-dependent than early vendor material suggested. Note quality is dominated by omissions
rather than fabrications. Fabrications are rarer and more severe per event. Simulated-encounter
studies find larger error rates than vendor evaluations, which is expected because the simulations
include content the products were not tuned on.

### What is claimed but not documented

Vendor material commonly reports three numbers: a percentage reduction in documentation time, a
"note accuracy" or "acceptance rate" in the high nineties, and a clinician satisfaction score. Each
is a real measurement of something, and none is a measurement of note safety.

- A time reduction percentage depends on the baseline. The same product shows a large relative
  saving at a site with a slow baseline and a small one at a fast site, as Section 6 shows.
- An acceptance rate counts notes signed with no edits, or with edits below a threshold. A note
  that had a fabricated sentence deleted before signing can still count as accepted.
- A satisfaction score measures the user's experience, which is correlated with time saved and
  weakly correlated with accuracy.

The honest summary, and the one supported by the peer-reviewed literature, is that ambient
documentation reduces documentation time for many clinicians, is associated with modest
improvements in burnout measures, and produces notes whose dominant defect is omission. Whether
that combination is acceptable is a local risk decision.

### How health systems actually deploy

Three deployment patterns are visible in public accounts. The first is enterprise licensing with
opt-in, which produces a self-selected user base and makes evaluation difficult. The second is
specialty-by-specialty rollout, which recognises that a template-heavy specialty such as orthopaedics
and a narrative specialty such as psychiatry have different error profiles. The third is a pilot
with a formal evaluation, which is the pattern this topic recommends and the least common.

The deployment detail that matters most for a data scientist is the integration. A product that
writes into a free-text field, with no link to the audio, cannot support provenance review. A
product that writes structured elements into the chart can be audited. The integration choice
determines what evidence is available after an adverse event, and it is usually made by an
informatics team that is not in the room when the clinical champion chooses the vendor.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Fabricated negative review of systems | Draft contains negatives the clinician does not remember asking | Template convention fills required fields | False documentation with legal weight | Require a source span for every negative; leave empty fields empty |
| Omission of a spoken fact | Reviewers find missing content only on audio review | Extraction misses a concept or drafting drops it | Lost diagnosis, lost safety information | Measure recall against audio, not against the draft |
| Laterality or number error | "left" and "right" differ between sections | ASR substitution plus no cross-section consistency check | Wrong-site procedure, wrong dose | Check laterality and numbers against the transcript and across sections |
| Speaker misattribution | Family history appears in the patient's problem list | Diarization or role assignment error | Wrong patient history, wrong risk | Role confidence gating; flag low-confidence turns |
| Template filling for required fields | Every note has a complete review of systems | Generation optimises for a complete-looking note | Systematic fabrication across the population | Empty-field policy; audit completion rates by field |
| Copy-forward amplification | Draft repeats prior-note content not discussed today | Model conditioned on the previous note | Stale findings presented as current | Separate "imported" provenance; require confirmation for carried content |
| Silent ASR degradation | Mean confidence drops for a subgroup or a room | Accent, noise, or microphone change | Unequal error rates hidden in an aggregate | Monitor confidence and WER by site, device, and language |
| Abbreviation expansion error | Unusual expansions of short tokens | Context-free normalisation | Wrong concept, wrong code | Keep raw form; flag ambiguous abbreviations |
| Phase misclassification | Plan items appear in the history | Segmentation error | Wrong section, wrong billing, wrong interpretation | Evaluate section placement separately from content |
| Consent gap for a late participant | No consent event for a third speaker | Check-in consent does not cover the room | Unlawful recording | Per-session announcement; stop on objection |
| Retention beyond policy | Audio exists after the stated deletion date | Vendor default retention or backup lag | Privacy breach, regulatory exposure | Contractual deletion terms; periodic deletion audit |
| Edit log discarded | No record of what clinicians change | Product designed for output, not learning | No training signal, no audit trail | Log edits as first-class events |
| Automation bias at signature | High sign rate with low flag-open rate | Fluent interface, time pressure | Attestation without review | Measure review depth; make flags salient |
| Metric substitution | WER in the executive summary | Easy to compute, familiar | Overstated quality | Report concept error and omission rate |
| Selection bias in the time claim | Savings measured only on volunteers | Self-selected fast adopters | ROI overstated for the whole group | Intention-to-treat style analysis; report non-users |
| Template change after validation | Note structure differs from the evaluated version | Product update | Validation no longer applies | Version-lock and re-validate on change |

The red flags share a structure. Each is a case where an artefact that looks complete is treated as
evidence that the process worked. A complete review of systems looks like a thorough history. A
signed note looks like a reviewed note. A high acceptance rate looks like accuracy. A high WER
reduction looks like quality. The countermeasure in every row is the same: measure the thing the
artefact is supposed to represent, not the artefact.

Two failure modes deserve separate mention because they are properties of the deployment rather
than the model. Selection bias in the time claim is almost universal in vendor reference sites,
because sites volunteer and volunteers are enthusiasts. Template change after validation is almost
invisible, because the product updates silently in the cloud and the validation report is dated
before the change.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The transcript is a faithful record of the encounter | Grounding compares note to transcript | ASR errors, especially negation and laterality | Audio review of a random sample; concept error rate |
| Speaker roles are recoverable from audio | Attribution depends on roles | Similar voices, >2 speakers, interpreter | Role permutation test; reviewed role accuracy |
| The clinician reviews the draft | Attestation is meaningful | Time pressure, interface design, trust | Flag-open rate; review time per note |
| The baseline documentation time is stable | Time savings need a counterfactual | Seasonal variation, template changes, staffing | Time-and-motion or EHR log baseline, measured before rollout |
| The pilot population resembles the deployment population | Results are generalisable | Self-selection, specialty mix, language mix | Compare users and non-users on observables |
| Audio quality is representative | WER depends on acoustics | Room changes, device changes, telehealth | Monitor by device and site |
| Consent generalises across the encounter | One consent event covers the visit | Late-joining family, interpreters | Audit consent events against speaker count |
| The vendor's retention terms match practice | Privacy claims depend on it | Backup lag, subcontractors, model training | Contract review; deletion audit |
| Note quality is measurable by clinician rating | Evaluation needs an outcome | Rater fatigue, generosity, specialty norms | Inter-rater agreement; blinded pairs |
| The template is stable across the evaluation | Section-level metrics need a fixed structure | Product updates, specialty templates | Version lock; record template version per note |

The largest limit is the counterfactual. Every time-saving claim rests on an estimate of what the
clinician would have done without the system, and that estimate is usually a pre-period average
that includes different patient mixes and different staffing. A randomised design is the only clean
answer, and the 2025 trials show it is feasible at a single institution. A health system that cannot
randomise can still improve on the naive comparison by matching on visit type, time of day, and
clinician, and by reporting the matched estimate rather than the raw before-and-after.

The second limit is that note quality and note safety are not the same construct. A note can be
accurate, complete, well organised, and still unsafe, because the information that mattered was not
in the conversation at all. Ambient documentation cannot record what was never asked, and a rise in
note completeness can coexist with a decline in history taking if clinicians learn that the system
will fill the gaps.

## 12. Exercises

### Exercise 1. Recompute the audit

Using the counts in Section 6, recompute WER, concept error rate, all three edit measures, and the
annual time saving for a baseline of 9.0 minutes.

**Deliverable:** A one-page calculation sheet with each formula, the substituted numbers, and the
result with its unit.

**Acceptance criteria:** Every metric carries a unit; the WER and concept error rate are within
0.001 of the values in Section 6; the time saving is stated in minutes per encounter and hours per
clinician-year; the baseline assumption is stated explicitly.

**Time:** 45 minutes.

### Exercise 2. Build a grounding checker

Write a standard-library Python function that takes a list of claims and a list of transcript turns
and returns, for each claim, the best-matching turn, a token-overlap score, and a label from the set
supported, partial, unsupported. Run it on a synthetic ten-turn transcript containing one
fabricated negative and one paraphrase.

**Deliverable:** The script and its output, plus a paragraph explaining which of the two error types
your scorer misses.

**Acceptance criteria:** The script runs with no third-party imports; the output identifies the
fabricated negative; the paragraph correctly explains that lexical overlap cannot distinguish a
negated claim from a claim whose words appear elsewhere.

**Time:** 90 minutes.

### Exercise 3. Design a role permutation test

Specify a test that detects a drafter ignoring speaker roles. State the manipulation, the similarity
metric, the threshold, the sample size, and the expected result for a working and a broken pipeline.

**Deliverable:** A two-page protocol.

**Acceptance criteria:** The manipulation is defined precisely enough to reproduce; the metric has a
stated range and unit; the threshold is justified rather than asserted; the protocol says what
result would falsify the pipeline's role handling.

**Time:** 60 minutes.

### Exercise 4. Critique a vendor claim

Find a public claim from an ambient documentation vendor that reports a percentage improvement.
Identify the study behind it, the design, the denominator, and the baseline.

**Deliverable:** A 400-word critique with the claim quoted, the source cited, and a statement of what
the evidence supports and what it does not.

**Acceptance criteria:** The source is real and checkable; the critique distinguishes vendor material
from peer-reviewed evidence; the critique states the baseline and the population; no claim is
described as false without evidence.

**Time:** 90 minutes.

### Exercise 5. Audit a consent flow

A clinic operates in California and Nevada, sees patients in person and by video, and sometimes has
family members join. Design the consent flow.

**Deliverable:** A process diagram, a script for the room announcement, a data model for the consent
record, and a list of the fields the record must carry.

**Acceptance criteria:** The flow covers late joiners, interpreters, and minors; the record
distinguishes consent to record from consent to retain and from consent to train; the flow cites the
relevant state rules without asserting a legal conclusion.

**Time:** 75 minutes.

### Exercise 6. Design a note-quality review

Design a review of 100 draft and signed note pairs using the PDQI-9 plus an error taxonomy with
omission, fabrication, misattribution, and terminology.

**Deliverable:** A review protocol, a rating form, a sampling plan, and an agreement analysis plan.

**Acceptance criteria:** Two reviewers rate a overlapping subsample; agreement is reported with a
statistic and its interpretation; the sampling plan states how notes are selected; the protocol
distinguishes draft-level from signed-level quality.

**Time:** 90 minutes.

### Exercise 7. Specify the omission panel

Design the interface element that shows the clinician what the encounter did not cover, relative to
the template.

**Deliverable:** A wireframe in text, a list of fields that trigger a prompt, and the logic that
decides when a field is "not discussed" rather than "empty".

**Acceptance criteria:** The design does not generate content to fill a field; it distinguishes "not
asked" from "asked and negative"; it states how the clinician dismisses a prompt and what is logged.

**Time:** 60 minutes.

### Exercise 8. Design the time study

Design a study to estimate the documentation time effect of an ambient system in a 40-clinician
group without randomisation.

**Deliverable:** A protocol with the counterfactual, the matching variables, the primary outcome,
the analysis, and the limitations.

**Acceptance criteria:** The primary outcome is defined with a unit and a data source; the matching
strategy addresses selection; the protocol states what the design cannot rule out; the analysis
plan is written before data collection.

**Time:** 120 minutes.

## 13. Self-check answers

**Exercise 1.** WER = 56 / 1,580 = 0.0354. Concept error = 3 / 42 = 0.0714. Sentence edit rate =
6 / 38 = 0.158. Word edit rate = 53 / 612 = 0.0866. Character edit ratio = 412 / 3,480 = 0.118.
Review time = 612/240 + 53/25 + 0.40 + 0.125 × 4.0 = 2.55 + 2.12 + 0.40 + 0.50 = 5.57 minutes.
At a 9.0-minute baseline, saving = 9.0 − 5.57 = 3.43 minutes per encounter. Annual = 3.43 × 14 ×
200 / 60 = 160.1 hours per clinician-year. At 120 dollars per hour, 19,208 dollars. Note that the
result is sensitive to the baseline: the same system saves 5.83 minutes at 11.4 and 0.93 at 6.5.

**Exercise 2.** A correct answer scores each claim against each turn by token overlap and reports
the maximum. The fabricated negative ("patient denies chest pain") will usually score low if the
transcript never contains "chest pain", so this simple scorer catches it. The paraphrase ("the
patient has difficulty climbing stairs" against "it hurts going up the steps") scores low despite
being supported, so the scorer produces a false flag. The deeper failure is the one described in
Section 7: when the transcript contains the words in another context, a fluent fabrication scores
high, and lexical overlap cannot detect it. The paragraph should say that the scorer conflates
"not mentioned" with "not supported".

**Exercise 3.** A strong protocol manipulates speaker labels only, holds the transcript text fixed,
regenerates the draft with the same model and settings, and compares the two drafts with token
Jaccard similarity. A working pipeline shows similarity below roughly 0.75, because the plan and
the history swap authorship. A broken pipeline shows above 0.95. The sample should be at least 20
encounters, because single-encounter variation is large. The falsification condition is a mean
similarity above 0.95 with a confidence interval that excludes 0.75.

**Exercise 4.** A strong critique names the study, the design (pre-post, randomised, or simulated),
the population, the baseline, and the denominator. It notes whether the outcome is time, note
quality, or satisfaction, and it states that a time result does not establish safety. It does not
call a vendor claim false without evidence; it says the claim is unsupported by the cited evidence
or that the evidence supports a narrower version.

**Exercise 5.** A strong flow has a per-session announcement, not only a check-in consent; a
mechanism to stop capture when a participant objects; a consent record with fields for the person,
role, timestamp, scope (record, retain, train), and revocation; and a separate path for
interpreter-mediated encounters. It cites California and Nevada rules as inputs to the design and
states that legal review is required. It handles minors through the guardian, and it treats a
late-joining family member as a new consent event.

**Exercise 6.** A strong protocol has two reviewers on at least 20 shared notes, a PDQI-9 form with
anchors for each attribute, an error taxonomy with examples, and an agreement statistic. It reports
the sampling frame and whether the sample is random or convenience. It distinguishes draft quality
from signed quality and does not attribute a signed-note improvement to the system without
accounting for clinician edits.

**Exercise 7.** A strong design lists the template fields, the evidence required to mark a field
"asked and negative" (a transcript span), the default state for an unaddressed field ("not
discussed"), and the log of dismissals. It never generates content to fill a field. It distinguishes
three states clearly: discussed and positive, discussed and negative, and not discussed.

**Exercise 8.** A strong protocol uses a matched comparison or a difference-in-differences design
with a pre-period, matches on visit type, clinician specialty, panel size, and time of day, defines
the primary outcome as minutes per encounter from EHR logs with a stated extraction rule, and
pre-specifies the analysis. It states that selection into use is the main threat, reports the
non-user group, and does not claim causation from a before-and-after comparison.

## 14. Interview questions

**1. What is the difference between word error rate and concept error rate, and why does it
matter?**

A strong answer defines both with their denominators, gives an example where a low WER coexists
with a high concept error rate (disfluency dominates the error count while a laterality error
changes meaning), and says that the concept rate is the one that predicts clinical harm. A weak
answer treats WER as a quality metric without qualification.

**2. How would you detect a drafting model that ignores speaker roles?**

A strong answer proposes the role permutation test, names a similarity metric and a threshold,
says that the manipulation holds the text fixed, and explains what result would falsify the
pipeline. A weak answer proposes reading some notes and looking for mistakes.

**3. A vendor reports 95 percent note accuracy. What do you ask?**

A strong answer asks for the denominator (sentences, notes, or concepts), the reference standard
and who produced it, the error taxonomy, and the rate of clinically significant error. It notes
that accuracy on an imbalanced error taxonomy is uninformative. A weak answer asks only which
model is used.

**4. Why are omissions harder to measure than fabrications?**

A strong answer explains that a fabrication is visible in the draft and an omission is not, so
draft-reading evaluations under-count omissions; the fix is comparison against the audio or the
transcript rather than against the reference note. A weak answer says omissions are rarer.

**5. What does grounding guarantee, and what does it not?**

A strong answer says grounding guarantees that a sentence has a source in the encounter, which
eliminates content with no origin, and does not guarantee that the source was understood correctly
or that the patient's statement was true. A weak answer equates grounding with correctness.

**6. Walk through the consent problem for a family member who joins the visit ten minutes in.**

A strong answer separates HIPAA from state recording law, says that check-in consent does not cover
a late joiner, proposes a per-session announcement and a stop mechanism, and notes that the
consent record needs a scope field for retention and training. A weak answer says the patient
already consented.

**7. When does an ambient product stop being administrative and become a device?**

A strong answer describes the intended-use test, gives examples of claims that cross the line (a
suggested diagnosis, a risk score), and says that the classification follows the manufacturer's
claims rather than the technology. A weak answer says ambient scribes are not devices because they
only write notes.

**8. How would you evaluate the time saving without randomising?**

A strong answer describes a matched or difference-in-differences design, names the matching
variables, states the primary outcome and its data source, and states what the design cannot rule
out. A weak answer proposes a before-and-after comparison of average note time.

**9. What would you put in the audit log, and why?**

A strong answer lists the transcript with offsets, the extraction output, the grounding labels, the
edit events, the model and template versions, the consent record, and the signature, and explains
that these are what allow reconstruction after an adverse event. A weak answer lists the note and
the audio.

**10. What is the strongest argument against deploying ambient documentation in your
organisation?**

A strong answer identifies the local baseline, the review-depth risk, the omission rate, the
consent complexity, or the absence of safety monitoring, and quantifies the concern. A weak answer
says the technology is not ready, without a measurable claim.

## 15. Key terms

- **Ambient clinical documentation** — generation of a note from passively captured encounter audio; see Section 3.
- **Attestation** — the clinician's signature asserting review and agreement.
- **Automation bias** — acceptance of automated output without adequate verification; see Section 10.
- **Business Associate Agreement (BAA)** — HIPAA contract governing a vendor's handling of PHI.
- **Character edit distance ratio** — Levenshtein distance between draft and signed note divided by draft length.
- **Clinical decision support (CDS)** — software supporting clinical decisions; a regulatory category.
- **Concept error rate (CER)** — share of clinically relevant concepts wrong or missing after processing.
- **Copy-forward** — carrying prior-note content into a new note; a source of stale findings.
- **Decision support intervention (DSI)** — ONC certification category for decision support in health IT.
- **Diarization** — partitioning audio by speaker identity.
- **Draft note** — machine-generated note before clinician editing.
- **Edit provenance** — the record of which clinician changed which span.
- **Entailment** — a relation in which a source span supports a claim.
- **Grounding** — linking a generated sentence to a supporting source span.
- **Hallucination** — generated content not supported by the encounter or any record.
- **History of present illness (HPI)** — narrative account of the current problem.
- **Intended use** — the manufacturer's stated purpose; drives regulation.
- **Misattribution** — a true fact assigned to the wrong person, time, or site.
- **Omission** — a relevant fact present in the encounter but absent from the note.
- **Pajama time** — EHR work outside scheduled hours; the window must be stated.
- **PDQI-9** — nine-attribute physician documentation quality instrument.
- **Protected health information (PHI)** — identifiable health information held by a covered entity or business associate.
- **Review of systems (ROS)** — structured symptom inventory, including negatives.
- **Signed note** — the attested, filed note.
- **SOAP note** — Subjective, Objective, Assessment, Plan structure.
- **Software as a Medical Device (SaMD)** — software intended for a medical purpose without hardware.
- **Speaker role assignment** — mapping diarized speakers to clinician, patient, family, or interpreter.
- **Time-in-note** — clinician time in the note interface per encounter.
- **Two-party consent** — state rule requiring all parties to agree to recording.
- **Word error rate (WER)** — edit distance at word level divided by reference word count.

## 16. Sources

All URLs accessed 27 October 2026. DOIs are given where verified; sources without a DOI are cited
by title and venue.

1. Sinsky C, Colligan L, Li L, et al. "Allocation of physician time in ambulatory practice: a time
   and motion study in 4 specialties." Annals of Internal Medicine. 2016;165(11):753-760.
   https://doi.org/10.7326/M16-0961
2. Tierney AA, Gayre G, Hoberman B, et al. "Ambient artificial intelligence scribes to alleviate
   the burden of clinical documentation." NEJM Catalyst Innovations in Care Delivery. 2024;5(3).
   https://doi.org/10.1056/CAT.23.0404
3. Lukac S, et al. "Ambient AI scribes in clinical practice: a randomized trial." NEJM AI. 2025.
   https://doi.org/10.1056/AIoa2501000
4. Afshar M, et al. "A pragmatic randomized controlled trial of ambient artificial intelligence to
   improve health practitioner well-being." NEJM AI. 2025. https://doi.org/10.1056/AIoa2500945
5. Draper et al. "Clinical AI scribes in primary care: accuracy, error severity and implications
   for clinical practice." BMJ Digital Health and AI. 2025.
   https://doi.org/10.1136/bmjdhai-2025-000092
6. Anderson et al. "Evaluating the quality and safety of ambient digital scribe platforms using
   simulated ambulatory encounters." Mayo Clinic Proceedings: Digital Health. 2025.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12605248/
7. "Accuracy and safety of AI-enabled scribe technology: instrument validation study." JMIR. 2025.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11811668/
8. US Food and Drug Administration. "Clinical Decision Support Software: Guidance for Industry and
   Food and Drug Administration Staff." September 2022.
   https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
9. US Food and Drug Administration. "Artificial Intelligence-Enabled Device Software Functions:
   Lifecycle Management and Marketing Submission Recommendations." Draft guidance, January 2025.
10. Office of the National Coordinator for Health IT. "HTI-1 Final Rule," 89 FR 1192, 9 January
    2024. https://www.federalregister.gov/documents/2024/01/09/2023-28857
11. Office of the National Coordinator for Health IT. "HTI-1 Decision Support Interventions (DSI)
    Fact Sheet." December 2023.
    https://healthit.gov/wp-content/uploads/2023/12/HTI-1_DSI_fact-sheet_508.pdf
12. US Department of Health and Human Services. "HIPAA Privacy Rule," 45 CFR Part 160 and Part 164,
    Subparts A and E. https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164
13. "Omnibus Crime Control and Safe Streets Act," 18 USC §2511 (interception and disclosure of
    wire, oral, or electronic communications).
14. Centers for Medicare and Medicaid Services. "2021 Physician Fee Schedule Final Rule:
    Evaluation and Management documentation." 85 FR 84472, December 2020.
15. European Parliament and Council. "Regulation (EU) 2024/1689 laying down harmonised rules on
    artificial intelligence (Artificial Intelligence Act)." Official Journal of the European
    Union, 12 July 2024.
16. "AI Scribes Are Not Productivity Tools (Yet)." NEJM AI, editorial, 2025.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`10_llm-safety-in-clinical-text.md`](10_llm-safety-in-clinical-text.md)
- Next topic: [`12_icsr-and-pharmacovigilance-nlp.md`](12_icsr-and-pharmacovigilance-nlp.md)
- Text sources and provenance: [`01_clinical-text-sources.md`](01_clinical-text-sources.md)
- Documentation behaviour and copy-forward: [`02_documentation-behaviour.md`](02_documentation-behaviour.md)
- Grounded summarisation: [`09_summarization-and-grounding.md`](09_summarization-and-grounding.md)
- Evaluation design: [`07_evaluation.md`](07_evaluation.md)
- Module survey: [`../../course/09_clinical_nlp.md`](../../course/09_clinical_nlp.md)
- Labs: [`../../labs/`](../../labs/)
