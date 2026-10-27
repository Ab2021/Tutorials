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
