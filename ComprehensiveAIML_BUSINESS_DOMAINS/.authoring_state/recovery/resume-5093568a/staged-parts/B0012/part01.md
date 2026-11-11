---
case_id: "CASE-18-02"
case_title: "Call-centre transcription quality for evidence-based objection reporting"
case_version: "2.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "hybrid"
domain: "Speech analytics / call-centre quality"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["human-in-the-loop", "authorization-scope"]
evidence_level: "illustrative"
source_basis: "constructed scenario plus derived arithmetic; no recording, vendor benchmark, or deployment result is asserted"
as_of_date: "2026-11-10"
owner_role: "Customer-operations analytics owner"
source_index: "INDEX.md"
review_trigger: "Audio channel, rubric, language route, consent policy, or review-capacity change"
---

# Call-Centre Transcription Quality: Evidence Before Aggregation

This manuscript is an `ILLUSTRATIVE` design exercise. The company, the 42,000 call legs, the corpus
statistics, the word error rates, the review timings, and every percentage below were constructed to
make the arithmetic checkable; none of them is a measurement of a real telephony estate. What is
*derived* is the arithmetic itself, the identities that relate counts to rates, and the failure
arguments. What is *open* is whether any particular vendor, language pair, or acoustic condition
matches these numbers. The reader should treat every figure as a worked example whose formula can be
re-run on real data, and should distrust any sentence in this file that reports a business outcome
without also reporting the denominator it came from.

The case concerns a single, narrow question that recurs in every speech-analytics programme: when a
customer says something that sounds like an objection, a complaint, or a refusal, how much of that
statement may a machine transcribe, and how much may an analyst then count? The answer defended here
is that transcription quality is not a property of the recogniser in isolation. It is a property of a
chain — authorised audio, stable call identity, diarised turns, span-level evidence, reviewer
adjudication, a pinned rubric version, and an aggregation denominator — and the chain fails at
whichever link is weakest, not at whichever link is newest.

## 1. The One-Line Framing

**Turn an authorised call recording into an evidence-anchored objection observation, or into an
explicit refusal to count, and never into an employee score.**

That single sentence carries four separate commitments that are easy to break independently.

The first is *authorised*: audio that has no consent or purpose record is not processed at all, and no
quality argument rescues it. The second is *anchored*: an objection count that cannot be replayed as an
audio span of bounded length is not a count, it is an impression. The third is *or refusal*: the system
must have a first-class output meaning "this call cannot support the claim", because a system that can
only emit observations will emit them for audio it cannot hear. The fourth is *never an employee score*:
the same transcript that supports a population-level objection rate would, with one join, support an
individual performance ranking, and the purpose limitation has to be built into the schema rather than
trusted to the analyst.

The comparison baseline is manual listening under an approved rubric. That baseline is not a gold
standard of truth; it is a second measurement process with its own variance, its own attention limits,
and its own tendency to record what the listener expected to hear. The case therefore does not ask
"does the model beat the humans". It asks a cheaper and more useful question: where does each process
lose information, and what would a combined process have to record so that its output can be audited
by a third party who was not present.

## 2. Context & Background

### 2.1 What the workflow actually looks like

A customer-operations team wants a monthly statement of the form "in this month, this share of
eligible calls contained a stated objection to the renewal price". Producing that statement requires
nine distinct artefacts, each of which can be absent:

1. A telephony record establishing that a call leg existed, when it started, when it ended, and which
   queue and agent handled it.
2. A capture record establishing which audio bytes were stored, on which channel, at which sample
   rate, and whether any interval was dropped by the recorder.
3. A consent or purpose record establishing that this audio may be processed for analytics, and under
   which policy version.
4. A segmentation of the audio into speech regions, silences, and overlaps.
5. A transcription of each speech region into tokens with time offsets.
6. A speaker assignment that distinguishes the caller from the agent, with an explicit confession when
   the assignment is uncertain.
7. A candidate span: the specific token range that appears to carry an objection, bound to a time
   interval that a reviewer can replay.
8. A reviewer disposition: accept, reject, reclassify, or cannot-determine, with a reason code.
9. An aggregation rule that fixes the population, the denominator, the rubric version, and the
   treatment of every non-accepted state.

A programme that automates steps 4 through 7 but leaves steps 1, 2, 3, 8, and 9 implicit has automated
the easy part and left the decision unauditable. Most of this case is about steps 8 and 9.

### 2.2 Why the obvious metrics mislead

Word error rate is the natural first metric and the least decisive one. It is a ratio of edit counts to
reference words, and it says nothing about whether the words that matter were preserved. A recogniser
can score 12% WER on a corpus and still lose the negation in "I did not agree to that", because "not"
is one word in eight and a deletion of it costs one eighth of a segment's error budget while inverting
the meaning of the span. Conversely a recogniser can score 24% WER driven entirely by filled pauses and
false starts that the objection rubric ignores, and be perfectly serviceable for the objection task.

The second misleading metric is confidence. An acoustic confidence score is a property of a token given
the model and the audio; it is not a calibrated probability that the token is correct, and it is
certainly not a probability that the span carries an objection. Programmes that threshold on mean
confidence and call the survivors "high quality" are thresholding an unvalidated proxy against an
unstated target.

The third is the aggregate itself. "9.1% of calls contained an objection" is a sentence with at least
four free parameters: which calls were eligible, which spans counted, which calls were silently
dropped, and which rubric version was applied. Change any one and the number moves by more than the
model ever will.

### 2.3 The organisational pressure

The pressure to publish a number is stronger than the pressure to bound it. A monthly report that says
"8.1% of eligible calls, 95% interval 7.4 to 8.8, 3.2% of calls unevidenced" is harder to read than
"8.1% of calls had an objection", and the second sentence survives editing while the first gets
trimmed. This case treats the trimming as the primary risk: the failure mode is not a bad model, it is
a good model whose caveats were deleted between the analyst's notebook and the executive summary.

## 3. The Business Question

The owner is the customer-operations analytics lead. The decision to be supported is whether to publish
a monthly objection rate for a defined population, and if so at what granularity. Three candidate
answers exist and the work must choose among them honestly rather than by preference.

**Answer A — publish at population level with an evidence floor.** The rate is published for the whole
eligible population, with an explicit unevidenced share, a confidence interval that accounts for
clustering by call and by agent, and no cell smaller than a stated size. This is the recommended answer
when the evidenced share is high enough that the unevidenced remainder cannot reverse the conclusion.

**Answer B — publish at queue level, no agent or team breakdown.** The rate is published per queue
because queue mix is a plausible driver and the sample supports it, but individual and team cells are
suppressed. This is the answer when queue-level differences exceed the sampling error and when the
purpose record covers queue-level analysis but not individual assessment.

**Answer C — do not publish; publish a method note and a measurement plan.** The rate is withheld
because the evidenced share is too low, or because the review queue cannot service the candidate volume
within the policy window, or because a required consent cohort is missing. The deliverable is then a
statement of what would have to change.

The question the case must answer operationally is: *what measurements, recorded at what grain, allow
an analyst to choose between A, B, and C without re-listening to the corpus?* That reframing turns an
open-ended quality question into a set of thresholds that can be computed from a call table.

Two constraints are non-negotiable and are stated as constraints rather than preferences. First, no
output may identify an individual agent's performance, because the consent record authorises
population analytics and not employment assessment. Second, no count may be published unless each
contributing call has a replayable span, because a count without a span cannot be defended in a dispute
and cannot be corrected after a rubric change.

## 4. Assumptions & Constraints

The table below is the register that a reviewer should attack first. Each row names a condition, the
check that would falsify it, the consequence if it fails, and the fallback that keeps the pipeline in a
defined state rather than an undefined one.

| ID | Assumption or limit | How it is checked | Consequence if false | Fallback state |
|---|---|---|---|---|
| A-01 | The audio is authorised for analytics under a recorded policy version | Consent/purpose lookup keyed by call ID at processing time, not at report time | Any derived count is unlawful, regardless of accuracy | `DENIED_CONSENT`; audio never read, row retained for volume accounting |
| A-02 | Call and leg identity are stable and unique | Reconciliation of telephony event count against stored recording count | Denominator inflation or double counting; spans attach to the wrong call | `IDENTITY_CONFLICT`; call excluded, reconciliation gap reported |
| A-03 | Capture bounds are known to the second | Recorder start/stop events compared with first and last speech timestamp | A "missing objection" may be an unrecorded interval | `CAPTURE_INCOMPLETE`; completeness claim withdrawn for the cohort |
| A-04 | The reviewer is blind to the model hypothesis when judging the audio | Sampled audit of console sessions showing hypothesis visibility | Human labels drift toward the machine; measured agreement is circular | Re-review of the audited subset under blind console |
| A-05 | The objection rubric is versioned and frozen per report | Report manifest records rubric hash and adjudication date | Month-over-month movement attributed to behaviour when it is definition | `RUBRIC_VERSION_MISMATCH`; cohorts reported separately, never merged |
| A-06 | Speaker role can be established for the span, or the span is not used | Role evidence recorded per span; caller-only claims rejected without it | Agent objections counted as customer objections | `ROLE_UNRESOLVED`; span routed to review, excluded from caller-only metrics |
| A-07 | A transcript token maps to a replayable time interval | Round-trip check: replay the offsets and confirm the word is audible | Evidence is decorative; disputes cannot be settled | `ANCHOR_BROKEN`; call excluded from evidence-bearing counts |
| A-08 | Review capacity is finite and known | Daily arrival versus service rate, measured weekly | Backlog becomes a sampling bias, because late calls differ | `THROTTLED`; report states achieved coverage, not intended coverage |
| A-09 | The 42,000-leg volume and its mix are representative of the month | Reconciliation against billing and queue statistics | Extrapolating rates to a population the sample does not cover | Report scoped to the measured cohort only |
| A-10 | No downstream consumer joins transcripts to HR data | Span rows carry no employee identifier; quarterly access review | Purpose limitation defeated by a single join after the fact | Purpose review escalation; access revoked pending decision |

The arithmetic in this file uses a single `ILLUSTRATIVE` intake: 42,000 call legs offered in the
reporting month, of which 38,000 are eligible after removing duplicates, zero-length legs, and legs
with no consent record. Those two numbers are constructed for the sake of the ratios and must be
replaced by the real reconciliation before any of the percentages below are quoted.

Three further constraints bound the method rather than the data. The first is that the reviewer's
attention is the scarce resource: at a constructed 22 minutes per candidate span, a 1,930-span
adjudication consumes 707.7 reviewer-hours, which is 4.3 full-time equivalents at 165 productive hours
a month. Any design that increases candidate volume without increasing precision is a capacity
regression, not an accuracy improvement. The second is that the audio never leaves the approved
processing boundary, so all model outputs must be accompanied by offsets rather than by copied audio.
The third is that the report must be reproducible from stored artefacts alone, which forbids any step
whose only record is an analyst's memory of a listening session.

## 5. Data, Tools & Method

### 5a. Data Specification

**Grain.** There are four grains, and they must not be confused because each carries a different
denominator:

- `call_leg` — one row per connected leg. The reporting denominator lives here.
- `turn` — one row per contiguous speech region attributed to one hypothesised speaker, bounded by
  silence or speaker change. Turns are the unit of diarisation error.
- `token` — one row per transcribed word with start and end offsets in milliseconds relative to the
  leg's audio epoch.
- `span` — one row per candidate objection, a token range plus rubric version, role evidence, and
  reviewer disposition. Spans, not calls, are the unit of evidence precision.

**Cut-off and clocks.** The extraction cut-off is the last moment at which a leg may be included. Three
clocks run in this system and must be named separately: *event time* (when the call happened), *ingest
time* (when the audio became readable), and *review time* (when a disposition was recorded). A call that
happened inside the month but was ingested after the cut-off is a late arrival; a call that happened
inside the month and was reviewed after the report freeze contributes a pending state, not an absence.
Mixing the clocks is the single most common source of month-over-month movement that has nothing to do
with customers.

**Population.** Eligible legs: connected, recorded, consent recorded, duration at least 30 seconds, and
attached to a queue in scope. Every exclusion is counted in its own bucket so that the eligible count is
reconstructible as `offered − duplicates − zero_length − no_consent − out_of_scope_queue`.

**Keys.** `call_id` (stable), `leg_id` (stable within call), `recording_hash` (content identity),
`policy_version`, `rubric_version`, `asr_version`, `diarizer_version`, `reviewer_id` (hashed), and
`disposition_id`. The recording hash is what makes a span reproducible: if the audio is re-encoded the
hash changes and every stored offset must be re-validated against the new media, not merely re-linked.

**Units.** Audio offsets in integer milliseconds; durations in seconds for reporting but milliseconds
for arithmetic; rates as proportions with four decimal places internally and one decimal place in
publication; cost in reviewer-minutes; intervals in percentage points.

**What is deliberately not available at decision time.** The transcript of a call that has not yet
reached the cut-off; the disposition of a span that is still queued; the true speaker role, which is a
hypothesis until adjudicated; and the eventual business outcome of the call. A model trained on any of
these and evaluated as if it were not is leaking the future into the present. The rule adopted here is
that every feature must be available at span creation time, and every label must post-date it.

### 5b. Tools & Stack

The stack is described by contract, not by product, because the argument must survive a vendor swap.

- **Authorised audio reader.** Preserves sample rate, channel mapping, gap markers, and the source
  hash. It refuses to read any leg whose consent lookup fails, and the refusal is logged with the
  policy version so that a later consent change can be re-evaluated without replaying the audio.
- **Voice activity detector.** Emits speech regions with a declared hangover and a minimum region
  length. Its output is an input to segmentation, not a truth about speech.
- **Recogniser.** Emits tokens with start/end offsets and a per-token acoustic score. The score is
  retained as a diagnostic and is never used as a calibrated probability without an explicit
  calibration study.
- **Diariser.** Emits speaker segments with a role hypothesis and a stability flag. Two-speaker
  telephone audio is the easy case; transfers, hold music, and speakerphone rooms are the hard ones.
- **Evidence binder.** Joins a token range to a time interval, a recording hash, and a rubric version,
  and validates the interval by replay. This component owns the `ANCHOR_BROKEN` state.
- **Review console.** Plays a bounded window around the span, shows the rubric definition, hides the
  model hypothesis on blind sessions, and forces a reason code on every non-accept disposition.
- **Aggregator.** Accepts only states permitted by the reporting policy for the chosen answer, computes
  the rate and the interval, and emits a manifest containing every version string.

### 5c. Method Summary

1. **Gate before reading.** Resolve consent, purpose, and call identity; write a decision row for every
   leg including the refusals. This step produces the denominator, so it must run before any
   transcription and must be complete even for legs that are never read.
2. **Segment with uncertainty retained.** Record speech regions, silences over a stated threshold, and
   overlaps. Overlapped speech is not discarded; it is marked, because a discarded overlap becomes an
   invisible gap in the denominator.
3. **Transcribe and diarise.** Produce tokens and turns with versions recorded. Nothing downstream may
   use a token whose offsets are absent.
4. **Propose spans.** Match rubric patterns over tokens to produce candidate spans, each with the time
   interval, the role hypothesis, and the pattern that fired.
5. **Adjudicate.** Reviewers accept, reject, reclassify, or mark undetermined, with a reason. The
   console is blind on audited sessions. Dispositions are the label source; model output is never the
   label.
6. **Aggregate under a pinned rule.** Fix population, denominator, rubric version, and treatment of
   undetermined and pending spans. Compute the rate, the interval, and the unevidenced share together,
   because a rate published without its unevidenced share is not interpretable.
7. **Publish with the manifest.** Every number travels with its versions, its coverage, and its
   limitation statement. A number that cannot travel with those is not published.

### 5d. Provenance, Quality, Exclusions, Leakage

**Lineage.** Every span resolves to a recording hash, an ASR version, a diariser version, a rubric
version, and a disposition. The chain is stored, not reconstructed, because reconstruction after a
version change is an inference and inferences fail silently.

**Exclusions, each in its own bucket.** Unauthorised audio; incomplete capture; unresolved identity;
unsupported language route; corrupted media; and calls shorter than the minimum duration. The buckets
must sum, together with the included count, to the offered count. A report that shows only the included
count has thrown away its own audit trail.

**Quality checks on the input side.** Channel presence on both directions; duration consistency between
telephony events and media; speech-region coverage, meaning the share of audio inside detected speech;
and the share of legs whose diariser emitted more than two speakers, which is a strong signal of a
transfer or a bad channel map.

**Leakage controls.** Splits are by caller and by call, never by turn, because two turns from the same
conversation share acoustics, vocabulary, and the same underlying objection. Where a caller appears
more than once, all their calls go to one side of the split. Temporal splits are used for the
production estimate: the rubric is tuned on earlier months and reported on a later month, so that the
reported rate is not a fitted quantity. When a device or queue changes mid-month, the cohort is
separated rather than mixed.

**The specific leak this case had to close.** Reviewers originally saw the model hypothesis beside the
audio. Under that console, agreement between model and reviewer was high and meaningless: the reviewer
was reading a suggestion and confirming it. The fix was a blind console plus a sampled audit of console
sessions; the measured effect of blindness is reported in section 9 as a design parameter, not as a
finding about any real reviewer.

### 5e. Reproducibility & Method-Selection Ledger

| Decision | Alternative rejected | Why rejected | What would reopen it |
|---|---|---|---|
| Micro word error rate over pooled counts | Macro mean of per-call rates | Macro gives a 40-second call the same weight as a 12-minute call, so a handful of short calls dominate | If the reporting question becomes "typical call quality" rather than "corpus accuracy" |
| Span-level precision and recall as the primary quality pair | Utterance-level accuracy | Utterance accuracy hides whether the one decisive word survived | If the downstream task stops depending on a specific phrase |
| Reviewer disposition as the label | Model output as the label | Circular; a model scored against itself always passes | Never |
| Diarisation error measured in seconds of turn misassignment | Speaker accuracy per turn | Turn-level accuracy rewards long turns and hides boundary error | If role errors prove to be turn-local rather than boundary-local |
| Rate published with an unevidenced share | Rate published alone | A rate with an unknown unevidenced remainder cannot be defended | If unevidenced share falls below the materiality threshold for three consecutive months |
| Clustered interval by call and agent | Binomial interval over spans | Spans within a call are correlated; a binomial interval is too narrow | If the design effect is measured near 1.0 on the real corpus |

Reproduction procedure, stated so that a second analyst can repeat it: take the frozen span table for
the month, filter to disposed spans, recompute the rate from counts rather than from stored rates,
recompute the interval from the cluster structure, and compare the manifest's version strings against
the stored artefacts. Any mismatch is a defect in the report, not a rounding difference. The
recomputation must produce the published rate to the published precision; if it does not, the report is
withdrawn and reissued rather than annotated.
