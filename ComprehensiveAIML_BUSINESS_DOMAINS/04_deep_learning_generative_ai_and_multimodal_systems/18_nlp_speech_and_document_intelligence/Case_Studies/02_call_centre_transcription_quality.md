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
## 6. Analysis Walkthrough

Every number in this section is computed from the `ILLUSTRATIVE` intake defined in section 5a and is
recomputed here rather than quoted from a stored rate. The corpus is 38,000 eligible legs with a
constructed mean duration of 384 seconds, which is 6.4 minutes, so the total audio is
38,000 × 384 / 3,600 = 4,053.3 hours. At a constructed 672 reference words per leg the reference corpus
holds 25,536,000 words; the scored evaluation sample is a 500-leg draw from it, 336,000 reference words,
which is the only corpus on which a word error rate is claimed.

### 6.1 Derivation one: the edit decomposition hides what the rubric cares about

```text
FORMULA [D1]: WER = (S + D + I) / N ;  accuracy = 1 - WER
WHERE:  S = substitutions, D = deletions, I = insertions (token counts, dimensionless integers)
        N = reference words after normalisation (count)
CONSTRAINT: N > 0; identical normalisation on both sides; unit edit cost; one scoring path policy
NUMERICAL EXAMPLE: S = 32,256 ; D = 8,064 ; I = 12,096 ; N = 336,000
        edits = 32,256 + 8,064 + 12,096 = 52,416
        WER = 52,416 / 336,000 = 0.15600 = 15.60%
        decomposition: 9.60% substitution + 2.40% deletion + 3.60% insertion
INTERPRETATION: 15.60% of reference words were edited, and the three components have
        different downstream consequences: deletions lose customer speech, insertions
        invent agent or caller speech, substitutions change one word into another.
```

The decomposition is more useful than the total because the three categories fail differently for an
objection rubric. A deletion removes evidence: the customer said something and the transcript does not
contain it, so no span can be proposed and the leg looks negative. An insertion manufactures evidence:
the recogniser emits a plausible phrase that never existed, a span is proposed on words nobody spoke,
and a reviewer who replays the audio rejects it, consuming review capacity to discover nothing. A
substitution corrupts evidence in place: the span exists, the words are nearly right, and the rubric
may fire on the wrong side of a negation.

At 8.06% of the 15.60% traced to disfluency handling, the largest single lever is normalisation. The
scored sample contains 33,000 filled pauses, false starts, and backchannels. Removing them from both
sides of the alignment removes 28,000 edits:

```text
removed tokens = 336,000 - 303,000 = 33,000
removed edits  =  52,416 -  24,416 = 28,000
share of all edits attributable to disfluency = 28,000 / 52,416 = 0.5342
raw WER         = 52,416 / 336,000 = 15.60%
normalised WER  = 24,416 / 303,000 =  8.06%
```

The interpretation must be stated narrowly, because this is exactly the number a programme is tempted
to publish. The 8.06% figure is correct for the declared normalisation and says nothing about whether
any decisive word survived. It is the rate a reader sees if the report quotes a normalised WER and
omits the normalisation policy, and the omission is what makes the number misleading rather than the
arithmetic.

### 6.2 Derivation two: span precision depends on how undetermined dispositions are counted

Flagged legs produced candidate spans, and each span received one of four dispositions from the
reviewer. The counts on the 4.24%-flag-rate design are:

| Disposition | Count | Meaning for evidence |
|---|---|---|
| accepted | 1,196 | the span is a real objection, role confirmed, replayable |
| rejected | 486 | the span is not an objection, or the audio does not support the words |
| reclassified | 163 | real speech, wrong category; usable for a different rubric, not this one |
| undetermined | 85 | reviewer could not decide; audio unclear, overlap, or rubric ambiguity |
| total | 1,930 | all spans that consumed reviewer time |

```text
FORMULA [D2]: precision_strict = A / (A + R + C + U)
              precision_broad  = (A + C) / (A + R + C + U)
WHERE: A = accepted, R = rejected, C = reclassified, U = undetermined (span counts)
CONSTRAINT: every span carries exactly one disposition; denominator is the full adjudicated set
NUMERICAL EXAMPLE: A = 1,196 ; R = 486 ; C = 163 ; U = 85 ; total = 1,930
        strict = 1,196 / 1,930 = 0.6197 = 61.97%
        broad  = 1,359 / 1,930 = 0.7041 = 70.41%
INTERPRETATION: the choice of numerator is worth 8.44 percentage points, which is larger
        than any model upgrade available at this stage, so the numerator must be declared.
```

The third variant is the one that reveals the policy question. Excluding the 85 undetermined spans
from the denominator gives 1,196 / 1,845 = 64.82%, which is defensible only if undetermined spans occur
at random with respect to being true objections. They do not: undetermined spans are concentrated in
overlapped speech and in transfers, which is exactly where a real objection is hardest to hear and
therefore where the true rate among undetermined spans is plausibly *higher* than among resolved ones.
Excluding them therefore flatters precision. Counting them as negative, which is the strict formula,
is the conservative choice and is the one adopted.

Recall was measured on a separate audit and cannot be recomputed from the same table, because the
denominator is human work rather than model output. In a 150-call audit with 63 hand-found objections,
the model had flagged 47 of them:

```text
recall = 47 / 63 = 0.7460 = 74.60%
F1     = 2 × (0.7041 × 0.7460) / (0.7041 + 0.7460) = 2 × 0.5253 / 1.4501 = 0.7245
```

Recall's denominator is the honest one and its cost is the reason it is measured rarely. Finding 63
objections required listening to 150 calls end to end, which at 6.4 minutes each is 960 listening
minutes, or 16 hours, to establish a recall estimate with a confidence interval wide enough that the
interval must itself be reported. A 63-item denominator cannot distinguish 70% recall from 80% recall.

### 6.3 Derivation three: the rate, the correction, and why the naive rate is wrong

The naive aggregate counts only evidenced objections and divides by all eligible legs. That mixes two
different populations: legs the model looked at and flagged, and legs the model looked at and passed
over. The correction re-weights:

```text
FORMULA [D3]: r̂ = (O_flag + (L - L_flag) × p̂_unflag) / L
WHERE: L = eligible legs (38,000)
       L_flag = legs carrying at least one flagged span (1,612)
       O_flag = legs with at least one accepted objection (1,082)
       p̂_unflag = observed objection rate in a random sample of unflagged legs (16/446)
CONSTRAINT: L_flag > 0; audit sample drawn without replacement from unflagged legs; sample
       representative of unflagged legs; a leg counts once regardless of span count
NUMERICAL EXAMPLE: L - L_flag = 36,388
       p̂_unflag = 16 / 446 = 0.035874
       estimated missed objections = 36,388 × 0.035874 = 1,305.4 legs
       r̂ = (1,082 + 1,305.4) / 38,000 = 2,387.4 / 38,000 = 0.06283 = 6.28%
       naive evidenced-only rate = 1,082 / 38,000 = 0.02847 = 2.85%
INTERPRETATION: the naive rate is 45.3% of the corrected rate, so reporting the evidenced
       count alone understates the phenomenon by a factor of 2.21, and it does so in the
       direction that makes the operation look better.
```

That factor is the case's central quantitative result. It is not a model-quality result at all: it
follows from coverage, not accuracy. Improving span precision from 70.4% to 90% would add 1,930 × 0.196
= 378 accepted spans, which at 1.197 spans per flagged leg is about 316 legs, whereas the unreviewed
population holds an estimated 1,305 objection legs. The programme's information about the phenomenon is
therefore limited primarily by what it never listens to, and any investment ordering that puts model
accuracy ahead of coverage is misallocated.

The interval must reflect two independent uncertainties, because r̂ is a sum of a census-like term and
an estimated term:

```text
sampling component: var(r̂) ≈ p(1-p)/L × deff + ((L - L_flag)/L)² × var(p̂_unflag)
deff = 1.8 (design effect from clustering spans within calls)
var(p̂_unflag) = p̂(1-p̂)/n × fpc,  fpc = (36,388 - 446)/(36,388 - 1) = 0.9878
             = 0.035874 × 0.964126 / 446 × 0.9878 = 7.6522e-5
se(p̂_unflag) = 0.008747 → contribution to r̂ = (0.95758) × 0.008747 = 0.008376
se(census term) = sqrt(0.06283 × 0.93717 / 38,000 × 1.8) = sqrt(2.7899e-6) = 0.001670
se(r̂) = sqrt(0.001670² + 0.008376²) = 0.008541
95% interval = 0.06283 ± 1.96 × 0.008541 = [0.04609, 0.07957] = [4.61%, 7.96%]
```

The audit term dominates: 0.008376 of a total standard error of 0.008541, which is 96.4% of the
variance. Enlarging the unflagged audit from 446 to 1,000 legs would roughly halve that component and
narrow the interval to about ±0.9 percentage points, whereas a better recogniser would barely move it.
This is a direct, computable statement about where the next unit of effort belongs.

### 6.4 Derivation four: reviewer capacity, and why the candidate set must be small

```text
FORMULA [D4]: H_review = n_span × m_span / 60 ;  FTE = H_review / h_month
WHERE: n_span = candidate spans adjudicated (count); m_span = mean minutes per span
       h_month = productive reviewer hours per month (165)
CONSTRAINT: m_span measured on the real console, including replay, rubric lookup and
       reason-code entry; excludes queue wait and training time
NUMERICAL EXAMPLE: 1,930 × 22 / 60 = 707.67 hours ; 707.67 / 165 = 4.29 FTE
       cost per evidenced objection = 707.67 / 1,082 = 0.654 hours = 39.2 minutes
       full-census alternative = 38,000 × 6.4 / 60 = 4,053.3 listening hours = 24.6 FTE
INTERPRETATION: reviewed flagging costs 4.3 FTE against 24.6 FTE for a census, so the
       machine-assisted route is 5.7× cheaper in labour, and each evidenced objection
       costs 39 minutes of human time.
```

The comparison is not "machine versus human" but "sampled human review of machine candidates versus
exhaustive human review". The exhaustive route has known and near-perfect recall and no model error; it
simply costs 5.7 times more and does not scale with volume. The machine route is only better if its
missed objections can be estimated, which is what the 446-leg audit does. Remove the audit and the
machine route has neither exhaustive recall nor a measurable one, which is the failure described in
section 12.

Compute cost is a rounding error against review cost and should be reported as such. At a
constructed real-time factor of 0.02 for recognition and the same again for diarisation, total
processing is 2 × 4,053.3 × 0.02 = 162.1 GPU-hours, against 707.7 reviewer-hours. The ratio is 4.4:1,
which means that any decision to reduce compute by sampling audio before transcription is a false
economy unless it also reduces candidate volume: transcribing more audio and reviewing fewer spans is
strictly better than the reverse.

### 6.5 A boundary where D3 breaks

The correction in D3 assumes the 446-leg audit is representative of the 36,388 unflagged legs. The
formula has no term that can detect a violation of that assumption, and the violation is not exotic.
Suppose the audit was drawn 60% from business hours and 40% out of hours, because reviewers work
business hours, while out-of-hours legs are actually 22% of the unflagged population. Suppose the true
objection rate is 6.00% in daytime and 1.20% out of hours, because out-of-hours callers are renewing
rather than disputing.

```text
audit-weighted p̂      = 0.60 × 0.0600 + 0.40 × 0.0120 = 0.04080
population-weighted p = 0.78 × 0.0600 + 0.22 × 0.0120 = 0.02256
r̂ from audit          = (1,082 + 36,388 × 0.04080) / 38,000 = 2,566.6 / 38,000 = 0.06754
true rate            = (1,082 + 36,388 × 0.02256) / 38,000 = 1,902.9 / 38,000 = 0.05008
bias                  = 0.06754 - 0.05008 = +0.01747, i.e. 1.75 percentage points too high
D3 interval           = [0.05084, 0.08424], which excludes the true 0.05008
```

The bias is slightly larger than the interval half-width of 1.67 points, so the published interval does
not merely fail to be tight, it excludes the truth. This is the failure mode that destroys trust in a
measurement programme: the arithmetic is impeccable, the interval is honestly computed, and the number
is still wrong in a direction that happens to be interesting. Nothing in the data reveals it; only the
sampling design does.

The validity condition of D3, stated plainly, is that the audit is a probability sample of the unflagged
population, stratified on every variable known to predict objections. The discriminating check is a
stratification comparison: compute p̂_unflag separately by hour band, queue, language route and channel,
then compare the observed spread against each cell's binomial standard error. At n = 446 split four
ways, cells of about 110 legs have a binomial standard error of sqrt(0.04 × 0.96 / 110) = 1.87 points,
which cannot resolve a 4.8-point design bias. That is why the check must be a design requirement — the
audit is drawn by stratified probability sampling with fixed cell quotas — rather than a diagnostic run
after the fact.

A second boundary arises at low denominators. A leg with 6 reference words and 3 edits has
WER = 3/6 = 50%, and a corpus of such legs can report an alarming micro WER driven by legs whose entire
content is "yes, speaking". The micro formula is arithmetically correct and substantively useless there,
because the reference denominator carries no information. The guard is a minimum reference length per
included leg, declared in advance, with excluded legs counted in their own bucket.

## 7. The Finding

**The binding constraint on this programme is coverage, not recognition accuracy, and the measured
quantity that exposes it is the ratio between the naive rate and the corrected rate.**

Four results support that sentence and each is falsifiable.

First, span precision of 70.4% at the broad numerator and recall of 74.6% on a 63-item audit is
adequate for population reporting and inadequate for individual casework. That asymmetry is structural:
a population rate tolerates per-span error because errors partially cancel, while a single customer's
dispute does not. The correct conclusion is not "improve precision to 90%" but "keep the evidence route
for population claims and keep humans on individual cases".

Second, the evidenced-only rate of 2.85% and the corrected rate of 6.28% differ by a factor of 2.21,
and the difference is entirely attributable to legs nobody reviewed. Any report published without the
correction is not a conservative estimate of the truth, it is an estimate of a different quantity, and
it will move in the wrong direction whenever review capacity tightens. That coupling between operations
pressure and a published business metric is the most dangerous property the design has, because the
metric degrades precisely when the team is asked to justify itself.

Third, the uncertainty budget is almost entirely in the audit: 96.4% of the variance of the corrected
rate comes from the 446-leg sample of unflagged legs. This inverts the usual investment logic. A model
upgrade that halves edits per word moves the published rate by less than a rounding step, while doubling
the audit sample moves the interval from ±1.7 points to ±0.9 points. The programme's measurement
priority follows from its variance decomposition, not from its tooling roadmap.

Fourth, the cost structure is review-dominated. At 39.2 reviewer-minutes per evidenced objection and
4.4:1 review-to-compute hours, the operating question is span precision per reviewer-minute, not word
error rate. A change that improves WER while adding candidates is a net loss, and a change that adds
candidates at equal precision but concentrates them in under-covered segments is a net gain.

The finding therefore resolves the business question in section 3 to **Answer A with a mandatory
correction term**, conditional on three gates that the evidence in section 9 either passes or fails:
the audit must be a stratified probability sample; the unevidenced share must be published alongside
the rate; and no cell below 250 legs may be published. If the audit cannot be drawn as a probability
sample, the correct output is Answer C, because a corrected rate resting on a convenience sample is
worse than no rate: it carries the authority of arithmetic without the validity of a sample.
## 8. System/Solution Design

### 8.1 The state machine is the design

The system's behaviour is defined by a per-span state machine with exactly seven states, and the point
of the design is that no state can be skipped and every non-terminal state has an owner.

| State | Entered when | Owner | Allowed next states |
|---|---|---|---|
| `OFFERED` | leg appears in the telephony extract | pipeline | `DENIED_CONSENT`, `IDENTITY_CONFLICT`, `INELIGIBLE_SHORT`, `SEGMENTED` |
| `SEGMENTED` | VAD produced regions and the reader verified the media hash | pipeline | `TRANSCRIBED`, `CAPTURE_INCOMPLETE` |
| `TRANSCRIBED` | tokens carry offsets and versions | pipeline | `SPAN_CANDIDATE`, `NO_CANDIDATE` |
| `SPAN_CANDIDATE` | at least one rubric pattern fired above the score floor | pipeline | `QUEUED`, `THROTTLED` |
| `QUEUED` | a review slot is allocated | review ops | `DISPOSED` |
| `DISPOSED` | reviewer recorded a disposition and reason code | reviewer | terminal |
| `THROTTLED` | capacity exhausted before allocation | review ops | `QUEUED` (when capacity returns) |

Three properties follow from this table and each closes a specific failure. First, `THROTTLED` is a
state rather than a silent drop, so throttled legs remain countable and the report can state achieved
coverage. Second, `NO_CANDIDATE` is distinguished from `SPAN_CANDIDATE`-then-`THROTTLED`, so a low flag
rate caused by a weak detector is never confused with a low flag rate caused by a full queue. Third,
`DISPOSED` is terminal and immutable; a correction requires a new disposition row referencing the
superseded one, which preserves the audit trail that an in-place update would destroy.

The refusal states — `DENIED_CONSENT`, `IDENTITY_CONFLICT`, `CAPTURE_INCOMPLETE` — are outcomes of the
same machine, not exceptions. That is deliberate: an exception path is a path nobody instruments, and an
uninstrumented refusal is indistinguishable from a leg that was never offered.

### 8.2 Span proposal and the score floor

Candidates are produced by matching versioned rubric patterns against tokens, where a pattern is a
phrase alternation plus a negation rule. The negation rule matters more than the phrase list. A pattern
"too expensive" fires on the utterance, but the same tokens fire inside "I never said it was too
expensive", which is a denial rather than an objection. The matcher therefore looks in a window of three
tokens before the phrase for negators and counterfactual markers, and it emits the span with a
`negated` flag that the reviewer sees.

The score floor is a single tunable and its sweep is the most useful artefact the programme produces,
because it converts a modelling preference into a capacity decision.

| Score floor | Candidate spans | Precision (broad) | Accepted spans | Recall on 63-item audit | Review hours | Reviewers (FTE) |
|---|---|---|---|---|---|---|
| 0.30 | 3,090 | 0.61 | 1,885 | 51/63 = 0.810 | 1,133 | 6.87 |
| 0.50 | 1,930 | 0.704 | 1,359 | 47/63 = 0.746 | 708 | 4.29 |
| 0.70 | 1,410 | 0.79 | 1,114 | 44/63 = 0.698 | 517 | 3.13 |
| 0.90 | 980 | 0.86 | 843 | 33/63 = 0.524 | 359 | 2.18 |

The table is `ILLUSTRATIVE` but internally consistent: as the floor rises, volume falls, precision rises,
and recall falls. The decision it supports is not "pick the best row". It is "pick the row whose recall
supports the claim you intend to make". For a population rate with a correction term, 0.50 is defensible
because missed objections are estimated by audit. For any claim of the form "we found all the objections
in this cohort", no row is defensible, because even 0.30 recall of 81.0% leaves 12 of 63 objections
unfound in a 150-call audit, and the audit's own interval on that recall is 0.69 to 0.89.

The cost column is what makes the sweep actionable in a business review. Moving from 0.50 to 0.70 saves
191 review hours and 1.16 FTE but costs 3 audit-found objections out of 63, which raises the estimated
missed-objection count in the unflagged population by roughly 3/63 × 36,388 × p̂. That trade is a
decision for the owner, not for the engineer, and presenting it as a table rather than as an accuracy
number is what allows the owner to make it.

### 8.3 Evidence binding and the replay contract

A span is only usable if a reviewer can hear it. The binder stores `start_ms`, `end_ms`, the
`recording_hash`, and the token indices; the console plays from `start_ms − 8,000` to `end_ms + 4,000`,
clamped to the media bounds and capped at 20,000 ms. For a 384-second leg the mean replay window is
about 13 seconds, which is 3.4% of the call; the review cost is not dominated by replay but by
contextual listening, which is why the 22-minute mean per span holds.

The replay contract has two failure tests, both run before a span enters `QUEUED`. The **audibility**
test decodes the window and confirms the energy profile is consistent with speech rather than silence;
this catches offset drift after re-encoding. The **hash** test confirms the media the binder referenced
is the media being played; this catches the case where a storage migration changed the container and
invalidated every stored offset. Failing either test sets `ANCHOR_BROKEN` and the leg leaves the
evidence-bearing count, which is a visible loss rather than a quiet one.

### 8.4 Review sampling design

The 446-leg audit of unflagged legs is a stratified probability sample with fixed quotas, drawn
independently of the reviewer's queue. Strata are queue family, hour band, language route, and channel
count. Allocation is proportional with a floor of 25 legs per stratum, and the estimator is the
stratum-weighted mean:

```text
FORMULA [D5]: p̂_strat = Σ_h (N_h / N) × (k_h / n_h)
WHERE: h indexes strata; N_h = unflagged legs in stratum h; n_h = audited legs in stratum h
       k_h = audited legs with a hand-found objection
CONSTRAINT: every n_h ≥ 1 and strata cover the unflagged population exactly;
       sampling within stratum is without replacement and independent of content
NUMERICAL EXAMPLE: two strata only, N_A = 28,383 (78%), N_B = 8,005 (22%)
       n_A = 268, k_A = 16 → 0.0597 ; n_B = 178, k_B = 2 → 0.0112
       p̂_strat = 0.78 × 0.0597 + 0.22 × 0.0112 = 0.04657 + 0.00246 = 0.04903
       unweighted pooled estimate would be 18/446 = 0.04036, which is 0.87 points lower
INTERPRETATION: if the strata differ in size and in rate, the unweighted sample mean is
       biased toward the oversampled stratum, so stratification weights are mandatory.
```

The design consequence is that the audit is planned before the month starts, with its quotas and its
cost. At 446 legs × 6.4 minutes = 2,858 listening minutes, the audit is 47.6 reviewer-hours, or 0.29
FTE — small against the 708 hours of candidate adjudication it corrects, which is exactly why it should
never be the thing cut first.

### 8.5 Access, purpose, and the schema as control

The purpose limitation is enforced structurally rather than by policy text. Span rows carry `call_id`,
`leg_id`, `queue_id`, and `reviewer_id` (hashed), and no `agent_id`. Agent identity exists in the
telephony system and is joined only in the queue-level aggregation job, which emits counts per queue
and never per agent. A team-level table would require a new view, a new purpose record, and a new access
review, which makes the prohibited use an explicit project rather than a query.

The transcript table is partitioned by month and by queue, so an analyst with queue-level access
physically cannot read another queue's text. Reviewer access is scoped to the queue they serve, and the
`reason_code` vocabulary is closed: adding a code is a schema change with a review, because reason codes
drift into free text and free text drifts into commentary about employees.

### 8.6 Operational protocol

The monthly cycle is a fixed sequence with explicit gates, run whether or not the numbers look good.

1. **Day 1** — freeze the extract at the event-time cut-off; reconcile offered against billed legs and
   publish the reconciliation gap before any rate.
2. **Day 2** — run gating, segmentation, transcription, and diarisation; write version strings into the
   manifest. Any leg whose pipeline run failed is retried once, then recorded as failed.
3. **Day 3** — generate candidates, apply the pinned score floor, and compute the capacity requirement.
   If the requirement exceeds available reviewer hours, apply the throttle rule and record the throttled
   count; never lower the floor silently to make the queue fit.
4. **Days 3–8** — adjudicate, with 5% of spans double-reviewed blind to measure reviewer agreement.
5. **Day 9** — draw and complete the stratified audit of unflagged legs.
6. **Day 10** — compute the corrected rate, the interval, the unevidenced share, and the per-queue
   cells; suppress any cell below 250 legs; write the manifest.
7. **Day 11** — publish, with the limitation statement attached to the table rather than in a footnote.

The protocol's value is that each gate has an owner and a timestamp, so a delay is attributable. The
most common real defect in such programmes is not a wrong number but a number published before the
audit completed, which is precisely what gate 5 prevents.

## 9. Evaluation & Validation

Evaluation is stated in eight fields because a metric without them cannot be audited. Everything in this
section is either `executed` on the constructed data or explicitly `not-executed`; no partially executed
item is described as done.

### 9.1 Definitions

| Field | Objection-span precision | Objection recall | Corrected rate |
|---|---|---|---|
| Numerator | accepted + reclassified spans (1,359) | audit objections the model flagged (47) | (1,082 + 1,305.4) legs |
| Denominator | all adjudicated spans (1,930) | audit objections found by humans (63) | eligible legs (38,000) |
| Population | 1,612 flagged legs, one month, in-scope queues | 150 randomly drawn flagged and unflagged calls | 38,000 eligible legs |
| Window | span month by event time | same month | same month, event time |
| Comparator | reviewer-with-model-visible console (retired) | human exhaustive review of the 150 calls | evidenced-only naive rate (2.85%) |
| Guardrails | 5% blind double-review agreement; audibility tests; throttled count | recall interval width | unevidenced share; cell suppression |
| Uncertainty | Wilson 95%: 0.683–0.724 | Wilson 95%: 0.627–0.837 | clustered 95%: 4.61%–7.96% |
| Status | executed (constructed) | executed (constructed) | executed (constructed) |

The comparator column is the part most programmes omit and it is the part that makes the number
meaningful. A precision of 70.4% is neither good nor bad in isolation; it is better than the 61.9%
achieved when undetermined spans were excluded from the denominator and worse than the 79% achieved at
a 0.70 floor, and both of those comparisons are more informative than the absolute value.

### 9.2 Guardrail measurements

**Blind double-review agreement.** Two hundred spans were adjudicated twice, once with the hypothesis
hidden and once with it visible, by different reviewers. Raw agreement was 150/200 = 75.0%; the expected
agreement under independence, given the marginals 142/200 and 132/200 positive, is
(142/200)(132/200) + (58/200)(68/200) = 0.4686 + 0.0986 = 0.5672; so

```text
κ = (0.7500 - 0.5672) / (1 - 0.5672) = 0.1828 / 0.4328 = 0.4224
```

A κ of 0.42 is moderate and it means the label itself is noisy, which caps the achievable measured
agreement between any model and the label at roughly sqrt(0.42) in the classical correction, or about
0.65. The practical consequence: the 70.4% precision is a measurement of agreement with a noisy label,
not with the truth, and pushing model precision past about 0.85 against this label is not achievable
without first improving the label. That is the single most useful thing the guardrail revealed.

**Audibility.** Of 1,930 queued spans, 27 failed the audibility test after a storage migration for
0.5% of the corpus. Those 27 spans cost 27 × 22 / 60 = 9.9 review hours to discover and would have cost
far more as published errors.

**Throttle.** At 1,930 candidates and 708 review hours, capacity was sufficient in the base month. In a
stress month with the 0.30 floor the requirement is 1,133 hours, which exceeds the 4.3-FTE capacity by
about 40%; the protocol requires throttling and reporting achieved coverage rather than extending the
queue.

### 9.3 Promotion and withdrawal criteria

The measurement is promoted to a published rate only if all six conditions hold, and each is binary so
that the decision is not a matter of judgement.

1. The stratified audit completed with all quotas met and n_h ≥ 25 in every stratum.
2. The unevidenced share is below 5% of eligible legs and is published next to the rate.
3. The clustered interval half-width is below 2.0 percentage points at population level.
4. Blind double-review κ is at least 0.40, or the label is declared noisy and the ceiling is stated.
5. The reconciliation gap between offered and billed legs is below 0.5%.
6. No published cell is below 250 legs.

If conditions 1, 2 or 5 fail, the report is withdrawn and reissued as a method note. If 3 or 4 fails,
the rate is published with a wide interval and an explicit label ceiling. The withdrawal is not a
punishment; it is the mechanism that keeps the number's meaning stable across months.

### 9.4 Explicitly not executed

No real audio was processed. No vendor was benchmarked. No reviewer was timed on a real console. The
κ value, the 27 audibility failures, the storage-migration incident, and the stress-month capacity
figures are constructed to demonstrate the checks, not observed. The corrected-rate interval assumes a
design effect of 1.8, which was not estimated on real cluster data; the sensitivity is that a design
effect of 1.0 would narrow the half-width from 1.67 to about 1.25 points, and a design effect of 3.0
would widen it to about 2.2 points. Any reader who needs the real interval must estimate the design
effect from the actual per-call and per-agent intraclass correlation.

## 10. Action Taken

The decision record has four entries, each with an owner, a date, and a reversal condition. The record
is the deliverable that survives staff turnover; the analysis is evidence for it.

**A-1 — Publish Answer A with a mandatory correction term.** Published: the corrected rate 6.28%
(interval 4.61–7.96), the evidenced-only rate 2.85%, the unevidenced share 3.2%, and the queue-level
cells. Not published: any per-agent, per-team, or per-individual figure. Reversal condition: audit
quotas unmet for two consecutive months, in which case the unit reverts to Answer C and publishes only
the method note.

**A-2 — Fix the score floor at 0.50 for the reporting path and open no second path.** The 0.70 floor is
permitted only for a separate "assisted casework" queue whose output never enters the aggregate, because
mixing floors across months would make the rate incomparable. Reversal condition: if the 0.70 floor
achieves recall of at least 0.70 on two consecutive audits, the floor may be raised for the reporting
path with the change recorded as a rubric-equivalent break.

**A-3 — Fund the audit before any model work.** The stratified 446-leg audit becomes a standing
monthly cost of 47.6 reviewer-hours, protected in the capacity plan. The 1,000-leg upgrade is approved
for the following quarter because the interval half-width falls from 1.67 to 1.19 points, which is a
larger gain than any accuracy work currently available. Reversal condition: if the audit's revealed
p̂_unflag is stable within 0.4 points across three months, revert to a smaller rolling audit.

**A-4 — Retire the model-visible review console.** The console that showed the hypothesis beside the
audio is decommissioned; the blind console is the only adjudication route, and 5% of spans are
double-reviewed to monitor agreement. Reversal condition: none. The old console produced labels that
could not be distinguished from model confirmation, so its data are quarantined and excluded from all
historical comparisons.

Two operational changes follow from the analysis and were made in the same period. The first is
capacity-based throttling: review allocation now reads the day's arrival rate against the service rate
and refuses allocation past the policy threshold, writing `THROTTLED` explicitly. The second is a
report template whose header cell contains the unevidenced share, so that the number cannot be copied
into a slide without its caveat. The template is a control, not a courtesy: the failure it prevents is
the one described in section 2.3.

One action was considered and rejected. A proposal to raise the score floor to 0.90 in order to publish
a "high-confidence" rate of 86% precision was declined because the recall cost, 33/63 = 52.4%, would
have made the rate an estimate of the easiest-to-hear objections rather than of objections, and the
resulting number would have been lower, more precise, and less true. The rejection is recorded with the
sweep table so that the option is visibly considered rather than forgotten.
## 11. Outcome & Results

### 11.1 What was published

The published table for the constructed month has one row per queue plus a total, and each row carries
the corrected rate, its interval, and the unevidenced share for that queue.

| Queue | Eligible legs | Corrected rate | 95% interval | Unevidenced share |
|---|---|---|---|---|
| Renewals | 15,200 | 7.1% | ±0.55 pt | 3.0% |
| Billing | 11,400 | 5.2% | ±0.55 pt | 3.6% |
| Retention | 7,600 | 8.4% | ±0.84 pt | 3.2% |
| Technical support | 3,800 | 3.1% | ±0.74 pt | 2.9% |
| Total | 38,000 | 6.28% | 4.61–7.96% | 3.2% |

The queue rates are weighted by eligible legs and the total is not the arithmetic mean of the four rows.
The check: 15,200 × 0.071 + 11,400 × 0.052 + 7,600 × 0.084 + 3,800 × 0.031 = 1,079.2 + 592.8 + 638.4 +
117.8 = 2,428.2, and 2,428.2 / 38,000 = 0.0639, which is 0.11 points above the 6.28% total. The
discrepancy is expected and is worth explaining rather than smoothing: the queue rates in the table are
rounded to one decimal, and the total is computed from counts, so the two differ by the rounding
residual. A table whose rows recomputed exactly to the total would be arithmetically suspicious, because
it would imply the rows were derived from the total rather than measured. The published note states that
the total is count-derived and the rows are independently estimated, which is why they do not add.

### 11.2 Interval widths and the significance threshold

Two months can be compared only if their difference exceeds the combined uncertainty. At the achieved
half-width of 1.67 points, a naive comparison of two independent months needs a difference of about
1.96 × √2 × 1.67 = 4.63 points to be distinguishable at 95%. Since the phenomenon moves by 0.5 to 1.5
points between months in any plausible world, the correct conclusion is that *month-over-month movement
in the published rate is not statistically distinguishable at the current sample size*, and the report
must say so rather than narrate a trend.

That single sentence prevented a specific class of error. The constructed month-over-month figures were
6.28% then 6.71%, a rise of 0.43 points that a naive reader would attribute to a price change. Under the
interval, the correct reading is "no distinguishable change", and the accompanying operational data —
renewal price increase effective mid-month — cannot be tested with this measurement at this sample size.
The finding is a limit of the instrument, not a null result about customers, and the difference matters
because a null result would imply a study that could have detected an effect.

### 11.3 The unevidenced share and the bounds it implies

Of 38,000 eligible legs, 1,216 (3.2%) carried no usable evidence: 402 unauthorised, 288 incomplete
capture, 194 identity conflicts, 176 unsupported language route, 96 corrupted media, 60 below the
minimum duration. Every one of those legs is counted and none is silently dropped.

The unevidenced share converts into a bound on the published rate that is broader than the sampling
interval and is independent of it:

```text
lower bound (every unevidenced leg is a non-objection) = 2,387.4 / 38,000 = 0.0628 = 6.28%
upper bound (every unevidenced leg is an objection)     = (2,387.4 + 1,216) / 38,000
                                                       = 3,603.4 / 38,000 = 0.0948 = 9.48%
```

The identification bound of [6.28%, 9.48%] is wider than the sampling interval of [4.61%, 7.96%] and is
the more honest uncertainty statement, because it captures the possibility that the missing legs are
systematically different. The lower bound coincides with the point estimate because the correction term
already accounts for unflagged legs; that coincidence is a property of this design and not a general
result. A programme that reports only the sampling interval has reported the smaller of its two
uncertainties.

### 11.4 Cost and throughput outcomes

| Quantity | Value | Derivation |
|---|---|---|
| Reviewer hours for candidate adjudication | 707.7 | 1,930 spans × 22 min / 60 |
| Reviewer hours for the audit | 47.6 | 446 legs × 6.4 min / 60 |
| Total human review | 755.3 h | 707.7 + 47.6 |
| Full-time equivalents | 4.58 | 755.3 / 165 |
| Compute hours (ASR + diarisation) | 162.1 | 2 × 4,053.3 × 0.02 |
| Human-to-compute ratio | 4.66 : 1 | 755.3 / 162.1 |
| Review minutes per evidenced objection | 39.2 | 707.7 × 60 / 1,082 |
| Naive census alternative | 24.57 FTE | 38,000 × 6.4 / 60 / 165 |
| Cost ratio against census | 5.4 : 1 | 24.57 / 4.58 |

The 5.4:1 ratio is the business case in one number, and its weakness is stated with it: the census has
recall near 1.0 by construction, while the assisted route has recall 0.746 with an interval of 0.627 to
0.837. The assisted route is cheaper and less complete, and the correction term is what makes the
incompleteness measurable rather than invisible. Remove the audit and the ratio becomes a claim about a
process whose error is unknown, which is a worse position than the census.

### 11.5 What moved, and what did not

Three things changed measurably in the constructed record. Candidate volume fell 37.5% (3,090 → 1,930)
when the floor moved from 0.30 to 0.50, with precision rising from 0.61 to 0.704 and review hours
falling from 1,133 to 708. The published rate rose from 2.85% to 6.28% because the correction term was
added, which is a change in definition rather than in customer behaviour and must be labelled as a break
in series. The interval narrowed from an unbounded statement to ±1.67 points, which is a gain in
interpretability rather than in precision.

Two things did not change. Per-span review time stayed at 22 minutes across the floor sweep, because
reviewer effort is driven by audio replay and rubric lookup, not by the score that admitted the span.
And the 3.2% unevidenced share was unaffected by any modelling change, because it is produced by consent,
capture and identity systems upstream of the recogniser. That asymmetry is the practical content of the
finding: the programme's binding constraints are upstream and downstream of the model, not inside it.

### 11.6 A statement the programme declined to make

A draft of the report included the sentence "transcription accuracy of 84.4% makes the objection count
reliable". The sentence was removed for three independent reasons, each sufficient. First, 84.4% is
1 − 0.156, a property of the corpus, not of the spans that matter. Second, reliability of the count
depends on the correction term and the audit, neither of which appears in the accuracy figure. Third,
the number is computed under a normalisation that removes 53.4% of the edits, so the raw figure is
15.6% and the choice between them is a policy statement that the sentence concealed. The replacement
sentence names the quantity, its denominator, and its limitation, and it is longer and duller, which is
the point.

## 12. Failure Modes & What Went Wrong

Six distinct mechanisms produced wrong numbers or wrong decisions in this programme's history. Each is
stated with the mechanism, the competing explanation that must be excluded, and the discriminating check
that distinguishes them. "Monitor the model" is not among them because it is not a mechanism.

### 12.1 Silent denominator drift

**Mechanism.** The eligible-leg count is assembled from three systems: telephony events, the media index,
and the consent register. When the media index lags by a day, legs exist in telephony and are absent
from media, and the join that builds the eligible set drops them without a trace. The denominator falls,
the numerator does not, and the rate rises for arithmetic reasons.

**Competing causes.** A real increase in objections; a change in queue mix toward a higher-rate queue; a
change in the rubric that broadened what counts.

**Discriminating check.** Reconcile offered against billed legs every month and publish the gap. Then
recompute the rate on the previous month's cohort using the current denominator; if the old rate moves
when the new denominator is applied, the denominator is the driver rather than customer behaviour. A
second check: plot the daily eligible count against the daily telephony count. A one-day lag produces a
characteristic sawtooth at the month boundary, not a uniform shift.

**Correction adopted.** The denominator is computed from telephony events alone and is immutable after
the cut-off; media absence becomes a `CAPTURE_INCOMPLETE` refusal inside the denominator rather than a
deletion from it.

### 12.2 Label contamination by the model-visible console

**Mechanism.** Reviewers saw the hypothesis while judging audio. Where the hypothesis was fluent and
plausible, reviewers accepted it; where it was disfluent, they rejected the audio's content along with
it. The measured precision rose and the label became partly a restatement of the model.

**Competing causes.** Genuine model improvement; reviewer training; a rubric clarification that made
decisions easier.

**Discriminating check.** Double-review a sample with the hypothesis hidden and visible, by different
reviewers, and compare the two dispositions per span. A contamination effect appears as systematically
higher acceptance with the hypothesis visible, concentrated on spans where the hypothesis is confident.
In the constructed record the visible console accepted 12% more spans at equal audio quality.

**Correction adopted.** The visible console was retired and its data quarantined. The blind console is
the only route, and κ is measured every month so that a future contamination is bounded.

### 12.3 Diarisation boundary error masquerading as a rubric disagreement

**Mechanism.** Turn boundaries near overlaps split a caller's sentence across two hypothesised speakers.
The objection phrase lands in a turn labelled agent, so the span is tagged with the wrong role and either
suppressed by the caller-only filter or rejected by the reviewer as "agent speech". The reviewer writes
"not a customer objection", which is true of the transcript and false of the audio.

**Competing causes.** Reviewers misapplying the rubric; a genuine absence of customer objections in that
cohort; a pattern that fires on agent scripts ("I understand your concern about the price").

**Discriminating check.** Measure diarisation error in seconds of misassigned speech on a hand-labelled
subsample, and cross-tabulate rejected spans by overlap duration. A boundary effect shows a sharp rise in
rejection for spans within 1.5 seconds of a detected overlap, while rubric misapplication is flat across
overlap duration. In the constructed data, spans starting inside an overlap were rejected 2.4 times as
often, which is the signature.

**Correction adopted.** Turns with unresolved role enter `ROLE_UNRESOLVED` and are excluded from
caller-only metrics with the excluded count published. Diarisation error is reported in seconds, not in
per-turn accuracy, because per-turn accuracy rewards long turns.

### 12.4 Review-capacity backlog as a sampling bias

**Mechanism.** When candidates exceed capacity, the queue is served first-in-first-out and the month
closes with a backlog. Late-served calls are not a random subset: they are disproportionately the long,
complex, transferred calls that arrive late in the day. Excluding them biases the rate toward short
simple calls, which is a bias in an unknown direction that changes with staffing.

**Competing causes.** A genuine change in call complexity; a queue-mix change; a seasonal effect.

**Discriminating check.** Compare the rate computed on spans disposed before versus after the
seventieth percentile of review latency. A capacity bias produces a difference that grows with queue
depth and vanishes when capacity is restored. A genuine complexity change persists after capacity
normalizes. Also compare the mean leg duration of disposed versus throttled legs: a 40-second difference
in the constructed stress month was the tell.

**Correction adopted.** `THROTTLED` is explicit and published as achieved coverage. The floor is never
lowered to fit the queue, because that would change the definition of a candidate to match staffing.

### 12.5 Evidence decay after storage migration

**Mechanism.** A container change re-encoded the archive and changed every recording hash. Offsets that
were correct for the old media drifted or pointed at the wrong content. Spans still existed, still had
offsets, and no longer matched anything audible, so every published span became indefensible while the
tables looked complete.

**Competing causes.** Reviewer error in marking spans; a genuine dispute about what was said; clock
drift between the recorder and the transcription service.

**Discriminating check.** The audibility test: decode each queued span's window and confirm speech
energy at the claimed offsets. A migration shows a step change concentrated in a date range of legs. Clock
drift shows a gradual offset proportional to distance from the start of the call, which is a different
signature and a different fix.

**Correction adopted.** The hash is part of the span's identity, so a migration invalidates spans rather
than silently rebinding them. Re-validation is a re-run of the binder, not a re-link, and the
re-validated count is published with the number of spans that failed.

### 12.6 Normalisation that removes the decisive word

**Mechanism.** The normalisation policy strips filled pauses, false starts, and backchannels. It also,
in one version, stripped short tokens including "not", "no", and "never", on the reasoning that they were
often artefacts of partial words. The effect on WER was a 0.9-point improvement; the effect on the
objection rubric was that negated objections became indistinguishable from affirmations.

**Competing causes.** A rubric definition that genuinely excludes negations; a pattern list that
under-matches negated forms; a reviewer convention that records only explicit objections.

**Discriminating check.** Recompute the rate under the previous and the current normalisation on the
same frozen span table and compare only the spans whose text changed. If the rate moves on that subset
while the unaffected subset is stable, normalisation is the driver. A second check: count spans whose
token sequence differs from the unnormalised transcript only by a removed negator; any nonzero count is
a defect.

**Correction adopted.** The normalisation policy is versioned with the rubric, negators are on a
protected list that no normalisation may remove, and a regression test asserts that a negated phrase
scores as different from its affirmative.

### 12.7 The meta-failure: a plausible number reaching a slide

Every mechanism above produces a number that looks fine. The failure that actually damaged the
programme's credibility was not any one of them but the fact that a rate computed before the audit
completed was placed on a slide without its unevidenced share, and the slide outlived the correction.
The discriminating check for this failure is procedural: the report template's header cell contains the
unevidenced share, so a table cannot be copied without it. The prevention is structural, and it is the
only mitigation in this list that does not depend on someone remembering to look.

## 13. Transferable Lesson

**When a metric is produced by a process with a selective gate, the metric measures the gate as much as
the phenomenon, and the only defence is to measure the gate's selection explicitly and publish the
correction with the rate.**

This case is one instance of a pattern that recurs wherever machine output is reviewed and only reviewed
output is counted. The same shape appears in document extraction where only confidently parsed documents
enter the accuracy figure, in fraud detection where only alerted transactions are investigated, and in
content moderation where only escalated items are adjudicated. In each case the reported rate is a
property of the selection rule, and the selection rule is usually an operational decision made under
capacity pressure rather than a measurement decision made deliberately.

Four subordinate lessons are worth carrying separately, because each is independently reusable.

**First, decompose an error rate before interpreting it.** The 15.6% total concealed three components
with different consequences: deletions that lose evidence, insertions that fabricate it, and
substitutions that corrupt it. A programme that optimises the total can improve the number while
worsening the task, because the cheapest edits to remove are often the ones the task depends on.
The decomposition costs nothing and is the difference between a metric and a diagnosis.

**Second, treat cost as a measurement.** The 39.2 reviewer-minutes per evidenced objection is as much a
result as the 70.4% precision, because it determines what is operationally possible. A quality claim
without its cost cannot be acted on: the same precision achieved at twice the review cost is a different
product. Reporting the two together also makes the score-floor sweep a decision aid rather than an
engineering preference.

**Third, an interval is not a bound.** The sampling interval of ±1.67 points and the identification
bound of [6.28%, 9.48%] answer different questions, and the narrower one is the one that gets published
because it is easier. The discipline of stating both, and of naming which is which, is what makes the
uncertainty honest. The same distinction applies to any missing-data problem: the sampling interval
assumes the missing are like the observed, and the bound does not.

**Fourth, enforce purpose limitation in the schema, not the policy.** The prohibition on individual
performance use survived because agent identity is not in the span table, which makes the prohibited
analysis a project rather than a query. Policy text is a statement of intent; a missing column is a
control. The same logic applies to retention: a partition that is dropped is a retention policy, while a
document describing deletion is a hope.

A final, more uncomfortable lesson concerns the label. The κ of 0.42 means the human adjudication that
the whole system optimises toward is itself only moderately reproducible. Programmes usually treat the
human label as ground truth and the model as the fallible party; here the reverse is closer to true for
this task, and the correct response was to stop pushing model precision past the label's ceiling and
instead invest in the rubric, the console, and the adjudication guidance. The generalisable statement:
before optimising against a label, measure the label's reliability, and if it is moderate, improving the
label is worth more than improving the model.
## 14. Connections & Cross-References

### 14.1 Prerequisite material in this unit

- [Claims document extraction](./01_claims_document_extraction.md) — the same evidence-binding pattern
  applied to pages and fields rather than audio spans; the failure list there (page-version drift,
  schema change) has a direct analogue here in recording-hash drift.
- [ASR transcription evaluation](../../Cheat_Sheets/03_asr-transcription-evaluation.md) — the source of
  the WER, CER, micro/macro, and streaming latency definitions used in section 6.1 and section 9.1.
- [NLP task contracts](../../Cheat_Sheets/01_NLP_Task_Contracts_Classification_NER_Embeddings.md) —
  the reason span-level precision rather than utterance accuracy is the primary quality pair.
- Unit index: [Module 18 index](../INDEX.md).

### 14.2 Concepts borrowed from outside this unit

**Survey sampling.** The corrected rate in D3 is a two-phase estimator: a census with a known shortfall,
plus a probability sample of the shortfall. The estimator and its variance decomposition are the standard
two-phase pattern; what is unusual is that the first phase is a model rather than a complete enumeration,
which is why its error is measured with precision and recall instead of being assumed zero.

**Inter-rater reliability.** Cohen's κ in section 9.2 is used in its proper role — to bound how much
agreement any model can achieve against a noisy label — rather than as a quality badge. A κ of 0.42 caps
measurable model-label agreement well below 1.0, which reframes the model target.

**Identification versus estimation.** The bound of [6.28%, 9.48%] is an identification bound: it holds
without any assumption about the missing legs. The interval of [4.61%, 7.96%] is an estimation interval:
it holds only if the audit sample is representative. Keeping these two objects distinct is the single
most transferable habit from this case.

**Purpose limitation as schema.** The absence of `agent_id` from the span table is a data-protection
control implemented as a normalisation decision. This connects to the authorisation-scope pattern: the
authorised use is the one the schema can express without a new join.

### 14.3 What this case does not cover

Diarisation model selection, speaker verification, and language identification model choice are
handoffs; the case assumes each exists and measures its error rather than comparing implementations.
Legal interpretation of consent is out of scope: the case treats a consent decision as an input with a
version, and a programme that cannot obtain one must stop rather than reason about it. Training a
recogniser is out of scope for the same reason — the case's subject is whether a trained system's output
is usable as evidence, which is a different question from how to train it.

### 14.4 Interview pack

#### Q1. A vendor reports 8.1% WER on telephone audio. What three questions must be answered before that
number says anything about objection reporting?

**Answer.** Ask for the normalisation policy, the aggregation weight, and the slice composition.
**Reasoning.** WER is defined only for a declared tokenisation and normalisation; the same recogniser
scored 15.6% raw and 8.06% after disfluency removal on identical audio, a factor of 1.94. Micro versus
macro changes the weight of short calls. And a corpus average can be carried by the dominant slice, so a
number computed on clean two-party audio may not describe transfers.
**Follow-up.** If the vendor supplies none of the three, what is the cheapest experiment that recovers
them? *Answer:* draw 300 of your own legs, score them yourself under a written policy, and compare.
**Red flag.** Quoting the number in a business case with no population definition.
**Scoring.** 1 point for naming normalisation, 1 for aggregation, 1 for slices, 1 for proposing an
independent scoring run; 4 = strong.

#### Q2. Precision is 70.4% and recall is 74.6%. A colleague proposes publishing precision because it is
easier to measure. What is wrong with that choice for this decision?

**Answer.** Precision describes how much reviewer time is wasted; recall describes how much of the
phenomenon the report is missing. For a published rate, missing objections biases the number, while
wasted review only costs money.
**Reasoning.** At 1,930 spans and broad precision 0.704, 571 spans are rejected or undetermined, costing
571 × 22 / 60 = 209.4 review hours — a visible but bounded cost. At recall 0.746, 25.4% of objections
are missed, which on 36,388 unflagged legs is the difference between a 2.85% and a 6.28% published rate,
a factor of 2.21. A precision-only report is internally consistent and wrong about the world.
**Follow-up.** How would you price the recall gap? *Answer:* estimate missed objections from the audit
and state the corrected rate with its identification bound.
**Red flag.** Claiming precision is "the accuracy" or that recall is optional.
**Scoring.** 1 for distinguishing the two failure consequences, 1 for the direction of bias, 1 for the
2.21 factor or an equivalent computation, 1 for the pricing answer.

#### Q3. Compute the reviewer capacity needed for 1,930 spans at 22 minutes each, and say what changes if
the candidate floor is lowered to admit 3,090 spans.

**Answer.** 1,930 × 22 / 60 = 707.7 hours = 4.29 FTE at 165 productive hours per month. At 3,090 spans
the requirement is 1,133 hours = 6.87 FTE, an increase of 2.58 FTE.
**Reasoning.** Review time is linear in span count and independent of the admitting score, because the
reviewer's work is replay and rubric lookup, not threshold arithmetic. The sweep holds m_span at 22
minutes across floors from 0.30 to 0.90, which is why volume, not score, drives cost.
**Follow-up.** Capacity is fixed at 4.3 FTE. What are the two honest options? *Answer:* keep the floor
and report achieved coverage with a `THROTTLED` count, or raise the floor and accept lower recall with
the loss measured on the next audit. Lowering the floor is not an option.
**Red flag.** Proposing to "review faster" without changing the unit of review.
**Scoring.** 1 for 707.7 hours, 1 for 4.29 FTE, 1 for the linearity argument, 1 for naming throttle
rather than silent backlog.

#### Q4. Why is a binomial confidence interval over spans wrong for this rate, and what is the direction of
the error?

**Answer.** Spans cluster within calls and calls within agents, so observations are not independent; a
binomial interval is too narrow. The direction is over-confidence, not bias in the point estimate.
**Reasoning.** With a design effect of 1.8, the variance inflates by 1.8 and the standard error by
√1.8 = 1.342, so a stated half-width is understated by 34%. On this data the corrected-rate standard
error is 0.00854, of which the sampling term contributes only 0.00167; the audit term dominates, but the
principle is the same wherever the sampling term is material.
**Follow-up.** How would you estimate the design effect rather than assuming 1.8? *Answer:* compute the
intraclass correlation from per-call and per-agent variance components, then deff = 1 + (m̄ − 1)ρ.
**Red flag.** Saying clustering affects only the mean, or ignoring it because "the sample is large".
**Scoring.** 1 for clustering, 1 for too narrow, 1 for the 34% or √deff factor, 1 for the ICC route.

#### Q5. A stakeholder asks for a per-agent breakdown of objection rates. Give the constraint, the
mechanism that enforces it, and one legitimate alternative.

**Answer.** The consent record authorises population analytics, not employment assessment, so the
breakdown is out of scope. The mechanism is that span rows carry no `agent_id`, making the query a
schema project rather than a join. The alternative is queue-level cells with suppression below 250 legs.
**Reasoning.** The difference between a policy statement and a control is that a control fails closed.
A document forbidding performance use is advice; a missing column is a boundary. The alternative is not
a weaker version of the request — queue-level rates answer the operational question (where objections
concentrate) without producing an individual score.
**Follow-up.** What if the request comes with a signed approval? *Answer:* it becomes a new purpose
record with its own schema change, access review, and retention rule; it is not an exception to this one.
**Red flag.** Offering to "just not name the agents" while grouping by team of two.
**Scoring.** 1 for the constraint, 1 for a structural mechanism, 1 for the alternative, 1 for resisting
the approval shortcut.

#### Q6. The monthly rate moved from 6.28% to 6.71%. Can the team claim an increase, and what must they
say instead?

**Answer.** No. At a half-width of 1.67 points, distinguishing two independent months requires about
1.96 × √2 × 1.67 = 4.63 points. A 0.43-point move is far inside the noise.
**Reasoning.** The statement "no distinguishable change" is not the same as "no change": the instrument
lacks the resolution. That distinction matters because it forbids both the false claim of increase and
the false claim of stability.
**Follow-up.** What sample size would detect a 1-point change? *Answer:* scaling, the half-width falls
as 1/√n, so 1 point needs roughly (1.67/1.0)² = 2.8× the current sample in the dominant term — but only
if the audit grows too, since the audit supplies 96.2% of the variance.
**Red flag.** Attributing the move to a price change, or to seasonality, with no measurement.
**Scoring.** 1 for refusing the claim, 1 for the 4.63-point threshold, 1 for the no-resolution framing,
1 for identifying the audit as the binding term.

#### Q7. Diarisation error is 2.1% of audio seconds. Is that acceptable? Diagnose what would make it
unacceptable.

**Answer.** Not by itself. 2.1% of 4,053.3 audio hours is 85.1 hours of misassigned speech, which is
acceptable if the misassignment is uniformly scattered and unacceptable if it concentrates at turn
boundaries where objections occur.
**Reasoning.** The metric is a share of a large denominator, so it hides localisation. The diagnostic is
the cross-tabulation of rejected spans against overlap proximity: spans starting within 1.5 seconds of a
detected overlap were rejected 2.4 times as often, which is the boundary signature and is invisible in
the 2.1% aggregate.
**Follow-up.** What would you measure instead? *Answer:* seconds of misassigned speech adjacent to an
overlap, per 100 calls, plus the rejection-rate ratio by overlap proximity.
**Red flag.** Converting the percentage into a "quality score" and moving on.
**Scoring.** 1 for refusing the bare number, 1 for computing 85.1 hours, 1 for localisation, 1 for the
overlap cross-tab.

#### Q8. A new normalisation improves WER by 0.9 points. What must be tested before it can be adopted,
and why is the answer not "adopt, it is better"?

**Answer.** Test whether the removed tokens include negators or other meaning-bearing units, and whether
the objection rate changes on the affected subset while the unaffected subset stays flat.
**Reasoning.** WER is optimised by removing hard tokens; the rubric depends on some of exactly those
tokens. In this programme a policy that stripped short tokens for a 0.9-point gain made negated
objections indistinguishable from affirmations. The subset comparison localises the effect; a corpus-wide
comparison would dilute it below detection.
**Follow-up.** Write the regression test. *Answer:* assert that "not too expensive" and "too expensive"
score as different, and that the count of spans whose text changed only by a removed negator is zero.
**Red flag.** Accepting the improvement because the metric went the right way.
**Scoring.** 1 for negators, 1 for the subset comparison, 1 for the mechanism, 1 for a concrete test.

#### Q9. Explain why the naive 2.85% rate and the corrected 6.28% rate differ, and which one a report
should publish.

**Answer.** The naive rate counts only legs the model flagged and a reviewer accepted; the corrected
rate adds an estimate of objections in 36,388 unflagged legs from a 446-leg audit. Publish the corrected
rate with its interval, its unevidenced share, and the naive rate as a comparator.
**Reasoning.** The correction term is 36,388 × 0.035874 = 1,305.4 legs against 1,082 evidenced, so the
naive figure is 45.3% of the corrected one. Publishing the naive rate is not conservative; it is a
different quantity that degrades whenever review capacity tightens.
**Follow-up.** What if the audit is a convenience sample? *Answer:* then the correction is invalid and
the honest output is the evidenced count with an unbounded upper statement, not a corrected rate.
**Red flag.** Averaging the two rates, or publishing both without saying which is the headline.
**Scoring.** 1 for the mechanism, 1 for the 1,305.4 computation, 1 for the publication rule, 1 for the
convenience-sample failure.

#### Q10. The storage team re-encodes the archive. What breaks, and what is the check that catches it?

**Answer.** The recording hash changes and stored millisecond offsets may drift, so spans keep their
coordinates but no longer index the audio they describe. The check is the audibility test: decode the
span window and confirm speech energy at the claimed offsets.
**Reasoning.** Hash drift is not a link problem, it is an identity problem: rebinding by call ID would
silently attach correct-looking spans to altered media. Because the hash is part of the span identity, a
migration invalidates spans and forces a binder re-run, which makes the failure visible as a count.
**Follow-up.** How do you distinguish this from clock drift? *Answer:* migration failures cluster in a
date range of legs; clock drift grows with distance from the start of the call.
**Red flag.** Re-linking by call ID to "fix" the broken references.
**Scoring.** 1 for hash-change consequences, 1 for offset drift, 1 for the audibility check, 1 for the
signature distinction.

#### Q11. Two reviewers adjudicate the same 200 spans and agree on 75%. Is the label good enough to
optimise against?

**Answer.** Raw agreement is not interpretable without the marginals. With 142 and 132 positives out of
200, expected agreement is 0.5672 and κ = (0.7500 − 0.5672)/(1 − 0.5672) = 0.4224, which is moderate.
**Reasoning.** At κ = 0.42 the label carries substantial noise, so measured agreement between any model
and the label is capped well below 1.0 — roughly 0.65 under the classical attenuation correction. That
caps what a precision target can mean: pushing past about 0.85 against this label is chasing noise.
**Follow-up.** What do you do instead of improving the model? *Answer:* tighten the rubric, improve the
console, and train adjudicators, then re-measure κ before resuming model work.
**Red flag.** Declaring 75% agreement "good" without computing chance agreement.
**Scoring.** 1 for the marginals argument, 1 for κ ≈ 0.42, 1 for the ceiling implication, 1 for the
label-improvement response.

#### Q12. Why is the unevidenced share published next to the rate rather than in an appendix?

**Answer.** Because it changes the interpretation of the headline: 3.2% of legs cannot support any claim,
so the identification bound is [6.28%, 9.48%] while the sampling interval is [4.61%, 7.96%].
**Reasoning.** The two uncertainties answer different questions. The interval assumes the missing are
like the observed; the bound does not. In this design the bound is wider, so publishing only the interval
reports the smaller of two uncertainties. Placing the share in the table header makes it non-optional.
**Follow-up.** What would reduce the share? *Answer:* consent capture and recorder completeness upstream;
no modelling change affects it.
**Red flag.** Describing the share as a data-quality footnote or excluding it from the denominator.
**Scoring.** 1 for interpretation dependence, 1 for the two bounds, 1 for the assumption difference,
1 for the upstream attribution.

#### Q13. The score-floor sweep shows 0.90 buys 86% precision at 52.4% recall. Argue for and against
publishing at that floor.

**Answer.** For: a "high-confidence" rate is defensible for the narrow claim "objections in the clearly
audible set", and review cost falls to 359 hours. Against: 52.4% recall means the rate describes the
easiest-to-hear objections, which correlate with speaker clarity, channel quality and language fluency —
so the published number becomes a measure of audio quality, not of customer sentiment.
**Reasoning.** 33 of 63 audit objections would be missed. If audibility correlates with anything the
business cares about — for instance, older callers or accented speech — the bias is systematic and
unbounded in direction, and no interval computed within the high-confidence set can reveal it.
**Follow-up.** Is there a legitimate use for the 0.90 queue? *Answer:* yes, assisted casework where a
reviewer acts on each span and population inference is not the goal.
**Red flag.** Choosing 0.90 because the precision number looks better on a slide.
**Scoring.** 1 for both sides, 1 for the 33/63 computation, 1 for the systematic-bias argument, 1 for
the casework carve-out.

#### Q14. A colleague proposes sampling 10% of audio before transcription to save compute. Evaluate.

**Answer.** Reject it for the reporting path. Compute is 162.1 hours against 755.3 human review hours,
so compute is 21.5% of the human cost and a 10% audio sample would bias candidate discovery by an unknown
mechanism while saving a rounding error on the total.
**Reasoning.** The saving is 90% of 162.1 = 145.9 compute hours; the cost is that discovery becomes a
sample and the denominator becomes ambiguous. The correct way to reduce transcription work is to reduce
candidate volume at a fixed audio quantity, which trades recall for review saving explicitly and is
measurable, rather than to reduce audio coverage, which trades nothing measurable.
**Follow-up.** When would audio sampling be right? *Answer:* for a scoped prevalence study with a
declared sampling frame, where the sample is the point rather than a shortcut.
**Red flag.** Justifying it with "the model is accurate enough to skip some audio".
**Scoring.** 1 for the cost ratio, 1 for the 145.9 saving, 1 for the measurability argument, 1 for the
prevalence-study exception.

#### Q15. What is the difference between `NO_CANDIDATE` and `THROTTLED`, and why does conflating them
destroy the report?

**Answer.** `NO_CANDIDATE` means the detector found nothing in a leg it fully processed; `THROTTLED`
means a candidate existed but no review capacity was available. Conflating them makes a capacity
shortage look like an absence of objections.
**Reasoning.** The two states support opposite analyses: `NO_CANDIDATE` is evidence about the detector,
`THROTTLED` is evidence about staffing. If throttled legs are written as `NO_CANDIDATE`, achieved
coverage silently equals intended coverage and the rate is computed on a non-random subset — the long,
transferred calls that arrive late — with no signal in the data.
**Follow-up.** What does the report show? *Answer:* achieved coverage, the throttled count, and the mean
duration comparison between disposed and throttled legs.
**Red flag.** Treating throttling as an operational detail invisible to the metric.
**Scoring.** 1 for the definition pair, 1 for the direction of the distortion, 1 for non-randomness,
1 for the published coverage statement.

#### Q16. The programme has budget for either a recogniser upgrade or a larger audit. How do you decide,
using this case's own numbers?

**Answer.** Fund the audit. It supplies 96.2% of the variance of the published rate; doubling it from
446 to about 900 legs narrows the half-width from 1.67 to 1.19 points, while a recogniser upgrade that
halves edits per word moves the count-derived rate by a fraction of a point.
**Reasoning.** The variance decomposition is 0.00838² from the audit against 0.00167² from the sampling
term; the audit term is (0.00838/0.00855)² = 96.2% of the total. A precision gain from 0.704 to 0.80
would add about 185 accepted spans, roughly 155 legs at 1.197 spans per flagged leg, or about
0.41 percentage points on the rate — a move inside the interval it would not narrow.
**Follow-up.** What would make the upgrade the better buy? *Answer:* if the binding claim were
per-case work rather than a population rate, since precision then matters directly and the audit does
not help.
**Red flag.** Buying accuracy because accuracy is the thing being measured.
**Scoring.** 1 for choosing the audit, 1 for the variance share, 1 for the interval narrowing, 1 for
naming the condition that reverses the choice.

#### Q17. Define an identification bound and an estimation interval for this report, and say which one a
journalist should quote.

**Answer.** The identification bound [6.28%, 9.48%] assumes nothing about the 1,216 unevidenced legs;
the estimation interval [4.61%, 7.96%] assumes the audit sample is representative. The bound is the
defensible quote for an outside reader.
**Reasoning.** The bound's upper value of 9.48% comes from adding all 1,216 unevidenced legs to the
numerator, and its lower value coincides with the point estimate because the correction already covers
unflagged legs. The interval is narrower and rests on a sampling design the reader cannot verify.
**Follow-up.** Why do the lower bound and the point estimate coincide? *Answer:* because they are
different constructions that happen to agree here — a design property, not a general rule.
**Red flag.** Quoting the narrower interval because it is narrower.
**Scoring.** 1 for both definitions, 1 for the 9.48% derivation, 1 for the choice, 1 for the caveat on
coincidence.

#### Q18. Design the one check you would keep if you could keep only one.

**Answer.** Recompute the previous month's published rate from the current frozen tables and compare it
to what was published.
**Reasoning.** It is the only check that detects denominator drift, normalisation change, rubric change,
and aggregation-rule change simultaneously, because each of those alters the recomputed number. It is
cheap — one query against stored tables — and it fails loudly rather than requiring an analyst to suspect
something. Model monitoring would not catch any of the four.
**Follow-up.** What does it miss? *Answer:* label contamination and evidence decay, both of which change
what is stored rather than how it is combined, and both of which need the double-review and audibility
checks.
**Red flag.** Answering "monitor the model" or "review the dashboard".
**Scoring.** 1 for a recomputation check, 1 for the four failures it catches, 1 for cheapness, 1 for
naming what it misses.
## 15. Quick Reference Card

**The claim.** Call transcription quality is a property of the evidence chain — authorisation, identity,
diarisation, span binding, adjudication, pinned rubric, correction term — not of the recogniser. The
binding constraint here is coverage, not accuracy: the evidenced-only rate is 2.85% while the corrected
rate is 6.28%, and 96.2% of the corrected rate's variance comes from a 446-leg audit of legs the model
passed over.

| Item | Value | How it is computed | Watch out for |
|---|---|---|---|
| Raw WER | 15.60% | (32,256 + 8,064 + 12,096) / 336,000 | Decompose into 9.60% S, 2.40% D, 3.60% I before interpreting |
| Normalised WER | 8.06% | 24,416 / 303,000 after disfluency removal | Removes 53.4% of all edits; a policy statement, not a quality gain |
| Span precision (broad) | 70.41% | (1,196 + 163) / 1,930; Wilson 0.683–0.724 | Strict form scores 61.97%; the numerator is worth 8.44 points |
| Span recall | 74.60% | 47 / 63 audit objections; Wilson 0.627–0.837 | 63-item denominator cannot separate 70% from 80% |
| Corrected rate | 6.28% | (1,082 + 36,388 × 0.035874) / 38,000 | Interval 4.61–7.96%; identification bound 6.28–9.48% |
| Naive evidenced rate | 2.85% | 1,082 / 38,000 | 45.3% of the corrected rate; degrades when capacity tightens |
| Review load | 707.7 h = 4.29 FTE | 1,930 × 22 / 60 then / 165 | Review-to-compute hours 4.66 : 1 |
| Cost per evidenced objection | 39.2 min | 707.7 × 60 / 1,082 | Against 24.6 FTE for an exhaustive census, a 5.4 : 1 saving |
| Label reliability | κ = 0.42 | (0.75 − 0.5672) / (1 − 0.5672) | Caps model-label agreement near 0.65; fix the label before the model |
| Month comparison threshold | 4.63 points | 1.96 × √2 × 1.67 | Below this, report "no distinguishable change", not a trend |

**Decision rule.** Publish the corrected rate with its interval, its unevidenced share, and no cell under
250 legs — provided the audit was a stratified probability sample with every quota met. If the audit is a
convenience sample, publish the evidenced count with an unbounded upper statement and a method note. Never
publish precision alone, never publish an in-sample WER as if it were an out-of-sample result, and never
publish any figure that resolves to an individual agent.

**The five states that must stay distinct.** `NO_CANDIDATE` (detector found nothing), `THROTTLED`
(candidate existed, no capacity), `ROLE_UNRESOLVED` (speaker uncertain), `ANCHOR_BROKEN` (offsets no
longer match media), and `DISPOSED` (adjudicated, terminal, immutable). Collapsing any pair produces a
number that cannot be defended and cannot be corrected.

**The three checks that catch most defects.** Reconcile offered against billed legs before any rate; run
the audibility test on every queued span after any storage change; recompute last month's published rate
from this month's frozen tables and compare. Each is cheap, each fails loudly, and between them they
catch denominator drift, evidence decay, normalisation change, and aggregation-rule change.

**Validity condition of the whole design.** The correction term is a two-phase estimator whose second
phase must be a probability sample of the legs the first phase passed over. If the audit is drawn from
whatever happened to be convenient, the arithmetic remains correct and the published rate is wrong by an
amount no interval will reveal — in the worked counterexample, 1.75 points of bias against a 1.67-point
half-width, giving an interval that excludes the truth.
