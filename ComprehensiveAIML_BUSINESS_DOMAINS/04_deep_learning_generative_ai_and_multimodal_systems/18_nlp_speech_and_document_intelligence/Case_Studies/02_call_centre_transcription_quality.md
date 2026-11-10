---
case_id: "CASE-18-02"
case_title: "Call-centre transcription quality for evidence-based objection reporting"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: "null"
status: "proposed"
content_status: "draft"
case_type: "hybrid"
domain: "Speech analytics / call-centre quality"
primary_pattern: "evaluation-mismatch"
secondary_patterns: "["human-in-the-loop", "authorization-scope"]"
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Customer-operations analytics owner"
source_index: "INDEX.md"
review_trigger: "Audio channel, rubric, language route, consent policy, or review-capacity change"
---

# Call-Centre Transcription Quality: Evidence Before Aggregation

## 1. The One-Line Framing

**Illustrative design exercise; no company recording, deployment, or outcome is asserted.**
Given an authorised call recording and its intake metadata, create evidence-linked transcript turns and route unreliable calls so a customer-operations analytics owner can report aggregate objections without converting uncertain audio into individual judgments, compared with manual listening and transcription.
Output is an evidence span, a bounded aggregate candidate, or an explicit route/abstention; it is not a performance score, customer decision, or employment action.

## 2. Context & Background

A call workflow may include capture, consent/access validation, transcription, diarization, evidence review, aggregation, and reporting.
The human baseline is authorised listening and transcription under an approved rubric.
The main trap is treating a plausible transcript, sentiment score, or keyword count as proof that a customer made a reportable objection.
Calls vary by channel, language, speaker, device, background noise, transfer, and capture completeness.

## 3. The Business Question

**Illustrative stakeholder question:** “Can we produce a reviewed aggregate objection report with traceable speaker and evidence spans while preserving manual review for uncertain calls?”
Owner: customer-operations analytics owner; reviewers approve transcription/evidence and reporting owners approve aggregate publication.
Actions: retain manual baseline, assist review on approved routes, route to specialist review, defer report, or reject unauthorised processing.
Hard constraints: consent/purpose gate, no individual performance use, evidence span for every reported count, speaker/role uncertainty explicit, and review capacity respected.

## 4. Assumptions & Constraints

| ID | Condition | Check | Impact | Fallback |
|---|---|---|---|---|
| A-01 | Audio use is authorised for this purpose. | Consent/access record. | Processing is forbidden. | Deny. |
| A-02 | Call/leg identity and capture bounds are stable. | Telephony metadata. | Wrong denominator/evidence. | Route. |
| A-03 | Gold review is independent of model output. | Blinded sampling. | Inflated quality. | Re-review. |
| C-01 | Review queue stays within approved service policy. | Arrival, service, age. | Delayed correction. | Throttle. |
| C-02 | Aggregate rubric/version is pinned. | Report manifest. | Incomparable cohorts. | Separate/reprocess. |

## 5. Data, Tools & Method

### 5a. Data Specification

- Unit: a call leg, speaker turn, and candidate objection span; aggregate only after approved turn evidence.
- Keys: call/leg ID, recording version/hash, channel, capture bounds, consent decision, timestamp contract, speaker-role hypothesis, rubric version.
- Inputs: authorised audio, telephony events, VAD/ASR/diarization outputs, language evidence, reviewer dispositions.
- Label: independent human transcript, speaker role, span-level objection rubric, and abstention reason.
- Split by caller/call, time, route/language, and when required device or queue; never randomize turns from the same call across splits.

### 5b. Tools & Stack

- Proposed canonical audio reader preserves sample rate, channel mapping, gap markers, and source hash.
- Proposed ASR emits token timing and confidence; proposed diarizer emits speaker segments and uncertainty.
- Proposed evidence service binds transcript tokens to recording offsets and rubric version.
- Proposed review console plays authorised audio around a span and records correction/reason.
- Proposed aggregate service accepts only reviewed or policy-approved evidence states.

### 5c. Method Summary

- 1. Check consent, purpose, call identity, and complete capture before any transcription.
- 2. Segment audio while retaining gaps, overlap, and timing uncertainty.
- 3. Produce turn hypotheses with language and speaker-role evidence.
- 4. Bind candidate objections to verbatim tokens and replayable time spans.
- 5. Apply hard gates for authorization, source completeness, role/evidence conflict, and rubric version.
- 6. Route uncertainty to reviewers; aggregate only under a pinned denominator and reporting cutoff.

### 5d. Data provenance, quality, exclusions and leakage controls

- Use source audio hash, call/leg identity, channel, capture start/end, consent decision, ASR/diarization versions, rubric version, and reviewer record as lineage.
- Exclude unauthorised audio, incomplete capture for completeness claims, unbound transfers, unsupported languages, and corrupted recording from automatic reporting.
- Do not use later supervisor notes, complaint outcomes, or reviewer corrections as extraction-time inputs.
- Sample accepted as well as routed calls so evaluation is not conditional on model selection.

### 5e. Reproducibility and method-selection ledger

| ID | Artifact | Purpose | Limitation |
|---|---|---|---|
| SRC-01 | illustrative authorised call manifest | input identity contract | no real calls inspected |
| CFG-01 | ASR/diarization/rubric bundle | reproducible candidate rules | proposed only |
| RUN-01 | manual workload calculation | capacity reasoning | not a service run |
| GOLD-01 | future blinded reviewer set | independent evaluation | unavailable |

## 6. Analysis Walkthrough

**ILLUSTRATIVE workload:** 900 eligible call legs/day; route fraction `r=0.35`; average reviewer time `m=3.0 staffed minutes/routed leg`; two reviewers each staff 8 hours/day.
Routed legs = `900 × 0.35 = 315`; review work = `315 × 3.0 = 945 minutes = 15.75 staffed hours/day`.
Two reviewers provide 16 staffed hours/day; nominal headroom is `16 - 15.75 = 0.25 hour = 15 minutes/day`.
At `r=0.45`, work is `900 × 0.45 × 3 = 1,215 minutes = 20.25 hours`, a 4.25-hour shortfall.
This is a constant-rate capacity calculation, not a tail-latency or quality guarantee.
```text
FORMULA [CC-01]: W = N * r * m
WHERE: W staffed minutes/day; N eligible call legs/day; r routed fraction; m staffed minutes/routed leg.
CONSTRAINT: constant averages; excludes rework, breaks, variation, and escalations.
NUMERICAL EXAMPLE: W=900*0.35*3=945 minutes=15.75 hours.
INTERPRETATION: a 15-minute nominal headroom is fragile and must not justify a safety threshold.
```

## 7. The Finding

The illustrative design may support proposed reviewer assistance only where consent, source identity, speaker/role evidence, and span-level rubric gates pass.
At the illustrative 35% route rate, two reviewers have only 15 minutes nominal daily headroom; any higher route rate or review time reverses feasibility.
No transcription quality, objection quality, customer impact, or deployment result is measured.

## 8. System/Solution Design

`Authorisation -> canonical audio/hash -> segmentation -> ASR/diarization -> span/rubric gate -> route policy -> reviewer -> aggregate manifest`.
A call is never aggregated from an unbound transcript; approved aggregate count stores call/leg, span, rubric, reviewer/route state, and version.
Recovery invalidates candidate spans if audio source, channel mapping, rubric, or model bundle changes.
Idempotency key is audio hash + call leg + configuration/rubric bundle; retries must not double count a call.

## 9. Evaluation & Validation

Report word error, speaker-role error, span precision/recall, objection precision/recall, evidence coverage, route rate, reviewer minutes, and aggregate denominator separately.
For every metric state numerator, denominator, reviewed population, time window, language/channel/device slice, and rubric version.
Use blinded independent gold and audit samples from accepted and routed populations.
Compare assisted review to authorised manual baseline, not only to a transcript benchmark.
A changed audio route, microphone, language, or rubric requires slice evaluation before promotion.

## 10. Action Taken

No production action was taken.
Proposed next action: define consent/access contract, obtain independent gold labels, measure manual baseline and staffed review time, then conduct a bounded non-production evaluation.
Stop if evidence spans cannot be replayed, consent is unclear, or queue policy cannot sustain routing.

## 11. Outcome & Results

**ILLUSTRATIVE / arithmetic verified / operational outcome not measured.**
Derived workload at 35% routing is 945 minutes or 15.75 staffed hours/day, leaving 15 nominal minutes for two 8-hour reviewers.
At 45% routing, workload is 20.25 hours/day, exceeding capacity by 4.25 hours.
No claim is made about WER, role accuracy, objection quality, customer behavior, or operational benefit.

## 12. Failure Modes & What Went Wrong

### FM-01 — channel clipping
- Symptom: evidence-based objection reporting is unreliable under channel clipping.
- Mechanism: audio samples saturate and consonants disappear.
- Diagnostic: clipping ratio and waveform peak histogram.
- Recovery: route to audio-quality review.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-02 — cross-talk
- Symptom: evidence-based objection reporting is unreliable under cross-talk.
- Mechanism: two speakers overlap.
- Diagnostic: overlap detector and diarization posterior.
- Recovery: preserve both channels and route.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-03 — speaker swap
- Symptom: evidence-based objection reporting is unreliable under speaker swap.
- Mechanism: diarization assigns turns to the wrong role.
- Diagnostic: role-consistency against known greeting/agent channel.
- Recovery: relabel with reviewer evidence.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-04 — code switching
- Symptom: evidence-based objection reporting is unreliable under code switching.
- Mechanism: language changes inside a turn.
- Diagnostic: word-level language posterior.
- Recovery: use supported bilingual review route.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-05 — domain term
- Symptom: evidence-based objection reporting is unreliable under domain term.
- Mechanism: product or claim term is out of vocabulary.
- Diagnostic: lexicon miss plus low token posterior.
- Recovery: flag term span for reviewer.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-06 — acronym ambiguity
- Symptom: evidence-based objection reporting is unreliable under acronym ambiguity.
- Mechanism: letters map to multiple business terms.
- Diagnostic: context alternatives and confidence margin.
- Recovery: retain verbatim span and ask reviewer.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-07 — VAD truncation
- Symptom: evidence-based objection reporting is unreliable under VAD truncation.
- Mechanism: voice activity detector cuts a short objection.
- Diagnostic: boundary padding audit.
- Recovery: rerun with protected boundary.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-08 — hold music
- Symptom: evidence-based objection reporting is unreliable under hold music.
- Mechanism: non-speech segment is transcribed as speech.
- Diagnostic: music/speech classifier.
- Recovery: remove segment and preserve timing.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-09 — background television
- Symptom: evidence-based objection reporting is unreliable under background television.
- Mechanism: external speech contaminates channel.
- Diagnostic: source/separation diagnostic.
- Recovery: route or mark evidence insufficient.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-10 — packet loss
- Symptom: evidence-based objection reporting is unreliable under packet loss.
- Mechanism: network gaps remove words.
- Diagnostic: packet-loss marker and discontinuity.
- Recovery: mark omission and abstain.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-11 — timestamp drift
- Symptom: evidence-based objection reporting is unreliable under timestamp drift.
- Mechanism: ASR times no longer align with call recording.
- Diagnostic: anchor-word offset check.
- Recovery: re-align or reject timestamps.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-12 — redaction error
- Symptom: evidence-based objection reporting is unreliable under redaction error.
- Mechanism: sensitive content is exposed or destroys needed evidence.
- Diagnostic: redaction span audit.
- Recovery: block report and repair policy.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-13 — summary hallucination
- Symptom: evidence-based objection reporting is unreliable under summary hallucination.
- Mechanism: aggregation asserts an objection absent from a span.
- Diagnostic: span-entailment check.
- Recovery: remove assertion and route.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-14 — negation loss
- Symptom: evidence-based objection reporting is unreliable under negation loss.
- Mechanism: “not satisfied” becomes “satisfied”.
- Diagnostic: negation cue comparison.
- Recovery: require human confirmation.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-15 — numeric confusion
- Symptom: evidence-based objection reporting is unreliable under numeric confusion.
- Mechanism: amount or date is misheard.
- Diagnostic: number grammar and audio replay.
- Recovery: retain raw audio evidence.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-16 — accent shift
- Symptom: evidence-based objection reporting is unreliable under accent shift.
- Mechanism: acoustic population differs from evaluation.
- Diagnostic: slice WER and calibration shift.
- Recovery: route new slice for gold labeling.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-17 — microphone change
- Symptom: evidence-based objection reporting is unreliable under microphone change.
- Mechanism: new device changes frequency response.
- Diagnostic: device metadata and spectral shift.
- Recovery: hold automatic route.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-18 — agent script change
- Symptom: evidence-based objection reporting is unreliable under agent script change.
- Mechanism: role language changes after training.
- Diagnostic: script/version drift.
- Recovery: refresh evaluation and route.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-19 — long silence
- Symptom: evidence-based objection reporting is unreliable under long silence.
- Mechanism: silence changes segmentation and conversation state.
- Diagnostic: silence-duration trace.
- Recovery: preserve event and avoid invented turn.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-20 — transfer between agents
- Symptom: evidence-based objection reporting is unreliable under transfer between agents.
- Mechanism: call leg changes speaker identity.
- Diagnostic: telephony transfer event.
- Recovery: split leg and re-establish roles.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-21 — duplicate recording
- Symptom: evidence-based objection reporting is unreliable under duplicate recording.
- Mechanism: same call is ingested twice.
- Diagnostic: call/content hash.
- Recovery: deduplicate aggregate count.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-22 — partial recording
- Symptom: evidence-based objection reporting is unreliable under partial recording.
- Mechanism: call begins after the relevant question.
- Diagnostic: capture-start marker.
- Recovery: abstain from completeness claim.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-23 — low volume
- Symptom: evidence-based objection reporting is unreliable under low volume.
- Mechanism: words are below intelligibility.
- Diagnostic: signal-to-noise estimate.
- Recovery: request/review recording.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-24 — reverberation
- Symptom: evidence-based objection reporting is unreliable under reverberation.
- Mechanism: echo smears phonemes.
- Diagnostic: room impulse indicator.
- Recovery: route quality slice.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-25 — emotion overreach
- Symptom: evidence-based objection reporting is unreliable under emotion overreach.
- Mechanism: prosody score is treated as intent.
- Diagnostic: evidence-span absence.
- Recovery: report only supported words.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-26 — sentiment aggregation
- Symptom: evidence-based objection reporting is unreliable under sentiment aggregation.
- Mechanism: one utterance is treated as whole-call outcome.
- Diagnostic: denominator and turn coverage.
- Recovery: require explicit aggregation rule.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-27 — late label
- Symptom: evidence-based objection reporting is unreliable under late label.
- Mechanism: review outcome arrives after reporting cutoff.
- Diagnostic: label timestamp.
- Recovery: hold metric update.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-28 — selective review
- Symptom: evidence-based objection reporting is unreliable under selective review.
- Mechanism: only difficult calls receive gold labels.
- Diagnostic: route propensity.
- Recovery: sample accepted calls.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-29 — queue overload
- Symptom: evidence-based objection reporting is unreliable under queue overload.
- Mechanism: reviewed calls exceed staffed capacity.
- Diagnostic: arrival/service/age.
- Recovery: throttle report release.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-30 — policy version drift
- Symptom: evidence-based objection reporting is unreliable under policy version drift.
- Mechanism: definition of objection changes.
- Diagnostic: rubric version mismatch.
- Recovery: reprocess or separate cohorts.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-31 — translation substitution
- Symptom: evidence-based objection reporting is unreliable under translation substitution.
- Mechanism: translated text replaces original evidence.
- Diagnostic: language/source mismatch.
- Recovery: retain original plus reviewed translation.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-32 — model rollback mismatch
- Symptom: evidence-based objection reporting is unreliable under model rollback mismatch.
- Mechanism: ASR model changes but downstream parser stays pinned.
- Diagnostic: release-bundle audit.
- Recovery: roll back bundle atomically.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-33 — consent scope
- Symptom: evidence-based objection reporting is unreliable under consent scope.
- Mechanism: audio use exceeds allowed purpose.
- Diagnostic: authorization check.
- Recovery: deny processing.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-34 — call merge
- Symptom: evidence-based objection reporting is unreliable under call merge.
- Mechanism: two calls are joined by identifier error.
- Diagnostic: telephony metadata conflict.
- Recovery: split and invalidate aggregate.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### FM-35 — reviewer fatigue
- Symptom: evidence-based objection reporting is unreliable under reviewer fatigue.
- Mechanism: long calls lower adjudication consistency.
- Diagnostic: agreement by duration.
- Recovery: split work and sample second review.
- Verification: replay the bounded audio span, preserve original timing, record disposition, and compare to an independent reviewer.
- Boundary: do not infer missing speech, speaker role, intent, or authorization from confidence alone.

### Failure recovery ledger

#### FM-01 state trace — channel clipping
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: channel clipping is detected before aggregate publication.
3. Cause hypothesis A: audio samples saturate and consonants disappear.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: clipping ratio and waveform peak histogram.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: route to audio-quality review.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-02 state trace — cross-talk
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: cross-talk is detected before aggregate publication.
3. Cause hypothesis A: two speakers overlap.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: overlap detector and diarization posterior.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: preserve both channels and route.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-03 state trace — speaker swap
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: speaker swap is detected before aggregate publication.
3. Cause hypothesis A: diarization assigns turns to the wrong role.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: role-consistency against known greeting/agent channel.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: relabel with reviewer evidence.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-04 state trace — code switching
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: code switching is detected before aggregate publication.
3. Cause hypothesis A: language changes inside a turn.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: word-level language posterior.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: use supported bilingual review route.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-05 state trace — domain term
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: domain term is detected before aggregate publication.
3. Cause hypothesis A: product or claim term is out of vocabulary.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: lexicon miss plus low token posterior.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: flag term span for reviewer.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-06 state trace — acronym ambiguity
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: acronym ambiguity is detected before aggregate publication.
3. Cause hypothesis A: letters map to multiple business terms.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: context alternatives and confidence margin.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: retain verbatim span and ask reviewer.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-07 state trace — VAD truncation
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: VAD truncation is detected before aggregate publication.
3. Cause hypothesis A: voice activity detector cuts a short objection.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: boundary padding audit.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: rerun with protected boundary.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-08 state trace — hold music
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: hold music is detected before aggregate publication.
3. Cause hypothesis A: non-speech segment is transcribed as speech.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: music/speech classifier.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: remove segment and preserve timing.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-09 state trace — background television
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: background television is detected before aggregate publication.
3. Cause hypothesis A: external speech contaminates channel.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: source/separation diagnostic.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: route or mark evidence insufficient.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-10 state trace — packet loss
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: packet loss is detected before aggregate publication.
3. Cause hypothesis A: network gaps remove words.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: packet-loss marker and discontinuity.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: mark omission and abstain.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-11 state trace — timestamp drift
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: timestamp drift is detected before aggregate publication.
3. Cause hypothesis A: ASR times no longer align with call recording.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: anchor-word offset check.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: re-align or reject timestamps.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-12 state trace — redaction error
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: redaction error is detected before aggregate publication.
3. Cause hypothesis A: sensitive content is exposed or destroys needed evidence.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: redaction span audit.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: block report and repair policy.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-13 state trace — summary hallucination
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: summary hallucination is detected before aggregate publication.
3. Cause hypothesis A: aggregation asserts an objection absent from a span.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: span-entailment check.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: remove assertion and route.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-14 state trace — negation loss
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: negation loss is detected before aggregate publication.
3. Cause hypothesis A: “not satisfied” becomes “satisfied”.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: negation cue comparison.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: require human confirmation.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-15 state trace — numeric confusion
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: numeric confusion is detected before aggregate publication.
3. Cause hypothesis A: amount or date is misheard.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: number grammar and audio replay.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: retain raw audio evidence.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-16 state trace — accent shift
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: accent shift is detected before aggregate publication.
3. Cause hypothesis A: acoustic population differs from evaluation.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: slice WER and calibration shift.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: route new slice for gold labeling.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-17 state trace — microphone change
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: microphone change is detected before aggregate publication.
3. Cause hypothesis A: new device changes frequency response.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: device metadata and spectral shift.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: hold automatic route.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-18 state trace — agent script change
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: agent script change is detected before aggregate publication.
3. Cause hypothesis A: role language changes after training.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: script/version drift.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: refresh evaluation and route.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-19 state trace — long silence
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: long silence is detected before aggregate publication.
3. Cause hypothesis A: silence changes segmentation and conversation state.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: silence-duration trace.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: preserve event and avoid invented turn.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-20 state trace — transfer between agents
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: transfer between agents is detected before aggregate publication.
3. Cause hypothesis A: call leg changes speaker identity.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: telephony transfer event.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: split leg and re-establish roles.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-21 state trace — duplicate recording
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: duplicate recording is detected before aggregate publication.
3. Cause hypothesis A: same call is ingested twice.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: call/content hash.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: deduplicate aggregate count.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-22 state trace — partial recording
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: partial recording is detected before aggregate publication.
3. Cause hypothesis A: call begins after the relevant question.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: capture-start marker.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: abstain from completeness claim.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-23 state trace — low volume
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: low volume is detected before aggregate publication.
3. Cause hypothesis A: words are below intelligibility.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: signal-to-noise estimate.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: request/review recording.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-24 state trace — reverberation
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: reverberation is detected before aggregate publication.
3. Cause hypothesis A: echo smears phonemes.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: room impulse indicator.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: route quality slice.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-25 state trace — emotion overreach
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: emotion overreach is detected before aggregate publication.
3. Cause hypothesis A: prosody score is treated as intent.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: evidence-span absence.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: report only supported words.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-26 state trace — sentiment aggregation
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: sentiment aggregation is detected before aggregate publication.
3. Cause hypothesis A: one utterance is treated as whole-call outcome.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: denominator and turn coverage.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: require explicit aggregation rule.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-27 state trace — late label
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: late label is detected before aggregate publication.
3. Cause hypothesis A: review outcome arrives after reporting cutoff.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: label timestamp.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: hold metric update.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-28 state trace — selective review
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: selective review is detected before aggregate publication.
3. Cause hypothesis A: only difficult calls receive gold labels.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: route propensity.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: sample accepted calls.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-29 state trace — queue overload
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: queue overload is detected before aggregate publication.
3. Cause hypothesis A: reviewed calls exceed staffed capacity.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: arrival/service/age.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: throttle report release.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-30 state trace — policy version drift
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: policy version drift is detected before aggregate publication.
3. Cause hypothesis A: definition of objection changes.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: rubric version mismatch.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: reprocess or separate cohorts.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-31 state trace — translation substitution
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: translation substitution is detected before aggregate publication.
3. Cause hypothesis A: translated text replaces original evidence.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: language/source mismatch.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: retain original plus reviewed translation.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-32 state trace — model rollback mismatch
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: model rollback mismatch is detected before aggregate publication.
3. Cause hypothesis A: ASR model changes but downstream parser stays pinned.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: release-bundle audit.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: roll back bundle atomically.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-33 state trace — consent scope
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: consent scope is detected before aggregate publication.
3. Cause hypothesis A: audio use exceeds allowed purpose.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: authorization check.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: deny processing.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-34 state trace — call merge
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: call merge is detected before aggregate publication.
3. Cause hypothesis A: two calls are joined by identifier error.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: telephony metadata conflict.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: split and invalidate aggregate.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

#### FM-35 state trace — reviewer fatigue
1. Preconditions: consent, call-leg identity, audio hash, and capture bounds are retained.
2. Trigger: reviewer fatigue is detected before aggregate publication.
3. Cause hypothesis A: long calls lower adjudication consistency.
4. Cause hypothesis B: stale release bundle, identity error, or source-version mismatch produces a similar symptom.
5. Discriminator: agreement by duration.
6. State transition: `CANDIDATE` becomes `ROUTED` or `BLOCKED`, not `AGGREGATED`.
7. Immediate control: split work and sample second review.
8. Correction: update reviewed evidence only under the same call-leg/source version; retain the original hypothesis.
9. Prevention: monitor rate by channel, language, device, route, model version, and rubric version.
10. Verification: a qualified reviewer can replay the cited span and see the final disposition.
11. Escalation: unresolved consent, identity, or authorization faults stop processing and go to the accountable owner.

## 13. Transferable Lesson

Speech analytics becomes defensible when a reported aggregate is traceable to authorised, replayable, time-bounded evidence rather than a fluent transcript alone.
A capacity model is part of the quality design: routing uncertainty to people fails if the queue cannot be staffed.
Role, timing, language, audio quality, rubric version, and selective labels are separate sources of bias that must not be averaged away.

## 14. Connections & Cross-References

Related material: Module 18 ASR evaluation and task-contract sheets; Module 20 permission-aware retrieval. These links do not establish evidence for this illustrative case.

### Interview pack

#### Q1. What is the unit of evaluation?
**Answer:** The unit is a call leg, speaker turn, evidence span, and declared aggregate denominator; a whole-call transcript is insufficient.
**Reasoning:** The reporting decision is valid only when evidence, authority, denominator, and operational capacity are jointly defined.
**Follow-up:** Which population and timestamp boundary would reverse this conclusion?
**Acceptable alternatives:** A different metric or workflow is acceptable when it preserves independent evidence and explicit authority.
**Red flags:** Treating a fluent transcript, average WER, or model confidence as sufficient proof.
**Scoring:** 3 = names contract and boundary; 2 = mechanism but incomplete boundary; 1 = generic model description.
**Answer support:** Sections 5–12.

#### Q2. Why is WER insufficient for objection reporting?
**Answer:** WER measures token differences, while objection reporting also requires correct speaker, span, rubric, authorization, and denominator.
**Reasoning:** The reporting decision is valid only when evidence, authority, denominator, and operational capacity are jointly defined.
**Follow-up:** Which population and timestamp boundary would reverse this conclusion?
**Acceptable alternatives:** A different metric or workflow is acceptable when it preserves independent evidence and explicit authority.
**Red flags:** Treating a fluent transcript, average WER, or model confidence as sufficient proof.
**Scoring:** 3 = names contract and boundary; 2 = mechanism but incomplete boundary; 1 = generic model description.
**Answer support:** Sections 5–12.

#### Q3. How do you prevent speaker leakage?
**Answer:** Split by caller/call and time, preserve telephony identity, and prohibit same-call turns across partitions.
**Reasoning:** The reporting decision is valid only when evidence, authority, denominator, and operational capacity are jointly defined.
**Follow-up:** Which population and timestamp boundary would reverse this conclusion?
**Acceptable alternatives:** A different metric or workflow is acceptable when it preserves independent evidence and explicit authority.
**Red flags:** Treating a fluent transcript, average WER, or model confidence as sufficient proof.
**Scoring:** 3 = names contract and boundary; 2 = mechanism but incomplete boundary; 1 = generic model description.
**Answer support:** Sections 5–12.

#### Q4. When should the workflow abstain?
**Answer:** Abstain for missing consent, incomplete capture, unknown route/language, unresolved speaker role, absent evidence, or capacity breach.
**Reasoning:** The reporting decision is valid only when evidence, authority, denominator, and operational capacity are jointly defined.
**Follow-up:** Which population and timestamp boundary would reverse this conclusion?
**Acceptable alternatives:** A different metric or workflow is acceptable when it preserves independent evidence and explicit authority.
**Red flags:** Treating a fluent transcript, average WER, or model confidence as sufficient proof.
**Scoring:** 3 = names contract and boundary; 2 = mechanism but incomplete boundary; 1 = generic model description.
**Answer support:** Sections 5–12.

#### Q5. How is reviewer capacity calculated?
**Answer:** Compute eligible legs times route fraction times staffed minutes, then compare with staffed capacity and variation boundaries.
**Reasoning:** The reporting decision is valid only when evidence, authority, denominator, and operational capacity are jointly defined.
**Follow-up:** Which population and timestamp boundary would reverse this conclusion?
**Acceptable alternatives:** A different metric or workflow is acceptable when it preserves independent evidence and explicit authority.
**Red flags:** Treating a fluent transcript, average WER, or model confidence as sufficient proof.
**Scoring:** 3 = names contract and boundary; 2 = mechanism but incomplete boundary; 1 = generic model description.
**Answer support:** Sections 5–12.

#### Q6. Why retain audio evidence?
**Answer:** Audio hash and replayable offsets let an authorised reviewer audit transcript and rubric claims without trusting generated text.
**Reasoning:** The reporting decision is valid only when evidence, authority, denominator, and operational capacity are jointly defined.
**Follow-up:** Which population and timestamp boundary would reverse this conclusion?
**Acceptable alternatives:** A different metric or workflow is acceptable when it preserves independent evidence and explicit authority.
**Red flags:** Treating a fluent transcript, average WER, or model confidence as sufficient proof.
**Scoring:** 3 = names contract and boundary; 2 = mechanism but incomplete boundary; 1 = generic model description.
**Answer support:** Sections 5–12.

#### Q7. How do you evaluate code-switching?
**Answer:** Measure independent gold by language/switching slice, role, channel, and domain terms; do not transfer aggregate WER.
**Reasoning:** The reporting decision is valid only when evidence, authority, denominator, and operational capacity are jointly defined.
**Follow-up:** Which population and timestamp boundary would reverse this conclusion?
**Acceptable alternatives:** A different metric or workflow is acceptable when it preserves independent evidence and explicit authority.
**Red flags:** Treating a fluent transcript, average WER, or model confidence as sufficient proof.
**Scoring:** 3 = names contract and boundary; 2 = mechanism but incomplete boundary; 1 = generic model description.
**Answer support:** Sections 5–12.

#### Q8. What is a valid baseline?
**Answer:** Authorized human listening/transcription under the current rubric is the baseline.
**Reasoning:** The reporting decision is valid only when evidence, authority, denominator, and operational capacity are jointly defined.
**Follow-up:** Which population and timestamp boundary would reverse this conclusion?
**Acceptable alternatives:** A different metric or workflow is acceptable when it preserves independent evidence and explicit authority.
**Red flags:** Treating a fluent transcript, average WER, or model confidence as sufficient proof.
**Scoring:** 3 = names contract and boundary; 2 = mechanism but incomplete boundary; 1 = generic model description.
**Answer support:** Sections 5–12.

#### Q9. How do timestamps affect aggregation?
**Answer:** A span must use a stable audio clock; drift can attach correct words to the wrong event or call segment.
**Reasoning:** The reporting decision is valid only when evidence, authority, denominator, and operational capacity are jointly defined.
**Follow-up:** Which population and timestamp boundary would reverse this conclusion?
**Acceptable alternatives:** A different metric or workflow is acceptable when it preserves independent evidence and explicit authority.
**Red flags:** Treating a fluent transcript, average WER, or model confidence as sufficient proof.
**Scoring:** 3 = names contract and boundary; 2 = mechanism but incomplete boundary; 1 = generic model description.
**Answer support:** Sections 5–12.

#### Q10. How is a new microphone introduced?
**Answer:** Treat it as a new slice: capture metadata, create independent gold, measure degradation, and keep review routing until approved.
**Reasoning:** The reporting decision is valid only when evidence, authority, denominator, and operational capacity are jointly defined.
**Follow-up:** Which population and timestamp boundary would reverse this conclusion?
**Acceptable alternatives:** A different metric or workflow is acceptable when it preserves independent evidence and explicit authority.
**Red flags:** Treating a fluent transcript, average WER, or model confidence as sufficient proof.
**Scoring:** 3 = names contract and boundary; 2 = mechanism but incomplete boundary; 1 = generic model description.
**Answer support:** Sections 5–12.

#### Q11. Why sample accepted calls?
**Answer:** Selective review makes routed quality visible but can hide accepted errors; independently sample accepted calls.
**Reasoning:** The reporting decision is valid only when evidence, authority, denominator, and operational capacity are jointly defined.
**Follow-up:** Which population and timestamp boundary would reverse this conclusion?
**Acceptable alternatives:** A different metric or workflow is acceptable when it preserves independent evidence and explicit authority.
**Red flags:** Treating a fluent transcript, average WER, or model confidence as sufficient proof.
**Scoring:** 3 = names contract and boundary; 2 = mechanism but incomplete boundary; 1 = generic model description.
**Answer support:** Sections 5–12.

#### Q12. What changes after rubric revision?
**Answer:** Pin rubric version in every aggregate, separate cohorts, and reprocess only with an approved comparison plan.
**Reasoning:** The reporting decision is valid only when evidence, authority, denominator, and operational capacity are jointly defined.
**Follow-up:** Which population and timestamp boundary would reverse this conclusion?
**Acceptable alternatives:** A different metric or workflow is acceptable when it preserves independent evidence and explicit authority.
**Red flags:** Treating a fluent transcript, average WER, or model confidence as sufficient proof.
**Scoring:** 3 = names contract and boundary; 2 = mechanism but incomplete boundary; 1 = generic model description.
**Answer support:** Sections 5–12.

#### Q13. Why is emotion not an objection label?
**Answer:** Prosody can be a signal for review, not proof of an objection; report only rubric-supported words and spans.
**Reasoning:** The reporting decision is valid only when evidence, authority, denominator, and operational capacity are jointly defined.
**Follow-up:** Which population and timestamp boundary would reverse this conclusion?
**Acceptable alternatives:** A different metric or workflow is acceptable when it preserves independent evidence and explicit authority.
**Red flags:** Treating a fluent transcript, average WER, or model confidence as sufficient proof.
**Scoring:** 3 = names contract and boundary; 2 = mechanism but incomplete boundary; 1 = generic model description.
**Answer support:** Sections 5–12.

#### Q14. How do you roll back safely?
**Answer:** Roll back audio parser, ASR, diarizer, rubric, aggregate mapping, and evidence manifest as one compatible release bundle.
**Reasoning:** The reporting decision is valid only when evidence, authority, denominator, and operational capacity are jointly defined.
**Follow-up:** Which population and timestamp boundary would reverse this conclusion?
**Acceptable alternatives:** A different metric or workflow is acceptable when it preserves independent evidence and explicit authority.
**Red flags:** Treating a fluent transcript, average WER, or model confidence as sufficient proof.
**Scoring:** 3 = names contract and boundary; 2 = mechanism but incomplete boundary; 1 = generic model description.
**Answer support:** Sections 5–12.

#### Q15. What makes a report auditable?
**Answer:** It records source hash, call leg, consent decision, capture bounds, span offsets, versions, reviewer disposition, and report inclusion.
**Reasoning:** The reporting decision is valid only when evidence, authority, denominator, and operational capacity are jointly defined.
**Follow-up:** Which population and timestamp boundary would reverse this conclusion?
**Acceptable alternatives:** A different metric or workflow is acceptable when it preserves independent evidence and explicit authority.
**Red flags:** Treating a fluent transcript, average WER, or model confidence as sufficient proof.
**Scoring:** 3 = names contract and boundary; 2 = mechanism but incomplete boundary; 1 = generic model description.
**Answer support:** Sections 5–12.

## 15. Quick Reference Card

| Field | Rule | Boundary |
|---|---|---|
| Input | Authorised audio + identity + capture contract | Do not process unclear consent. |
| Output | Evidence span, route, or abstention | Never an individual judgment. |
| Baseline | Authorised manual listening | Compare full workflow. |
| Capacity | W=N*r*m | Constant averages are not a queue guarantee. |
| Aggregate | Use pinned rubric and denominator | Do not mix versions. |
| Evidence | Audio hash + offsets + versions + disposition | Transcript alone is insufficient. |
| State | CANDIDATE -> ROUTED -> REVIEWED -> AGGREGATED | No direct candidate aggregation. |
| Evidence level | Illustrative, not-measured | No operational claim. |
