---
case_id: "CASE-18-02"
case_title: "Call-centre transcription quality for evidence-based objection reporting"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "hybrid"
domain: "Speech analytics / call-centre quality"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["human-in-the-loop", "authorization-scope"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Customer-operations analytics owner"
source_index: "INDEX.md"
review_trigger: "Audio channel, rubric, language route, consent policy, or review-capacity change"
---

# Call-Centre Transcription Quality: Evidence Before Aggregation

**Evidence disclaimer.** Every count in this manuscript is a constructed teaching input, labelled
ILLUSTRATIVE at its point of use. No contact centre, telephony platform, ASR vendor, or regulator
result is asserted, inspected, or reproduced. The arithmetic is verified: each derivation is
recomputed in the text so a reader can redo it with a calculator. Where the text says a quantity
"would" or "is projected to", it is a design projection conditional on the stated assumptions, not
an observation. The only claims carrying external authority are the formula definitions, which
follow standard speech-recognition and queueing conventions cited in §14. Where evidence is absent
the text writes `unknown` or `not-measured` rather than inventing a number.

## 1. The One-Line Framing

Turn an authorised call recording plus its telephony metadata into evidence-linked transcript
spans, then decide per call whether to publish an objection count, route the call to a human
reviewer, or withhold it — because a fluent transcript is not proof that a customer objected.

The decision unit is not the call and not the transcript. It is the **objection span**: a bounded
interval of authorised audio, attributed to a speaker role, matching a written rubric, carrying a
replayable offset and a version manifest. Everything downstream — the aggregate count, its
denominator, its confidence interval, the reviewer queue that produced it — is a function of how
many such spans survive scrutiny.

Three outputs are admissible and they are not interchangeable:

1. **Aggregate candidate** — a span that passed every gate, with machine-checkable provenance, that
   may enter a published count under a pinned denominator.
2. **Route** — a span that a human must hear before anyone may count it, with the routing reason
   recorded, so the queue is auditable rather than a silent dumping ground.
3. **Abstention** — an explicit refusal (consent unclear, capture incomplete, role unresolved,
   language unsupported, rubric ambiguous) that removes the call from the numerator *and* the
   denominator, with the removal counted.

The output is never a performance score for an agent, never a customer risk decision, and never an
employment signal. That boundary is a design constraint, not a disclaimer bolted on at the end: it
is the reason the system refuses to emit per-agent aggregates at all.

## 2. Context & Background

### 2.1 What the workflow actually looks like

A contact centre is a chain of lossy handoffs, and every hop leaves a mark on the audio that later
shows up as an apparent model defect.

- **Capture.** A call arrives over SIP, is answered by an IVR or a queue, and is bridged to an
  agent. Recording may be trunk-side (before the PBX), station-side (at the handset), or
  screen-and-voice via the agent desktop. Trunk-side recording hears what the network delivered;
  station-side recording hears what the headset microphone produced. The two disagree about level,
  noise floor, and whether the customer's voice is attenuated by the agent's headset sidetone.
- **Codec.** Narrowband G.711 at 8 kHz with a nominal 300–3400 Hz passband is still common on
  PSTN-facing legs; wideband Opus at 16 or 48 kHz appears on softphone and WebRTC legs. The same
  sentence spoken by the same person has measurably different acoustic information depending on
  which leg carried it.
- **Loss and concealment.** Jitter buffers conceal lost packets by repeating the last frame. The
  result is not silence but a stretched vowel, which an acoustic model happily transcribes as a
  longer word.
- **Bridging and hold.** A consult transfer puts the customer on hold music, sometimes with a
  recorded announcement, sometimes with silence and comfort noise. Hold segments are speech-like
  to a voice-activity detector and contain no customer speech at all.
- **Dual-channel versus mixed.** Where the two parties arrive on separate channels, diarisation is
  nearly free and speaker attribution is a metadata lookup. Where they arrive mixed, attribution
  becomes an inference problem with its own error rate.
- **Consent and purpose.** Many jurisdictions require an announcement, a lawful basis, or both
  before a recording may be used for quality analysis beyond the original service purpose.
  Re-purposing a service recording for aggregate objection analytics is a distinct purpose in most
  privacy regimes, and the lawful basis is often narrower than the recording's existence suggests.
- **Retention.** Audio is typically retained for a short window (weeks to a few months) while
  derived text or features may be retained longer. Any claim that a report is auditable must
  survive the audio's deletion schedule; a span whose audio has expired is not evidence, it is a
  recollection with a timestamp.

### 2.2 Why the naive pipeline fails

The tempting pipeline is: transcribe everything, keyword-match objection phrases, count per day,
chart the trend. Each of the four steps has a specific failure that the naive design hides:

- **Transcribe everything** assumes every recording is authorised and complete. Neither is true for
  a whole day of traffic: consent varies by campaign, and captures fail (dropped bridge, storage
  timeout, late start).
- **Keyword-match** assumes the transcript is faithful at the exact words that matter. Entity-like
  tokens — product names, amounts, dates, references — are precisely where word error concentrates,
  and they are also where objection language lives.
- **Count per day** assumes the denominator is stable. It is not: the set of processed calls changes
  with capture success and consent coverage, so a rising count can be a rising numerator or a
  shrinking denominator.
- **Chart the trend** assumes the label means the same thing across time. A rubric edit, a model
  upgrade, or a new queue changes the meaning of the count without changing its name.

The trap this case is built around is the fourth one, because it is invisible in the data: two
months of counts plotted side by side look comparable even when the second month was produced by a
different rubric over a different route mix.

### 2.3 Who is in the room

Four roles with genuinely different objectives, and the design fails if any of them is modelled as
a rubber stamp:

| Role | Wants | Will push back on |
|---|---|---|
| Customer-operations analytics owner | A defensible number with an interval | A number whose denominator it cannot explain |
| Quality reviewer | Fewer, better-targeted items; replay that works first time | A queue that arrives stale and unreplayable |
| Privacy / compliance officer | Purpose limitation, minimisation, retention proof | Silent re-purposing of service recordings |
| Contact-centre operations lead | No agent-facing scorecard built from the output | Any per-agent slice escaping into workforce management |

The analytics owner is the decision owner for this case. The compliance officer holds a veto that
the design encodes as a hard gate rather than a review step, because a review step can be skipped
under deadline pressure and a gate cannot.

### 2.4 What this case is not

It is not a vendor comparison of ASR engines, not a claim about any model's accuracy, and not an
argument that automated objection detection should replace listening. It is an argument that the
*evidence contract* — what a span must carry before it may be counted — determines whether the
report survives an audit, and that the contract costs reviewer minutes that must be budgeted like
any other capacity constraint.

## 3. The Business Question

**Stakeholder question (ILLUSTRATIVE framing):** "Our quality team listens to a sample of calls and
files objection notes by hand. Can we produce a monthly aggregate of customer objections by
category, with each count traceable to a replayable moment in an authorised recording, without
turning the output into an agent scorecard and without breaking either our review capacity or our
consent commitments?"

Decomposed into the five decisions the system must actually make:

1. **Admissibility.** Is this recording usable for this purpose at all? Binary, per call, decided
   before any transcription compute is spent. Fails closed.
2. **Span candidacy.** Does this stretch of speech contain a rubric-matching objection? If yes,
   where exactly does it start and end, and which tokens carry the claim?
3. **Attribution.** Which speaker role said it — customer, agent, IVR, or unknown? A rubric-matching
   sentence spoken by the agent while reading a script is not a customer objection.
4. **Publication.** Does this span enter the aggregate as-is, go to a reviewer, or leave the count?
   This is where calibration and review capacity meet.
5. **Denominator.** Which calls are in scope for the report, and what happens to the ones removed
   for consent, capture failure, or unsupported language?

Decision options, stated so they can be compared rather than traded off rhetorically:

| Option | What it produces | Cost profile | Failure mode |
|---|---|---|---|
| A. Manual listening only | Reviewed notes from a sampling frame | High minutes per call, small sample | Coverage so thin that category trends are noise |
| B. Transcribe, count matches, publish | Fast aggregate | Low marginal cost | Transcript error becomes report error, silently |
| C. Transcribe, gate, route uncertainty, sample the rest | Aggregate with interval and audit trail | Low base cost plus bounded reviewer load | Reviewer queue becomes the bottleneck if unbudgeted |
| D. Transcribe, publish per-agent rates | Agent-level coaching input | Moderate | Out of scope by design; employment-adjacent use |

Option D is listed to be rejected explicitly. It is the option an operations lead will ask for,
and the only honest answer is that the sample sizes and the attribution error rate cannot support
per-agent inference even when the aggregate is sound.

Hard constraints, each of which can independently sink the project:

- **C-1 Consent and purpose.** No processing without a recorded lawful basis for this purpose.
- **C-2 No per-person output.** Aggregates only, above a minimum cell size, never joined to agent
  identifiers.
- **C-3 Evidence for every count.** A published number must resolve to spans, and every span must
  resolve to audio that still exists and can be replayed.
- **C-4 Explicit attribution uncertainty.** An unknown-role span cannot be counted as a customer
  objection; it must be routed or dropped, and the drop counted.
- **C-5 Capacity honesty.** The routing policy must be feasible at the peak hour, not merely on the
  daily average.
- **C-6 Version pinning.** Rubric, model, audio parser, and aggregation mapping move as one
  release bundle; mixed-version cohorts are reported separately or not at all.

## 4. Assumptions & Constraints

Every assumption below is a place where the design could be wrong in a way the data will not
announce. The check column names the artifact that would falsify it; the fallback names the
behaviour when the check fails. A design that cannot state its fallback is not ready.

| ID | Assumption | Check that can falsify it | Impact if false | Fallback |
|---|---|---|---|---|
| A-01 | The recording is authorised for aggregate quality analytics, not merely for service delivery | Consent/purpose record keyed to campaign and date | Processing is unlawful; every derived count is unusable | Deny the call; count the denial |
| A-02 | Call and leg identity are stable from capture to report | CTI event replay versus recording header | Spans attach to the wrong customer; denominator double-counts | Quarantine the call; exclude from both numerator and denominator |
| A-03 | Capture covers the full conversation, including transfers | Bridge-event timeline versus audio duration | Truncated calls look like calls with no objection | Exclude; never impute missing speech |
| A-04 | Gold annotation is independent of model output | Blinded double annotation with adjudication | Measured quality is inflated by anchoring | Re-annotate blinded; discard the anchored set |
| A-05 | Dual-channel audio exists for the majority of legs | Channel-count histogram per queue | Attribution error rises sharply on mixed legs | Route all mixed-channel spans |
| A-06 | Reviewer service time is roughly stationary across the queue mix | Time-stamped console events per disposition | Capacity plan understates load on complex spans | Re-estimate per stratum; staff to the worst stratum |
| A-07 | Rubric phrases are stable within a reporting month | Rubric version stamp per span | Month-over-month comparison is meaningless | Split the series at the version change |
| A-08 | Reviewers can replay the audio referenced by a span | Replay success rate on a sampled audit | "Evidence-linked" degenerates to "asserted" | Block publication for the affected cohort |
| A-09 | The audit sample is drawn from accepted spans, not routed ones | Sampling-frame log with inclusion probabilities | Accepted-error rate is invisible; precision is overstated | Resample from the full accepted frame |
| A-10 | Language coverage matches the traffic mix | Language identification versus queue configuration | Silent degradation on the uncovered tail | Abstain on unsupported languages; report coverage |
| A-11 | Audio retention outlives the reporting window | Retention schedule versus report cadence | Older counts become unauditable | Shorten the comparison window or freeze spans earlier |
| A-12 | Reviewer headcount is available at the peak hour | Roster overlap with the arrival curve | Queue explodes exactly when traffic peaks | Cap routing at the peak-feasible share |
| A-13 | Objection language is expressed in words, not only in prosody | Comparison of lexical matches to human-noted objections | Recall ceiling is set by phrasing, not by audio quality | Report "stated objections only"; route prosody-only candidates |
| A-14 | Aggregate cells are large enough to publish | Cell-size check per category and month | Small cells become de-facto individual disclosure | Suppress cells below the threshold; report suppression count |

Two structural constraints sit outside the assumption table because they are not falsifiable, they
are chosen:

- **No imputation of missing speech.** Where the capture is incomplete, the call is excluded. A
  model asked to "fill in" a dropped segment produces text that no reviewer can verify, which
  destroys the evidence chain that the whole design exists to protect.
- **No retro-fitted labels.** A span is labelled with the rubric version in force at review time,
  not the version current at report time. Re-labelling history to match today's rubric is how a
  trend line becomes fiction.

The cost of these constraints is real and should be stated up front: exclusion reduces the
denominator, and a reduced denominator makes every per-100-calls figure larger. §6 quantifies that
effect rather than waving it away, because a 7% exclusion and a 30% exclusion have very different
consequences for whether a monthly comparison is worth publishing.
## 5. Data, Tools & Method

### 5a. Data Specification

**Grain.** Three nested grains, and mixing them is the most common cause of a wrong number:

| Grain | One row is | Primary key | Used for |
|---|---|---|---|
| Call leg | One recorded audio stream from bridge to teardown | `call_id` + `leg_id` + `recording_version` | Admissibility, denominators, capacity |
| Turn | One contiguous same-speaker interval | Above + `turn_index` | Attribution error, diarisation metrics |
| Objection span | One rubric-matching stretch inside a turn | Above + `span_start_ms`, `span_end_ms` | Counting, review, publication |

The span grain is the only one that may enter a published objection count. A turn without a span
is attribution evidence; a call without a span is a denominator member, not a zero.

**Cut-off and time semantics.** Two clocks must never be conflated:

- `event_time` — when the conversation happened (bridge start, teardown).
- `processing_time` — when transcription, review, or aggregation completed.

A report labelled "calls on 12 March" is defined on `event_time` inside a stated UTC window,
including calls that started before midnight and ended after. Late-arriving records are admitted
only until a declared close of books. Without that declaration, a monthly figure silently drifts
upward for weeks as stragglers land, and nobody can reproduce last month's number.

**Population.** All call legs entering queues in scope, for campaigns with a recorded lawful basis,
where the recording exists and its capture bounds are known. Excluded, with reasons counted:

- Legs without a lawful basis for aggregate analytics (even when recorded for service quality).
- Legs whose capture is incomplete relative to the bridge-event timeline.
- Legs in languages or dialects outside the declared supported set.
- Legs whose recording hash fails integrity verification.
- Duplicate legs (same recording delivered twice under different transport identifiers).

**Keys and joins.** `call_id` is the telephony correlation identifier; `leg_id` disambiguates
transfers and conference joins. Every derived artifact carries `recording_sha256` so that a
re-encode or re-delivery cannot masquerade as new evidence. The join to review is by
`(call_id, leg_id, recording_sha256, span_start_ms, span_end_ms)`, deliberately verbose because a
shorter key silently merges spans after an offset correction.

**Units.** Durations in milliseconds for spans, seconds for audio totals, staffed minutes for
capacity. Token counts are whitespace-delimited after a frozen normalisation. Confidence is a
probability in `[0,1]` produced by a specific model revision and is not comparable across
revisions.

**Not available at decision time** — the discipline that keeps the evaluation honest:

| Signal | Why it is attractive | Why it is unavailable |
|---|---|---|
| Supervisor follow-up notes | They name the real objection | Written after the fact, often after a complaint |
| Complaint or escalation outcome | Strong label proxy | Occurs days later; unavailable when the call is scored |
| Reviewer's own correction text | Improves the transcript | It is the thing being evaluated, not an input to it |
| Churn or cancellation flag | Tempting target | Post-dates the call by weeks and is confounded by price and coverage |
| Agent tenure or team | Explains phrasing | Would create a per-agent slice by the back door |
| Rubric version released later in the month | Looks like a bug fix | Using it would re-label history to today's definition |

### 5b. Tools & Stack

The stack is described by the contract each component must honour, not by product names, because
the argument holds for any vendor meeting the contract.

| Component | Contract it must honour | What breaks without it |
|---|---|---|
| Canonical audio reader | Preserve sample rate, channel map, gap markers, and source hash; refuse to repair silently | Re-sampling or gap-filling shifts every span offset downstream |
| Voice-activity segmenter | Emit speech intervals with explicit uncertainty at boundaries; never emit "speech" for music or comfort noise | Hold music becomes hallucinated objection text |
| ASR engine | Emit token text, token start/end in the canonical clock, and a per-token confidence from a stated revision | Without timing there is no span; without revision no confidence is interpretable |
| Diariser | Emit speaker segments with a posterior and an explicit unknown label | Forced two-speaker output turns silence into an agent turn |
| Language identifier | Emit per-turn and per-token language evidence with a supported/unsupported decision | Code-switched turns get transcribed in one language and silently corrupted |
| Evidence service | Bind tokens to recording offsets and to a configuration manifest; serve replay from the original bytes | The audit trail becomes a screenshot |
| Review console | Play a bounded window around the span, record disposition and reason code, enforce blinding to model confidence | Reviewers anchor on the model and stop being an independent check |
| Aggregation service | Accept only reviewed or policy-approved span states, under a pinned rubric and denominator | Numbers appear from states nobody approved |
| Release bundler | Treat audio parser, ASR, diariser, rubric, and aggregation mapping as one atomic version | A partial rollback produces a cohort that never existed |

Two tooling choices deserve explicit justification because they are frequently made badly:

- **Streaming versus batch.** Streaming buys nothing here. Objection reporting is a monthly
  aggregate; a five-minute transcription delay is irrelevant, and batch allows retries,
  re-segmentation, and deterministic replay. Streaming is only justified if the same pipeline must
  also drive live agent assist, which is a different product with different risk.
- **Word-level versus utterance-level confidence.** Utterance-level confidence cannot support a
  span-level decision, because the span is a sub-interval of the utterance. A confident utterance
  can contain one badly-recognised entity, and that entity is often the objection.

### 5c. Method Summary

Nineteen steps in five phases. Steps are ordered because each depends on the previous gate, not
because the order is a preference.

**Phase 1 — Admissibility (decided before any ASR compute).**

1. Resolve `call_id`/`leg_id` against CTI events; reject unresolved identity.
2. Verify the lawful basis covers aggregate analytics for this campaign and date.
3. Verify recording integrity against the delivered hash.
4. Compare audio duration to the bridge-event interval; reject if the shortfall exceeds the stated
   tolerance.
5. Record the admissibility disposition with a reason code, including denials.

**Phase 2 — Signal preparation.**

6. Read the audio canonically; preserve gaps as gaps and channels as channels.
7. Segment speech, marking hold music, silence, and overlap explicitly.
8. Detect the language mixture per turn; mark unsupported segments.

**Phase 3 — Hypothesis generation.**

9. Transcribe with token timing and per-token confidence.
10. Diarise with posteriors; map speakers to roles using metadata where available.
11. Attach a role hypothesis plus an explicit `unknown` state; never force a binary.
12. Generate candidate spans against the pinned rubric, with the matching tokens quoted.

**Phase 4 — Gating and routing.**

13. Apply hard gates: admissibility, span replayability, role resolution, rubric version match.
14. Compute a calibrated confidence for each surviving candidate span.
15. Route spans below the calibrated threshold to review, with the trigger recorded.
16. Enforce peak-hour routing capacity; queue overflow is a routing failure, not a silent delay.

**Phase 5 — Aggregation and audit.**

17. Draw an independent audit sample from *accepted* spans, not from routed ones.
18. Compute the aggregate with a Wilson interval on the audit-estimated precision, under a pinned
    denominator and cell-size floor.
19. Publish the manifest: denominator, exclusions by reason, versions, interval, suppression count.

The critical design property is that phases 1 and 4 are *gates*, not checks. A gate can only be
passed or failed, and failure is recorded. A check can be "noted" and overridden, which is what
happens to every soft control under a deadline.

### 5d. Data provenance, quality, exclusions and leakage controls

**Lineage chain.** For every published count the system can produce, without human recall:

`audio hash → call/leg identity → consent record id → capture interval → segmenter version →
ASR revision → diariser revision → rubric version → span offsets → routing decision → reviewer id →
review timestamp → aggregation mapping version → report id`.

A break anywhere in that chain demotes the count from "evidence-linked" to "estimated from a
pipeline we trust", and the report must say which it is.

**Exclusion taxonomy, with counts retained.** Exclusions are not silent. Every excluded leg is
counted in a reason bucket: `NO_BASIS`, `IDENTITY_UNRESOLVED`, `CAPTURE_INCOMPLETE`,
`HASH_MISMATCH`, `LANGUAGE_UNSUPPORTED`, `DUPLICATE`, `AUDIO_UNREADABLE`. The sum of exclusions
plus the processed count must equal the in-scope population, and that identity is checked at
publication time as a hard assertion.

**Leakage controls.** Four separate leaks, each with a different fix:

1. **Annotation leakage.** The gold annotator saw the model transcript. Fix: blind annotation with
   adjudication of disagreements, and compare against a second blinded pass to bound anchoring.
2. **Selection leakage.** The audit sample was drawn only from calls the model flagged. Fix: draw
   from the full accepted frame with known inclusion probabilities.
3. **Split leakage.** Turn-level random splits put the same call in train and test, so the model
   memorises the caller. Fix: split by caller and by time; never by turn.
4. **Version leakage.** Spans scored under rubric v3 are aggregated under rubric v4. Fix: rubric
   version is part of the primary key of every aggregate.

**Quality gates on the labels themselves.** Before any model metric is computed, the gold set is
checked for (a) double-annotated agreement above a stated floor, (b) adjudicated disagreement rate,
and (c) coverage across channel, language, and queue. A gold set that is 90% wideband softphone and
10% legacy headset cannot measure the legacy stratum, which is exactly where the error lives.

### 5e. Reproducibility & Method-Selection Ledger

| ID | Artifact | Purpose | Limitation | Reproducible |
|---|---|---|---|---|
| SRC-01 | Illustrative authorised call manifest (600 calls, 250 consent-eligible subset) | Fixes the population and its exclusions | Constructed; no real calls inspected | n/a — constructed |
| SRC-02 | Illustrative token-level gold for 300 calls (146,880 reference tokens) | Supports the WER decomposition | Constructed counts; not an annotation study | Arithmetic yes, measurement no |
| CFG-01 | Segmenter + ASR + diariser + rubric bundle v3.2 | One atomic version identity | Proposed, not built | Yes, by version label |
| GOLD-01 | Blinded double-annotated objection spans (412 gold spans over 600 calls) | Span precision and recall | Constructed; annotator agreement asserted, not observed | Arithmetic yes |
| CAL-01 | Confidence-bin table, 1,200 candidate spans | Calibration and threshold selection | Constructed bin counts | Arithmetic yes |
| RUN-01 | Staffing and queue calculation | Capacity feasibility at peak | Uses Erlang-C assumptions, no abandonment or retrials | Arithmetic yes |
| RUN-02 | Audit sampling calculation with finite population correction | Sample size for the accepted-frame audit | Assumes a single population proportion | Arithmetic yes |
| AUD-01 | 150-span audit of accepted spans, 141 correct | Establishes published precision | Constructed | Arithmetic yes |

**Why these methods and not the obvious alternatives.**

| Choice made | Alternative rejected | Reason for rejection |
|---|---|---|
| Token-grounded spans | Free-text objection summary | A summary cannot be replayed, so it cannot be audited |
| Rubric matching plus role gate | Sentiment or emotion classification | Prosody is not an objection; it does not survive a rubric test |
| Calibrated threshold on span confidence | Fixed confidence cut at 0.90 | Untested cuts are arbitrary, and confidence is not calibrated by construction |
| Wilson interval on audit-estimated precision | Normal-approximation interval | Proportions near 0.95 with n=150 break the normal approximation |
| Erlang-C peak analysis | Daily-average utilisation | The average says feasible while the peak says collapse |
| Finite population correction | Uncapped sample-size formula | The accepted frame is only a few hundred spans per day |
| Single atomic release bundle | Independent component upgrades | Independent upgrades create cohorts that never existed in production |

## 6. Analysis Walkthrough

Six derivations, each recomputed from stated inputs. Every number is ILLUSTRATIVE. The point of the
walkthrough is not the numbers; it is that each one can be *checked*, and that each one has a
boundary where the formula stops applying. Those boundaries are stated, not omitted.

### 6.1 Derivation CC-01 — Why aggregate WER understates the error that matters

A single word error rate is the first number anyone produces and the last number anyone should act
on. The reason is structural: word errors are not uniformly distributed across token types, and
objection reporting cares almost exclusively about the token types where errors concentrate.

```text
FORMULA [CC-01]: WER = (S + D + I) / N, and for any subset E ⊂ tokens,
                 WER_E = (S_E + D_E + I_E) / N_E
WHERE: S substitutions, D deletions, I insertions (counts, dimensionless);
       N reference tokens; E a declared token subset (entity-like tokens here);
       all four counts from one frozen alignment and one normalisation policy.
CONSTRAINT: N > 0 and N_E > 0; identical tokenisation on both sides; the subset E is
       defined by the gold annotation, not by the model output.
NUMERICAL EXAMPLE: A 300-call illustrative sample yields 1,382.4 speech-minutes and
       146,880 reference tokens, i.e. 106.25 tokens per speech-minute (147.6 words per
       audio-minute at a 0.72 speech fraction). Errors: S = 7,412, D = 1,306, I = 2,015,
       so S+D+I = 10,733 and WER = 10,733 / 146,880 = 0.07307.
       Of those tokens, 9,180 (6.25%) are entity-like and carry 1,412 errors (13.16% of all
       errors), giving WER_E = 1,412 / 9,180 = 0.15381. The remaining 137,700 tokens carry
       10,733 − 1,412 = 9,321 errors: WER_nonE = 9,321 / 137,700 = 0.06769.
INTERPRETATION: Entity error is 2.27 times background error (0.15381 / 0.06769), so a
       headline of 7.3% describes mostly function words and morphology while the tokens
       that carry product names, amounts, and references fail at 15.4%.
```

The multiplication check matters: 6.25% of tokens contribute 13.16% of errors, so
`0.0625 × 0.15381 + 0.9375 × 0.06769 = 0.00961 + 0.06346 = 0.07307`, which reproduces the pooled
figure exactly. This is not decoration; it is the proof that the decomposition is consistent and
that no errors were double-counted between the two subsets.

**Where entity error actually sits (ILLUSTRATIVE, same 9,180-token subset).**

| Entity category | Tokens | Errors | Category WER |
|---|---|---|---|
| Account reference | 1,980 | 402 | 20.30% |
| Plan or product name | 612 | 192 | 31.37% |
| Monetary amount | 1,844 | 268 | 14.53% |
| Date or time | 1,290 | 151 | 11.71% |
| Other named entity | 2,350 | 311 | 13.23% |
| Agent identifier | 1,104 | 88 | 7.97% |
| **Total** | **9,180** | **1,412** | **15.38%** |

Category totals recompute to 9,180 tokens and 1,412 errors, so the table is closed. Plan names are
the worst category at 31.37%: product names are rare in training data, often multi-word, and
frequently pronounced as an acronym that the model expands incorrectly. Account references at
20.30% fail on digit grouping — the model hears "four four seven one" and writes "4471" or
"44 71" depending on the reference's own convention.

**Why this drives the design.** An objection is usually anchored to an entity: *"you charged me
forty-five pounds instead of thirty-five"*, *"I never agreed to the premium plan"*. If the amount
or the plan name is wrong, the objection may still be detected as an objection while being
misclassified as to type, or it may be missed entirely because the rubric phrase names the entity.
A 7.3% headline supports a claim like "objection detection is broadly accurate"; the 15.4% entity
figure does not, and the second is the honest one for this use.

**Boundary where CC-01 breaks.** The decomposition assumes E is defined on gold tokens and is
disjoint from its complement. It fails when the alignment crosses the subset boundary: if the model
substitutes a two-token plan name for one token, the substitution may be counted in `E` while the
deletion lands outside it, and the two subset rates no longer sum to the pooled rate. The check is
the multiplication identity above — when it fails to reproduce the pooled WER to three decimals, a
boundary crossing exists and the subset figure must be recomputed by re-aligning on gold boundaries
rather than by partitioning an existing alignment. A second boundary: if `N_E` is small (a single
queue, one day), the subset rate is unstable, and a 3-error difference over 40 tokens moves the rate
by 7.5 points.

### 6.2 Derivation CC-02 — Diarisation error with a speech-time denominator

Word error says nothing about attribution. A transcript can be word-perfect and still attribute the
customer's sentence to the agent, which inverts the meaning of the report.

```text
FORMULA [CC-02]: DER = (FA + MISS + CONF) / T_speech
WHERE: FA = falsely-alarmed speech time (seconds marked speech where reference is non-speech);
       MISS = missed speech time (reference speech marked non-speech);
       CONF = speaker-confusion time (speech correctly detected, wrong speaker);
       T_speech = total reference speech time in seconds.
CONSTRAINT: all four measured on one frozen reference segmentation; overlap regions are
       handled by one declared policy (scored once, or excluded from all three terms);
       T_speech > 0.
NUMERICAL EXAMPLE: 600 calls × 384 s = 230,400 s of audio; at a 0.72 speech fraction,
       T_speech = 165,888 s. FA = 878 s, MISS = 1,372 s, CONF = 551 s, total 2,801 s.
       DER = 2,801 / 165,888 = 0.016885 = 1.6885%.
       Component shares: FA 31.35%, MISS 48.98%, CONF 19.67% (878/2,801, 1,372/2,801,
       551/2,801), which sum to 100.00%.
INTERPRETATION: Nearly half the diarisation error is missed speech rather than wrong
       speakers, so tuning the clustering model would address only the 19.67% CONF share.
```

**The turn-level view, which behaves differently.** A second measure is needed because reporters
think in turns, not seconds: 84 of 2,880 scored turns carry a wrong role label, i.e. 2.917% turn
error, but those turns hold 1,845 s, which is 1.112% of speech time. Turn error is 2.6 times the
time-weighted error because misattributed turns are short — interruptions and backchannels. This
inversion is the single most useful diagnostic in the case: a team that reads only turn error will
believe attribution is three times worse than the audio suggests, and a team that reads only
seconds will under-weight the fact that customers interrupt.

**Why missed speech dominates.** Three mechanisms, distinguishable by where the misses sit:

| Mechanism | Signature in the miss distribution | Discriminating check |
|---|---|---|
| Segmenter threshold too conservative | Misses concentrated at turn onsets and offsets | Compare miss time inside 300 ms of a reference boundary versus the turn interior |
| Low-level customer speech | Misses concentrated on one channel, correlated with measured level | Histogram miss time by channel RMS |
| Music and announcement exclusion | Misses clustered on hold intervals | Join misses to hold-marker timestamps |

The second is the one that matters for objections, because a customer speaking quietly and
hesitantly — the acoustic profile of someone raising a complaint — is precisely the profile a
conservative segmenter drops. If the miss time is concentrated on the customer channel at low RMS,
the fix is not a better clusterer but a channel-aware threshold, and no amount of speaker-model
work will recover it.

**Boundary where CC-02 breaks.** DER assumes the reference segmentation is the truth and that
overlap is scored by one declared rule. It fails in three ways worth naming. First, with heavy
overlap, an "excluded from all three terms" policy reports a low DER on a conversation that is
mostly unintelligible — the denominator silently drops the hard part. Second, DER is unbounded
above and can exceed 100% when a segmenter hallucinates speech on music, so a clipped report at
100% loses the signal that the segmenter is broken. Third, and most important here: a low DER does
not imply correct *role* labels. A diariser can be internally consistent (two clusters, stable
within a call) and consistently wrong about which cluster is the customer, because cluster-to-role
mapping is a separate inference. That is why §8 maps roles from telephony metadata first and treats
the diariser's clusters as a consistency check, not as an identity source.
### 6.3 Derivation CC-03 — Span precision, recall, and the inflation factor

The metric that actually governs publication is not word error but span-level correctness: did the
system mark the right interval as an objection, and did it mark any interval that is not one?

```text
FORMULA [CC-03]: P = TP / (TP + FP);  R = TP / (TP + FN);  F1 = 2TP / (2TP + FP + FN)
WHERE: TP spans match a gold span by the declared match rule; FP spans have no gold
       counterpart; FN gold spans have no model counterpart. Match rule: temporal
       intersection-over-union >= 0.5 AND the same rubric category.
CONSTRAINT: one-to-one matching (a model span may satisfy at most one gold span);
       spans overlapping a routed call are excluded from both numerator and denominator
       so that routing does not flatter precision.
NUMERICAL EXAMPLE: over 600 calls, gold holds 412 objection spans; the pipeline proposes
       468. Matching gives TP = 287, FP = 181, FN = 125. Check: TP+FP = 468 (proposed),
       TP+FN = 412 (gold). P = 287/468 = 0.6132; R = 287/412 = 0.6966;
       F1 = 2(287)/(2(287)+181+125) = 574/880 = 0.6523.
INTERPRETATION: A proposed-span count of 468 overstates gold by 468/412 − 1 = 13.59%,
       while at the same time missing three gold spans in ten. The error is not a
       scaling problem; it is a two-sided problem.
```

The count is 13.59% too high and simultaneously 30.34% incomplete on the gold side — two biases in
opposite directions that a total-count comparison partially cancels. This is the most dangerous
configuration in the whole case: a dashboard that plots "objections per day" from the raw proposed
count will look *approximately right* while being composed of the wrong spans. The cancellation is
coincidental, and it evaporates the moment the rubric changes the phrase list.

**Per-100-calls view.** Gold density is 412/600 = 0.687 spans per call; proposed density is
468/600 = 0.780 spans per call. Over a 900-call day that is 618.3 gold versus 702.0 proposed — a
gap of 83.7 spans per day, composed of roughly 27 missing spans and 111 spurious ones if the daily
mix matches the sample. The two errors do not net out in the category breakdown, which is what the
report is actually for.

**Where the errors concentrate.**

| Error class | Count | Share of FP or FN | Characteristic |
|---|---|---|---|
| Agent script phrase matched as customer objection | 64 | 35.4% of FP | Script line reads "I understand your concern about the charge" |
| Negation missed | 41 | 22.7% of FP | "I don't have a problem with that" matched on the phrase "problem with" |
| Fragment split into two spans | 38 | 21.0% of FP | One objection interrupted by an agent backchannel becomes two |
| Weak or partial speech | 29 | 16.0% of FP | Boundary placed inside the objection, IoU falls below 0.5 |
| Boundary truncation | 9 | 4.9% of FP | Span starts after the objection's first clause |
| Objection inside a routed call | 96 | 76.8% of FN | Counted as routed, so excluded from the TP set by the matching rule |
| Objection phrased outside the rubric | 22 | 17.6% of FN | Real objection, no rubric phrase, no candidate generated |
| Objection split across a transfer | 7 | 5.6% of FN | Conversation continues on a new leg with a new call leg id |

This table is the design input: 35.4% of false positives come from ignoring speaker role, which is
a gate problem, not a model problem. Fixing role attribution removes the largest single FP class
for the cost of a metadata join.

**Boundary where CC-03 breaks.** The IoU ≥ 0.5 rule is a convention, not a property. It fails when
gold spans are much longer than model spans: a 6-second gold objection matched by a correct 2-second
core has IoU = 1/3 and scores as both a FP and a FN, double-penalising one honest detection. It
also fails for nested or overlapping objections — a customer who raises a charge objection and then
a service objection in one breath produces overlapping gold spans that a one-to-one rule cannot
represent. The check is to report the span-length distribution alongside P and R; if model spans
are systematically shorter than gold, IoU is measuring boundary policy rather than detection, and
boundary-tolerant matching (token-level overlap on the objection's head phrase) must be used
instead, with both numbers reported.

### 6.4 Derivation CC-04 — Calibration, and why an untested confidence cut is a guess

A confidence score is only usable for routing if it is calibrated: among spans the model scores at
0.94, about 94% should actually be correct. Uncalibrated confidence produces a threshold that looks
principled and behaves arbitrarily.

```text
FORMULA [CC-04]: ECE = Σ_b (n_b / n) · | mean(conf_b) − acc_b |
WHERE: b indexes confidence bins; n_b spans in bin b; n = Σ n_b;
       mean(conf_b) mean predicted probability in bin b; acc_b observed accuracy in bin b.
CONSTRAINT: bins declared before inspection; each bin needs enough spans for acc_b to be
       stable; accuracy measured on held-out data, not the data used to fit the calibrator.
NUMERICAL EXAMPLE: 1,200 candidate spans in ten bins. Predicted minus observed by bin:
       +0.031, +0.018, +0.014, +0.002, +0.003, −0.001, +0.075, +0.065, +0.040, +0.003.
       Weighted by bin share the absolute gaps give ECE = 0.0267 = 2.67 percentage points.
INTERPRETATION: Small overall, but the error is concentrated: the three highest-confidence
       bins over-predict by 4.0 to 7.5 points, exactly where a high threshold is applied.
```

**The bin table, closed and recomputed (ILLUSTRATIVE).**

| Bin | Spans `n_b` | Mean predicted | Observed accuracy | Gap (pred − obs) |
|---|---|---|---|---|
| 0.52–0.56 | 36 | 0.556 | 0.525 | +0.031 |
| 0.56–0.60 | 54 | 0.593 | 0.575 | +0.018 |
| 0.60–0.64 | 72 | 0.639 | 0.625 | +0.014 |
| 0.64–0.68 | 96 | 0.677 | 0.675 | +0.002 |
| 0.68–0.74 | 114 | 0.728 | 0.725 | +0.003 |
| 0.74–0.80 | 84 | 0.774 | 0.775 | −0.001 |
| 0.80–0.90 | 138 | 0.900 | 0.825 | +0.075 |
| 0.90–0.94 | 156 | 0.940 | 0.875 | +0.065 |
| 0.94–0.97 | 174 | 0.965 | 0.925 | +0.040 |
| 0.97–1.00 | 276 | 0.978 | 0.975 | +0.003 |
| **Total** | **1,200** | **0.8498** | **0.8233** | **+0.0265** |

Bin counts sum to 1,200 as required. Overall predicted is 1,019.8/1,200 = 0.8498 and overall
observed is 987.9/1,200 = 0.8233, so the aggregate over-prediction is 2.65 points, consistent with
the ECE of 2.67 computed from absolute gaps. The apparent inconsistency in the middle of the table —
a bin at 0.90 predicted scoring only 0.825 — is the whole point: the model's confidence saturates
after about 0.90 and stops carrying information.

**Threshold selection, computed rather than asserted.** Three candidate cuts:

| Cut on `p` | Spans accepted | Coverage | Empirical precision | Spans routed |
|---|---|---|---|---|
| ≥ 0.85 | 744 | 62.0% | 680.5/744 = 91.45% | 456 |
| ≥ 0.94 | 606 | 50.5% | 566.6/606 = 93.49% | 594 |
| ≥ 0.96 | 450 | 37.5% | 430.1/450 = 95.57% | 750 |

Moving the cut from 0.85 to 0.96 buys 4.1 points of precision and costs 24.5 points of coverage —
172 spans per 1,200, which at 702 candidates per day is 100.6 spans, or 180.3 staffed minutes of
extra review per day. That trade is a staffing decision disguised as a modelling decision, which is
why §6.6 computes it in minutes rather than in accuracy points.

**The asymmetry that decides the cut.** A false positive enters a published count as a fact and is
never revisited. A false negative is a span that a reviewer never sees, and its absence is invisible
in the data. Both are bad, but only the false positive is *published*; the negative merely
undercounts. That asymmetry argues for the higher cut, and the capacity model decides how high it
can go.

**Boundary where CC-04 breaks.** ECE is a marginal, bin-averaged quantity and says nothing about
conditional calibration. A model can have ECE of 2.7 points overall while being badly
over-confident on one channel — say legacy headsets, where the acoustic evidence is weakest — and
that is the stratum where the threshold is applied most often. The check is to compute ECE per
channel, per language, and per queue. A second boundary: the observed accuracy in each bin is
itself estimated from a finite number of spans, so a bin with 36 spans has a standard error of
about `sqrt(0.525×0.475/36) = 0.083` — 8.3 points — larger than the gap it is used to measure. Bins
below roughly 50 spans cannot support a calibration claim, and the two lowest bins in this table are
therefore reported to show the shape, not to justify a decision.

### 6.5 Derivation CC-05 — Staffing: the daily average says feasible, the peak hour says no

The naive capacity calculation multiplies arrivals by service time and compares with staffed
minutes. It is correct arithmetic applied to the wrong moment.

```text
FORMULA [CC-05, daily average]: W = A · r · s ;  U = W / (k · h · 60)
WHERE: A eligible candidate spans per day (spans, not calls — only candidates are routed);
       r routed share of candidates (dimensionless); s effective staffed minutes per
       routed span (minutes); k reviewers; h productive staffed hours per reviewer per day.
CONSTRAINT: stationary arrivals and service times; no rework, breaks, or escalations;
       excludes queueing delay entirely — this is a load test, not a service test.
NUMERICAL EXAMPLE: A = 900 calls × 0.78 candidates/call = 702 candidates/day. At
       r = 0.38, 266.8 spans route. Nominal handle time 1.6 min, rework factor 1.12
       (12% of dispositions reopened), so s = 1.6 × 1.12 = 1.792 min.
       W = 702 × 0.38 × 1.792 = 478.0 staffed minutes. With k = 2 reviewers at
       h = 7.0 productive hours, capacity = 840 min, so U = 478.0/840 = 56.91%.
INTERPRETATION: 57% average utilisation reads as comfortable headroom. It is not a
       service guarantee, because a queue's delay is governed by the peak hour.
```

The rework factor deserves its own justification because it is the most commonly omitted term. A
reviewer who opens a span, cannot resolve it, and reopens it after consulting the rubric spends
the time twice. If 12% of dispositions are reopened at full cost, effective service time is 12%
above nominal. Omit it and the plan understates load by 10.7% (`1.792/1.6 = 1.12`), which is
invisible on a 57% utilisation and fatal on a 90% one.

**Now the same day at the peak hour.** Arrivals are not uniform. In this illustration 900 legs
over an 8-hour service window is a mean of 112.5 legs/hour, and the busiest hour runs 1.32 times
the mean — 148.5 legs/hour — a modest peaking factor that any contact centre will recognise as
conservative.

```text
FORMULA [CC-05b]: Erlang-C delay probability for c reviewers and offered load a,
       a = λ · s / 60  (λ routed spans per hour, s minutes per span, so a is dimensionless)
       C(c, a) = [ a^c / (c!(1 − a/c)) ] / [ Σ_{k=0}^{c−1} a^k/k! + a^c/(c!(1 − a/c)) ]
       W_q = C(c, a) / (c·μ − λ),  μ = 1/s (spans per minute per reviewer)
       P(wait > t) = C(c, a) · exp(−(cμ − λ) t)
CONSTRAINT: Poisson arrivals, exponential service, no abandonment, no priority classes,
       no batching. All four are violated in reality to some degree; results are
       directional, not contractual.
NUMERICAL EXAMPLE: at the peak, candidates = 148.5 × 0.78 = 115.83 spans/hour. At
       r = 0.38, λ = 44.02 spans/hour = 0.7336/min. With s = 1.792 min, μ = 0.5580/min,
       a = 0.7336/0.5580 = 1.3146, so a/c = 0.6573 with c = 2 and C = 0.5214.
       W_q = 0.5214 / (2(0.5580) − 0.7336) = 0.5214 / 0.3824 = 1.363 min.
       P(wait > 10 min) = 0.5214 · exp(−0.3824 × 10) = 0.5214 × 0.02196 = 1.14%.
INTERPRETATION: The same policy that shows 57% daily utilisation puts 52% of peak-hour
       spans in a queue, with a 1.36-minute mean wait and about 1.1% waiting over ten
       minutes — roughly 0.5 spans in the peak hour.
```

**The sensitivity table that decides the policy.** Recomputed at each routing share, two reviewers:

| Routed share `r` | Offered load `a` | Utilisation `a/c` | Delay prob. `C` | Mean wait | P(wait > 10 min) |
|---|---|---|---|---|---|
| 0.20 | 0.692 | 0.346 | 0.178 | 0.24 min | 0.01% |
| 0.30 | 1.038 | 0.519 | 0.355 | 0.66 min | 0.17% |
| 0.35 | 1.211 | 0.605 | 0.457 | 1.04 min | 0.56% |
| 0.38 | 1.315 | 0.657 | 0.521 | 1.36 min | 1.14% |
| 0.40 | 1.384 | 0.692 | 0.566 | 1.65 min | 1.82% |
| 0.45 | 1.557 | 0.778 | 0.681 | 2.76 min | 5.74% |
| 0.50 | 1.730 | 0.865 | 0.802 | 5.32 min | 17.75% |
| 0.55 | 1.903 | 0.951 | 0.928 | 17.09 min | 53.90% |

The shape is the whole lesson: from `r = 0.30` to `r = 0.45` the routed volume rises by half, but
the ten-minute tail rises by a factor of 34 (0.17% to 5.74%). Beyond `r = 0.50` the queue is not
slow, it is unstable in practice, because the exponential-service assumption is the optimistic case
and real handle times are heavier-tailed. At `r = 0.60` the offered load reaches 2.076 with two
reviewers — strictly infeasible, since `a/c > 1` means arrivals exceed service capacity and the
queue grows without bound.

**Three reviewers change the picture, at a price.** At `r = 0.60` with `c = 3`: `a = 2.076`,
`a/c = 0.692`, `C = 0.4805`, `W_q = 0.932 min`, `P(wait > 10) = 0.28%`. So the third reviewer
supports a 60% routing share with a better tail than two reviewers deliver at 38%. The comparison
that matters is not "2 versus 3 reviewers" but "38% routing with 2" versus "60% routing with 3":
the third reviewer buys 22 points of additional coverage — 154 more spans per day at 702
candidates — for 420 additional staffed minutes per day.

**Where the daily-average model actually fails.** It fails by saying yes. At `r = 0.55` the
daily average gives `W = 702 × 0.55 × 1.792 = 691.9` minutes against 840 capacity, i.e. 82.37%
utilisation — apparently fine — while the peak-hour tail is 53.9%. A staffing plan built on the
daily figure would approve a policy under which half the peak-hour spans wait more than ten
minutes; since the review queue is not the customer-facing queue, nobody notices until the
backlog appears three days later.

**Boundary where CC-05 breaks.** Erlang-C assumes Poisson arrivals and exponential service. Three
violations matter here and each pushes the true delay higher than computed. First, arrivals are not
Poisson at the peak: calls cluster after an outage or a campaign drop, producing bursts the model
smooths away. Second, handle time is not exponential but right-skewed — a long, contested call can
consume twenty minutes — and the model's mean wait is optimistic when the tail is heavy. Third, no
abandonment: a real reviewer who finds a twenty-minute queue stops working the queue and does
something else, which removes capacity exactly when it is needed. The practical consequence is to
treat the computed tail as a *lower bound* and to add a hard cap on routed volume per hour rather
than relying on the model's probability.

### 6.6 Derivation CC-06 — Publishing an aggregate with an audit-estimated numerator

The aggregate has two components: spans a reviewer confirmed, and spans the model accepted that no
reviewer saw. The second component is estimated from a sample, so the published figure carries
sampling error that the first component does not.

```text
FORMULA [CC-06]: N_hat = N_confirmed + N_auto · p_hat
                 with p_hat from an audit of n accepted spans: p_hat = x/n,
                 Wilson interval [ (p̂ + z²/2n ± z√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n) ]
CONSTRAINT: the audit sample is drawn from the accepted frame with known inclusion
       probabilities; accepted spans are exchangeable with unaudited accepted spans;
       the confirmed count is treated as exact (it was reviewed, not sampled).
NUMERICAL EXAMPLE: 702 candidates/day; 266.8 routed and reviewed, of which 178 confirm
       (66.67% of routed); 435.2 accepted without review. An audit of n = 150 accepted
       spans finds x = 141 correct, p̂ = 0.94, Wilson 95% interval [0.8899, 0.9681].
       N_hat = 178 + 435.2 × 0.94 = 178 + 409.1 = 587.1 spans/day.
       Interval: 178 + 435.2×0.8899 = 565.3 to 178 + 435.2×0.9681 = 599.3.
       Per 900 calls: 587.1/900 = 65.23 per 100 calls, interval 62.81 to 66.59.
INTERPRETATION: The interval is ±1.9 per 100 calls, which is tight enough to compare
       months. The dominant error is not sampling; it is the 6% accepted-error rate
       applied to 62% of the volume.
```

Recomputing the Wilson interval by hand, so the numbers are checkable:
`z = 1.96`, `z² = 3.8416`, `p̂ = 0.94`, `n = 150`. Centre term
`(0.94 + 3.8416/300)/(1 + 3.8416/150) = (0.94 + 0.012805)/(1.025611) = 0.952805/1.025611 = 0.92902`.
Half-width term `1.96 × sqrt(0.94×0.06/150 + 3.8416/90000) = 1.96 × sqrt(0.000376 + 0.00004268) =
1.96 × sqrt(0.00041868) = 1.96 × 0.020462 = 0.040105`; divided by `1.025611` gives `0.039103`. So
the interval is `0.92902 ± 0.03910 = [0.88992, 0.96812]`, matching the reported `[0.8899, 0.9681]`.

**Why not the normal approximation.** At `p̂ = 0.94` and `n = 150` the textbook interval
`0.94 ± 1.96·sqrt(0.94·0.06/150) = 0.94 ± 0.03803` gives `[0.9020, 0.9780]`, whose upper end
exceeds the plausible range and whose lower end is 1.2 points optimistic versus Wilson. The
asymmetry is real: with 9 failures in 150, the tail above 0.97 is thinner than the tail below 0.90,
and the Wilson form captures that. The practical effect on the published figure is ±1.2 per 100
calls, which is most of the month-over-month signal the report is meant to carry.

**How large should the audit be?** For a ±3-point margin at 95% confidence on `p ≈ 0.94`:
`n₀ = z²p(1−p)/e² = 3.8416 × 0.94 × 0.06 / 0.0009 = 0.216666/0.0009 = 240.7`. With a finite
population correction against an accepted frame of 435.2 spans/day,
`n = n₀ / (1 + (n₀−1)/N) = 240.7/(1 + 239.7/435.2) = 240.7/1.5508 = 155.2`. So 156 spans per day —
34.5% of the accepted frame — buys ±3 points, and the illustration's 150 is within one span of that
requirement, giving an actual margin of 3.08 points with the correction
(`1.96 × sqrt(0.94×0.06/150) × sqrt((435.2−150)/(435.2−1)) = 1.96 × 0.019391 × 0.8105 = 0.0308`).

**The denominator trap, quantified.** 837 of 900 calls are processed (93.0%), the rest excluded for
consent, capture, language, or integrity reasons. Reporting per *processed* call gives
`587.1/837 = 70.14` per 100 calls rather than `587.1/900 = 65.23` — a 7.53% inflation, produced
entirely by the choice of denominator. Neither number is wrong; publishing one while describing the
other is. The rule is that the denominator must be named in the same sentence as the rate, and the
exclusion count must be visible in the same table.

**Break-even on accepting a span unreviewed.** A reviewer costs `1.792 min × £46/hour / 60 = £1.3739`
per span reviewed. Accepting a span unreviewed saves that cost and admits a 6% error rate. The
break-even cost of one published error is therefore `£1.3739 / 0.06 = £22.90`. If a spurious
objection in the published count costs the organisation less than £22.90 in remediation — a
re-check, a correction, an apology for a miscategorised trend — then accepting at `p = 0.94` is
economically correct. If it costs more, the threshold must move. That is a finance question the
design should force into the open rather than decide implicitly through an arbitrary 0.90 cut.

**Boundary where CC-06 breaks.** The estimator is unbiased only if the audited accepted spans are
exchangeable with the unaudited ones. Three realistic violations: (a) the audit is drawn from the
same day it reports, so late-arriving accepted spans are never auditable — a stale-audit bias;
(b) the audit is drawn after a model update, so `p̂` describes a different release than the spans it
is applied to — a version-mixing bias; (c) reviewers of routed spans have already seen neighbouring
spans from the same call, so the confirmed count is not independent of the accepted count when
multiple spans come from one call — a clustering effect that shrinks the effective sample size. The
check for (c) is to compute the design effect from the intra-call correlation of correctness; if
spans cluster by call, `n = 150` spans may carry the information of far fewer independent
observations.
## 7. The Finding

**The finding, stated once and then supported.**

A monthly objection aggregate from this pipeline is publishable only as a *triangulated* figure: a
reviewer-confirmed count plus a sampled estimate of the unreviewed remainder, under a named
denominator, with the exclusion count visible. Every simpler construction is wrong in a specific,
identifiable way.

Seven findings, in order of how much they change the design:

1. **Headline word error is the wrong yardstick and the right warning sign.** 7.31% overall against
   15.38% on entity-like tokens means the pipeline is adequate for readability and inadequate for
   the tokens that carry objections. The design must be evaluated on span precision and recall
   (61.32% and 69.66%), not on WER.

2. **Two-sided span error partially cancels in the total and does not cancel anywhere else.**
   Proposed 468 spans against 412 gold — 13.59% high — while missing 30.34% of gold. Any total-count
   comparison will therefore look acceptable while the category breakdown is composed of the wrong
   spans. Publication must report precision from an audit sample, not just a total.

3. **Attribution, not acoustics, is the largest controllable error source.** 35.4% of false-positive
   spans are agent script lines matched as customer objections, and 48.98% of diarisation error is
   missed speech rather than speaker confusion. Both are fixed by gates and channel-aware
   segmentation, not by a better acoustic model.

4. **Confidence is calibrated in aggregate and mis-calibrated where it is used.** ECE of 2.67
   points overall hides a 7.5-point over-prediction in the 0.80–0.90 bin. A threshold is only
   defensible after per-stratum calibration, because the stratum with the weakest evidence is also
   the one that reaches the threshold most often.

5. **The daily capacity calculation is not a feasibility test.** At a 38% routing share the day
   shows 56.9% utilisation while the peak hour shows 52.1% of spans queued. Feasibility must be
   assessed at the peak with a service-level tail, and the feasible share under a 5%-over-ten-minutes
   guardrail is 44.4% — not the 66.8% the daily average would permit.

6. **The audit is the publication's load-bearing element.** 62% of the published numerator is
   unreviewed, so the published figure's accuracy is set by a 150-span sample. At ±3.08 points on
   `p̂ = 0.94`, the interval on the aggregate is ±1.9 per 100 calls — comparable to a typical
   month-over-month movement, which means short-horizon comparisons are mostly sampling noise.

7. **Denominator choice moves the headline more than a model upgrade.** Reporting per processed call
   rather than per in-scope call inflates the rate by 7.53% with no change to any span. That is a
   larger effect than the difference between the 0.85 and 0.96 confidence cuts on the point
   estimate, and it is decided by a sentence in a report template.

**What is explicitly not found.** No claim is made about:
- Whether any real ASR system reaches these figures (they are constructed).
- Whether the objection categories correspond to any real taxonomy.
- Whether the 6% accepted-error rate would survive a second audit, a different channel mix, or a
  model upgrade.
- Whether reviewers would actually be available at the peak hour.
- Whether the legal basis described would suffice in any jurisdiction.

The finding is a *design* finding: given the structure of the problem, these are the terms a
defensible report must contain, and these are the quantities that must be measured before any claim
is published.

## 8. System/Solution Design

### 8.1 State machine — the core of the design

Every span occupies exactly one state, transitions are recorded with actor and timestamp, and only
two states may enter a published count.

```text
CANDIDATE ──gate_fail──────────────────────────────► BLOCKED
    │                                                   ▲
    ├──route (calibration or capacity)──► ROUTED         │
    │                                       │           │
    │                                       ├─confirm──► REVIEWED ──► AGGREGATED
    │                                       └─reject───► REJECTED
    └──accept (calibration + audit)──────► ACCEPTED ──audit_fail──► ROUTED
```

Four properties are non-negotiable and each is enforced in code, not in a runbook:

- **No transition into `AGGREGATED` that bypasses `REVIEWED` or `ACCEPTED`.** A span cannot be
  aggregated directly from `CANDIDATE`.
- **`BLOCKED` is terminal for the report, not for the record.** A blocked span keeps its audio
  reference, its reason code, and its timestamps; it can be re-examined, but it can never be
  silently resurrected into a later month's count.
- **`ACCEPTED → ROUTED` is the only audit-driven demotion.** When an audit finds an accepted span
  wrong, that span is routed; the audit also corrects the estimate `p̂` for the whole cohort, which
  is the more important effect.
- **Version is a property of the state, not of the record.** A span reviewed under rubric v3 and
  aggregated under v4 is invalid; the aggregation service refuses the transition.

### 8.2 Component contracts

| Layer | Responsibility | Explicit refusal |
|---|---|---|
| Admissibility gate | Resolve identity, lawful basis, integrity, capture completeness | Refuses to emit a transcript for a call that failed any check |
| Canonical reader | Preserve sample rate, channels, gaps, hash | Refuses to resample or to repair a gap |
| Segmenter | Speech intervals with uncertainty, hold/music labelled | Refuses to label music as speech |
| ASR | Tokens with timing and revision-stamped confidence | Refuses to emit text without timing |
| Diariser | Speaker segments with posteriors and `unknown` | Refuses a forced two-speaker output |
| Role mapper | Metadata-first role assignment, diariser as consistency check | Refuses to assign "customer" from acoustics alone |
| Rubric matcher | Candidate spans with quoted matching tokens | Refuses to emit a category without a quoted span |
| Calibrator | Calibrated probability per span, per stratum | Refuses to output a probability from an unfitted stratum |
| Router | Threshold plus capacity cap, with reason code | Refuses to queue beyond the cap; emits `CAPACITY_DEFER` |
| Evidence service | Replay from original bytes at recorded offsets | Refuses to serve a re-encoded surrogate |
| Aggregator | Accepts only `REVIEWED`/`ACCEPTED`; pinned rubric and denominator | Refuses mixed-version cohorts |
| Publisher | Manifest, interval, exclusions, suppression count | Refuses to publish a cohort with an incomplete lineage chain |

### 8.3 Role attribution, done in the right order

Role attribution is where most pipelines quietly fail, because the intuitive order is backwards.

**Wrong order:** diarise → assume cluster 1 is the customer → attach roles.

**Right order:** use telephony metadata to establish which channel carried the agent, fall back to
the diariser only to check consistency, and emit `unknown` when neither is decisive.

The reason is that cluster numbering is arbitrary and unstable across calls. A diariser that is
perfectly consistent within each call will label the customer as speaker 0 in one call and speaker 1
in the next, and any global mapping is a coin flip. Metadata — which extension, which trunk, which
desktop — is decisive where it exists and absent exactly where transfers happen.

Three rules follow, and they are cheap:

| Condition | Role assignment | Rationale |
|---|---|---|
| Dual-channel plus agent extension known | Deterministic from channel | No inference needed; error only if the bridge crossed channels |
| Mixed channel, agent greeting matched with high confidence | Agent for the greeting turns, customer for the rest of that turn's addressee pattern | Greeting is the most predictable utterance in the call |
| Mixed channel, neither decisive | `unknown`, route the span | A wrong role inverts the report's meaning |

The third rule is the one that gets argued about, because routing every `unknown` looks expensive.
The arithmetic that settles it: 35.4% of false positives are script lines attributed to the
customer. Removing them requires no model improvement, and the review cost of the `unknown` spans is
bounded by how often metadata fails, which is a measurable quantity rather than an opinion.

### 8.4 Recovery and idempotency

The pipeline is re-run constantly — after a rubric change, a parser fix, a model rollback. Three
mechanisms prevent re-runs from corrupting history.

- **Idempotency key:** `recording_sha256 + leg_id + config_bundle_version`. A retry with the same
  key is a no-op, never a second count. The key deliberately excludes wall-clock time, so a retry
  after a six-hour outage still maps to the original record.
- **Immutable span identity:** spans are identified by `(call_id, leg_id, hash, start_ms, end_ms)`
  under a specific bundle version. A re-segmentation that shifts an offset by 40 ms produces a new
  span identity, which is correct: it is different evidence, even if it looks similar.
- **Atomic bundle rollback:** the audio parser, segmenter, ASR, diariser, rubric, and aggregation
  mapping roll back together. Partial rollback is prohibited because it produces cohorts that never
  existed — for example, new spans scored under an old rubric, which no annotation process can
  validate after the fact.

### 8.5 Privacy architecture — minimisation as a structural property

The design constraint "no per-agent output" is enforced structurally rather than by policy:

- **No agent identifier in the span table.** The `agent_id` column does not exist. Role is `agent` or
  `customer` or `unknown`; identity is not carried. A downstream analyst cannot join to a person
  because there is nothing to join on.
- **Cell-size floor at publication.** Any category × month cell below 20 spans is suppressed and the
  suppression count is published. Suppression without the count invites the reader to assume the
  cell was empty.
- **Redaction at replay, not at storage.** The audio is retained per the retention schedule; the
  replay service masks declared sensitive intervals (card numbers spoken aloud, third-party
  identification) at playback. Redacting stored audio destroys evidence; redacting at replay
  preserves it while limiting exposure.
- **Purpose binding.** The consent record is joined at admissibility and carried through to the
  manifest, so a later claim that the data was used only for an authorised purpose is checkable
  rather than asserted.

### 8.6 What the design deliberately does not do

Four non-capabilities, each of which someone will request:

1. **No real-time scoring.** The pipeline is batch. Live assist is a different product with
   different latency, different failure tolerance, and different consent exposure.
2. **No speaker identification across calls.** Building a voiceprint to track a caller across
   interactions is a different lawful basis and a different risk class.
3. **No emotion classification as an objection label.** Prosody can prioritise review; it cannot
   establish that an objection was stated.
4. **No automated remediation trigger.** The aggregate informs a decision by a named owner; it does
   not open cases, refund customers, or dispatch coaching. Automation of consequence requires a
   different evidence standard than automation of information.

## 9. Evaluation & Validation

### 9.1 Metric contract

Nine metrics, each with its numerator, denominator, population, window, comparator, and status. A
metric without a declared denominator is not a metric; it is an anecdote with units.

| Metric | Numerator | Denominator | Population | Window | Comparator | Status |
|---|---|---|---|---|---|---|
| WER (pooled) | S+D+I = 10,733 | 146,880 reference tokens | 300 sampled calls, all channels | One frozen corpus | Prior release on the same corpus | Computed (illustrative) |
| Entity WER | 1,412 entity errors | 9,180 entity tokens | Same 300 calls | Same corpus | Pooled WER as reference point | Computed (illustrative) |
| Span precision | 287 matched | 468 proposed | 600 calls, gate-passing spans | Same corpus | Manual baseline precision on the same calls | Computed (illustrative) |
| Span recall | 287 matched | 412 gold | 600 calls | Same corpus | Manual baseline recall | Computed (illustrative) |
| DER | 2,801 s error | 165,888 s speech | 600 calls | Same corpus | Prior segmenter revision | Computed (illustrative) |
| Turn role accuracy | 2,796 correct | 2,880 turns | 600 calls, dual + mixed | Same corpus | Metadata-only attribution | Computed (illustrative) |
| Calibration (ECE) | 0.0267 weighted gap | 1,200 candidate spans | All gate-passing candidates | Same corpus | Target ≤ 0.02 | Computed (illustrative) |
| Published precision | 141 correct | 150 audited | Accepted-frame sample | One reporting day | Target ≥ 0.93 | Computed (illustrative) |
| Peak queue tail | 1.14% over 10 min | Peak-hour routed spans | 38% routing policy | Busiest hour | Guardrail ≤ 5% | Computed (illustrative) |

### 9.2 Guardrails

Guardrails are *stopping conditions*, not dashboards. Each is checked before publication and each
has a defined action on breach.

| Guardrail | Threshold | Action on breach |
|---|---|---|
| Published precision | ≥ 0.93 | Block publication; route the whole cohort until re-audited |
| Replay success on audited spans | ≥ 0.99 | Block; the evidence chain is broken |
| Excluded-leg share | ≤ 10% | Publish with an explicit coverage warning; investigate the exclusion reason mix |
| Peak queue tail over 10 min | ≤ 5% | Reduce the routing share for the next window; never silently extend the queue |
| Mixed-version cohorts | 0 | Refuse the aggregation; split by version |
| Cells below the size floor | Suppressed and counted | Publish the suppression count alongside every table |
| Reviewer agreement (κ) on a 120-span re-review | ≥ 0.60 | Recalibrate the rubric before trusting the confirmation rate |

The agreement guardrail deserves explanation because it is the least obvious. If two reviewers
disagree about whether a span is an objection, the "confirmed" count is not a measurement of the
pipeline — it is a measurement of reviewer taste. In this illustration, a 120-span blind re-review
gives both-agree-positive 71, both-agree-negative 30, first-only 8, second-only 11. Observed
agreement `p_o = (71+30)/120 = 0.8417`; chance agreement `p_e = (79/120)(82/120) + (41/120)(38/120)
= 0.4498 + 0.1082 = 0.5581`; `κ = (0.8417 − 0.5581)/(1 − 0.5581) = 0.2836/0.4419 = 0.642`. That is
substantial but far from perfect, and it caps the meaningful precision of the confirmation step:
with κ = 0.64, a "confirmed" span carries roughly a 5–8% chance that a second competent reviewer
would disagree. The 6% accepted-error rate and the reviewer disagreement rate are therefore of the
same order, which is a useful sanity check on the audit design.

### 9.3 Population and slice discipline

Aggregate metrics hide the strata that matter. Every metric above is computed per slice before it
is pooled, and the slice list is fixed in advance:

| Slice | Why it is required | Risk if pooled away |
|---|---|---|
| Channel width (narrowband / wideband) | Wideband carries more acoustic evidence | Narrowband failures blend into a wideband majority |
| Capture path (trunk / station) | Station-side attenuates the customer | Customer speech loss looks like model weakness |
| Queue / campaign | Consent coverage and language mix differ | Coverage bias appears as quality difference |
| Language | Supported set differs by queue | Unsupported tail drags the average |
| Rubric category | Phrase length and ambiguity differ | One bad category disappears into nine good ones |
| Report month | Version and mix change over time | Trend is a version artefact |

Illustrative slice table for WER (weights and rates are constructed; they multiply back to the
pooled 7.31%):

| Slice | Share of tokens | Slice WER | Contribution to pooled WER |
|---|---|---|---|
| Narrowband clean (ISDN) | 0.42 | 5.55% | 2.331 |
| Narrowband mobile | 0.28 | 7.75% | 2.170 |
| Wideband softphone | 0.19 | 8.50% | 1.615 |
| Legacy amplified headset | 0.11 | 10.90% | 1.199 |
| **Total** | **1.00** | — | **7.315** |

Check: `0.42×5.55 + 0.28×7.75 + 0.19×8.50 + 0.11×10.90 = 2.331 + 2.170 + 1.615 + 1.199 = 7.315`,
which is the pooled `7.31%` to two decimals. The last two slices are only 30% of tokens but
contribute 38.5% of the weighted error; a change that improves wideband by one point moves the
headline by only 0.19.

### 9.4 Uncertainty

Three uncertainty sources are reported separately because they behave differently:

1. **Sampling uncertainty** in the audit estimate: ±3.08 points on `p̂`, propagating to ±1.9 per 100
   calls. Shrinks with more audit spans; computable in advance.
2. **Reviewer disagreement:** κ = 0.642, contributing an unquantified but non-trivial band around the
   confirmation rate. Does not shrink with more audit; shrinks only with rubric work.
3. **Model and version uncertainty:** the entire measurement is conditional on one bundle. A model
   upgrade invalidates the comparison rather than widening an interval, which is a different kind of
   uncertainty — it is a discontinuity, not a variance.

The published interval covers only the first. That is stated in the manifest, because a reader who
assumes the interval covers all three will over-trust a month-over-month difference.

### 9.5 Executed versus not executed

| Activity | Status | Note |
|---|---|---|
| Arithmetic in §6 | Executed | Recomputed; multiplication identities checked |
| WER, DER, span metrics on constructed corpus | Executed as arithmetic, not as measurement | Counts constructed, not observed |
| Calibration and threshold selection | Executed on constructed bins | Real calibration requires held-out data |
| Erlang-C capacity analysis | Executed | Assumptions stated; directional only |
| Audit sample-size calculation | Executed | Includes finite population correction |
| Reviewer agreement computation | Executed on constructed 2×2 | κ arithmetic verified |
| Production deployment | Not executed | No environment, no approvals |
| Independent gold annotation | Not executed | Constructed counts only |
| Legal review of consent basis | Not executed | Out of scope; compliance owner decision |
| Longitudinal validation across months | Not executed | Would require real cohorts |
## 10. Action Taken

**This section describes a proposed decision record. No production action was taken.**

The decision is structured as a staged commitment with explicit reversal conditions, because the
binding constraint is evidence, not engineering effort.

### 10.1 Decision record

| Decision | Choice | Alternatives considered | Reversal condition |
|---|---|---|---|
| D-1 Report shape | Triangulated count: confirmed + audited estimate, under a named denominator | Raw proposed count; manual-only sample | Audit precision falls below 0.93 for two consecutive months |
| D-2 Routing policy | 38% of candidates, capped at the peak-feasible share | Fixed confidence cut with no capacity cap; route-everything | Peak tail exceeds 5% for a full week |
| D-3 Role handling | Metadata-first, diariser as consistency check, `unknown` routed | Diariser-cluster mapping; forcing two speakers | Metadata coverage rises above 98% and diariser accuracy is validated in that stratum |
| D-4 Confidence cut | Calibrated 0.94 on the span probability, per stratum | Fixed 0.90; raw logit cut | Per-stratum ECE exceeds 0.05 |
| D-5 Audit | 150 accepted spans per day, drawn with known inclusion probabilities | No audit; audit only when something looks wrong | Acceptance rate change exceeds 2 points |
| D-6 Scope | Aggregate only, cells ≥ 20, no agent identifiers stored | Per-agent rate reporting; per-team comparison | Never — structural constraint, requires a new lawful basis and a new case |
| D-7 Rollout | One queue, one language, one month, batch only | All queues at once; streaming | Any guardrail breach during the pilot month |

### 10.2 What the pilot would need to measure, in order

The sequencing is deliberate: each measurement only makes sense if the previous one has been
completed, because a later number computed on a broken earlier artifact is worse than no number.

1. **Admissibility coverage.** What fraction of in-scope legs passes the gate, by queue and by
   reason code? If coverage is below 85%, the denominator is too small for monthly comparison and
   the project stops here regardless of model quality.
2. **Replay feasibility.** Can a reviewer open an audit sample of spans and hear the objection in
   under thirty seconds? If replay fails on more than 1% of sampled spans, the evidence architecture
   is broken and no precision claim is meaningful.
3. **Gold annotation.** Blinded, double-annotated, adjudicated, with agreement reported. Without
   this, precision and recall are undefined.
4. **Calibration by stratum.** Only after gold exists. Fit the calibrator on one half, evaluate on the
   other; report ECE per channel and per queue.
5. **Review service time.** Time-stamped from first open to disposition, per stratum, including
   reopens. This replaces the 1.6-minute nominal figure with a measurement.
6. **Queue behaviour.** Compare observed wait distribution to the Erlang-C projection. A large
   divergence usually means service times are heavier-tailed than assumed, which changes the
   feasible routing share.
7. **Audit-estimated precision.** Only meaningful once routing is stable, because `p̂` describes the
   accepted population, which changes when the routing threshold changes.

The last point contains a subtle trap worth naming: if the threshold is tuned during the measurement
period, `p̂` and the routing share are measured under different policies, and the published interval
is computed from a mixture. Freeze the policy for the measurement window.

### 10.3 Stop conditions

Four conditions under which the work stops rather than degrades:

- **Consent coverage below the threshold needed for a monthly cell.** Aggregation at that coverage
  would describe the consenting subset and be reported as describing all calls. That is the single
  most damaging failure available here, because it looks like success.
- **Replay success below 99%.** Without replay the design's central claim — evidence-linked
  counting — is false, and the report is an unverifiable estimate dressed in provenance metadata.
- **Reviewer agreement below κ = 0.45 on a re-review.** At that level the confirmation step measures
  reviewer preference; more reviewers will not fix it and more data will not fix it.
- **No named owner for the aggregate's consequences.** A number published without an accountable
  owner becomes a number used by whoever finds it, for whatever purpose, which is exactly the
  per-agent scorecard risk the design exists to prevent.

## 11. Outcome & Results

**Evidence level: illustrative, not-measured. Arithmetic verified; no operational outcome observed.**

What can be stated as computed, with the recomputation shown:

| Result | Value | How it was obtained |
|---|---|---|
| Pooled WER on the constructed corpus | 7.31% | 10,733/146,880, and reproduced by the slice-weighted sum |
| Entity WER | 15.38% | 1,412/9,180; 2.27× the non-entity rate |
| Entity error concentration | 6.25% of tokens, 13.16% of errors | Subset counts; multiplication identity closes |
| Diarisation error | 1.69% of speech time | 2,801 s / 165,888 s |
| Turn-level role error | 2.92% of turns, 1.11% of speech seconds | 84/2,880 and 1,845/165,888 |
| Span precision / recall / F1 | 61.32% / 69.66% / 65.23% | 287/468, 287/412, 574/880 |
| Proposed-count inflation | +13.59% | 468/412 − 1 |
| Calibration error (ECE) | 2.67 points | Weighted bin gaps over 1,200 spans |
| Accepted coverage at the 0.94 cut | 50.5% of candidates | 606/1,200 bins at or above 0.94 |
| Daily effective review load | 478.0 min at 56.91% utilisation | 702 × 0.38 × 1.792 against 840 |
| Peak delay probability (c=2, r=0.38) | 52.14% | Erlang-C with a = 1.3146 |
| Peak ten-minute tail | 1.14% | C·exp(−0.3824×10) |
| Feasible routing share, 5% tail guardrail | 44.4% | Bisection on the same formula |
| Feasible routing share, daily average | 66.8% | 840/(702×1.792) |
| Published rate | 65.23 per 100 calls, [62.81, 66.59] | 587.1/900 with Wilson bounds |
| Denominator-induced inflation | +7.53% | 70.14/65.23 − 1 |
| Audit sample needed for ±3 points | 156 spans (34.5% of the frame) | n₀ = 240.7 with FPC 1.5508 |
| Reviewer agreement | κ = 0.642 | 2×2 on 120 re-reviewed spans |
| Annual review cost at the 38% share | £91,623 | 478.1 min/day × £46/h × 250 days |

**What the numbers say as a whole.** The pipeline is good enough to be worth reviewing and not good
enough to publish directly. That combination is the normal case, and it is the case that gets
mishandled most often: a 7.3% WER and 61% span precision look like "nearly there", so the number
gets published with a caveat, and the caveat does not survive contact with a reader who wants a
trend. The arithmetic says the defensible configuration is a reviewed core plus an audited estimate,
which costs £91,623 a year at this volume and yields an interval of ±1.9 per 100 calls.

**What is deliberately not claimed.**

- No reduction in objection volume, because the pipeline has not changed any customer's behaviour.
- No improvement over manual listening, because the manual baseline was never measured here; the
  comparison is a design requirement, not a result.
- No claim that the 38% routing share is optimal — it is the share that survives the peak-hour
  guardrail with two reviewers, which is a constraint, not an optimum.
- No claim about reviewer throughput under real conditions; the 1.6-minute nominal handle time with a
  1.12 rework factor is an assumption with an explicit falsification test in §10.2 step 5.
- No claim that the objection taxonomy is exhaustive; recall of 69.66% is measured against a rubric
  that itself misses objections phrased outside its phrase list.

**The one comparison that is meaningful.** Within the constructed data, two configurations can be
compared because they share all inputs:

| Configuration | Published rate per 100 calls | Interval width | Annual review cost | Precision claim |
|---|---|---|---|---|
| Publish the raw proposed count | 78.00 (702/900) | None | £0 | Unsupported — precision 61.32% |
| Reviewed core + audited estimate | 65.23 | ±1.89 | £91,623 | Supported — precision 95.25% |

The raw count is 19.6% higher than the defensible figure and carries no interval, which is the worst
of both properties: it looks precise and it is wrong by roughly a fifth.

## 12. Failure Modes & What Went Wrong

Fourteen distinct mechanisms. Each has its own symptom, its own competing causes, and a check that
distinguishes between them — because "monitor the model" is not a diagnostic, and two failures with
the same symptom but different fixes are the norm in this pipeline.

### FM-01 — Hold music transcribed as objection language

**Mechanism.** A segmenter running on energy and spectral flux treats music as speech. The ASR then
transcribes it, and a language model with a strong prior produces fluent, plausible phrases. Because
objection phrases are short and generic ("I'm not happy with that"), a hallucinated phrase can match
the rubric exactly. The failure is not random noise; it is systematically shaped by the language
model's prior toward conversational text.

**Competing causes.** (a) Music treated as speech; (b) genuine customer speech over music on a
consult-transfer leg; (c) an announcement containing the same words ("if you are not happy with the
service, press 2").

**Discriminating check.** Join candidate spans to hold-marker timestamps from the telephony event
stream. If a candidate's interval lies entirely inside a hold interval, it is (a). If it partially
overlaps, it may be (b) and needs a listener. If it aligns with an IVR announcement segment, it is
(c) and the fix is an exclusion rule for announcement audio, not a segmenter change.

**Correction.** Exclude hold intervals before ASR using the event stream, not the audio, because the
event stream is ground truth for bridging. Where events are missing, require an explicit
music-detection label and route rather than accept.

**Prevention.** A negative-control corpus of hold and announcement audio, run with every release;
any non-empty output on that corpus is a release blocker.

### FM-02 — Script line counted as a customer objection

**Mechanism.** The agent reads a compliance script: "I understand your concern about the charge."
The rubric matches phrases that describe objections, and this sentence contains one. Role attribution
is absent or wrong, so the span enters the count. This is 35.4% of all false positives in the
constructed data — the single largest class.

**Competing causes.** (a) Role attribution absent; (b) role present but inverted; (c) the customer
genuinely echoed the agent's words ("yeah, my concern about the charge is…") and the span is real.

**Discriminating check.** Compare the span's speaker to the metadata-derived role, and additionally
check position: script lines cluster in the first 20 seconds and immediately after an objection, in a
predictable agent-response pattern. Customer echoes appear after the agent's line, never before. The
ordering check separates (a) from (c) without listening to audio.

**Correction.** Apply the role gate before rubric matching, not after. Order matters: matching first
and filtering later means the confidence was computed in a context that included agent speech.

**Prevention.** A script-corpus regression test: run the matcher over the published script library
and require zero matches in the customer role.

### FM-03 — Negation loss flips the polarity of a span

**Mechanism.** The customer says "I don't have a problem with the charge, it's the timing." The
transcript drops or misplaces "n't" — a clitic with low acoustic salience, frequently reduced to a
nasal murmur — and the text reads "I have a problem with the charge". The rubric match is now
confident and wrong. 22.7% of false positives have this structure in the constructed data.

**Competing causes.** (a) ASR dropped the negation; (b) the negation is present in the transcript but
the matcher's phrase window excludes it (window starts after "don't"); (c) the customer genuinely
said both, in successive clauses.

**Discriminating check.** Re-read the token sequence from the *turn start*, not the span start. If the
negation exists in the transcript upstream of the span, it is (b) — a windowing bug, cheaper to fix
than any model change. If it is absent from the transcript, listen to a two-second pre-roll: a
murmured nasal before the verb confirms (a).

**Correction.** Expand the matching window to the full clause and treat a preceding negation within
the same clause as a veto, with the veto recorded as a reason code rather than a silent drop.

**Prevention.** A negation-focused gold set (statements and their negated twins) scored on every
release; the metric is polarity accuracy, not WER, because WER barely moves when a clitic vanishes.

### FM-04 — Span boundary splits one objection into two

**Mechanism.** An agent backchannel — "mm-hmm", "right" — lands inside the customer's objection. The
segmenter breaks the turn, the matcher emits two spans, and both are counted. The gold set holds one
span. Result: recall looks fine, precision is inflated downward, and the aggregate double-counts a
single objection. 21.0% of false positives in the constructed data.

**Competing causes.** (a) Backchannel-induced split; (b) two genuinely distinct objections in one
breath; (c) a genuine turn break where the customer paused for several seconds.

**Discriminating check.** Measure the inter-span gap and the interrupting token. A gap under 800 ms
with a one-token agent interjection is (a). A gap over three seconds with a complete clause on each
side is more likely (b) or (c). The distinction matters because merging (b) destroys real information.

**Correction.** Merge spans separated by a short agent backchannel within the same rubric category,
and record the merge with both original offsets so the merge is reversible.

**Prevention.** Report span-length distribution and inter-span gap distribution per release; a shift
in either is the earliest visible sign of a segmentation change.

### FM-05 — Transfer splits the conversation and loses half the objection

**Mechanism.** The customer states half the objection, is transferred, and finishes on the second
leg — which has a different `leg_id`. Any per-leg pipeline sees two incomplete fragments, neither of
which matches the rubric. The objection vanishes from the count without appearing as an error
anywhere. 5.6% of false negatives.

**Competing causes.** (a) Genuine transfer split; (b) a dropped bridge that lost the middle of the
conversation; (c) two separate calls that happen to be adjacent in time.

**Discriminating check.** Walk the CTI event chain: a transfer produces a linked-legs record with a
`consult` or `blind_transfer` event and overlapping participant sets. A dropped bridge produces a gap
in the event timeline with no successor leg. Adjacent-but-unlinked calls share no participant.

**Correction.** Build a conversation-level grouping key from the event chain and match spans across
legs within the group, while keeping legs as the audio-evidence unit.

**Prevention.** Publish a per-conversation coverage figure (legs grouped / legs total); a fall in that
ratio precedes any fall in recall.

### FM-06 — Entity substitution changes the objection's category

**Mechanism.** The customer objects to a 15% price rise; the ASR transcribes "fifteen" as "fifty".
The span is still an objection, still matched, but the category assignment keys on the amount and now
points at a different tariff. The count is right in total and wrong in composition, which is worse for
a report organised by category.

**Competing causes.** (a) Acoustic confusion of the numeral; (b) a genuine "fifty" that the annotator
misheard; (c) the amount is correct but the customer's own reference was wrong.

**Discriminating check.** Cross-check numerals against the structured record for that account — the
billed amount, the tariff change date. Where structured data contradicts the transcript, the
transcript is not automatically wrong; the disagreement itself is the finding, and it needs a
listener.

**Correction.** Never overwrite transcript text with structured data. Attach a discrepancy flag and
route the span; the transcript is evidence of what was said, the record is evidence of what was
billed, and conflating them destroys one of the two.

**Prevention.** Track category-composition stability week to week; a composition shift larger than the
total shift is the signature of this failure.

### FM-07 — Language coverage gap reported as a quality gap

**Mechanism.** A queue carries 12% of calls in a language outside the declared supported set. Those
calls fail language identification, are excluded, and the remaining 88% is reported as if it
represented the queue. If the excluded calls also differ in objection rate — plausible, since
billing-dispute queues attract non-dominant-language callers — the reported rate is biased by an
amount nobody can see.

**Competing causes.** (a) Genuine coverage gap; (b) code-switching inside otherwise supported calls
causing false unsupported labels; (c) a language-identification model that is weak on short turns and
misclassifies brief customer responses.

**Discriminating check.** Compare the language label distribution to the queue's declared
configuration and to the consent records (which often carry a language preference). If the two
disagree systematically for short turns, it is (c), and the fix is a minimum-duration rule before
labelling.

**Correction.** Report coverage per language and per queue in the same table as the rate, and suppress
the queue-level rate entirely when coverage falls below the declared floor.

**Prevention.** A coverage guardrail (§9.2) with a stated floor, checked at publication rather than
monitored afterwards.

### FM-08 — Review queue exceeds the peak-feasible share

**Mechanism.** The routing share is set from the daily average and then applied uniformly. The peak
hour pushes the offered load past the two-reviewer capacity, the queue grows through the day, and
spans reviewed the next morning are attributed to the previous day's report. Two errors follow: a
late span enters the wrong month, and a backlog of unreviewed spans silently becomes "accepted" if
any timeout path exists.

**Competing causes.** (a) Understaffing at the peak; (b) heavier-than-assumed service times on complex
spans; (c) a spike in candidate volume from a model or rubric change, not from traffic.

**Discriminating check.** Compare candidate arrivals per hour against call arrivals per hour. If
candidates rose while calls were flat, it is (c). If arrivals are flat and queue time rose, compare
measured service time per stratum to the assumed 1.792 minutes — that separates (a) from (b).

**Correction.** Cap routed volume per hour, emit `CAPACITY_DEFER` for overflow, and never allow a
deferred span to be counted as accepted.

**Prevention.** A publication-time assertion that every span in the report has a review timestamp
inside the reporting window. This is a one-line check that catches the entire family.

### FM-09 — Stale calibration after a channel change

**Mechanism.** A site migrates from amplified headsets to USB headsets. The acoustic profile changes,
confidence becomes miscalibrated in the new stratum, and the 0.94 cut now accepts spans at 86%
accuracy rather than 94%. Nothing in the pipeline breaks; the numbers simply drift, and the published
precision guardrail is checked against an audit that is itself drawn from the new stratum — which
would catch it, if the audit were stratified rather than pooled.

**Competing causes.** (a) Genuine calibration drift from the device change; (b) a model rollback that
restored an older, less-calibrated revision; (c) an audit drawn from the wrong frame.

**Discriminating check.** Compute ECE per device stratum. If the new-device stratum shows ECE above
0.05 while the old stratum is unchanged, it is (a). If all strata moved together, check the deployed
revision hash — that is (b).

**Correction.** Stratified audit by device family, with a minimum per-stratum sample; recalibrate the
new stratum before admitting its spans to the count.

**Prevention.** Treat device family as a first-class slice in the metric contract with its own
guardrail, so a new device family arrives with an empty audit and therefore zero admission.

### FM-10 — Rubric revision re-labels history

**Mechanism.** The objection taxonomy gains a category in month three. Spans aggregated in month four
under the new rubric include a category that did not exist in months one and two. A trend chart shows
a rise at the revision date that is entirely definitional. This is the most common way an analytics
programme produces a false narrative, because the chart is internally consistent and the annotation is
correct.

**Competing causes.** (a) Definitional change; (b) a genuine change in customer behaviour coinciding
with the revision; (c) a change in the queue mix at the same time.

**Discriminating check.** Re-score a frozen, stratified sample of month-one audio under both rubric
versions. If the new category claims spans that were previously uncategorised, it is (a). If the
previously categorised spans also moved, the revision changed more than it claims.

**Correction.** Split the series at the revision and publish both, or re-score the historical frozen
sample under both versions and publish the bridged series with the bridge's uncertainty stated.

**Prevention.** Every rubric version ships with a re-score of the frozen audit sample as an acceptance
test; a version that cannot reproduce the prior version's totals on the frozen sample is not
comparable and must not be presented as a continuation.

### FM-11 — Span offsets drift after re-processing

**Mechanism.** The segmenter is upgraded and produces slightly different boundaries. Span offsets
shift by tens of milliseconds. A reviewer opening yesterday's span hears the right words; a reviewer
opening a re-processed span at the same offset hears the tail of the previous sentence. Replay
"works" but points at the wrong place, so the audit trail is technically present and substantively
wrong. In the constructed data, 4.06% of proposed spans are unreplayable at their recorded offsets
after a re-processing pass.

**Competing causes.** (a) Offset drift from a segmenter change; (b) a channel-mapping change that
swapped left and right; (c) a recording re-delivery with a different hash.

**Discriminating check.** The recording hash settles (c) immediately. For (a) versus (b), compare
energy on each channel for a sample of spans against the expected role: a channel swap puts customer
speech where the agent's was, producing an energy inversion rather than a shift.

**Correction.** Bind offsets to the bundle version that produced them, and never re-point an existing
span at a re-processed audio file. Re-processing creates new spans.

**Prevention.** A replay-success audit (§9.2, floor 0.99) run on every release rather than on a
schedule, because the failure is created by the release.

### FM-12 — Audit sample drawn from a convenient frame

**Mechanism.** The audit is drawn from spans that reviewers have already opened, because that set is
cheap. Those spans are not exchangeable with the accepted population: reviewers open spans that look
suspicious. The resulting `p̂` is either far too low (if suspicious spans dominate) or, worse, too
high if the auditors avoid the hard cases. The estimate looks precise and is biased in an unknown
direction.

**Competing causes.** (a) Frame selection from reviewed spans; (b) frame selection from one day's
traffic; (c) genuine improvement in accepted accuracy after a model update.

**Discriminating check.** Compare the audit frame's feature distribution (channel, span length,
confidence) to the full accepted frame. A material difference is (a) or (b); the day-level check
separates them. A frame that matches and an estimate that rose is (c), which is good news requiring
no correction.

**Correction.** Draw the audit from the accepted frame with a logged sampling seed and inclusion
probabilities, and publish the frame's summary statistics beside the estimate.

**Prevention.** Make the sampling log part of the report manifest, so the frame is auditable by
anyone reading the number, not just by the person who drew it.

### FM-13 — Reviewer fatigue masquerading as model drift

**Mechanism.** Reviewers work a queue that is 60% negative (most routed spans are not objections). Over
a shift, confirmation rates fall: the twentieth similar span gets a faster, less careful disposition.
Because routed volume rises through the day as the queue accumulates, late-shift items are
disproportionately reviewed by tired reviewers. The confirmation rate then correlates with time of
day, and an analyst reading the weekly trend concludes the model degrades in the afternoon.

**Competing causes.** (a) Fatigue; (b) a real diurnal pattern in objection phrasing (morning calls
about billing, evening calls about delivery); (c) a batch-processing schedule that sends different
audio types at different times.

**Discriminating check.** Randomise the queue order for a two-week trial and re-measure. If the
time-of-day effect disappears under randomisation, it is (a). If it persists, it is (b) and the
phenomenon is real. Check (c) by joining span timestamps to batch run identifiers.

**Correction.** Randomise queue order, cap continuous review time, and require a second review on a
sample of dispositions made in the final hour of a shift.

**Prevention.** Report confirmation rate by queue position and by shift hour as a standard cut; a
monotone decline across queue position is the signature, and it is visible without any experiment.

### FM-14 — Aggregate cell reveals an individual

**Mechanism.** A category-by-team table has a cell with three spans, all from the same agent and the
same week. The cell size floor is set on the *category*, not on the cross-tabulation, so the cell
publishes. Anyone with rosters can identify the individual, and the design's central privacy promise
is broken by a table layout rather than by a data leak.

**Competing causes.** (a) Cell-size floor applied at the wrong grain; (b) a legitimate small
population that genuinely has three objections; (c) a join that reintroduced an identifier the span
table was designed not to carry.

**Discriminating check.** Enumerate every published cell and compute its distinct-caller and
distinct-agent counts (from a keyed, non-published join). Any cell with fewer than the floor of
distinct subjects is (a). A cell that passes but whose underlying identifiers were joinable is (c),
which means the span table was altered.

**Correction.** Apply the floor at the finest published grain, not at the top-level category, and
suppress whole rows rather than individual cells to avoid reconstruction by subtraction.

**Prevention.** A publication-time check that computes cell sizes on the actual output table,
including any table derived by subtraction or differencing across months.
## 13. Transferable Lesson

### 13.1 The general form

Any system that converts unstructured human signal (audio, images, pages, logs) into a countable
business quantity faces the same three-way tension, and this case is one instance of it:

- The quantity must be **traceable**, or it cannot be defended after the fact.
- Traceability requires **human attention** on the uncertain cases, which is a finite resource with a
  queue's behaviour, not a constant.
- Human attention is itself **noisy**, so the confirmation step has an error rate that must be
  measured rather than assumed.

The naive design resolves the tension by dropping one leg. Drop traceability and you get a fast number
nobody can defend. Drop the human and you get a number whose error is invisible. Drop the noise model
on the human and you overstate precision while quoting a confirmation count.

### 13.2 Five rules that generalise beyond call centres

**Rule 1 — Evaluate at the grain of the decision, not at the grain of the model.** WER is a model
metric at the token grain; the decision is made per span and per published figure. In this case 7.31%
WER coexisted with 61.32% span precision, and only the second number governed whether publication was
safe. The general test: name the decision, name the unit it is made on, and require a metric computed
on that unit. Any metric whose unit differs from the decision unit is a proxy and must be labelled as
one.

**Rule 2 — Two-sided errors do not cancel; they hide.** Proposed 468 against 412 gold looks like a 13.6%
overcount and is actually 27 missed findings plus 111 invented ones. Totals are the one place where
these errors partially offset, which is exactly the place dashboards are built. The general test:
report precision and recall separately, from an independently sampled frame, even when the total looks
right.

**Rule 3 — A capacity plan is a service-level plan or it is fiction.** The daily average said 56.9%
utilisation and the peak said 52.1% of spans queued. Every human-in-the-loop design has this structure:
a mean that looks comfortable and a tail that decides the experience. The general test: state the
service-level guardrail (here, ≤5% waiting over ten minutes), then compute the maximum feasible load,
then set the automation threshold to respect it — rather than setting the threshold on accuracy and
discovering the capacity consequence afterwards.

**Rule 4 — Calibration is per stratum or it is decoration.** An ECE of 2.67 points overall concealed a
7.5-point overprediction in the bin where the threshold sat. Thresholds are applied where evidence is
weakest, which is by construction the stratum where calibration is worst. The general test: compute the
calibration curve per slice, and refuse to apply a threshold in a stratum whose curve has not been
measured.

**Rule 5 — The denominator is a decision, and it belongs in the sentence.** Excluding 7% of calls
inflated the rate by 7.53% with no change to any span. The general test: no rate is published without
its denominator and its exclusion count in the same table, and the exclusion reasons are enumerated
rather than summed.

### 13.3 The counter-lesson: what would make this design wrong

A transferable lesson that cannot recognise its own limits is a slogan. Three conditions under which
the recommendation in this case inverts:

- **If the decision were per-call and immediate** (route this caller to retention now), the evidence
  architecture would be unaffordable and the right answer would be a cheap, well-calibrated,
  high-recall classifier with no aggregate claim at all. Precision would matter less than speed;
  the span contract would be over-engineering.
- **If the human review were demonstrably worse than the model** — lower agreement, no better
  precision on the audited sample — then routing to humans is a ritual that adds cost and delay. The
  honest design would audit the model directly and drop the review layer, publishing the audit-based
  interval.
- **If the lawful basis forbade retention of audio beyond a few days**, the evidence chain could not
  survive to the reporting date, and the defensible output would be an aggregate with a destroyed
  audit trail — which means the report must be produced within the retention window or not at all.
  That is a scheduling constraint the design must accept, not a problem to solve with more metadata.

### 13.4 The one-sentence version

Publish what you can replay, estimate what you cannot, and budget the difference in staffed minutes at
the peak hour rather than on the daily average.

## 14. Connections & Cross-References

### 14.1 Internal links

| Artifact | Relationship to this case |
|---|---|
| [`../Cheat_Sheets/03_asr-transcription-evaluation.md`](../Cheat_Sheets/03_asr-transcription-evaluation.md) | Supplies the WER/CER derivations, micro-versus-macro aggregation, and the failure catalogue that §6.1 and §6.2 instantiate at case scale |
| [`../Cheat_Sheets/01_NLP_Task_Contracts_Classification_NER_Embeddings.md`](../Cheat_Sheets/01_NLP_Task_Contracts_Classification_NER_Embeddings.md) | Supplies the task-contract discipline (unit, label set, abstention) that §5a applies to span-level objection labelling |
| [`./01_claims_document_extraction.md`](./01_claims_document_extraction.md) | Sibling case with the same evidence-linked abstention pattern on a document grain; contrast the extraction unit (field) with the unit here (time-bounded audio span) |
| [`../INDEX.md`](../INDEX.md) | Unit topic map and status table |
| [`./INDEX.md`](./INDEX.md) | Case catalog; this case is CASE-18-02 |

### 14.2 External method sources

Named because the methods are standard and a reader should be able to check them, not because any
measurement here is attributed to them:

- Word error rate and its alignment derivation: the standard minimum-edit-distance formulation used in
  speech-recognition evaluation since the 1990s.
- Diarisation error rate: the NIST RT-style definition, with falsely-alarmed, missed, and confusion
  components over speech time.
- Erlang-C: the classical queueing delay formula for M/M/c, standard in workforce management.
- Wilson score interval: the standard binomial interval recommended for proportions near the boundary
  and for small samples.
- Cohen's kappa: the standard chance-corrected agreement statistic.
- Finite population correction: standard survey sampling, applicable here because the accepted frame is
  small relative to a per-day audit.

### 14.3 Related patterns elsewhere in the course

| Pattern | Where it recurs | What changes |
|---|---|---|
| evaluation-mismatch | Any ranking, detection, or extraction system | The metric's unit must match the decision's unit |
| human-in-the-loop | Moderation, underwriting, claims, clinical coding | The binding constraint is the review queue's peak behaviour |
| authorization-scope | Consent-gated analytics, permission-aware retrieval | The gate must be structural, not procedural |
| data-leakage | Any supervised pipeline | Annotation, selection, split, and version leaks are distinct |
| model-monitoring | Deployed pipelines | A guardrail must have a defined action on breach, not a chart |

### Interview pack

#### Q1. A monthly report shows 65 objections per 100 calls. State the four things you need before you can reproduce it.

**Answer:** The denominator (in-scope calls: 900, or processed calls: 837 — the two differ by 7.53%);
the numerator's construction (confirmed spans plus an audited estimate of the unreviewed remainder);
the audit frame and its sample size (150 accepted spans, `p̂ = 0.94`, Wilson `[0.8899, 0.9681]`); and
the version bundle (rubric, ASR, diariser, aggregation mapping) in force for that month.

**Reasoning:** The point estimate is a composite `N = N_confirmed + N_auto·p̂`. Without the split you
cannot tell whether a change came from reviewers, from the model, or from the sampling; without the
denominator you cannot tell whether the population changed; without the bundle you cannot tell whether
the definition changed.

**Follow-up:** If the denominator moved from 900 to 837 while the numerator was unchanged, what happens
to the published rate and would you publish it?

**Red flags:** Quoting the rate as if it were a direct count; treating the audit as a formality.

**Scoring:** 3 = names all four and can compute the denominator effect; 2 = names three; 1 = describes
the pipeline without the composite structure.

#### Q2. Compute the effect of switching the published denominator from in-scope to processed calls.

**Answer:** The numerator is `178 + 435.2 × 0.94 = 587.1` spans. Per in-scope call: `587.1/900 = 65.23`
per 100. Per processed call: `587.1/837 = 70.14` per 100. The change is `70.14/65.23 − 1 = +7.53%`.

**Reasoning:** 63 calls (7.0%) were excluded for consent, capture, language, or integrity reasons. The
numerator is identical; only the population description changed. Because the excluded calls are not a
random sample of calls — consent coverage and language mix differ by queue — the processed-call rate is
also *biased*, not merely rescaled.

**Follow-up:** Which denominator is correct if the business question is "how often do our customers
object?" versus "how often do customers in the analysable population object?"

**Red flags:** Choosing the larger number because it is larger; claiming the two are interchangeable
because they are both "per 100 calls".

**Scoring:** 3 = computes both, names the bias, and distinguishes the questions; 2 = computes both; 1 =
asserts the rate is "about the same either way".

#### Q3. Span precision is 61.32% and recall is 69.66%. A colleague says the errors roughly cancel, so the total is usable. Diagnose the flaw.

**Answer:** The totals cancel only in aggregate: 468 proposed against 412 gold is +13.59%, which looks
acceptable. The composition does not cancel — 181 false positives and 125 false negatives are different
spans, and the category table built from them is a mixture of invented categories and missing ones.

**Reasoning:** Cancellation is a property of sums, not of distributions. If the 181 spurious spans
concentrate in one rubric category (script lines about charges) and the 125 missed spans concentrate in
another (weak-speech objections about delivery), the two category rates move in opposite directions
while the headline is stable. A report organised by category is therefore wrong in both directions.

**Follow-up:** What single check would reveal the cancellation is coincidental rather than structural?

**Red flags:** Accepting the cancellation; proposing a fudge factor to rescale the total.

**Scoring:** 3 = separates total from composition and predicts the category-level divergence; 2 =
recognises the two error types are different; 1 = accepts cancellation or proposes rescaling.

#### Q4. A span has model confidence 0.965. The bin table says spans at that confidence are 92.5% correct. What do you do with it, and why not simply lower the threshold?

**Answer:** The span enters the accepted set only if the 0.965 bin is calibrated for its stratum; on the
constructed numbers that bin over-predicts by 4.0 points (0.965 predicted vs 0.925 observed). Lowering
the threshold does not fix miscalibration — it enlarges the population exposed to it, because the
0.80–0.90 bin over-predicts by 7.5 points, the worst in the table.

**Reasoning:** A threshold is a decision rule applied to a probability; if the probability is wrong in
the region where the rule fires, the rule is wrong regardless of where it is set. The fix is
recalibration (isotonic or Platt on held-out data), then threshold selection on the recalibrated score.

**Follow-up:** Quantify the cost of lowering the cut from 0.94 to 0.85 in reviewer minutes at 702
candidates per day.

**Red flags:** Treating confidence as accuracy; assuming monotone bins imply monotone correctness.

**Scoring:** 3 = separates calibration from thresholding and quantifies the exposure; 2 = recognises
miscalibration; 1 = proposes a threshold change as the remedy.

#### Q5. At 702 candidates/day and a 38% routing share, is the review team feasible? Show the calculation that decides it.

**Answer:** Not from the daily average. Daily: `702 × 0.38 × 1.792 = 478.0` minutes against `2 × 7 × 60 =
840`, i.e. 56.9% utilisation — apparently comfortable. Peak hour: arrivals `148.5 × 0.78 = 115.83`
candidates/hour, so `λ = 115.83 × 0.38 / 60 = 0.7336` spans/min. With `μ = 1/1.792 = 0.5580`, offered
load `a = 1.3146`, `a/c = 0.6573`, Erlang-C `C = 0.5214`, mean wait `1.36` min, and
`P(wait > 10 min) = 0.5214·exp(−0.3824×10) = 1.14%`. That passes a 5% guardrail; the maximum feasible
share under a 5% guardrail is 44.4%, so 38% has margin but only about six points of it.

**Reasoning:** The daily mean is a load test. Delay is a queue property governed by the peak hour and by
the gap between capacity and arrival rate, which shrinks fast: at 50% routing the same team's ten-minute
tail is 17.75%.

**Follow-up:** With three reviewers, what share can be sustained, and what does it cost?

**Red flags:** Comparing minutes-per-day to minutes-per-day and declaring feasibility; ignoring the
service-level tail.

**Scoring:** 3 = computes both views and states the feasible share; 2 = computes the daily view and
recognises it is insufficient; 1 = does the daily division only.

#### Q6. Three reviewers sustain a 60% routing share (tail 0.28%). Two sustain 38% (tail 1.14%). Which is the better decision and what numbers decide it?

**Answer:** The third reviewer admits `702 × 0.22 = 154.4` additional spans per day, at an extra
`702 × 0.22 × 1.792 = 276.8` staffed minutes per day — about £53,000 a year at £46/hour over 250 days.
Two reviewers at 38% cost about £91,623 a year. So the question is whether 154 more reviewed spans a day
is worth £53,000, which depends on what those spans buy: if routing the extra 22% removes the highest-
confidence errors, it raises published precision; if it merely duplicates what the audit already
estimates, it is £53,000 for a slightly narrower interval.

**Reasoning:** The comparison must be made on the *increment*, not on the total. The relevant quantity is
the marginal cost per unit of decision quality, and the audit already provides an unbiased estimate of
the accepted population — so extra review competes with the audit, not with the status quo.

**Follow-up:** At what accepted-error rate does extra review beat a larger audit sample?

**Red flags:** Choosing three reviewers "for safety"; comparing total costs rather than increments.

**Scoring:** 3 = frames it as a marginal comparison against the audit; 2 = computes the increment; 1 =
picks an option on intuition.

#### Q7. Your audit finds `p̂ = 0.94` from 150 spans. How large should the audit be, and why is 150 nearly right?

**Answer:** For ±3 points at 95% confidence with `p ≈ 0.94`, `n₀ = z²p(1−p)/e² = 3.8416 × 0.94 × 0.06 /
0.0009 = 240.7`. With a finite population correction against a 435.2-span daily accepted frame,
`n = 240.7/(1 + 239.7/435.2) = 155.2`, so 156 spans. A 150-span audit therefore achieves ±3.08 points,
not ±3.00 — close enough, and the FPC is what makes it possible at all: without the correction the same
margin needs 241 spans, 55% of the frame.

**Reasoning:** The correction is legitimate because the sample is a large fraction (34.5%) of a small
population. Ignoring it overstates the required sample by 55%, which is the difference between an audit
that fits the day and one that consumes it.

**Follow-up:** What happens to the required sample if the audit must be stratified by four channels?

**Red flags:** Using the uncorrected formula; sampling 10% "because that is the rule".

**Scoring:** 3 = computes n₀, applies the FPC, and notes the stratified consequence; 2 = computes n₀
only; 1 = picks a round number.

#### Q8. A reviewer and a second reviewer re-review 120 spans blind: 71 both-objection, 30 both-not, 8 first-only, 11 second-only. Interpret.

**Answer:** `p_o = (71+30)/120 = 0.8417`; marginals give `p_e = (79/120)(82/120) + (41/120)(38/120) =
0.4498 + 0.1082 = 0.5581`; `κ = (0.8417 − 0.5581)/(1 − 0.5581) = 0.642`. That is substantial agreement,
and it caps the meaning of the "confirmed" count: about 6–8% of confirmed spans would be disagreed with
by a second competent reviewer — the same order as the 6% accepted-error rate.

**Reasoning:** Kappa corrects for the high base rate of agreement that arises because most spans are
objections. Raw agreement of 84% sounds strong; against a chance rate of 56% it is moderate-to-
substantial, and the disagreement rate is what bounds the confirmation step's contribution to precision.

**Follow-up:** Which guardrail does this justify, and what would you do at κ = 0.45?

**Red flags:** Reporting 84% agreement as if it were the quality of the count; ignoring the base rate.

**Scoring:** 3 = computes kappa and interprets the cap on the confirmed count; 2 = computes kappa; 1 =
quotes raw agreement.

#### Q9. A customer says "I don't have a problem with the charge, it's the timing." The pipeline counts it as a charge objection. Walk the diagnosis.

**Answer:** Three candidates, distinguished without guessing. First, check whether the negation is
present in the transcript upstream of the span — if `don't` is there, the matcher's window started after
it (a windowing bug, cheapest fix). Second, if absent from the transcript, listen to a two-second
pre-roll: a reduced nasal before "have" confirms the clitic was dropped acoustically. Third, if the
transcript is correct and complete, check whether the clause was split by a segmenter boundary, which
would put the negation in a different span.

**Reasoning:** The three causes have different fixes at very different costs: window change, acoustic
model or language-model bias against clitics, or segmentation merge. Diagnosing in cost order avoids
retraining a model to fix a regex.

**Follow-up:** What metric would you add so this class of error is visible on every release?

**Red flags:** Retraining immediately; assuming the transcript is faithful because WER is low.

**Scoring:** 3 = orders the diagnosis by cost and names the check for each; 2 = names two causes; 1 =
attributes it to "model error" generically.

#### Q10. Design the check that distinguishes "hold music hallucinated as speech" from "customer speaking over music during a consult transfer".

**Answer:** Join the candidate span's interval to the telephony event stream. Fully inside a hold
interval with no participant bridge → hallucination on music. Partially overlapping a hold interval that
also contains a consult participant → genuine speech over music, needs a listener. A third case: aligned
with an IVR announcement segment, which is a script-match, not either of the above.

**Reasoning:** The event stream is ground truth for bridging in a way audio never is, so it separates
causes that sound identical. This is a general principle: when two mechanisms produce the same acoustic
signature, use a non-acoustic record to discriminate.

**Follow-up:** What negative-control corpus would you run with each release, and what result blocks it?

**Red flags:** Listening to a sample and generalising; adding a music classifier as the first response.

**Scoring:** 3 = uses the event stream and enumerates three cases; 2 = proposes a music detector; 1 =
suggests listening to more audio.

#### Q11. Recall is 69.66%. Where are the missing spans, and which is cheapest to recover?

**Answer:** 96 of the 125 false negatives (76.8%) are objections inside calls that were routed — they
exist but are excluded from the matched set by the evaluation's own rule. 22 (17.6%) are objections
phrased outside the rubric phrase list. 7 (5.6%) are split across a transfer.

The cheapest recovery is the routing interaction: those 96 spans are *found*, just not counted, so the
fix is an evaluation-design question (count routed spans in recall with their routed status) rather than
a modelling one. The transfer case needs a conversation-level grouping key. The out-of-rubric case is
the only one requiring rubric or model work, and it is the smallest bucket.

**Reasoning:** A recall number that excludes routed spans understates detection and misdirects effort
toward modelling. Reporting "recall over all gold spans, with routed spans marked" separates detection
from publication.

**Follow-up:** If routed spans were counted in recall, what would recall become, and why is that number
not directly comparable to precision?

**Red flags:** Assuming recall is a model property; going straight to more training data.

**Scoring:** 3 = decomposes the misses and identifies the evaluation-design artefact; 2 = names two
buckets; 1 = treats recall as a single number to improve.

#### Q12. A model upgrade lifts pooled WER from 7.31% to 7.05%. Can you publish the improvement? What must you check first?

**Answer:** Not yet. Check four things: (a) the slice table — a pooled gain can conceal a regression on
legacy headsets, which contribute 1.199 of the 7.315 weighted points from only 11% of tokens; (b) entity
WER, since that is what governs span quality and it can move opposite to pooled WER; (c) span precision
and recall on the same frozen gold, because token-level gains do not automatically become span-level
gains; (d) calibration, because a new revision's confidence is uncalibrated until refitted.

**Reasoning:** A 0.26-point pooled gain is small relative to the sampling uncertainty of a 300-call
corpus, and it is smaller than the 7.53% denominator effect — so it is not, by itself, a decision-relevant
change.

**Follow-up:** What would make a 0.26-point gain decision-relevant?

**Red flags:** Publishing on the headline; comparing across corpora; ignoring recalibration.

**Scoring:** 3 = lists all four checks and contextualises the magnitude; 2 = names two; 1 = publishes.

#### Q13. The rubric gains a category in month four. What breaks, and what do you publish?

**Answer:** The trend breaks at the revision, because the same count means something different on either
side. Publish either a split series with the revision marked, or a bridged series produced by re-scoring
a frozen stratified sample of earlier audio under both versions — with the bridge's uncertainty stated.
Do not splice the two series into one line.

**Reasoning:** A definitional change is a discontinuity, not a variance. Adding it to a confidence
interval is a category error: no amount of data makes month three and month four comparable.

**Follow-up:** What acceptance test should a rubric version ship with?

**Red flags:** Re-labelling history silently; plotting a single line through the revision.

**Scoring:** 3 = names the discontinuity and the bridge method; 2 = says "split the series"; 1 =
continues the series.

#### Q14. An operations lead asks for per-agent objection rates from the same pipeline. What is your answer and what is the strongest reason?

**Answer:** No — and the strongest reason is not privacy policy, it is inferential validity. The span
table does not carry agent identifiers by design, and even if it did, the per-agent denominator is a few
hundred calls a month with a 61.32% span precision and a 6% accepted-error rate. The noise on a per-agent
rate would swamp any real difference, while the number would still be read as a ranking.

**Reasoning:** Refusing on privacy grounds invites a policy exception; refusing on measurement grounds
does not. The honest statement is that the aggregate is defensible and the per-agent slice is not, from
the same data, at the same precision.

**Follow-up:** What would it take to make a per-agent claim defensible, and would it still be this
pipeline?

**Red flags:** Refusing only on policy; offering a "rough" per-agent number as a compromise.

**Scoring:** 3 = gives the inferential reason and quantifies it; 2 = refuses on policy only; 1 = offers a
caveated per-agent number.

#### Q15. Replay success on audited spans falls to 96%. What does that mean for the published figure, and what is the first check?

**Answer:** It means 4% of spans cannot be replayed at their recorded offsets, so the claim "evidence-
linked" is false for that cohort and publication must be blocked. The first check is the recording hash:
if the hash changed, the audio was re-delivered or re-encoded and the spans belong to a different byte
stream. If the hash matches, check the bundle version that produced the offsets — a segmenter upgrade
shifts boundaries, and old offsets point at the wrong place.

**Reasoning:** Replay is the load-bearing element of the whole design; at 96% the audit trail is present
in form and absent in substance. The 4.06% unreplayable rate in the constructed data is exactly this
failure, and it is created by re-processing, so it appears right after a release rather than drifting.

**Follow-up:** Why is an offset bound to a bundle version rather than to the recording alone?

**Red flags:** Accepting 96% as "good enough"; re-pointing old spans at newly processed audio.

**Scoring:** 3 = blocks publication and orders the checks by likelihood; 2 = recognises the severity; 1 =
proposes a manual workaround.

#### Q16. Give the smallest set of numbers you would put in front of the decision owner each month.

**Answer:** Seven: (1) in-scope calls and the exclusion count by reason; (2) candidates and the routed
share; (3) confirmed objections and the audit `p̂` with its interval; (4) the resulting rate per 100
in-scope calls with bounds; (5) published precision against the 0.93 guardrail; (6) peak-hour queue tail
against the 5% guardrail; (7) the bundle version and the rubric version.

**Reasoning:** Each corresponds to a way the number can be wrong: population, workload, numerator
construction, uncertainty, precision, feasibility, definition. Anything else is commentary; anything
fewer allows one of the seven failure routes to pass unobserved.

**Follow-up:** Which of the seven would you drop first under pressure, and what risk does that accept?

**Red flags:** A dashboard with thirty metrics and no guardrails; omitting the exclusion count.

**Scoring:** 3 = lists the seven with their failure route; 2 = lists five or six; 1 = lists a WER and a
count.

## 15. Quick Reference Card

| Field | Rule | Boundary / failure it prevents |
|---|---|---|
| Decision unit | Objection span: authorised audio interval + role + rubric match + replayable offset | A transcript, a call, or a turn cannot be counted |
| Admissibility | Consent, identity, integrity, capture completeness — gated before any compute | Fails closed; denial counted, never imputed |
| Evaluation grain | Span precision and recall from a blinded gold set | Pooled WER hid a 15.38% entity error behind 7.31% |
| Role | Metadata first, diariser as consistency check, `unknown` routed | Prevents 35.4% script-line false positives |
| Calibration | Per stratum, then threshold; never a fixed cut on raw confidence | Overall ECE 2.67 points concealed a 7.5-point bin gap |
| Routing | Calibrated probability plus a peak-hour capacity cap | Daily 56.9% utilisation coexisted with a 52.1% peak queue |
| Feasible share | 44.4% at a 5% ten-minute tail with two reviewers | The daily average would have permitted 66.8% |
| Numerator | `N_confirmed + N_auto·p̂`, with a Wilson interval on `p̂` | A raw proposed count is 19.6% high and carries no interval |
| Denominator | In-scope calls, named in the same sentence as the rate | Processed-call denominators inflate the rate by 7.53% |
| Versions | Parser, segmenter, ASR, diariser, rubric, mapping move atomically | Mixed-version cohorts are refused, not reconciled |
| Replay | ≥0.99 success from original bytes at bound offsets | At 96% "evidence-linked" is false and publication blocks |
| Scope | Aggregates only, cells ≥20 at the published grain, no agent identifiers stored | Small-cell differencing is the residual risk |
| Evidence level | Illustrative, not-measured | No operational, vendor, or legal claim is made |
