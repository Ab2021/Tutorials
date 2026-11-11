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
