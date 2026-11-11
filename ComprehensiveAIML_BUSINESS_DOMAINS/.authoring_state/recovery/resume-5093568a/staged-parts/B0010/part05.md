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
