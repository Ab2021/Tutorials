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

The 5.4:1 ratio is the business case in one number. It is deliberately computed against total human
review, 4.58 FTE including the audit, rather than against candidate adjudication alone, which would give
24.57 / 4.29 = 5.7:1 and flatter the design by omitting the very work that makes its recall measurable.
Its weakness is stated with it: the census has recall near 1.0 by construction, while the assisted route
has recall 0.746 with an interval of 0.627 to 0.837. The assisted route is cheaper and less complete, and the correction term is what makes the
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
