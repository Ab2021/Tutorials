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
