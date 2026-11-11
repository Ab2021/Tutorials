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
