## 12. Quick Reference Card

| Item | Value and condition |
|---|---|
| Core decision | Trust a candidate pipeline only when eligibility is checked at decision time, candidate recall is measured on a matched cohort, and underfill plus latency are reported alongside it |
| Candidate recall | `|C_k ∩ R| / |R ∩ E|`, defined only when the eligible labelled count is positive; the denominator uses the eligible universe, never the whole catalog |
| Aggregates | macro weights requests equally; pooled (identical to relevant-count weighting) weights by relevant-item count; declare the choice before seeing results |
| Depth requirement | smallest m with `P(Binomial(m, p) ≥ k) ≥ 1 − α`; p = 0.02, k = 20, α = 0.05 gives m = 1390 (ratio 69.5); p = 0.08 gives m = 344 (ratio 17.2) |
| Eligibility attrition | `1 − r1 − r2 + (r1·r2 + ω)`; positive co-rejection ω raises joint survival, so independence under-estimates the surviving set |
| Approval gate | fail closed on authorization, push eligibility into traversal or partition the index, and keep eligibility decisions out of the index so staleness affects quality only |
| Cost anchors | exact scan is bandwidth-bound (6.144 GB per request at 12M × 128 float32); ANN at depth 1390 touches ~742 KiB; ranker cost scales with candidate count at roughly 2.6 ms per candidate |
| Use when | a ranker or slate budget limits how many items can be scored and eligibility is a nontrivial share of the catalog |
| Avoid when | the eligible labelled population cannot be assembled point-in-time, the label horizon is immature, or the comparison differs in k, H, or eligibility vintage |
| Signature failure | a full slate with weak personalisation: fallback rows fill the slots because the primary path under-delivers, and slate fill hides it |
